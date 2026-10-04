"""Atomic widths, signedness, contention, target rejection and device scope."""

import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from llvmlite import binding as llvm

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.llvm_gen import generate_llvm_ir
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang import ir
from tack.lang.atomic_support import ATOMIC_DTYPES
from tack.lang.ir_type_annotate import annotate_types
from tack.runtime.dispatch import get_backend
from tack.runtime.kernel_utils import resolve_variant

TYPES = (tack.i8, tack.u8, tack.i16, tack.u16, tack.i32, tack.u32,
         tack.i64, tack.u64, tack.f32, tack.f64)
GENERATORS = [('cpu', generate_llvm_ir), ('cuda', generate_cuda_source),
              ('hip', generate_hip_source), ('metal', generate_msl_source),
              ('level_zero', generate_opencl_source)]


@tack.kernel
def atomic_add(data, out, n):
    for i in range(n):
        tack.atomic_add(out, 0, data[i])


@tack.kernel
def atomic_min(data, out, n):
    for i in range(n):
        tack.atomic_min(out, 0, data[i])


@tack.kernel
def atomic_max(data, out, n):
    for i in range(n):
        tack.atomic_max(out, 0, data[i])


KERNELS = {'add': atomic_add, 'min': atomic_min, 'max': atomic_max}


def _field(values, dtype):
    array = np.asarray(values, dtype=dtype.numpy_dtype)
    result = tack.field(dtype, array.shape)
    result.from_numpy(array)
    return result


def _check_floating_extrema_zero(dtype, op, zero):
    # Numeric zeros are valid extrema. Their sign on a tie is unspecified.
    direction = 1 if op == 'min' else -1
    data = _field([zero] * 4097, dtype)
    out = _field([direction * 4], dtype)
    KERNELS[op](data, out, 4097)
    assert out.to_numpy()[0] == 0
    for stored in [0.0, -0.0]:
        out.fill(stored)
        data.fill(direction * 2)
        KERNELS[op](data, out, 4097)
        assert out.to_numpy()[0] == 0
        data.fill(-direction * 2)
        KERNELS[op](data, out, 4097)
        assert out.to_numpy()[0] == -direction * 2


@pytest.mark.parametrize('op', ['min', 'max'])
@pytest.mark.parametrize('zero', [0.0, -0.0])
def test_f32_atomic_extrema_numeric_zeros(backend, op, zero):
    _check_floating_extrema_zero(tack.f32, op, zero)


@pytest.mark.parametrize('op', ['min', 'max'])
@pytest.mark.parametrize('zero', [0.0, -0.0])
def test_f64_atomic_extrema_numeric_zeros(f64_backend, op, zero):
    be = get_backend()
    if tack.f64 not in be.supported_atomic_dtypes:
        pytest.skip('Backend has f64 fields but no f64 atomics')
    _check_floating_extrema_zero(tack.f64, op, zero)


def _wrap(value, dtype):
    if dtype in (tack.f32, tack.f64):
        return dtype.numpy_dtype.type(value)
    value = int(value) % (1 << dtype.bits)
    if dtype.name.startswith('i') and value >= 1 << (dtype.bits - 1):
        value -= 1 << dtype.bits
    return value


def _typed_ir(dtype, op):
    target = ir.IRParam('out', dtype)
    target._is_field = True
    atomic = ir.IRAtomicOp(op, ir.IRName('out'), ir.IRConstant(0), ir.IRConstant(1))
    func = ir.IRFunction('atomic_contract', [target], [
        ir.IRParallelFor('i', ir.IRConstant(0), ir.IRConstant(513), [atomic]),
    ])
    annotate_types(func)
    return func, atomic


@pytest.mark.parametrize('dtype', TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize('op', KERNELS)
@pytest.mark.parametrize('arch,generator', GENERATORS, ids=[x[0] for x in GENERATORS])
def test_direct_generators_enforce_width_domain(arch, generator, dtype, op):
    func, node = _typed_ir(dtype, op)
    node.dtype = tack.f32  # A stale annotation cannot authorize a target.
    if dtype not in ATOMIC_DTYPES[arch]:
        with pytest.raises(TypeError, match=f'atomic_{op}.*target dtype.*{dtype.name}'):
            generator(func)
    else:
        source = generator(func)
        if arch == 'cpu':
            llvm.parse_assembly(str(source)).verify()
        if arch == 'level_zero':
            assert 'memory_scope_device' in source
            assert 'atomic_cmpxchg(' not in source


@pytest.mark.parametrize('arch,generator', GENERATORS, ids=[x[0] for x in GENERATORS])
@pytest.mark.parametrize('target_kind', ['scalar', 'local', 'shared', 'texture'])
def test_direct_generators_reject_other_atomic_address_spaces(arch, generator, target_kind):
    func, _ = _typed_ir(tack.f32, 'add')
    if target_kind in ('scalar', 'texture'):
        func.params[0]._is_field = target_kind == 'texture'
        func.params[0]._is_texture = target_kind == 'texture'
    else:
        allocation = ir.IRLocalAlloc if target_kind == 'local' else ir.IRSharedAlloc
        func.params.clear()
        func.body.insert(0, allocation('out', tack.f32, ir.IRConstant(1)))
    with pytest.raises(TypeError, match='atomic_add.*global field parameter'):
        generator(func)


@pytest.mark.parametrize('op', KERNELS)
@pytest.mark.parametrize('dtype', TYPES, ids=lambda t: t.name)
def test_atomic_results_and_cached_inputs(backend, dtype, op):
    be = get_backend()
    if dtype not in be.supported_dtypes:
        pytest.skip('target does not support this field dtype')
    info = np.iinfo(dtype.numpy_dtype) if dtype.name[0] in 'iu' else None
    values = ([1, info.max, 2, info.min + 1] if info is not None else [0.25, -2, 8, 0.5])
    data = _field(values, dtype)
    initial = {'add': 0, 'min': info.max if info else 99,
               'max': info.min if info else -99}[op]
    out = _field([initial], dtype)
    kernel = KERNELS[op]
    if dtype not in be.supported_atomic_dtypes:
        with pytest.raises(TypeError, match=f'atomic_{op}.*target dtype.*{dtype.name}'):
            kernel(data, out, len(values))
        assert out.to_numpy()[0] == initial
        return
    for current in (values, list(reversed(values))):
        data.from_numpy(np.asarray(current, dtype=dtype.numpy_dtype))
        out.fill(initial)
        kernel(data, out, len(values))
        expected = (_wrap(sum(map(int, current)), dtype) if info is not None else
                    sum(current)) if op == 'add' else getattr(np, op)(data.to_numpy())
        assert out.to_numpy()[0] == expected
    # Empty calls do not update storage, and atomics need no full workgroup.
    out.fill(initial)
    kernel(data, out, 0)
    assert out.to_numpy()[0] == initial


@pytest.mark.parametrize('op', KERNELS)
def test_atomic_value_converts_to_unsigned_target(backend, op):
    data = _field([2**31 + 256, 512, 1024], tack.f32)
    out = _field([0 if op != 'min' else 2**32 - 1], tack.u32)
    KERNELS[op](data, out, 3)
    expected = {'add': 2**31 + 1792, 'min': 512, 'max': 2**31 + 256}[op]
    assert out.to_numpy()[0] == expected


@pytest.mark.parametrize('op', KERNELS)
@pytest.mark.parametrize('dtype', [tack.f32, tack.f64], ids=lambda t: t.name)
def test_cpu_concurrent_updates_are_atomic(dtype, op):
    tack.init(arch=tack.cpu)
    be = get_backend()
    count, workers = 32768, 8
    values = np.random.default_rng(6017).permutation(count).astype(dtype.numpy_dtype) + 1
    data = _field(values, dtype)
    out = _field([0 if op == 'add' else count + 1 if op == 'min' else -1], dtype)
    kernel = KERNELS[op]
    # Compile through the real variant pipeline, then exercise real concurrent
    # call_range workers without depending on the adaptive threading policy.
    variant, args = resolve_variant(be, kernel, (data, out, count), {}, build=be._build_variant)
    compiled = variant.payload
    prefix = compiled.bind(args)
    # Exactly representable sums, even in f32, irrespective of scheduling.
    if op == 'add':
        data.fill(0.25)
    expected = {'add': count / 4, 'min': 1, 'max': count}[op]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(4):
            out.fill(0 if op == 'add' else count + 1 if op == 'min' else -1)
            gate = threading.Barrier(workers)

            def run(slot, gate=gate):
                gate.wait()
                compiled.call_range(prefix, slot * (count // workers), (slot + 1) * (count // workers))

            futures = [pool.submit(run, slot) for slot in range(workers)]
            for future in futures:
                future.result()
            assert out.to_numpy()[0] == expected
    if op != 'add':
        source = tack.inspect(kernel, data, out, count)
        assert 'load atomic' in source and 'cmpxchg' in source


@pytest.mark.parametrize('mode', ['dispatch', 'ir', 'source', 'optimized'])
def test_cpu_rejects_unaligned_atomic_storage(mode, monkeypatch):
    tack.init(arch=tack.cpu)
    data = _field([1], tack.i32)
    raw = bytearray(5)
    array = np.ndarray((1,), np.int32, buffer=raw, offset=1)
    array.fill(7)
    out = tack.field_from_ptr(array, tack.i32, (1,), writable=True)
    be = get_backend()

    def forbidden_build(*args):
        pytest.fail('unaligned atomic target reached code generation')

    monkeypatch.setattr(be, '_build_variant', forbidden_build)
    with pytest.raises(ValueError, match='atomic target.*4-byte alignment'):
        if mode == 'dispatch':
            atomic_add(data, out, 1)
        else:
            tack.inspect(atomic_add, data, out, 1, mode=mode)
    assert array[0] == 7


def test_cached_atomic_alignment_without_ir_scan(monkeypatch):
    from tack.runtime import kernel_utils

    tack.init(arch=tack.cpu)
    data, out = _field([1], tack.i32), _field([0], tack.i32)
    atomic_add(data, out, 1)

    def forbidden_scan(*args, **kwargs):
        pytest.fail('cached atomic launch repeated structural analysis')

    monkeypatch.setattr(kernel_utils, 'check_atomic_support', forbidden_scan)
    atomic_add(data, out, 1)
    assert out.to_numpy()[0] == 2
    raw = bytearray(5)
    array = np.ndarray((1,), np.int32, buffer=raw, offset=1)
    array.fill(7)
    unaligned = tack.field_from_ptr(array, tack.i32, (1,), writable=True)
    with pytest.raises(ValueError, match='4-byte alignment'):
        atomic_add(data, unaligned, 1)
    assert array[0] == 7


@pytest.mark.parametrize('dtype', [tack.i32, tack.u32, tack.f32], ids=lambda t: t.name)
@pytest.mark.parametrize('op', KERNELS)
def test_opencl_device_scope_syntax(dtype, op, tmp_path):
    clang = shutil.which('clang')
    if not clang:
        pytest.skip('requires clang')
    func, _ = _typed_ir(dtype, op)
    source = tmp_path / 'atomic.cl'
    source.write_text(generate_opencl_source(func))
    result = subprocess.run([clang, '-x', 'cl', '-cl-std=CL2.0', '-fsyntax-only', str(source)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('arch,generator', GENERATORS[1:3], ids=['cuda', 'hip'])
@pytest.mark.parametrize('dtype', [tack.i64, tack.u64, tack.f64], ids=lambda t: t.name)
@pytest.mark.parametrize('op', KERNELS)
def test_cuda_hip_wide_atomic_syntax(arch, generator, dtype, op, tmp_path):
    clang = shutil.which('clang++')
    if not clang:
        pytest.skip('requires clang++')
    func, _ = _typed_ir(dtype, op)
    body = generator(func).replace('#include <hip/hip_runtime.h>\n', '')
    preamble = '''#define __device__
#define __global__
struct Dim { unsigned int x; };
extern Dim threadIdx, blockIdx, blockDim;
unsigned long long atomicCAS(unsigned long long*, unsigned long long, unsigned long long);
double __longlong_as_double(long long);
long long __double_as_longlong(double);
'''
    source = tmp_path / 'atomic.cpp'
    source.write_text(preamble + body)
    result = subprocess.run([clang, '-std=c++17', '-fsyntax-only', str(source)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('dtype', [tack.i32, tack.u32, tack.f32], ids=lambda t: t.name)
@pytest.mark.parametrize('op', KERNELS)
def test_cross_workgroup_atomic_contention(backend, dtype, op):
    count = 4099  # Multiple groups plus a partial final group.
    values = np.random.default_rng(6018).permutation(count).astype(dtype.numpy_dtype) + 1
    if dtype is tack.u32:
        values[count // 2] = 2**32 - 1
    if dtype is tack.f32 and op == 'add':
        values.fill(0.25)
    data = _field(values, dtype)
    info = np.iinfo(dtype.numpy_dtype) if dtype is not tack.f32 else None
    initial = {'add': 0, 'min': info.max if info else count + 1,
               'max': info.min if info else -1}[op]
    out = _field([initial], dtype)
    expected = (_wrap(sum(map(int, values)), dtype) if info else sum(map(float, values))) \
        if op == 'add' else getattr(np, op)(values)
    for _ in range(4):
        out.fill(initial)
        KERNELS[op](data, out, count)
        assert out.to_numpy()[0] == expected


@pytest.mark.parametrize('mode', ['dispatch', 'ir', 'source', 'optimized'])
@pytest.mark.parametrize('op', KERNELS)
def test_gpu_unsupported_width_rejected_before_compilation(workgroup_backend, mode, op, monkeypatch):
    be = get_backend()
    data, out = _field([1], tack.i16), _field([7], tack.i16)

    def forbidden_build(*args):
        pytest.fail('unsupported atomic width reached device compilation')

    monkeypatch.setattr(be, '_build_variant', forbidden_build)
    with pytest.raises(TypeError, match=f'atomic_{op}.*target dtype.*i16'):
        if mode == 'dispatch':
            KERNELS[op](data, out, 0)  # Structural rejection even on an empty launch.
        else:
            tack.inspect(KERNELS[op], data, out, 0, mode=mode)
    assert out.to_numpy()[0] == 7


@tack.kernel
def aliased_atomic_updates(data, left, right):
    for i in range(data.shape[0]):
        tack.atomic_add(left, 0, data[i])
        tack.atomic_add(right, 0, data[i])


def test_atomic_updates_through_aliased_fields(backend):
    data = _field(np.full(513, 0.25), tack.f32)
    out = _field([0], tack.f32)
    for alias in (out, out.reshape((1,))):
        out.fill(0)
        aliased_atomic_updates(data, out, alias)
        assert out.to_numpy()[0] == 513 / 2


@tack.kernel
def publish_global_with_barrier(data, out):
    for i in range(data.shape[0]):
        data[i] = i + 1
        tack.barrier()
        out[i] = data[(i // 256) * 256 + (tack.thread_id() + 1) % 256]


def test_workgroup_barrier_publishes_global_field_updates(workgroup_backend):
    data, out = _field(np.zeros(512), tack.i32), _field(np.zeros(512), tack.i32)
    expected = np.arange(1, 513, dtype=np.int32).reshape(2, 256)
    for _ in range(4):
        data.fill(0)
        publish_global_with_barrier(data, out)
        np.testing.assert_array_equal(out.to_numpy(), np.roll(expected, -1, axis=1).ravel())


@pytest.mark.parametrize('arch,generator', GENERATORS[1:], ids=[x[0] for x in GENERATORS[1:]])
def test_user_barrier_fences_global_and_shared_memory(arch, generator):
    func, _ = _typed_ir(tack.i32, 'add')
    func.body[0].body = [ir.IRBarrier()]
    source = generator(func)
    if arch == 'metal':
        assert 'mem_flags::mem_threadgroup | mem_flags::mem_device' in source
    elif arch == 'level_zero':
        assert 'CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE' in source
    else:
        assert '__syncthreads()' in source


@tack.func
def inlined_atomic_extrema(data, lo, hi, index):
    tack.atomic_min(lo, 0, data[index])
    tack.atomic_max(hi, 0, data[index])


@tack.func
def nested_atomic_extrema(data, lo, hi, index):
    inlined_atomic_extrema(data, lo, hi, index)


@tack.kernel
def atomic_extrema_through_inlining(data, lo, hi):
    for i in range(data.shape[0]):
        nested_atomic_extrema(data, lo, hi, i)


def test_inlined_atomic_targets_resolve_to_global_fields(backend):
    data = _field([2**31 + 5, 2], tack.u32)
    lo, hi = _field([2**32 - 1], tack.u32), _field([0], tack.u32)
    for _ in range(2):
        atomic_extrema_through_inlining(data, lo, hi)
        assert lo.to_numpy()[0] == 2
        assert hi.to_numpy()[0] == 2**31 + 5
    source = tack.inspect(atomic_extrema_through_inlining, data, lo, hi)
    assert source


@pytest.mark.parametrize('arch,generator', GENERATORS, ids=[x[0] for x in GENERATORS])
def test_atomic_alias_cannot_hide_a_nonfield_target(arch, generator):
    func, node = _typed_ir(tack.f32, 'add')
    func.body[0].body.insert(0, ir.IRAssign('alias', ir.IRName('out')))
    func.body[0].body.insert(1, ir.IRAssign('alias', ir.IRConstant(0)))
    node.field = ir.IRName('alias')
    with pytest.raises(TypeError, match='global field parameter'):
        generator(func)


@pytest.mark.parametrize('dtype', [tack.i64, tack.u64, tack.f64], ids=lambda t: t.name)
@pytest.mark.parametrize('op', KERNELS)
def test_wide_cas_helpers_host_sanitized(dtype, op, tmp_path):
    from tack.codegen.atomics import cuda_atomic64_helpers

    clang = shutil.which('clang++')
    if not clang:
        pytest.skip('requires clang++')
    ctype = {'i64': 'long long', 'u64': 'unsigned long long', 'f64': 'double'}[dtype.name]
    initial = (2**63 - 6 if dtype is tack.i64 else 2**64 - 6) if op == 'add' else (
        np.iinfo(dtype.numpy_dtype).max if op == 'min' and dtype is not tack.f64 else
        99 if op == 'min' else -99 if dtype is not tack.u64 else 0)
    if dtype is tack.f64 and op == 'add':
        initial = 0
    contribution = 0.25 if dtype is tack.f64 and op == 'add' else (
        1 if op == 'add' else -3 if dtype is not tack.u64 else 2**63 + 5)
    other = 8 if op != 'add' else contribution
    expected = _wrap(initial + 8 * 256 * contribution, dtype) if op == 'add' else (
        min(initial, contribution, other) if op == 'min' else max(initial, contribution, other))

    def literal(value):
        return str(int(value)) + ('ull' if dtype is tack.u64 else 'll') if dtype is not tack.f64 else repr(float(value))

    preamble = '''#include <thread>
#include <vector>
#include <cstring>
#define __device__
unsigned long long atomicCAS(unsigned long long* ptr, unsigned long long expected, unsigned long long next) {
    __atomic_compare_exchange_n(ptr, &expected, next, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
}
double __longlong_as_double(long long bits) { double value; std::memcpy(&value, &bits, 8); return value; }
long long __double_as_longlong(double value) { long long bits; std::memcpy(&bits, &value, 8); return bits; }
'''
    program = f'''
int main() {{
    {ctype} output = {literal(initial)};
    std::vector<std::thread> threads;
    for (int k = 0; k < 8; ++k) threads.emplace_back([&output, k]() {{
        for (int j = 0; j < 256; ++j)
            tack_atomic_{op}_{dtype.name}(&output, k % 2 ? {literal(contribution)} : {literal(other)});
    }});
    for (auto& thread : threads) thread.join();
    return output == {literal(expected)} ? 0 : 1;
}}
'''
    source, binary = tmp_path / 'cas.cpp', tmp_path / 'cas'
    source.write_text(preamble + '\n'.join(cuda_atomic64_helpers({(op, dtype)})) + program)
    result = subprocess.run([clang, '-std=c++17', '-O2', '-pthread', '-fsanitize=undefined',
                             '-fno-sanitize-recover=all', str(source), '-o', str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
