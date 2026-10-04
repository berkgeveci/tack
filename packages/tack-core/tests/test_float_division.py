"""Floating // and % preserve precision, divisor sign, and exceptional classes."""

import itertools
import shutil
import subprocess

import numpy as np
import pytest

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.identifiers import kernel_entry_name
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang import ir
from tack.lang.inspect_kernel import _prepare_ir
from tack.lang.ir_traversal import walk_ir

INTS = [tack.i8, tack.u8, tack.i16, tack.u16, tack.i32, tack.u32, tack.i64, tack.u64]
F32_PAIRS = [(tack.f32, tack.f32), *itertools.product([tack.f32], INTS),
             *itertools.product(INTS, [tack.f32])]
F64_PAIRS = [(tack.f64, tack.f64), (tack.f32, tack.f64), (tack.f64, tack.f32),
             *itertools.product([tack.f64], INTS), *itertools.product(INTS, [tack.f64])]


@tack.kernel
def _divmod(a, b, out):
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        out[i, 0] = x // y
        out[i, 1] = x % y
        out[i, 2] = (x // y) + 0.25
        out[i, 3] = (x % y) * 0.5


def _field(values, dtype):
    values = np.asarray(values, dtype=dtype.numpy_dtype)
    field = tack.field(dtype, values.shape)
    field.from_numpy(values)
    return field


def _oracle(a, b, dtype):
    # Cast independently first: NumPy's mixed integer/float promotion differs
    # from Tack's f64-if-present-otherwise-f32 rule.
    a, b = np.asarray(a, dtype=dtype.numpy_dtype), np.asarray(b, dtype=dtype.numpy_dtype)
    with np.errstate(all='ignore'):
        q, r = np.divmod(a, b)
        return np.column_stack([q, r, q + dtype.numpy_dtype.type(0.25),
                                r * dtype.numpy_dtype.type(0.5)])


def _assert_results(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    signed = (expected == 0) | np.isinf(expected)
    np.testing.assert_array_equal(np.signbit(actual[signed]), np.signbit(expected[signed]))
    zero = expected == 0
    np.testing.assert_array_equal(actual[zero], expected[zero])
    finite = np.isfinite(expected) & ~zero
    np.testing.assert_array_max_ulp(actual[finite], expected[finite], maxulp=4)


def _finite_pairs(dtype):
    scalar = dtype.numpy_dtype.type
    values = [0.0, -0.0, 0.1, -0.1, 0.3, -0.3, 1.0, -1.0,
              3.0, -3.0, 6.0, -6.0, 7.5, -7.5, 2.0**30, -(2.0**30)]
    pairs = list(itertools.product(values, [v for v in values if v != 0]))
    # Neighbours of rounded multiples expose floor(a / b) and cancellation.
    for b in [scalar(0.1), scalar(-0.1), scalar(3.0), scalar(-3.0)]:
        for q in [1, 3, 10, 59, 60, 2**20]:
            a = scalar(b * scalar(q))
            pairs.extend((v, b) for v in [a, np.nextafter(a, scalar(-np.inf)),
                                         np.nextafter(a, scalar(np.inf))])
    # Keep the portable set clear of denormal inputs/results; their general
    # device contract is separate. Quotients exceed integer storage widths.
    limit = 100 if dtype is tack.f32 else 900
    pairs.extend([(scalar(2.0**limit), scalar(3.0)),
                  (scalar(-(2.0**limit)), scalar(3.0)),
                  (scalar(2.0**-limit), scalar(2.0**limit)),
                  (scalar(-(2.0**-limit)), scalar(2.0**limit)),
                  (scalar(2.0**limit), scalar(2.0**-limit))])
    rng = np.random.default_rng(5706 + dtype.bits)
    for _ in range(256):
        a, b = rng.uniform(-8, 8, 2) * np.exp2(rng.integers(-15, 15, 2))
        pairs.append((scalar(a), scalar(b)))
    return np.asarray(pairs, dtype=dtype.numpy_dtype)


def _check_pairs(pairs, dtype):
    a, b = (_field(pairs[:, i], dtype) for i in range(2))
    out = tack.field(dtype, (len(pairs), 4))
    _divmod(a, b, out)
    _assert_results(out.to_numpy(), _oracle(pairs[:, 0], pairs[:, 1], dtype))


def test_f32_finite_boundaries_and_large_quotients(backend):
    _check_pairs(_finite_pairs(tack.f32), tack.f32)


def test_f64_finite_boundaries_and_large_quotients(f64_backend):
    _check_pairs(_finite_pairs(tack.f64), tack.f64)


def _exceptional_pairs(dtype):
    return np.asarray(list(itertools.product(
        [0.0, -0.0, 1.0, -1.0, 7.0, -7.0, np.inf, -np.inf, np.nan],
        [1.0, -1.0, 3.0, -3.0, np.inf, -np.inf, np.nan])), dtype=dtype.numpy_dtype)


def test_f32_signed_zero_nan_and_infinity(backend):
    _check_pairs(_exceptional_pairs(tack.f32), tack.f32)


def test_f64_signed_zero_nan_and_infinity(f64_backend):
    _check_pairs(_exceptional_pairs(tack.f64), tack.f64)


def test_signed_zero_expressions_and_nan_guards(backend):
    @tack.kernel
    def expressions(a, b, out):
        for i in range(a.shape[0]):
            x, y = a[i], b[i]
            out[i, 0] = (-x) // y
            out[i, 1] = (-x) % y
            out[i, 2] = x // y if y != 0.0 else 12.0
            out[i, 3] = x % y if x else 13.0
            out[i, 4] = (-0.0) // y
            out[i, 5] = (-0.0) % y

    x = np.asarray([0.0, -0.0, np.nan, 1.0, -1.0], dtype=np.float32)
    y = np.asarray([3.0, -3.0, 2.0, np.nan, np.inf], dtype=np.float32)
    out = tack.field(tack.f32, (len(x), 6))
    expressions(_field(x, tack.f32), _field(y, tack.f32), out)
    with np.errstate(all='ignore'):
        qneg, rneg = np.divmod(-x, y)
        q, r = np.divmod(x, y)
        qzero, rzero = np.divmod(np.float32(-0.0), y)
    expected = np.column_stack([qneg, rneg, np.where(y != 0, q, np.float32(12)),
                                np.where(x.astype(bool), r, np.float32(13)), qzero, rzero])
    _assert_results(out.to_numpy(), expected)


def _mixed_values(dtype):
    if dtype in (tack.f32, tack.f64):
        return [0.25, -0.25, 0.1, -0.1, 3.0, -3.0, 7.5, -7.5]
    info = np.iinfo(dtype.numpy_dtype)
    return sorted({1, 2, 3, 7, int(info.max), int(info.max) - 1,
                   *([int(info.min), -1, -3, -7] if info.min < 0 else [])})


def _check_mixed(left_type, right_type, result_type):
    values = list(itertools.product(_mixed_values(left_type), _mixed_values(right_type)))
    a = _field([x for x, _ in values], left_type)
    b = _field([y for _, y in values], right_type)
    out = tack.field(result_type, (len(values), 4))
    _divmod(a, b, out)
    _assert_results(out.to_numpy(), _oracle(a.to_numpy(), b.to_numpy(), result_type))


@pytest.mark.parametrize('left_type,right_type', F32_PAIRS, ids=lambda t: t.name)
def test_f32_mixed_integer_promotion(backend, left_type, right_type):
    _check_mixed(left_type, right_type, tack.f32)


@pytest.mark.parametrize('left_type,right_type', F64_PAIRS, ids=lambda t: t.name)
def test_f64_mixed_precision_and_integer_promotion(f64_backend, left_type, right_type):
    _check_mixed(left_type, right_type, tack.f64)


@tack.func
def _touch_float(a, state, i):
    state[i] = state[i] + 1
    return a[i] + tack.f32(state[i])


def test_operands_evaluate_once_in_order_and_zero_guards(backend):
    @tack.kernel
    def guarded(a, b, state, out):
        for i in range(a.shape[0]):
            if b[i] != 0.0:
                out[i, 0] = _touch_float(a, state, i) // _touch_float(b, state, i)
                out[i, 1] = _touch_float(a, state, i) % _touch_float(b, state, i)
            else:
                out[i, 0] = 123.0
                out[i, 1] = 456.0
            out[i, 2] = a[i] // b[i] if b[i] != 0.0 else 789.0
            out[i, 3] = a[i] % b[i] if b[i] != 0.0 else 321.0

    a_values = np.asarray([7.0, -7.0, 0.0, 1.0], dtype=np.float32)
    b_values = np.asarray([3.0, 3.0, 0.0, -0.0], dtype=np.float32)
    a, b = _field(a_values, tack.f32), _field(b_values, tack.f32)
    state = _field(np.zeros(4, dtype=np.int32), tack.i32)
    out = tack.field(tack.f32, (4, 4))
    guarded(a, b, state, out)
    expected = np.array([[8.0 // 5.0, 10.0 % 7.0, 7.0 // 3.0, 7.0 % 3.0],
                         [-6.0 // 5.0, -4.0 % 7.0, -7.0 // 3.0, -7.0 % 3.0],
                         [123.0, 456.0, 789.0, 321.0],
                         [123.0, 456.0, 789.0, 321.0]], dtype=np.float32)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    np.testing.assert_array_equal(state.to_numpy(), [4, 4, 0, 0])


def test_destination_f64_does_not_widen_f32_operations(f64_backend):
    @tack.kernel
    def precision(a, b, out):
        for i in range(a.shape[0]):
            out[i, 0] = a[i] // b[i]
            out[i, 1] = a[i] % b[i]
            out[i, 2] = tack.f64(a[i]) // tack.f64(b[i])
            out[i, 3] = tack.f64(a[i]) % tack.f64(b[i])

    a = _field([2**30], tack.f32)
    b = _field([3.0], tack.f32)
    out = tack.field(tack.f64, (1, 4))
    precision(a, b, out)
    q32, r32 = np.divmod(a.to_numpy(), b.to_numpy())
    q64, r64 = np.divmod(a.to_numpy().astype(np.float64), b.to_numpy().astype(np.float64))
    expected = np.column_stack([q32, r32, q64, r64])
    assert expected[0, 0] != expected[0, 2]
    np.testing.assert_array_equal(out.to_numpy(), expected)


@pytest.mark.parametrize('dtype', [tack.f32, tack.f64], ids=lambda t: t.name)
def test_annotations_keep_floating_result_types(dtype):
    tack.init(arch=tack.cpu)
    a, b = _field([7.5], dtype), _field([3.0], dtype)
    out = tack.field(dtype, (1, 4))
    function, _ = _prepare_ir(_divmod, (a, b, out))
    operations = [n for n in walk_ir(function)
                  if isinstance(n, ir.IRBinOp) and n.op in ('//', '%')]
    assert len(operations) == 4
    assert all(n.dtype is dtype for n in operations)
    integer_args = _field([7], tack.i32), _field([3], tack.i32), tack.field(tack.i32, (1, 4))
    integer_function, _ = _prepare_ir(_divmod, integer_args)
    integer_operations = [n for n in walk_ir(integer_function)
                          if isinstance(n, ir.IRBinOp) and n.op in ('//', '%')]
    assert all(n.dtype is tack.i32 for n in integer_operations)


@pytest.mark.parametrize('dtype', [tack.f32, tack.f64], ids=lambda t: t.name)
@pytest.mark.parametrize('generate', [generate_cuda_source, generate_hip_source,
                                     generate_msl_source, generate_opencl_source],
                         ids=['cuda', 'hip', 'metal', 'opencl'])
def test_generators_keep_floating_result_types_and_compile_opencl(dtype, generate, tmp_path):
    if dtype is tack.f64 and generate is generate_msl_source:
        pytest.skip('MSL has no f64')
    tack.init(arch=tack.cpu)
    a, b = _field([7.5], dtype), _field([3.0], dtype)
    out = tack.field(dtype, (1, 4))
    function, _ = _prepare_ir(_divmod, (a, b, out))
    source = generate(function)
    assert f'__tack_floordiv_{dtype.name}__' in source
    assert f'__tack_mod_{dtype.name}__' in source
    if generate is generate_opencl_source:
        clang = shutil.which('clang')
        if clang is None:
            pytest.skip('Clang is required for OpenCL syntax validation')
        path = tmp_path / 'float_division.cl'
        path.write_text(source)
        result = subprocess.run([clang, '-target', 'x86_64-unknown-linux-gnu', '-x', 'cl',
                                 '-cl-std=CL1.2', '-fsyntax-only', str(path)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('dtype', [tack.f32, tack.f64], ids=lambda t: t.name)
@pytest.mark.parametrize('generate', [generate_cuda_source, generate_hip_source],
                         ids=['cuda-host', 'hip-host'])
def test_generated_cpp_executes_boundaries_and_exceptional_inputs(dtype, generate, tmp_path):
    clang = shutil.which('clang++')
    if clang is None:
        pytest.skip('Clang++ is required for generated C++ validation')
    tack.init(arch=tack.cpu)
    pairs = np.concatenate([_finite_pairs(dtype), _exceptional_pairs(dtype)])
    a, b = (_field(pairs[:, i], dtype) for i in range(2))
    out = tack.field(dtype, (len(pairs), 4))
    function, _ = _prepare_ir(_divmod, (a, b, out))
    source = '\n'.join(line for line in generate(function).splitlines()
                       if not line.startswith('#include'))
    ctype = 'float' if dtype is tack.f32 else 'double'
    entry = kernel_entry_name(function.name)
    source = '''
#include <cmath>
#include <cstdio>
#define __global__
#define __device__
struct Grid { unsigned int x; };
Grid blockIdx = {0}, blockDim = {1}, threadIdx = {0};
''' + source + f'''
int main(int argc, char** argv) {{
    {ctype} a[{len(pairs)}], b[{len(pairs)}], out[{len(pairs) * 4}];
    FILE* left = std::fopen(argv[1], "rb");
    FILE* right = std::fopen(argv[2], "rb");
    if (!left || !right) return 2;
    if (std::fread(a, sizeof({ctype}), {len(pairs)}, left) != {len(pairs)}) return 3;
    if (std::fread(b, sizeof({ctype}), {len(pairs)}, right) != {len(pairs)}) return 4;
    std::fclose(left); std::fclose(right);
    for (unsigned int i = 0; i < {len(pairs)}; ++i) {{
        blockIdx.x = i;
        {entry}(a, b, out, {len(pairs)});
    }}
    return std::fwrite(out, sizeof({ctype}), {len(pairs) * 4}, stdout) == {len(pairs) * 4} ? 0 : 5;
}}
'''
    cpp, executable = tmp_path / 'kernel.cpp', tmp_path / 'kernel'
    cpp.write_text(source)
    result = subprocess.run([clang, '-std=c++14', '-O2', '-fsanitize=undefined',
                             '-fno-sanitize-recover=all', str(cpp), '-o', str(executable)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    left, right = tmp_path / 'a.bin', tmp_path / 'b.bin'
    pairs[:, 0].tofile(left)
    pairs[:, 1].tofile(right)
    result = subprocess.run([str(executable), str(left), str(right)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    actual = np.frombuffer(result.stdout, dtype=dtype.numpy_dtype).reshape(-1, 4)
    _assert_results(actual, _oracle(pairs[:, 0], pairs[:, 1], dtype))
