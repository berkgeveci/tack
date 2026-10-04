"""Unsupported cooperative execution fails before any kernel side effects."""

import numpy as np
import pytest

import tack
from tack.codegen.llvm_gen import LLVMCodeGen
from tack.lang.ir_traversal import clone_ir
from tack.runtime.dispatch import get_backend


@tack.kernel
def shared_kernel(out):
    for i in range(out.shape[0]):
        out[i] = 99
        scratch = tack.shared(tack.i32, 256)
        scratch[0] = i


@tack.kernel
def shared_like_kernel(out):
    for i in range(out.shape[0]):
        out[i] = 99
        scratch = tack.shared_like(out, 256)
        scratch[0] = i


@tack.kernel
def barrier_kernel(out):
    for i in range(out.shape[0]):
        out[i] = 99
        if i < 0:
            tack.barrier()


@tack.kernel
def thread_id_kernel(out):
    for i in range(out.shape[0]):
        out[i] = tack.thread_id()


@tack.kernel
def block_sum_kernel(out):
    for i in range(out.shape[0]):
        out[i] = tack.block_sum(float(i))


@tack.kernel
def block_min_kernel(out):
    for i in range(out.shape[0]):
        out[i] = tack.block_min(float(i))


@tack.kernel
def block_max_kernel(out):
    for i in range(out.shape[0]):
        out[i] = tack.block_max(float(i))


@tack.func
def cooperative_value(value):
    tack.barrier()
    return value


@tack.kernel
def inlined_barrier_kernel(out):
    for i in range(out.shape[0]):
        out[i] = cooperative_value(i)


_UNSUPPORTED = [
    (shared_kernel, 'shared'), (shared_like_kernel, 'shared_like'),
    (barrier_kernel, 'barrier'), (thread_id_kernel, 'thread_id'),
    (block_sum_kernel, 'block_sum'), (block_min_kernel, 'block_min'),
    (block_max_kernel, 'block_max'), (inlined_barrier_kernel, 'barrier'),
]


@pytest.mark.parametrize('kernel,feature', _UNSUPPORTED,
                         ids=[kernel.name for kernel, _ in _UNSUPPORTED])
@pytest.mark.parametrize('mode', ['dispatch', 'ir', 'source', 'optimized', 'llvm'])
def test_cpu_rejects_workgroup_primitives(kernel, feature, mode):
    tack.init(arch=tack.cpu)
    out = tack.field(tack.i32, (256,))
    out.fill(-7)
    message = f"{kernel.name}.*CPU backend does not support workgroup.*{feature}"
    with pytest.raises(RuntimeError, match=message) as error:
        if mode == 'dispatch':
            kernel(out)
        elif mode == 'llvm':
            LLVMCodeGen(clone_ir(kernel.get_ir().functions[0])).generate()
        else:
            tack.inspect(kernel, out, mode=mode)
    if mode == 'dispatch':
        assert isinstance(error.value.__cause__, NotImplementedError)
    else:
        assert isinstance(error.value, NotImplementedError)
    np.testing.assert_array_equal(out.to_numpy(), np.full(256, -7, np.int32))
    assert not get_backend()._cache.get(kernel)


def test_direct_llvm_checks_mutable_ir_despite_cached_features():
    from tack.lang import ir
    from tack.lang.workgroup_support import check_workgroup_support

    func = ir.IRFunction(name='mutable', params=[], body=[])
    check_workgroup_support(func, supports_workgroups=False,
                            backend_label='CPU', cache_features=True)
    generator = LLVMCodeGen(func)
    func.body.append(ir.IRBarrier())
    with pytest.raises(NotImplementedError, match='mutable.*CPU.*barrier'):
        generator.generate()


@tack.kernel
def private_scratch(data, out, total):
    for i in range(data.shape[0]):
        tmp = tack.local_array(tack.i32, 2)
        copy = tack.local_array_like(data, 2)
        tmp[0] = data[i]
        tmp[1] = tmp[0] * 2
        copy[0] = tmp[1]
        out[i] = copy[0]
        tack.atomic_add(total, 0, tmp[0])


def test_cpu_private_arrays_atomics_and_host_reductions(monkeypatch):
    from tack.lang import workgroup_support

    tack.init(arch=tack.cpu)
    data = tack.field(tack.i32, (512,))
    out = tack.field(tack.i32, (512,))
    total = tack.field(tack.i32, (1,))
    values = np.arange(512, dtype=np.int32)
    data.from_numpy(values)
    walks = []
    original = workgroup_support.walk_ir

    def counted_walk(root):
        walks.append(root)
        return original(root)

    monkeypatch.setattr(workgroup_support, 'walk_ir', counted_walk)
    private_scratch(data, out, total)
    initial_walks = len(walks)
    assert initial_walks > 0
    for offset in (1, 17):
        data.from_numpy(values + offset)
        total.fill(0)
        private_scratch(data, out, total)
        np.testing.assert_array_equal(out.to_numpy(), (values + offset) * 2)
        assert total.to_numpy()[0] == np.sum(values + offset)
        assert out.sum() == np.sum((values + offset) * 2)
        assert out.min() == offset * 2
        assert out.max() == (511 + offset) * 2
    assert len(walks) == initial_walks  # Warm dispatch reuses feature metadata.


@tack.kernel
def exchange_neighbors(data, out):
    for i in range(data.shape[0]):
        scratch = tack.shared_like(data, 256)
        tid = tack.thread_id()
        scratch[tid] = data[i]
        tack.barrier()
        out[i] = scratch[(tid + 1) % 256]


@pytest.mark.parametrize('dtype', [tack.i32, tack.f32])
def test_workgroups_exchange_across_lanes_and_reject_cpu(workgroup_backend, dtype):
    data = tack.field(dtype, (512,))
    out = tack.field(dtype, (512,))
    values = np.arange(512, dtype=dtype.numpy_dtype)
    for mode in ('ir', 'source'):
        assert tack.inspect(exchange_neighbors, data, out, mode=mode)
    for offset in (0, 1000):
        data.from_numpy(values + offset)
        exchange_neighbors(data, out)
        expected = np.roll((values + offset).reshape(2, 256), -1, axis=1).ravel()
        np.testing.assert_array_equal(out.to_numpy(), expected)

    # Reusing the same frontend template on another backend must be checked,
    # even after a successful GPU compilation and cached dispatch.
    tack.init(arch=tack.cpu)
    cpu_data = tack.field(dtype, (512,))
    cpu_out = tack.field(dtype, (512,))
    with pytest.raises(RuntimeError, match='CPU.*barrier, shared_like, thread_id'):
        exchange_neighbors(cpu_data, cpu_out)
    np.testing.assert_array_equal(cpu_out.to_numpy(), np.zeros(512))
