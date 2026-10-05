"""Collectives require a proven uniform path and complete native workgroups."""

from types import SimpleNamespace

import numpy as np
import pytest

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang.inspect_kernel import _prepare_ir
from tack.runtime.dispatch import get_backend


@tack.kernel
def varying_branch(data, out, n):
    for i in range(n):
        if i % 2 == 0:
            tack.barrier()
        out[i] = data[i]


@tack.kernel
def varying_reduction(data, out, n):
    for i in range(n):
        if tack.thread_id() < 128:
            out[i] = tack.block_sum(data[i])


@tack.kernel
def varying_bound(data, out, n):
    for i in range(n):
        for j in range(i % 3):
            tack.barrier()
        out[i] = data[i]


@tack.kernel
def carried_condition(data, out, n):
    for i in range(n):
        limit = 2
        j = 0
        while j < limit:
            tack.barrier()
            limit = 1 + i % 2
            j += 1
        out[i] = data[i]


@tack.kernel
def loop_break_after_barrier(data, out, n):
    for i in range(n):
        for j in range(3):
            tack.barrier()
            if i % 2:
                break
        out[i] = data[i]


@tack.kernel
def loop_continue(data, out, n):
    for i in range(n):
        for j in range(3):
            if i % 2:
                continue
            tack.barrier()
        out[i] = data[i]


@tack.kernel
def parallel_continue(data, out, n):
    for i in range(n):
        if i % 2:
            continue
        tack.barrier()
        out[i] = data[i]


@tack.kernel
def joined_condition(data, out, n):
    for i in range(n):
        participate = 1
        if i % 2:
            participate = 0
        if participate:
            tack.barrier()
        out[i] = data[i]


@tack.kernel
def memory_condition(data, out, n):
    for i in range(n):
        if data[0] > 0:
            tack.barrier()
        out[i] = data[i]


@tack.kernel
def conditional_reduction(data, out, n):
    for i in range(n):
        out[i] = tack.block_sum(data[i]) if n > 0 else 0.0


@tack.kernel
def short_circuit_reduction(data, out, n):
    for i in range(n):
        out[i] = n > 0 and tack.block_sum(data[i]) > 0


@tack.kernel
def while_reduction_condition(data, out, n):
    for i in range(n):
        j = 0
        while tack.block_sum(data[i]) > j:
            j += 1
        out[i] = data[i]


@tack.func
def early_device_return(value):
    if value % 2:
        return value
    tack.barrier()
    return value


@tack.kernel
def inlined_return(data, out, n):
    for i in range(n):
        out[i] = early_device_return(i)


_INVALID = [varying_branch, varying_reduction, varying_bound, carried_condition,
            loop_break_after_barrier, loop_continue, parallel_continue,
            joined_condition, memory_condition, conditional_reduction,
            short_circuit_reduction, while_reduction_condition, inlined_return]
_GENERATORS = [generate_cuda_source, generate_hip_source,
               generate_msl_source, generate_opencl_source]


def _inputs(n=512):
    data = tack.field(tack.f32, (n,))
    out = tack.field(tack.f32, (n,))
    values = np.arange(n, dtype=np.float32) + 1
    data.from_numpy(values)
    out.fill(-7)
    return data, out, values


@pytest.mark.parametrize('kernel', _INVALID, ids=lambda k: k.name)
@pytest.mark.parametrize('generate', _GENERATORS, ids=lambda f: f.__name__)
def test_direct_generators_reject_unsafe_participation(kernel, generate):
    tack.init(arch=tack.cpu)
    data, out, _ = _inputs()
    prepared, _ = _prepare_ir(kernel, (data, out, 512))
    with pytest.raises(ValueError, match='requires uniform workgroup participation'):
        generate(prepared)


@pytest.mark.parametrize('kernel', _INVALID, ids=lambda k: k.name)
@pytest.mark.parametrize('mode', ['dispatch', 'ir', 'source', 'optimized'])
def test_gpu_rejects_before_compilation(workgroup_backend, kernel, mode, monkeypatch):
    data, out, _ = _inputs()
    backend = get_backend()
    backend._cache.pop(kernel, None)

    def forbidden_build(*args):
        pytest.fail('unsafe collective reached shader compilation')

    monkeypatch.setattr(backend, '_build_variant', forbidden_build)
    with pytest.raises(ValueError, match='requires uniform workgroup participation'):
        if mode == 'dispatch':
            kernel(data, out, 512)
        else:
            tack.inspect(kernel, data, out, 512, mode=mode)
    np.testing.assert_array_equal(out.to_numpy(), np.full(512, -7))
    assert not backend._cache.get(kernel)


@tack.kernel
def full_groups_only(data, out, n):
    for i in range(n):
        tack.barrier()
        out[i] = data[i]


@pytest.mark.parametrize('count', [1, 255, 257, 513])
@pytest.mark.parametrize('mode', ['dispatch', 'ir', 'source', 'optimized'])
def test_cold_partial_groups_rejected(workgroup_backend, count, mode, monkeypatch):
    data, out, _ = _inputs(768)
    backend = get_backend()
    backend._cache.pop(full_groups_only, None)

    def forbidden_build(*args):
        pytest.fail('partial collective group reached shader compilation')

    monkeypatch.setattr(backend, '_build_variant', forbidden_build)
    with pytest.raises(ValueError, match=f'complete 256-lane groups.*count {count}'):
        if mode == 'dispatch':
            full_groups_only(data, out, count)
        else:
            tack.inspect(full_groups_only, data, out, count, mode=mode)
    np.testing.assert_array_equal(out.to_numpy(), np.full(768, -7))


def test_cached_launch_rechecks_count_without_reanalysis(workgroup_backend, monkeypatch):
    from tack.runtime import kernel_utils

    data, out, values = _inputs(768)
    full_groups_only(data, out, 256)

    def forbidden_analysis(*args):
        pytest.fail('cached launch repeated structural participation analysis')

    monkeypatch.setattr(kernel_utils, 'check_workgroup_participation', forbidden_analysis)
    for count in (512, 0, -5, 768):
        out.fill(-7)
        full_groups_only(data, out, count)
        expected = np.full(768, -7, dtype=np.float32)
        expected[:max(count, 0)] = values[:max(count, 0)]
        np.testing.assert_array_equal(out.to_numpy(), expected)
    for count in (1, 255, 257, 513):
        out.fill(-7)
        with pytest.raises(ValueError, match='partial workgroup'):
            full_groups_only(data, out, count)
        np.testing.assert_array_equal(out.to_numpy(), np.full(768, -7))


@tack.kernel
def uniform_control(data, out, n, enabled, rounds):
    for i in range(n):
        if enabled:
            j = 0
            acc = 0.0
            while j < rounds:
                total = tack.block_sum(data[i])
                if total > 0:
                    tack.barrier()
                acc += total
                j += 1
            out[i] = acc
        else:
            out[i] = data[i]


def test_uniform_controls_and_packed_scalars(workgroup_backend):
    data, out, values = _inputs()
    for enabled, rounds in [(1, 2), (0, 2), (1, 3), (1, 0)]:
        # Public inspection uses unpacked IR; runtime codegen checks packs.
        for mode in ('ir', 'source'):
            assert tack.inspect(uniform_control, data, out, 512, enabled, rounds, mode=mode)
        uniform_control(data, out, 512, enabled, rounds)
        expected = (np.repeat(values.reshape(2, 256).sum(axis=1) * rounds, 256)
                    if enabled else values)
        np.testing.assert_array_equal(out.to_numpy(), expected)


def test_repeated_reductions_protect_result_reads(workgroup_backend):
    @tack.kernel
    def kernel(data, out, rounds):
        for i in range(data.shape[0]):
            acc = 0.0
            for j in range(rounds):
                # No user barrier after the collective: it must protect
                # its own result broadcast before reusing shared storage.
                total = tack.block_sum(data[i] + float(j))
                acc += total
            out[i] = acc

    data, out, values = _inputs()
    for rounds in (64, 128):
        kernel(data, out, rounds)
        expected = values.reshape(2, 256).sum(axis=1) * rounds
        expected += 256 * (rounds * (rounds - 1) // 2)
        np.testing.assert_array_equal(out.to_numpy(), np.repeat(expected, 256))


@tack.kernel
def shared_tree(data, out):
    for i in range(data.shape[0]):
        scratch = tack.shared_like(data, 256)
        tid = tack.thread_id()
        scratch[tid] = data[i]
        tack.barrier()
        stride = 128
        while stride > 0:
            if tid < stride:
                scratch[tid] = scratch[tid] + scratch[tid + stride]
            tack.barrier()
            stride //= 2
        out[i] = scratch[0]


def test_varying_memory_updates_rejoin_before_barrier(workgroup_backend):
    data, out, values = _inputs()
    shared_tree(data, out)
    np.testing.assert_array_equal(out.to_numpy(),
                                  np.repeat(values.reshape(2, 256).sum(axis=1), 256))


def test_varying_loop_without_collectives_can_rejoin(workgroup_backend):
    @tack.kernel
    def kernel(data, out):
        for i in range(data.shape[0]):
            for j in range(i % 3):
                if i % 2:
                    break
            tack.barrier()
            out[i] = data[i]

    data, out, values = _inputs()
    kernel(data, out)
    np.testing.assert_array_equal(out.to_numpy(), values)


@pytest.mark.parametrize('count', [1, 255, 257, 513])
def test_shared_and_thread_id_allow_partial_groups(workgroup_backend, count):
    @tack.kernel
    def kernel(out):
        for i in range(out.shape[0]):
            scratch = tack.shared(tack.i32, 256)
            tid = tack.thread_id()
            scratch[tid] = tid
            out[i] = scratch[tid]

    out = tack.field(tack.i32, (count,))
    kernel(out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(count) % 256)


def test_stepped_range_checks_logical_count(workgroup_backend):
    @tack.kernel
    def kernel(out, end):
        for i in range(3, end, 2):
            tack.barrier()
            out[i] = i

    out = tack.field(tack.i32, (768,))
    out.fill(-7)
    kernel(out, 515)  # 256 iterations; the endpoint itself is not divisible by 256.
    expected = np.full(768, -7, np.int32)
    expected[3:515:2] = np.arange(3, 515, 2)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    out.fill(-7)
    with pytest.raises(ValueError, match='iteration count 257'):
        kernel(out, 516)
    np.testing.assert_array_equal(out.to_numpy(), np.full(768, -7))


def test_ndrange_checks_flattened_count(workgroup_backend):
    @tack.kernel
    def kernel(out, width):
        for i, j in tack.ndrange(16, width):
            tack.barrier()
            out[i, j] = i * 1000 + j

    out = tack.field(tack.i32, (16, 20))
    out.fill(-7)
    kernel(out, 16)
    expected = np.full((16, 20), -7, np.int32)
    expected[:, :16] = np.arange(16)[:, None] * 1000 + np.arange(16)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    out.fill(-7)
    with pytest.raises(ValueError, match='iteration count 272'):
        kernel(out, 17)
    np.testing.assert_array_equal(out.to_numpy(), np.full((16, 20), -7))


def test_thread_id_in_condition_is_discovered(workgroup_backend):
    @tack.kernel
    def kernel(out):
        for i in range(out.shape[0]):
            if tack.thread_id() % 2 == 0:
                out[i] = 1
            else:
                out[i] = 2

    out = tack.field(tack.i32, (7,))
    kernel(out)
    np.testing.assert_array_equal(out.to_numpy(), np.where(np.arange(7) % 2 == 0, 1, 2))


@pytest.mark.parametrize('limit', [128, 256, 512])
def test_metal_pipeline_limit_is_enforced(limit):
    from tack.runtime.metal import CompiledMetalKernel

    pipeline = SimpleNamespace(maxTotalThreadsPerThreadgroup=lambda: limit,
                               threadExecutionWidth=lambda: 32)
    if limit < 256:
        with pytest.raises(ValueError, match='require 256 lanes.*selects 128'):
            CompiledMetalKernel(None, None, pipeline, 'collective', [], [],
                                requires_full_workgroups=True)
    else:
        compiled = CompiledMetalKernel(None, None, pipeline, 'collective', [], [],
                                       requires_full_workgroups=True)
        with pytest.raises(ValueError, match='partial workgroup'):
            compiled([], 257)  # Rejected before accessing a command queue.
    ordinary = CompiledMetalKernel(None, None, pipeline, 'ordinary', [], [])
    assert ordinary._workgroup_size == min(limit, 256)


@pytest.mark.parametrize('limit', [128, 256])
def test_level_zero_compiled_limit_is_enforced(limit):
    from tack.runtime.level_zero_backend import CompiledL0Kernel

    if limit < 256:
        with pytest.raises(ValueError, match='require 256 lanes.*selects 128'):
            CompiledL0Kernel(None, None, 'collective', [], [], limit,
                             requires_full_workgroups=True)
    else:
        compiled = CompiledL0Kernel(None, None, 'collective', [], [], limit,
                                    requires_full_workgroups=True)
        with pytest.raises(ValueError, match='partial workgroup'):
            compiled([], 257, None)  # Rejected before loading the driver.


@pytest.mark.parametrize('x_limit,total_limit', [(128, 512), (512, 128)])
def test_level_zero_checks_both_device_limits_before_driver(x_limit, total_limit, monkeypatch):
    from tack.runtime import level_zero_backend as level_zero

    backend = level_zero.LevelZeroBackend.__new__(level_zero.LevelZeroBackend)
    backend._compute_props = SimpleNamespace(maxGroupSizeX=x_limit,
                                             maxTotalGroupSize=total_limit)

    def forbidden_driver():
        pytest.fail('unsupported collective reached the Level Zero driver')

    monkeypatch.setattr(level_zero, '_get_ze', forbidden_driver)
    with pytest.raises(ValueError, match='require 256 lanes.*selects 128'):
        backend._compile_kernel(full_groups_only.get_ir().functions[0])
