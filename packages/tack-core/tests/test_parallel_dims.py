"""Multi-dimensional parallel loops: ``tack.ndrange`` without division.

A kernel's parallel loop over a two- or three-dimensional ``ndrange``
keeps one index variable per dimension, and each backend binds them from
a launch of the same shape: CPU workers walk their chunk of the flat range
row by row, GPUs launch a 2D or 3D grid. Workgroup kernels and loops of
four or more dimensions keep the flat loop, which recovers the indices by
division; so does a GPU launch whose extents pass the device's grid
limits, through a flag the kernel checks at run time.

Every test compares with NumPy, on every backend reachable here. The CPU
chunk tests call the compiled function on chunk boundaries chosen to start
and end mid-row and to carry across rows and planes, since the dispatcher
picks its own boundaries by timing.
"""

import copy

import numpy as np
import pytest

import tack
from tack.lang import parallel_dims
from tack.lang.parallel_dims import launch_geometry


@tack.kernel
def _fill2(a):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        a[i, j] = i * 1000 + j


@tack.kernel
def _fill3(a, lo):
    for i, j, k in tack.ndrange((lo, a.shape[0]), a.shape[1], (1, a.shape[2])):
        a[i, j, k] = i * 1_000_000 + j * 1000 + k


@tack.kernel
def _scalar_extents(a, rows, cols):
    for i, j in tack.ndrange(rows, cols):
        a[i * cols + j] = i * cols + j + 1


@tack.kernel
def _skip_odd_columns(a):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        if j % 2 == 1:
            continue
        a[i, j] = 1


@tack.kernel
def _row_sums(a, sums):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        tack.atomic_add(sums, i, a[i, j])


@tack.kernel
def _fill4(a):
    for i, j, k, m in tack.ndrange(a.shape[0], a.shape[1], a.shape[2], a.shape[3]):
        a[i, j, k, m] = i * 1000 + j * 100 + k * 10 + m


@tack.kernel
def _nested(a):
    for r in range(a.shape[0]):
        for j, k in tack.ndrange(a.shape[1], a.shape[2]):
            a[r, j, k] = r * 100 + j * 10 + k


def _i32(shape, fill=-1):
    f = tack.field(tack.i32, shape=shape)
    f.fill(fill)
    return f


SHAPES_2D = [(1, 1), (3, 4), (7, 1), (1, 9), (5, 3), (300, 257), (2, 100_000), (100_000, 2)]
SHAPES_3D = [((2, 2, 2), 0), ((4, 5, 6), 1), ((9, 3, 2), 0), ((65, 70, 33), 2), ((3, 1, 500), 0)]


@pytest.mark.parametrize("shape", SHAPES_2D, ids=str)
def test_2d_indices(backend, shape):
    a = _i32(shape)
    _fill2(a)
    i, j = np.indices(shape)
    np.testing.assert_array_equal(a.to_numpy(), i * 1000 + j)


@pytest.mark.parametrize("shape, lo", SHAPES_3D, ids=str)
def test_3d_indices_with_ranges(backend, shape, lo):
    a = _i32(shape)
    _fill3(a, lo)
    i, j, k = np.indices(shape)
    want = np.where((i >= lo) & (k >= 1), i * 1_000_000 + j * 1000 + k, -1)
    np.testing.assert_array_equal(a.to_numpy(), want)


def test_continue_atomics_and_empty_ranges(backend):
    a = _i32((5, 7), 0)
    _skip_odd_columns(a)
    np.testing.assert_array_equal(a.to_numpy(), np.tile(np.arange(7) % 2 == 0, (5, 1)))

    values = np.arange(6 * 50, dtype=np.int32).reshape(6, 50)
    a = tack.field(tack.i32, shape=(6, 50))
    a.from_numpy(values)
    sums = _i32((6,), 0)
    _row_sums(a, sums)
    np.testing.assert_array_equal(sums.to_numpy(), values.sum(axis=1))

    for shape in [(0, 5), (5, 0), (0, 0)]:
        _fill2(_i32(shape))                     # runs nothing, raises nothing
    a = _i32((3, 4, 5))
    _fill3(a, 3)                                # an empty (start, end) pair
    assert (a.to_numpy() == -1).all()


def test_negative_extents_run_nothing(backend):
    """Two negative sizes multiply to a positive count; the launch is empty all the same."""
    a = _i32((12,))
    _scalar_extents(a, -3, -4)
    assert (a.to_numpy() == -1).all()


def test_one_variant_for_every_extent(backend):
    sizes = [(3, 5), (7, 11), (1, 300), (64, 64)]
    for rows, cols in sizes:
        a = _i32((rows * cols,))
        _scalar_extents(a, rows, cols)
        np.testing.assert_array_equal(a.to_numpy(), np.arange(rows * cols) + 1)
    from tack.runtime.dispatch import get_backend
    assert len(get_backend()._cache[_scalar_extents]) == 1


@tack.kernel
def _accumulate(a, n):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        for k in range(n):
            a[i, j] += i + j


def test_a_sequential_loop_storing_inside(backend):
    """Metal compiles such a loop's kernel body as a function of its own (__tack_body__)."""
    a = _i32((6, 9), 0)
    _accumulate(a, 5)
    i, j = np.indices((6, 9))
    np.testing.assert_array_equal(a.to_numpy(), 5 * (i + j))


def test_four_dimensions_and_nested_ndrange_stay_flat(backend):
    a = _i32((2, 3, 4, 5))
    _fill4(a)
    i, j, k, m = np.indices((2, 3, 4, 5))
    np.testing.assert_array_equal(a.to_numpy(), i * 1000 + j * 100 + k * 10 + m)
    a = _i32((3, 4, 5))
    _nested(a)
    r, j, k = np.indices((3, 4, 5))
    np.testing.assert_array_equal(a.to_numpy(), r * 100 + j * 10 + k)


# ── The IR ──────────────────────────────────────────────────────────

def _loop(kernel):
    return parallel_dims.parallel_loop(kernel.get_ir().functions[0])


def test_two_and_three_dimensions_keep_their_dimensions():
    tack.init(arch=tack.cpu)
    for kernel, n in ((_fill2, 2), (_fill3, 3)):
        loop = _loop(kernel)
        assert loop.dims is not None and len(loop.dims) == n
        ops = {node.op for node in _walk(loop.body) if hasattr(node, 'op')}
        assert not ops & {'//', '%'}, f"{kernel.name} still divides: {ops}"
    for kernel in (_fill4, _nested):
        assert _loop(kernel).dims is None


def _walk(stmts):
    from tack.lang.ir_traversal import walk_ir
    return walk_ir(stmts)


@tack.kernel
def _shared_rows(a):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        scratch = tack.shared(tack.i32, 256)
        scratch[tack.thread_id()] = i * 1000 + j
        tack.barrier()
        a[i, j] = scratch[tack.thread_id()]


def test_workgroup_kernels_keep_the_flat_loop():
    tack.init(arch=tack.cpu)
    assert _loop(_shared_rows).dims is None


def test_workgroup_kernels_still_compute_on_workgroup_backends(workgroup_backend):
    a = _i32((16, 32))                  # 512 iterations: whole groups of 256
    _shared_rows(a)
    i, j = np.indices((16, 32))
    np.testing.assert_array_equal(a.to_numpy(), i * 1000 + j)


# ── CPU chunks ──────────────────────────────────────────────────────

def _compile_cpu(kernel, args):
    from tack.lang.ir_optimize import optimize_ir
    from tack.lang.ir_resolve import resolve_ir
    from tack.lang.ir_type_annotate import annotate_types
    from tack.lang.type_inference import infer_param_types
    from tack.runtime.cpu import _compile_kernel
    from tack.runtime.kernel_utils import _get_launch

    ir_func = copy.deepcopy(kernel.get_ir().functions[0])
    _, extents = _get_launch(ir_func, args)
    resolve_ir(ir_func, {p.name: a for p, a in zip(ir_func.params, args)
                         if hasattr(a, "_buffer")})
    infer_param_types(ir_func, tuple(args))
    optimize_ir(ir_func)
    annotate_types(ir_func)
    return _compile_kernel(ir_func), extents


@pytest.mark.parametrize("cuts", [
    [0, 1, 2, 3, 59],                     # one element at a time, then the rest
    [0, 7, 9, 23, 31, 59],                # mid-row starts and ends
    [0, 4, 8, 12, 59],                    # whole rows
    [0, 58, 59],
], ids=str)
def test_cpu_chunks_start_and_end_anywhere_in_2d(cuts):
    tack.init(arch=tack.cpu)
    a = _i32((15, 4))                     # 60 elements, rows of 4
    compiled, extents = _compile_cpu(_fill2, [a])
    prefix = compiled.bind([a], extents)
    for start, end in zip(cuts, [*cuts[1:], 60]):
        compiled.call_range(prefix, start, end)
    i, j = np.indices((15, 4))
    np.testing.assert_array_equal(a.to_numpy(), i * 1000 + j)


def test_cpu_chunks_carry_across_planes_in_3d():
    tack.init(arch=tack.cpu)
    shape = (4, 3, 5)
    a = _i32(shape)
    compiled, extents = _compile_cpu(_fill3, [a, 0])
    prefix = compiled.bind([a, 0], extents)
    total = 4 * 3 * 4                      # k runs from 1, so 4 per row
    for start in range(0, total, 7):       # chunks of 7 cross rows and planes
        compiled.call_range(prefix, start, min(start + 7, total))
    compiled.call_range(prefix, 5, 5)      # an empty chunk does nothing
    i, j, k = np.indices(shape)
    np.testing.assert_array_equal(a.to_numpy(),
                                  np.where(k >= 1, i * 1_000_000 + j * 1000 + k, -1))


def test_cpu_empty_chunk_with_a_zero_extent_does_not_divide():
    """The dispatcher's probes run empty chunks; a zero extent must not reach a division."""
    tack.init(arch=tack.cpu)
    a = _i32((4,))
    compiled, _ = _compile_cpu(_scalar_extents, [a, 0, 0])
    compiled.call_range(compiled.bind([a, 0, 0], (0, 0)), 0, 0)


def test_cpu_fan_out_across_workers():
    """The dispatcher's own fan-out, on worker chunks that end mid-row."""
    tack.init(arch=tack.cpu, num_threads=7)
    backend = tack.runtime.dispatch.get_backend()
    shape = (300, 257)                      # 77100 elements: 7 chunks of 11015
    a = _i32(shape)
    _fill2(a)                               # compile the variant
    compiled = next(iter(backend._cache[_fill2].values())).payload
    _, extents = tack.runtime.kernel_utils._get_launch(
        next(iter(backend._cache[_fill2].values())).ir, [a])
    chunks = []
    real = compiled.call_range
    compiled.call_range = lambda prefix, s, e: (chunks.append((s, e)), real(prefix, s, e))
    a.fill(-1)
    backend._parallel_execute(compiled, compiled.bind([a], extents), 0, 300 * 257)
    del compiled.call_range
    assert len(chunks) == 7 and any(s % 257 for s, _ in chunks)
    i, j = np.indices(shape)
    np.testing.assert_array_equal(a.to_numpy(), i * 1000 + j)


# ── GPU launches ────────────────────────────────────────────────────

@pytest.mark.parametrize("extents, grid, block", [
    ((100, 300), (2, 100, 1), (256, 1, 1)),
    ((100_000, 3), (1, 1563, 1), (4, 64, 1)),
    ((10, 20, 30), (1, 3, 10), (32, 8, 1)),
    ((1000, 1, 1), (1, 1, 4), (1, 1, 256)),
    ((5, 5), (1, 1, 1), (8, 8, 1)),
], ids=str)
def test_launch_geometry(extents, grid, block):
    got = launch_geometry(extents, max_grid=(2**31 - 1, 65535, 65535),
                          max_block=(1024, 1024, 1024))
    assert got == (grid, block, False)
    for size, g, b in zip(extents[::-1], grid, block):
        assert g * b >= size and b <= 256


def test_launch_geometry_falls_back_past_the_grid_limits():
    """Blocks take up to 256 threads along a narrow dimension, so only extents
    past 65535 blocks of them -- about 16.8M along y or z -- need the fallback."""
    limits = {"max_grid": (2**31 - 1, 65535, 65535), "max_block": (1024, 1024, 64)}
    assert launch_geometry((100_000, 1), **limits) == ((1, 391, 1), (1, 256, 1), False)
    grid, block, flat = launch_geometry((20_000_000, 1), **limits)
    assert flat and grid == (78_125, 1, 1) and block == (256, 1, 1)
    grid, block, flat = launch_geometry((3_000_000, 2, 3), **limits)
    assert flat and grid == (-(-18_000_000 // 256), 1, 1)
    # The z block is capped by the device (64 on NVIDIA), not just by 256.
    grid, block, flat = launch_geometry((1000, 1, 1), **limits)
    assert not flat and block == (1, 1, 64) and grid == (1, 1, 16)


@pytest.mark.parametrize("shape", [(70, 3), (5, 6, 7)], ids=str)
def test_gpu_flat_fallback(backend, shape, monkeypatch):
    """With the grid limits shrunk, the kernel's flat fallback computes the same thing."""
    from tack.runtime.dispatch import get_backend
    be = get_backend()
    if not hasattr(be, '_launch_limits') and not hasattr(be, '_compute_props'):
        pytest.skip(f"{backend} launches multi-dimensional grids without limits")
    if hasattr(be, '_launch_limits'):
        monkeypatch.setattr(be, '_launch_limits', {"max_grid": (2**31 - 1, 1, 1),
                                                   "max_block": (1024, 1024, 64)})
    else:
        props = copy.copy(be._compute_props)
        props.maxGroupCountY = props.maxGroupCountZ = 1
        monkeypatch.setattr(be, '_compute_props', props)
    a = _i32(shape)
    if len(shape) == 2:
        _fill2(a)
        i, j = np.indices(shape)
        want = i * 1000 + j
    else:
        _fill3(a, 0)
        i, j, k = np.indices(shape)
        want = np.where(k >= 1, i * 1_000_000 + j * 1000 + k, -1)
    np.testing.assert_array_equal(a.to_numpy(), want)
