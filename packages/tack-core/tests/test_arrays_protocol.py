"""The arrays of `tack.arrays` are indexable in any kernel.

Every implicit array defines `__getitem__` (its `get`) and `__len__`, so a
plain kernel written for stored arrays takes them as they are: `a[k]`,
`len(a)`, composed to any depth.
"""

import numpy as np
import pytest

import tack
from tack import arrays

pytestmark = pytest.mark.quick


@tack.kernel
def _scaled(a, out, s):
    for i in range(len(a)):
        out[i] = a[i] * s


@tack.kernel
def _vectors(a, out):
    for i in range(len(a)):
        out[i] = a[i]


def _run(a, n, dtype=tack.f32):
    out = tack.zeros((n,), dtype)
    _scaled(a, out, 2)
    return out.numpy()


def test_a_plain_kernel_indexes_implicit_arrays(backend):
    np.testing.assert_allclose(_run(arrays.CountingArray(5, 1.0, 0.5), 5),
                               2 * (1.0 + 0.5 * np.arange(5)))
    np.testing.assert_allclose(_run(arrays.ConstantArray(3.0, 4), 4), np.full(4, 6.0))
    ids = tack.array(np.array([3, 0, 2], np.int32))
    base = tack.array(np.array([10.0, 20.0, 30.0, 40.0], np.float32))
    np.testing.assert_allclose(_run(arrays.Permutation(ids, base), 3), [80, 20, 60])
    joined = arrays.Concatenate(arrays.CountingArray(2, 0.0, 1.0), base)
    np.testing.assert_allclose(_run(joined, 6), 2 * np.array([0, 1, 10, 20, 30, 40]))
    view = arrays.View(base, 1, 2)
    np.testing.assert_allclose(_run(view, 2), [40, 60])


def test_the_same_kernel_takes_a_stored_array(backend):
    stored = tack.array(np.arange(4, dtype=np.float32))
    np.testing.assert_allclose(_run(stored, 4), [0, 2, 4, 6])
    np.testing.assert_allclose(_run(arrays.CountingArray(4, 0.0, 1.0), 4), [0, 2, 4, 6])


def test_len_bounds_the_launch(backend):
    out = tack.full((8,), -1.0)
    _scaled(arrays.CountingArray(3, 1.0, 1.0), out, 1)
    np.testing.assert_allclose(out.numpy(), [1, 2, 3, -1, -1, -1, -1, -1])


def test_vector_arrays(backend):
    x = tack.array(np.array([0.0, 1.0], np.float32))
    y = tack.array(np.array([5.0, 6.0, 7.0], np.float32))
    grid = arrays.CartesianProduct(x, y)
    out = tack.zeros((len(x.numpy()) * 3,), tack.vec3())
    _vectors(grid, out)
    np.testing.assert_allclose(out.numpy(), grid.numpy())
    soa = arrays.Components(arrays.CountingArray(3, 0.0, 1.0), arrays.ConstantArray(9.0, 3))
    out2 = tack.zeros((3,), tack.vec2())
    _vectors(soa, out2)
    np.testing.assert_allclose(out2.numpy(), [[0, 9], [1, 9], [2, 9]])


@tack.func
def _square(k):
    return tack.f32(k) * tack.f32(k)


@tack.func
def _plus_one(v):
    return v + 1.0


def test_functions_and_transforms(backend):
    squares = arrays.Function(_square, 4)
    np.testing.assert_allclose(_run(squares, 4), 2 * np.arange(4) ** 2)
    shifted = arrays.Transform(_plus_one, squares)
    np.testing.assert_allclose(_run(shifted, 4), 2 * (np.arange(4) ** 2 + 1))


def test_a_new_length_does_not_recompile(backend):
    from tack.runtime.dispatch import get_backend
    from tack.runtime.kernel_utils import drop_variants
    drop_variants(_scaled, lambda key: True)
    for n in (3, 5, 9):
        out = tack.zeros((n,))
        _scaled(arrays.CountingArray(n, 0.0, 1.0), out, 1)
        np.testing.assert_allclose(out.numpy(), np.arange(n))
    assert len(get_backend()._cache[_scaled]) == 1


def test_algorithms_take_implicit_arrays(backend):
    from tack import algorithms as alg
    c = arrays.CountingArray(10, 1.0, 1.0)                  # 1 .. 10
    down = arrays.CountingArray(10, 9, -1, tack.i32)        # 9 .. 0
    out = tack.empty((10,))
    assert alg.exclusive_scan(c, out, 10) == 55
    np.testing.assert_allclose(out.numpy(), np.cumsum(np.arange(1, 11)) - np.arange(1, 11))
    np.testing.assert_array_equal(alg.argsort(down).numpy(), np.arange(10)[::-1])
    keys, values = alg.sort_by_key(down, c)
    np.testing.assert_array_equal(keys.numpy(), np.arange(10))
    np.testing.assert_allclose(values.numpy(), np.arange(10, 0, -1))
    np.testing.assert_allclose(alg.gather(c, tack.array([3, 0])).numpy(), [4, 1])
    runs, totals = alg.reduce_by_key(arrays.ConstantArray(2, 4, tack.i32), c)
    np.testing.assert_array_equal(runs.numpy(), [2])
    np.testing.assert_allclose(totals.numpy(), [10])
    assert alg.dot(c, c) == 385 and alg.count_nonzero(c) == 10 and alg.absmax(c) == 10
    assert alg.var(c) == pytest.approx(np.var(np.arange(1, 11)))
    counts, edges = alg.histogram(c, bins=2)
    np.testing.assert_array_equal(counts.numpy(), [5, 5])


def test_algorithms_take_vector_implicit_arrays(backend):
    """A vector implicit array gives what the same vectors stored give. The
    outputs used to be sized by the component type, so the kernels stored a
    vector into a scalar array and failed to compile."""
    from tack import algorithms as alg
    x, y, z = (np.array(v, np.float32) for v in ([1, 1, 4, 4], [2, 2, 5, 0], [3, 3, 6, -2]))
    implicit = arrays.Components(tack.array(x), tack.array(y), tack.array(z))
    stored = tack.array(np.stack([x, y, z], axis=1), dtype=tack.vec3())
    keys = tack.array(np.array([0, 0, 1, 1], np.int32))
    for a in (implicit, stored):
        np.testing.assert_allclose(alg.gather(a, tack.array([2, 0])).numpy(),
                                   [[4, 5, 6], [1, 2, 3]])
        distinct, counts = alg.unique(a)
        np.testing.assert_allclose(distinct.numpy(), [[1, 2, 3], [4, 5, 6], [4, 0, -2]])
        np.testing.assert_array_equal(counts.numpy(), [2, 1, 1])
        runs, totals = alg.reduce_by_key(keys, a)
        np.testing.assert_allclose(totals.numpy(), [[2, 4, 6], [8, 5, 4]])
        flat = np.stack([x, y, z], axis=1).ravel()
        assert alg.norm(a) == pytest.approx(np.linalg.norm(flat))
        assert alg.var(a) == pytest.approx(np.var(flat))
        assert alg.absmax(a) == 6 and alg.count_nonzero(a) == 11


def test_strided_takes_host_data_as_the_other_arrays_do(backend):
    s = arrays.Strided(np.arange(10, dtype=np.float32), 2, 1)
    assert s.dtype is tack.f32 and s.size == 5
    np.testing.assert_array_equal(arrays.materialize(s).numpy(), [1, 3, 5, 7, 9])


@pytest.mark.parametrize("dtype", ["i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64",
                                   "f32", "f64"])
def test_cast_converts_to_every_scalar_type(backend, dtype):
    dtype = getattr(tack, dtype)
    if dtype not in tack.get_default_device()._backend.supported_dtypes:
        pytest.skip(f"{dtype.name} is not supported on {backend}")
    c = arrays.Cast(tack.array(np.array([1.5, 30.0, 2.0], np.float32)), dtype)
    assert c.dtype is dtype
    np.testing.assert_array_equal(arrays.materialize(c).numpy(),
                                  np.array([1.5, 30.0, 2.0]).astype(dtype.numpy_dtype))


def test_reductions_of_implicit_arrays_are_a_stored_arrays(backend):
    c = arrays.CountingArray(10, 1.0, 1.0)
    assert (c.sum(), c.min(), c.max(), c.mean()) == (55, 1, 10, 5.5)
    soa = arrays.Components(arrays.CountingArray(3, 0.0, 1.0), arrays.ConstantArray(9.0, 3))
    assert soa.sum() == 30 and soa.mean() == 5           # every component, as Array
    wide = arrays.CountingArray(3, 2**40, 2**40, tack.i64)
    assert wide.sum() == 6 * 2**40


@tack.kernel
def _fill(dst, n):
    for i in range(n):
        dst[i] = tack.f32(i) + 1.0


@tack.kernel
def _fill3(dst, n):
    for i in range(n):
        dst[i] = tack.vec3(tack.f32(i), 10.0 + i, 20.0 + i)


@tack.kernel
def _bump(dst, n):
    for i in range(n):
        dst[i] += 100.0


def test_writes_go_through_to_stored_arrays(backend):
    x, y, z = (tack.zeros((4,)) for _ in range(3))
    _fill3(arrays.Components(x, y, z), 4)                   # structure of arrays
    np.testing.assert_allclose([x.numpy(), y.numpy(), z.numpy()],
                               [np.arange(4), 10 + np.arange(4), 20 + np.arange(4)])
    b = tack.zeros((6,))
    _fill(arrays.View(b, 2, 3), 3)
    np.testing.assert_allclose(b.numpy(), [0, 0, 1, 2, 3, 0])
    b = tack.zeros((6,))
    _fill(arrays.Strided(b, 2, 1), 3)
    np.testing.assert_allclose(b.numpy(), [0, 1, 0, 2, 0, 3])
    b = tack.zeros((4,))
    _fill(arrays.Permutation(tack.array([3, 1, 0, 2]), b), 4)    # a scatter
    np.testing.assert_allclose(b.numpy(), [3, 2, 4, 1])
    first, second = tack.zeros((2,)), tack.zeros((3,))
    _fill(arrays.Concatenate(first, second), 5)
    np.testing.assert_allclose([*first.numpy(), *second.numpy()], [1, 2, 3, 4, 5])
    v = tack.zeros((3,), tack.vec3())
    _fill(arrays.ExtractComponent(v, 1), 3)
    np.testing.assert_allclose(v.numpy(), [[0, 1, 0], [0, 2, 0], [0, 3, 0]])
    _bump(arrays.Permutation(tack.array([1, 0]), arrays.Components(x, y)), 2)
    np.testing.assert_allclose(x.numpy(), [100, 101, 2, 3])
    np.testing.assert_allclose(y.numpy(), [110, 111, 12, 13])


def test_computed_and_read_only_arrays_refuse_writes(backend):
    with pytest.raises(TypeError, match="storing into a CountingArray"):
        _fill(arrays.CountingArray(3), 3)
    with pytest.raises(TypeError, match="storing into a CountingArray"):
        _fill(arrays.Permutation(tack.array([0, 1]), arrays.CountingArray(3)), 2)
    if backend == "cpu":
        host = np.zeros(4, np.float32)
        read_only = tack.array_from_ptr(host, (4,), tack.f32)
        with pytest.raises(ValueError, match="'dst.base.values' may be written, but its "
                                             "array is read-only"):
            _fill(arrays.View(read_only, 0, 4), 4)
