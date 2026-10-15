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
