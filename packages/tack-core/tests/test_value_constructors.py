"""Kernel values built as NumPy and GLSL spell them: tack.vec3(x, y, z),
tack.mat2(...), tack.identity(n); a[()] for a zero-dimensional array."""

import numpy as np
import pytest

import tack
from tack.lang.source_validation import UnsupportedSyntaxError

pytestmark = pytest.mark.quick


@tack.kernel
def _vectors(xs, out):
    for i in range(xs.shape[0]):
        x = xs[i]
        a = tack.vec3(x, 2.0 * x, 3.0)           # three scalars
        b = tack.vec3(x)                         # one scalar, every component
        c = tack.vec3(a + b)                     # a 3-vector
        out[i] = c + tack.vec3(0.0, 0.0, tack.vec2(1.0, 2.0).sum())


def test_vector_constructors(backend):
    xs = tack.array(np.arange(4, dtype=np.float32))
    out = tack.empty((4,), tack.vec3())
    _vectors(xs, out)
    x = np.arange(4, dtype=np.float32)
    np.testing.assert_allclose(out.numpy(), np.stack([2 * x, 3 * x, x + 6.0], 1))


@tack.kernel
def _matrices(v, out, rect):
    for i in range(v.shape[0]):
        a = tack.mat2([1.0, 2.0], [3.0, 4.0])    # rows
        b = tack.mat2(1.0, 0.0, 0.0, 2.0)        # row-major scalars
        c = tack.mat2(v[i], v[i])                # vectors as rows
        out[i] = (a @ b + c + tack.identity(2)) @ v[i]
        rect[i] = tack.mat([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])


def test_matrix_constructors(backend):
    host = np.array([[1, 2], [3, 4]], np.float32)
    v = tack.array(host, tack.vec2())
    out = tack.empty((2,), tack.vec2())
    rect = tack.empty((2,), tack.mat(2, 3))
    _matrices(v, out, rect)
    a = np.array([[1, 2], [3, 4]]) @ np.array([[1, 0], [0, 2]])
    want = [(a + np.stack([row, row]) + np.eye(2)) @ row for row in host]
    np.testing.assert_allclose(out.numpy(), want)
    np.testing.assert_array_equal(rect.numpy(), np.tile([[1, 2, 3], [4, 5, 6]], (2, 1, 1)))


def test_wrong_shapes_name_the_constructor(backend):
    @tack.kernel
    def two_for_three(out):
        for i in range(out.shape[0]):
            out[i] = tack.vec3(1.0, 2.0)

    with pytest.raises(UnsupportedSyntaxError, match=r"tack\.vec3\(\)"):
        two_for_three(tack.empty((1,), tack.vec3()))

    @tack.kernel
    def not_square(out):
        for i in range(out.shape[0]):
            out[i] = tack.mat2([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    with pytest.raises(UnsupportedSyntaxError, match=r"tack\.mat2\(\)"):
        not_square(tack.empty((1,), tack.mat2()))


@tack.kernel
def _bump(counter, out):
    for i in range(out.shape[0]):
        out[i] = counter[()] + i


def test_zero_dimensional_arrays_index_with_an_empty_tuple(backend):
    counter = tack.full((), 5, tack.i32)
    out = tack.empty((3,), tack.i32)
    _bump(counter, out)
    np.testing.assert_array_equal(out.numpy(), [5, 6, 7])
    assert counter[()] == 5


def test_none_is_not_an_index(backend):
    @tack.kernel
    def old(counter, out):
        for i in range(out.shape[0]):
            out[i] = counter[None]

    with pytest.raises(UnsupportedSyntaxError, match=r"a\[\(\)\]"):
        old(tack.zeros((), tack.i32), tack.empty((1,), tack.i32))
    with pytest.raises(TypeError, match=r"a\[\(\)\]"):
        tack.zeros((), tack.i32)[None]


def test_taichi_names_are_gone():
    for name in ("Vector", "Matrix", "template", "data_oriented", "ndrange", "field"):
        assert not hasattr(tack, name), name
