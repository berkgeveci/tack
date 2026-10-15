"""Templates holding templates: ``self.base.get(k)`` through any depth.

A device-class object may hold others as instance attributes. Their methods,
arrays, runtime scalars and constants resolve as the outer object's do, and
the whole tree is one specialization: values change without recompiling,
a different nesting compiles anew.
"""

import numpy as np
import pytest

import tack


@tack.device_class
class Counting:
    def __init__(self, start, step):
        self.start = start
        self.step = step

    @tack.func
    def get(self, k):
        return self.start + self.step * k


@tack.device_class
class Explicit:
    def __init__(self, values):
        self.values = values

    @tack.func
    def get(self, k):
        return self.values[k]


@tack.device_class
class Permuted:
    def __init__(self, ids, base):
        self.ids = ids
        self.base = base

    @tack.func
    def get(self, k):
        return self.base.get(self.ids[k])


@tack.device_class
class Scaled:
    FACTOR = 3                       # a nested class's constant

    def __init__(self, base):
        self.base = base

    @tack.func
    def get(self, k):
        return self.FACTOR * self.base.get(k)


@tack.kernel
def _read(arr, out):
    for i in range(out.shape[0]):
        out[i] = arr.get(i)


@tack.kernel
def _read_step(arr, out):
    for i in range(out.shape[0]):
        out[i] = arr.base.step * i + arr.ids[i]


def _ints(values):
    values = np.asarray(values, np.int32)
    array = tack.empty(values.shape, tack.i32)
    tack.copyto(array, values)
    return array


def _run(arr, n):
    out = tack.empty((n,), tack.i32)
    _read(arr, out)
    return out.numpy()


def test_one_level(backend):
    ids = _ints([3, 0, 2, 1])
    np.testing.assert_array_equal(_run(Permuted(ids, Counting(10, 5)), 4), [25, 10, 20, 15])
    np.testing.assert_array_equal(_run(Permuted(ids, Explicit(_ints([7, 8, 9, 6]))), 4),
                                  [6, 7, 9, 8])


def test_two_levels_and_constants(backend):
    inner = Permuted(_ints([1, 2, 3, 0]), Counting(0, 1))
    np.testing.assert_array_equal(_run(Permuted(_ints([3, 2, 1, 0]), inner), 4), [0, 3, 2, 1])
    np.testing.assert_array_equal(_run(Scaled(Permuted(_ints([1, 0]), Counting(2, 1))), 2),
                                  [9, 6])


def test_a_kernel_reads_through_the_tree(backend):
    out = tack.empty((3,), tack.i32)
    _read_step(Permuted(_ints([5, 6, 7]), Counting(0, 4)), out)
    np.testing.assert_array_equal(out.numpy(), [5, 10, 15])


def test_nested_vector_arrays(backend):
    @tack.kernel
    def norms(arr, out):
        for i in range(out.shape[0]):
            out[i] = arr.get(i).norm()

    vectors = tack.empty((2,), tack.vec(3, tack.f32))
    tack.copyto(vectors, np.array([[3, 4, 0], [0, 0, 2]], np.float32))
    out = tack.empty((2,))
    norms(Permuted(_ints([1, 0]), Explicit(vectors)), out)
    np.testing.assert_allclose(out.numpy(), [2, 5])


def test_values_do_not_recompile_and_nestings_do(backend):
    @tack.kernel
    def read(arr, out):
        for i in range(out.shape[0]):
            out[i] = arr.get(i)

    ids = _ints([1, 0])
    out = tack.empty((2,), tack.i32)
    read(Permuted(ids, Counting(1, 1)), out)
    read(Permuted(ids, Counting(7, 3)), out)
    np.testing.assert_array_equal(out.numpy(), [10, 7])
    assert len(read._ir_cache) == 1
    read(Permuted(ids, Explicit(_ints([4, 9]))), out)
    np.testing.assert_array_equal(out.numpy(), [9, 4])
    assert len(read._ir_cache) == 2


def test_a_template_holding_itself_is_refused(backend):
    loop = Permuted(_ints([0]), Counting(0, 1))
    loop.base = loop
    with pytest.raises(TypeError, match="holds itself"):
        _run(loop, 1)


@tack.func
def _one(k):
    return 1


@tack.func
def _two(k):
    return 2


@tack.device_class
class _Leaf:
    def __init__(self, f):
        self.get = f


@tack.device_class
class _Clash:
    """``base__get`` and the nested ``base.get`` flatten to one name."""

    def __init__(self, name, value):
        self.base = _Leaf(_one)
        setattr(self, name, value)


@tack.kernel
def _read_both(obj, out):
    for i in range(1):
        out[i] = obj.base__get(i) + obj.base.get(i)


@pytest.mark.parametrize("value", [_two, 2.0], ids=["function", "scalar"])
def test_flattened_names_may_not_collide(backend, value):
    """A function attribute that flattened onto a nested one's name dropped the
    other from the cache key, so replacing it ran stale code (the outside
    review's 1-a). Every kind of attribute is refused on such a collision."""
    out = tack.empty((1,), tack.i32)
    with pytest.raises(Exception, match="flatten to the same name"):
        _read_both(_Clash("base__get", value), out)
