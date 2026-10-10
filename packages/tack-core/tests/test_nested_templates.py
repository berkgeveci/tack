"""Templates holding templates: ``self.base.get(k)`` through any depth.

A data-oriented object may hold others as instance attributes. Their methods,
fields, runtime scalars and constants resolve as the outer object's do, and
the whole tree is one specialization: values change without recompiling,
a different nesting compiles anew.
"""

import numpy as np
import pytest

import tack


@tack.data_oriented
class Counting:
    def __init__(self, start, step):
        self.start = start
        self.step = step

    @tack.func
    def get(self, k):
        return self.start + self.step * k


@tack.data_oriented
class Explicit:
    def __init__(self, values):
        self.values = values

    @tack.func
    def get(self, k):
        return self.values[k]


@tack.data_oriented
class Permuted:
    def __init__(self, ids, base):
        self.ids = ids
        self.base = base

    @tack.func
    def get(self, k):
        return self.base.get(self.ids[k])


@tack.data_oriented
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
    field = tack.field(tack.i32, shape=values.shape)
    field.from_numpy(values)
    return field


def _run(arr, n):
    out = tack.field(tack.i32, shape=(n,))
    _read(arr, out)
    return out.to_numpy()


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
    out = tack.field(tack.i32, shape=(3,))
    _read_step(Permuted(_ints([5, 6, 7]), Counting(0, 4)), out)
    np.testing.assert_array_equal(out.to_numpy(), [5, 10, 15])


def test_nested_vector_fields(backend):
    @tack.kernel
    def norms(arr, out):
        for i in range(out.shape[0]):
            out[i] = arr.get(i).norm()

    vectors = tack.Vector.field(3, tack.f32, shape=(2,))
    vectors.from_numpy(np.array([[3, 4, 0], [0, 0, 2]], np.float32))
    out = tack.field(tack.f32, shape=(2,))
    norms(Permuted(_ints([1, 0]), Explicit(vectors)), out)
    np.testing.assert_allclose(out.to_numpy(), [2, 5])


def test_values_do_not_recompile_and_nestings_do(backend):
    @tack.kernel
    def read(arr, out):
        for i in range(out.shape[0]):
            out[i] = arr.get(i)

    ids = _ints([1, 0])
    out = tack.field(tack.i32, shape=(2,))
    read(Permuted(ids, Counting(1, 1)), out)
    read(Permuted(ids, Counting(7, 3)), out)
    np.testing.assert_array_equal(out.to_numpy(), [10, 7])
    assert len(read._ir_cache) == 1
    read(Permuted(ids, Explicit(_ints([4, 9]))), out)
    np.testing.assert_array_equal(out.to_numpy(), [9, 4])
    assert len(read._ir_cache) == 2


def test_a_template_holding_itself_is_refused(backend):
    loop = Permuted(_ints([0]), Counting(0, 1))
    loop.base = loop
    with pytest.raises(TypeError, match="holds itself"):
        _run(loop, 1)
