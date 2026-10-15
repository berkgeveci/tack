"""Python's indexing protocol on device classes.

A `@tack.device_class` that defines `__getitem__`, `__setitem__` and
`__len__` as `@tack.func` methods is indexable in kernels: `obj[i]`,
`obj[i] = v`, `obj[i] += v` and `len(obj)` call them, and an instance tuple
of numbers (`self.shape = (nx, ny)`) is read as `obj.shape[k]`. One kernel
then takes a stored array or any indexable object, each its own variant.
"""

import numpy as np
import pytest

import tack
from tack.runtime.kernel_utils import drop_variants

pytestmark = pytest.mark.quick


@tack.device_class
class Counting:
    def __init__(self, n, start, step):
        self.n, self.start, self.step = n, start, step

    @tack.func
    def __getitem__(self, k):
        return self.start + self.step * k

    @tack.func
    def __len__(self):
        return self.n


@tack.device_class
class Doubled:
    """Holds an indexable and indexes it: `self.base[k]`, `len(self.base)`."""

    def __init__(self, base):
        self.base = base

    @tack.func
    def __getitem__(self, k):
        return 2.0 * self.base[k]

    @tack.func
    def __len__(self):
        return len(self.base)


@tack.device_class
class Grid:
    """A writable 2-D view of a flat array: `k` is the index vector."""

    def __init__(self, values, nx, ny):
        self.values = values
        self.shape = (nx, ny)

    @tack.func
    def __getitem__(self, k):
        return self.values[k[1] * self.shape[0] + k[0]]

    @tack.func
    def __setitem__(self, k, v):
        self.values[k[1] * self.shape[0] + k[0]] = v


class Shifted(Counting):
    """Overrides `__getitem__` and inherits `__len__`, along the MRO."""

    @tack.func
    def __getitem__(self, k):
        return self.start + self.step * k + 100.0


@tack.device_class
class ReadOnly:
    def __init__(self, n):
        self.n = n

    @tack.func
    def __getitem__(self, k):
        return k * 1.0


@tack.device_class
class NotIndexable:
    def __init__(self, n):
        self.n = n


@tack.kernel
def _double(a, out):
    for i in range(out.shape[0]):
        out[i] = a[i] * 2.0


@tack.kernel
def _copy_len(a, out):
    for i in range(len(a)):
        out[i] = a[i]


@tack.kernel
def _bump(g):
    for i, j in tack.ndindex(g.shape[0], g.shape[1]):
        g[i, j] += 1.0
        g[i, j] = g[i, j] * 10.0


def test_one_kernel_takes_an_array_or_an_indexable(backend):
    out = tack.zeros((5,))
    _double(tack.arange(5, dtype=tack.f32), out)
    np.testing.assert_allclose(out.numpy(), [0, 2, 4, 6, 8])
    _double(Counting(5, 1.0, 0.5), out)
    np.testing.assert_allclose(out.numpy(), [2, 3, 4, 5, 6])
    _double(Doubled(Counting(5, 1.0, 0.5)), out)             # nested
    np.testing.assert_allclose(out.numpy(), [4, 6, 8, 10, 12])
    _double(Shifted(5, 1.0, 0.5), out)                       # an override
    np.testing.assert_allclose(out.numpy(), [202, 203, 204, 205, 206])


def test_len_is_a_launch_bound_and_a_value(backend):
    out = tack.zeros((8,))
    _copy_len(Counting(5, 1.0, 1.0), out)
    np.testing.assert_allclose(out.numpy(), [1, 2, 3, 4, 5, 0, 0, 0])
    _copy_len(Doubled(Counting(3, 0.0, 1.0)), out)           # len through nesting
    np.testing.assert_allclose(out.numpy(), [0, 2, 4, 4, 5, 0, 0, 0])

    @tack.kernel
    def length(a, out):
        for i in range(out.shape[0]):
            out[i] = float(len(a)) + float(len(a.base))

    length(Doubled(Counting(7, 0.0, 1.0)), out)
    np.testing.assert_allclose(out.numpy(), np.full(8, 14.0))


def test_stores_augmented_stores_and_a_vector_index(backend):
    values = tack.zeros((6,))
    _bump(Grid(values, 3, 2))
    np.testing.assert_allclose(values.numpy(), np.full(6, 10.0))

    @tack.kernel
    def mark(g):
        for j in range(g.shape[1]):
            g[[0, j]] = 7.0                                  # a vector index, written out

    mark(Grid(values, 3, 2))
    np.testing.assert_allclose(values.numpy(), [7, 10, 10, 7, 10, 10])


def test_an_augmented_store_evaluates_its_index_once(backend):
    values = tack.zeros((4,))
    counter = tack.zeros((1,), tack.i32)

    @tack.kernel
    def once(g, counter):
        for _ in range(1):
            g[[tack.atomic_add(counter, 0, 1), 0]] += 1.0

    once(Grid(values, 4, 1), counter)
    assert counter[0] == 1
    np.testing.assert_allclose(values.numpy(), [1, 0, 0, 0])


def test_tuple_attributes(backend):
    out = tack.zeros((4,))

    @tack.kernel
    def extents(g, out):
        for i in range(1):
            out[0] = float(g.shape[0])
            out[1] = float(g.shape[-1])
            out[2] = float(len(g.shape))

    extents(Grid(tack.zeros((6,)), 3, 2), out)
    np.testing.assert_allclose(out.numpy(), [3, 2, 2, 0])


def test_changing_tuple_values_reuses_the_variant(backend):
    from tack.runtime.dispatch import get_backend
    drop_variants(_bump, lambda key: True)
    for nx, ny in ((3, 2), (4, 4), (2, 5)):
        values = tack.zeros((nx * ny,))
        _bump(Grid(values, nx, ny))
        np.testing.assert_allclose(values.numpy(), np.full(nx * ny, 10.0))
    assert len(get_backend()._cache[_bump]) == 1


def test_a_store_without_setitem_is_refused(backend):
    @tack.kernel
    def write(a):
        for i in range(4):
            a[i] = 0.0

    @tack.kernel
    def update(a):
        for i in range(4):
            a[i] += 1.0

    with pytest.raises(TypeError, match="storing into a ReadOnly, whose class defines "
                                        "no @tack.func __setitem__"):
        write(ReadOnly(4))
    with pytest.raises(TypeError, match="__setitem__"):
        update(ReadOnly(4))


def test_indexing_or_len_without_the_method_is_refused(backend):
    out = tack.zeros((4,))
    with pytest.raises(TypeError, match="indexing a NotIndexable, whose class defines "
                                        "no @tack.func __getitem__"):
        _double(NotIndexable(4), out)
    with pytest.raises(TypeError, match=r"len\(\) of a ReadOnly, whose class defines "
                                        "no @tack.func __len__"):
        _copy_len(ReadOnly(4), out)


def test_an_atomic_on_an_indexable_is_refused(backend):
    @tack.kernel
    def accumulate(a):
        for i in range(4):
            tack.atomic_add(a, i, 1.0)

    with pytest.raises(TypeError, match="a Counting has none to update"):
        accumulate(Counting(4, 0.0, 1.0))


def test_a_tuple_attribute_takes_a_literal_index(backend):
    @tack.kernel
    def runtime_index(g, out, k):
        for i in range(1):
            out[0] = float(g.shape[k])

    @tack.kernel
    def past_the_end(g, out):
        for i in range(1):
            out[0] = float(g.shape[2])

    g = Grid(tack.zeros((6,)), 3, 2)
    with pytest.raises(TypeError, match="literal index from -2 to 1"):
        runtime_index(g, tack.zeros((1,)), 0)
    with pytest.raises(TypeError, match="literal index from -2 to 1"):
        past_the_end(g, tack.zeros((1,)))


def test_entry_names_are_reserved():
    @tack.device_class
    class Clash:
        def __init__(self):
            self.shape__0 = 1

    @tack.kernel
    def touch(c, out):
        for i in range(1):
            out[0] = 1.0

    with pytest.raises(TypeError, match="how a tuple's entries are named"):
        touch(Clash(), tack.zeros((1,)))


@tack.device_class
class Tile:
    shape = (4, 2)                            # a class tuple: compiled in

    def __init__(self, values):
        self.values = values

    @tack.func
    def __getitem__(self, k):
        return self.values[k[1] * self.shape[0] + k[0]]


@tack.kernel
def _tile_total(t, out):
    for i, j in tack.ndindex(t.shape[0], t.shape[1]):
        tack.atomic_add(out, 0, t[i, j] * tack.f32(i + 10 * j))


def test_class_tuples_are_constants(backend):
    out = tack.zeros((1,))
    _tile_total(Tile(tack.ones((8,))), out)
    assert out[0] == sum(i + 10 * j for i in range(4) for j in range(2))
    if backend == "cpu":
        ir = tack.inspect(_tile_total, Tile(tack.ones((8,))), out, mode="ir")
        assert "* 4" in ir                    # the row length is a literal
    swapped = Tile(tack.ones((8,)))
    swapped.shape = (2, 4)                    # an instance override, still compiled in
    out.fill(0)
    _tile_total(swapped, out)
    assert out[0] == sum(i + 10 * j for i in range(2) for j in range(4))


@tack.device_class
class Wrapped:
    """`__len__` returns its array's length, which bounds a launch."""

    def __init__(self, values):
        self.values = values

    @tack.func
    def __getitem__(self, k):
        return self.values[k]

    @tack.func
    def __len__(self):
        return self.values.shape[0]


def test_len_may_return_an_arrays_length(backend):
    from tack.runtime.dispatch import get_backend
    drop_variants(_copy_len, lambda key: True)
    for n in (3, 6):
        out = tack.zeros((8,))
        _copy_len(Wrapped(tack.arange(n, dtype=tack.f32)), out)
        np.testing.assert_allclose(out.numpy(), [*range(n), *[0] * (8 - n)])
    assert len(get_backend()._cache[_copy_len]) == 1
