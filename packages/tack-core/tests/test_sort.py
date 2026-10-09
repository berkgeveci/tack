"""Sorting and segmented reductions, checked against NumPy's stable sort.

The radix sort is portable (no workgroup primitives), so every case runs
on every available backend. The sizes straddle the kernels' chunk size
and the digit-pass count depends on the key range, so the parameters below
cover one, two, four and eight passes as well as the all-equal shortcut.
"""

import numpy as np
import pytest

import tack
from tack import algorithms
from tack.algorithms import sort as sort_module

_KEY_TYPES = [
    (tack.i32, np.int32),
    (tack.u32, np.uint32),
    (tack.i64, np.int64),
    (tack.u64, np.uint64),
]


def _field(dtype, values):
    f = tack.field(dtype=dtype, shape=(len(values),))
    if len(values):
        f.from_numpy(values)
    return f


def _random_keys(rng, np_dtype, n, lo, hi):
    if n == 0:
        return np.zeros(0, dtype=np_dtype)
    return rng.integers(lo, hi, size=n, dtype=np_dtype, endpoint=True)


def _check_argsort(dtype, keys):
    perm = algorithms.argsort(_field(dtype, keys)).to_numpy()
    assert perm.dtype == np.int32
    np.testing.assert_array_equal(perm, np.argsort(keys, kind="stable"))


# --- argsort ------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 3, 255, 256, 257, 1000, 4097],
                         ids=lambda n: f"n{n}")
@pytest.mark.parametrize("dtype, np_dtype", _KEY_TYPES, ids=lambda t: getattr(t, "name", ""))
def test_argsort_matches_a_stable_numpy_sort(backend, dtype, np_dtype, n):
    """Sizes around the chunk boundary, keys with many duplicates so
    stability is exercised, for every key dtype."""
    rng = np.random.default_rng(n)
    keys = _random_keys(rng, np_dtype, n, 0, 50)
    _check_argsort(dtype, keys)


@pytest.mark.parametrize("dtype, np_dtype, lo, hi", [
    (tack.i32, np.int32, -2**31, 2**31 - 1),
    (tack.u32, np.uint32, 0, 2**32 - 1),
    (tack.i64, np.int64, -2**63, 2**63 - 1),
    (tack.u64, np.uint64, 0, 2**64 - 1),
], ids=["i32", "u32", "i64", "u64"])
def test_argsort_over_the_whole_key_range(backend, dtype, np_dtype, lo, hi):
    """Full-width keys take every digit pass; signed keys mix both signs,
    including the extremes."""
    rng = np.random.default_rng(7)
    keys = _random_keys(rng, np_dtype, 3000, lo, hi)
    keys[:4] = [lo, hi, 0, -1 if lo < 0 else 1]
    _check_argsort(dtype, keys)


@pytest.mark.parametrize("dtype, np_dtype", _KEY_TYPES, ids=lambda t: getattr(t, "name", ""))
def test_argsort_of_equal_keys_is_the_identity(backend, dtype, np_dtype):
    keys = np.full(700, 42, dtype=np_dtype)
    _check_argsort(dtype, keys)


def test_argsort_of_a_prefix_ignores_the_rest(backend):
    keys = np.array([9, 3, 5, 1, 8, 7, 2, 6], dtype=np.int32)
    perm = algorithms.argsort(_field(tack.i32, keys), 5).to_numpy()
    np.testing.assert_array_equal(perm, [3, 1, 2, 4, 0])


def test_argsort_passes_depend_on_the_key_spread(backend, monkeypatch):
    """Keys spanning one byte take one digit pass, wherever they sit in the
    key range; the pass count follows the spread, not the key width."""
    passes = []
    counted = sort_module._count_digits

    def counting(*args):
        passes.append(args[3])  # the shift
        counted(*args)

    monkeypatch.setattr(sort_module, "_count_digits", counting)
    rng = np.random.default_rng(3)

    keys = rng.integers(10**9, 10**9 + 200, size=600, dtype=np.int64)
    _check_argsort(tack.i64, keys)
    assert passes == [0]

    passes.clear()
    keys = rng.integers(-2**40, -2**40 + 60000, size=600, dtype=np.int64)
    _check_argsort(tack.i64, keys)
    assert passes == [0, 8]

    passes.clear()
    keys = np.array([2**39, 2**39 + 1, 5, 2**39 - 1], dtype=np.int64)  # high words 0 to 128
    _check_argsort(tack.i64, keys)
    assert len(passes) == 5  # a spread of 40 bits

    passes.clear()
    keys = np.array([2**32 + 2, 2**32 - 2, 2**32, 2**32 - 1, 2**32 + 1], dtype=np.int64)
    _check_argsort(tack.i64, keys)
    assert passes == [0]  # across a 32-bit word boundary, the spread is still 4


def test_argsort_finds_the_key_range_over_many_chunks(backend):
    """The key range is reduced chunk by chunk, then over the chunks' ranges:
    70,000 keys take three levels, and the extremes sit late in the last
    chunks, where a level that dropped a partial chunk would miss them."""
    rng = np.random.default_rng(11)
    keys = rng.integers(-2**40, 2**40, size=70_000, dtype=np.int64)
    keys[-3] = -2**62
    keys[-1] = 2**62
    _check_argsort(tack.i64, keys)


def test_argsort_refuses_other_key_dtypes(backend):
    for dtype in (tack.f32, tack.i16, tack.u8):
        with pytest.raises(TypeError, match="sort keys must be one of"):
            algorithms.argsort(tack.field(dtype=dtype, shape=(4,)))


# --- sort_by_key and gather -----------------------------------------------------

def test_sort_by_key_carries_values_stably(backend):
    keys = np.array([3, 1, 3, 2, 1, 3, 2, 1], dtype=np.int32)
    values = np.arange(8, dtype=np.float32) * 0.5
    skeys, svalues = algorithms.sort_by_key(_field(tack.i32, keys), _field(tack.f32, values))
    order = np.argsort(keys, kind="stable")
    np.testing.assert_array_equal(skeys.to_numpy(), keys[order])
    np.testing.assert_array_equal(svalues.to_numpy(), values[order])
    assert svalues.dtype is tack.f32


@pytest.mark.parametrize("vdtype, np_vdtype", [
    (tack.u8, np.uint8), (tack.i64, np.int64), (tack.f32, np.float32),
], ids=["u8", "i64", "f32"])
def test_sort_by_key_keeps_the_values_dtype(backend, vdtype, np_vdtype):
    rng = np.random.default_rng(11)
    keys = rng.integers(0, 1000, size=1500, dtype=np.uint32)
    values = rng.integers(0, 100, size=1500).astype(np_vdtype)
    _, svalues = algorithms.sort_by_key(_field(tack.u32, keys), _field(vdtype, values))
    assert svalues.dtype is vdtype
    np.testing.assert_array_equal(svalues.to_numpy(), values[np.argsort(keys, kind="stable")])


def test_sort_by_key_without_values(backend):
    keys = np.array([5, -2, 9, -2], dtype=np.int32)
    skeys, svalues = algorithms.sort_by_key(_field(tack.i32, keys))
    np.testing.assert_array_equal(skeys.to_numpy(), [-2, -2, 5, 9])
    assert svalues is None


def test_sort_by_key_of_nothing(backend):
    skeys, svalues = algorithms.sort_by_key(
        tack.field(dtype=tack.i32, shape=(4,)), tack.field(dtype=tack.f32, shape=(4,)), 0)
    assert skeys.size == 0
    assert svalues.size == 0


def test_gather_applies_a_permutation(backend):
    src = _field(tack.f32, np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32))
    idx = _field(tack.i32, np.array([3, 3, 0, 2], dtype=np.int32))
    np.testing.assert_array_equal(algorithms.gather(src, idx).to_numpy(), [13.0, 13.0, 10.0, 12.0])
    np.testing.assert_array_equal(algorithms.gather(src, idx, 2).to_numpy(), [13.0, 13.0])


# --- unique and reduce_by_key ---------------------------------------------------

def test_unique_counts_runs_of_a_sorted_field(backend):
    keys = np.array([1, 1, 1, 4, 4, 6, 9, 9, 9, 9], dtype=np.int32)
    ukeys, counts = algorithms.unique(_field(tack.i32, keys))
    np.testing.assert_array_equal(ukeys.to_numpy(), [1, 4, 6, 9])
    np.testing.assert_array_equal(counts.to_numpy(), [3, 2, 1, 4])
    assert ukeys.dtype is tack.i32
    assert counts.dtype is tack.i32


@pytest.mark.parametrize("n", [1, 255, 256, 257, 3001], ids=lambda n: f"n{n}")
def test_unique_matches_numpy_on_sorted_random_keys(backend, n):
    rng = np.random.default_rng(n)
    keys = np.sort(rng.integers(0, 40, size=n, dtype=np.int64))
    ukeys, counts = algorithms.unique(_field(tack.i64, keys))
    expected, expected_counts = np.unique(keys, return_counts=True)
    np.testing.assert_array_equal(ukeys.to_numpy(), expected)
    np.testing.assert_array_equal(counts.to_numpy(), expected_counts)


def test_unique_groups_adjacent_equal_keys_only(backend):
    """Keys only need equal values adjacent, not ascending; every run is
    kept in order of appearance."""
    keys = np.array([7, 7, 2, 2, 7], dtype=np.int32)
    ukeys, counts = algorithms.unique(_field(tack.i32, keys))
    np.testing.assert_array_equal(ukeys.to_numpy(), [7, 2, 7])
    np.testing.assert_array_equal(counts.to_numpy(), [2, 2, 1])


def test_unique_of_float_keys(backend):
    keys = np.array([0.5, 0.5, 1.5, 2.5, 2.5], dtype=np.float32)
    ukeys, counts = algorithms.unique(_field(tack.f32, keys))
    np.testing.assert_array_equal(ukeys.to_numpy(), [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(counts.to_numpy(), [2, 1, 2])


def test_unique_of_nothing(backend):
    ukeys, counts = algorithms.unique(tack.field(dtype=tack.i32, shape=(3,)), 0)
    assert ukeys.size == 0
    assert counts.size == 0


def _sequential_reduce(keys, values, op):
    """Reduce each run in element order, in the values' dtype, as the
    kernels do; exact for integers and the same rounding for floats."""
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    out = np.empty(len(starts), dtype=values.dtype)
    with np.errstate(over="ignore"):  # u8 sums wrap, as the kernels' do
        for r, (s, e) in enumerate(zip(starts, ends)):
            acc = values[s]
            for v in values[s + 1:e]:
                acc = {"sum": lambda a, b: a + b, "min": min, "max": max}[op](acc, v)
            out[r] = acc
    return keys[starts], out


@pytest.mark.parametrize("op", ["sum", "min", "max"])
@pytest.mark.parametrize("vdtype, np_vdtype", [
    (tack.i32, np.int32), (tack.f32, np.float32), (tack.u8, np.uint8),
], ids=["i32", "f32", "u8"])
def test_reduce_by_key_reduces_each_run_in_order(backend, op, vdtype, np_vdtype):
    rng = np.random.default_rng(5)
    n = 2000
    keys = np.sort(rng.integers(0, 60, size=n, dtype=np.int32))
    values = rng.integers(0, 200, size=n).astype(np_vdtype)
    if np_vdtype is np.float32:
        values = (values / 7.0).astype(np.float32)
    ukeys, reduced = algorithms.reduce_by_key(
        _field(tack.i32, keys), _field(vdtype, values), op=op)
    expected_keys, expected = _sequential_reduce(keys, values, op)
    np.testing.assert_array_equal(ukeys.to_numpy(), expected_keys)
    assert reduced.dtype is vdtype
    np.testing.assert_array_equal(reduced.to_numpy(), expected)


def test_reduce_by_key_over_a_prefix(backend):
    keys = _field(tack.i32, np.array([2, 2, 5, 5, 5, 8], dtype=np.int32))
    values = _field(tack.i32, np.array([1, 2, 3, 4, 5, 6], dtype=np.int32))
    ukeys, sums = algorithms.reduce_by_key(keys, values, 4)
    np.testing.assert_array_equal(ukeys.to_numpy(), [2, 5])
    np.testing.assert_array_equal(sums.to_numpy(), [3, 7])


def test_reduce_by_key_of_nothing(backend):
    ukeys, sums = algorithms.reduce_by_key(
        tack.field(dtype=tack.i32, shape=(3,)), tack.field(dtype=tack.f32, shape=(3,)), 0)
    assert ukeys.size == 0
    assert sums.size == 0


def test_reduce_by_key_refuses_an_unknown_op(backend):
    keys = tack.field(dtype=tack.i32, shape=(3,))
    with pytest.raises(ValueError, match="op must be one of"):
        algorithms.reduce_by_key(keys, keys, op="mean")


# --- bounds ---------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda a, b, n: algorithms.argsort(a, n),
    lambda a, b, n: algorithms.sort_by_key(a, b, n),
    lambda a, b, n: algorithms.gather(b, a, n),
    lambda a, b, n: algorithms.unique(a, n),
    lambda a, b, n: algorithms.reduce_by_key(a, b, n),
], ids=["argsort", "sort_by_key", "gather", "unique", "reduce_by_key"])
def test_a_count_past_any_field_is_refused(backend, call):
    a = tack.field(dtype=tack.i32, shape=(8,))
    b = tack.field(dtype=tack.i32, shape=(8,))
    with pytest.raises(ValueError, match="outside"):
        call(a, b, 9)
    with pytest.raises(ValueError, match="outside"):
        call(a, b, -1)


@pytest.mark.parametrize("call", [
    lambda a, b, n: algorithms.sort_by_key(a, b, n),
    lambda a, b, n: algorithms.reduce_by_key(a, b, n),
], ids=["sort_by_key", "reduce_by_key"])
def test_a_count_past_the_second_field_is_refused(backend, call):
    a = tack.field(dtype=tack.i32, shape=(8,))
    b = tack.field(dtype=tack.i32, shape=(4,))
    with pytest.raises(ValueError, match="outside"):
        call(a, b, 5)  # within a, past b


def test_sort_by_key_refuses_values_shorter_than_the_keys(backend):
    keys = tack.field(dtype=tack.i32, shape=(8,))
    values = tack.field(dtype=tack.f32, shape=(4,))
    with pytest.raises(ValueError, match="outside"):
        algorithms.sort_by_key(keys, values)


# --- the sort as a building block -----------------------------------------------

def test_point_to_cell_links_from_a_sorted_connectivity(backend):
    """The use the sort exists for: invert a connectivity array into the
    cells around each point, as a CSR structure."""
    conn = np.array([0, 1, 2,  1, 2, 3,  2, 3, 0], dtype=np.int32)  # three triangles
    cell_of_entry = np.repeat(np.arange(3, dtype=np.int32), 3)
    points, cells = algorithms.sort_by_key(_field(tack.i32, conn), _field(tack.i32, cell_of_entry))
    upoints, counts = algorithms.unique(points)
    np.testing.assert_array_equal(upoints.to_numpy(), [0, 1, 2, 3])
    np.testing.assert_array_equal(counts.to_numpy(), [2, 2, 3, 2])
    offsets = tack.field(dtype=tack.i32, shape=(4,))
    algorithms.exclusive_scan(counts, offsets, 4)
    np.testing.assert_array_equal(offsets.to_numpy(), [0, 2, 4, 7])
    links = cells.to_numpy()
    assert sorted(links[4:7]) == [0, 1, 2]  # point 2 touches every triangle
