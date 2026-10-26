"""bucket_order: the stable order of keys that lead with a bounded id, found by
bucketing instead of sorting -- what face, edge, cell-point and point-link
derivation and contour point merging use. Keys are ids, or tuples of 2 or 4
ids, in i32 or i64."""

import numpy as np
import pytest

import tack
from tack.dataset import buckets
from tack.dataset.buckets import bucket_groups, bucket_order, key_order, run_offsets

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

ID_TYPES = [tack.i32, tack.i64]


def _keys(values, dtype):
    values = np.asarray(values)
    width = values.shape[1] if values.ndim == 2 else 0
    field = (tack.empty((len(values),), tack.vec(width, dtype)) if width
             else tack.empty((len(values),), dtype))
    if len(values):
        tack.copyto(field, values.astype(dtype.numpy_dtype))
    return field


def _stable(columns):
    """np.lexsort's order of rows of ``columns`` (first column most significant),
    ties by index."""
    columns = np.asarray(columns).reshape(len(columns), -1)
    return np.lexsort((np.arange(columns.shape[0]), *columns.T[::-1]))


@pytest.mark.parametrize("dtype", ID_TYPES, ids=lambda t: t.name)
def test_one_key_is_the_stable_order(backend, dtype):
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 3000, 20_000)
    order, starts = bucket_order(_keys(keys, dtype), 3000)
    np.testing.assert_array_equal(order.numpy(), np.argsort(keys, kind="stable"))
    counts = np.bincount(keys, minlength=3000)
    np.testing.assert_array_equal(starts.numpy(), np.concatenate([[0], np.cumsum(counts)]))


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("dtype", ID_TYPES, ids=lambda t: t.name)
def test_tuples_are_the_stable_lexicographic_order(backend, dtype, width):
    rng = np.random.default_rng(width)
    keys = np.c_[rng.integers(0, 3000, 20_000), rng.integers(0, 4, (20_000, width - 1))]
    order, _ = bucket_order(_keys(keys, dtype), 3000)
    np.testing.assert_array_equal(order.numpy(), _stable(keys))
    offsets, runs = run_offsets(_keys(keys, dtype), order)
    distinct = np.unique(keys, axis=0)
    assert runs == len(distinct)
    np.testing.assert_array_equal(np.diff(offsets.numpy()),
                                  np.unique(keys, axis=0, return_counts=True)[1])


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("dtype", ID_TYPES, ids=lambda t: t.name)
def test_key_order_of_few_keys_among_many_buckets(backend, dtype, width):
    """``key_order`` radix sorts keys far fewer than their buckets (a contour's
    crossings among a mesh's points) and buckets the rest: the stable order
    either way, ties included."""
    rng = np.random.default_rng(width)
    for n, nbuckets in ((300, 100_000), (300, 400)):
        keys = np.c_[rng.integers(0, nbuckets, n), rng.integers(0, 4, (n, width - 1))]
        keys[1::2] = keys[0::2]                   # every key twice: ties by index
        values = keys[:, 0] if width == 1 else keys
        order = key_order(_keys(values, dtype), nbuckets, dtype)
        np.testing.assert_array_equal(order.numpy(), _stable(keys))


@pytest.mark.parametrize("indices", ID_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("width", [1, 2, 4])
def test_buckets_either_side_of_the_local_sort(backend, width, indices):
    """Buckets of up to 16 entries are sorted in a local array, larger ones (up
    to the radix sort's threshold) in place: sizes 0 to 40, in reverse order
    within each so that every entry moves, with ties to keep stable."""
    rng = np.random.default_rng(width)
    sizes = np.r_[np.arange(41), 15, 16, 17]
    first = np.repeat(np.arange(len(sizes)), sizes)
    rest = rng.integers(0, 3, (len(first), width - 1))
    keys = np.c_[first, rest][::-1] if width > 1 else first[::-1]
    order, starts = bucket_order(_keys(keys, tack.i32), len(sizes), indices)
    assert order.dtype == indices
    expected = _stable(keys if width > 1 else keys[:, None])
    np.testing.assert_array_equal(order.numpy(), expected)
    np.testing.assert_array_equal(starts.numpy(), np.r_[0, np.cumsum(sizes)])


def test_ids_past_32_bits(backend):
    """Second components past 2**32 compare in full: nothing is packed."""
    keys = np.array([[1, 2**40 + 5], [1, 2**40 + 1], [0, 2**33], [1, 7]])
    order, _ = bucket_order(_keys(keys, tack.i64), 2)
    np.testing.assert_array_equal(order.numpy(), [2, 3, 1, 0])


def test_indices_take_the_type_asked_for(backend):
    order, starts = bucket_order(_keys([2, 0, 1], tack.i32), 3, tack.i64)
    assert order.dtype == starts.dtype == tack.i64
    np.testing.assert_array_equal(order.numpy(), [1, 2, 0])


@pytest.mark.parametrize("width", [1, 2])
def test_a_huge_bucket_falls_back_to_the_radix_sort(backend, monkeypatch, width):
    """Buckets are sorted by insertion, so one with thousands of entries -- a
    point every face shares -- would be quadratic: the radix sort takes over,
    one component at a time."""
    first = np.r_[np.full(2000, 7), np.arange(100)]
    keys = first if width == 1 else np.c_[first, np.r_[np.arange(2000)[::-1], np.zeros(100)]]
    sorted_by = []
    original = buckets.argsort
    monkeypatch.setattr(buckets, "argsort",
                        lambda k, **kw: sorted_by.append(1) or original(k, **kw))
    order, starts = bucket_order(_keys(keys, tack.i32), 100)
    assert len(sorted_by) == width, "the radix sort was not used for each component"
    np.testing.assert_array_equal(order.numpy(), _stable(keys if width == 2 else keys[:, None]))
    assert starts.numpy()[-1] == len(first)


def test_no_keys(backend):
    order, starts = bucket_order(_keys(np.zeros(0, int), tack.i32), 5)
    assert order.shape == (0,)
    np.testing.assert_array_equal(starts.numpy(), np.zeros(6))


def test_a_key_outside_the_buckets_is_refused(backend):
    with pytest.raises(ValueError, match="outside"):
        bucket_order(_keys([[9, 1]], tack.i32), 5)
    with pytest.raises(ValueError, match="outside"):
        bucket_order(_keys([-1], tack.i64), 5)


@pytest.mark.parametrize("dtype", ID_TYPES, ids=lambda t: t.name)
def test_groups_hold_each_buckets_entries(backend, dtype):
    """bucket_groups: the same buckets as bucket_order, each holding the same
    entries in some order -- one bucket far past the size bucket_order sorts
    by insertion, as a locator's points outside its grid are."""
    rng = np.random.default_rng(2)
    keys = np.r_[rng.integers(0, 3000, 20_000), np.full(5000, 3000)]
    rng.shuffle(keys)
    order, starts = bucket_groups(_keys(keys, dtype), 3001)
    order, starts = order.numpy(), starts.numpy()
    counts = np.bincount(keys, minlength=3001)
    np.testing.assert_array_equal(starts, np.concatenate([[0], np.cumsum(counts)]))
    assert sorted(order) == list(range(len(keys)))
    for b in (0, 1234, 2999, 3000):
        group = order[starts[b]:starts[b + 1]]
        np.testing.assert_array_equal(np.sort(group), np.flatnonzero(keys == b))


def test_no_keys_to_group(backend):
    order, starts = bucket_groups(_keys(np.zeros(0, int), tack.i32), 5)
    assert order.shape == (0,)
    np.testing.assert_array_equal(starts.numpy(), np.zeros(6))


@pytest.mark.parametrize("dtype", ID_TYPES, ids=lambda t: t.name)
def test_groups_leave_the_skipped_bucket_empty(backend, dtype):
    """skip: its entries are not placed, and the other buckets are as without it."""
    rng = np.random.default_rng(3)
    keys = np.r_[rng.integers(0, 50, 2000), np.full(700, 50), rng.integers(51, 60, 300)]
    rng.shuffle(keys)
    order, starts = bucket_groups(_keys(keys, dtype), 60, skip=50)
    order, starts = order.numpy(), starts.numpy()
    counts = np.bincount(keys, minlength=60)
    counts[50] = 0
    np.testing.assert_array_equal(starts, np.concatenate([[0], np.cumsum(counts)]))
    placed = order[:starts[-1]]
    assert sorted(placed) == sorted(np.flatnonzero(keys != 50))
    for b in (0, 49, 51, 59):
        group = order[starts[b]:starts[b + 1]]
        np.testing.assert_array_equal(np.sort(group), np.flatnonzero(keys == b))

