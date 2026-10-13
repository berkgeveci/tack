"""bucket_order: the stable order of keys that lead with a bounded id, found by
bucketing instead of sorting -- what face, edge, cell-point and point-link
derivation and contour point merging use. Keys are ids, or tuples of 2 or 4
ids, in i32 or i64."""

import numpy as np
import pytest

import tack
from tack.data import buckets
from tack.data.buckets import bucket_order, run_offsets

ID_TYPES = [tack.i32, tack.i64]


def _keys(values, dtype):
    values = np.asarray(values)
    width = values.shape[1] if values.ndim == 2 else 0
    field = (tack.Vector.field(width, dtype, shape=(len(values),)) if width
             else tack.field(dtype, shape=(len(values),)))
    if len(values):
        field.from_numpy(values.astype(dtype.numpy_dtype))
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
    np.testing.assert_array_equal(order.to_numpy(), np.argsort(keys, kind="stable"))
    counts = np.bincount(keys, minlength=3000)
    np.testing.assert_array_equal(starts.to_numpy(), np.concatenate([[0], np.cumsum(counts)]))


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("dtype", ID_TYPES, ids=lambda t: t.name)
def test_tuples_are_the_stable_lexicographic_order(backend, dtype, width):
    rng = np.random.default_rng(width)
    keys = np.c_[rng.integers(0, 3000, 20_000), rng.integers(0, 4, (20_000, width - 1))]
    order, _ = bucket_order(_keys(keys, dtype), 3000)
    np.testing.assert_array_equal(order.to_numpy(), _stable(keys))
    offsets, runs = run_offsets(_keys(keys, dtype), order)
    distinct = np.unique(keys, axis=0)
    assert runs == len(distinct)
    np.testing.assert_array_equal(np.diff(offsets.to_numpy()),
                                  np.unique(keys, axis=0, return_counts=True)[1])


def test_ids_past_32_bits(backend):
    """Second components past 2**32 compare in full: nothing is packed."""
    keys = np.array([[1, 2**40 + 5], [1, 2**40 + 1], [0, 2**33], [1, 7]])
    order, _ = bucket_order(_keys(keys, tack.i64), 2)
    np.testing.assert_array_equal(order.to_numpy(), [2, 3, 1, 0])


def test_indices_take_the_type_asked_for(backend):
    order, starts = bucket_order(_keys([2, 0, 1], tack.i32), 3, tack.i64)
    assert order.dtype == starts.dtype == tack.i64
    np.testing.assert_array_equal(order.to_numpy(), [1, 2, 0])


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
    np.testing.assert_array_equal(order.to_numpy(), _stable(keys if width == 2 else keys[:, None]))
    assert starts.to_numpy()[-1] == len(first)


def test_no_keys(backend):
    order, starts = bucket_order(_keys(np.zeros(0, int), tack.i32), 5)
    assert order.shape == (0,)
    np.testing.assert_array_equal(starts.to_numpy(), np.zeros(6))


def test_a_key_outside_the_buckets_is_refused(backend):
    with pytest.raises(ValueError, match="outside"):
        bucket_order(_keys([[9, 1]], tack.i32), 5)
    with pytest.raises(ValueError, match="outside"):
        bucket_order(_keys([-1], tack.i64), 5)
