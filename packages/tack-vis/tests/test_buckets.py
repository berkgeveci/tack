"""bucket_order: the stable order of keys that lead with a bounded id, found by
bucketing instead of sorting -- what face, edge, cell-point and point-link
derivation and contour point merging use."""

import numpy as np
import pytest

import tack
from tack.data import buckets
from tack.data.buckets import bucket_order


def _u64(values):
    field = tack.field(tack.u64, shape=(len(values),))
    if len(values):
        field.from_numpy(np.asarray(values, np.uint64))
    return field


def _keys(rng, n, nbuckets):
    first = rng.integers(0, nbuckets, n).astype(np.uint64)
    second = rng.integers(0, 50, n).astype(np.uint64)       # many equal keys
    return (first << np.uint64(32)) | second


def test_one_key_is_the_stable_order(backend):
    rng = np.random.default_rng(1)
    keys = _keys(rng, 20_000, 3000)
    order, starts = bucket_order(_u64(keys), 3000)
    np.testing.assert_array_equal(order.to_numpy(), np.argsort(keys, kind="stable"))
    counts = np.bincount((keys >> np.uint64(32)).astype(np.int64), minlength=3000)
    np.testing.assert_array_equal(starts.to_numpy(), np.concatenate([[0], np.cumsum(counts)]))


def test_two_keys_are_the_stable_lexicographic_order(backend):
    rng = np.random.default_rng(2)
    hi = _keys(rng, 20_000, 3000)
    lo = rng.integers(0, 4, 20_000).astype(np.uint64)
    order, _ = bucket_order(_u64(hi), 3000, lo=_u64(lo))
    np.testing.assert_array_equal(order.to_numpy(),
                                  np.lexsort((np.arange(20_000), lo, hi)))


def test_buckets_from_other_bits(backend):
    """Contour within cells keys a crossing as cell * 16 + edge: the bucket is the
    cell, from bit 4 up."""
    rng = np.random.default_rng(3)
    keys = (rng.integers(0, 500, 5000) * 16 + rng.integers(0, 12, 5000)).astype(np.uint64)
    order, _ = bucket_order(_u64(keys), 500, shift=4)
    np.testing.assert_array_equal(order.to_numpy(), np.argsort(keys, kind="stable"))


def test_a_huge_bucket_falls_back_to_the_radix_sort(backend, monkeypatch):
    """Buckets are sorted by insertion, so one with thousands of entries -- a
    point every face shares -- would be quadratic: the radix sort takes over."""
    keys = np.concatenate([np.full(2000, 7 << 32, np.uint64) | np.arange(2000)[::-1].astype(
        np.uint64), np.arange(100, dtype=np.uint64) << np.uint64(32)])
    sorted_by = []
    original = buckets.argsort
    monkeypatch.setattr(buckets, "argsort", lambda k: sorted_by.append(1) or original(k))
    order, starts = bucket_order(_u64(keys), 100)
    assert sorted_by, "the radix sort was not used"
    np.testing.assert_array_equal(order.to_numpy(), np.argsort(keys, kind="stable"))
    assert starts.to_numpy()[-1] == keys.size


def test_no_keys(backend):
    order, starts = bucket_order(_u64([]), 5)
    assert order.shape == (0,)
    np.testing.assert_array_equal(starts.to_numpy(), np.zeros(6))


def test_a_key_past_the_last_bucket_is_refused(backend):
    with pytest.raises(ValueError, match="past the last"):
        bucket_order(_u64([(9 << 32) | 1]), 5)
