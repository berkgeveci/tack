"""Sorting keys that lead with a bounded id, by bucketing.

Every sort the dataset code needs is of u64 keys whose leading bits are a
point or cell id -- a face's smallest point, an edge's low point, a cell
-- so the keys fall into as many buckets as there are points or cells,
each holding a handful of entries. VTK finds faces this way
(``vtkStaticFaceHashLinksTemplate``, keyed by the smallest point id).
``bucket_order`` counts the entries per bucket, scans the counts, scatters
each entry's index into its bucket, and sorts each bucket by (key, index):
the same stable order a radix sort of the whole key gives, in a few
launches, where a radix sort of 64-bit keys made up to seven passes over
all of them, each a count, a scan and a scatter.

Buckets are sorted by insertion, one thread each; a bucket larger than
``_LARGEST`` (a point shared by thousands of faces) would make that
quadratic, so then the radix sort is used instead.
"""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import argsort, gather

_LARGEST = 256          # entries in one bucket, past which the radix sort is used
_CHUNK = 1024           # buckets one thread looks through for the largest


@tack.kernel
def _bucket_counts(hi, shift, counts, bad, n):
    for i in range(n):
        b = tack.i64(hi[i] >> tack.u64(shift))
        if b < counts.shape[0]:
            tack.atomic_add(counts, tack.i32(b), 1)
        else:
            tack.atomic_add(bad, 0, 1)


@tack.kernel
def _largest_counts(counts, largest, chunk, nchunks):
    for c in range(nchunks):
        most = 0
        for b in range(c * chunk, min(c * chunk + chunk, counts.shape[0])):
            most = max(most, counts[b])
        largest[c] = most


@tack.kernel
def _bucket_scatter(hi, shift, starts, fill, order, n):
    for i in range(n):
        b = tack.i32(hi[i] >> tack.u64(shift))
        slot = starts[b] + tack.atomic_add(fill, b, 1)
        order[slot] = i


@tack.kernel
def _close(starts, n, total):
    for i in range(1):
        starts[n] = total


@tack.kernel
def _sort_buckets(hi, starts, order, nbuckets):
    for b in range(nbuckets):
        begin = starts[b]
        for i in range(begin + 1, starts[b + 1]):
            e = order[i]
            key = hi[e]
            j = i - 1
            while j >= begin and (hi[order[j]] > key or (hi[order[j]] == key and order[j] > e)):
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = e


@tack.kernel
def _sort_buckets_2(hi, lo, starts, order, nbuckets):
    for b in range(nbuckets):
        begin = starts[b]
        for i in range(begin + 1, starts[b + 1]):
            e = order[i]
            key = hi[e]
            low = lo[e]
            j = i - 1
            while j >= begin and (
                    hi[order[j]] > key
                    or (hi[order[j]] == key
                        and (lo[order[j]] > low or (lo[order[j]] == low and order[j] > e)))):
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = e


def bucket_order(hi, nbuckets, lo=None, shift=32):
    """``(order, starts)``: the stable permutation sorting entries by ``(hi, lo)`` (or
    ``hi`` alone), as ``argsort`` would, for u64 keys ``hi`` whose bits from ``shift``
    up are a bucket in ``[0, nbuckets)`` -- an i32 field of ``hi.shape[0]`` entries --
    and where each bucket starts in it, ``nbuckets + 1`` offsets."""
    n = int(hi.shape[0])
    starts = tack.field(tack.i32, shape=(nbuckets + 1,))
    if n == 0:
        starts.fill(0)
        return tack.field(tack.i32, shape=(0,)), starts
    counts = tack.zeros(tack.i32, (nbuckets,))
    bad = tack.zeros(tack.i32, (1,))
    _bucket_counts(hi, shift, counts, bad, n)
    if bad[0]:
        raise ValueError(f"{bad[0]} keys lie past the last of {nbuckets} buckets")
    exclusive_scan(counts, starts, nbuckets)
    _close(starts, nbuckets, n)
    nchunks = (nbuckets + _CHUNK - 1) // _CHUNK
    largest = tack.field(tack.i32, shape=(nchunks,))
    _largest_counts(counts, largest, _CHUNK, nchunks)
    if int(largest.to_numpy().max()) > _LARGEST:
        if lo is None:
            return argsort(hi), starts
        by_lo = argsort(lo)
        return gather(by_lo, argsort(gather(hi, by_lo))), starts
    fill = tack.zeros(tack.i32, (nbuckets,))
    order = tack.field(tack.i32, shape=(n,))
    _bucket_scatter(hi, shift, starts, fill, order, n)
    if lo is None:
        _sort_buckets(hi, starts, order, nbuckets)
    else:
        _sort_buckets_2(hi, lo, starts, order, nbuckets)
    return order, starts
