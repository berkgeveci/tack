"""Sorting keys that lead with a bounded id, by bucketing.

Every sort the dataset code needs is of keys that lead with a point or cell
id -- a face's smallest point, an edge's low point, a cell -- so the keys
fall into as many buckets as there are points or cells, each holding a
handful of entries. VTK finds faces this way
(``vtkStaticFaceHashLinksTemplate``, keyed by the smallest point id).
``bucket_order`` counts the entries per bucket, scans the counts, scatters
each entry's index into its bucket, and sorts each bucket by (key, index):
the same stable order a radix sort of the whole key gives, in a few
launches.

A key is an id array, or a vector field of two or four ids (an edge's two
points, a face's four), compared component by component; its first
component is its bucket. Ids are the topology's ``id_dtype``, so a key
takes no more room than the ids it holds, and nothing is packed: 64-bit ids
sort as 32-bit ones do.

Buckets are sorted by insertion, one thread each; a bucket larger than
``_LARGEST`` (a point shared by thousands of faces) would make that
quadratic, so then the radix sort is used instead, one component at a time.
"""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import _index_dtype, argsort, gather
from tack.runtime.dispatch import on_inputs_device

_LARGEST = 256          # entries in one bucket, past which the radix sort is used
_CHUNK = 1024           # buckets one thread looks through for the largest


# ── Keys ────────────────────────────────────────────────────────────
#
# Template wrappers, so one kernel serves every key width: each gives entry
# i's bucket, whether entry i's key comes before entry j's in one bucket (by
# key, then by index, which makes the order stable), and whether two keys are
# equal.

@tack.device_class
class _Keys1:
    def __init__(self, values):
        self.values = values

    @tack.func
    def bucket(self, i):
        return self.values[i]

    @tack.func
    def before(self, i, j):
        # Asked only of entries of one bucket, whose keys are their bucket: the
        # index alone orders them, without reading two keys at random (a third
        # of point_links' time at 20M hexahedra).
        return i < j

    @tack.func
    def same(self, i, j):
        return self.values[i] == self.values[j]


@tack.device_class
class _Keys2:
    def __init__(self, values):
        self.values = values

    @tack.func
    def bucket(self, i):
        return self.values[i][0]

    @tack.func
    def before(self, i, j):
        a = self.values[i]
        b = self.values[j]
        return a[0] < b[0] or (a[0] == b[0] and (a[1] < b[1] or (a[1] == b[1] and i < j)))

    @tack.func
    def same(self, i, j):
        a = self.values[i]
        b = self.values[j]
        return a[0] == b[0] and a[1] == b[1]


@tack.device_class
class _Keys4:
    def __init__(self, values):
        self.values = values

    @tack.func
    def bucket(self, i):
        return self.values[i][0]

    @tack.func
    def before(self, i, j):
        a = self.values[i]
        b = self.values[j]
        less = 0
        if a[0] != b[0]:
            less = 1 if a[0] < b[0] else 0
        elif a[1] != b[1]:
            less = 1 if a[1] < b[1] else 0
        elif a[2] != b[2]:
            less = 1 if a[2] < b[2] else 0
        elif a[3] != b[3]:
            less = 1 if a[3] < b[3] else 0
        else:
            less = 1 if i < j else 0
        return less

    @tack.func
    def same(self, i, j):
        a = self.values[i]
        b = self.values[j]
        return a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]


_KEYS = {1: _Keys1, 2: _Keys2, 4: _Keys4}


def _width(values):
    return getattr(values, "_vector_n", None) or 1


def _count(values):
    """The number of keys."""
    return values.size


def keys_of(values):
    """The template over ``values``: an id array, or a vector field of 2 or 4 ids."""
    width = _width(values)
    if width not in _KEYS:
        raise ValueError(f"keys have 1, 2 or 4 components, not {width}")
    return _KEYS[width](values)


# ── Bucketing ───────────────────────────────────────────────────────

@tack.kernel
def _bucket_counts(keys, counts, bad, n, nbuckets):
    for i in range(n):
        b = keys.bucket(i)
        if b >= 0 and b < nbuckets:
            tack.atomic_add(counts, b, 1)
        else:
            tack.atomic_add(bad, 0, 1)


@tack.kernel
def _largest_counts(counts, largest, chunk, nchunks, nbuckets):
    for c in range(nchunks):
        most = 0
        for b in range(c * chunk, min(c * chunk + chunk, nbuckets)):
            most = max(most, counts[b])
        largest[c] = most


@tack.kernel
def _bucket_scatter(keys, starts, fill, order, n):
    for i in range(n):
        b = keys.bucket(i)
        slot = starts[b] + tack.atomic_add(fill, b, 1)
        order[slot] = i


@tack.kernel
def _close(starts, n, total):
    for i in range(1):
        starts[n] = total


# A bucket of up to 16 entries is read into a local array, sorted there and
# written back once: by insertion in global memory, each shift is a store that
# the next comparison reads back, which on an MI210 cost three times the
# comparisons themselves (faces of 1M hexahedra, about 6 entries a bucket:
# 2.48 ms, 0.76 sorted locally; an H100 0.40 either way). Larger buckets, up to
# _LARGEST, keep the insertion in place.
@tack.kernel
def _sort_buckets(keys, starts, order, nbuckets):
    for b in range(nbuckets):
        begin = starts[b]
        end = starts[b + 1]
        if end - begin <= 16:
            local = tack.local_array_like(order, 16)
            m = end - begin
            for i in range(m):
                local[i] = order[begin + i]
            for i in range(1, m):
                e = local[i]
                j = i - 1
                while j >= 0 and keys.before(e, local[j]):
                    local[j + 1] = local[j]
                    j -= 1
                local[j + 1] = e
            for i in range(m):
                order[begin + i] = local[i]
        else:
            for i in range(begin + 1, end):
                e = order[i]
                j = i - 1
                while j >= begin and keys.before(e, order[j]):
                    order[j + 1] = order[j]
                    j -= 1
                order[j + 1] = e


@tack.kernel
def _component(values, c, out):
    for i in range(out.shape[0]):
        out[i] = values[i][c]


def _radix_order(values, n, indices):
    """The stable order of the keys by a radix sort, one component at a time from the
    last: each sort keeps the order the later components gave ties."""
    width = _width(values)
    if width == 1:
        return argsort(values, index_dtype=indices)
    order = None
    for c in reversed(range(width)):
        component = tack.empty((n,), values.scalar_dtype)
        _component(values, c, component)
        if order is None:
            order = argsort(component, index_dtype=indices)
        else:
            order = gather(order, argsort(gather(component, order), index_dtype=indices))
    return order


@on_inputs_device
def bucket_order(values, nbuckets, index_dtype=None):
    """``(order, starts)``: the stable permutation sorting ``values`` -- an id array,
    or a vector field of 2 or 4 ids compared component by component -- as
    ``argsort`` would, for keys whose first component is a bucket in ``[0,
    nbuckets)``; and where each bucket starts in it, ``nbuckets + 1`` offsets.
    Both are ``index_dtype``, by default ``i32`` unless there are too many entries."""
    keys = keys_of(values)
    n = _count(values)
    indices = _index_dtype(max(n, nbuckets), index_dtype)
    starts = tack.empty((nbuckets + 1,), indices)
    if n == 0:
        starts.fill(0)
        return tack.empty((0,), indices), starts
    counts = tack.zeros((nbuckets,), tack.i32)
    bad = tack.zeros((1,), tack.i32)
    _bucket_counts(keys, counts, bad, n, nbuckets)
    if bad[0]:
        raise ValueError(f"{bad[0]} keys lie outside the {nbuckets} buckets")
    exclusive_scan(counts, starts, nbuckets, total=False, close=True)
    nchunks = (nbuckets + _CHUNK - 1) // _CHUNK
    largest = tack.empty((nchunks,), tack.i32)
    _largest_counts(counts, largest, _CHUNK, nchunks, nbuckets)
    if int(largest.numpy().max()) > _LARGEST:
        return _radix_order(values, n, indices), starts
    fill = tack.zeros((nbuckets,), tack.i32)
    order = tack.empty((n,), indices)
    _bucket_scatter(keys, starts, fill, order, n)
    _sort_buckets(keys, starts, order, nbuckets)
    return order, starts


#: Keys at least this many times fewer than the buckets are radix sorted by
#: ``key_order``: a bucket sort's counts and offsets would be mostly empty.
_SPARSE = 8


@on_inputs_device
def key_order(values, nbuckets, index_dtype=None):
    """The stable permutation sorting ``values`` as ``bucket_order``'s does, without
    the buckets' offsets. Keys much fewer than the buckets (a contour's crossings
    among a mesh's points) are radix sorted, in memory and time proportional to
    the keys; ``bucket_order``'s counts and offsets are proportional to the
    buckets, which at 20M points was 230 MB to sort a few million keys."""
    n = _count(values)
    if n and nbuckets > _SPARSE * n:
        return _radix_order(values, n, _index_dtype(max(n, nbuckets), index_dtype))
    return bucket_order(values, nbuckets, index_dtype)[0]


@tack.kernel
def _tally(keys, skip, counts):
    for i in range(keys.shape[0]):
        if keys[i] != skip:
            tack.atomic_add(counts, keys[i], 1)


@tack.kernel
def _close_counted(starts, counts, n):
    # The end of the last bucket: its start and its count.
    for i in range(1):
        starts[n] = starts[n - 1] + counts[n - 1]


@tack.kernel
def _scatter(keys, skip, starts, fill, order):
    for i in range(keys.shape[0]):
        b = keys[i]
        if b != skip:
            order[starts[b] + tack.atomic_add(fill, b, 1)] = i


@on_inputs_device
def bucket_groups(keys, nbuckets, index_dtype=None, skip=None):
    """``(order, starts)`` as ``bucket_order`` gives them for an id array ``keys``,
    but with each bucket's entries in no particular order (atomics place them):
    for grouping that needs only which entries share a bucket, such as points
    by the bin or the cell they lie in. Counting, a scan and a scatter, without
    reading anything back; every key must lie in ``[0, nbuckets)``, unchecked.

    Entries whose key is ``skip`` are not placed: that bucket comes out empty,
    and ``order`` keeps room for every entry, its tail unused. For a bucket no
    one reads -- a locator's points outside its grid, probe's points in no
    cell -- which would otherwise put all of them on one counter.

    ``bucket_order`` sorts each bucket, and a bucket past ``_LARGEST`` entries
    sends it to the radix sort: the points outside a locator's grid, or in no
    cell, all share one bucket, so a probe took two radix sorts, 50 of its 57
    launches."""
    n = _count(keys)
    indices = _index_dtype(max(n, nbuckets), index_dtype)
    starts = tack.empty((nbuckets + 1,), indices)
    if n == 0:
        starts.fill(0)
        return tack.empty((0,), indices), starts
    skip = -1 if skip is None else int(skip)
    counts = tack.zeros((nbuckets,), tack.i32)
    _tally(keys, skip, counts)
    exclusive_scan(counts, starts, nbuckets, total=False)
    _close_counted(starts, counts, nbuckets)
    counts.fill(0)                      # now each bucket's fill
    order = tack.empty((n,), indices)
    _scatter(keys, skip, starts, counts, order)
    return order, starts


# ── Runs of equal keys ──────────────────────────────────────────────

@tack.kernel
def _flag_runs(keys, order, flags):
    # Runs of equal keys in sorted order, read through the permutation rather
    # than from sorted copies of the keys.
    for i in range(flags.shape[0]):
        flags[i] = 0 if i > 0 and keys.same(order[i], order[i - 1]) else 1


@tack.kernel
def _scatter_run_starts(flags, run_ids, offsets, n, nruns):
    for i in range(n):
        if flags[i] == 1:
            offsets[run_ids[i]] = i
        if i == 0:
            offsets[nruns] = n


@on_inputs_device
def run_offsets(values, order):
    """``(offsets, nruns)``: where each run of equal keys starts among ``values``
    taken in ``order`` (as ``bucket_order`` gives it), ``nruns + 1`` offsets of
    ``order``'s type."""
    n = int(order.shape[0])
    if n == 0:
        offsets = tack.empty((1,), order.dtype)
        offsets.fill(0)
        return offsets, 0
    flags = tack.empty((n,), tack.i32)
    run_ids = tack.empty((n,), order.dtype)
    _flag_runs(keys_of(values), order, flags)
    nruns = exclusive_scan(flags, run_ids, n)
    offsets = tack.empty((nruns + 1,), order.dtype)
    _scatter_run_starts(flags, run_ids, offsets, n, nruns)
    return offsets, nruns
