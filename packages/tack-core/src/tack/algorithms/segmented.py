"""Segmented operations: reductions, prefix sums and broadcasts over segments.

A collection of variable-length segments is flat ``values`` and CSR ``offsets``:
segment ``k`` is ``values[offsets[k]:offsets[k + 1]]``, ``offsets`` holding
``nseg + 1`` non-decreasing indices (``i32`` or ``i64``), the last the end of the
last segment. Empty segments are kept: they are what start flags alone cannot
say. Elements outside ``[offsets[0], offsets[nseg])`` belong to no segment and
are left alone.

    from tack.algorithms import segment_reduce, segment_scan, segment_broadcast

    sums = segment_reduce(values, offsets)               # one per segment
    mins = segment_reduce(values, offsets, op="min")
    prefix = segment_scan(values, offsets)               # restarts at each segment
    spread = segment_broadcast(sums, offsets, n)         # back to the elements
    ids = segment_ids(offsets, n)                        # each element's segment

**How.** The values go in chunks of ``_CHUNK`` elements, one thread each, on
every backend alike. A thread walks the segments its chunk overlaps: one that
starts and ends inside is finished there; one that runs on past the chunk's end
leaves its part as the chunk's *open* partial, and one that began in an earlier
chunk and ends inside leaves its *head* partial. A segment over several chunks
is then the open partials of those chunks, which lie side by side -- the same
problem a level up, solved the same way -- and its last chunk's head. A level
divides the elements by ``_CHUNK``, so one enormous segment costs a few
launches, not a thread that walks it alone; many small ones cost one pass.
Nothing uses atomics, so results are the same bits on every run and backend:
floating sums group by chunk (``ordered=True`` sums each segment in element
order, one thread per segment, as ``reduce_by_key`` does). Integer sums wrap at
the values' width.

The ``reduce_by_key`` this complements reduces runs one thread each: O(n) work
but O(L) span for a run of L (Blelloch's segmented instructions, *Vector
Models for Data-Parallel Computing*, §4.3 and Chapter 5).
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.arrays import element_type, size_of
from tack.runtime.dispatch import on_inputs_device

__all__ = ["segment_broadcast", "segment_ids", "segment_reduce", "segment_scan"]

_CHUNK = 256          # elements one thread walks

_OPS = {"sum": 0, "min": 1, "max": 2}


# ── Shared walking ──────────────────────────────────────────────────

@tack.func
def _first_overlapping(stops, s1, nseg, i):
    """The first segment whose stop is past ``i``: the first that may hold ``i``
    or start after it (``nseg`` for none). Stops are non-decreasing."""
    lo = tack.i64(0)
    hi = tack.i64(nseg)
    while lo < hi:
        mid = (lo + hi) // 2
        if stops[mid + s1] <= i:
            lo = mid + 1
        else:
            hi = mid
    return lo


@tack.func
def _combine(op, a, b):
    out = a + b
    if op == 1:
        out = min(a, b)
    elif op == 2:
        out = max(a, b)
    return out


# ── Reduce ──────────────────────────────────────────────────────────

@tack.func
def _first_reaching(stops, s1, nseg, i):
    """The first segment whose stop is at or past ``i``: the first a chunk starting
    at ``i`` walks, empty segments at ``i`` included (``nseg`` for none)."""
    lo = tack.i64(0)
    hi = tack.i64(nseg)
    while lo < hi:
        mid = (lo + hi) // 2
        if stops[mid + s1] < i:
            lo = mid + 1
        else:
            hi = mid
    return lo


@tack.kernel
def _reduce_chunks(values, starts, s0, stops, s1, nseg, n, chunk, nchunks, op, out, empty,
                   opened, open_seg, head):
    # Each chunk's segments: finished ones, and the empty ones starting in it (the
    # last chunk's up to n), into out; the one running on past the chunk into
    # opened[c] (its part here, open_seg[c] its id, -1 for none); the one begun
    # before and ending here into head[c].
    for c in range(nchunks):
        lo = tack.i64(c) * chunk
        hi = min(lo + chunk, tack.i64(n))
        last_chunk = 1 if c == nchunks - 1 else 0
        open_seg[c] = -1
        k = _first_reaching(stops, s1, nseg, lo)
        walking = 1
        while walking == 1 and k < nseg:
            b = tack.i64(starts[k + s0])
            e = tack.i64(stops[k + s1])
            if b >= hi and not (last_chunk == 1 and b == hi):
                walking = 0
            elif e <= b:
                if b >= lo:
                    out[k] = empty[0]
                k += 1
            else:
                if e > lo:
                    first = max(b, lo)
                    acc = values[first]
                    for i in range(first + 1, min(e, hi)):
                        acc = _combine(op, acc, values[i])
                    if e > hi:
                        opened[c] = acc
                        open_seg[c] = k
                    elif b < lo:
                        head[c] = acc
                    else:
                        out[k] = acc
                k += 1


@tack.kernel
def _run_flags(open_seg, flags, nchunks):
    # 1 where a run of chunks open on one segment begins.
    for c in range(nchunks):
        f = 0
        if open_seg[c] >= 0 and (c == 0 or open_seg[c - 1] != open_seg[c]):
            f = 1
        flags[c] = f


@tack.kernel
def _runs(open_seg, flags, run_ids, run_starts, run_stops, run_seg, nchunks):
    # Each run's chunks [start, stop) and segment: a segment over several chunks
    # is its open partials there, then the head of the chunk it ends in (stop).
    for c in range(nchunks):
        k = open_seg[c]
        if k >= 0:
            r = run_ids[c] + flags[c] - 1
            if flags[c] == 1:
                run_starts[r] = c
                run_seg[r] = k
            if c == nchunks - 1 or open_seg[c + 1] != k:
                run_stops[r] = c + 1


@tack.kernel
def _finish_runs(run_stops, run_seg, upper, head, op, out, nruns):
    for r in range(nruns):
        out[run_seg[r]] = _combine(op, upper[r], head[run_stops[r]])


@tack.kernel
def _all_empty(out, empty, nseg):
    for k in range(nseg):
        out[k] = empty[0]


@tack.kernel
def _reduce_ordered(values, starts, s0, stops, s1, op, out, empty, nseg):
    # One thread a segment, in element order: reduce_by_key's sums, exactly.
    for k in range(nseg):
        b = tack.i64(starts[k + s0])
        e = tack.i64(stops[k + s1])
        if e <= b:
            out[k] = empty[0]
        else:
            acc = values[b]
            for i in range(b + 1, e):
                acc = _combine(op, acc, values[i])
            out[k] = acc


def _open_runs(open_seg, nchunks):
    """The runs of chunks open on one segment: ``(starts, stops, segments, count)``,
    at most a run a chunk; one scalar is read, the count."""
    flags = tack.empty((nchunks,), tack.i32)
    run_ids = tack.empty((nchunks,), tack.i64)
    _run_flags(open_seg, flags, nchunks)
    nruns = int(exclusive_scan(flags, run_ids, nchunks))
    run_starts = tack.empty((max(nruns, 1),), tack.i64)
    run_stops = tack.empty((max(nruns, 1),), tack.i64)
    run_seg = tack.empty((max(nruns, 1),), tack.i64)
    if nruns:
        _runs(open_seg, flags, run_ids, run_starts, run_stops, run_seg, nchunks)
    return run_starts, run_stops, run_seg, nruns


def _reduce(values, starts, s0, stops, s1, nseg, n, op, out, empty):
    """``out[k]``, segment ``k``'s reduction, for segments given as ranges
    ``[starts[k + s0], stops[k + s1])`` over ``values[:n]`` (ordered, apart, gaps
    allowed): each chunk finishes what it can, and the segments over several
    chunks -- at most one open at a chunk's end -- go a level up, where their open
    partials are the values and the runs of chunks the segments."""
    if n == 0:
        _all_empty(out, empty, nseg)
        return
    nchunks = -(-n // _CHUNK)
    kind = element_type(values)
    opened = tack.empty((nchunks,), kind)
    head = tack.empty((nchunks,), kind)
    open_seg = tack.empty((nchunks,), tack.i64)
    _reduce_chunks(values, starts, s0, stops, s1, nseg, n, _CHUNK, nchunks, op, out, empty,
                   opened, open_seg, head)
    if nchunks == 1:
        return
    run_starts, run_stops, run_seg, nruns = _open_runs(open_seg, nchunks)
    if nruns:
        upper = tack.empty((nruns,), kind)
        _reduce(opened, run_starts, 0, run_stops, 0, nruns, nchunks, op, upper, empty)
        _finish_runs(run_stops, run_seg, upper, head, op, out, nruns)


# ── Scan ────────────────────────────────────────────────────────────

@tack.kernel
def _scan_chunks(values, starts, s0, stops, s1, nseg, n, chunk, nchunks, op, inclusive, out,
                 opened, open_seg):
    # Each segment's running result within the chunk, from the chunk's start for
    # one begun before (the carry comes later); the part of the one running on
    # into opened[c].
    for c in range(nchunks):
        lo = tack.i64(c) * chunk
        hi = min(lo + chunk, tack.i64(n))
        open_seg[c] = -1
        k = _first_reaching(stops, s1, nseg, lo)
        walking = 1
        while walking == 1 and k < nseg:
            b = tack.i64(starts[k + s0])
            e = tack.i64(stops[k + s1])
            if b >= hi:
                walking = 0
            else:
                if e > lo and e > b:
                    first = max(b, lo)
                    acc = values[first]
                    if inclusive == 1:
                        out[first] = acc
                    else:
                        out[first] = acc - acc
                    for i in range(first + 1, min(e, hi)):
                        value = values[i]
                        if inclusive == 1:
                            acc = _combine(op, acc, value)
                            out[i] = acc
                        else:
                            out[i] = acc
                            acc = _combine(op, acc, value)
                    if e > hi:
                        opened[c] = acc
                        open_seg[c] = k
                k += 1


@tack.kernel
def _scan_carry(stops, s1, n, chunk, nchunks, op, inclusive, out, open_seg, carried):
    # A chunk that begins inside the segment open at the chunk before's end: its
    # elements of that segment take, before them, the segment's parts in the
    # chunks before (carried[c - 1], the open partials' inclusive scan).
    for c in range(1, nchunks):
        k = open_seg[c - 1]
        if k >= 0:
            lo = tack.i64(c) * chunk
            last = min(tack.i64(stops[k + s1]), min(lo + chunk, tack.i64(n)))
            carry = carried[c - 1]
            if inclusive == 1:
                for i in range(lo, last):
                    out[i] = _combine(op, carry, out[i])
            else:
                out[lo] = carry
                for i in range(lo + 1, last):
                    out[i] = _combine(op, carry, out[i])


def _scan(values, starts, s0, stops, s1, nseg, n, op, inclusive, out):
    if n == 0:
        return
    nchunks = -(-n // _CHUNK)
    kind = element_type(values)
    opened = tack.empty((nchunks,), kind)
    open_seg = tack.empty((nchunks,), tack.i64)
    _scan_chunks(values, starts, s0, stops, s1, nseg, n, _CHUNK, nchunks, op,
                 1 if inclusive else 0, out, opened, open_seg)
    if nchunks == 1:
        return
    run_starts, run_stops, _, nruns = _open_runs(open_seg, nchunks)
    if nruns:
        carried = tack.empty((nchunks,), kind)
        _scan(opened, run_starts, 0, run_stops, 0, nruns, nchunks, op, True, carried)
        _scan_carry(stops, s1, n, _CHUNK, nchunks, op, 1 if inclusive else 0, out, open_seg,
                    carried)


# ── Broadcast and ids ───────────────────────────────────────────────

@tack.kernel
def _broadcast_chunks(source, starts, stops, names, nseg, n, chunk, nchunks, out, ids):
    # The long segments' elements, by chunk: range k is segment names[k].
    for c in range(nchunks):
        lo = tack.i64(c) * chunk
        hi = min(lo + chunk, tack.i64(n))
        k = _first_overlapping(stops, 0, nseg, lo)
        walking = 1
        while walking == 1 and k < nseg:
            b = tack.i64(starts[k])
            e = tack.i64(stops[k])
            first = max(b, lo)
            if first >= hi:
                walking = 0
            else:
                for i in range(first, min(e, hi)):
                    if ids == 1:
                        out[i] = names[k]
                    else:
                        out[i] = source[names[k]]
                k += 1


# ── Short segments a thread each, long ones by chunks ───────────────
#
# A segment of at most _CHUNK elements is one thread's, in element order (so a
# float sum is the ordered one's): a GPU has a thread a segment, where one a
# chunk of many short segments left most of it idle (2.5M segments of about 4
# elements: 3.8 ms against 0.6 on an M1 Max's GPU). Longer ones go through the
# chunks, which walk only their ranges. The split is by length alone, so the
# result is the same on every backend.

@tack.kernel
def _reduce_short(values, offsets, op, limit, out, empty, long_flags, long_count, nseg):
    for k in range(nseg):
        b = tack.i64(offsets[k])
        e = tack.i64(offsets[k + 1])
        f = 0
        if e <= b:
            out[k] = empty[0]
        elif e - b <= limit:
            acc = values[b]
            for i in range(b + 1, e):
                acc = _combine(op, acc, values[i])
            out[k] = acc
        else:
            f = 1
            tack.atomic_add(long_count, 0, 1)
        long_flags[k] = f


@tack.kernel
def _scan_short(values, offsets, op, inclusive, limit, out, long_flags, long_count, nseg):
    for k in range(nseg):
        b = tack.i64(offsets[k])
        e = tack.i64(offsets[k + 1])
        f = 0
        if e - b > limit:
            f = 1
            tack.atomic_add(long_count, 0, 1)
        elif e > b:
            acc = values[b]
            if inclusive == 1:
                out[b] = acc
            else:
                out[b] = acc - acc
            for i in range(b + 1, e):
                value = values[i]
                if inclusive == 1:
                    acc = _combine(op, acc, value)
                    out[i] = acc
                else:
                    out[i] = acc
                    acc = _combine(op, acc, value)
        long_flags[k] = f


@tack.kernel
def _broadcast_short(source, offsets, limit, out, ids, long_flags, long_count, nseg):
    for k in range(nseg):
        b = tack.i64(offsets[k])
        e = tack.i64(offsets[k + 1])
        f = 0
        if e - b > limit:
            f = 1
            tack.atomic_add(long_count, 0, 1)
        else:
            for i in range(b, e):
                if ids == 1:
                    out[i] = k
                else:
                    out[i] = source[k]
        long_flags[k] = f


@tack.kernel
def _long_ranges(offsets, long_flags, slots, long_starts, long_stops, long_ids, nseg):
    for k in range(nseg):
        if long_flags[k] == 1:
            r = slots[k]
            long_starts[r] = offsets[k]
            long_stops[r] = offsets[k + 1]
            long_ids[r] = k


@tack.kernel
def _scatter_long(long_ids, long_out, out, nlong):
    for r in range(nlong):
        out[long_ids[r]] = long_out[r]


def _long(offsets, long_flags, long_count, nseg):
    """The long segments' ranges and ids, ``(starts, stops, ids, count)``. The count
    (one atomic a long segment, rare) is read first, and their places are scanned
    only if there are any: most calls have none, and a scan of every segment's
    flag cost more than the short segments' own pass on a GPU."""
    nlong = int(long_count[0])
    if not nlong:
        return None, None, None, 0
    slots = tack.empty((nseg,), tack.i64)
    exclusive_scan(long_flags, slots, nseg, total=False)
    long_starts = tack.empty((nlong,), tack.i64)
    long_stops = tack.empty((nlong,), tack.i64)
    long_ids = tack.empty((nlong,), tack.i64)
    _long_ranges(offsets, long_flags, slots, long_starts, long_stops, long_ids, nseg)
    return long_starts, long_stops, long_ids, nlong


# ── Host side ───────────────────────────────────────────────────────

def _segments(offsets):
    """The segment count of CSR ``offsets`` (``nseg + 1`` integers)."""
    if element_type(offsets) not in (tack.i32, tack.i64):
        raise TypeError(f"offsets are i32 or i64, not {element_type(offsets).name}")
    count = size_of(offsets)
    if count < 1:
        raise ValueError("offsets need at least one entry: the end of the last segment")
    return count - 1


def _check(offsets, nseg, n):
    """In debug mode (``tack.set_debug``), offsets non-decreasing from 0 or more to at
    most ``n``: a read of the whole array, as topologies check their ids."""
    from tack.runtime import dispatch

    if not dispatch._debug:
        return
    host = np.asarray(offsets.numpy()).reshape(-1)
    if host[0] < 0 or host[-1] > n or (nseg and (np.diff(host) < 0).any()):
        raise ValueError(f"offsets must be non-decreasing within [0, {n}]: "
                         f"from {host[0]} to {host[-1]}")


def _identity(op, kind, empty):
    """An empty segment's value: ``empty`` if given, else 0 for a sum and, for min and
    max, the type's greatest and least (infinities for floating types)."""
    if empty is not None:
        return empty
    if op == "sum":
        return 0
    numpy_dtype = np.dtype(kind.scalar_dtype.numpy_dtype if hasattr(kind, "scalar_dtype")
                           else kind.numpy_dtype)
    if numpy_dtype.kind == "f":
        return float("inf") if op == "min" else float("-inf")
    info = np.iinfo(numpy_dtype)
    return int(info.max) if op == "min" else int(info.min)


@on_inputs_device
def segment_reduce(values, offsets, op="sum", out=None, empty=None, ordered=False):
    """Each segment's reduction: ``out[k]`` is the sum (``op="min"``, ``"max"``: the
    least, the greatest) of ``values[offsets[k]:offsets[k + 1]]``, for the
    ``size(offsets) - 1`` segments, in ``values``' element type (a vector's
    componentwise). An empty segment gets ``empty``: by default 0 for a sum, and
    the type's greatest or least -- an infinity for floats -- for min and max.

    Floating sums group by chunks of 256 elements, the same on every backend
    and run; ``ordered=True`` sums each segment in element order, one thread a
    segment, as ``reduce_by_key`` does (a long segment is then one thread's
    loop). Integer sums wrap at the values' width; min and max are exact either
    way. Returns ``out`` (made if not given)."""
    if op not in _OPS:
        raise ValueError(f"op is one of {', '.join(_OPS)}, not {op!r}")
    nseg = _segments(offsets)
    n = size_of(values)
    _check(offsets, nseg, n)
    kind = element_type(values)
    if out is None:
        out = tack.empty((nseg,), kind)
    elif size_of(out) < nseg:
        raise ValueError(f"out has {size_of(out)} elements for {nseg} segments")
    if not nseg:
        return out
    # One element of the values' type, a vector's every component the identity.
    identity = tack.full((1,), _identity(op, kind, empty), kind)
    if ordered:
        _reduce_ordered(values, offsets, 0, offsets, 1, _OPS[op], out, identity, nseg)
        return out
    long_flags = tack.empty((nseg,), tack.i32)
    long_count = tack.zeros((1,), tack.i32)
    _reduce_short(values, offsets, _OPS[op], _CHUNK, out, identity, long_flags, long_count,
                  nseg)
    starts, stops, ids, nlong = _long(offsets, long_flags, long_count, nseg)
    if nlong:
        long_out = tack.empty((nlong,), kind)
        _reduce(values, starts, 0, stops, 0, nlong, n, _OPS[op], long_out, identity)
        _scatter_long(ids, long_out, out, nlong)
    return out


@on_inputs_device
def segment_scan(values, offsets, op="sum", inclusive=True, out=None):
    """Each segment's prefix sums (``op="min"``, ``"max"``: running least or
    greatest), restarting at every segment: ``out[i]`` for ``i`` in segment ``k`` is
    the reduction of ``values[offsets[k]:i + 1]`` (``inclusive``), or of
    ``values[offsets[k]:i]`` with zero at the segment's first element (an
    exclusive sum; exclusive is for sums). Elements outside every segment are
    left as they are. ``out`` may be ``values``. Floating sums group by chunks of
    256 elements, the same on every backend and run. Returns ``out``."""
    if op not in _OPS:
        raise ValueError(f"op is one of {', '.join(_OPS)}, not {op!r}")
    if not inclusive and op != "sum":
        raise ValueError("an exclusive scan is a sum's: op='sum'")
    nseg = _segments(offsets)
    n = size_of(values)
    _check(offsets, nseg, n)
    if out is None:
        out = tack.empty((n,), element_type(values))
    elif size_of(out) < n:
        raise ValueError(f"out has {size_of(out)} elements for {n} values")
    if nseg and n:
        long_flags = tack.empty((nseg,), tack.i32)
        long_count = tack.zeros((1,), tack.i32)
        _scan_short(values, offsets, _OPS[op], 1 if inclusive else 0, _CHUNK, out, long_flags,
                    long_count, nseg)
        starts, stops, _, nlong = _long(offsets, long_flags, long_count, nseg)
        if nlong:
            _scan(values, starts, 0, stops, 0, nlong, n, _OPS[op], inclusive, out)
    return out


@on_inputs_device
def segment_broadcast(segment_values, offsets, n, out=None):
    """Each segment's value spread over its elements: ``out[i] = segment_values[k]``
    for every ``i`` in segment ``k``, ``out`` of ``n`` elements (elements outside
    every segment left as they are). The inverse of ``segment_reduce``'s
    placement: per-segment results back to the elements. Returns ``out``."""
    nseg = _segments(offsets)
    _check(offsets, nseg, n)
    if size_of(segment_values) < nseg:
        raise ValueError(f"{size_of(segment_values)} values for {nseg} segments")
    if out is None:
        out = tack.empty((n,), element_type(segment_values))
    _spread(segment_values, offsets, nseg, n, out, 0)
    return out


@on_inputs_device
def segment_ids(offsets, n, dtype=None):
    """Each element's segment: ``ids[i] = k`` for ``i`` in segment ``k``, -1 outside
    every segment, in ``dtype`` (by default ``offsets``' type)."""
    nseg = _segments(offsets)
    _check(offsets, nseg, n)
    ids = tack.full((n,), -1, dtype or element_type(offsets)) if n else \
        tack.empty((0,), dtype or element_type(offsets))
    _spread(ids, offsets, nseg, n, ids, 1)
    return ids


def _spread(source, offsets, nseg, n, out, ids):
    """Each segment's value (``ids``: its number) over its elements: short segments a
    thread each, long ones by chunks."""
    if not (nseg and n):
        return
    long_flags = tack.empty((nseg,), tack.i32)
    long_count = tack.zeros((1,), tack.i32)
    _broadcast_short(source, offsets, _CHUNK, out, ids, long_flags, long_count, nseg)
    starts, stops, names, nlong = _long(offsets, long_flags, long_count, nseg)
    if nlong:
        _broadcast_chunks(source, starts, stops, names, nlong, n, _CHUNK, -(-n // _CHUNK), out,
                          ids)
