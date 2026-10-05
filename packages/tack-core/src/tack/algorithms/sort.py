"""Sorting and segmented reductions on tack fields.

A stable least-significant-digit radix sort, with ``unique`` and
``reduce_by_key`` on top of it.  Like the scans, everything here is a
sequence of ordinary kernels on the active backend: no shared memory,
barriers or workgroup collectives, so it runs on the CPU as well as on the
GPU backends.

Usage:
    from tack import algorithms

    perm = algorithms.argsort(keys)                       # stable permutation
    skeys, svals = algorithms.sort_by_key(keys, values)   # new, sorted fields
    ukeys, counts = algorithms.unique(skeys)              # run-length summary
    ukeys, sums = algorithms.reduce_by_key(skeys, svals)  # one value per key

How the sort works
------------------
Keys are first mapped to unsigned integers whose numeric order is the
keys' order: unsigned keys are copied, signed keys have their sign bit
flipped.  The smallest mapped key is subtracted on the fly, so the number
of 8-bit digit passes depends on the key *range*, not the key width: cell
ids below 2^16 take two passes, not four.

Each pass splits the range into chunks of ``_CHUNK`` elements.  One
thread per chunk counts its digits into a private 256-entry histogram and
writes the counts bucket-major (``counts[digit * nchunks + chunk]``), so
one exclusive scan over that array gives every (digit, chunk) pair its
first output slot in stable order.  A second kernel, again one thread per
chunk, walks its chunk in order and scatters each element to the next
slot for its digit.  Within a chunk the walk is sequential and across
chunks the slots are disjoint, so the pass is stable; the sort carries a
permutation rather than the caller's values, and ``sort_by_key`` gathers
the values once at the end.

Segments
--------
``unique`` and ``reduce_by_key`` take keys that are already sorted (or at
least grouped: equal keys adjacent).  They flag the first element of each
run, scan the flags to number the runs, and scatter each run's start into
an offsets array with the usual sentinel at the end.  Reductions then run
one thread per run, serially over its elements, which keeps them exact and
reproducible for every dtype at the price of load imbalance when a few
runs are very long.
"""

import tack
from tack.algorithms.copy import copy
from tack.algorithms.scan import exclusive_scan

# Elements per chunk in the counting and scatter kernels. Larger chunks
# mean fewer threads with longer serial loops and a shorter scan. The
# kernels' private histograms are sized by the radix, not the chunk, so
# this is a runtime argument and free to tune.
_CHUNK = 256
_RADIX_BITS = 8
_RADIX = 1 << _RADIX_BITS  # 256: the local_array sizes in the kernels

_KEY_DTYPES = {
    tack.i32: tack.u32,
    tack.u32: tack.u32,
    tack.i64: tack.u64,
    tack.u64: tack.u64,
}

_INDEX_LIMIT = 2**31 - 1


# --- key mapping and range -------------------------------------------------

@tack.kernel
def _map_keys_32(keys, mapped, flip, n):
    """mapped[i] = keys[i] as an unsigned integer, with the sign bit flipped
    for signed keys so unsigned order is numeric order.  ``flip`` is 0 for
    unsigned keys.  The u32 step keeps a negative i32 from sign-extending
    into the high word."""
    for i in range(n):
        mapped[i] = tack.u64(tack.u32(keys[i])) ^ tack.u64(flip)


@tack.kernel
def _map_keys_64(keys, mapped, flip, n):
    """The 64-bit counterpart of ``_map_keys_32``."""
    for i in range(n):
        mapped[i] = tack.u64(keys[i]) ^ tack.u64(flip)


@tack.kernel
def _mapped_lo_hi_range(mapped, lo_min, lo_max, hi_min, hi_max, n):
    """Min and max of the low and high 32-bit words of the mapped keys,
    with the 32-bit atomics every backend has."""
    for i in range(n):
        k = mapped[i]
        lo = tack.u32(k & tack.u64(4294967295))
        hi = tack.u32(k >> 32)
        tack.atomic_min(lo_min, 0, lo)
        tack.atomic_max(lo_max, 0, lo)
        tack.atomic_min(hi_min, 0, hi)
        tack.atomic_max(hi_max, 0, hi)


# --- one radix pass ---------------------------------------------------------

@tack.kernel
def _count_digits(mapped, counts, base, shift, n, chunk, nchunks):
    """counts[d * nchunks + c] = number of elements in chunk c whose digit
    at ``shift`` is d, after subtracting ``base``."""
    for c in range(nchunks):
        hist = tack.local_array(tack.i32, 256)
        for b in range(256):
            hist[b] = 0
        start = c * chunk
        end = min(start + chunk, n)
        for i in range(start, end):
            d = tack.i32(((mapped[i] - tack.u64(base)) >> shift) & tack.u64(255))
            hist[d] = hist[d] + 1
        for b in range(256):
            counts[b * nchunks + c] = hist[b]


@tack.kernel
def _scatter_digits(mapped_in, perm_in, mapped_out, perm_out, offsets,
                    base, shift, n, chunk, nchunks):
    """Move every element of chunk c to the slots the scanned counts gave
    its digit, in chunk order, so the pass is stable."""
    for c in range(nchunks):
        slot = tack.local_array(tack.i32, 256)
        for b in range(256):
            slot[b] = offsets[b * nchunks + c]
        start = c * chunk
        end = min(start + chunk, n)
        for i in range(start, end):
            d = tack.i32(((mapped_in[i] - tack.u64(base)) >> shift) & tack.u64(255))
            pos = slot[d]
            slot[d] = pos + 1
            mapped_out[pos] = mapped_in[i]
            perm_out[pos] = perm_in[i]


@tack.kernel
def _iota(perm, n):
    for i in range(n):
        perm[i] = i


@tack.kernel
def _gather(src, indices, dst, n):
    for i in range(n):
        dst[i] = src[indices[i]]


# --- segments ---------------------------------------------------------------

@tack.kernel
def _flag_run_starts(keys, flags, n):
    for i in range(n):
        if i == 0:
            flags[i] = 1
        elif keys[i] != keys[i - 1]:
            flags[i] = 1
        else:
            flags[i] = 0


@tack.kernel
def _scatter_run_starts(flags, run_ids, offsets, n, nruns):
    """offsets[r] = first index of run r; offsets[nruns] = n.

    ``run_ids`` is the exclusive scan of ``flags``, which is the run index
    at a run's first element and one past it elsewhere, so only flagged
    elements may use it."""
    for i in range(n):
        if flags[i] == 1:
            offsets[run_ids[i]] = i
        if i == 0:
            offsets[nruns] = n


@tack.kernel
def _run_lengths(offsets, counts, nruns):
    for r in range(nruns):
        counts[r] = offsets[r + 1] - offsets[r]


@tack.kernel
def _reduce_runs_sum(values, offsets, out, nruns):
    for r in range(nruns):
        start = offsets[r]
        end = offsets[r + 1]
        acc = values[start]
        for i in range(start + 1, end):
            acc = acc + values[i]
        out[r] = acc


@tack.kernel
def _reduce_runs_min(values, offsets, out, nruns):
    for r in range(nruns):
        start = offsets[r]
        end = offsets[r + 1]
        acc = values[start]
        for i in range(start + 1, end):
            acc = min(acc, values[i])
        out[r] = acc


@tack.kernel
def _reduce_runs_max(values, offsets, out, nruns):
    for r in range(nruns):
        start = offsets[r]
        end = offsets[r + 1]
        acc = values[start]
        for i in range(start + 1, end):
            acc = max(acc, values[i])
        out[r] = acc


_REDUCERS = {
    "sum": _reduce_runs_sum,
    "min": _reduce_runs_min,
    "max": _reduce_runs_max,
}


# --- host side --------------------------------------------------------------

def _count(n, *fields):
    """Resolve and check the element count: every field is read or written
    on [0, n)."""
    n = fields[0].size if n is None else int(n)
    for f in fields:
        if not 0 <= n <= f.size:
            raise ValueError(
                f"n={n} is outside [0, {f.size}] for a field of {f.size} elements")
    if n > _INDEX_LIMIT:
        raise ValueError(
            f"n={n} exceeds the i32 index limit {_INDEX_LIMIT} of the sort")
    return n


def _key_width(keys):
    mapped = _KEY_DTYPES.get(keys.dtype)
    if mapped is None:
        names = ", ".join(t.name for t in _KEY_DTYPES)
        raise TypeError(
            f"sort keys must be one of {names}; got {keys.dtype.name}")
    return mapped


def _read(field):
    return field.to_numpy()[0].item()


def argsort(keys, n=None):
    """Stable permutation that sorts the first ``n`` keys ascending.

    ``keys`` is an ``i32``, ``u32``, ``i64`` or ``u64`` field.  Returns a
    new ``i32`` field ``perm`` of ``n`` elements with ``keys[perm[0]] <=
    keys[perm[1]] <= ...``; equal keys keep their original order.  ``n``
    defaults to the whole field and may be zero.
    """
    _key_width(keys)
    n = _count(n, keys)
    perm = tack.field(dtype=tack.i32, shape=(n,))
    if n == 0:
        return perm
    _iota(perm, n)
    if n == 1:
        return perm

    flip = 0 if keys.dtype in (tack.u32, tack.u64) else 1 << (keys.dtype.bits - 1)
    mapped = tack.field(dtype=tack.u64, shape=(n,))
    map_keys = _map_keys_32 if keys.dtype.bits == 32 else _map_keys_64
    map_keys(keys, mapped, flip, n)

    # The passes needed depend on the spread of the keys, not their width.
    # The base is the smallest mapped key, so digits above the spread are
    # all zero and those passes can be skipped.
    lo_min = tack.field(dtype=tack.u32, shape=(1,))
    lo_max = tack.field(dtype=tack.u32, shape=(1,))
    hi_min = tack.field(dtype=tack.u32, shape=(1,))
    hi_max = tack.field(dtype=tack.u32, shape=(1,))
    lo_min.fill(0xFFFFFFFF)
    lo_max.fill(0)
    hi_min.fill(0xFFFFFFFF)
    hi_max.fill(0)
    _mapped_lo_hi_range(mapped, lo_min, lo_max, hi_min, hi_max, n)
    lo_lo, lo_hi = _read(lo_min), _read(lo_max)
    hi_lo, hi_hi = _read(hi_min), _read(hi_max)
    if hi_lo == hi_hi:
        # One high word: the low word alone orders the keys.
        base = (hi_lo << 32) | lo_lo
        spread = lo_hi - lo_lo
    else:
        # The low words of different high words do not compare, so only the
        # high word's spread can be subtracted; the low word takes 4 passes.
        base = hi_lo << 32
        spread = ((hi_hi - hi_lo) << 32) | 0xFFFFFFFF
    passes = (spread.bit_length() + _RADIX_BITS - 1) // _RADIX_BITS
    if passes == 0:
        return perm  # every key is equal: the identity is the stable order

    nchunks = (n + _CHUNK - 1) // _CHUNK
    counts = tack.field(dtype=tack.i32, shape=(_RADIX * nchunks,))
    offsets = tack.field(dtype=tack.i32, shape=(_RADIX * nchunks,))
    mapped_b = tack.field(dtype=tack.u64, shape=(n,))
    perm_b = tack.field(dtype=tack.i32, shape=(n,))
    src_keys, src_perm, dst_keys, dst_perm = mapped, perm, mapped_b, perm_b
    for p in range(passes):
        shift = p * _RADIX_BITS
        _count_digits(src_keys, counts, base, shift, n, _CHUNK, nchunks)
        exclusive_scan(counts, offsets, _RADIX * nchunks)
        _scatter_digits(src_keys, src_perm, dst_keys, dst_perm, offsets,
                        base, shift, n, _CHUNK, nchunks)
        src_keys, dst_keys = dst_keys, src_keys
        src_perm, dst_perm = dst_perm, src_perm
    if src_perm is not perm:
        copy(src_perm, perm, n)
    return perm


def gather(src, indices, n=None):
    """New field ``out`` with ``out[i] = src[indices[i]]`` for ``i < n``.

    ``indices`` is an integer field; ``n`` defaults to its size.  The
    output has ``src``'s dtype.  Indices must lie in ``[0, src.size)``.
    """
    n = _count(n, indices)
    out = tack.field(dtype=src.dtype, shape=(n,))
    if n:
        _gather(src, indices, out, n)
    return out


def sort_by_key(keys, values=None, n=None):
    """Sort the first ``n`` keys ascending, carrying ``values`` along.

    Returns ``(sorted_keys, sorted_values)`` as new fields of ``n``
    elements; ``sorted_values`` is ``None`` when no values are given.  The
    sort is stable, so equal keys keep the order of their values.  See
    ``argsort`` for the key dtypes.
    """
    fields = (keys,) if values is None else (keys, values)
    n = _count(n, *fields)
    perm = argsort(keys, n)
    sorted_keys = gather(keys, perm, n)
    sorted_values = None if values is None else gather(values, perm, n)
    return sorted_keys, sorted_values


def _run_offsets(sorted_keys, n):
    """Offsets of the runs of equal adjacent keys: an ``i32`` field of
    ``nruns + 1`` entries with ``offsets[nruns] == n``.  Returns
    ``(offsets, nruns)``; an empty input has no runs."""
    if n == 0:
        return tack.field(dtype=tack.i32, shape=(1,)), 0
    flags = tack.field(dtype=tack.i32, shape=(n,))
    run_ids = tack.field(dtype=tack.i32, shape=(n,))
    _flag_run_starts(sorted_keys, flags, n)
    nruns = exclusive_scan(flags, run_ids, n)
    offsets = tack.field(dtype=tack.i32, shape=(nruns + 1,))
    _scatter_run_starts(flags, run_ids, offsets, n, nruns)
    return offsets, nruns


def unique(sorted_keys, n=None):
    """The distinct keys of a sorted field and how often each occurs.

    ``sorted_keys`` must have equal keys adjacent (sorted, or grouped).
    Returns ``(keys, counts)``: ``keys`` is a new field of the same dtype
    holding each distinct key once, in order of first appearance, and
    ``counts`` is an ``i32`` field of the run lengths.  Works for any
    dtype that supports ``!=``.
    """
    n = _count(n, sorted_keys)
    offsets, nruns = _run_offsets(sorted_keys, n)
    keys = tack.field(dtype=sorted_keys.dtype, shape=(nruns,))
    counts = tack.field(dtype=tack.i32, shape=(nruns,))
    if nruns:
        _gather(sorted_keys, offsets, keys, nruns)
        _run_lengths(offsets, counts, nruns)
    return keys, counts


def reduce_by_key(sorted_keys, values, n=None, op="sum"):
    """Reduce the values of each run of equal keys to one value.

    ``sorted_keys`` must have equal keys adjacent, as for ``unique``;
    ``values`` is any field with at least ``n`` elements.  ``op`` is
    ``"sum"``, ``"min"`` or ``"max"``.  Returns ``(keys, reduced)``: the
    distinct keys and a new field of ``values``' dtype with one entry per
    key.  Each run is reduced serially in element order, so the result is
    exact for integers and reproducible for floats; integer sums wrap at
    the values' width.
    """
    reducer = _REDUCERS.get(op)
    if reducer is None:
        raise ValueError(f"op must be one of {', '.join(_REDUCERS)}; got {op!r}")
    n = _count(n, sorted_keys, values)
    offsets, nruns = _run_offsets(sorted_keys, n)
    keys = tack.field(dtype=sorted_keys.dtype, shape=(nruns,))
    reduced = tack.field(dtype=values.dtype, shape=(nruns,))
    if nruns:
        _gather(sorted_keys, offsets, keys, nruns)
        reducer(values, offsets, reduced, nruns)
    return keys, reduced


__all__ = ["argsort", "gather", "reduce_by_key", "sort_by_key", "unique"]
