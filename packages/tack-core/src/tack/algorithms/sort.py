"""Sorting and segmented reductions on tack fields.

A stable least-significant-digit radix sort, with ``unique`` and
``reduce_by_key`` on top of it. On the CPU, everything here is a sequence of
ordinary kernels. On backends with workgroups, each radix pass sorts tiles
of keys in shared memory instead (see "one radix pass, shaped for GPUs"
below); the result is the same permutation.

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

On the CPU, each pass splits the range into chunks of ``_CHUNK`` elements.  One
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
def _chunk_range(mins_in, maxs_in, mins, maxs, n, chunk, nchunks):
    """Each chunk's smallest and largest key, one thread per chunk.

    No atomics: on a CPU, atomics on one address from every thread take
    turns on its cache line, and the four per key this replaced made the
    sort four times slower on eight threads than on one.
    """
    for c in range(nchunks):
        start = c * chunk
        end = min(start + chunk, n)
        lo = mins_in[start]
        hi = maxs_in[start]
        for i in range(start + 1, end):
            lo = min(lo, mins_in[i])
            hi = max(hi, maxs_in[i])
        mins[c] = lo
        maxs[c] = hi


def _key_range(mapped, n):
    """The smallest and largest of the first ``n`` (at least one) mapped keys:
    chunk ranges, then ranges of those, until one remains."""
    mins = maxs = mapped
    while True:
        nchunks = (n + _CHUNK - 1) // _CHUNK
        out_mins = tack.field(dtype=tack.u64, shape=(nchunks,))
        out_maxs = tack.field(dtype=tack.u64, shape=(nchunks,))
        _chunk_range(mins, maxs, out_mins, out_maxs, n, _CHUNK, nchunks)
        if nchunks == 1:
            return _read(out_mins), _read(out_maxs)
        mins, maxs, n = out_mins, out_maxs, nchunks


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


# --- one radix pass, shaped for GPUs ---------------------------------------
#
# The chunk kernels above give one thread 256 keys and a private 256-entry
# table: right for CPU threads, slow on a GPU, which then runs few threads
# whose tables spill to slow memory. On backends with workgroups a pass
# takes tiles of _TILE keys, one workgroup each, and sorts every tile by its
# digit in shared memory first: eight stable one-bit splits, each placing a
# key by a workgroup prefix sum of the zeros before it. Tack has no atomics
# on shared memory, and needs none here. In the sorted tile each digit is
# one run, whose first and last positions give the tile's count for it --
# laid out digit-major like the chunk counts, then scanned into offsets --
# and the tile is written back in that order. A plain kernel then moves
# each key to its digit's offset plus its place in the run, so keys leave
# in runs rather than one by one. Every split is stable and the tiles keep
# their order, so the pass is stable.

# Keys per tile and per lane of the workgroup's 256. Digits are kept as u8 and
# tile positions as u16 so two copies of each fit Metal's 32 KB of
# threadgroup memory; longer tiles mean longer runs per digit to scatter and
# a shorter table of counts to scan.
_TILE = tack.constant(4096)
_PER_LANE = tack.constant(16)


@tack.kernel
def _sort_tiles(mapped, perm, tile_keys, tile_perm, counts, starts, base, shift, n, ntiles):
    digit = tack.shared(tack.u8, _TILE)
    local = tack.shared(tack.u16, _TILE)
    digit_b = tack.shared(tack.u8, _TILE)
    local_b = tack.shared(tack.u16, _TILE)
    zeros = tack.shared(tack.i32, 256)
    first = tack.shared(tack.i32, 256)
    last = tack.shared(tack.i32, 256)
    for i in range(ntiles * 256):
        t = tack.thread_id()
        tile = i // 256
        valid = min(_TILE, n - tile * _TILE)     # keys in this tile; the rest pad it
        for k in range(_PER_LANE):
            q = t * _PER_LANE + k
            # Padding sorts last: digit 255, after the real 255s by stability.
            d = 255
            if q < valid:
                d = tack.i32(((mapped[tile * _TILE + q] - tack.u64(base)) >> shift)
                             & tack.u64(255))
            digit[q] = d
            local[q] = q
        first[t] = -1
        last[t] = -1
        tack.barrier()
        for bit in range(8):
            mine = 0
            for k in range(_PER_LANE):
                if (tack.i32(digit[t * _PER_LANE + k]) >> bit) & 1 == 0:
                    mine += 1
            zeros[t] = mine
            tack.barrier()
            step = 1
            while step < 256:
                below = zeros[t - step] if t >= step else 0
                tack.barrier()
                zeros[t] = zeros[t] + below
                tack.barrier()
                step = step * 2
            total = zeros[255]
            z = zeros[t] - mine                  # zeros before this lane's keys
            o = t * _PER_LANE - z                # ones before them
            for k in range(_PER_LANE):
                q = t * _PER_LANE + k
                d = tack.i32(digit[q])
                to = 0
                if (d >> bit) & 1 == 0:
                    to = z
                    z += 1
                else:
                    to = total + o
                    o += 1
                digit_b[to] = d
                local_b[to] = local[q]
            tack.barrier()
            for k in range(_PER_LANE):
                q = t * _PER_LANE + k
                digit[q] = digit_b[q]
                local[q] = local_b[q]
            tack.barrier()
        for k in range(_PER_LANE):
            q = t * _PER_LANE + k
            if q < valid:
                d = tack.i32(digit[q])
                if q == 0 or tack.i32(digit[q - 1]) != d:
                    first[d] = q
                if q == valid - 1 or tack.i32(digit[q + 1]) != d:
                    last[d] = q
                src = tile * _TILE + tack.i32(local[q])
                tile_keys[tile * _TILE + q] = mapped[src]
                tile_perm[tile * _TILE + q] = perm[src]
        tack.barrier()
        counts[t * ntiles + tile] = last[t] - first[t] + 1 if first[t] >= 0 else 0
        starts[tile * 256 + t] = first[t]


@tack.kernel
def _scatter_tiles(tile_keys, tile_perm, mapped_out, perm_out, offsets, starts,
                   base, shift, n, ntiles):
    for i in range(n):
        tile = i // _TILE
        d = tack.i32(((tile_keys[i] - tack.u64(base)) >> shift) & tack.u64(255))
        pos = offsets[d * ntiles + tile] + (i - tile * _TILE - starts[tile * 256 + d])
        mapped_out[pos] = tile_keys[i]
        perm_out[pos] = tile_perm[i]


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
    base, largest = _key_range(mapped, n)
    spread = largest - base
    passes = (spread.bit_length() + _RADIX_BITS - 1) // _RADIX_BITS
    if passes == 0:
        return perm  # every key is equal: the identity is the stable order

    from tack.runtime.dispatch import get_backend

    mapped_b = tack.field(dtype=tack.u64, shape=(n,))
    perm_b = tack.field(dtype=tack.i32, shape=(n,))
    if get_backend().supports_workgroups:
        # Tiles sorted into the second buffers, then scattered back: the
        # pass's result is where its input was.
        ntiles = (n + _TILE - 1) // _TILE
        counts = tack.field(dtype=tack.i32, shape=(_RADIX * ntiles,))
        offsets = tack.field(dtype=tack.i32, shape=(_RADIX * ntiles,))
        starts = tack.field(dtype=tack.i32, shape=(_RADIX * ntiles,))
        for p in range(passes):
            shift = p * _RADIX_BITS
            _sort_tiles(mapped, perm, mapped_b, perm_b, counts, starts, base, shift, n,
                        ntiles)
            exclusive_scan(counts, offsets, _RADIX * ntiles)
            _scatter_tiles(mapped_b, perm_b, mapped, perm, offsets, starts, base, shift, n,
                           ntiles)
        return perm
    nchunks = (n + _CHUNK - 1) // _CHUNK
    counts = tack.field(dtype=tack.i32, shape=(_RADIX * nchunks,))
    offsets = tack.field(dtype=tack.i32, shape=(_RADIX * nchunks,))
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
