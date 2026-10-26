"""Parallel prefix sum (scan) on tack arrays.

Reduce, then scan, by chunks: one thread per chunk of ``_CHUNK``
elements totals it, the chunk totals are scanned the same way, and each
chunk then writes its running sums from its offset. That is a few launches
per factor of ``_CHUNK`` -- three levels for a million elements -- where
the Blelloch scan this replaced made two per factor of two, 44 for a
million, and most of a filter's time went to them. No shared memory or
barriers -- works on all backends.

Usage:
    import tack
    from tack import algorithms

    total = algorithms.exclusive_scan(counts, offsets, n)
    algorithms.inclusive_scan(input_array, output_array, n)
"""

import tack
from tack.runtime.dispatch import on_inputs_device

_CHUNK = 256  # elements one thread scans in turn
# A level of at most this many elements is scanned by one thread in one
# launch: below it the recursion's launches cost more than the work (the
# chunk totals of a million elements, 3907 of them, took two more levels
# and five launches of a few microseconds' work each).
_SERIAL_MAX = 4096


@tack.kernel
def _chunk_sums(src, dst, sums, n, chunk, nchunks):
    """Copy each chunk into ``dst`` -- converting to its dtype on the store --
    and total it in that dtype."""
    for c in range(nchunks):
        start = c * chunk
        end = min(start + chunk, n)
        dst[start] = src[start]
        total = dst[start]
        for i in range(start + 1, end):
            dst[i] = src[i]
            total += dst[i]
        sums[c] = total


@tack.kernel
def _chunk_scan(data, offsets, n, chunk, nchunks, inclusive):
    """Each chunk's running sums, in place, starting from its offset."""
    for c in range(nchunks):
        running = offsets[c]
        for i in range(c * chunk, min(c * chunk + chunk, n)):
            value = data[i]
            if inclusive == 1:
                running += value
                data[i] = running
            else:
                data[i] = running
                running += value


# On backends with workgroups the chunk kernels above run few threads with
# long serial loops; there a tile of _TILE elements, _PER_LANE to each of a
# workgroup's 256 lanes, is scanned by the workgroup: each lane totals its
# elements in order, the lane totals are scanned in shared memory, and each
# lane then writes running sums from its offset. Same levels and dtype rules
# as the chunks; between lanes the additions run in tree order.

_TILE = tack.constant(2048)
_PER_LANE = tack.constant(8)


@tack.kernel
def _tile_sums(src, dst, sums, n, ntiles):
    """Copy each tile into ``dst``, converting on the store, and total it there."""
    lanes = tack.shared_like(dst, 256)
    for i in range(ntiles * 256):
        t = tack.thread_id()
        start = (i // 256) * _TILE + t * _PER_LANE
        lanes[t] = 0
        if start < n:
            dst[start] = src[start]
            total = dst[start]
            for g in range(start + 1, min(start + _PER_LANE, n)):
                dst[g] = src[g]
                total += dst[g]
            lanes[t] = total
        tack.barrier()
        step = 1
        while step < 256:
            below = lanes[t - step] if t >= step else lanes[t] * 0
            tack.barrier()
            if t >= step:
                lanes[t] = lanes[t] + below
            tack.barrier()
            step = step * 2
        if t == 0:
            sums[i // 256] = lanes[255]


@tack.kernel
def _tile_apply(data, offsets, n, ntiles, inclusive):
    """Each tile's running sums, in place, from its offset."""
    lanes = tack.shared_like(data, 256)
    for i in range(ntiles * 256):
        t = tack.thread_id()
        start = (i // 256) * _TILE + t * _PER_LANE
        lanes[t] = 0
        if start < n:
            total = data[start]
            for g in range(start + 1, min(start + _PER_LANE, n)):
                total += data[g]
            lanes[t] = total
        tack.barrier()
        step = 1
        while step < 256:
            below = lanes[t - step] if t >= step else lanes[t] * 0
            tack.barrier()
            if t >= step:
                lanes[t] = lanes[t] + below
            tack.barrier()
            step = step * 2
        # The lanes before this one, read from its neighbour rather than
        # subtracted, so floating sums are not rounded twice.
        running = offsets[i // 256]
        if t > 0:
            running = running + lanes[t - 1]
        for g in range(start, min(start + _PER_LANE, n)):
            value = data[g]
            if inclusive == 1:
                running += value
                data[g] = running
            else:
                data[g] = running
                running += value


@tack.kernel
def _scan_serial(src, dst, total, n, inclusive, close):
    """One thread scans ``[0, n)`` into ``dst`` and leaves the sum in
    ``total[0]`` -- and with ``close`` in ``dst[n]`` too, the end of the
    offsets -- every value converted to ``dst``'s dtype on the copy so the
    sums form in it, as the chunk kernels' do."""
    for i in range(1):
        dst[0] = src[0]
        running = dst[0] * 0
        for k in range(n):
            dst[k] = src[k]
            value = dst[k]
            if inclusive == 1:
                running += value
                dst[k] = running
            else:
                dst[k] = running
                running += value
        total[0] = running
        if close == 1:
            dst[n] = running


def _count(n, *arrays):
    """Check the element count: the scan reads and writes [0, n)."""
    n = int(n)
    for f in arrays:
        if not 0 <= n <= f.size:
            raise ValueError(
                f"n={n} is outside [0, {f.size}] for an array of {f.size} elements")
    return n


def _total(array, n):
    """array[n - 1] as a Python number: one element read on the host, which
    on every backend is a copy of just that element and no launch."""
    return array[n - 1]


@tack.kernel
def _close_from(dst, n, offsets, sums, nchunks):
    # dst[n]: the last chunk's offset plus its sum, the whole's total.
    for i in range(1):
        dst[n] = offsets[nchunks - 1] + sums[nchunks - 1]


@tack.kernel
def _close_empty(dst):
    for i in range(1):
        dst[0] = dst[0] * 0


def _scan(src, dst, n, inclusive, want_total=True, close=False):
    """Scan ``src`` into ``dst`` (which may be ``src``) in ``dst``'s dtype and return
    the total: chunk sums, then -- recursively -- their exclusive scan as the
    chunks' offsets, then each chunk's own running sums. A few launches per
    level, and a level per factor of ``_CHUNK``. Without ``want_total`` the
    total is never read to the host, so nothing waits for the device; the
    result is ``None``."""
    from tack.runtime.dispatch import get_backend

    tiles = get_backend().supports_workgroups
    size = int(_TILE) if tiles else _CHUNK
    if n == 0:
        return 0 if want_total else None
    if n <= (size if tiles else _SERIAL_MAX):
        # One launch; with workgroups a level of one tile is one thread's
        # worth anyway. The total is read as one element, not a launch.
        total = tack.empty((1,), dst.dtype)
        _scan_serial(src, dst, total, n, 1 if inclusive else 0, 1 if close else 0)
        return total[0] if want_total else None
    nchunks = (n + size - 1) // size
    sums = tack.empty((nchunks,), dst.dtype)
    if tiles:
        _tile_sums(src, dst, sums, n, nchunks)
    else:
        _chunk_sums(src, dst, sums, n, _CHUNK, nchunks)
    offsets = tack.empty((nchunks,), dst.dtype)
    total = _scan(sums, offsets, nchunks, inclusive=False, want_total=want_total)
    if tiles:
        _tile_apply(dst, offsets, n, nchunks, 1 if inclusive else 0)
    else:
        _chunk_scan(dst, offsets, n, _CHUNK, nchunks, 1 if inclusive else 0)
    if close:
        _close_from(dst, n, offsets, sums, nchunks)
    return total


@on_inputs_device
def exclusive_scan(input_array, output_array, n, total=True, close=False):
    """Compute exclusive prefix sum on the inputs' device.

    output[i] = sum(input[0..i-1]), output[0] = 0.

    The scan runs in the output array's dtype; input values convert to it
    on the copy, so any dtype the backend allocates works.

    Args:
        input_array: array with input values.
        output_array: array for the output offsets.
        n: number of elements, at most either array's size. Zero writes
            nothing and returns 0.
        total: whether to return the total. Reading it waits for the
            device (on Metal, for every launch queued so far); a caller
            that needs only the offsets passes ``total=False`` and gets
            ``None``, and the scan stays queued.
        close: also write the total at ``output[n]``, so ``output`` is the
            ``n + 1`` offsets of CSR rows (``output[0]`` is 0 when ``n`` is
            0). In the scan's own launch where it is one, as for any array
            of at most a tile or 4096 elements; ``output`` needs ``n + 1``
            elements.

    Returns:
        The total of all input elements in the output's dtype, as a Python
        int or float; ``None`` with ``total=False``.
    """
    n = _count(n, input_array, output_array)
    if close and output_array.size < n + 1:
        raise ValueError(f"exclusive_scan with close writes output[{n}]: the output has "
                         f"{output_array.size} elements")
    if n == 0:
        # Nothing to scan; the empty sum is 0, as for Array.sum().
        if close:
            _close_empty(output_array)
        return 0 if total else None
    # Scan in the output's dtype, as the inclusive scan does: an i32 work
    # buffer truncated float inputs and wrapped wider integers.
    return _scan(input_array, output_array, n, inclusive=False, want_total=total, close=close)


@on_inputs_device
def inclusive_scan(input_array, output_array, n, total=True):
    """Compute inclusive prefix sum on the inputs' device.

    output[i] = sum(input[0..i]).

    The scan runs in the output array's dtype; input values convert to it
    on the copy, so any dtype the backend allocates works.

    Args:
        input_array: array with input values.
        output_array: array for the output sums.
        n: number of elements, at most either array's size. Zero writes
            nothing and returns 0.
        total: whether to return the total; see ``exclusive_scan``.

    Returns:
        The total of all input elements in the output's dtype, as a Python
        int or float; ``None`` with ``total=False``.
    """
    n = _count(n, input_array, output_array)
    if n == 0:
        return 0 if total else None
    return _scan(input_array, output_array, n, inclusive=True, want_total=total)
