"""Parallel prefix sum (scan) on tack fields.

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
    algorithms.inclusive_scan(input_field, output_field, n)
"""

import tack

_CHUNK = 256  # elements one thread scans in turn


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


@tack.kernel
def _zero_first(field):
    for i in range(1):
        field[0] = 0


@tack.kernel
def _read_last(src, dst, idx):
    """Copy a single element src[idx] into dst[0]."""
    for i in range(1):
        dst[0] = src[idx]


def _count(n, *fields):
    """Check the element count: the scan reads and writes [0, n)."""
    n = int(n)
    for f in fields:
        if not 0 <= n <= f.size:
            raise ValueError(
                f"n={n} is outside [0, {f.size}] for a field of {f.size} elements")
    return n


def _total(field, n):
    """field[n - 1] as a Python number, read in the field's own dtype."""
    result = tack.field(dtype=field.dtype, shape=(1,))
    _read_last(field, result, n - 1)
    return result.to_numpy()[0].item()


def _scan(src, dst, n, inclusive):
    """Scan ``src`` into ``dst`` (which may be ``src``) in ``dst``'s dtype and return
    the total: chunk sums, then -- recursively -- their exclusive scan as the
    chunks' offsets, then each chunk's own running sums. A few launches per
    level, and a level per factor of ``_CHUNK``."""
    nchunks = (n + _CHUNK - 1) // _CHUNK
    sums = tack.field(dtype=dst.dtype, shape=(nchunks,))
    _chunk_sums(src, dst, sums, n, _CHUNK, nchunks)
    offsets = tack.field(dtype=dst.dtype, shape=(nchunks,))
    if nchunks == 1:
        _zero_first(offsets)
        total = _total(sums, 1)
    else:
        total = _scan(sums, offsets, nchunks, inclusive=False)
    _chunk_scan(dst, offsets, n, _CHUNK, nchunks, 1 if inclusive else 0)
    return total


def exclusive_scan(input_field, output_field, n):
    """Compute exclusive prefix sum on the active backend.

    output[i] = sum(input[0..i-1]), output[0] = 0.

    The scan runs in the output field's dtype; input values convert to it
    on the copy, so any dtype the backend allocates works.

    Args:
        input_field: field with input values.
        output_field: field for the output offsets.
        n: number of elements, at most either field's size. Zero writes
            nothing and returns 0.

    Returns:
        The total of all input elements in the output's dtype, as a Python
        int or float.
    """
    n = _count(n, input_field, output_field)
    if n == 0:
        # Nothing to scan; the empty sum is 0, as for Field.sum().
        return 0
    # Scan in the output's dtype, as the inclusive scan does: an i32 work
    # buffer truncated float inputs and wrapped wider integers.
    return _scan(input_field, output_field, n, inclusive=False)


def inclusive_scan(input_field, output_field, n):
    """Compute inclusive prefix sum on the active backend.

    output[i] = sum(input[0..i]).

    The scan runs in the output field's dtype; input values convert to it
    on the copy, so any dtype the backend allocates works.

    Args:
        input_field: field with input values.
        output_field: field for the output sums.
        n: number of elements, at most either field's size. Zero writes
            nothing and returns 0.

    Returns:
        The total of all input elements in the output's dtype, as a Python
        int or float.
    """
    n = _count(n, input_field, output_field)
    if n == 0:
        return 0
    return _scan(input_field, output_field, n, inclusive=True)
