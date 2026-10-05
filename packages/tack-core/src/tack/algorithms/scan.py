"""Parallel prefix sum (scan) on tack fields.

Implements a Blelloch-style work-efficient scan using a series of
kernel launches with doubling/halving strides.  No shared memory or
barriers required — works on all backends.

Usage:
    import tack
    from tack import algorithms

    total = algorithms.exclusive_scan(counts, offsets, n)
    algorithms.inclusive_scan(input_field, output_field, n)
"""

import tack


@tack.kernel
def _copy_field(src, dst, n):
    for i in range(n):
        dst[i] = src[i]


@tack.kernel
def _upsweep(data, stride, n, count):
    """Up-sweep (reduce) phase: accumulate at stride boundaries."""
    for i in range(count):
        k = (i + 1) * stride * 2 - 1
        if k < n:
            data[k] = data[k] + data[k - stride]


@tack.kernel
def _downsweep(data, stride, n, count):
    """Down-sweep phase: propagate partial sums back down."""
    for i in range(count):
        k = (i + 1) * stride * 2 - 1 + stride
        if k < n:
            data[k] = data[k] + data[k - stride]


@tack.kernel
def _shift_right(src, dst, n):
    """Convert inclusive scan to exclusive by shifting right, inserting 0."""
    for i in range(n):
        if i == 0:
            dst[i] = 0
        else:
            dst[i] = src[i - 1]


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


def _blelloch_scan_inplace(work, n):
    """Run Blelloch up-sweep + down-sweep on a work buffer (in-place).

    After this, work contains an inclusive prefix sum.
    """
    # Up-sweep (reduce) phase
    stride = 1
    while stride < n:
        count = n // (stride * 2)
        if count > 0:
            _upsweep(work, stride, n, count)
        stride *= 2

    # Down-sweep phase
    stride //= 4
    while stride >= 1:
        count = n // (stride * 2)
        if count > 0:
            _downsweep(work, stride, n, count)
        stride //= 2


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
    work = tack.field(dtype=output_field.dtype, shape=(n,))
    _copy_field(input_field, work, n)
    _blelloch_scan_inplace(work, n)
    _shift_right(work, output_field, n)

    # Total = last element of the inclusive scan
    return _total(work, n)


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
    _copy_field(input_field, output_field, n)
    _blelloch_scan_inplace(output_field, n)
    return _total(output_field, n)
