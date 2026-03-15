"""Parallel prefix sum (scan) on pgc fields.

Implements a Blelloch-style work-efficient scan using a series of
kernel launches with doubling/halving strides.  No shared memory or
barriers required — works on all backends.

Usage:
    import pgc
    from pgc import algorithms

    total = algorithms.exclusive_scan(counts, offsets, n)
    algorithms.inclusive_scan(input_field, output_field, n)
"""

import pgc


@pgc.kernel
def _copy_field(src, dst, n):
    for i in range(n):
        dst[i] = src[i]


@pgc.kernel
def _upsweep(data, stride, n):
    """Up-sweep (reduce) phase: accumulate at stride boundaries."""
    for i in range(n):
        k = (i + 1) * stride * 2 - 1
        if k < n:
            data[k] = data[k] + data[k - stride]


@pgc.kernel
def _downsweep(data, stride, n):
    """Down-sweep phase: propagate partial sums back down."""
    for i in range(n):
        k = (i + 1) * stride * 2 - 1 + stride
        if k < n:
            data[k] = data[k] + data[k - stride]


@pgc.kernel
def _shift_right(src, dst, n):
    """Convert inclusive scan to exclusive by shifting right, inserting 0."""
    for i in range(n):
        if i == 0:
            dst[i] = 0
        else:
            dst[i] = src[i - 1]


def _blelloch_scan_inplace(work, n):
    """Run Blelloch up-sweep + down-sweep on a work buffer (in-place).

    After this, work contains an inclusive prefix sum.
    """
    # Up-sweep (reduce) phase
    stride = 1
    while stride < n:
        _upsweep(work, stride, n)
        stride *= 2

    # Down-sweep phase
    stride //= 4
    while stride >= 1:
        _downsweep(work, stride, n)
        stride //= 2


def exclusive_scan(input_field, output_field, n):
    """Compute exclusive prefix sum on the GPU.

    output[i] = sum(input[0..i-1]), output[0] = 0.

    Args:
        input_field: pgc.field(i32) with input values.
        output_field: pgc.field(i32) for output offsets.
        n: number of elements.

    Returns:
        int: total sum of all input elements.
    """
    work = pgc.field(dtype=pgc.i32, shape=(n,))
    _copy_field(input_field, work, n)
    _blelloch_scan_inplace(work, n)
    _shift_right(work, output_field, n)

    # Total = last element of inclusive scan
    work_np = work.to_numpy()
    return int(work_np[n - 1])


def inclusive_scan(input_field, output_field, n):
    """Compute inclusive prefix sum on the GPU.

    output[i] = sum(input[0..i]).

    Args:
        input_field: pgc.field(i32) with input values.
        output_field: pgc.field(i32) for output sums.
        n: number of elements.

    Returns:
        int: total sum of all input elements.
    """
    _copy_field(input_field, output_field, n)
    _blelloch_scan_inplace(output_field, n)

    out_np = output_field.to_numpy()
    return int(out_np[n - 1])
