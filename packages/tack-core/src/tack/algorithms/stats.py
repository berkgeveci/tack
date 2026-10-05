"""Statistics and analysis on tack fields.

Each function runs a kernel on the active backend that combines into a
one-element f32 or i32 accumulator with atomic operations, then reads that
accumulator back with to_numpy().  var/std call data.sum(), and histogram
without a range calls data.min()/data.max(), which copy non-f32 fields to
the host on GPU backends.
"""

import tack

# ================================================================
# KERNELS
# ================================================================

@tack.kernel
def _sum_sq_diff(data, mean_val, out, n):
    """Sum of squared differences from mean: Σ(x[i] - mean)²."""
    for i in range(n):
        diff = data[i] - mean_val
        tack.atomic_add(out, 0, diff * diff)


@tack.kernel
def _abs_sum(data, out, n):
    """Sum of absolute values: Σ|x[i]|."""
    for i in range(n):
        val = data[i]
        if val < 0.0:
            val = 0.0 - val
        tack.atomic_add(out, 0, val)


@tack.kernel
def _sq_sum(data, out, n):
    """Sum of squares: Σx[i]²."""
    for i in range(n):
        tack.atomic_add(out, 0, data[i] * data[i])


@tack.kernel
def _abs_max(data, out, n):
    """Max absolute value."""
    for i in range(n):
        val = data[i]
        if val < 0.0:
            val = 0.0 - val
        tack.atomic_max(out, 0, val)


@tack.kernel
def _count_nz(data, out, n):
    """Count non-zero elements."""
    for i in range(n):
        if data[i] != 0.0:
            tack.atomic_add(out, 0, 1)


@tack.kernel
def _dot_product(a, b, out, n):
    """Dot product: Σa[i]*b[i]."""
    for i in range(n):
        tack.atomic_add(out, 0, a[i] * b[i])


@tack.kernel
def _histogram_kernel(data, counts, lo, inv_bin_width, n_bins, n):
    """Bin data into histogram using atomics."""
    for i in range(n):
        val = data[i]
        b = int((val - lo) * inv_bin_width)
        if b < 0:
            b = 0
        if b >= n_bins:
            b = n_bins - 1
        tack.atomic_add(counts, b, 1)


# ================================================================
# PUBLIC API
# ================================================================

def _count(n, *fields):
    """Resolve and check the element count: every kernel reads [0, n)."""
    if n is None:
        n = fields[0].size
    n = int(n)
    for f in fields:
        if not 0 <= n <= f.size:
            raise ValueError(
                f"n={n} is outside [0, {f.size}] for a field of {f.size} elements")
    return n


def _prefix(data, n):
    """The first n elements of data, as a field the host reductions accept."""
    if n == data.size:
        return data
    from tack.algorithms.copy import copy
    head = tack.field(dtype=data.dtype, shape=(n,))
    copy(data, head, n)
    return head


def var(data, n=None):
    """Population variance of a field: Σ(x - mean)² / n.

    Runs two passes over the first n elements: Field.sum() for the mean,
    then a kernel adding the squared differences into an f32 accumulator.
    Returns a NumPy float32; an empty range gives NaN, as Field.mean() does.
    """
    n = _count(n, data)
    if n == 0:
        return float('nan')
    mean_val = _prefix(data, n).sum() / n
    acc = tack.field(dtype=tack.f32, shape=(1,))
    acc.fill(0.0)
    _sum_sq_diff(data, mean_val, acc, n)
    return acc.to_numpy()[0] / n


def std(data, n=None):
    """Population standard deviation of a field."""
    from math import sqrt
    return sqrt(var(data, n))


def norm(data, ord=2, n=None):
    """Vector norm of a field.

    ord=1: L1 norm (sum of absolute values)
    ord=2: L2 norm (Euclidean)
    ord=inf: L-infinity (max absolute value)

    Any other ord raises ValueError.
    """
    n = _count(n, data)
    if ord == 1:
        acc = tack.field(dtype=tack.f32, shape=(1,))
        acc.fill(0.0)
        _abs_sum(data, acc, n)
        return float(acc.to_numpy()[0])
    if ord == 2:
        from math import sqrt
        acc = tack.field(dtype=tack.f32, shape=(1,))
        acc.fill(0.0)
        _sq_sum(data, acc, n)
        return sqrt(float(acc.to_numpy()[0]))
    if ord == float('inf'):
        acc = tack.field(dtype=tack.f32, shape=(1,))
        acc.fill(0.0)
        _abs_max(data, acc, n)
        return float(acc.to_numpy()[0])
    raise ValueError(f"Unsupported norm order: {ord}")


def absmax(data, n=None):
    """Maximum absolute value of a field."""
    n = _count(n, data)
    acc = tack.field(dtype=tack.f32, shape=(1,))
    acc.fill(0.0)
    _abs_max(data, acc, n)
    return float(acc.to_numpy()[0])


def count_nonzero(data, n=None):
    """Count non-zero elements in a field."""
    n = _count(n, data)
    acc = tack.field(dtype=tack.i32, shape=(1,))
    acc.fill(0)
    _count_nz(data, acc, n)
    return int(acc.to_numpy()[0])


def dot(a, b, n=None):
    """Dot product of two fields: Σa[i]*b[i]."""
    n = _count(n, a, b)
    acc = tack.field(dtype=tack.f32, shape=(1,))
    acc.fill(0.0)
    _dot_product(a, b, acc, n)
    return float(acc.to_numpy()[0])


def histogram(data, bins=10, range=None, n=None):
    """Compute a histogram of field values with atomic bin counts.

    Values below the range are counted in the first bin and values above it
    in the last, unlike numpy.histogram, which drops them.

    Args:
        data: input field of any dtype
        bins: number of bins (at least 1)
        range: (min, max) tuple. If None, uses the minimum and maximum of
            the first n elements, and n must then be at least 1.
        n: number of elements (default: data.size)

    Returns:
        (counts, bin_edges) where counts is a tack.field of i32,
        bin_edges is a numpy array of (bins + 1) float64 edges.
    """
    import numpy as np
    n = _count(n, data)
    if range is None:
        if n == 0:
            raise ValueError("histogram of no elements needs an explicit range")
        head = _prefix(data, n)
        lo = float(head.min())
        hi = float(head.max())
    else:
        lo, hi = float(range[0]), float(range[1])

    # Avoid division by zero for constant fields
    if hi == lo:
        hi = lo + 1.0

    bin_width = (hi - lo) / bins
    inv_bw = 1.0 / bin_width

    counts = tack.field(dtype=tack.i32, shape=(bins,))
    counts.fill(0)
    _histogram_kernel(data, counts, lo, inv_bw, bins, n)

    edges = np.linspace(lo, hi, bins + 1)
    return counts, edges
