"""GPU-accelerated statistics and analysis on tack fields.

All operations run entirely on the active backend — no host roundtrips
unless noted. Uses atomic operations for reductions and histogram binning.
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

def var(data, n=None):
    """Population variance of a field: Σ(x - mean)² / n.

    Runs two GPU passes: one for the mean, one for the squared differences.
    """
    if n is None:
        n = data.size
    mean_val = data.sum() / n
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
    """
    if n is None:
        n = data.size
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
    if n is None:
        n = data.size
    acc = tack.field(dtype=tack.f32, shape=(1,))
    acc.fill(0.0)
    _abs_max(data, acc, n)
    return float(acc.to_numpy()[0])


def count_nonzero(data, n=None):
    """Count non-zero elements in a field."""
    if n is None:
        n = data.size
    acc = tack.field(dtype=tack.i32, shape=(1,))
    acc.fill(0)
    _count_nz(data, acc, n)
    return int(acc.to_numpy()[0])


def dot(a, b, n=None):
    """Dot product of two fields: Σa[i]*b[i]."""
    if n is None:
        n = a.size
    acc = tack.field(dtype=tack.f32, shape=(1,))
    acc.fill(0.0)
    _dot_product(a, b, acc, n)
    return float(acc.to_numpy()[0])


def histogram(data, bins=10, range=None, n=None):
    """Compute a histogram of field values on GPU using atomics.

    Args:
        data: input field (f32)
        bins: number of bins
        range: (min, max) tuple. If None, uses data.min()/data.max().
        n: number of elements (default: data.size)

    Returns:
        (counts, bin_edges) where counts is a tack.field of i32,
        bin_edges is a numpy array of (bins + 1) float64 edges.
    """
    import numpy as np
    if n is None:
        n = data.size
    if range is None:
        lo = float(data.min())
        hi = float(data.max())
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
