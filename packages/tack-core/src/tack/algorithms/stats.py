"""Statistics and analysis on tack arrays.

Each statistic forms its terms -- |x|, x², (x - mean)², a·b, x ≠ 0 -- in a
temporary array, in the precision of its inputs (f64 if any is f64,
otherwise f32), and reduces that array with ``Array.sum()`` or
``Array.max()``. They therefore follow the array-reduction contract: f64
inputs accumulate in f64, and the error budget and repeatability are those
of array reductions. ``count_nonzero`` scans its 0/1 flags into i64. (They used to combine into a
one-element f32 or i32 accumulator with atomics: f64 data was summed in
f32, counts wrapped past 2^31, and the order of the additions was
whatever order the launches arrived in.) ``histogram`` still bins with
atomics: each bin's count is an exact integer whatever the order.
"""

import tack

# ================================================================
# KERNELS
# ================================================================
# ``one`` is 1.0 in the accumulation precision: a float scalar takes f64
# when an array argument is f64, so ``x * one`` converts an f32 or integer
# input once, before any arithmetic, to the precision of ``out``.

@tack.kernel
def _converted(data, one, out, n):
    """x[i] in the accumulation precision."""
    for i in range(n):
        out[i] = data[i] * one


@tack.kernel
def _squared_diffs(data, mean_val, one, out, n):
    """(x[i] - mean)²."""
    for i in range(n):
        diff = data[i] * one - mean_val
        out[i] = diff * diff


@tack.kernel
def _abs_values(data, one, out, n):
    """|x[i]|."""
    for i in range(n):
        out[i] = abs(data[i] * one)


@tack.kernel
def _squares(data, one, out, n):
    """x[i]²."""
    for i in range(n):
        x = data[i] * one
        out[i] = x * x


@tack.kernel
def _nonzero(data, out, n):
    """1 where x[i] != 0 (a NaN is nonzero; -0.0 is zero), else 0."""
    for i in range(n):
        out[i] = 1 if data[i] != 0.0 else 0


@tack.kernel
def _products(a, b, one, out, n):
    """a[i]·b[i]."""
    for i in range(n):
        out[i] = (a[i] * one) * (b[i] * one)


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

def _count(n, *arrays):
    """Resolve and check the element count, and flatten vector arrays.

    Every kernel reads [0, n) scalar elements. A vector array is reduced
    over all its scalar components in storage order through a storage view, since the kernels index scalars and a
    vector element in `a[i] * b[i]` has no scalar meaning.
    Returns ``(n, *arrays)`` with the views substituted.
    """
    flat = tuple(f._scalar_view() if hasattr(f, '_vector_n') else f for f in arrays)
    if n is None:
        n = flat[0].size
    n = int(n)
    for f in flat:
        if not 0 <= n <= f.size:
            raise ValueError(
                f"n={n} is outside [0, {f.size}] for an array of {f.size} elements")
    return (n, *flat)


def _prefix(data, n):
    """The first n elements of data, as an array the host reductions accept."""
    if n == data.size:
        return data
    from tack.algorithms.copy import copy
    head = tack.empty((n,), data.dtype)
    copy(data, head, n)
    return head


def _precision(*arrays):
    """The accumulation precision of a statistic of ``arrays``: f64 if any is f64."""
    return tack.f64 if any(f.dtype is tack.f64 for f in arrays) else tack.f32


def var(data, n=None):
    """Population variance of an array: Σ(x - mean)² / n.

    Two passes over the first n elements: the mean by ``Array.sum()``, then
    the squared differences, formed in the data's precision (f64 for f64
    data, otherwise f32) and summed by ``Array.sum()``. Returns a Python
    float; an empty range gives NaN, as ``Array.mean()`` does.
    """
    n, data = _count(n, data)
    if n == 0:
        return float('nan')
    if data.dtype in (tack.f32, tack.f64):
        mean_val = _prefix(data, n).sum() / n
    else:
        # An integer sum wraps (2^62 four times is 0): the values convert
        # first, as every other term does.
        mean_val = _terms(_converted, n, data).sum() / n
    terms = tack.empty((n,), _precision(data))
    _squared_diffs(data, mean_val, 1.0, terms, n)
    return terms.sum() / n


def std(data, n=None):
    """Population standard deviation of an array."""
    from math import sqrt
    return sqrt(var(data, n))


def _terms(kernel, n, *arrays):
    """``kernel``'s terms over the first n elements, in the arrays' precision."""
    terms = tack.empty((n,), _precision(*arrays))
    kernel(*arrays, 1.0, terms, n)
    return terms


def norm(data, ord=2, n=None):
    """Vector norm of an array, in the data's precision (f64 for f64 data,
    otherwise f32), as a Python float.

    ord=1: L1 norm (sum of absolute values)
    ord=2: L2 norm (Euclidean)
    ord=inf: L-infinity (max absolute value)

    Any other ord raises ValueError. An empty range gives 0.0.
    """
    if ord not in (1, 2, float('inf')):
        raise ValueError(f"Unsupported norm order: {ord}")
    n, data = _count(n, data)
    if n == 0:
        return 0.0
    if ord == 1:
        return _terms(_abs_values, n, data).sum()
    if ord == 2:
        from math import sqrt
        return sqrt(_terms(_squares, n, data).sum())
    return _terms(_abs_values, n, data).max()


def absmax(data, n=None):
    """Maximum absolute value of an array, as a Python float; 0.0 when empty."""
    return norm(data, float('inf'), n)


def count_nonzero(data, n=None):
    """The number of nonzero elements of an array, as a Python int: its 0/1 flags
    scanned into i64 on the device, so the count is exact past 2^31 and only
    the total comes back to the host."""
    from tack.algorithms.scan import exclusive_scan

    n, data = _count(n, data)
    if n == 0:
        return 0
    flags = tack.empty((n,), tack.i32)
    _nonzero(data, flags, n)
    return int(exclusive_scan(flags, tack.empty((n,), tack.i64), n))


def dot(a, b, n=None):
    """Dot product of two arrays, Σa[i]·b[i], in their precision (f64 if either
    is f64, otherwise f32), as a Python float; 0.0 when empty."""
    n, a, b = _count(n, a, b)
    if n == 0:
        return 0.0
    return _terms(_products, n, a, b).sum()


def histogram(data, bins=10, range=None, n=None):
    """Compute a histogram of array values with atomic bin counts.

    Values below the range are counted in the first bin and values above it
    in the last, unlike numpy.histogram, which drops them.

    Args:
        data: input array of any dtype
        bins: number of bins (at least 1)
        range: (min, max) tuple. If None, uses the minimum and maximum of
            the first n elements, and n must then be at least 1.
        n: number of elements (default: data.size)

    Returns:
        (counts, bin_edges) where counts is a Tack array of i32,
        bin_edges is a numpy array of (bins + 1) float64 edges.
    """
    import numpy as np
    n, data = _count(n, data)
    if int(bins) < 1:
        raise ValueError(f"histogram needs at least 1 bin, not {bins}")
    if range is None:
        if n == 0:
            raise ValueError("histogram of no elements needs an explicit range")
        head = _prefix(data, n)
        lo = float(head.min())
        hi = float(head.max())
        if not (np.isfinite(lo) and np.isfinite(hi)):
            # As numpy.histogram: bins cannot divide a range that is not finite.
            raise ValueError(f"autodetected range of [{lo}, {hi}] is not finite; "
                             "pass a finite range")
    else:
        lo, hi = float(range[0]), float(range[1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"histogram range [{lo}, {hi}] is not finite")
        if lo > hi:
            raise ValueError(f"histogram range [{lo}, {hi}] has its minimum above its maximum")

    # Avoid division by zero for constant arrays
    if hi == lo:
        hi = lo + 1.0

    bin_width = (hi - lo) / bins
    inv_bw = 1.0 / bin_width

    counts = tack.empty((bins,), tack.i32)
    counts.fill(0)
    _histogram_kernel(data, counts, lo, inv_bw, bins, n)

    edges = np.linspace(lo, hi, bins + 1)
    return counts, edges
