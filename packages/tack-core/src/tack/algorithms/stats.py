"""Statistics and analysis on tack fields.

Each statistic forms its terms -- |x|, x², (x - mean)², a·b, x ≠ 0 -- in a
temporary field, in the precision of its inputs (f64 if any is f64,
otherwise f32), and reduces that field with ``Field.sum()`` or
``Field.max()``. They therefore follow the field-reduction contract: f64
inputs accumulate in f64, and the error budget and repeatability are those
of field reductions. ``count_nonzero`` scans its 0/1 flags into i64. (They used to combine into a
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
# when a field argument is f64, so ``x * one`` converts an f32 or integer
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

def _count(n, *fields):
    """Resolve and check the element count, and flatten vector fields.

    Every kernel reads [0, n) scalar elements. A vector field is reduced
    over all its components in storage order -- its ``size`` already counts
    them -- through a reshape view, since the kernels index scalars and a
    vector element in `a[i] * b[i]` has no scalar meaning.
    Returns ``(n, *fields)`` with the views substituted.
    """
    flat = tuple(f.reshape((f.size,)) if hasattr(f, '_vector_n') else f for f in fields)
    if n is None:
        n = flat[0].size
    n = int(n)
    for f in flat:
        if not 0 <= n <= f.size:
            raise ValueError(
                f"n={n} is outside [0, {f.size}] for a field of {f.size} elements")
    return (n, *flat)


def _prefix(data, n):
    """The first n elements of data, as a field the host reductions accept."""
    if n == data.size:
        return data
    from tack.algorithms.copy import copy
    head = tack.field(dtype=data.dtype, shape=(n,))
    copy(data, head, n)
    return head


def _precision(*fields):
    """The accumulation precision of a statistic of ``fields``: f64 if any is f64."""
    return tack.f64 if any(f.dtype is tack.f64 for f in fields) else tack.f32


def var(data, n=None):
    """Population variance of a field: Σ(x - mean)² / n.

    Two passes over the first n elements: the mean by ``Field.sum()``, then
    the squared differences, formed in the data's precision (f64 for f64
    data, otherwise f32) and summed by ``Field.sum()``. Returns a Python
    float; an empty range gives NaN, as ``Field.mean()`` does.
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
    terms = tack.field(dtype=_precision(data), shape=(n,))
    _squared_diffs(data, mean_val, 1.0, terms, n)
    return terms.sum() / n


def std(data, n=None):
    """Population standard deviation of a field."""
    from math import sqrt
    return sqrt(var(data, n))


def _terms(kernel, n, *fields):
    """``kernel``'s terms over the first n elements, in the fields' precision."""
    terms = tack.field(dtype=_precision(*fields), shape=(n,))
    kernel(*fields, 1.0, terms, n)
    return terms


def norm(data, ord=2, n=None):
    """Vector norm of a field, in the data's precision (f64 for f64 data,
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
    """Maximum absolute value of a field, as a Python float; 0.0 when empty."""
    return norm(data, float('inf'), n)


def count_nonzero(data, n=None):
    """The number of nonzero elements of a field, as a Python int: its 0/1 flags
    scanned into i64 on the device, so the count is exact past 2^31 and only
    the total comes back to the host."""
    from tack.algorithms.scan import exclusive_scan

    n, data = _count(n, data)
    if n == 0:
        return 0
    flags = tack.field(dtype=tack.i32, shape=(n,))
    _nonzero(data, flags, n)
    return int(exclusive_scan(flags, tack.field(dtype=tack.i64, shape=(n,)), n))


def dot(a, b, n=None):
    """Dot product of two fields, Σa[i]·b[i], in their precision (f64 if either
    is f64, otherwise f32), as a Python float; 0.0 when empty."""
    n, a, b = _count(n, a, b)
    if n == 0:
        return 0.0
    return _terms(_products, n, a, b).sum()


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
    n, data = _count(n, data)
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
