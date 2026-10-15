"""``statistics``: descriptive statistics of a scalar field's values, as Viskores' Statistics filter gives them: a dict of ``n``, ``min``, ``max``, ``sum``, ``mean``, the central moment sums ``m2``, ``m3``, ``m4``, ``sample_variance`` (``m2 / (n - 1)``), ``population_variance`` (``m2 / n``), their standard deviations, ``skewness`` (``sqrt(n) m3 / m2^1.5``) and ``kurtosis`` (``n m4 / m2^2``, not the excess), the last two 0 for a constant field. Moments are taken about the mean, in a second pass."""

import math

import numpy as np

import tack
from tack.arrays import materialize, size_of, width_of
from tack.dataset.algorithms._common import _converted, _field, _floating_dtype

__all__ = ['statistics']


@tack.kernel
def _central_powers(values, mean, p2, p3, p4):
    for i in range(p2.shape[0]):
        d = values[i] - mean
        d2 = d * d
        p2[i] = d2
        p3[i] = d2 * d
        p4[i] = d2 * d2


def statistics(data, field):
    """Descriptive statistics of a scalar field's values, as Viskores' Statistics
    filter gives them: a dict of ``n``, ``min``, ``max``, ``sum``, ``mean``, the
    central moment sums ``m2``, ``m3``, ``m4``, ``sample_variance``
    (``m2 / (n - 1)``), ``population_variance`` (``m2 / n``), their standard
    deviations, ``skewness`` (``sqrt(n) m3 / m2^1.5``) and ``kurtosis``
    (``n m4 / m2^2``, not the excess), the last two 0 for a constant field.
    Moments are taken about the mean, in a second pass."""
    field = _field(data, field)
    if width_of(field.values):
        raise TypeError("statistics takes a scalar field")
    source = materialize(field.values)
    values = _converted(source, _floating_dtype(source))
    n = size_of(field.values)
    if n == 0:
        raise ValueError("statistics of no values")
    # Integer sums are exact in f64 up to 2**53 and cannot wrap; summing the
    # converted f32 values was off by 57 for a million i32 values of 1001.
    # NumPy reduces integer fields on the host anyway.
    total = values.sum() if values is source else float(
        np.sum(source.numpy(), dtype=np.float64))
    mean = total / n
    # The powers of differences from a mean are fractions, whatever the values are.
    p2, p3, p4 = (tack.empty((n,), values.dtype) for _ in range(3))
    _central_powers(values, mean, p2, p3, p4)
    m2, m3, m4 = p2.sum(), p3.sum(), p4.sum()
    skewness = math.sqrt(n) * m3 / m2 ** 1.5 if m2 else 0.0
    kurtosis = n * m4 / (m2 * m2) if m2 else 0.0
    return {"n": n, "min": source.min(), "max": source.max(), "sum": total, "mean": mean,
            "m2": m2, "m3": m3, "m4": m4,
            "sample_variance": m2 / (n - 1) if n > 1 else 0.0,
            "population_variance": m2 / n,
            "sample_stddev": math.sqrt(m2 / (n - 1)) if n > 1 else 0.0,
            "population_stddev": math.sqrt(m2 / n),
            "skewness": skewness, "kurtosis": kurtosis}
