"""``statistics``: descriptive statistics of a scalar field's values, as Viskores' Statistics filter gives them: a ``Statistics`` (a NamedTuple) of ``n``, ``min``, ``max``, ``sum``, ``mean``, the central moment sums ``m2``, ``m3``, ``m4``, ``sample_variance`` (``m2 / (n - 1)``), ``population_variance`` (``m2 / n``), their standard deviations, ``skewness`` (``sqrt(n) m3 / m2^1.5``) and ``kurtosis`` (``n m4 / m2^2``, not the excess), the last two 0 for a constant field. Moments are taken about the mean, in a second pass."""

import math
from typing import NamedTuple

import numpy as np

import tack
from tack.arrays import size_of, width_of
from tack.dataset.algorithms._common import _converted, _field, _floating_dtype
from tack.dataset.ghosts import owned_values
from tack.dataset.partitioned import PartitionedDataSet, fields_of
from tack.runtime.dispatch import on_inputs_device

__all__ = ['Statistics', 'statistics']


@tack.kernel
def _central_powers(values, mean, p2, p3, p4):
    for i in range(p2.shape[0]):
        d = values[i] - mean
        d2 = d * d
        p2[i] = d2
        p3[i] = d2 * d
        p4[i] = d2 * d2


class Statistics(NamedTuple):
    """What ``statistics`` returns: Python numbers, by name (``s.mean``)."""
    n: int
    min: float
    max: float
    sum: float
    mean: float
    m2: float
    m3: float
    m4: float
    sample_variance: float
    population_variance: float
    sample_stddev: float
    population_stddev: float
    skewness: float
    kurtosis: float


@on_inputs_device
def _moments(data, field):
    """``(n, min, max, sum, mean, m2, m3, m4)`` of one dataset's scalar field, over the
    values it owns (not its ghosts), the moments about their own mean in a
    second pass; ``None`` for no values."""
    field = _field(data, field)
    if width_of(field.values):
        raise TypeError("statistics takes a scalar field")
    source = owned_values(data, field)
    n = size_of(source)
    if n == 0:
        return None
    values = _converted(source, _floating_dtype(source))
    # Integer sums are exact in f64 up to 2**53 and cannot wrap; summing the
    # converted f32 values was off by 57 for a million i32 values of 1001.
    # NumPy reduces integer fields on the host anyway.
    total = values.sum() if values is source else float(
        np.sum(source.numpy(), dtype=np.float64))
    mean = total / n
    # The powers of differences from a mean are fractions, whatever the values are.
    p2, p3, p4 = (tack.empty((n,), values.dtype) for _ in range(3))
    _central_powers(values, mean, p2, p3, p4)
    return n, source.min(), source.max(), total, mean, p2.sum(), p3.sum(), p4.sum()


def _merged(a, b):
    """The moments of two sets of values from each one's (Pebay's pairwise update,
    "Formulas for robust, one-pass parallel computation of covariances and
    arbitrary-order statistical moments", 2008)."""
    na, mina, maxa, suma, meana, m2a, m3a, m4a = a
    nb, minb, maxb, sumb, meanb, m2b, m3b, m4b = b
    n = na + nb
    d = meanb - meana
    m2 = m2a + m2b + d * d * na * nb / n
    m3 = (m3a + m3b + d ** 3 * na * nb * (na - nb) / n ** 2
          + 3.0 * d * (na * m2b - nb * m2a) / n)
    m4 = (m4a + m4b + d ** 4 * na * nb * (na * na - na * nb + nb * nb) / n ** 3
          + 6.0 * d * d * (na * na * m2b + nb * nb * m2a) / n ** 2
          + 4.0 * d * (na * m3b - nb * m3a) / n)
    total = suma + sumb
    # NaN propagates through the extrema, as an array's own do.
    return (n, float(np.min([mina, minb])), float(np.max([maxa, maxb])), total, total / n,
            m2, m3, m4)


def _over_pieces(data, each):
    """``each(piece)`` for every piece of a partitioned dataset over every rank of its
    world, in the pieces' global order."""
    from tack import communication as tc

    pieces = list(data)
    blocks = tc.Blocks([piece.device for piece in pieces], data.world)
    return blocks.all_gather({g: each(piece) for g, piece in zip(blocks.local, pieces)})


def _summary(n, low, high, total, mean, m2, m3, m4):
    skewness = math.sqrt(n) * m3 / m2 ** 1.5 if m2 else 0.0
    kurtosis = n * m4 / (m2 * m2) if m2 else 0.0
    return Statistics(n=n, min=low, max=high, sum=total, mean=mean,
                      m2=m2, m3=m3, m4=m4,
                      sample_variance=m2 / (n - 1) if n > 1 else 0.0,
                      population_variance=m2 / n,
                      sample_stddev=math.sqrt(m2 / (n - 1)) if n > 1 else 0.0,
                      population_stddev=math.sqrt(m2 / n),
                      skewness=skewness, kurtosis=kurtosis)


@on_inputs_device
def statistics(data, field):
    """Descriptive statistics of a scalar field's values, as Viskores' Statistics
    filter gives them: a ``Statistics`` of ``n``, ``min``, ``max``, ``sum``, ``mean``, the
    central moment sums ``m2``, ``m3``, ``m4``, ``sample_variance``
    (``m2 / (n - 1)``), ``population_variance`` (``m2 / n``), their standard
    deviations, ``skewness`` (``sqrt(n) m3 / m2^1.5``) and ``kurtosis``
    (``n m4 / m2^2``, not the excess), the last two 0 for a constant field.
    Moments are taken about the mean, in a second pass. Ghosts are left out:
    cells and points ``data.ghosts`` marks a duplicate (another piece owns
    them) or hidden.

    Over a ``PartitionedDataSet``, ``field`` names a field every piece has: each
    piece's moments, over what it owns, are taken on its own device and
    combined pairwise, in the order of the pieces over every rank of
    ``data.world`` -- the same result however the pieces are shared out. Where
    every shared cell and point has one owner, the result is the whole's up to
    rounding. Over an ``AMRDataSet``, the same over every box of every level:
    refined cells and points are left out, so each region of space counts
    once, at its finest level."""
    from tack.dataset.amr import AMRDataSet

    if isinstance(data, AMRDataSet):
        data = data._as_pieces()
    if isinstance(data, PartitionedDataSet):
        fields_of(data, field, "statistics")
        moments = None
        for part in _over_pieces(data, lambda piece: _moments(piece, field)):
            if part is not None:
                moments = part if moments is None else _merged(moments, part)
    else:
        moments = _moments(data, field)
    if moments is None:
        raise ValueError("statistics of no values")
    return _summary(*moments)
