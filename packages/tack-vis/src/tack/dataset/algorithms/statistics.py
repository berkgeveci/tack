"""``statistics``: descriptive statistics of a scalar field's values, as Viskores' Statistics filter gives them: a ``Statistics`` (a NamedTuple) of ``n``, ``min``, ``max``, ``sum``, ``mean``, the central moment sums ``m2``, ``m3``, ``m4``, ``sample_variance`` (``m2 / (n - 1)``), ``population_variance`` (``m2 / n``), their standard deviations, ``skewness`` (``sqrt(n) m3 / m2^1.5``) and ``kurtosis`` (``n m4 / m2^2``, not the excess), the last two 0 for a constant field. Moments are taken about the mean, in a second pass."""

import math
from typing import NamedTuple, Optional

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


@tack.kernel
def _weighted(values, weights, wx, w2):
    for i in range(wx.shape[0]):
        w = weights[i]
        wx[i] = w * values[i]
        w2[i] = w * w


@tack.kernel
def _weighted_powers(values, weights, mean, p2, p3, p4):
    for i in range(p2.shape[0]):
        w = weights[i]
        d = values[i] - mean
        d2 = d * d
        p2[i] = w * d2
        p3[i] = w * d2 * d
        p4[i] = w * d2 * d2


class Statistics(NamedTuple):
    """What ``statistics`` returns: Python numbers, by name (``s.mean``), and the
    field's ``unit`` and the weights' (``weight_unit``: ``"dimensionless"``
    without weights, a measure's unit for ``"volume"``); ``unit_of(name)`` gives
    each number's."""
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
    #: The sum of the weights: ``n`` without them.
    weight: float = 0.0
    #: The values' unit (``Field.unit``), or ``None`` if unknown.
    unit: Optional[str] = None
    #: The weights' unit: ``"dimensionless"`` without weights.
    weight_unit: Optional[str] = None

    def unit_of(self, name):
        """The unit of the statistic ``name`` (``"mean"``, ``"sum"``, ...), or
        ``None`` if unknown: ``min``, ``max`` and ``mean`` are in the values'
        unit, ``sum`` in it times the weights', the moment sums ``m2`` ... ``m4``
        in its powers times the weights', the variances in its square and the
        standard deviations in it (a difference of degrees Celsius is in
        kelvin), ``skewness``, ``kurtosis`` and ``n`` pure numbers, ``weight``
        in the weights' unit."""
        from tack.dataset import units

        u, w = self.unit, self.weight_unit
        if name not in self._fields or name in ("unit", "weight_unit"):
            raise KeyError(f"no statistic {name!r}")
        if name in ("min", "max", "mean"):
            return u
        if name == "sum":
            return u if w == "dimensionless" else units.derived(u, w)
        if name in ("m2", "m3", "m4"):
            return units.derived((units.delta(u), int(name[1])), w)
        if name.endswith("variance"):
            return units.derived((units.delta(u), 2))
        if name.endswith("stddev"):
            return units.delta(u)
        if name == "weight":
            return w
        return "dimensionless"                   # n, skewness, kurtosis


@on_inputs_device
def _moments(data, field, weights=None):
    """``(n, W, W2, min, max, sum, mean, m2, m3, m4)`` of one dataset's scalar field,
    over the values it owns (not its ghosts), the moments about their own mean
    in a second pass, weighted by ``weights`` (``W`` their sum, ``W2`` the sum of
    their squares; both ``n`` without); ``None`` for no values."""
    field = _field(data, field)
    if width_of(field.values):
        raise TypeError("statistics takes a scalar field")
    source = owned_values(data, field)
    return _array_moments(source, None if weights is None
                          else owned_values(data, _weights(data, field, weights)))


def _units(data, field, weights):
    """``(unit, weight_unit)`` of one dataset's ``field`` weighted by ``weights``."""
    from tack.dataset.algorithms._common import _measure_unit

    field = _field(data, field)
    if weights is None:
        return field.unit, "dimensionless"
    if isinstance(weights, str) and weights == "volume":
        return field.unit, _measure_unit(data)
    return field.unit, _field(data, weights).unit


def _weights(data, field, weights):
    """The ``Field`` of weights: each cell's measure for ``"volume"``, or a field of
    ``data`` (a name or a ``Field``) on the same entities as ``field``."""
    from tack.dataset.algorithms.cell_geometry import cell_geometry
    from tack.dataset.algorithms.ghost_cells import _kind

    kind = _kind(field)
    if isinstance(weights, str) and weights == "volume":
        if kind != "cells":
            raise TypeError("volume weights are a cell's measure: they take a cell field")
        return cell_geometry(data).measure
    weights = _field(data, weights)
    if _kind(weights) != kind or width_of(weights.values):
        raise TypeError("weights are a scalar field on the same entities as the field "
                        "(points or cells), or \"volume\"")
    return weights


@on_inputs_device
def _array_moments(source, weights=None):
    """``_moments`` of an array of values (and of weights, one per value)."""
    n = size_of(source)
    if n == 0:
        return None
    if weights is not None:
        return _weighted_moments(source, weights, n)
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
    return (n, float(n), float(n), source.min(), source.max(), total, mean, p2.sum(),
            p3.sum(), p4.sum())


def _weighted_moments(source, weights, n):
    from tack import arrays

    values = _converted(source, _floating_dtype(source))
    weights = _converted(arrays.materialize(weights), values.dtype)
    wx, w2 = tack.empty((n,), values.dtype), tack.empty((n,), values.dtype)
    _weighted(values, weights, wx, w2)
    total_weight = weights.sum()
    if not total_weight > 0:
        raise ValueError("statistics with weights needs a positive sum of weights")
    total = wx.sum()
    mean = total / total_weight
    p2, p3, p4 = (tack.empty((n,), values.dtype) for _ in range(3))
    _weighted_powers(values, weights, mean, p2, p3, p4)
    return (n, total_weight, w2.sum(), source.min(), source.max(), total, mean, p2.sum(),
            p3.sum(), p4.sum())


def _merged(a, b):
    """The moments of two sets of values from each one's (Pebay's pairwise update,
    "Formulas for robust, one-pass parallel computation of covariances and
    arbitrary-order statistical moments", 2008)."""
    count_a, wa, w2a, mina, maxa, suma, meana, m2a, m3a, m4a = a
    count_b, wb, w2b, minb, maxb, sumb, meanb, m2b, m3b, m4b = b
    # Weights stand for counts; without weights they are the counts.
    na, nb = wa, wb
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
    return (count_a + count_b, n, w2a + w2b, float(np.min([mina, minb])),
            float(np.max([maxa, maxb])), total, total / n, m2, m3, m4)


def _over_pieces(data, each):
    """``each(piece)`` for every piece of a partitioned dataset over every rank of its
    world, in the pieces' global order."""
    from tack import communication as tc

    pieces = list(data)
    blocks = tc.Blocks([piece.device for piece in pieces], data.world)
    return blocks.all_gather({g: each(piece) for g, piece in zip(blocks.local, pieces)})


def _summary(n, weight, weight2, low, high, total, mean, m2, m3, m4):
    # With weights, W stands for n; the sample variance takes them as
    # reliability weights (NumPy's aweights), m2 / (W - W2 / W), which is
    # m2 / (n - 1) without them.
    skewness = math.sqrt(weight) * m3 / m2 ** 1.5 if m2 else 0.0
    kurtosis = weight * m4 / (m2 * m2) if m2 else 0.0
    dof = weight - weight2 / weight
    sample = m2 / dof if n > 1 and dof > 0 else 0.0
    return Statistics(n=n, min=low, max=high, sum=total, mean=mean,
                      m2=m2, m3=m3, m4=m4,
                      sample_variance=sample,
                      population_variance=m2 / weight,
                      sample_stddev=math.sqrt(sample),
                      population_stddev=math.sqrt(m2 / weight),
                      skewness=skewness, kurtosis=kurtosis, weight=weight)


@on_inputs_device
def statistics(data, field, weights=None, level=None):
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
    once, at its finest level. Over an ``OctreeDataSet``, its leaves (or the
    nodes of one ``level``), every rank's.

    ``weights`` weighs each value: ``"volume"``, each cell's measure (a cell
    field's), or a scalar field on the same entities, by name (a density, for
    mass-weighted moments). Then ``mean`` is ``sum(w x) / W``, ``sum`` is ``sum(w
    x)``, the moments are ``sum(w (x - mean)^k)``, ``weight`` is ``W``, the sum of
    the weights, the population variance ``m2 / W``, the sample variance
    ``m2 / (W - sum(w^2) / W)`` (reliability weights, NumPy's ``aweights``), and
    skewness and kurtosis take ``W`` for ``n``. Cells of different sizes -- an
    AMR hierarchy's levels, an octree's leaves -- then count by their size."""
    from tack.dataset.amr import AMRDataSet
    from tack.dataset.octree import OctreeDataSet

    if isinstance(data, OctreeDataSet):
        return data._statistics(field, weights, level)
    if level is not None:
        raise TypeError("level= picks an octree's level; this is not an OctreeDataSet")
    if isinstance(data, AMRDataSet):
        data = data._as_pieces()
    if isinstance(data, PartitionedDataSet):
        fields_of(data, field, "statistics")
        from tack.dataset import units

        # Every rank's pieces' units, so a rank holding none agrees with the rest.
        every = [pair for part in data.world._allgather(
            [_units(piece, field, weights) for piece in data]) for pair in part]
        unit = units.common("statistics' pieces' field", [u for u, _ in every])
        weight_unit = units.common("statistics' pieces' weights", [w for _, w in every])
        moments = None
        for part in _over_pieces(data, lambda piece: _moments(piece, field, weights)):
            if part is not None:
                moments = part if moments is None else _merged(moments, part)
    else:
        unit, weight_unit = _units(data, field, weights)
        moments = _moments(data, field, weights)
    if moments is None:
        raise ValueError("statistics of no values")
    return _summary(*moments)._replace(unit=unit, weight_unit=weight_unit)
