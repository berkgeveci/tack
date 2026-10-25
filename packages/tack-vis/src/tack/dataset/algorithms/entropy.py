"""``entropy``: the Shannon entropy, in bits, of a scalar field's values in ``bins`` equal bins over their range: ``-sum(p log2 p)`` over the bins' shares, as Viskores' Entropy filter."""

import numpy as np

from tack.arrays import size_of, width_of
from tack.dataset.algorithms._common import _field
from tack.dataset.algorithms.statistics import _over_pieces
from tack.dataset.ghosts import owned_values
from tack.dataset.partitioned import PartitionedDataSet, fields_of
from tack.runtime.dispatch import on_inputs_device

__all__ = ['entropy']


@on_inputs_device
def _extrema(data, field):
    """The minimum and maximum of the values of one dataset's scalar field that it
    owns, or ``None`` for no values."""
    field = _field(data, field)
    if width_of(field.values):
        raise TypeError("entropy takes a scalar field")
    values = owned_values(data, field)
    if size_of(values) == 0:
        return None
    return values.min(), values.max()


@on_inputs_device
def _counts(data, field, bins, value_range, weights=None):
    from tack.algorithms import histogram

    field = _field(data, field)
    values = owned_values(data, field)
    if weights is not None:
        from tack.dataset.algorithms.statistics import _weights

        w = owned_values(data, _weights(data, field, weights))
        return _weighted_counts(values.numpy(), np.asarray(w.numpy(), float), bins,
                                value_range)
    counts, _ = histogram(values, bins=bins, range=value_range)
    return np.asarray(counts.numpy() if hasattr(counts, "numpy") else counts, float)


def _weighted_counts(values, weights, bins, value_range):
    """The weights' sum in each of ``bins`` equal bins over ``value_range`` (the
    values' own when None), values outside counted in the end bins, as the
    unweighted histogram counts them."""
    values = np.asarray(values, np.float64)
    lo, hi = value_range if value_range is not None else (values.min(), values.max())
    span = hi - lo
    index = np.zeros(len(values), np.int64) if span == 0 else \
        np.clip(((values - lo) / span * bins).astype(np.int64), 0, bins - 1)
    return np.bincount(index, weights=weights, minlength=bins).astype(float)


@on_inputs_device
def entropy(data, field, bins=10, weights=None, level=None):
    """The Shannon entropy, in bits, of a scalar field's values in ``bins`` equal bins
    over their range: ``-sum(p log2 p)`` over the bins' shares, as Viskores'
    Entropy filter. Ghosts are left out: cells and points ``data.ghosts``
    marks a duplicate or hidden.

    Over a ``PartitionedDataSet``, ``field`` names a field every piece has: the
    range is the whole's, and each piece's histogram over what it owns, made on
    its own device, is summed, over every rank of ``data.world``. Over an
    ``AMRDataSet``, the same over every box of every level, refined cells and
    points left out; over an ``OctreeDataSet``, its leaves (or one ``level``'s
    nodes). With ``weights`` (``"volume"`` or a field, as ``statistics`` takes
    them) each bin's share is the weights' sum in it over theirs."""
    from tack.dataset.amr import AMRDataSet
    from tack.dataset.octree import OctreeDataSet

    if isinstance(data, OctreeDataSet):
        return data._entropy(field, bins, weights, level)
    if level is not None:
        raise TypeError("level= picks an octree's level; this is not an OctreeDataSet")
    if isinstance(data, AMRDataSet):
        data = data._as_pieces()
    if isinstance(data, PartitionedDataSet):
        fields_of(data, field, "entropy")
        extrema = [e for e in _over_pieces(data, lambda piece: _extrema(piece, field))
                   if e is not None]
        if not extrema:
            raise ValueError("entropy of no values")
        value_range = (float(np.min([e[0] for e in extrema])),
                       float(np.max([e[1] for e in extrema])))

        def counts(piece):
            if _extrema(piece, field) is None:
                return np.zeros(bins)
            return _counts(piece, field, bins, value_range, weights)
        p = sum(_over_pieces(data, counts))
    else:
        if _extrema(data, field) is None:
            raise ValueError("entropy of no values")
        # The histogram's own range, as before: it refuses one that is not finite.
        p = _counts(data, field, bins, None, weights)
    return _bits(p)


def _bits(p):
    """The entropy, in bits, of bins' counts (or weights)."""
    p = np.asarray(p, float)
    p = p[p > 0] / p.sum()
    return 0.0 - float((p * np.log2(p)).sum())     # 0.0, not -0.0, for one value
