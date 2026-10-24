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
def _counts(data, field, bins, value_range):
    from tack.algorithms import histogram

    counts, _ = histogram(owned_values(data, _field(data, field)), bins=bins,
                          range=value_range)
    return np.asarray(counts.numpy() if hasattr(counts, "numpy") else counts, float)


@on_inputs_device
def entropy(data, field, bins=10):
    """The Shannon entropy, in bits, of a scalar field's values in ``bins`` equal bins
    over their range: ``-sum(p log2 p)`` over the bins' shares, as Viskores'
    Entropy filter. Ghosts are left out: cells and points ``data.ghosts``
    marks a duplicate or hidden.

    Over a ``PartitionedDataSet``, ``field`` names a field every piece has: the
    range is the whole's, and each piece's histogram over what it owns, made on
    its own device, is summed, over every rank of ``data.world``."""
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
            return _counts(piece, field, bins, value_range)
        p = sum(_over_pieces(data, counts))
    else:
        if _extrema(data, field) is None:
            raise ValueError("entropy of no values")
        # The histogram's own range, as before: it refuses one that is not finite.
        p = _counts(data, field, bins, None)
    p = p[p > 0] / p.sum()
    return 0.0 - float((p * np.log2(p)).sum())     # 0.0, not -0.0, for one value
