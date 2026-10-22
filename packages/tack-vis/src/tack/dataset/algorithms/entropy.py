"""``entropy``: the Shannon entropy, in bits, of a scalar field's values in ``bins`` equal bins over their range: ``-sum(p log2 p)`` over the bins' shares, as Viskores' Entropy filter."""

import numpy as np

from tack.arrays import materialize, size_of, width_of
from tack.dataset.algorithms._common import _field
from tack.dataset.partitioned import PartitionedDataSet, fields_of
from tack.runtime.dispatch import on_inputs_device

__all__ = ['entropy']


@on_inputs_device
def _extrema(data, field):
    """One dataset's scalar field's minimum and maximum, or ``None`` for no values."""
    field = _field(data, field)
    if width_of(field.values):
        raise TypeError("entropy takes a scalar field")
    if size_of(field.values) == 0:
        return None
    values = materialize(field.values)
    return values.min(), values.max()


@on_inputs_device
def _counts(data, field, bins, value_range):
    from tack.algorithms import histogram

    counts, _ = histogram(materialize(_field(data, field).values), bins=bins,
                          range=value_range)
    return np.asarray(counts.numpy() if hasattr(counts, "numpy") else counts, float)


@on_inputs_device
def entropy(data, field, bins=10):
    """The Shannon entropy, in bits, of a scalar field's values in ``bins`` equal bins
    over their range: ``-sum(p log2 p)`` over the bins' shares, as Viskores'
    Entropy filter.

    Over a ``PartitionedDataSet``, ``field`` names a field every piece has: the
    range is the whole's, and each piece's histogram over it, made on its own
    device, is summed. A point on the boundary between two pieces is in both,
    and counts once per piece."""
    if isinstance(data, PartitionedDataSet):
        fields_of(data, field, "entropy")
        pieces = [(piece, _extrema(piece, field)) for piece in data]
        pieces = [(piece, e) for piece, e in pieces if e is not None]
        if not pieces:
            raise ValueError("entropy of no values")
        value_range = (float(np.min([e[0] for _, e in pieces])),
                       float(np.max([e[1] for _, e in pieces])))
        p = sum(_counts(piece, field, bins, value_range) for piece, _ in pieces)
    else:
        if _extrema(data, field) is None:
            raise ValueError("entropy of no values")
        # The histogram's own range, as before: it refuses one that is not finite.
        p = _counts(data, field, bins, None)
    p = p[p > 0] / p.sum()
    return 0.0 - float((p * np.log2(p)).sum())     # 0.0, not -0.0, for one value
