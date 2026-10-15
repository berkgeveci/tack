"""``entropy``: the Shannon entropy, in bits, of a scalar field's values in ``bins`` equal bins over their range: ``-sum(p log2 p)`` over the bins' shares, as Viskores' Entropy filter."""

import numpy as np

from tack.arrays import materialize, size_of, width_of
from tack.dataset.algorithms._common import _field

__all__ = ['entropy']


def entropy(data, field, bins=10):
    """The Shannon entropy, in bits, of a scalar field's values in ``bins`` equal bins
    over their range: ``-sum(p log2 p)`` over the bins' shares, as Viskores'
    Entropy filter."""
    field = _field(data, field)
    from tack.algorithms import histogram

    if width_of(field.values):
        raise TypeError("entropy takes a scalar field")
    if size_of(field.values) == 0:
        raise ValueError("entropy of no values")
    counts, _ = histogram(materialize(field.values), bins=bins)
    p = np.asarray(counts.numpy() if hasattr(counts, "numpy") else counts, float)
    p = p[p > 0] / p.sum()
    return 0.0 - float((p * np.log2(p)).sum())     # 0.0, not -0.0, for one value
