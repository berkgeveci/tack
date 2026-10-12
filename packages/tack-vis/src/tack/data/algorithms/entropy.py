"""``entropy``: the Shannon entropy, in bits, of a scalar field's values in ``bins`` equal bins over their range: ``-sum(p log2 p)`` over the bins' shares, as Viskores' Entropy filter."""

import numpy as np

from tack.data.arrays import materialize, width_of

__all__ = ['entropy']


def entropy(field, bins=10):
    """The Shannon entropy, in bits, of a scalar field's values in ``bins`` equal bins
    over their range: ``-sum(p log2 p)`` over the bins' shares, as Viskores'
    Entropy filter."""
    from tack.algorithms import histogram

    if width_of(field.values):
        raise TypeError("entropy takes a scalar field")
    counts, _ = histogram(materialize(field.values), bins=bins)
    p = np.asarray(counts.to_numpy() if hasattr(counts, "to_numpy") else counts, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())
