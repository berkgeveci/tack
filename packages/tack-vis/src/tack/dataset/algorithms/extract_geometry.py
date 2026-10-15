"""``extract_geometry``: the whole cells inside ``function``'s region (all their points at or below zero), or, with ``inside=False``, outside it (all above); with ``boundary``, also the cells it cuts (some point on the kept side). Cells are not split -- that is clip. Points are compacted and fields carried, as by ``threshold``."""

import numpy as np

from tack.dataset.algorithms._extract import _complement, _keep_cells
from tack.dataset.algorithms.implicit_values import implicit_values
from tack.dataset.algorithms.threshold import _threshold_flags, threshold

__all__ = ['extract_geometry']


def extract_geometry(data, function, inside=True, boundary=False, fields=None):
    """The whole cells inside ``function``'s region (all their points at or below zero),
    or, with ``inside=False``, outside it (all above); with ``boundary``, also the
    cells it cuts (some point on the kept side). Cells are not split -- that is
    clip. Points are compacted and fields carried, as by ``threshold``."""

    values = implicit_values(data, function)
    if inside:
        return threshold(data, values, -np.inf, 0.0, all_points=not boundary, fields=fields)
    # Outside is what is not inside, exactly: all points above zero is no point at
    # or below it, and some point above is not all at or below.
    keep = _threshold_flags(data, values, -np.inf, 0.0, all_points=boundary)
    _complement(keep)
    return _keep_cells(data, keep, fields)
