"""``extract_geometry``: the whole cells inside ``function``'s region (all their points at or below zero), or, with ``inside=False``, outside it (all above); with ``boundary``, also the cells it cuts (some point on the kept side). Cells are not split -- that is clip. Points are compacted and fields carried, as by ``threshold``."""

import numpy as np

from tack.data.algorithms.implicit_values import implicit_values
from tack.data.algorithms.threshold import threshold

__all__ = ['extract_geometry']


def extract_geometry(data, function, inside=True, boundary=False, fields=None):
    """The whole cells inside ``function``'s region (all their points at or below zero),
    or, with ``inside=False``, outside it (all above); with ``boundary``, also the
    cells it cuts (some point on the kept side). Cells are not split -- that is
    clip. Points are compacted and fields carried, as by ``threshold``."""

    values = implicit_values(data, function)
    lower, upper = (-np.inf, 0.0) if inside else (np.nextafter(0.0, 1.0), np.inf)
    return threshold(data, values, lower, upper, all_points=not boundary, fields=fields)
