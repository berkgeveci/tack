"""``extract_points``: the points inside ``function``'s region (at or below zero), or outside it (above), as vertex cells. Needs an ``H1`` geometry: one position per point."""

import numpy as np

from tack.data.algorithms._extract import _keep_points, _point_flags
from tack.data.algorithms.implicit_values import implicit_values
from tack.data.carry import _point_values
from tack.data.spaces import H1

__all__ = ['extract_points']


def extract_points(data, function, inside=True, fields=None):
    """The points inside ``function``'s region (at or below zero), or outside it
    (above), as vertex cells. Needs an ``H1`` geometry: one position per point."""

    if not isinstance(data.geometry.space, H1):
        raise TypeError("extract_points needs an H1 geometry: one position per point")
    values = _point_values(implicit_values(data, function).values, data.num_points)
    lower, upper = (-np.inf, 0.0) if inside else (np.nextafter(0.0, 1.0), np.inf)
    return _keep_points(data, _point_flags(data, values, lower, upper), fields)
