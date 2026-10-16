"""``extract_points``: the points inside ``function``'s region (at or below zero), or outside it (above), as vertex cells. Needs an ``H1`` geometry: one position per point."""

import numpy as np

from tack.dataset.algorithms._extract import _complement, _keep_points, _point_flags
from tack.dataset.algorithms.implicit_values import implicit_values
from tack.dataset.carry import _point_values
from tack.dataset.spaces import H1
from tack.runtime.dispatch import on_inputs_device

__all__ = ['extract_points']


@on_inputs_device
def extract_points(data, function, inside=True, fields=None):
    """The points inside ``function``'s region (at or below zero), or outside it
    (above), as vertex cells. Needs an ``H1`` geometry: one position per point."""

    if not isinstance(data.geometry.space, H1):
        raise TypeError("extract_points needs an H1 geometry: one position per point")
    values = _point_values(implicit_values(data, function).values, data.num_points)
    # Outside is what is not inside: "above zero" as a range would need a bound
    # between zero and the smallest positive value, which f32 kernels (and GPUs
    # that flush denormals) do not keep.
    flags = _point_flags(data, values, -np.inf, 0.0)
    if not inside:
        _complement(flags)
    return _keep_points(data, flags, fields)
