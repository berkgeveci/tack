"""``mask_points``: every ``stride``-th point (0, stride, 2 * stride, ...) as vertex cells, as Viskores' ``MaskPoints``."""

import tack
from tack.data.algorithms._extract import _every, _keep_points

__all__ = ['mask_points']


def mask_points(data, stride, fields=None):
    """Every ``stride``-th point (0, stride, 2 * stride, ...) as vertex cells, as
    Viskores' ``MaskPoints``."""
    if int(stride) < 1:
        raise ValueError(f"stride must be at least 1, not {stride!r}")
    flags = tack.field(tack.i32, shape=(data.num_points,))
    if data.num_points:
        _every(flags, int(stride))
    return _keep_points(data, flags, fields)
