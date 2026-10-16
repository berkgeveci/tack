"""``mask_points``: every ``stride``-th point (0, stride, 2 * stride, ...) as vertex cells, as Viskores' ``MaskPoints``."""

import tack
from tack.dataset.algorithms._extract import _every, _keep_points
from tack.runtime.dispatch import on_inputs_device

__all__ = ['mask_points']


@on_inputs_device
def mask_points(data, stride, fields=None):
    """Every ``stride``-th point (0, stride, 2 * stride, ...) as vertex cells, as
    Viskores' ``MaskPoints``."""
    if int(stride) < 1:
        raise ValueError(f"stride must be at least 1, not {stride!r}")
    flags = tack.empty((data.num_points,), tack.i32)
    if data.num_points:
        _every(flags, int(stride))
    return _keep_points(data, flags, fields)
