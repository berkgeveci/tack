"""``mask``: every ``stride``-th cell (0, stride, 2 * stride, ...), as Viskores' ``Mask``."""

import tack
from tack.dataset.algorithms._extract import _every, _keep_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['mask']


@on_inputs_device
def mask(data, stride, fields=None):
    """Every ``stride``-th cell (0, stride, 2 * stride, ...), as Viskores' ``Mask``."""
    if int(stride) < 1:
        raise ValueError(f"stride must be at least 1, not {stride!r}")
    keep = tack.empty((data.num_cells,), tack.i32)
    if data.num_cells:
        _every(keep, int(stride))
    return _keep_cells(data, keep, fields)
