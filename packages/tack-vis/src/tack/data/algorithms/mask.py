"""``mask``: every ``stride``-th cell (0, stride, 2 * stride, ...), as Viskores' ``Mask``."""

import tack
from tack.data.algorithms._extract import _every, _keep_cells

__all__ = ['mask']


def mask(data, stride, fields=None):
    """Every ``stride``-th cell (0, stride, 2 * stride, ...), as Viskores' ``Mask``."""
    if int(stride) < 1:
        raise ValueError(f"stride must be at least 1, not {stride!r}")
    keep = tack.field(tack.i32, shape=(data.num_cells,))
    if data.num_cells:
        _every(keep, int(stride))
    return _keep_cells(data, keep, fields)
