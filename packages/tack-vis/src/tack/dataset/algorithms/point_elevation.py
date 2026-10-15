"""``point_elevation``: each geometry value's position along the line from ``low_point`` to ``high_point``, clamped to it and mapped to ``range`` (VTK's vtkElevationFilter, Viskores' PointElevation): a field in the geometry's space."""

import numpy as np

import tack
from tack.dataset.algorithms._common import _scalars
from tack.dataset.arrays import materialize, size_of
from tack.dataset.dataset import Field

__all__ = ['point_elevation']


@tack.kernel
def _elevations(positions, lx, ly, lz, dx, dy, dz, length2, low, span, out):
    for i in range(out.shape[0]):
        p = positions[i]
        s = ((p[0] - lx) * dx + (p[1] - ly) * dy + (p[2] - lz) * dz) / length2
        out[i] = low + min(max(s, 0.0), 1.0) * span


def point_elevation(data, low_point=(0, 0, 0), high_point=(0, 0, 1), range=(0.0, 1.0)):
    """Each geometry value's position along the line from ``low_point`` to
    ``high_point``, clamped to it and mapped to ``range`` (VTK's
    vtkElevationFilter, Viskores' PointElevation): a field in the geometry's
    space."""
    low = np.asarray(low_point, float)
    direction = np.asarray(high_point, float) - low
    length2 = float(direction @ direction)
    if length2 == 0:
        raise ValueError("the low and high points must differ")
    n = size_of(data.geometry.values)
    out = _scalars(n, data.dtype)
    if n:
        _elevations(materialize(data.geometry.values), *map(float, low), *map(float, direction),
                    length2, float(range[0]), float(range[1] - range[0]), out)
    return Field(data.geometry.space, out)
