"""``point_cloud``: every point as a vertex cell, with the point data (Viskores' ConvertToPointCloud); cell data is left behind."""

import tack
from tack.dataset.algorithms._extract import _keep_points
from tack.runtime.dispatch import on_inputs_device

__all__ = ['point_cloud']


@on_inputs_device
def point_cloud(data, fields=None):
    """Every point as a vertex cell, with the point data (Viskores'
    ConvertToPointCloud); cell data is left behind."""
    flags = tack.full((data.num_points,), 1, tack.i32) if data.num_points else \
        tack.empty((0,), tack.i32)
    return _keep_points(data, flags, fields)
