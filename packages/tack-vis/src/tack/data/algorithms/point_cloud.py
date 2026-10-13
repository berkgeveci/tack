"""``point_cloud``: every point as a vertex cell, with the point data (Viskores' ConvertToPointCloud); cell data is left behind."""

import tack
from tack.data.algorithms._extract import _keep_points

__all__ = ['point_cloud']


def point_cloud(data, fields=None):
    """Every point as a vertex cell, with the point data (Viskores'
    ConvertToPointCloud); cell data is left behind."""
    flags = tack.full(tack.i32, (data.num_points,), 1) if data.num_points else \
        tack.field(tack.i32, shape=(0,))
    return _keep_points(data, flags, fields)
