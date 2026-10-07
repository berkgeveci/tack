"""Datasets, and running a kernel over their cells one shape at a time.

A ``DataSet`` is a host object: point coordinates, a cell set, and named
point and cell data. Kernels never see it whole. ``for_each_shape`` runs a
kernel once per shape present, passing it the cells of that shape as a
cell view (see ``tack.data.cell_set``), followed by the caller's own
arguments::

    @tack.kernel
    def cell_centers(cells, out):
        for c in range(cells.num_cells):
            pc = cells.parametric_center()
            x = tack.Vector([0.0, 0.0, 0.0])
            for j in range(cells.NUM_POINTS):
                x += cells.shape_function(j, pc) * cells.point(c, j)
            out[cells.cell_id(c)] = x

    out = tack.Vector.field(3, tack.f32, shape=(data.cells.num_cells,))
    tack.data.for_each_shape(cell_centers, data, out)

Point and cell data are passed as ordinary arguments; index point data by
``cells.point_id(c, j)`` and cell data by ``cells.cell_id(c)``.
"""

import numpy as np

import tack
from tack.lang.field import Field

__all__ = ["DataSet", "for_each_shape"]


def _point_field(points, dtype):
    if isinstance(points, Field):
        if getattr(points, "_vector_n", None) != 3 or len(points._vector_shape()) != 2:
            raise ValueError("points must be a one-dimensional field of 3-vectors")
        return points
    points = np.ascontiguousarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (num_points, 3), not {points.shape}")
    field = tack.Vector.field(3, dtype, shape=(points.shape[0],))
    if points.size:
        field.from_numpy(points.astype(dtype.numpy_dtype))
    return field


class DataSet:
    """Point coordinates, a cell set, and named point and cell data.

    ``points`` is a ``(num_points, 3)`` NumPy array, copied to a field of
    ``dtype`` (f32 unless given), or a field of 3-vectors. ``cells`` is an
    ``ExplicitCellSet``, ``SingleTypeCellSet`` or ``StructuredCellSet``.
    ``point_data`` and ``cell_data`` map names to fields.
    """

    def __init__(self, points, cells, point_data=None, cell_data=None, dtype=tack.f32):
        self.points = _point_field(points, dtype)
        self.cells = cells
        self.point_data = dict(point_data or {})
        self.cell_data = dict(cell_data or {})

    @property
    def num_points(self):
        return self.points.shape[0] // 3

    @property
    def num_cells(self):
        return self.cells.num_cells


def for_each_shape(kernel, data, *args):
    """Run ``kernel(cells, *args)`` once per shape present in ``data``.

    ``data`` is a ``DataSet`` or a cell set. ``cells`` is a cell view: the
    shape's methods and constants, ``num_cells``, ``point_id(c, j)`` and
    ``cell_id(c)``, and for a ``DataSet`` also ``point(c, j)`` and
    ``gather_points(c, pts)``. Each shape compiles its own variant of the
    kernel once; a later call with a set of the same shape reuses it.
    """
    if isinstance(data, DataSet):
        views = data.cells.views(data.points)
    else:
        views = data.views()
    for view in views:
        kernel(view, *args)
