"""``to_cells``: ``field`` averaged over each cell's corners: a ``Constant`` field."""

import tack
from tack.data.algorithms._common import _all
from tack.data.arrays import materialize
from tack.data.carry import _like, _take
from tack.data.dataset import Field, for_each
from tack.data.spaces import H1, Constant, Values

__all__ = ['to_cells']


@tack.kernel
def _corner_averages(cells, u, out):
    for c in cells:
        total = u.corner_value(c, 0)
        for j in range(1, cells.NUM_POINTS):
            total += u.corner_value(c, j)
        out[cells.entity_id(c)] = total / cells.NUM_POINTS


@tack.kernel
def _point_averages(cells, values, out):
    for c in cells:
        n = cells.num_points(c)
        total = values[cells.point_id(c, 0)]
        for j in range(1, n):
            total += values[cells.point_id(c, j)]
        out[cells.entity_id(c)] = total / n


def to_cells(data, field):
    """``field`` averaged over each cell's corners: a ``Constant`` field.

    For point data -- ``H1`` of order 1, or values on points -- each cell gets
    the mean of its points' values, as VTK's vtkPointDataToCellData does. Any
    field with a basis goes through its view's corner values: a DG field
    averages each cell's own, an order-2 field its values at the corners (as
    the filters read it), and a cell constant comes back as itself. Scalars or
    vectors; the values must be floating point. On polyhedra and polygons, each
    cell averages its distinct points' values.
    """
    space = field.space
    if isinstance(space, Values) and space.on == "points":
        field = Field(H1(data), field.values)
        space = field.space
    if not space.interpolated or space.on not in ("cells", "points"):
        raise TypeError(f"to_cells averages a field on the points or cells, not {space!r}")
    if field.values.dtype not in (tack.f32, tack.f64):
        raise TypeError(f"to_cells needs floating-point values, not {field.values.dtype.name}")
    out = _like(field.values, data.num_cells)
    if not getattr(data.topology, "reference_cells", True):
        if space.on != "points":
            return Field(Constant(data), _take(field.values, _all(data.num_cells)))
        for_each(_point_averages, data, "cells", materialize(field.values), out)
        return Field(Constant(data), out)
    for_each(_corner_averages, data, "cells", field, out)
    return Field(Constant(data), out)
