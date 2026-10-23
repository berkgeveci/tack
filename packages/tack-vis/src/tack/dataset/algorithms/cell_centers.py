"""``cell_centers``: each cell's center, where the geometry maps its parametric center, or on a cell with no reference element the mean of its points: a field on cells."""

import tack
from tack.arrays import materialize
from tack.dataset.algorithms._common import _vectors
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['cell_centers']


@tack.kernel
def _centers(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = cells.position(c, cells.parametric_center())


@on_inputs_device
def cell_centers(data):
    """Each cell's center, where the geometry maps its parametric center: a field on cells.
    That is ``values_at_centers(data, data.geometry)``, and not the centroid on a
    cell that is not affine; ``cell_geometry(data).centroid`` is the centroid, on
    every topology, polyhedra included.

    A polyhedron, polygon or polyline has no parametric center: its center is its
    *cell point*, the mean of its distinct points, where its interpolation fans
    its faces from. That is not the centroid either, unless the points are
    spread evenly."""
    out = _vectors(data.num_cells, data.dtype)
    if not getattr(data.topology, "reference_cells", True):
        from tack.dataset.algorithms.project import _distinct_point_means

        for_each(_distinct_point_means, data, "cells", materialize(data.geometry.values), out)
    else:
        for_each(_centers, data, "cells", out)
    return Field(Values(data, "cells"), out)
