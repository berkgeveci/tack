"""``cell_centers``: each cell's center, where the geometry maps its parametric center: a field on cells."""

import tack
from tack.dataset.algorithms._common import _vectors
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import Values, reference_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['cell_centers']


@tack.kernel
def _centers(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = cells.position(c, cells.parametric_center())


@on_inputs_device
def cell_centers(data):
    """Each cell's center, where the geometry maps its parametric center: a field on cells."""
    reference_cells(data.topology, "cell_centers")
    out = _vectors(data.num_cells, data.dtype)
    for_each(_centers, data, "cells", out)
    return Field(Values(data, "cells"), out)
