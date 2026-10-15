"""``values_at_centers``: ``field`` evaluated at each cell's parametric center, through its basis."""

import tack
from tack.dataset.algorithms._common import _field
from tack.dataset.carry import _like
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import Values, reference_cells

__all__ = ['values_at_centers']


@tack.kernel
def _at_centers(cells, u, out):
    for c in cells:
        out[cells.entity_id(c)] = u.value(c, cells.parametric_center())


def values_at_centers(data, field):
    """``field`` evaluated at each cell's parametric center, through its basis."""
    field = _field(data, field)
    reference_cells(data.topology, "values_at_centers")
    out = _like(field.values, data.num_cells)
    for_each(_at_centers, data, "cells", field, out)
    return Field(Values(data, "cells"), out)
