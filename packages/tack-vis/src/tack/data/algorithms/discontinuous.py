"""``discontinuous``: ``field`` (``H1`` or ``Constant``) as an ``L2`` field: each cell's value at each of its corners, stored per cell. The same function, in the DG layout; edit the values and cells disagree where they meet."""

import tack
from tack.data.algorithms._common import _field
from tack.data.carry import _like
from tack.data.dataset import Field, for_each
from tack.data.spaces import L2

__all__ = ['discontinuous']


@tack.kernel
def _corner_values(cells, u, offsets, out):
    for c in cells:
        first = offsets[cells.entity_id(c)]
        for j in range(cells.NUM_POINTS):
            out[first + j] = u.value(c, cells.parametric_point(j))


def discontinuous(data, field):
    """``field`` (``H1`` or ``Constant``) as an ``L2`` field: each cell's value at each
    of its corners, stored per cell. The same function, in the DG layout; edit the
    values and cells disagree where they meet."""
    field = _field(data, field)
    space = L2(data)
    out = _like(field.values, space.size)
    for_each(_corner_values, data, "cells", field, space.offsets, out)
    return Field(space, out)
