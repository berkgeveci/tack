"""``divergence``: each cell's outward sum of ``flux``, values on faces measured out of side 0 (``Values(data, "faces", oriented=True)``, or plain face values taken so): a face's value counts plus for its side-0 cell and minus for its side-1 cell."""

import tack
from tack.data.algorithms._faces import _on_faces
from tack.data.arrays import materialize
from tack.data.carry import _like
from tack.data.dataset import Field, for_each
from tack.data.spaces import Values

__all__ = ['divergence']


@tack.kernel
def _outward_sums(cells, flux, out):
    for c in cells:
        total = flux[0] * 0.0
        for f in range(cells.num_faces(c)):     # the shape's count, or a polyhedron's
            sign = 1.0 if cells.face_side(c, f) == 0 else -1.0
            total += sign * flux[cells.face_id(c, f)]
        out[cells.entity_id(c)] = total


def divergence(data, flux):
    """Each cell's outward sum of ``flux``, values on faces measured out of side 0
    (``Values(data, "faces", oriented=True)``, or plain face values taken so): a
    face's value counts plus for its side-0 cell and minus for its side-1 cell."""
    if not _on_faces(data, flux):
        raise TypeError("divergence sums a field on faces")
    data.topology.faces()                          # the incidence the cell views need
    out = _like(flux.values, data.num_cells)
    for_each(_outward_sums, data, "cells", materialize(flux.values), out)
    return Field(Values(data, "cells"), out)
