"""``net_flux``: each cell's outward sum of ``flux``, values on faces measured out of side 0 (``Values(data, "faces", oriented=True)``, or plain face values taken so): a face's value counts plus for its side-0 cell and minus for its side-1 cell."""

import tack
from tack.arrays import materialize
from tack.dataset.algorithms._common import _field
from tack.dataset.algorithms._faces import _on_faces
from tack.dataset.carry import _like
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['net_flux']


@tack.kernel
def _outward_sums(cells, flux, out):
    for c in cells:
        total = flux[0] * 0.0
        for f in range(cells.num_faces(c)):     # the shape's count, or a polyhedron's
            sign = 1.0 if cells.face_side(c, f) == 0 else -1.0
            total += sign * flux[cells.face_id(c, f)]
        out[cells.entity_id(c)] = total


@on_inputs_device
def net_flux(data, flux):
    """Each cell's outward sum of ``flux``, values on faces measured out of side 0
    (``Values(data, "faces", oriented=True)``, or plain face values taken so): a
    face's value counts plus for its side-0 cell and minus for its side-1 cell.

    Given each face's integrated flux (``upwind_flux``, or a normal component times
    the area), it is the net outflow of each cell, ``V`` times the divergence;
    ``divergence`` divides by the cells' measure. Nothing is weighted: the
    values are summed as they are, so face area vectors sum to zero on a closed
    cell. Cells of a lower dimension than the mesh's have no faces: zero."""
    flux = _field(data, flux)
    if not _on_faces(data, flux):
        raise TypeError("net_flux sums a field on faces")
    data.topology.faces()                          # the incidence the cell views need
    out = _like(flux.values, data.num_cells)
    out.fill(0)
    # Cells of a lower dimension (a surface's triangles among tetrahedra) have
    # no faces of the mesh's, so no flux: zero. Nor has a mesh of vertices any.
    values = materialize(flux.values)
    for group in data.launch_groups("cells", []):
        if group.count and group.shape.DIMENSION == data.topology.dimension > 0:
            _outward_sums(data.domain_view("cells", group), values, out)
    return Field(Values(data, "cells"), out, flux.unit)
