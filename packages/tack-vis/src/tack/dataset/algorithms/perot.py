"""``perot``: a vector per cell from each face's normal component -- Perot's reconstruction, ``u_c = (1/V) sum_f A_f u_f (r_f - r_c)``, as C-grid models (MPAS, ICON) need to show their face-normal velocity at cells."""

import tack
from tack.dataset.algorithms._common import _field, _vectors
from tack.dataset.algorithms._faces import _on_faces
from tack.dataset.algorithms.cell_geometry import cell_geometry
from tack.dataset.algorithms.face_centers import face_centers
from tack.dataset.algorithms.face_geometry import face_geometry
from tack.dataset.arrays import materialize
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values

__all__ = ['perot']


@tack.kernel
def _perot(cells, flux, areas, centers, centroids, volumes, out):
    for c in cells:
        e = cells.entity_id(c)
        r = centroids[e]
        u = tack.Vector([0.0, 0.0, 0.0])
        for k in range(cells.num_faces(c)):
            f = cells.face_id(c, k)
            outward = 1.0 - 2.0 * cells.face_side(c, k)
            u += (outward * areas[f] * flux[f]) * (centers[f] - r)
        out[e] = u / volumes[e]


def perot(data, flux):
    """A vector per cell from each face's normal component -- Perot's reconstruction,
    ``u_c = (1/V) sum_f A_f u_f (r_f - r_c)``, as C-grid models (MPAS, ICON) need to
    show their face-normal velocity at cells.

    ``flux`` is a scalar field on faces, the component along each face's normal out
    of its side 0; the side each cell is on gives the sign. Face centers, areas
    and cell centroids and volumes are this module's, so a uniform field's normal
    components give it back exactly on cells with planar faces. First order, as
    Perot's method is. A cell of no volume gives NaN."""
    flux = _field(data, flux)
    if not _on_faces(data, flux):
        raise TypeError("perot reconstructs from a field on faces")
    _, areas = face_geometry(data)
    centers = face_centers(data)
    volumes, centroids = cell_geometry(data)
    out = _vectors(data.num_cells, data.dtype)
    for group in data.launch_groups("cells", []):
        if group.count and group.shape.DIMENSION == data.topology.dimension:
            _perot(data.domain_view("cells", group), materialize(flux.values), areas.values,
                   centers.values, centroids.values, volumes.values, out)
    return Field(Values(data, "cells"), out)
