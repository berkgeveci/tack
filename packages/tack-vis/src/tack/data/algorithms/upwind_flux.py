"""``upwind_flux``: the flux of ``field`` carried by a constant ``velocity`` through each face, out of side 0: ``(v . n) * area * u``, ``u`` taken from the upwind side at the face's center -- the side the flow leaves -- as a DG advection scheme does. A boundary face takes its one side's value whichever way the flow goes. ``divergence`` of the result is each cell's net outflow."""

import tack
from tack.data.algorithms._common import _field
from tack.data.algorithms._faces import _on_cells
from tack.data.algorithms.face_geometry import face_geometry
from tack.data.algorithms.traces import traces
from tack.data.arrays import materialize
from tack.data.carry import _like
from tack.data.dataset import Field, for_each
from tack.data.spaces import Values

__all__ = ['upwind_flux']


@tack.kernel
def _upwind(faces, t, normals, areas, vx, vy, vz, out):
    for f in faces:
        e = faces.entity_id(f)
        pc = faces.parametric_center()
        vn = normals[e].dot(tack.Vector([vx, vy, vz]))
        upwind = t.value(f, 0, pc)
        if vn < 0.0 and faces.num_sides(f) == 2:
            upwind = t.value(f, 1, pc)
        out[e] = vn * areas[e] * upwind


@tack.kernel
def _cell_upwind(faces, values, normals, areas, vx, vy, vz, out):
    for f in faces:
        e = faces.entity_id(f)
        vn = normals[e].dot(tack.Vector([vx, vy, vz]))
        cell = faces.side_cell(f, 0)
        if vn < 0.0 and faces.num_sides(f) == 2:
            cell = faces.side_cell(f, 1)
        out[e] = vn * areas[e] * values[cell]


def upwind_flux(data, field, velocity):
    """The flux of ``field`` carried by a constant ``velocity`` through each face, out
    of side 0: ``(v . n) * area * u``, ``u`` taken from the upwind side at the face's
    center -- the side the flow leaves -- as a DG advection scheme does. A boundary
    face takes its one side's value whichever way the flow goes. ``divergence`` of
    the result is each cell's net outflow."""
    field = _field(data, field)
    normals, areas = face_geometry(data)
    out = _like(field.values, data.topology.faces().num_faces)
    vx, vy, vz = (float(v) for v in velocity)
    if _on_cells(data, field):
        for_each(_cell_upwind, data, "faces", materialize(field.values), normals.values,
                 areas.values, vx, vy, vz, out)
    else:
        for_each(_upwind, data, "faces", traces(data, field), normals.values, areas.values,
                 vx, vy, vz, out)
    return Field(Values(data, "faces", oriented=True), out)
