"""``face_centers``: each face's area centroid: a field on faces. The face is fanned from its points' mean, so both its cells -- which store it once -- see the same fan."""

import tack
from tack.dataset.algorithms._common import _vectors
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['face_centers']


@tack.kernel
def _face_centers(faces, out):
    for f in faces:
        size = faces.num_points(f)
        origin = faces.corner_position(f, 0)
        apex = tack.vec3(0.0, 0.0, 0.0)
        for j in range(size):
            apex += faces.corner_position(f, j) - origin
        apex = apex / size
        total = 0.0
        weighted = tack.vec3(0.0, 0.0, 0.0)
        for j in range(size):
            a = faces.corner_position(f, j) - origin
            b = faces.corner_position(f, j + 1 if j + 1 < size else 0) - origin
            area = (a - apex).cross(b - apex).norm()
            total += area
            weighted += area * (a + b + apex)
        if size < 3 or total == 0.0:
            # An edge, a 2D mesh's face, or a polyline's end -- or a face of no
            # area: its points' mean.
            out[faces.entity_id(f)] = origin + apex
        else:
            out[faces.entity_id(f)] = origin + weighted / (3.0 * total)


@on_inputs_device
def face_centers(data):
    """Each face's area centroid: a field on faces. The face is fanned from its
    points' mean, so both its cells -- which store it once -- see the same fan. A
    2D mesh's face, an edge, has its midpoint, and a polyline's end its point."""
    out = _vectors(data.topology.faces().num_faces, data.dtype)
    for_each(_face_centers, data, "faces", out)
    return Field(Values(data, "faces"), out, data.unit)
