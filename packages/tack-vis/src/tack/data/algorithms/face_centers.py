"""``face_centers``: each face's area centroid: a field on faces. The face is fanned from its points' mean, so both its cells -- which store it once -- see the same fan."""

import tack
from tack.data.algorithms._common import _vectors
from tack.data.dataset import Field, for_each
from tack.data.spaces import Values

__all__ = ['face_centers']


@tack.kernel
def _face_centers(faces, out):
    for f in faces:
        size = faces.face_size(f)
        origin = faces.point(f, 0)
        apex = tack.Vector([0.0, 0.0, 0.0])
        for j in range(size):
            apex += faces.point(f, j) - origin
        apex = apex / size
        total = 0.0
        weighted = tack.Vector([0.0, 0.0, 0.0])
        for j in range(size):
            a = faces.point(f, j) - origin
            b = faces.point(f, j + 1 if j + 1 < size else 0) - origin
            area = (a - apex).cross(b - apex).norm()
            total += area
            weighted += area * (a + b + apex)
        out[faces.entity_id(f)] = origin + weighted / (3.0 * total)


def face_centers(data):
    """Each face's area centroid: a field on faces. The face is fanned from its
    points' mean, so both its cells -- which store it once -- see the same fan."""
    out = _vectors(data.topology.faces().num_faces, data.dtype)
    for_each(_face_centers, data, "faces", out)
    return Field(Values(data, "faces"), out)
