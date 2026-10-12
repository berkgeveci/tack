"""``face_geometry``: ``(normal, area)``: fields on faces, of any size of polygon. A face's normal points out of its side 0; on a non-planar face it is the mean normal."""

import tack
from tack.data.algorithms._common import _vectors
from tack.data.dataset import Field, for_each
from tack.data.spaces import Values

__all__ = ['face_geometry']


@tack.kernel
def _face_geometry(faces, normals, areas):
    for f in faces:
        size = faces.face_size(f)                # the shape's constant, or a polygon's
        origin = faces.point(f, 0)
        n = tack.Vector([0.0, 0.0, 0.0])
        for j in range(1, size - 1):
            # Newell's normal, from the face's first point: a fan of triangles.
            n += (faces.point(f, j) - origin).cross(faces.point(f, j + 1) - origin)
        length = n.norm()
        normals[faces.entity_id(f)] = n / length
        areas[faces.entity_id(f)] = 0.5 * length


def face_geometry(data):
    """``(normal, area)``: fields on faces, of any size of polygon. A face's normal
    points out of its side 0; on a non-planar face it is the mean normal."""
    faces = data.topology.faces()
    dtype = data.dtype
    normals = _vectors(faces.num_faces, dtype)
    areas = tack.field(dtype, shape=(faces.num_faces,))
    for_each(_face_geometry, data, "faces", normals, areas)
    return Field(Values(data, "faces"), normals), Field(Values(data, "faces"), areas)
