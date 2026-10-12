"""``face_geometry``: ``(normal, area)``: fields on faces, of any size of polygon. A face's normal points out of its side 0; on a non-planar face it is the mean normal."""

from typing import NamedTuple

import tack
from tack.data.algorithms._common import _vectors
from tack.data.dataset import Field, for_each
from tack.data.spaces import Values

__all__ = ['FaceGeometry', 'face_geometry']


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


@tack.kernel
def _edge_geometry(cells, normals, areas):
    # A 2D cell's faces are its edges: each one's outward normal is its direction
    # crossed with the cell's normal (Newell's, from its edges going round it),
    # its "area" its length. Side 0's cell measures it.
    for c in cells:
        origin = cells.side_position(c, 0, 0)
        n = origin - origin
        for k in range(cells.num_faces(c)):
            n += (cells.side_position(c, k, 0) - origin).cross(cells.side_position(c, k, 1)
                                                                - origin)
        for k in range(cells.num_faces(c)):
            if cells.face_side(c, k) == 0:
                d = cells.side_position(c, k, 1) - cells.side_position(c, k, 0)
                out = d.cross(n)
                f = cells.face_id(c, k)
                normals[f] = out / out.norm()
                areas[f] = d.norm()


class FaceGeometry(NamedTuple):
    """``face_geometry``'s fields, on the faces."""

    normal: Field
    area: Field


def face_geometry(data):
    """``FaceGeometry(normal, area)``: fields on faces. A face's normal points out of
    its side 0. A 3D mesh's faces are polygons of any size, the normal the mean
    one on a non-planar face; a 2D mesh's faces are edges, the normal in the
    plane of side 0's cell and the "area" the edge's length."""
    faces = data.topology.faces()
    dtype = data.dtype
    normals = _vectors(faces.num_faces, dtype)
    areas = tack.field(dtype, shape=(faces.num_faces,))
    if data.topology.dimension == 2:
        for_each(_edge_geometry, data, "cells", normals, areas)
    else:
        for_each(_face_geometry, data, "faces", normals, areas)
    return FaceGeometry(Field(Values(data, "faces"), normals), Field(Values(data, "faces"), areas))
