"""``face_geometry``: ``(normal, area)``: fields on faces, of any size of polygon. A face's normal points out of its side 0; on a non-planar face it is the mean normal."""

from typing import NamedTuple

import tack
from tack.dataset.algorithms._common import _vectors
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['FaceGeometry', 'face_geometry']


@tack.kernel
def _face_geometry(faces, normals, areas):
    for f in faces:
        size = faces.num_points(f)                # the shape's constant, or a polygon's
        origin = faces.corner_position(f, 0)
        n = tack.vec3(0.0, 0.0, 0.0)
        for j in range(1, size - 1):
            # Newell's normal, from the face's first point: a fan of triangles.
            n += (faces.corner_position(f, j) - origin).cross(faces.corner_position(f, j + 1) - origin)
        length = n.norm()
        normals[faces.entity_id(f)] = n / length if length > 0.0 else n   # none: no area
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
                length = out.norm()
                f = cells.face_id(c, k)
                normals[f] = out / length if length > 0.0 else out
                areas[f] = d.norm()


@tack.kernel
def _end_geometry(cells, normals, areas):
    # A polyline's facets are its ends: each one's normal is the chain's direction
    # there, pointing out of it, and its "area" 1. Side 0's cell measures it.
    for c in cells:
        last = cells.line_num_points(c) - 1
        for k in range(2):
            if cells.face_side(c, k) == 0:
                at = 0 if k == 0 else last
                inside = 1 if k == 0 else last - 1
                d = (cells.point_position(cells.line_point_id(c, at))
                     - cells.point_position(cells.line_point_id(c, inside)))
                length = d.norm()
                f = cells.face_id(c, k)
                normals[f] = d / length if length > 0.0 else d
                areas[f] = 1.0


class FaceGeometry(NamedTuple):
    """``face_geometry``'s fields, on the faces."""

    normal: Field
    area: Field


@on_inputs_device
def face_geometry(data):
    """``FaceGeometry(normal, area)``: fields on faces. A face's normal points out of
    its side 0. A 3D mesh's faces are polygons of any size, the normal the mean
    one on a non-planar face; a 2D mesh's faces are edges, the normal in the
    plane of side 0's cell and the "area" the edge's length; a polyline mesh's
    faces are the polylines' ends, the normal along the line out of side 0 and
    the "area" 1."""
    faces = data.topology.faces()
    dtype = data.dtype
    normals = _vectors(faces.num_faces, dtype)
    areas = tack.empty((faces.num_faces,), dtype)
    if data.topology.dimension == 2:
        for_each(_edge_geometry, data, "cells", normals, areas)
    elif data.topology.dimension == 1:
        for_each(_end_geometry, data, "cells", normals, areas)
    else:
        for_each(_face_geometry, data, "faces", normals, areas)
    return FaceGeometry(Field(Values(data, "faces"), normals), Field(Values(data, "faces"), areas))
