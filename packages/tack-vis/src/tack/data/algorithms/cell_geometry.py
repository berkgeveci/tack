"""``cell_geometry``: ``(measure, centroid)`` of each cell: fields on cells. A 3D cell's volume, by the divergence theorem over its faces -- polyhedral or of a shape -- with faces taken as fans from their points' means, so a cell with non-planar faces is measured as that polyhedron (a hexahedron with warped faces is not quite its trilinear volume); its faces must be wound out of it. A polygon's area, along its own normal (Newell's), for a polygonal topology."""

from typing import NamedTuple

import tack
from tack.data import shapes
from tack.data.algorithms._common import _vectors
from tack.data.dataset import Field
from tack.data.spaces import Values

__all__ = ['CellGeometry', 'cell_geometry']


@tack.kernel
def _cell_geometry(cells, volumes, centroids):
    for c in cells:
        # The divergence theorem over the cell's faces, each walked outward and
        # fanned from its points' mean, taken from the cell's first point so
        # cells far from the origin keep their precision.
        origin = cells.side_position(c, 0, 0)
        six = 0.0
        weighted = tack.Vector([0.0, 0.0, 0.0])
        for k in range(cells.num_faces(c)):
            size = cells.side_size(c, k)
            apex = tack.Vector([0.0, 0.0, 0.0])
            for j in range(size):
                apex += cells.side_position(c, k, j) - origin
            apex = apex / size
            for j in range(size):
                a = cells.side_position(c, k, j) - origin
                b = cells.side_position(c, k, j + 1 if j + 1 < size else 0) - origin
                t = a.dot(b.cross(apex))
                six += t
                weighted += t * (a + b + apex)
        volumes[cells.entity_id(c)] = six / 6.0
        centroids[cells.entity_id(c)] = origin + weighted / (4.0 * six)


@tack.kernel
def _polygon_geometry(cells, areas, centroids):
    for c in cells:
        # Newell's area vector over the polygon's edges, walked its way, then a fan
        # from its first point weighted by each triangle's area along the normal.
        origin = cells.side_position(c, 0, 0)
        n = tack.Vector([0.0, 0.0, 0.0])
        for k in range(cells.num_faces(c)):
            n += (cells.side_position(c, k, 0) - origin).cross(
                cells.side_position(c, k, 1) - origin)
        unit = n / n.norm()
        twice = 0.0
        weighted = tack.Vector([0.0, 0.0, 0.0])
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - origin
            b = cells.side_position(c, k, 1) - origin
            t = a.cross(b).dot(unit)
            twice += t
            weighted += t * (a + b)
        areas[cells.entity_id(c)] = 0.5 * twice
        centroids[cells.entity_id(c)] = origin + weighted / (3.0 * twice)


@tack.kernel
def _line_geometry(cells, lengths, centroids):
    for c in cells:
        # Each segment's length, and its midpoint weighted by it, from the chain's
        # first point.
        origin = cells.point_position(cells.line_point(c, 0))
        total = 0.0
        weighted = origin - origin
        for j in range(cells.line_size(c) - 1):
            a = cells.point_position(cells.line_point(c, j)) - origin
            b = cells.point_position(cells.line_point(c, j + 1)) - origin
            length = (b - a).norm()
            total += length
            weighted += length * (a + b)
        lengths[cells.entity_id(c)] = total
        centroids[cells.entity_id(c)] = origin + weighted / (2.0 * total)


class CellGeometry(NamedTuple):
    """``cell_geometry``'s fields, on the cells."""

    measure: Field
    centroid: Field


def cell_geometry(data):
    """``CellGeometry(measure, centroid)`` of each cell: fields on cells. A 3D cell's volume, by the
    divergence theorem over its faces -- polyhedral or of a shape -- with faces
    taken as fans from their points' means, so a cell with non-planar faces is
    measured as that polyhedron (a hexahedron with warped faces is not quite its
    trilinear volume); its faces must be wound out of it. A 2D cell's area --
    triangle, quad, pixel or polygon -- along its own normal (Newell's), from
    its edges going round it. A polyline's length, and the centroid of its
    segments."""
    if data.topology.dimension > 1:
        data.topology.faces()                      # the incidence the cell views need
    n = data.num_cells
    volumes = tack.field(data.dtype, shape=(n,))
    centroids = _vectors(n, data.dtype)
    for group in data.launch_groups("cells", []):
        if group.count and group.shape.DIMENSION == 3:
            _cell_geometry(data.domain_view("cells", group), volumes, centroids)
        elif group.count and group.shape.DIMENSION == 2:
            _polygon_geometry(data.domain_view("cells", group), volumes, centroids)
        elif group.count and group.shape is shapes.PolyLine:
            _line_geometry(data.domain_view("cells", group), volumes, centroids)
    return CellGeometry(Field(Values(data, "cells"), volumes), Field(Values(data, "cells"), centroids))
