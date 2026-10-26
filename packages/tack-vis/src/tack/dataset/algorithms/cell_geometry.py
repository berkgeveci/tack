"""``cell_geometry``: a ``CellGeometry(measure, centroid)`` of each cell: fields on cells. A 3D cell's volume, by the divergence theorem over its faces -- polyhedral or of a shape -- with faces taken as fans from their points' means, so a cell with non-planar faces is measured as that polyhedron (a hexahedron with warped faces is not quite its trilinear volume); its faces must be wound out of it. A polygon's area, along its own normal (Newell's), for a polygonal topology."""

from typing import NamedTuple

import tack
from tack.dataset import shapes
from tack.dataset.algorithms._common import _measure_unit, _table, _vectors
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['CellGeometry', 'cell_geometry']


@tack.kernel
def _cell_geometry(cells, volumes, centroids):
    for c in cells:
        # The divergence theorem over the cell's faces, each walked outward and
        # fanned from its points' mean, taken from the cell's first point so
        # cells far from the origin keep their precision.
        origin = cells.side_position(c, 0, 0)
        six = 0.0
        weighted = tack.vec3(0.0, 0.0, 0.0)
        apexes = tack.vec3(0.0, 0.0, 0.0)
        for k in range(cells.num_faces(c)):
            size = cells.side_num_points(c, k)
            apex = tack.vec3(0.0, 0.0, 0.0)
            for j in range(size):
                apex += cells.side_position(c, k, j) - origin
            apex = apex / size
            apexes += apex
            for j in range(size):
                a = cells.side_position(c, k, j) - origin
                b = cells.side_position(c, k, j + 1 if j + 1 < size else 0) - origin
                t = a.dot(b.cross(apex))
                six += t
                weighted += t * (a + b + apex)
        volumes[cells.entity_id(c)] = six / 6.0
        if six != 0.0:
            centroids[cells.entity_id(c)] = origin + weighted / (4.0 * six)
        else:
            # No volume to weight by: the mean of its faces' centers of points.
            centroids[cells.entity_id(c)] = origin + apexes / cells.num_faces(c)


@tack.kernel
def _polygon_geometry(cells, areas, centroids):
    for c in cells:
        # Newell's area vector over the polygon's edges, walked its way, then a fan
        # from its first point weighted by each triangle's area along the normal.
        origin = cells.side_position(c, 0, 0)
        n = tack.vec3(0.0, 0.0, 0.0)
        for k in range(cells.num_faces(c)):
            n += (cells.side_position(c, k, 0) - origin).cross(
                cells.side_position(c, k, 1) - origin)
        length = n.norm()
        unit = n / length if length > 0.0 else n
        twice = 0.0
        weighted = tack.vec3(0.0, 0.0, 0.0)
        corners = tack.vec3(0.0, 0.0, 0.0)
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - origin
            b = cells.side_position(c, k, 1) - origin
            t = a.cross(b).dot(unit)
            twice += t
            weighted += t * (a + b)
            corners += a
        areas[cells.entity_id(c)] = 0.5 * twice
        if twice != 0.0:
            centroids[cells.entity_id(c)] = origin + weighted / (3.0 * twice)
        else:
            # No area to weight by: the mean of its corners.
            centroids[cells.entity_id(c)] = origin + corners / cells.num_faces(c)


#: A 2D shape's corners in order round it.
_ROUND = {shapes.Triangle: (0, 1, 2), shapes.Quad: (0, 1, 2, 3), shapes.Pixel: (0, 1, 3, 2)}


@tack.kernel
def _corner_polygon_geometry(cells, order, areas, centroids):
    # As _polygon_geometry, round the shape's corners: a triangle, quad or pixel
    # among cells of a higher dimension, whose views have no sides of it.
    for c in cells:
        size = order.shape[0]
        origin = cells.corner_position(c, order[0])
        n = tack.vec3(0.0, 0.0, 0.0)
        for k in range(size):
            n += (cells.corner_position(c, order[k]) - origin).cross(
                cells.corner_position(c, order[k + 1 if k + 1 < size else 0]) - origin)
        length = n.norm()
        unit = n / length if length > 0.0 else n
        twice = 0.0
        weighted = tack.vec3(0.0, 0.0, 0.0)
        corners = tack.vec3(0.0, 0.0, 0.0)
        for k in range(size):
            a = cells.corner_position(c, order[k]) - origin
            b = cells.corner_position(c, order[k + 1 if k + 1 < size else 0]) - origin
            t = a.cross(b).dot(unit)
            twice += t
            weighted += t * (a + b)
            corners += a
        areas[cells.entity_id(c)] = 0.5 * twice
        if twice != 0.0:
            centroids[cells.entity_id(c)] = origin + weighted / (3.0 * twice)
        else:
            centroids[cells.entity_id(c)] = origin + corners / size


@tack.kernel
def _line_geometry(cells, lengths, centroids):
    for c in cells:
        # Each segment's length, and its midpoint weighted by it, from the chain's
        # first point.
        origin = cells.point_position(cells.line_point_id(c, 0))
        total = 0.0
        weighted = origin - origin
        points = origin - origin
        for j in range(cells.line_num_points(c) - 1):
            a = cells.point_position(cells.line_point_id(c, j)) - origin
            b = cells.point_position(cells.line_point_id(c, j + 1)) - origin
            length = (b - a).norm()
            total += length
            weighted += length * (a + b)
            points += b
        lengths[cells.entity_id(c)] = total
        if total != 0.0:
            centroids[cells.entity_id(c)] = origin + weighted / (2.0 * total)
        else:
            # No length to weight by: the mean of its points.
            centroids[cells.entity_id(c)] = origin + points / cells.line_num_points(c)


@tack.kernel
def _corner_geometry(cells, lengths, centroids):
    # A line's length and midpoint; a vertex's point, measuring nothing.
    for c in cells:
        a = cells.corner_position(c, 0)
        b = cells.corner_position(c, cells.NUM_POINTS - 1)
        lengths[cells.entity_id(c)] = (b - a).norm()
        centroids[cells.entity_id(c)] = 0.5 * (a + b)


class CellGeometry(NamedTuple):
    """``cell_geometry``'s fields, on the cells."""

    measure: Field
    centroid: Field


@on_inputs_device
def cell_geometry(data):
    """``CellGeometry(measure, centroid)`` of each cell: fields on cells. A 3D cell's volume, by the
    divergence theorem over its faces -- polyhedral or of a shape -- with faces
    taken as fans from their points' means, so a cell with non-planar faces is
    measured as that polyhedron (a hexahedron with warped faces is not quite its
    trilinear volume); its faces must be wound out of it. A 2D cell's area --
    triangle, quad, pixel or polygon -- along its own normal (Newell's), from
    its edges going round it. A line's or a polyline's length, and the
    centroid of its segments; a vertex's point, with measure 0."""
    if data.topology.dimension > 1 and not getattr(data.topology, "reference_cells", True):
        data.topology.faces()                      # polyhedra and polygons walk theirs
    n = data.num_cells
    volumes = tack.empty((n,), data.dtype)
    centroids = _vectors(n, data.dtype)
    for group in data.launch_groups("cells", []):
        if group.count and group.shape.DIMENSION == 3:
            _cell_geometry(data.domain_view("cells", group), volumes, centroids)
        elif group.count and group.shape.DIMENSION < data.topology.dimension and \
                group.shape in _ROUND:
            _corner_polygon_geometry(data.domain_view("cells", group),
                                     _table(("round", group.shape), _ROUND[group.shape]),
                                     volumes, centroids)
        elif group.count and group.shape.DIMENSION == 2:
            _polygon_geometry(data.domain_view("cells", group), volumes, centroids)
        elif group.count and group.shape is shapes.PolyLine:
            _line_geometry(data.domain_view("cells", group), volumes, centroids)
        elif group.count:
            _corner_geometry(data.domain_view("cells", group), volumes, centroids)
    return CellGeometry(Field(Values(data, "cells"), volumes, _measure_unit(data)),
                        Field(Values(data, "cells"), centroids, data.unit))
