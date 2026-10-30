"""``integrate``: the measure of a dataset and the integrals of its fields over it, by
quadrature in each cell's reference element through its own map -- curved cells and
fields of any order exactly, to rounding (VTK's vtkIntegrateAttributes)."""

import math
from typing import NamedTuple

import numpy as np

import tack
from tack import arrays
from tack.dataset import shapes, units
from tack.dataset.algorithms._common import _field, _with_basis
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['Integrals', 'integrate']


class Integrals(NamedTuple):
    """``integrate``'s answer: ``measure`` (the total volume, area or length, in
    ``measure_unit``) and, per field, its integral (a float, or a tuple for a vector
    field) in ``values`` and that integral's unit in ``units``."""

    measure: float
    measure_unit: object
    values: dict
    units: dict


@tack.func
def _element(m, dimension):
    # The measure element of a cell of ``dimension`` mapped by Jacobian m: a volume's
    # |det J|, a surface's |J0 x J1|, a line's |J0| (columns past the cell's
    # dimension are zero).
    out = abs(m.determinant())
    if dimension == 2:
        a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
        b = tack.vec3(m[0, 1], m[1, 1], m[2, 1])
        out = a.cross(b).norm()
    elif dimension == 1:
        out = tack.vec3(m[0, 0], m[1, 0], m[2, 0]).norm()
    return out


@tack.kernel
def _cell_measures(cells, points, weights, measures, centroids):
    # Each cell's measure and centroid: the quadrature's weights times the measure
    # element, and the positions those weigh.
    for c in cells:
        e = cells.entity_id(c)
        m = weights[0] * 0.0
        x = cells.position(c, points[0]) * 0.0
        for q in range(weights.shape[0]):
            d = weights[q] * _element(cells.geometry_jacobian(c, points[q]), cells.DIMENSION)
            m += d
            x += d * cells.position(c, points[q])
        measures[e] = m
        centroids[e] = x / m if m > 0.0 else cells.position(c, cells.parametric_center())


@tack.kernel
def _cell_integrals(cells, u, points, weights, out):
    # Each cell's integral of the field: the quadrature's weights times the measure
    # element times the field's value, by its own basis.
    for c in cells:
        total = u.value(c, points[0]) * 0.0
        for q in range(weights.shape[0]):
            d = weights[q] * _element(cells.geometry_jacobian(c, points[q]), cells.DIMENSION)
            total += d * u.value(c, points[q])
        out[cells.entity_id(c)] = total


@tack.kernel
def _drop_cells(cells, values):
    # Cells of a lower dimension than the mesh's: no part of its measure.
    for c in cells:
        values[cells.entity_id(c)] = 0.0


def _top_cells(data, values):
    """``values`` (one per cell) with the cells below the mesh's dimension zeroed."""
    top = data.topology.dimension
    for group in data.topology.groups():
        if group.count and group.shape.DIMENSION < top:
            _drop_cells(data.domain_view("cells", group), values)
    return values


@tack.kernel
def _polyhedron_integrals(cells, u, out):
    # A point field is linear on a polyhedron's sub-tetrahedra (each face fanned
    # from its points' mean, the fans from the cell's points' mean, values too):
    # each one's volume times the mean of its corners' values. Positions from the
    # cell's first point, as cell_geometry takes them.
    for c in cells:
        origin = cells.point_position(cells.point_id(c, 0))
        xc = origin - origin
        uc = u[cells.point_id(c, 0)] * 0.0
        for j in range(cells.num_points(c)):
            xc += cells.point_position(cells.point_id(c, j)) - origin
            uc += u[cells.point_id(c, j)]
        xc = xc / cells.num_points(c)
        uc = uc / cells.num_points(c)
        total = uc * 0.0
        for k in range(cells.num_faces(c)):
            size = cells.side_num_points(c, k)
            xf = origin - origin
            uf = uc * 0.0
            for j in range(size):
                xf += cells.side_position(c, k, j) - origin
                uf += u[cells.side_point_id(c, k, j)]
            xf = xf / size
            uf = uf / size
            for j in range(size):
                nxt = j + 1 if j + 1 < size else 0
                a = cells.side_position(c, k, j) - origin - xc
                b = cells.side_position(c, k, nxt) - origin - xc
                six = a.dot(b.cross(xf - xc))
                total += six * (uc + uf + u[cells.side_point_id(c, k, j)]
                                + u[cells.side_point_id(c, k, nxt)])
        out[cells.entity_id(c)] = total / 24.0


@tack.kernel
def _polygon_integrals(cells, u, out):
    # Linear on the triangles fanning a polygon from its points' mean: each one's
    # area along the polygon's normal (Newell's) times its corners' mean value.
    for c in cells:
        origin = cells.side_position(c, 0, 0)
        n = origin - origin
        for k in range(cells.num_faces(c)):
            n += (cells.side_position(c, k, 0) - origin).cross(
                cells.side_position(c, k, 1) - origin)
        length = n.norm()
        unit = n / length if length > 0.0 else n
        xc = origin - origin
        uc = u[cells.point_id(c, 0)] * 0.0
        for j in range(cells.num_points(c)):
            xc += cells.point_position(cells.point_id(c, j)) - origin
            uc += u[cells.point_id(c, j)]
        xc = xc / cells.num_points(c)
        uc = uc / cells.num_points(c)
        total = uc * 0.0
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - origin - xc
            b = cells.side_position(c, k, 1) - origin - xc
            total += a.cross(b).dot(unit) * (uc + u[cells.side_point_id(c, k, 0)]
                                             + u[cells.side_point_id(c, k, 1)])
        out[cells.entity_id(c)] = total / 6.0


@tack.kernel
def _polyline_integrals(cells, u, out):
    # Linear on each segment: its length times its ends' mean value.
    for c in cells:
        total = u[cells.line_point_id(c, 0)] * 0.0
        for j in range(cells.line_num_points(c) - 1):
            p, q = cells.line_point_id(c, j), cells.line_point_id(c, j + 1)
            total += (cells.point_position(q) - cells.point_position(p)).norm() * (u[p] + u[q])
        out[cells.entity_id(c)] = total / 2.0


def quadrature(shape, n):
    """``(points, weights)``: ``n``-point Gauss-Legendre rules along each axis of
    ``shape``'s reference element, collapsed onto a simplex (Duffy: the triangle
    ``(u, v (1 - u))`` weighted by ``1 - u``, the tetrahedron likewise) and a wedge
    (the triangle's rule times the segment's). A pyramid's reference element is the
    unit cube -- its map collapses the top itself -- so it takes the cube's.
    ``points`` are ``(m, 3)``, the weights sum to the element's measure, and a
    polynomial of degree ``2 n - 1`` along each axis of the cube (fewer collapsed)
    integrates exactly."""
    d = shape.DIMENSION
    if d == 0:
        return np.zeros((1, 3)), np.ones(1)     # a vertex: its point, measure 0
    x, w = np.polynomial.legendre.leggauss(int(n))
    x, w = (x + 1) / 2, w / 2
    if shape in (shapes.Triangle, shapes.Tetra, shapes.Wedge):
        u, v = np.meshgrid(x, x, indexing="ij")
        wu, wv = np.meshgrid(w, w, indexing="ij")
        if shape is shapes.Tetra:
            u, v, t = np.meshgrid(x, x, x, indexing="ij")
            wu, wv, wt = np.meshgrid(w, w, w, indexing="ij")
            points = np.stack([u, v * (1 - u), t * (1 - u) * (1 - v)], -1).reshape(-1, 3)
            return points, (wu * wv * wt * (1 - u) ** 2 * (1 - v)).reshape(-1)
        tri = np.stack([u, v * (1 - u)], -1).reshape(-1, 2)
        tri_w = (wu * wv * (1 - u)).reshape(-1)
        if shape is shapes.Triangle:
            return np.c_[tri, np.zeros(len(tri))], tri_w
        points = np.array([[a, b, z] for a, b in tri for z in x])
        return points, np.outer(tri_w, w).reshape(-1)
    grids = np.meshgrid(*([x] * d), indexing="ij")
    weights = np.prod(np.meshgrid(*([w] * d), indexing="ij"), axis=0).reshape(-1)
    points = np.stack([g.reshape(-1) for g in grids], -1)
    return np.c_[points, np.zeros((len(points), 3 - d))], weights


def _order(space):
    """The polynomial order a space's values carry along an axis."""
    if isinstance(space, (Constant, Values)):
        return 0 if isinstance(space, Constant) or space.on == "cells" else 1
    return max(int(getattr(space, "order", 1)), 0)


def _tables(data, shape, n):
    """The rule of ``n`` points an axis for ``shape`` as device arrays in the
    geometry's float type."""
    points, weights = quadrature(shape, n)
    dtype = data.dtype
    p = tack.empty((len(points),), tack.vec(3, dtype))
    tack.copyto(p, points.astype(dtype.numpy_dtype))
    w = tack.empty((len(weights),), dtype)
    tack.copyto(w, weights.astype(dtype.numpy_dtype))
    return p, w


def _points_per_axis(data, field_order):
    """Enough Gauss points an axis for a field of ``field_order`` on ``data``'s
    geometry: the integrand's degree along an axis is at most the field's plus the
    measure element's (the dimension times the geometry's order), and a collapsed
    rule loses one more an axis."""
    g = _order(data.geometry.space)
    dimension = data.topology.dimension
    return max(1, (field_order + dimension * g + dimension) // 2 + 1)


def cell_measures(data, n=None):
    """``(measures, centroids)`` of every cell with a reference element, by quadrature
    (``n`` points an axis, by default enough for the geometry): the curved cells'
    ``cell_geometry``."""
    vec = tack.vec(3, data.dtype)
    measures = tack.empty((data.num_cells,), data.dtype)
    centroids = tack.empty((data.num_cells,), vec)
    n = _points_per_axis(data, 0) if n is None else n
    for group in data.launch_groups("cells", []):
        if group.count:
            _cell_measures(data.domain_view("cells", group), *_tables(data, group.shape, n),
                           measures, centroids)
    return measures, centroids


def _cell_values(data, field):
    """Each cell's integral of ``field`` (a field with a basis), by quadrature."""
    field = _with_basis(data, field)
    width = arrays.width_of(field.values)
    dtype = arrays.dtype_of(field.values)
    out = tack.zeros((data.num_cells,), tack.vec(width, dtype) if width else dtype)
    n = _points_per_axis(data, _order(field.space))
    top = data.topology.dimension
    for group in data.launch_groups("cells", [field]):
        if group.count and group.shape.DIMENSION == top:
            _cell_integrals(data.domain_view("cells", group), field.view(group),
                            *_tables(data, group.shape, n), out)
    return out


def _on_points(space):
    """Whether ``space`` holds one value a point: linear on polyhedra's pieces."""
    return ((isinstance(space, H1) and space.order == 1 and space.dofs is None
             and space.basis is None)
            or (isinstance(space, Values) and space.on == "points"))


def _no_reference_values(data, field):
    """Each polyhedron's, polygon's or polyline's integral of a point field, exact
    on the pieces it is linear on."""
    values = arrays.materialize(field.values)
    width = arrays.width_of(values)
    dtype = arrays.dtype_of(values)
    out = tack.zeros((data.num_cells,), tack.vec(width, dtype) if width else dtype)
    data.topology.faces()
    for group in data.launch_groups("cells", []):
        if not group.count:
            continue
        cells = data.domain_view("cells", group)
        if group.shape.DIMENSION == 3:
            _polyhedron_integrals(cells, values, out)
        elif group.shape.DIMENSION == 2:
            _polygon_integrals(cells, values, out)
        elif group.shape is shapes.PolyLine:
            _polyline_integrals(cells, values, out)
    return out


def _piece(data, names):
    """One dataset's owned cells' measure and integrals, summed in cell order (f64)."""
    from tack.dataset.algorithms.cell_geometry import cell_geometry
    from tack.dataset.ghosts import owned_values, visible

    data = visible(data)
    measures = Field(Values(data, "cells"), _top_cells(data, arrays.materialize(
        cell_geometry(data).measure.values)))
    measure = float(np.sum(np.asarray(owned_values(data, measures).numpy(), np.float64)))
    values = {}
    for name in names:
        field = data.fields[name]
        if not getattr(data.topology, "reference_cells", True):
            space = _with_basis(data, field).space
            if _on_points(space):
                per_cell = _top_cells(data, _no_reference_values(data, field))
                values[name] = np.asarray(owned_values(data, Field(Values(data, "cells"),
                                                                   per_cell)).numpy(),
                                          np.float64).sum(axis=0)
                continue
            if not isinstance(space, Constant):
                raise NotImplementedError(
                    f"integrating {space!r} needs a reference element (polygons and "
                    "polyhedra integrate cell data and point data)")
            # Constant in each cell: its value times the cell's measure.
            cell_values = owned_values(data, Field(Values(data, "cells"),
                                                   arrays.materialize(field.values))).numpy()
            owned = np.asarray(owned_values(data, measures).numpy(), np.float64)
            cell_values = np.asarray(cell_values, np.float64)
            values[name] = (cell_values * (owned if cell_values.ndim == 1
                                           else owned[:, None])).sum(axis=0)
            continue
        per_cell = _cell_values(data, field)
        values[name] = np.asarray(owned_values(data, Field(Values(data, "cells"), per_cell))
                                  .numpy(), np.float64).sum(axis=0)
    return measure, values


@on_inputs_device
def integrate(data, fields=None):
    """The measure of ``data`` -- its cells' total volume, area or length -- and the
    integral over it of each of ``fields`` (names; by default every field with a
    basis or on the cells), as VTK's ``vtkIntegrateAttributes`` gives them:
    ``Integrals(measure, measure_unit, values, units)``, ``values[name]`` a float or
    a tuple for a vector field.

    Each cell is integrated by Gauss quadrature in its reference element through
    its own map (``quadrature``: collapsed onto simplices and wedges), with enough
    points for the field's order and the geometry's, so a curved geometry and a
    field of any order (``H1``, ``L2``, point and cell data) integrate exactly, to
    rounding. Polyhedra, polygons and polylines take their measure from
    ``cell_geometry`` and integrate cell data and point data, which is linear on
    their pieces (sub-tetrahedra, triangles, segments), exactly. Only the cells of the mesh's dimension count, as in VTK: a volume's
    integrals leave out its surfaces, lines and vertices. Only owned cells count
    (ghost marks), and a ``PartitionedDataSet`` sums its pieces over every rank, in
    the pieces' order."""
    from tack.dataset.algorithms._common import _measure_unit
    from tack.dataset.partitioned import PartitionedDataSet

    if isinstance(data, PartitionedDataSet):
        from tack.dataset.algorithms.statistics import _over_pieces

        pieces = list(data)
        names = _names(pieces[0], fields) if pieces else []
        parts = _over_pieces(data, lambda piece: _piece(piece, names))
        measure = math.fsum(m for m, _ in parts)
        values = {name: np.sum([v[name] for _, v in parts], axis=0) for name in names}
        first = pieces[0] if pieces else None
    else:
        names = _names(data, fields)
        measure, values = _piece(data, names)
        first = data
    out_units = {name: units.derived((first.fields[name].unit, 1),
                                     (_measure_unit(first), 1)) if first is not None else None
                 for name in names}
    return Integrals(measure=measure, measure_unit=_measure_unit(first) if first else None,
                     values={name: (float(v) if np.ndim(v) == 0 else tuple(map(float, v)))
                             for name, v in values.items()},
                     units=out_units)


def _names(data, fields):
    if fields is None:
        return [name for name, field in data.fields.items()
                if isinstance(field.space, (H1, L2, Constant))
                or (isinstance(field.space, Values) and field.space.on in ("points", "cells"))]
    names = [fields] if isinstance(fields, str) else list(fields)
    for name in names:
        _field(data, name)
    return names
