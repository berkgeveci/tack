"""``project``: a field in another space of the same topology -- the one way to move
a field between spaces."""


import tack
from tack.arrays import CartesianProduct, CountingArray, UniformCoordinates, dtype_of, materialize
from tack.dataset import bases, ids, shapes
from tack.dataset.algorithms._common import _field, _with_basis
from tack.dataset.algorithms._links import _average_cell_values, _average_links
from tack.dataset.algorithms.integrate import (
    _element,
    _no_reference_values,
    _order,
    _points_per_axis,
    _tables,
)
from tack.dataset.algorithms.traces import traces
from tack.dataset.buckets import bucket_order
from tack.dataset.carry import _like, _take
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import H1, L2, Constant, SideTraces, Values, reference_cells
from tack.dataset.topology import UnstructuredTopology, corner_layout
from tack.runtime.dispatch import on_inputs_device

__all__ = ['project']


# ── Onto cells ──────────────────────────────────────────────────────

@tack.func
def _node_mean(u, c):
    # The mean of the field's own values in the cell, as vtkPointDataToCellData takes
    # it: every point a VTK cell has, of any order.
    total = u.dof(c, 0)
    for k in range(1, u.NODES):
        total += u.dof(c, k)
    return total / u.NODES


@tack.func
def _corner_mean(cells, u, c):
    # The mean of the cell's corner values: its average where the field is linear
    # along each axis of a cell its map does not distort.
    total = u.corner_value(c, 0)
    for j in range(1, cells.NUM_POINTS):
        total += u.corner_value(c, j)
    return total / cells.NUM_POINTS


@tack.func
def _integral_mean(cells, u, c, points, weights):
    # The field's integral over the cell divided by the cell's measure, by quadrature
    # through the cell's own map; a cell without measure takes its nodes' mean.
    m = weights[0] * 0.0
    total = u.value(c, points[0]) * 0.0
    for q in range(weights.shape[0]):
        d = weights[q] * _element(cells.geometry_jacobian(c, points[q]), cells.DIMENSION)
        m += d
        total += d * u.value(c, points[q])
    return total / m if m > 0.0 else _node_mean(u, c)


@tack.kernel
def _node_means(cells, u, place, out):
    for c in cells:
        out[place[cells.entity_id(c)]] = _node_mean(u, c)


@tack.kernel
def _corner_means(cells, u, place, out):
    for c in cells:
        out[place[cells.entity_id(c)]] = _corner_mean(cells, u, c)


@tack.kernel
def _integral_means(cells, u, points, weights, place, out):
    for c in cells:
        out[place[cells.entity_id(c)]] = _integral_mean(cells, u, c, points, weights)


@tack.kernel
def _tensor_cell_nodes(cells, u, t, offsets, out):
    # An L2 cell's values on a tensor basis: the source at the target's nodes,
    # stored where the target's layout keeps them.
    for c in cells:
        for n in range((t.ORDER + 1) ** cells.DIMENSION):
            out[t.dof_index(c, n)] = u.value(c, t.node_parametric(n))


@tack.kernel
def _lattice_cell_nodes(cells, u, t, offsets, out):
    # The same on a simplex's or wedge's lattice: its NODES nodes, not (p + 1)^d
    # (the tensor count ran past them: values written past the cell's own).
    for c in cells:
        for n in range(t.NODES):
            out[t.dof_index(c, n)] = u.value(c, t.node_parametric(n))


@tack.kernel
def _cell_nodes(cells, u, t, offsets, out):
    # An order-1 L2 cell's values: the source's at its corners.
    for c in cells:
        first = offsets[cells.entity_id(c)]
        for j in range(cells.NUM_POINTS):
            out[first + j] = u.value(c, cells.parametric_point(j))


@tack.kernel
def _corner_copies(cells, u, offsets, out):
    # An order-1 L2 cell's values from point values: its corners' own, copied, so
    # integers come across too.
    for c in cells:
        first = offsets[cells.entity_id(c)]
        for j in range(cells.NUM_POINTS):
            out[first + j] = u.corner_value(c, j)


@tack.kernel
def _distinct_point_means(cells, values, out):
    # A polyhedral, polygonal or polyline cell: the mean over its distinct points.
    for c in cells:
        n = cells.num_points(c)
        total = values[cells.point_id(c, 0)]
        for j in range(1, n):
            total += values[cells.point_id(c, j)]
        out[cells.entity_id(c)] = total / n


@tack.kernel
def _divided(integrals, measures, out, n):
    # A cell's integral over its measure, where it has one (out keeps the rest).
    for c in range(n):
        if measures[c] > 0.0:
            out[c] = integrals[c] / measures[c]


# Shapes whose straight (order-1) map is affine: a field linear along each axis
# averages to its corners' mean there, as on any box of an image or rectilinear grid.
_AFFINE = (shapes.Vertex, shapes.Line, shapes.Triangle, shapes.Tetra, shapes.Pixel,
           shapes.Voxel)

_AVERAGES = ("integral", "nodes")


def _onto_cells(data, field, groups, place, out, average):
    """Each cell of ``groups`` (launch groups for ``field``) its one value of
    ``field``, at ``out[place[cell]]``: by ``average``, the field's integral over the
    cell divided by the cell's measure, or the mean of the field's nodes."""
    source = field.space
    straight = _order(data.geometry.space) <= 1
    boxes = isinstance(data.geometry.values, (UniformCoordinates, CartesianProduct))
    for group in groups:
        if not group.count:
            continue
        cells, u = data.domain_view("cells", group), field.view(group)
        if average == "nodes":
            _node_means(cells, u, place, out)
            continue
        order = int(group.keys[source]) if source.varies else _order(source)
        if order <= 1 and (boxes or (straight and group.shape in _AFFINE)):
            _corner_means(cells, u, place, out)       # the same average, without quadrature
        else:
            n = _points_per_axis(data, order, group.shape)
            _integral_means(cells, u, *_tables(data, group.shape, n), place, out)


def _polyhedral_means(data, field, average):
    """Each polyhedron's, polygon's or polyline's one value of a point field: its
    integral, linear on the cell's pieces, over its measure, or its distinct
    points' mean (also where a cell has no measure)."""
    from tack.dataset.algorithms.cell_geometry import cell_geometry

    values = materialize(field.values)
    out = _like(values, data.num_cells)
    for_each(_distinct_point_means, data, "cells", values, out)
    if average == "integral" and data.num_cells:
        measures = materialize(cell_geometry(data).measure.values)
        _divided(_no_reference_values(data, field), measures, out, data.num_cells)
    return out


# ── Onto points and nodes ───────────────────────────────────────────

@tack.kernel
def _corner_contributions(cells, u, start, contributions):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            contributions[first + j] = u.corner_value(c, j)


@tack.kernel
def _node_contributions(cells, u, t, start, count, keys, contributions):
    # Each cell's value at each of its first count nodes of the order-2 target (all
    # NUM_QUADRATIC, or serendipity's corners and edges), keyed by the node.
    for c in cells:
        first = start + cells.index(c) * count
        for k in range(count):
            keys[first + k] = t.dof_index(c, k)
            contributions[first + k] = u.value(c, cells.quadratic_node(k))


@tack.kernel
def _listed_node_contributions(cells, u, t, start, node_of, count, keys, contributions):
    # Each cell's value at each node of an H1 target read as a basis of any order
    # (the nodes it stores, lexicographic: node_of), keyed by the node.
    for c in cells:
        first = start + cells.index(c) * count
        for k in range(count):
            n = node_of[k]
            keys[first + k] = t.dof_index(c, n)
            contributions[first + k] = u.value(c, t.node_parametric(n))


@tack.kernel
def _cell_contributions(point_offsets, values, contributions, n_cells):
    for c in range(n_cells):
        for e in range(point_offsets[c], point_offsets[c + 1]):
            contributions[e] = values[c]


@tack.kernel
def _first_values(source, out, n):
    for i in range(n):
        out[i] = source[i]


def _start(data, field, out):
    """``out`` zeroed, but where the source is continuous its point values first:
    a point no cell uses keeps its value, which no cell can give it."""
    out.fill(0)
    if isinstance(field.space, H1) and data.num_points:
        _first_values(materialize(field.values), out, data.num_points)


def _onto_points(data, field, values):
    """Order-1 H1: each point the mean of what the cells around it give it -- their
    corner values -- summed in a fixed order through the point links."""
    topology = data.topology
    groups = topology.groups() if isinstance(topology, UnstructuredTopology) else ()
    if len(groups) == 1 and isinstance(field.space, Constant) and groups[0].count:
        # A value a cell, cells of one shape, indexed by an integer: each
        # corner's contribution is its cell's value, read through the links,
        # without a contribution a corner (8 values a hexahedron: 640 MB at
        # 20M).
        out = _like(values, data.num_points)
        out.fill(0)
        link_offsets, entries = topology.point_links()
        _average_cell_values(link_offsets, entries, groups[0].view(), values, out)
        return out
    if not getattr(topology, "reference_cells", True):
        offsets, point_ids = topology.cell_points()
        total = int(point_ids.shape[0])
        contributions = _like(values, total)
        if total:
            _cell_contributions(offsets, materialize(field.values), contributions,
                                data.num_cells)
    else:
        start_of, total = corner_layout(topology)
        contributions = _like(values, total)
        for group in data.launch_groups("cells", [field]):
            if group.count:
                _corner_contributions(data.domain_view("cells", group), field.view(group),
                                      start_of[group.root], contributions)
    out = _like(values, data.num_points)
    _start(data, field, out)
    if total:
        link_offsets, entries = topology.point_links()
        _average_links(link_offsets, entries, contributions, out)
    return out


def _quadratic_count(space, shape):
    """The order-2 nodes a space stores on a ``shape`` cell, in the quadratic shapes'
    order: all of them, or serendipity's corners and edges (the first ones)."""
    if space.basis == bases.SERENDIPITY:
        return shape.NUM_POINTS + shape.NUM_EDGES
    return shape.NUM_QUADRATIC


def _onto_nodes(data, field, space, values):
    """H1 of order 2 and up: each node the mean of the cells around it evaluated there."""
    groups = data.topology.groups()
    starts, total = {}, 0
    nodes = {}
    for group in groups:
        starts[group] = total
        if space._any_order(group.shape):
            nodes[group] = space.stored_nodes(group.shape)
            per_cell = len(nodes[group])
        else:
            per_cell = _quadratic_count(space, group.shape)
        total += group.count * per_cell
    idt = ids.at_least(data.id_dtype, space.size)
    keys = tack.empty((total,), idt)
    contributions = _like(values, total)
    target = Field(space, _like(values, space.size))
    for group in data.launch_groups("cells", [field, target]):
        if group.count:
            cells, u, t = data.domain_view("cells", group), field.view(group), target.view(group)
            listed = nodes.get(group.root)
            if listed is None:
                _node_contributions(cells, u, t, starts[group.root],
                                    _quadratic_count(space, group.shape), keys, contributions)
            else:
                _listed_node_contributions(cells, u, t, starts[group.root],
                                           tack.array(listed), len(listed), keys, contributions)
    out = target.values
    _start(data, field, out)              # an order-2 field's points come first
    if total:
        order, offsets = bucket_order(keys, space.size, idt)
        _average_links(offsets, order, contributions, out)
    return out


# ── The entry point ─────────────────────────────────────────────────

def _one_value_a_cell(space):
    """Whether ``space`` holds one value in some cell: ``Constant``, order-0 ``L2``,
    or an ``L2`` with an order per cell."""
    return isinstance(space, Constant) or (isinstance(space, L2) and (space.varies
                                                                     or space.order == 0))


@on_inputs_device
def project(data, field, space, *, average="integral"):
    """``field`` (a ``Field`` or a name) in ``space``, a space of the same topology:
    a new field. Values on points or cells count as order-1 ``H1`` and
    ``Constant``, both as source and as target.

    Into ``H1``, each node takes the mean of the source evaluated there by the
    cells around it: the source's own value wherever it is continuous (so a
    continuous field of lower order comes across exactly), and an average where
    cells disagree -- a cell field's, as vtkCellDataToPointData takes it, or a DG
    field's corner values. Into ``Constant`` (or order-0 ``L2``), each cell takes
    the source's average over it, its integral divided by the cell's measure, by
    quadrature through the cell's own map: exact for any order, basis and curved
    geometry. ``average="nodes"`` takes instead the mean of the source's own
    values in the cell (its ``NODES``), as vtkPointDataToCellData and Viskores'
    CellAverage do: the same on a linear field over a tetrahedron or a box, not
    on a pyramid, a distorted cell or a field of higher order. Into order-1
    ``L2``, each cell takes the source's values at its corners, exactly whatever
    the source's order is up to 1. Into ``SideTraces``, each face side the
    source's values from its cell (``traces``). Sums run in a fixed order, so
    results do not vary from run to run.

    Values on faces or edges have no basis to project, and are refused as a
    source or target. On polyhedra, polygons and polylines -- no reference
    element -- only cell values and point values project, one into the other:
    a cell's average is its integral, linear on the pieces the cell is cut
    into, over its measure (``average="nodes"``: the mean over its distinct
    points). The values must be floating point, scalars or vectors, except from
    order-1 ``H1`` into order-1 ``L2``, which copies them and takes integers too.
    """
    if average not in _AVERAGES:
        raise ValueError(f"average is one of {', '.join(map(repr, _AVERAGES))}, "
                         f"not {average!r}")
    field = _with_basis(data, _field(data, field))
    source = field.space
    if space.topology is not data.topology:
        raise ValueError("project moves a field between spaces of one topology")
    if isinstance(space, Values) and space.on in ("points", "cells"):
        projected = project(data, field, H1(data) if space.on == "points" else Constant(data),
                            average=average)
        return Field(space, projected.values, field.unit)
    if average != "integral" and not _one_value_a_cell(space):
        raise ValueError(f"average chooses how a cell's one value is taken, into Constant or "
                         f"order-0 L2; {space!r} has none")
    for which, s in (("from", source), ("into", space)):
        if not s.interpolated:
            raise TypeError(f"project works {which} a space with a basis, or values on the "
                            f"points or cells; {s!r} has no basis to project")
    if isinstance(source, SideTraces):
        raise TypeError("a field of traces has two values on each face; project the field "
                        "they were taken from")
    values = field.values
    if space is source:
        return Field(space, _take(values, tack.arange(space.size, dtype=data.id_dtype)),
                     field.unit)
    if dtype_of(values) not in (tack.f32, tack.f64):
        if (isinstance(space, L2) and not space.varies and space.order == 1
                and isinstance(source, H1) and source.order == 1
                and getattr(data.topology, "reference_cells", True)):
            out = _like(values, space.size)
            for group in data.launch_groups("cells", [field]):
                if group.count:
                    _corner_copies(data.domain_view("cells", group), field.view(group),
                                   space.offsets, out)
            return Field(space, out, field.unit)
        raise TypeError(f"project averages and interpolates floating-point values, not "
                        f"{dtype_of(values).name}")
    if isinstance(space, SideTraces):
        return traces(data, field)
    from tack.dataset import _bernstein

    if _bernstein.is_bernstein(space):
        # Into the equispaced nodes, then their Bernstein coefficients.
        nodal = project(data, field, _bernstein.lagrange_companion(space), average=average)
        return _bernstein.from_lagrange(data, nodal, space)
    if not getattr(data.topology, "reference_cells", True):
        if isinstance(space, H1) and space.order == 1 and isinstance(source, Constant):
            return Field(space, _onto_points(data, field, values), field.unit)
        if isinstance(space, Constant) and isinstance(source, H1) and source.order == 1:
            return Field(space, _polyhedral_means(data, field, average), field.unit)
        reference_cells(data.topology, f"project from {source!r} to {space!r}")
    if isinstance(space, H1):
        if space.order == 1:
            return Field(space, _onto_points(data, field, values), field.unit)
        return Field(space, _onto_nodes(data, field, space, values), field.unit)
    out = _like(values, space.size)
    if isinstance(space, Constant) or (isinstance(space, L2) and not space.varies
                                       and space.order == 0):
        _onto_cells(data, field, data.launch_groups("cells", [field]),
                    CountingArray(data.num_cells, dtype=data.id_dtype), out, average)
        return Field(space, out, field.unit)
    target = Field(space, out)
    groups = data.launch_groups("cells", [field, target])
    if space.varies:
        # An order per cell: the order-0 cells' one value each, then the others'.
        _onto_cells(data, field, [g for g in groups if g.keys[space] == 0], space.offsets,
                    out, average)
        groups = [g for g in groups if g.keys[space] != 0]
    for group in groups:
        if group.count:
            kernel = _cell_nodes
            if getattr(space, "tensor", False):
                kernel = (_tensor_cell_nodes if bases.tensor_shape(group.shape)
                          else _lattice_cell_nodes)
            kernel(data.domain_view("cells", group), field.view(group), target.view(group),
                   space.offsets, out)
    return Field(space, out, field.unit)
