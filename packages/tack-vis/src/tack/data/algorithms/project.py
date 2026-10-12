"""``project``: a field in another space of the same topology -- the one way to move
a field between spaces, of which ``to_points``, ``to_cells`` and ``discontinuous``
are shortcuts."""

import tack
from tack.data import ids
from tack.data.algorithms._common import _field
from tack.data.algorithms._links import _average_links
from tack.data.algorithms.traces import traces
from tack.data.arrays import materialize
from tack.data.buckets import bucket_order
from tack.data.carry import _like, _take
from tack.data.dataset import Field, for_each
from tack.data.spaces import H1, L2, Constant, SideTraces, Values, reference_cells
from tack.data.topology import corner_layout

__all__ = ['project']


# ── Onto cells ──────────────────────────────────────────────────────

@tack.kernel
def _cell_means(cells, u, out):
    # The mean of the cell's corner values, as vtkPointDataToCellData takes it.
    for c in cells:
        total = u.corner_value(c, 0)
        for j in range(1, cells.NUM_POINTS):
            total += u.corner_value(c, j)
        out[cells.entity_id(c)] = total / cells.NUM_POINTS


@tack.kernel
def _cell_nodes(cells, u, t, offsets, out):
    # An L2 cell's values: its corner mean at order 0, else its corners' values.
    for c in cells:
        first = offsets[cells.entity_id(c)]
        if t.ORDER == 0:
            total = u.corner_value(c, 0)
            for j in range(1, cells.NUM_POINTS):
                total += u.corner_value(c, j)
            out[first] = total / cells.NUM_POINTS
        else:
            for j in range(cells.NUM_POINTS):
                out[first + j] = u.value(c, cells.parametric_point(j))


@tack.kernel
def _distinct_point_means(cells, values, out):
    # A polyhedral, polygonal or polyline cell: the mean over its distinct points.
    for c in cells:
        n = cells.num_points(c)
        total = values[cells.point_id(c, 0)]
        for j in range(1, n):
            total += values[cells.point_id(c, j)]
        out[cells.entity_id(c)] = total / n


# ── Onto points and nodes ───────────────────────────────────────────

@tack.kernel
def _corner_contributions(cells, u, start, contributions):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            contributions[first + j] = u.corner_value(c, j)


@tack.kernel
def _node_contributions(cells, u, t, start, keys, contributions):
    # Each cell's value at each node of the order-2 target, keyed by the node.
    for c in cells:
        first = start + cells.index(c) * cells.NUM_QUADRATIC
        for k in range(cells.NUM_QUADRATIC):
            keys[first + k] = t.dof_index(c, k)
            contributions[first + k] = u.value(c, cells.quadratic_node(k))


@tack.kernel
def _cell_contributions(point_offsets, values, contributions, n_cells):
    for c in range(n_cells):
        for e in range(point_offsets[c], point_offsets[c + 1]):
            contributions[e] = values[c]


def _onto_points(data, field, values):
    """Order-1 H1: each point the mean of what the cells around it give it -- their
    corner values -- summed in a fixed order through the point links."""
    topology = data.topology
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
                                      start_of[id(group.root)], contributions)
    out = _like(values, data.num_points)
    out.fill(0)
    if total:
        link_offsets, entries = topology.point_links()
        _average_links(link_offsets, entries, contributions, out)
    return out


def _onto_nodes(data, field, space, values):
    """Order-2 H1: each node the mean of the cells around it evaluated there."""
    groups = data.topology.groups()
    starts, total = {}, 0
    for group in groups:
        starts[id(group)] = total
        total += group.count * group.shape.NUM_QUADRATIC
    idt = ids.at_least(data.id_dtype, space.size)
    keys = tack.field(idt, shape=(total,))
    contributions = _like(values, total)
    target = Field(space, _like(values, space.size))
    for group in data.launch_groups("cells", [field, target]):
        if group.count:
            _node_contributions(data.domain_view("cells", group), field.view(group),
                                target.view(group), starts[id(group.root)], keys,
                                contributions)
    out = target.values
    out.fill(0)
    if total:
        order, offsets = bucket_order(keys, space.size, idt)
        _average_links(offsets, order, contributions, out)
    return out


# ── The entry point ─────────────────────────────────────────────────

def _with_basis(data, field):
    """Values on points or cells read as the space they are: order-1 H1, Constant."""
    space = field.space
    if isinstance(space, Values) and space.on == "points":
        return Field(H1(data), field.values)
    if isinstance(space, Values) and space.on == "cells":
        return Field(Constant(data), field.values)
    return field


def project(data, field, space):
    """``field`` (a ``Field`` or a name) in ``space``, a space of the same topology:
    a new field. Values on points or cells count as order-1 ``H1`` and
    ``Constant``, both as source and as target.

    Into ``H1``, each node takes the mean of the source evaluated there by the
    cells around it: the source's own value wherever it is continuous (so a
    continuous field of lower order comes across exactly), and an average where
    cells disagree -- a cell field's, as vtkCellDataToPointData takes it, or a DG
    field's corner values. Into ``Constant`` (or order-0 ``L2``), each cell takes
    the mean of its corner values, as vtkPointDataToCellData does. Into order-1
    ``L2``, each cell takes the source's values at its corners, exactly whatever
    the source's order is up to 1. Into ``SideTraces``, each face side the
    source's values from its cell (``traces``). Sums run in a fixed order, so
    results do not vary from run to run.

    Values on faces or edges have no basis to project, and are refused as a
    source or target. On polyhedra, polygons and polylines -- no reference
    element -- only cell values and point values project, one into the other,
    each cell averaging over its distinct points. The values must be floating
    point, scalars or vectors.
    """
    field = _with_basis(data, _field(data, field))
    source = field.space
    if space.topology is not data.topology:
        raise ValueError("project moves a field between spaces of one topology")
    if isinstance(space, Values) and space.on in ("points", "cells"):
        projected = project(data, field, H1(data) if space.on == "points" else Constant(data))
        return Field(space, projected.values)
    for which, s in (("from", source), ("into", space)):
        if not s.interpolated:
            raise TypeError(f"project works {which} a space with a basis, or values on the "
                            f"points or cells; {s!r} has no basis to project")
    if isinstance(source, SideTraces):
        raise TypeError("a field of traces has two values on each face; project the field "
                        "they were taken from")
    values = field.values
    if space is source:
        return Field(space, _take(values, tack.arange(space.size, data.id_dtype)))
    if values.dtype not in (tack.f32, tack.f64):
        raise TypeError(f"project averages and interpolates floating-point values, not "
                        f"{values.dtype.name}")
    if isinstance(space, SideTraces):
        return traces(data, field)
    if not getattr(data.topology, "reference_cells", True):
        if isinstance(space, H1) and space.order == 1 and isinstance(source, Constant):
            return Field(space, _onto_points(data, field, values))
        if isinstance(space, Constant) and isinstance(source, H1) and source.order == 1:
            out = _like(values, data.num_cells)
            for_each(_distinct_point_means, data, "cells", materialize(values), out)
            return Field(space, out)
        reference_cells(data.topology, f"project from {source!r} to {space!r}")
    if isinstance(space, H1):
        if space.order == 1:
            return Field(space, _onto_points(data, field, values))
        return Field(space, _onto_nodes(data, field, space, values))
    out = _like(values, space.size)
    if isinstance(space, Constant) or (isinstance(space, L2) and not space.varies
                                       and space.order == 0):
        for group in data.launch_groups("cells", [field]):
            if group.count:
                _cell_means(data.domain_view("cells", group), field.view(group), out)
        return Field(space, out)
    target = Field(space, out)
    for group in data.launch_groups("cells", [field, target]):
        if group.count:
            _cell_nodes(data.domain_view("cells", group), field.view(group), target.view(group),
                        space.offsets, out)
    return Field(space, out)
