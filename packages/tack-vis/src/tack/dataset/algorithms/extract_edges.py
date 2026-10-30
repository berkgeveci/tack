"""``extract_edges``: the edges of set ``name`` (all of them by default) as line cells on the same points, each running from its lower point id to its higher. Values on edges become the lines' cell values, and edge sets cell sets; point fields (``H1``, values on points) stay on the same points -- an order-2 field's or geometry's values at the points, the lines being linear. Cell fields are left behind: an edge has as many cells as meet there. An ``L2`` geometry's edges are each cell's own."""

import numpy as np

import tack
from tack import arrays
from tack.arrays import CountingArray
from tack.dataset import ids as id_types
from tack.dataset import shapes
from tack.dataset.carry import Same, Take, carry
from tack.dataset.sets import Set
from tack.dataset.spaces import H1, L2
from tack.dataset.topology import UnstructuredTopology, uniform_cells
from tack.dataset.views import _axis, _before
from tack.runtime.dispatch import on_inputs_device

__all__ = ['extract_edges']


@tack.kernel
def _edge_cells(ids, rows, connectivity, n):
    for i in range(n):
        row = rows[ids[i]]
        connectivity[2 * i] = row[0]
        connectivity[2 * i + 1] = row[1]


@tack.kernel
def _curved_edges(ids, rows, positions, trace, nodes, weights, q, bernstein, steps, level, n,
                  points, connectivity, out):
    # Edge i (edge ids[i]) as level segments: its level - 1 inner points on the
    # geometry's trace, at k / level from its first point by position (so every
    # piece computes the same bits), numbered from its lower point id. steps[k] is
    # k / level in the geometry's type (an integer over a literal is f32).
    for i in range(n):
        e = ids[i]
        row = rows[e]
        flip = _before(positions[row[1]], positions[row[0]])
        previous = row[0]
        for k in range(1, level):
            kk = level - k if flip else k
            t = steps[kk]
            x = trace[e * q] * _axis(t, 0, nodes, weights, q, bernstein)
            for j in range(1, q):
                x += trace[e * q + j] * _axis(t, j, nodes, weights, q, bernstein)
            p = points + i * (level - 1) + k - 1
            out[p] = x
            connectivity[2 * (i * level + k - 1)] = previous
            connectivity[2 * (i * level + k - 1) + 1] = p
            previous = p
        connectivity[2 * (i * level + level - 1)] = previous
        connectivity[2 * (i * level + level - 1) + 1] = row[1]


@tack.kernel
def _edge_counts(cells, counts):
    for c in cells:
        counts[cells.entity_id(c)] = cells.NUM_EDGES


@tack.kernel
def _own_edges(cells, starts, steps, level, out, connectivity):
    # Each of a cell's edges as level segments through level + 1 points of its own
    # on the cell's map, from the edge's first corner.
    for c in cells:
        first = starts[cells.entity_id(c)]
        for k in range(cells.NUM_EDGES):
            a = cells.parametric_point(cells.edge_corner(k, 0))
            b = cells.parametric_point(cells.edge_corner(k, 1))
            edge = first + k
            for t in range(level + 1):
                p = edge * (level + 1) + t
                out[p] = cells.position(c, a + (b - a) * steps[t])
                if t < level:
                    connectivity[2 * (edge * level + t)] = p
                    connectivity[2 * (edge * level + t) + 1] = p + 1


@tack.kernel
def _copy_points(values, out, n):
    for i in range(n):
        out[i] = values[i]


@on_inputs_device
def extract_edges(data, edges=None, fields=None, subdivision=None):
    """The ``edges`` -- an edge ``Set``, a set's name in ``data.sets``, or edge ids (a
    field or an array); by default every edge of the topology (``edges()``) --
    as a dataset of line cells on the same points, cell ``i`` running from edge
    ``i``'s lower point id to its higher (``edges().rows``). Values on edges
    (``Values(data, "edges")``) become the output's cell values, and edge sets
    its cell sets; point fields (``H1``, values on points) and point sets stay
    on the same points -- an order-2 field's or geometry's values at the
    points, the lines being linear. Cell and face fields are left behind: an
    edge lies on every cell and face around it.

    An ``L2`` geometry has no shared points for the lines to stand on: its cells
    need not meet (a periodic mesh's seam, a mesh displaced by a DG field), so
    each cell's edges are drawn on their own points, in cell order, along the
    cell's map (``subdivision`` segments each), with no fields kept.

    ``subdivision=n`` follows a curved geometry: each edge is ``n`` line cells
    through ``n - 1`` new points on the geometry's trace along it, appended after
    the points (edge ``i``'s from its lower point id), so a mesh's outline shows
    its cells as they are. The output's geometry is the points' positions and
    the new ones; no other field is kept, and no ghost marks. A straight
    geometry's edges are lines already, and are not subdivided."""
    from tack.dataset.ghosts import refuse_hidden, visible

    if edges is None:
        data = visible(data)
    else:
        refuse_hidden(data, "extract_edges of chosen edges")
    if isinstance(data.geometry.space, L2):
        if edges is not None or fields:
            raise NotImplementedError("extract_edges of an L2 geometry draws every cell's "
                                      "own edges, without fields")
        return _own_cell_edges(data, int(subdivision or 1))
    if not isinstance(data.geometry.space, H1):
        raise TypeError("extract_edges keeps the points, so needs an H1 geometry")
    topology = data.topology
    found = topology.edges()
    idt = data.id_dtype
    if edges is None:
        n = found.num_edges
        chosen = CountingArray(n, dtype=idt)              # every edge, in order
        edges_to_cells = Same()
    else:
        chosen = _edge_ids(data, edges, found.num_edges)
        n = chosen.shape[0]
        edges_to_cells = Take(chosen)
    types, offsets = uniform_cells(shapes.LINE, n, idt)       # all lines: implicit
    connectivity = tack.empty((2 * n,), idt)
    if n:
        _edge_cells(chosen, found.rows, connectivity, n)
    if (subdivision is not None and int(subdivision) > 1
            and data.geometry.space.order > 1):
        if fields:
            raise ValueError("extract_edges keeps only the geometry when it subdivides edges")
        return _subdivided(data, chosen, n, int(subdivision))
    lines = UnstructuredTopology(types, offsets, connectivity, num_points=data.num_points)
    return carry(data, lines, points=Same(), edges_to_cells=edges_to_cells, fields=fields)


def _subdivided(data, chosen, n, level):
    """``chosen``'s ``n`` edges, each ``level`` line cells along the curved geometry's
    trace. The geometry's first values are the points' (every ``H1`` layout's)."""
    from tack.dataset import DataSet
    from tack.dataset.dataset import _entity_traces
    from tack.dataset.ghosts import drop_marks

    trace, _, basis = _entity_traces(data, faces=False)
    positions = arrays.materialize(data.geometry.values)
    num_points = data.num_points
    total = num_points + n * (level - 1)
    idt = id_types.at_least(data.id_dtype, total)
    out = tack.empty((total,), arrays.element_type(positions))
    connectivity = tack.empty((2 * n * level,), idt)
    if num_points:
        _copy_points(positions, out, num_points)
    if n:
        _curved_edges(chosen, data.topology.edges().rows, positions, trace,
                      basis["trace_nodes"], basis["trace_weights"], basis["trace_q"],
                      basis["trace_bernstein"], _steps(level, positions), level, n, num_points,
                      connectivity, out)
    types, offsets = uniform_cells(shapes.LINE, n * level, idt)
    lines = UnstructuredTopology(types, offsets, connectivity, num_points=total)
    return drop_marks(data, DataSet(lines, out, **data._frame()),
                      "extract_edges' subdivided edges have no marks")


def _steps(level, like):
    """``k / level`` for ``k`` from 0 to ``level``, in ``like``'s float type."""
    dtype = arrays.dtype_of(like)
    steps = tack.empty((level + 1,), dtype)
    tack.copyto(steps, (np.arange(level + 1) / level).astype(dtype.numpy_dtype))
    return steps


def _own_cell_edges(data, level):
    """Every cell's edges as ``level`` line cells each, on points of their own
    along the cell's map (an ``L2`` geometry's edges)."""
    from tack.algorithms.scan import exclusive_scan
    from tack.dataset import DataSet
    from tack.dataset.ghosts import drop_marks

    counts = tack.zeros((data.num_cells,), tack.i64)
    for group in data.launch_groups("cells", []):
        if group.count:
            _edge_counts(data.domain_view("cells", group), counts)
    starts = tack.empty((data.num_cells,), tack.i64)
    n = exclusive_scan(counts, starts, data.num_cells) if data.num_cells else 0
    total = n * (level + 1)
    idt = id_types.at_least(data.id_dtype, total)
    out = tack.empty((total,), tack.vec(3, data.dtype))
    connectivity = tack.empty((2 * n * level,), idt)
    steps = _steps(level, out)
    for group in data.launch_groups("cells", []):
        if group.count and group.shape.NUM_EDGES:
            _own_edges(data.domain_view("cells", group), starts, steps, level, out,
                       connectivity)
    types, offsets = uniform_cells(shapes.LINE, n * level, idt)
    lines = UnstructuredTopology(types, offsets, connectivity, num_points=total)
    return drop_marks(data, DataSet(lines, out, **data._frame()),
                      "extract_edges' edges of an L2 geometry have no marks")


def _edge_ids(data, edges, count):
    """``edges`` -- an edge set, its name, or edge ids -- as a field of the topology's
    ids: a set of this topology's edges, or ids within its edge count."""
    if isinstance(edges, str):
        edges = data.sets[edges]
    if isinstance(edges, Set):
        if edges.on != "edges":
            raise ValueError(f"extract_edges takes edges, not a set of {edges.on}")
        if edges.topology is not data.topology:
            raise ValueError("extract_edges takes a set of this dataset's edges, not "
                             "one of another topology")
        return edges.ids
    ids = id_types.as_ids(edges, data.id_dtype)
    id_types.check_bounds(ids, count, "edge")
    return ids
