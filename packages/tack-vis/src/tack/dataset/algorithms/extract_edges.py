"""``extract_edges``: the edges of set ``name`` (all of them by default) as line cells on the same points, each running from its lower point id to its higher. Values on edges become the lines' cell values, and edge sets cell sets; point fields (``H1``, values on points) stay on the same points -- an order-2 field's or geometry's values at the points, the lines being linear. Cell fields are left behind: an edge has as many cells as meet there. The geometry must be ``H1``."""

import tack
from tack import arrays
from tack.arrays import CountingArray
from tack.dataset import ids as id_types
from tack.dataset import shapes
from tack.dataset.carry import Same, Take, carry
from tack.dataset.sets import Set
from tack.dataset.spaces import H1
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
def _curved_edges(ids, rows, positions, trace, nodes, weights, q, bernstein, level, n,
                  points, connectivity, out):
    # Edge i (edge ids[i]) as level segments: its level - 1 inner points on the
    # geometry's trace, at k / level from its first point by position (so every
    # piece computes the same bits), numbered from its lower point id.
    for i in range(n):
        e = ids[i]
        row = rows[e]
        flip = _before(positions[row[1]], positions[row[0]])
        previous = row[0]
        for k in range(1, level):
            kk = level - k if flip else k
            t = (kk + 0.0) / level
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
    edge lies on every cell and face around it. The geometry must be ``H1``:
    an ``L2`` one has no shared points for the lines to stand on.

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
                      basis["trace_bernstein"], level, n, num_points, connectivity, out)
    types, offsets = uniform_cells(shapes.LINE, n * level, idt)
    lines = UnstructuredTopology(types, offsets, connectivity, num_points=total)
    return drop_marks(data, DataSet(lines, out, **data._frame()),
                      "extract_edges' subdivided edges have no marks")


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
