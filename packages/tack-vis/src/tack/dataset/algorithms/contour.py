"""``contour``: the surface where ``field`` crosses ``value`` -- or each of several
values, in one pass over the cells -- as triangles."""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.arrays import CountingArray, size_of
from tack.dataset import ids, shapes
from tack.dataset.algorithms._common import (
    _close,
    _compact,
    _crossing,
    _field,
    _first_end,
    _in_unit_of,
    _scalar_floating,
    _with_basis,
)
from tack.dataset.buckets import key_order, run_offsets
from tack.dataset.carry import Interpolate, Pieces, _like, carry
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1, Values
from tack.dataset.topology import UnstructuredTopology, uniform_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['contour']


def _levels(value, field):
    """The isovalues as a field of the field's floating type, their number, and
    whether several were given (a sequence: the output then records each
    cell's value)."""
    several = np.ndim(value) > 0
    host = np.asarray(value, arrays.dtype_of(field.values).numpy_dtype).reshape(-1)
    if not host.size:
        raise ValueError("contour needs at least one value")
    return tack.array(host), int(host.size), several


@tack.kernel
def _level_values(levels, which, out):
    for i in range(out.shape[0]):
        out[i] = levels[which[i]]


def _with_levels(out, levels, which, several, unit=None):
    """``out`` with each cell's isovalue (``contour_value``, in the contoured
    field's ``unit``) when several were given."""
    if several:
        from tack.dataset.spaces import Constant

        values = _like(levels, out.num_cells)
        if out.num_cells:
            _level_values(levels, which, values)
        out.fields["contour_value"] = Field(Constant(out), values, unit)
    return out


@tack.func
def _cell_case(cells, u, c, level):
    case = 0
    for j in range(cells.NUM_POINTS):
        if u.corner_value(c, j) >= level:
            case |= 1 << j
    return case


def _scanned(counts, n, odt):
    """``(starts, total)``: the exclusive scan of ``counts`` (i32, one per cell) in
    ``odt``, in place when ``odt`` is i32 -- one array a cell, not two."""
    if not n:
        return tack.empty((0,), odt), 0
    starts = counts if odt == counts.dtype else tack.empty((n,), odt)
    return starts, exclusive_scan(counts, starts, n)


@tack.kernel
def _contour_counts(cells, u, levels, nlevels, counts):
    # Every level's triangles, in one pass over the cells.
    for c in cells:
        total = 0
        for v in range(nlevels):
            total += cells.contour_count(_cell_case(cells, u, c, levels[v]))
        counts[cells.entity_id(c)] = total


@tack.kernel
def _contour_points(cells, u, levels, nlevels, per_cell, starts, points, edges, weights,
                    sources, keys, which):
    # A crossing's key holds its level: (low point * nlevels + level, high point),
    # or (within cells) (cell * nlevels + level, local edge), so each level's
    # points merge among themselves.
    for c in cells:
        cell = cells.entity_id(c)
        at = 3 * starts[cell]
        for v in range(nlevels):
            level = levels[v]
            case = _cell_case(cells, u, c, level)
            for k in range(3 * cells.contour_count(case)):
                e = cells.contour_edge(case, k)
                ja = cells.edge_corner(e, 0)
                jb = cells.edge_corner(e, 1)
                # From the end first by position (_first_end), so every cell and
                # every piece sharing the edge computes the same point, bit for bit,
                # when their values agree there.
                if not _first_end(cells.corner_position(c, ja), cells.corner_position(c, jb),
                                  cells.point_id(c, ja), cells.point_id(c, jb)):
                    ja, jb = jb, ja
                va = u.corner_value(c, ja)
                vb = u.corner_value(c, jb)
                w = _crossing(level, va, vb)
                xa = cells.corner_position(c, ja)
                xb = cells.corner_position(c, jb)
                a = cells.point_id(c, ja)
                b = cells.point_id(c, jb)
                sources[at] = cell
                which[at] = v
                # A crossing at a corner whose value is the level is that corner,
                # keyed by it, so every edge's crossing there merges into one point.
                hit = -1
                if va == level:
                    hit = ja
                elif vb == level:
                    hit = jb
                if hit >= 0:
                    p = cells.point_id(c, hit)
                    points[at] = cells.corner_position(c, hit)
                    edges[at] = [p, p]
                    weights[at] = 0.0
                    if per_cell == 1:
                        keys[at] = [cell * nlevels + v, _CORNER_KEY + hit]
                    else:
                        keys[at] = [p * nlevels + v, p]
                else:
                    points[at] = xa + w * (xb - xa)
                    edges[at] = [a, b]
                    weights[at] = w
                    if per_cell == 1:
                        keys[at] = [cell * nlevels + v, e]
                    else:
                        keys[at] = [min(a, b) * nlevels + v, max(a, b)]
                at += 1


# Within-cell keys of corner hits: (cell, _CORNER_KEY + corner), past every
# local edge number.
_CORNER_KEY = tack.constant(64, tack.i32)


@tack.func
def _same(p, q):
    return p[0] == q[0] and p[1] == q[1]


@tack.kernel
def _distinct_triangles(keys, keep):
    # A triangle two of whose points are one point (two hits at one corner, or a
    # hit and its edge's crossing there) has no area: dropped, as warp-simdata
    # drops it.
    for t in range(keep.shape[0]):
        p = keys[3 * t]
        q = keys[3 * t + 1]
        r = keys[3 * t + 2]
        keep[t] = 0 if _same(p, q) or _same(q, r) or _same(p, r) else 1


@tack.kernel
def _kept_triangles(keep, slots, points, edges, weights, sources, keys, which, out_points,
                    out_edges, out_weights, out_sources, out_keys, out_which):
    for t in range(keep.shape[0]):
        if keep[t] == 1:
            s = slots[t]
            for j in range(3):
                out_points[3 * s + j] = points[3 * t + j]
                out_edges[3 * s + j] = edges[3 * t + j]
                out_weights[3 * s + j] = weights[3 * t + j]
                out_sources[3 * s + j] = sources[3 * t + j]
                out_keys[3 * s + j] = keys[3 * t + j]
                out_which[3 * s + j] = which[3 * t + j]


@tack.kernel
def _first_of_runs(order, offsets, points, edges, weights, merged, merged_edges,
                   merged_weights, count):
    for r in range(count):
        first = order[offsets[r]]
        merged[r] = points[first]
        merged_edges[r] = edges[first]
        merged_weights[r] = weights[first]


@tack.kernel
def _run_of_each(order, offsets, run_of, count):
    for r in range(count):
        for i in range(offsets[r], offsets[r + 1]):
            run_of[order[i]] = r


@tack.kernel
def _triangle_cells(sources, which, out, out_which):
    for t in range(out.shape[0]):
        out[t] = sources[3 * t]
        out_which[t] = which[3 * t]


@tack.func
def _crosses(va, vb, iso):
    # One end at or above the level and the other below it (a NaN is below).
    return (1 if va >= iso else 0) != (1 if vb >= iso else 0)


@tack.func
def _emit_crossing(a, b, values, positions, iso, v, nlevels, e, at, keys, ends, weights,
                   sources, which):
    # The crossing of segment (a, b): named by its (low, high) points and the
    # level, or by the end whose value is the level, so every segment's crossing
    # there is one point; interpolated from the end first by position, as the
    # shape paths are.
    lo = min(a, b)
    hi = max(a, b)
    first = a if _first_end(positions[a], positions[b], a, b) else b
    second = b if first == a else a
    if values[lo] == iso:
        keys[at] = [lo * nlevels + v, lo]
        ends[at] = [lo, lo]
        weights[at] = 0.0
    elif values[hi] == iso:
        keys[at] = [hi * nlevels + v, hi]
        ends[at] = [hi, hi]
        weights[at] = 0.0
    else:
        keys[at] = [lo * nlevels + v, hi]
        ends[at] = [first, second]
        weights[at] = _crossing(iso, values[first], values[second])
    sources[at] = e
    which[at] = v


@tack.kernel
def _line_crossing_counts(cells, values, levels, nlevels, counts):
    for c in cells:
        a = cells.point_id(c, 0)
        b = cells.point_id(c, 1)
        k = 0
        for v in range(nlevels):
            if _crosses(values[a], values[b], levels[v]):
                k += 1
        counts[cells.entity_id(c)] = k


@tack.kernel
def _line_crossings(cells, values, positions, levels, nlevels, starts, keys, ends, weights,
                    sources, which):
    for c in cells:
        e = cells.entity_id(c)
        a = cells.point_id(c, 0)
        b = cells.point_id(c, 1)
        at = starts[e]
        for v in range(nlevels):
            if _crosses(values[a], values[b], levels[v]):
                _emit_crossing(a, b, values, positions, levels[v], v, nlevels, e, at, keys,
                               ends, weights, sources, which)
                at += 1


@tack.kernel
def _chain_crossing_counts(cells, values, levels, nlevels, counts):
    for c in cells:
        k = 0
        for j in range(cells.line_num_points(c) - 1):
            a = cells.line_point_id(c, j)
            b = cells.line_point_id(c, j + 1)
            for v in range(nlevels):
                if _crosses(values[a], values[b], levels[v]):
                    k += 1
        counts[cells.entity_id(c)] = k


@tack.kernel
def _chain_crossings(cells, values, positions, levels, nlevels, starts, keys, ends, weights,
                     sources, which):
    for c in cells:
        e = cells.entity_id(c)
        at = starts[e]
        for j in range(cells.line_num_points(c) - 1):
            a = cells.line_point_id(c, j)
            b = cells.line_point_id(c, j + 1)
            for v in range(nlevels):
                if _crosses(values[a], values[b], levels[v]):
                    _emit_crossing(a, b, values, positions, levels[v], v, nlevels, e, at,
                                   keys, ends, weights, sources, which)
                    at += 1


@tack.kernel
def _first_crossings(order, offsets, ends, weights, sources, which, out_ends, out_weights,
                     out_sources, out_which, count):
    for r in range(count):
        first = order[offsets[r]]
        out_ends[r] = ends[first]
        out_weights[r] = weights[first]
        out_sources[r] = sources[first]
        out_which[r] = which[first]


@tack.kernel
def _crossing_positions(positions, ends, weights, out):
    for i in range(out.shape[0]):
        end = ends[i]
        xa = positions[end[0]]
        out[i] = xa + weights[i] * (positions[end[1]] - xa)


def _contour_vertices(data, field, levels, nlevels, several, merge_points, fields):
    """A 1D mesh's contour: a vertex cell at each crossing of its segments -- lines'
    or polylines' -- one per distinct crossing with ``merge_points``."""
    if not isinstance(field.space, H1) or field.space.order != 1:
        raise NotImplementedError(f"contour of lines takes point data (H1 of order 1), "
                                  f"not {field.space!r}")
    if not isinstance(data.geometry.space, H1) or data.geometry.space.order != 1:
        raise NotImplementedError("contour of lines takes positions on the points")
    values = arrays.materialize(field.values)
    n_cells = data.num_cells
    idt = data.id_dtype
    kdt = ids.at_least(idt, data.num_points * nlevels)
    groups = [g for g in data.launch_groups("cells", []) if g.count and
              g.shape in (shapes.Line, shapes.PolyLine)]
    counts = tack.zeros((n_cells,), tack.i32)
    for group in groups:
        count_kernel = (_chain_crossing_counts if group.shape is shapes.PolyLine
                        else _line_crossing_counts)
        count_kernel(data.domain_view("cells", group), values, levels, nlevels, counts)
    odt = ids.for_output(data, n_cells * nlevels + 1)
    starts, n = _scanned(counts, n_cells, odt)
    keys = tack.empty((n,), tack.vec(2, kdt))
    ends = tack.empty((n,), tack.vec(2, idt))
    weights = tack.empty((n,), arrays.dtype_of(field.values))
    sources = tack.empty((n,), idt)
    which = tack.empty((n,), tack.i32)
    positions = arrays.materialize(data.geometry.values)
    for group in groups:
        emit = _chain_crossings if group.shape is shapes.PolyLine else _line_crossings
        emit(data.domain_view("cells", group), values, positions, levels, nlevels, starts,
             keys, ends, weights, sources, which)
    if merge_points and n:
        order = key_order(keys, data.num_points * nlevels, odt)
        offsets, count = run_offsets(keys, order)
        merged = (_like(ends, count), _like(weights, count), _like(sources, count),
                  _like(which, count))
        _first_crossings(order, offsets, ends, weights, sources, which, *merged, count)
        ends, weights, sources, which = merged
        n = count
    crossings = tack.empty((n,), tack.vec(3, data.dtype))
    if n:
        _crossing_positions(positions, ends, weights, crossings)
    types, offsets = uniform_cells(shapes.VERTEX, n, odt)
    points = UnstructuredTopology(types, offsets, CountingArray(n, dtype=odt), num_points=n)
    return _with_levels(carry(data, points, points=Interpolate(ends, weights),
                              cells=Pieces(sources), geometry=Field(H1(points), crossings),
                              fields=fields), levels, which, several, field.unit)


@tack.kernel
def _flag_cells(cells, flags):
    for c in cells:
        flags[cells.entity_id(c)] = 1


def _cells_of_dimension(data, field, dimension):
    """``data``'s cells of ``dimension`` alone, on the points they use, and
    ``field`` on them."""
    from tack.dataset.algorithms._extract import _keep_cells
    from tack.dataset.dataset import DataSet
    from tack.dataset.ghosts import keep_marks

    keep = tack.zeros((data.num_cells,), tack.i32)
    for group in data.topology.groups():
        if group.count and group.shape.DIMENSION == dimension:
            _flag_cells(group.view(), keep)
    name = "__tack_contour__"
    staged = keep_marks(data, DataSet(data.topology, data.geometry,
                                      fields={**data.fields, name: field},
                                      sets=dict(data.sets), **data._frame()))
    subset = _keep_cells(staged, keep)
    return subset, subset.fields.pop(name)


@on_inputs_device
def contour(data, field, value, merge_points=True, fields=None, dimension=None,
            subdivision=None, tolerance=None, exact=True):
    """Where ``field`` crosses ``value``: a surface of a 3D mesh, as triangles
    (as polygons on polyhedra); lines of a 2D mesh.

    ``value`` may be a sequence: the surfaces of every value, in one dataset
    made in one pass over the cells (each cell compared with every value),
    with each output cell's value in a cell field ``contour_value``. Points
    are not shared between values.

    ``field`` is a scalar field with a basis, or its name: ``H1`` of order 1
    (point data), or ``L2`` (DG, an order per cell too).

    On segments, quads, pixels, hexahedra and voxels, and on triangles,
    tetrahedra and wedges (into Freudenthal's sub-simplices, a wedge's in
    layers), a field above order 1 or a
    curved geometry is cut on linear sub-cells (``tack.dataset.subdivide``): each
    cell whose field's Bernstein coefficients can hold a value is cut into
    ``subdivision`` lattice steps along each axis -- by default the largest
    order of the field and geometry -- and the surface is the sub-cells'; a
    lattice point shared by cells is evaluated once, so the surface has no
    cracks. Each crossing of a sub-cell's edge is where the field itself takes
    the value along it (``exact``, the default), found from the field's own
    values on the edge, face or cell holding it, so a vertex lies on the true
    surface (to rounding) whatever the steps, and neighbouring cells and pieces
    put it at the same bits; with ``exact=False``, where the line between the
    lattice values does, as a linear contour of the sub-cells.
    ``subdivision=1`` with ``exact=False`` is the cut of the cells linearized at
    their corners. ``tolerance`` (in the field's unit) chooses the fewest steps whose
    estimated error is within it instead: what linearizing a cell loses is
    bounded by its Bernstein coefficients' distance from the corners'
    interpolation, divided by about the square of the steps; a curved
    geometry's own bend is not counted. Give one or the other. Elsewhere an order-2 ``H1`` field is refused: its cells would
    need subdividing, and their corner values alone are another, linear field. Each 3D cell contributes the
    triangles of VTK's case tables for its corner values, interpolated along
    its edges; in a mesh with 3D cells, cells of lower dimension contribute
    nothing, and a cell constant has no level set inside its cells.

    ``dimension`` (3, 2 or 1) chooses which cells to contour, by default the
    mesh's highest: the cells of that dimension alone, as a mesh of their own on
    the points they use (no cells of it, no contour), so a mesh of tetrahedra
    and the triangles of its boundary gives the tetrahedra's surface, or with
    ``dimension=2`` the triangles' lines. Cells of other dimensions are never
    mixed in.

    A 2D mesh -- triangles, quads, pixels or polygons -- gives line cells from
    its point values: each cell's crossings, found going round its edges, are
    joined in pairs, each line with the higher values on its right as the
    cell winds. Lines share their points through the mesh's faces (its edges),
    whatever ``merge_points`` says -- unless the field or the geometry is
    ``L2``: then each cell's lines have their own points, from its own values
    and corners. No cells give no lines.

    A 1D mesh -- lines or polylines -- gives a vertex cell at each crossing of
    its segments, from its point values, interpolated from the end first by position;
    a crossing at a point whose value is the level is that point, and with
    ``merge_points`` every crossing is one vertex, each taking the cell fields
    of the first cell to cross there.

    With ``merge_points``, triangles share their points: by global edge when
    the field and the geometry are both continuous, so the surface is
    watertight; otherwise only within each cell, so a DG field's surface
    keeps its jumps and an ``L2`` geometry's its gaps. ``H1`` fields carry
    onto the surface's points, interpolated along the same edges; cell
    fields (``Constant``, values on cells) onto its triangles, from the cell
    each came from.
    """
    value = _in_unit_of(_field(data, field), value, "contour's value")
    from tack.dataset.ghosts import visible

    data, field = visible(data, field)
    field = _field(data, field)
    space = field.space
    from tack.dataset.algorithms.subdivide import subdivided_contour, subdivides

    if tolerance is not None:
        tolerance = _in_unit_of(field, tolerance, "contour's tolerance")
    # Polyhedra, polygons and polylines have no lattice to subdivide: linear,
    # they are cut as they are (as threshold and clip cut them).
    if (dimension is None or dimension == data.topology.dimension) and getattr(
            data.topology, "reference_cells", True) and (
            subdivides(data, field) or (subdivision is not None and subdivision > 1)
            or tolerance is not None):
        _scalar_floating("contour", field)
        levels, nlevels, _ = _levels(value, field)
        return subdivided_contour(
            data, field, value, levels, nlevels, subdivision,
            lambda refined, name: contour(refined, name, value, merge_points=merge_points,
                                          fields=fields, dimension=dimension),
            tolerance=tolerance, exact=exact)
    if isinstance(space, H1) and space.order != 1:
        from tack.dataset.algorithms.subdivide import _cell_family

        # A mesh subdivision does not take (mixed families, pyramids) says so.
        _cell_family(data, f"contour of an order-{space.order} field (by subdivision)")
        raise NotImplementedError(
            f"contour of an order-{space.order} H1 field is not built: it would subdivide "
            "each cell, and its corner values alone give the contour of a linear field")
    _scalar_floating("contour", field)
    levels, nlevels, several = _levels(value, field)
    top = data.topology.dimension
    if dimension is None:
        dimension = top if top >= 1 else 3       # no cells (or vertices): no surface
    if dimension not in (1, 2, 3):
        raise ValueError(f"contour's dimension is 3, 2 or 1 (the cells to contour), "
                         f"not {dimension!r}")
    mixed = any(g.count and g.shape.DIMENSION != dimension for g in data.topology.groups())
    if mixed or dimension > top:
        data, field = _cells_of_dimension(data, field, dimension)
    if dimension == 2:
        return _contour_lines(data, field, levels, nlevels, several, fields)
    if dimension == 1:
        return _contour_vertices(data, field, levels, nlevels, several, merge_points, fields)
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_contour(data, field, levels, nlevels, several, fields)
    field = _with_basis(data, field)
    space = field.space
    if not space.interpolated or space.on not in ("cells", "points"):
        raise TypeError(f"contour needs a field with a basis on the cells, not {space!r}")
    per_cell = 0 if isinstance(space, H1) and isinstance(data.geometry.space, H1) else 1
    groups = [g for g in data.launch_groups("cells", [field])
              if g.count and g.shape.CONTOUR_TRIANGLES]
    n_cells = data.num_cells
    idt = data.id_dtype
    odt = ids.for_output(data, 15 * n_cells * nlevels)     # up to five triangles a cell
    # Keys hold the level beside a point or cell id.
    kdt = ids.at_least(idt, (n_cells if per_cell else data.num_points) * nlevels)
    counts = tack.zeros((n_cells,), tack.i32)
    for group in groups:
        _contour_counts(data.domain_view("cells", group), field.view(group), levels, nlevels,
                        counts)
    starts, triangles = _scanned(counts, n_cells, odt)

    n = 3 * triangles
    points = tack.empty((n,), tack.vec(3, data.dtype))
    edges = tack.empty((n,), tack.vec(2, idt))
    weights = tack.empty((n,), arrays.dtype_of(field.values))
    sources = tack.empty((n,), idt)
    keys = tack.empty((n,), tack.vec(2, kdt))
    which = tack.empty((n,), tack.i32)
    for group in groups:
        _contour_points(data.domain_view("cells", group), field.view(group), levels, nlevels,
                        per_cell, starts, points, edges, weights, sources, keys, which)
    if triangles:
        keep = tack.empty((triangles,), tack.i32)
        _distinct_triangles(keys, keep)
        slots = tack.empty((triangles,), odt)
        kept = exclusive_scan(keep, slots, triangles)
        if kept < triangles:
            arrays_in = (points, edges, weights, sources, keys, which)
            arrays_out = tuple(_like(a, 3 * kept) for a in arrays_in)
            _kept_triangles(keep, slots, *arrays_in, *arrays_out)
            points, edges, weights, sources, keys, which = arrays_out
            triangles, n = kept, 3 * kept
    triangle_cells = tack.empty((triangles,), idt)
    triangle_levels = tack.empty((triangles,), tack.i32)
    if triangles:
        _triangle_cells(sources, which, triangle_cells, triangle_levels)

    if merge_points and n:
        # A key is the edge's (low, high) points, or (within cells) (cell, local edge),
        # its first part times the levels plus the level.
        order = key_order(keys, (n_cells if per_cell else data.num_points) * nlevels,
                                odt)
        offsets, count = run_offsets(keys, order)
        merged = tack.empty((count,), tack.vec(3, data.dtype))
        merged_edges = tack.empty((count,), tack.vec(2, idt))
        merged_weights = tack.empty((count,), weights.dtype)
        _first_of_runs(order, offsets, points, edges, weights, merged, merged_edges,
                       merged_weights, count)
        rows = tack.empty((n,), odt)
        _run_of_each(order, offsets, rows, count)
        points, edges, weights = merged, merged_edges, merged_weights
    else:
        rows, count = CountingArray(n, dtype=odt), n          # each triangle its own points

    # Every cell a triangle, three points each: the types and offsets are
    # implicit, and the connectivity is the triangles' rows of merged points.
    types, tri_offsets = uniform_cells(shapes.TRIANGLE, triangles, odt)
    surface = UnstructuredTopology(types, tri_offsets, rows, num_points=count)
    return _with_levels(carry(data, surface, points=Interpolate(edges, weights),
                              cells=Pieces(triangle_cells), geometry=Field(H1(surface), points),
                              fields=fields), levels, triangle_levels, several,
                        field.unit)


@tack.func
def _value_range(cells, c, values):
    # The cell's least and greatest values, and whether one is NaN: enough to say
    # which levels it straddles without walking its points again for each.
    lo = values[cells.point_id(c, 0)]
    hi = lo
    nan = 0
    for j in range(cells.num_points(c)):
        x = values[cells.point_id(c, j)]
        lo = min(lo, x)                     # a number is preferred to NaN
        hi = max(hi, x)
        if x != x:
            nan = 1
    return lo, hi, nan


@tack.func
def _straddles(lo, hi, nan, iso):
    # Whether the cell has points on both sides of the level: at or above it, and
    # not (below it, or NaN). Most do not, and need not walk their faces.
    return hi >= iso and (lo < iso or nan == 1)


@tack.func
def _straddles_any(lo, hi, nan, levels, nlevels):
    found = 0
    for v in range(nlevels):
        if _straddles(lo, hi, nan, levels[v]):
            found = 1
            break
    return found == 1


@tack.kernel
def _lopez_counts(cells, values, levels, nlevels, counts):
    # Each face's rising crossings, of every level; cells on one side of every
    # level keep their faces' zeros.
    for c in cells:
        lo, hi, nan = _value_range(cells, c, values)
        if not _straddles_any(lo, hi, nan, levels, nlevels):
            continue
        for k in range(cells.num_faces(c)):
            n = cells.side_num_points(c, k)
            keyed = 0
            for v in range(nlevels):
                iso = levels[v]
                if _straddles(lo, hi, nan, iso):
                    for j in range(n):
                        va = values[cells.side_point_id(c, k, j)]
                        vb = values[cells.side_point_id(c, k, j + 1 if j + 1 < n else 0)]
                        if not va >= iso and vb >= iso:     # below: not at or above (NaN too)
                            keyed += 1
            counts[cells.entry(c, k)] = keyed


@tack.func
def _edge_key(a, b, v, nlevels):
    # An edge's crossing of level v: (low point * nlevels + v, high point).
    return [min(a, b) * nlevels + v, max(a, b)]


@tack.kernel
def _lopez_pairs(cells, values, levels, nlevels, starts, pair_from, pair_to):
    for c in cells:
        lo, hi, nan = _value_range(cells, c, values)
        if not _straddles_any(lo, hi, nan, levels, nlevels):
            continue
        for k in range(cells.num_faces(c)):
            n = cells.side_num_points(c, k)
            at = starts[cells.entry(c, k)]
            for v in range(nlevels):
                iso = levels[v]
                if _straddles(lo, hi, nan, iso):
                    at = _face_pairs(cells, c, k, n, values, iso, v, nlevels, at, pair_from,
                                     pair_to)


@tack.func
def _face_pairs(cells, c, k, n, values, iso, v, nlevels, at, pair_from, pair_to):
    # Face k's rising crossings of level v, each paired with the next crossing
    # going round; returns the next free slot.
    for j in range(n):
        va = values[cells.side_point_id(c, k, j)]
        vb = values[cells.side_point_id(c, k, j + 1 if j + 1 < n else 0)]
        if not va >= iso and vb >= iso:     # below: not at or above (NaN too)
            nxt = j
            for m in range(1, n):
                j2 = j + m if j + m < n else j + m - n
                wa = values[cells.side_point_id(c, k, j2)]
                wb = values[cells.side_point_id(c, k, j2 + 1 if j2 + 1 < n else 0)]
                if (1 if wa >= iso else 0) != (1 if wb >= iso else 0):
                    nxt = j2
                    break
            pair_from[at] = _edge_key(
                cells.side_point_id(c, k, j), cells.side_point_id(c, k, j + 1 if j + 1 < n else 0),
                v, nlevels)
            pair_to[at] = _edge_key(
                cells.side_point_id(c, k, nxt),
                cells.side_point_id(c, k, nxt + 1 if nxt + 1 < n else 0), v, nlevels)
            at += 1
    return at


@tack.kernel
def _lopez_trace(cells, pair_offsets, pair_from, pair_to, emit, poly_counts, poly_offsets,
                 poly_starts, poly_cells, poly_levels, vertices, used, nlevels, bad):
    # used: one flag per pair, zero on entry (a cell's pairs, of every level, can
    # pass any scratch size). A pair's key holds its level, so a polygon's pairs
    # are of one level.
    for c in cells:
        e = cells.entity_id(c)
        first = pair_offsets[cells.entry(c, 0)]
        m = pair_offsets[cells.entry(c, cells.num_faces(c))] - first
        polys = 0
        out = first
        for s in range(m):
            if used[first + s] == 0:
                if emit == 1:
                    poly_starts[poly_offsets[e] + polys] = out
                    poly_cells[poly_offsets[e] + polys] = e
                    poly_levels[poly_offsets[e] + polys] = tack.i32(pair_from[first + s][0]
                                                                    % nlevels)
                cur = s
                for step in range(m):
                    used[first + cur] = 1
                    if emit == 1:
                        vertices[out] = first + cur          # the crossing, by its pair
                    out += 1
                    target = pair_to[first + cur]
                    nxt = -1
                    for i in range(m):
                        other = pair_from[first + i]
                        if other[0] == target[0] and other[1] == target[1]:
                            nxt = i
                    if nxt == s:
                        break
                    if nxt < 0 or used[first + nxt] == 1:
                        tack.atomic_add(bad, 0, 1)
                        break
                    cur = nxt
                polys += 1
        if emit == 0:
            poly_counts[e] = polys


@tack.kernel
def _crossing_names(pair_from, values, levels, nlevels, names):
    # A crossing's name is its edge, or the point at an end whose value is the
    # level, so every edge's crossing at such a point is one iso-vertex, as the
    # shape path's contour names it; with the level, as its key holds it:
    # (point * nlevels + level, other point).
    for i in range(names.shape[0]):
        key = pair_from[i]
        v = key[0] % nlevels
        a = key[0] // nlevels
        b = key[1]
        iso = levels[v]
        if values[a] == iso:
            names[i] = [a * nlevels + v, a]
        elif values[b] == iso:
            names[i] = [b * nlevels + v, b]
        else:
            names[i] = key


@tack.kernel
def _iso_vertices(names, order, offsets, values, levels, nlevels, positions, weights, out,
                  ends, vertex_of_pair, count):
    # One per run of equal crossing names, interpolated from the end first by
    # position, as the shape path's contour interpolates, so the two agree on
    # every point, and so do pieces.
    for r in range(count):
        key = names[order[offsets[r]]]
        v = key[0] % nlevels
        a = key[0] // nlevels
        b = key[1]
        if a != b and not _first_end(positions[a], positions[b], a, b):
            a, b = b, a
        if a == b:
            out[r] = positions[a]
            weights[r] = 0.0
        else:
            w = _crossing(levels[v], values[a], values[b])
            xa = positions[a]
            out[r] = xa + w * (positions[b] - xa)
            weights[r] = w
        ends[r] = [a, b]
        for i in range(offsets[r], offsets[r + 1]):
            vertex_of_pair[order[i]] = r


@tack.kernel
def _distinct_sizes(poly_starts, vertices, sizes, n):
    # Each polygon's vertices without a repeat of the one before it (going round):
    # two crossings at one point become one vertex. Fewer than three: dropped.
    for g in range(n):
        begin = poly_starts[g]
        end = poly_starts[g + 1]
        kept = 0
        for i in range(begin, end):
            prev = vertices[end - 1] if i == begin else vertices[i - 1]
            if vertices[i] != prev:
                kept += 1
        if kept == 0 and end > begin:
            kept = 1
        sizes[g] = kept if kept >= 3 else 0


@tack.kernel
def _distinct_vertices(poly_starts, vertices, sizes, out_starts, out_vertices, poly_cells,
                       out_cells, poly_levels, out_levels, slots, n):
    for g in range(n):
        if sizes[g] > 0:
            begin = poly_starts[g]
            end = poly_starts[g + 1]
            at = out_starts[g]
            for i in range(begin, end):
                prev = vertices[end - 1] if i == begin else vertices[i - 1]
                if vertices[i] != prev:
                    out_vertices[at] = vertices[i]
                    at += 1
            out_cells[slots[g]] = poly_cells[g]
            out_levels[slots[g]] = poly_levels[g]


@tack.kernel
def _nonzero(sizes, keep):
    for g in range(keep.shape[0]):
        keep[g] = 1 if sizes[g] > 0 else 0


@tack.kernel
def _kept_starts(out_starts, keep, slots, starts, n):
    for g in range(n):
        if keep[g] == 1:
            starts[slots[g]] = out_starts[g]


@tack.kernel
def _renumber_vertices(vertices, vertex_of_pair):
    for i in range(vertices.shape[0]):
        vertices[i] = vertex_of_pair[vertices[i]]


def _lopez_polygons(data, values, levels, nlevels, weight_dtype):
    """A polyhedral topology's iso-polygons of point ``values`` at ``levels``, by López's
    tracing: ``(poly_starts, vertices, poly_cells, poly_levels, polygons, ends,
    weights, positions, count)`` -- each polygon's iso-vertices (CSR, in the
    order traced), its cell and level, and per iso-vertex the points it lies
    between (``ends``: ``(a, a)`` at a point whose value is the level, else the
    edge's end first by position, ``_first_end``), its weight from ``a`` and its
    position.
    Polygons come in cell order."""
    from tack.dataset.polyhedra import SizeBuckets

    t = data.topology
    entries = size_of(t.cell_faces)
    idt = data.id_dtype
    # A face's crossings are at most its points; every face is in two cells at most.
    odt = ids.for_output(data, 2 * size_of(t.face_points) * nlevels)
    kdt = ids.at_least(idt, data.num_points * nlevels)      # keys hold the level
    counts = tack.zeros((entries,), tack.i32)
    for group in data.launch_groups("cells", []):
        if group.count:
            _lopez_counts(data.domain_view("cells", group), values, levels, nlevels, counts)
    pair_offsets = tack.empty((entries + 1,), odt)
    pairs = exclusive_scan(counts, pair_offsets, entries, close=True)
    pair_from = tack.empty((pairs,), tack.vec(2, kdt))
    pair_to = tack.empty((pairs,), tack.vec(2, kdt))
    if pairs:
        for group in data.launch_groups("cells", []):
            if group.count:
                _lopez_pairs(data.domain_view("cells", group), values, levels, nlevels,
                             pair_offsets, pair_from, pair_to)
    n = data.num_cells
    poly_counts = tack.zeros((n,), tack.i32)
    poly_offsets = tack.empty((n + 1,), odt)
    bad = tack.zeros((1,), tack.i32)
    buckets = SizeBuckets(t)
    traced = [g for g in data.launch_groups("cells", [], keys=[buckets]) if g.count]
    dummy = tack.empty((1,), odt)
    dummy_levels = tack.empty((1,), tack.i32)
    used = tack.zeros((max(pairs, 1),), tack.i32)
    if pairs:
        for group in traced:
            _lopez_trace(data.domain_view("cells", group), pair_offsets, pair_from, pair_to,
                         0, poly_counts, poly_offsets, dummy, dummy, dummy_levels, dummy, used,
                         nlevels, bad)
    polygons = exclusive_scan(poly_counts, poly_offsets, n) if n and pairs else 0
    _close(poly_offsets, n, polygons)
    poly_starts = tack.empty((polygons + 1,), odt)
    poly_cells = tack.empty((polygons,), idt)
    poly_levels = tack.empty((polygons,), tack.i32)
    vertices = tack.empty((pairs,), odt)
    if polygons:
        used.fill(0)
        for group in traced:
            _lopez_trace(data.domain_view("cells", group), pair_offsets, pair_from, pair_to,
                         1, poly_counts, poly_offsets, poly_starts, poly_cells, poly_levels,
                         vertices, used, nlevels, bad)
    _close(poly_starts, polygons, pairs)
    if bad[0]:
        raise ValueError(f"{bad[0]} iso-polygons do not close: the cells' faces are not "
                         "wound consistently (check_winding)")

    # Each crossing is named by its edge in every cell around that edge, or by its
    # point where the value there is the isovalue: one iso-vertex per run of
    # equal names.
    names = tack.empty((pairs,), tack.vec(2, kdt))
    if pairs:
        _crossing_names(pair_from, values, levels, nlevels, names)
    order = key_order(names, data.num_points * nlevels, odt)
    offsets, count = run_offsets(names, order)
    positions = tack.empty((count,), tack.vec(3, data.dtype))
    ends = tack.empty((count,), tack.vec(2, idt))
    weights = tack.empty((count,), weight_dtype)
    vertex_of_pair = tack.empty((pairs,), odt)
    if count:
        _iso_vertices(names, order, offsets, values, levels, nlevels,
                      arrays.materialize(data.geometry.values), weights, positions, ends,
                      vertex_of_pair, count)
    if pairs:
        _renumber_vertices(vertices, vertex_of_pair)
    if polygons:
        # A polygon through such a point may name it twice in a row: once.
        sizes = tack.empty((polygons,), tack.i32)
        _distinct_sizes(poly_starts, vertices, sizes, polygons)
        out_starts = tack.empty((polygons + 1,), odt)
        total = exclusive_scan(sizes, out_starts, polygons)
        if total < pairs:
            keep = tack.empty((polygons,), tack.i32)
            _nonzero(sizes, keep)
            slots = tack.empty((polygons,), odt)
            kept = exclusive_scan(keep, slots, polygons)
            out_vertices = tack.empty((total,), odt)
            out_cells = tack.empty((kept,), idt)
            out_levels = tack.empty((kept,), tack.i32)
            _distinct_vertices(poly_starts, vertices, sizes, out_starts, out_vertices,
                               poly_cells, out_cells, poly_levels, out_levels, slots, polygons)
            starts = tack.empty((kept + 1,), odt)
            if kept:
                _kept_starts(out_starts, keep, slots, starts, polygons)
            _close(starts, kept, total)
            poly_starts, vertices, poly_cells, polygons = starts, out_vertices, out_cells, kept
            poly_levels = out_levels
            # Iso-vertices only dropped polygons used are left over, unused.
    return (poly_starts, vertices, poly_cells, poly_levels, polygons, ends, weights,
            positions, count)


def _polyhedral_contour(data, field, levels, nlevels, several, fields=None):
    """``contour`` of a polyhedral topology: iso-polygons by López's tracing, as a
    ``PolygonalTopology`` -- polygons, not triangles."""
    from tack.dataset.polyhedra import PolygonalTopology

    if not (isinstance(field.space, H1) or field.space is Values(data, "points")):
        raise TypeError("a polyhedral topology contours point data")
    values = arrays.materialize(field.values)
    (poly_starts, vertices, poly_cells, poly_levels, polygons, ends, weights, positions,
     count) = _lopez_polygons(data, values, levels, nlevels, arrays.dtype_of(field.values))
    surface = PolygonalTopology(poly_starts, vertices, num_points=count)
    return _with_levels(carry(data, surface, points=Interpolate(ends, weights),
                              cells=Pieces(poly_cells), geometry=Field(H1(surface), positions),
                              fields=fields), levels, poly_levels, several,
                        field.unit)


# ── Lines of 2D meshes ──────────────────────────────────────────────
#
# Several levels: a face's crossing of level v is slot f * nlevels + v of the
# per-face arrays, and a crossing at point p of level v is slot
# nfv + p * nlevels + v of the points' (nfv the faces' slots), so each level's
# points are its own.

@tack.kernel
def _line_counts(cells, values, levels, nlevels, counts):
    # A 2D cell's crossings of each level, going round its edges: half as many lines.
    for c in cells:
        total = 0
        for v in range(nlevels):
            iso = levels[v]
            crossings = 0
            for k in range(cells.num_faces(c)):
                above_a = values[cells.side_point_id(c, k, 0)] >= iso
                above_b = values[cells.side_point_id(c, k, 1)] >= iso
                if above_a != above_b:
                    crossings += 1
            total += crossings // 2
        counts[cells.entity_id(c)] = total


@tack.func
def _level_lines(cells, c, values, iso, v, nlevels, at, line_faces, line_cells, line_levels,
                 crossed, ends, weights, positions, hits):
    # Level v's lines of cell c from slot at; returns the next free slot. Going
    # round the cell from its first rising crossing, each rising crossing (below
    # to above) pairs with the next falling one: a line cutting off a run of
    # points above the level, which lies on its right.
    n = cells.num_faces(c)
    first = -1
    for k in range(n):
        if first < 0:
            if (not values[cells.side_point_id(c, k, 0)] >= iso       # below (NaN too)
                    and values[cells.side_point_id(c, k, 1)] >= iso):
                first = k
    if first >= 0:
        rising = 0
        for i in range(n):
            k = (first + i) % n
            a = cells.side_point_id(c, k, 0)
            b = cells.side_point_id(c, k, 1)
            above_a = values[a] >= iso
            above_b = values[b] >= iso
            if above_a != above_b:
                f = cells.face_id(c, k) * nlevels + v
                if not above_a:
                    rising = f
                else:
                    line_faces[at] = [rising, f]
                    line_cells[at] = cells.entity_id(c)
                    line_levels[at] = v
                    at += 1
                if cells.face_side(c, k) == 0:
                    # The face's side 0 records its iso-point, from its lower point
                    # id, as every contour interpolates.
                    lo = 0 if a < b else 1
                    p = cells.side_point_id(c, k, lo)
                    q = cells.side_point_id(c, k, 1 - lo)
                    w = _crossing(iso, values[p], values[q])
                    xp = cells.side_position(c, k, lo)
                    crossed[f] = 1
                    ends[f] = [p, q]
                    weights[f] = w
                    positions[f] = xp + w * (cells.side_position(c, k, 1 - lo) - xp)
                    # At an end whose value is the level the crossing is that
                    # point, shared by every face crossing there.
                    hits[f] = p if values[p] == iso else (q if values[q] == iso else -1)
    return at


@tack.kernel
def _line_pieces(cells, values, levels, nlevels, starts, line_faces, line_cells, line_levels,
                 crossed, ends, weights, positions, hits):
    for c in cells:
        at = starts[cells.entity_id(c)]
        for v in range(nlevels):
            at = _level_lines(cells, c, values, levels[v], v, nlevels, at, line_faces,
                              line_cells, line_levels, crossed, ends, weights, positions, hits)


@tack.kernel
def _distinct_lines(line_faces, hits, keep):
    # A line whose two crossings are one point (both at a point whose value is the
    # level) has no length: dropped.
    for i in range(keep.shape[0]):
        pair = line_faces[i]
        h = hits[pair[0]]
        keep[i] = 0 if h >= 0 and h == hits[pair[1]] else 1


@tack.kernel
def _kept_lines(keep, slots, line_faces, line_cells, line_levels, out_faces, out_cells,
                out_levels):
    for i in range(keep.shape[0]):
        if keep[i] == 1:
            out_faces[slots[i]] = line_faces[i]
            out_cells[slots[i]] = line_cells[i]
            out_levels[slots[i]] = line_levels[i]


@tack.func
def _point_slot(fv, hits, nfv, nlevels):
    # The point a face's crossing is: its own slot, or its point's, of its level.
    h = hits[fv]
    return fv if h < 0 else nfv + h * nlevels + fv % nlevels


@tack.kernel
def _used_points(line_faces, hits, used, nfv, nlevels):
    for i in range(line_faces.shape[0]):
        pair = line_faces[i]
        for j in range(2):
            used[_point_slot(pair[j], hits, nfv, nlevels)] = 1


@tack.kernel
def _iso_points(kept, ends, weights, positions, geometry, nfv, nlevels, out_ends,
                out_weights, out_positions):
    for i in range(kept.shape[0]):
        k = kept[i]
        if k < nfv:
            out_ends[i] = ends[k]
            out_weights[i] = weights[k]
            out_positions[i] = positions[k]
        else:
            p = (k - nfv) // nlevels
            out_ends[i] = [p, p]
            out_weights[i] = 0.0
            out_positions[i] = geometry[p]


@tack.kernel
def _line_cells(line_faces, hits, point_of, nfv, nlevels, connectivity, n):
    for i in range(n):
        pair = line_faces[i]
        for j in range(2):
            connectivity[2 * i + j] = point_of[_point_slot(pair[j], hits, nfv, nlevels)]


@tack.kernel
def _cell_line_counts(cells, u, levels, nlevels, counts):
    # As _line_counts, from the cell's own corner values.
    for c in cells:
        total = 0
        for v in range(nlevels):
            iso = levels[v]
            crossings = 0
            for k in range(cells.NUM_FACES):
                above_a = u.corner_value(c, cells.face_corner(k, 0)) >= iso
                above_b = u.corner_value(c, cells.face_corner(k, 1)) >= iso
                if above_a != above_b:
                    crossings += 1
            total += crossings // 2
        counts[cells.entity_id(c)] = total


@tack.func
def _level_cell_lines(cells, u, c, iso, v, at, line_cells, line_levels, ends, weights,
                      positions):
    # As _level_lines, but each line has its own two points, from its cell's own
    # values and corners: line i's are points 2i (rising) and 2i + 1 (falling).
    n = cells.NUM_FACES
    first = -1
    for k in range(n):
        if first < 0:
            if (not u.corner_value(c, cells.face_corner(k, 0)) >= iso       # below (NaN too)
                    and u.corner_value(c, cells.face_corner(k, 1)) >= iso):
                first = k
    if first >= 0:
        for i in range(n):
            k = (first + i) % n
            ja = cells.face_corner(k, 0)
            jb = cells.face_corner(k, 1)
            above_a = u.corner_value(c, ja) >= iso
            if above_a != (u.corner_value(c, jb) >= iso):
                # From the end first by position, as every contour interpolates.
                lo = ja
                hi = jb
                if not _first_end(cells.corner_position(c, ja), cells.corner_position(c, jb),
                                  cells.point_id(c, ja), cells.point_id(c, jb)):
                    lo = jb
                    hi = ja
                vl = u.corner_value(c, lo)
                vh = u.corner_value(c, hi)
                w = _crossing(iso, vl, vh)
                x = cells.corner_position(c, lo)
                slot = 2 * at + (1 if above_a else 0)
                # At a corner whose value is the level the crossing is the corner.
                if vl == iso or vh == iso:
                    j = lo if vl == iso else hi
                    ends[slot] = [cells.point_id(c, j), cells.point_id(c, j)]
                    weights[slot] = 0.0
                    positions[slot] = cells.corner_position(c, j)
                else:
                    ends[slot] = [cells.point_id(c, lo), cells.point_id(c, hi)]
                    weights[slot] = w
                    positions[slot] = x + w * (cells.corner_position(c, hi) - x)
                if above_a:
                    line_cells[at] = cells.entity_id(c)
                    line_levels[at] = v
                    at += 1
    return at


@tack.kernel
def _cell_line_points(cells, u, levels, nlevels, starts, line_cells, line_levels, ends,
                      weights, positions):
    for c in cells:
        at = starts[cells.entity_id(c)]
        for v in range(nlevels):
            at = _level_cell_lines(cells, u, c, levels[v], v, at, line_cells, line_levels,
                                   ends, weights, positions)


@tack.kernel
def _distinct_cell_lines(ends, keep):
    # A line both of whose points are one corner has no length: dropped.
    for i in range(keep.shape[0]):
        p = ends[2 * i]
        q = ends[2 * i + 1]
        keep[i] = 0 if p[0] == p[1] and _same(p, q) else 1


@tack.kernel
def _kept_cell_lines(keep, slots, line_cells, line_levels, ends, weights, positions,
                     out_cells, out_levels, out_ends, out_weights, out_positions):
    for i in range(keep.shape[0]):
        if keep[i] == 1:
            s = slots[i]
            out_cells[s] = line_cells[i]
            out_levels[s] = line_levels[i]
            for j in range(2):
                out_ends[2 * s + j] = ends[2 * i + j]
                out_weights[2 * s + j] = weights[2 * i + j]
                out_positions[2 * s + j] = positions[2 * i + j]


def _contour_cell_lines(data, field, levels, nlevels, several, fields=None):
    """``contour`` of a 2D shape mesh whose field or geometry is per cell (``L2``):
    each cell's lines from its own values and corners, their points not shared,
    so a DG field keeps its jumps and an ``L2`` geometry its gaps."""
    if not field.space.interpolated or field.space.on not in ("cells", "points"):
        raise TypeError(f"contour needs a field with a basis on the cells, not "
                        f"{field.space!r}")
    groups = [g for g in data.launch_groups("cells", [field])
              if g.count and g.shape.DIMENSION == 2]
    n = data.num_cells
    idt = data.id_dtype
    counts = tack.zeros((n,), tack.i32)
    for group in groups:
        _cell_line_counts(data.domain_view("cells", group), field.view(group), levels, nlevels,
                          counts)
    odt = ids.for_output(data, 4 * n * nlevels)  # at most two lines, four points, a cell
    starts, lines = _scanned(counts, n, odt)
    line_cells = tack.empty((lines,), idt)
    line_levels = tack.empty((lines,), tack.i32)
    ends = tack.empty((2 * lines,), tack.vec(2, idt))
    weights = tack.empty((2 * lines,), arrays.dtype_of(field.values))
    positions = tack.empty((2 * lines,), tack.vec(3, data.dtype))
    if lines:
        for group in groups:
            _cell_line_points(data.domain_view("cells", group), field.view(group), levels,
                              nlevels, starts, line_cells, line_levels, ends, weights,
                              positions)
        keep = tack.empty((lines,), tack.i32)
        _distinct_cell_lines(ends, keep)
        slots = tack.empty((lines,), odt)
        kept = exclusive_scan(keep, slots, lines)
        if kept < lines:
            arrays_in = (line_cells, line_levels, ends, weights, positions)
            arrays_out = (_like(line_cells, kept), _like(line_levels, kept),
                          *(_like(a, 2 * kept) for a in arrays_in[2:]))
            _kept_cell_lines(keep, slots, *arrays_in, *arrays_out)
            line_cells, line_levels, ends, weights, positions = arrays_out
            lines = kept
    # Each line its own two points: everything about the cells is implicit.
    types, offsets = uniform_cells(shapes.LINE, lines, odt)
    out = UnstructuredTopology(types, offsets, CountingArray(2 * lines, dtype=odt),
                               num_points=2 * lines)
    return _with_levels(carry(data, out, points=Interpolate(ends, weights),
                              cells=Pieces(line_cells), geometry=Field(H1(out), positions),
                              fields=fields), levels, line_levels, several,
                        field.unit)


def _contour_lines(data, field, levels, nlevels, several, fields=None):
    """``contour`` of a 2D mesh -- shapes or polygons -- as line cells: each cell's
    crossings found going round its edges, which are the mesh's faces, so lines
    share their points through them. A field or geometry per cell (``L2``) gives
    each cell its own lines' points instead."""
    on_points = isinstance(field.space, H1) or field.space is Values(data, "points")
    if not on_points or not isinstance(data.geometry.space, H1):
        if isinstance(field.space, Values):
            if field.space.on not in ("points", "cells"):
                raise TypeError("contour needs a field with a basis on the cells")
            from tack.dataset.spaces import Constant

            field = Field((H1 if field.space.on == "points" else Constant)(data), field.values,
                          field.unit)
        if not getattr(data.topology, "reference_cells", True):
            raise TypeError("contour of polygons takes point data")
        return _contour_cell_lines(data, field, levels, nlevels, several, fields)
    values = arrays.materialize(field.values)
    faces = data.topology.faces()
    nf = faces.num_faces
    nfv = nf * nlevels
    idt = data.id_dtype
    n = data.num_cells
    groups = [g for g in data.launch_groups("cells", [])
              if g.count and g.shape.DIMENSION == 2]
    counts = tack.zeros((n,), tack.i32)
    for group in groups:
        _line_counts(data.domain_view("cells", group), values, levels, nlevels, counts)
    np_ = data.num_points
    odt = ids.for_output(data, (2 * nf + np_) * nlevels)
    sdt = ids.at_least(idt, nfv + np_ * nlevels)            # face-and-level slots
    starts, lines = _scanned(counts, n, odt)
    line_faces = tack.empty((lines,), tack.vec(2, sdt))
    line_cells = tack.empty((lines,), idt)
    line_levels = tack.empty((lines,), tack.i32)
    crossed = tack.zeros((nfv,), tack.i32)
    ends = tack.empty((nfv,), tack.vec(2, idt))
    weights = tack.empty((nfv,), values.dtype)
    positions = tack.empty((nfv,), tack.vec(3, data.dtype))
    hits = tack.empty((nfv,), idt)
    if lines:
        for group in groups:
            _line_pieces(data.domain_view("cells", group), values, levels, nlevels, starts,
                         line_faces, line_cells, line_levels, crossed, ends, weights,
                         positions, hits)
        keep = tack.empty((lines,), tack.i32)
        _distinct_lines(line_faces, hits, keep)
        slots = tack.empty((lines,), odt)
        kept_lines = exclusive_scan(keep, slots, lines)
        if kept_lines < lines:
            out_faces = tack.empty((kept_lines,), tack.vec(2, sdt))
            out_cells = tack.empty((kept_lines,), idt)
            out_levels = tack.empty((kept_lines,), tack.i32)
            _kept_lines(keep, slots, line_faces, line_cells, line_levels, out_faces,
                        out_cells, out_levels)
            line_faces, line_cells, line_levels, lines = (out_faces, out_cells, out_levels,
                                                          kept_lines)
    # The points: a face's own crossing, or the point it crosses at, shared by
    # every face crossing there -- each level's its own -- those the kept lines
    # use, in order.
    used = tack.zeros((nfv + np_ * nlevels,), tack.i32)
    if lines:
        _used_points(line_faces, hits, used, nfv, nlevels)
    point_of, kept, count = _compact(used, odt)
    out_ends = tack.empty((count,), tack.vec(2, idt))
    out_weights = tack.empty((count,), values.dtype)
    out_positions = tack.empty((count,), tack.vec(3, data.dtype))
    if count:
        _iso_points(kept, ends, weights, positions, arrays.materialize(data.geometry.values),
                    nfv, nlevels, out_ends, out_weights, out_positions)
    types, offsets = uniform_cells(shapes.LINE, lines, odt)
    connectivity = tack.empty((2 * lines,), odt)
    if lines:
        _line_cells(line_faces, hits, point_of, nfv, nlevels, connectivity, lines)
    out = UnstructuredTopology(types, offsets, connectivity, num_points=count)
    return _with_levels(carry(data, out, points=Interpolate(out_ends, out_weights),
                              cells=Pieces(line_cells), geometry=Field(H1(out), out_positions),
                              fields=fields), levels, line_levels, several,
                        field.unit)
