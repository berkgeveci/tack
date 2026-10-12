"""``contour``: the surface where ``field`` crosses ``isovalue``, as triangles."""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.data import arrays, ids, shapes
from tack.data.algorithms._common import _close, _compact, _field, _scalar_floating
from tack.data.buckets import bucket_order, run_offsets
from tack.data.carry import Interpolate, Pieces, carry
from tack.data.dataset import Field
from tack.data.spaces import H1, Values
from tack.data.topology import UnstructuredTopology

__all__ = ['contour']


@tack.kernel
def _contour_counts(cells, u, isovalue, counts):
    for c in cells:
        case = 0
        for j in range(cells.NUM_POINTS):
            if u.corner_value(c, j) >= isovalue:
                case |= 1 << j
        counts[cells.entity_id(c)] = cells.contour_count(case)


@tack.kernel
def _contour_points(cells, u, isovalue, per_cell, starts, points, edges, weights, sources,
                    keys):
    for c in cells:
        case = 0
        for j in range(cells.NUM_POINTS):
            if u.corner_value(c, j) >= isovalue:
                case |= 1 << j
        cell = cells.entity_id(c)
        first = 3 * starts[cell]
        for k in range(3 * cells.contour_count(case)):
            e = cells.contour_edge(case, k)
            ja = cells.edge_corner(e, 0)
            jb = cells.edge_corner(e, 1)
            # Always from the lower point id, so cells sharing an edge compute
            # the same point, bit for bit, when their values agree there.
            if cells.point_id(c, jb) < cells.point_id(c, ja):
                ja, jb = jb, ja
            va = u.corner_value(c, ja)
            w = (isovalue - va) / (u.corner_value(c, jb) - va)
            xa = cells.corner_position(c, ja)
            points[first + k] = xa + w * (cells.corner_position(c, jb) - xa)
            a = cells.point_id(c, ja)
            b = cells.point_id(c, jb)
            edges[first + k] = [a, b]
            weights[first + k] = w
            sources[first + k] = cell
            if per_cell == 1:
                keys[first + k] = [cell, e]
            else:
                keys[first + k] = [a, b]


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
def _triangle_cells(sources, out):
    for t in range(out.shape[0]):
        out[t] = sources[3 * t]


@tack.kernel
def _triangles(rows, types, offsets, connectivity, count):
    for t in range(count):
        types[t] = tack.u8(shapes.TRIANGLE)
        offsets[t] = 3 * t
        for k in range(3):
            connectivity[3 * t + k] = rows[3 * t + k]
        if t == 0:
            offsets[count] = 3 * count


def contour(data, field, isovalue, merge_points=True, fields=None):
    """Where ``field`` crosses ``isovalue``: a surface of a 3D mesh, as triangles
    (as polygons on polyhedra); lines of a 2D mesh.

    ``field`` is a scalar field with a basis, or its name: ``H1`` (point
    data), or ``L2`` (DG, an order per cell too). Each 3D cell contributes the
    triangles of VTK's case tables for its corner values, interpolated along
    its edges; in a mesh with 3D cells, cells of lower dimension contribute
    nothing, and a cell constant has no level set inside its cells.

    A 2D mesh -- triangles, quads, pixels or polygons -- gives line cells from
    its point values: each cell's crossings, found going round its edges, are
    joined in pairs, each line with the higher values on its right as the
    cell winds. Lines share their points through the mesh's faces (its edges),
    whatever ``merge_points`` says. A polyline mesh is refused.

    With ``merge_points``, triangles share their points: by global edge when
    the field and the geometry are both continuous, so the surface is
    watertight; otherwise only within each cell, so a DG field's surface
    keeps its jumps and an ``L2`` geometry's its gaps. ``H1`` fields carry
    onto the surface's points, interpolated along the same edges; cell
    fields (``Constant``, values on cells) onto its triangles, from the cell
    each came from.
    """
    field = _field(data, field)
    space = field.space
    _scalar_floating("contour", field)
    dimension = data.topology.dimension
    if dimension == 2:
        return _contour_lines(data, field, isovalue, fields)
    if dimension < 2:
        raise NotImplementedError("contour of lines (which would give points) is not built")
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_contour(data, field, isovalue, fields)
    if not space.interpolated or space.on not in ("cells", "points"):
        raise TypeError(f"contour needs a field with a basis on the cells, not {space!r}")
    per_cell = 0 if isinstance(space, H1) and isinstance(data.geometry.space, H1) else 1
    groups = [g for g in data.launch_groups("cells", [field])
              if g.count and g.shape.CONTOUR_TRIANGLES]
    n_cells = data.num_cells
    idt = data.id_dtype
    odt = ids.for_output(data, 15 * n_cells)     # up to five triangles a cell
    counts = tack.zeros(tack.i32, (n_cells,))
    for group in groups:
        _contour_counts(data.domain_view("cells", group), field.view(group), isovalue, counts)
    starts = tack.field(odt, shape=(n_cells,))
    triangles = exclusive_scan(counts, starts, n_cells) if n_cells else 0

    n = 3 * triangles
    points = tack.Vector.field(3, data.dtype, shape=(n,))
    edges = tack.Vector.field(2, idt, shape=(n,))
    weights = tack.field(arrays.dtype_of(field.values), shape=(n,))
    sources = tack.field(idt, shape=(n,))
    keys = tack.Vector.field(2, idt, shape=(n,))
    for group in groups:
        _contour_points(data.domain_view("cells", group), field.view(group), isovalue,
                        per_cell, starts, points, edges, weights, sources, keys)
    triangle_cells = tack.field(idt, shape=(triangles,))
    if triangles:
        _triangle_cells(sources, triangle_cells)

    if merge_points and n:
        # A key is the edge's (low, high) points, or (within cells) (cell, local edge).
        order, _ = bucket_order(keys, n_cells if per_cell else data.num_points, odt)
        offsets, count = run_offsets(keys, order)
        merged = tack.Vector.field(3, data.dtype, shape=(count,))
        merged_edges = tack.Vector.field(2, idt, shape=(count,))
        merged_weights = tack.field(weights.dtype, shape=(count,))
        _first_of_runs(order, offsets, points, edges, weights, merged, merged_edges,
                       merged_weights, count)
        rows = tack.field(odt, shape=(n,))
        _run_of_each(order, offsets, rows, count)
        points, edges, weights = merged, merged_edges, merged_weights
    else:
        rows, count = tack.arange(n, odt), n

    types = tack.field(tack.u8, shape=(triangles,))
    tri_offsets = tack.zeros(odt, (triangles + 1,))
    connectivity = tack.field(odt, shape=(n,))
    if triangles:
        _triangles(rows, types, tri_offsets, connectivity, triangles)
    surface = UnstructuredTopology(types, tri_offsets, connectivity, num_points=count)
    return carry(data, surface, points=Interpolate(edges, weights),
                 cells=Pieces(triangle_cells), geometry=Field(H1(surface), points),
                 fields=fields)


@tack.kernel
def _lopez_counts(cells, values, iso, counts):
    # Most cells lie on one side of the isovalue and need not walk their faces;
    # their counts stay zero.
    for c in cells:
        below = 0
        above = 0
        for j in range(cells.num_points(c)):
            if values[cells.point_id(c, j)] >= iso:
                above = 1
            else:
                below = 1
        if below == 0 or above == 0:
            continue
        for k in range(cells.num_faces(c)):
            n = cells.side_num_points(c, k)
            keyed = 0
            for j in range(n):
                va = values[cells.side_point_id(c, k, j)]
                vb = values[cells.side_point_id(c, k, j + 1 if j + 1 < n else 0)]
                if va < iso and vb >= iso:
                    keyed += 1
            counts[cells.entry(c, k)] = keyed


@tack.func
def _edge_key(a, b):
    return [min(a, b), max(a, b)]


@tack.kernel
def _lopez_pairs(cells, values, iso, starts, pair_from, pair_to):
    for c in cells:
        below = 0
        above = 0
        for j in range(cells.num_points(c)):
            if values[cells.point_id(c, j)] >= iso:
                above = 1
            else:
                below = 1
        if below == 0 or above == 0:
            continue
        for k in range(cells.num_faces(c)):
            n = cells.side_num_points(c, k)
            at = starts[cells.entry(c, k)]
            for j in range(n):
                va = values[cells.side_point_id(c, k, j)]
                vb = values[cells.side_point_id(c, k, j + 1 if j + 1 < n else 0)]
                if va < iso and vb >= iso:
                    nxt = j
                    for m in range(1, n):
                        j2 = j + m if j + m < n else j + m - n
                        wa = values[cells.side_point_id(c, k, j2)]
                        wb = values[cells.side_point_id(c, k, j2 + 1 if j2 + 1 < n else 0)]
                        if (1 if wa >= iso else 0) != (1 if wb >= iso else 0):
                            nxt = j2
                            break
                    pair_from[at] = _edge_key(
                        cells.side_point_id(c, k, j), cells.side_point_id(c, k, j + 1 if j + 1 < n else 0))
                    pair_to[at] = _edge_key(
                        cells.side_point_id(c, k, nxt),
                        cells.side_point_id(c, k, nxt + 1 if nxt + 1 < n else 0))
                    at += 1


@tack.kernel
def _lopez_trace(cells, pair_offsets, pair_from, pair_to, emit, poly_counts, poly_offsets,
                 poly_starts, poly_cells, vertices, bad):
    for c in cells:
        e = cells.entity_id(c)
        first = pair_offsets[cells.entry(c, 0)]
        m = pair_offsets[cells.entry(c, cells.num_faces(c))] - first
        used = tack.local_array(tack.i32, cells.MAX_SCRATCH)
        for i in range(m):
            used[i] = 0
        polys = 0
        out = first
        for s in range(m):
            if used[s] == 0:
                if emit == 1:
                    poly_starts[poly_offsets[e] + polys] = out
                    poly_cells[poly_offsets[e] + polys] = e
                cur = s
                for step in range(m):
                    used[cur] = 1
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
                    if nxt < 0 or used[nxt] == 1:
                        tack.atomic_add(bad, 0, 1)
                        break
                    cur = nxt
                polys += 1
        if emit == 0:
            poly_counts[e] = polys


@tack.kernel
def _iso_vertices(pair_from, order, offsets, values, iso, positions, weights, out, ends,
                  vertex_of_pair, count):
    # One per run of equal crossing keys, interpolated from the lower point id, as
    # the shape path's contour interpolates, so the two agree on every point.
    for r in range(count):
        key = pair_from[order[offsets[r]]]
        a = key[0]
        b = key[1]
        va = values[a]
        w = (iso - va) / (values[b] - va)
        xa = positions[a]
        out[r] = xa + w * (positions[b] - xa)
        ends[r] = [a, b]
        weights[r] = w
        for i in range(offsets[r], offsets[r + 1]):
            vertex_of_pair[order[i]] = r


@tack.kernel
def _renumber_vertices(vertices, vertex_of_pair):
    for i in range(vertices.shape[0]):
        vertices[i] = vertex_of_pair[vertices[i]]


def _polyhedral_contour(data, field, isovalue, fields=None):
    """``contour`` of a polyhedral topology: iso-polygons by López's tracing, as a
    ``PolygonalTopology`` -- polygons, not triangles."""
    from tack.data.polyhedra import PolygonalTopology, SizeBuckets

    t = data.topology
    if not (isinstance(field.space, H1) or field.space is Values(data, "points")):
        raise TypeError("a polyhedral topology contours point data")
    values = arrays.materialize(field.values)
    entries = int(t.cell_faces.shape[0])
    idt = data.id_dtype
    # A face's crossings are at most its points; every face is in two cells at most.
    odt = ids.for_output(data, 2 * int(t.face_points.shape[0]))
    counts = tack.zeros(tack.i32, (entries,))
    for group in data.launch_groups("cells", []):
        if group.count:
            _lopez_counts(data.domain_view("cells", group), values, isovalue, counts)
    pair_offsets = tack.field(odt, shape=(entries + 1,))
    pairs = exclusive_scan(counts, pair_offsets, entries) if entries else 0
    _close(pair_offsets, entries, pairs)
    pair_from = tack.Vector.field(2, idt, shape=(pairs,))
    pair_to = tack.Vector.field(2, idt, shape=(pairs,))
    if pairs:
        for group in data.launch_groups("cells", []):
            if group.count:
                _lopez_pairs(data.domain_view("cells", group), values, isovalue,
                             pair_offsets, pair_from, pair_to)
    n = data.num_cells
    poly_counts = tack.zeros(tack.i32, (n,))
    poly_offsets = tack.field(odt, shape=(n + 1,))
    bad = tack.zeros(tack.i32, (1,))
    buckets = SizeBuckets(t)
    traced = [g for g in data.launch_groups("cells", [], keys=[buckets]) if g.count]
    dummy = tack.field(odt, shape=(1,))
    if pairs:
        for group in traced:
            _lopez_trace(data.domain_view("cells", group), pair_offsets, pair_from, pair_to,
                         0, poly_counts, poly_offsets, dummy, dummy, dummy, bad)
    polygons = exclusive_scan(poly_counts, poly_offsets, n) if n and pairs else 0
    _close(poly_offsets, n, polygons)
    poly_starts = tack.field(odt, shape=(polygons + 1,))
    poly_cells = tack.field(idt, shape=(polygons,))
    vertices = tack.field(odt, shape=(pairs,))
    if polygons:
        for group in traced:
            _lopez_trace(data.domain_view("cells", group), pair_offsets, pair_from, pair_to,
                         1, poly_counts, poly_offsets, poly_starts, poly_cells, vertices, bad)
    _close(poly_starts, polygons, pairs)
    if bad[0]:
        raise ValueError(f"{bad[0]} iso-polygons do not close: the cells' faces are not "
                         "wound consistently (check_winding)")

    # Each crossing is named by its edge in every cell around that edge: one
    # iso-vertex per run of equal names.
    order, _ = bucket_order(pair_from, data.num_points, odt)
    offsets, count = run_offsets(pair_from, order)
    positions = tack.Vector.field(3, data.dtype, shape=(count,))
    ends = tack.Vector.field(2, idt, shape=(count,))
    weights = tack.field(arrays.dtype_of(field.values), shape=(count,))
    vertex_of_pair = tack.field(odt, shape=(pairs,))
    if count:
        _iso_vertices(pair_from, order, offsets, values, isovalue,
                      arrays.materialize(data.geometry.values), weights, positions, ends,
                      vertex_of_pair, count)
    if pairs:
        _renumber_vertices(vertices, vertex_of_pair)
    surface = PolygonalTopology(poly_starts, vertices, num_points=count)
    return carry(data, surface, points=Interpolate(ends, weights), cells=Pieces(poly_cells),
                 geometry=Field(H1(surface), positions), fields=fields)


# ── Lines of 2D meshes ──────────────────────────────────────────────

@tack.kernel
def _line_counts(cells, values, iso, counts):
    # A 2D cell's crossings, going round its edges: half as many lines.
    for c in cells:
        crossings = 0
        for k in range(cells.num_faces(c)):
            above_a = values[cells.side_point_id(c, k, 0)] >= iso
            above_b = values[cells.side_point_id(c, k, 1)] >= iso
            if above_a != above_b:
                crossings += 1
        counts[cells.entity_id(c)] = crossings // 2


@tack.kernel
def _line_pieces(cells, values, iso, starts, line_faces, line_cells, crossed, ends, weights,
                 positions):
    for c in cells:
        n = cells.num_faces(c)
        # Going round the cell from its first rising crossing, each rising crossing
        # (below to above) pairs with the next falling one: a line cutting off a run
        # of points above the isovalue, which lies on its right.
        first = -1
        for k in range(n):
            if first < 0:
                if (values[cells.side_point_id(c, k, 0)] < iso
                        and values[cells.side_point_id(c, k, 1)] >= iso):
                    first = k
        if first < 0:
            continue
        at = starts[cells.entity_id(c)]
        rising = 0
        for i in range(n):
            k = (first + i) % n
            a = cells.side_point_id(c, k, 0)
            b = cells.side_point_id(c, k, 1)
            above_a = values[a] >= iso
            above_b = values[b] >= iso
            if above_a != above_b:
                f = cells.face_id(c, k)
                if not above_a:
                    rising = f
                else:
                    line_faces[at] = [rising, f]
                    line_cells[at] = cells.entity_id(c)
                    at += 1
                if cells.face_side(c, k) == 0:
                    # The face's side 0 records its iso-point, from its lower point
                    # id, as every contour interpolates.
                    lo = 0 if a < b else 1
                    p = cells.side_point_id(c, k, lo)
                    q = cells.side_point_id(c, k, 1 - lo)
                    w = (iso - values[p]) / (values[q] - values[p])
                    xp = cells.side_position(c, k, lo)
                    crossed[f] = 1
                    ends[f] = [p, q]
                    weights[f] = w
                    positions[f] = xp + w * (cells.side_position(c, k, 1 - lo) - xp)


@tack.kernel
def _iso_points(kept, ends, weights, positions, out_ends, out_weights, out_positions):
    for v in range(kept.shape[0]):
        f = kept[v]
        out_ends[v] = ends[f]
        out_weights[v] = weights[f]
        out_positions[v] = positions[f]


@tack.kernel
def _line_cells(line_faces, point_of_face, types, offsets, connectivity, n):
    for i in range(n):
        pair = line_faces[i]
        types[i] = tack.u8(shapes.LINE)
        offsets[i] = 2 * i
        connectivity[2 * i] = point_of_face[pair[0]]
        connectivity[2 * i + 1] = point_of_face[pair[1]]
        if i == n - 1:
            offsets[i + 1] = 2 * (i + 1)


def _contour_lines(data, field, isovalue, fields=None):
    """``contour`` of a 2D mesh -- shapes or polygons -- as line cells: each cell's
    crossings found going round its edges, which are the mesh's faces, so lines
    share their points through them."""
    if not (isinstance(field.space, H1) or field.space is Values(data, "points")):
        raise TypeError("contour of a 2D mesh takes a field on the points")
    values = arrays.materialize(field.values)
    faces = data.topology.faces()
    nf = faces.num_faces
    idt = data.id_dtype
    n = data.num_cells
    groups = [g for g in data.launch_groups("cells", [])
              if g.count and g.shape.DIMENSION == 2]
    counts = tack.zeros(tack.i32, (n,))
    for group in groups:
        _line_counts(data.domain_view("cells", group), values, isovalue, counts)
    odt = ids.for_output(data, 2 * nf)
    starts = tack.field(odt, shape=(n,))
    lines = exclusive_scan(counts, starts, n) if n else 0
    line_faces = tack.Vector.field(2, idt, shape=(lines,))
    line_cells = tack.field(idt, shape=(lines,))
    crossed = tack.zeros(tack.i32, (nf,))
    ends = tack.Vector.field(2, idt, shape=(nf,))
    weights = tack.field(values.dtype, shape=(nf,))
    positions = tack.Vector.field(3, data.dtype, shape=(nf,))
    if lines:
        for group in groups:
            _line_pieces(data.domain_view("cells", group), values, isovalue, starts, line_faces,
                         line_cells, crossed, ends, weights, positions)
    point_of_face, kept, count = _compact(crossed, odt)
    out_ends = tack.Vector.field(2, idt, shape=(count,))
    out_weights = tack.field(values.dtype, shape=(count,))
    out_positions = tack.Vector.field(3, data.dtype, shape=(count,))
    if count:
        _iso_points(kept, ends, weights, positions, out_ends, out_weights, out_positions)
    types = tack.field(tack.u8, shape=(lines,))
    offsets = tack.zeros(odt, (lines + 1,))
    connectivity = tack.field(odt, shape=(2 * lines,))
    if lines:
        _line_cells(line_faces, point_of_face, types, offsets, connectivity, lines)
    out = UnstructuredTopology(types, offsets, connectivity, num_points=count)
    return carry(data, out, points=Interpolate(out_ends, out_weights), cells=Pieces(line_cells),
                 geometry=Field(H1(out), out_positions), fields=fields)
