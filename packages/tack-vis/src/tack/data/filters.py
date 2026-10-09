"""Filters over the dataset API: contour, slice, threshold and external faces.

Ported from the ``vis/data-model`` prototype, where they read point and cell
arrays; here they read fields through their spaces, so what they accept
grows with the spaces:

- ``contour`` takes any field with a basis -- ``H1``, ``L2`` (an order per
  cell too), on an ``H1`` or ``L2`` geometry -- and evaluates it at each
  cell's corners. A continuous field on a continuous geometry merges its
  points by global edge, so the surface is watertight; anything
  discontinuous merges them only within each cell, so the surface keeps the
  jumps between cells, as the data has them.
- ``slice_plane`` contours each point's signed distance to a plane, computed
  in the geometry's own space, so an ``L2`` geometry's cells each slice
  their own corners.
- ``threshold`` keeps the cells whose field lies in a range, testing the
  field's corner values (or cell values) through its view, and carries
  every field it can: point and cell data gathered, ``L2`` blocks copied
  cell by cell, either kind of geometry.
- ``external_faces`` is the boundary face set as a surface
  (``algorithms.extract_surface``).

Outputs are new datasets on ``UnstructuredTopology``, with ``H1`` geometry
except where ``threshold`` keeps an ``L2`` one.

All four work on a cell's corners, so a quadratic field or a curved geometry
is linearized: contour and slice cut each cell by its corner values and
positions, and threshold and external faces keep an order-2 field's (or
geometry's) values at the points. Cutting curved cells by their quadratic
functions -- subdividing them -- is still to come.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import _run_offsets, gather
from tack.data import arrays, shapes
from tack.data.algorithms import _like, _take, boundary_faces, extract_surface
from tack.data.buckets import bucket_order
from tack.data.dataset import DataSet, Field
from tack.data.spaces import H1, L2, Constant, Values
from tack.data.topology import UnstructuredTopology

__all__ = ["contour", "external_faces", "slice_plane", "threshold"]


def _field(data, field):
    """``field``, or the field of that name."""
    return data.fields[field] if isinstance(field, str) else field


def _scalar_floating(name, field):
    if arrays.width_of(field.values):
        raise TypeError(f"{name} needs a scalar field, not one of "
                        f"{arrays.width_of(field.values)}-vectors")
    dtype = arrays.dtype_of(field.values)
    if dtype not in (tack.f32, tack.f64):
        raise TypeError(f"{name} needs floating-point values; the field is {dtype.name}")


# ── Contour ─────────────────────────────────────────────────────────
#
# A cell's case: bit j is 1 when the field at corner j is at or above the
# isovalue. VTK's tables (tack.data.shapes) give the triangles of each case,
# facing away from the larger values.

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
            ja = cells.edge_point(e, 0)
            jb = cells.edge_point(e, 1)
            # Always from the lower point id, so cells sharing an edge compute
            # the same point, bit for bit, when their values agree there.
            if cells.point_id(c, jb) < cells.point_id(c, ja):
                ja, jb = jb, ja
            va = u.corner_value(c, ja)
            w = (isovalue - va) / (u.corner_value(c, jb) - va)
            xa = cells.point(c, ja)
            points[first + k] = xa + w * (cells.point(c, jb) - xa)
            a = cells.point_id(c, ja)
            b = cells.point_id(c, jb)
            edges[first + k] = [a, b]
            weights[first + k] = w
            sources[first + k] = cell
            if per_cell == 1:
                keys[first + k] = tack.u64(cell) * tack.u64(16) + tack.u64(e)
            else:
                keys[first + k] = (tack.u64(a) << tack.u64(32)) | tack.u64(b)


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
def _interpolate_edges(values, edges, weights, out, n):
    for i in range(n):
        ab = edges[i]
        va = values[ab[0]]
        out[i] = va + weights[i] * (values[ab[1]] - va)


@tack.kernel
def _triangles(rows, types, offsets, connectivity, count):
    for t in range(count):
        types[t] = tack.u8(shapes.TRIANGLE)
        offsets[t] = 3 * t
        for k in range(3):
            connectivity[3 * t + k] = rows[3 * t + k]
        if t == 0:
            offsets[count] = 3 * count


def contour(data, field, isovalue, merge_points=True):
    """The surface where ``field`` crosses ``isovalue``, as triangles.

    ``field`` is a scalar field with a basis, or its name: ``H1`` (point
    data), or ``L2`` (DG, an order per cell too). Each 3D cell contributes the
    triangles of VTK's case tables for its corner values, interpolated along
    its edges; cells of lower dimension contribute nothing, and a cell
    constant has no level set inside its cells.

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
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_contour(data, field, isovalue)
    if not space.interpolated or space.on not in ("cells", "points"):
        raise TypeError(f"contour needs a field with a basis on the cells, not {space!r}")
    per_cell = 0 if isinstance(space, H1) and isinstance(data.geometry.space, H1) else 1
    groups = [g for g in data.launch_groups("cells", [field])
              if g.count and g.shape.CONTOUR_TRIANGLES]
    n_cells = data.num_cells
    counts = tack.zeros(tack.i32, (n_cells,))
    for group in groups:
        _contour_counts(data.domain_view("cells", group), field.view(group), isovalue, counts)
    starts = tack.field(tack.i32, shape=(n_cells,))
    triangles = exclusive_scan(counts, starts, n_cells) if n_cells else 0

    n = 3 * triangles
    points = tack.Vector.field(3, data.dtype, shape=(n,))
    edges = tack.Vector.field(2, tack.i32, shape=(n,))
    weights = tack.field(arrays.dtype_of(field.values), shape=(n,))
    sources = tack.field(tack.i32, shape=(n,))
    keys = tack.field(tack.u64, shape=(n,))
    for group in groups:
        _contour_points(data.domain_view("cells", group), field.view(group), isovalue,
                        per_cell, starts, points, edges, weights, sources, keys)
    triangle_cells = tack.field(tack.i32, shape=(triangles,))
    if triangles:
        _triangle_cells(sources, triangle_cells)

    if merge_points and n:
        # A key leads with the edge's low point, or (within cells) is cell * 16 + edge.
        order, _ = (bucket_order(keys, n_cells, shift=4) if per_cell
                    else bucket_order(keys, data.num_points))
        offsets, count = _run_offsets(gather(keys, order), n)
        merged = tack.Vector.field(3, data.dtype, shape=(count,))
        merged_edges = tack.Vector.field(2, tack.i32, shape=(count,))
        merged_weights = tack.field(weights.dtype, shape=(count,))
        _first_of_runs(order, offsets, points, edges, weights, merged, merged_edges,
                       merged_weights, count)
        rows = tack.field(tack.i32, shape=(n,))
        _run_of_each(order, offsets, rows, count)
        points, edges, weights = merged, merged_edges, merged_weights
    else:
        rows, count = tack.arange(n, tack.i32), n

    types = tack.field(tack.u8, shape=(triangles,))
    tri_offsets = tack.zeros(tack.i32, (triangles + 1,))
    connectivity = tack.field(tack.i32, shape=(n,))
    if triangles:
        _triangles(rows, types, tri_offsets, connectivity, triangles)
    surface = UnstructuredTopology(types, tri_offsets, connectivity, num_points=count)
    fields = {}
    for name, f in data.fields.items():
        if name == "shape" or arrays.dtype_of(f.values) not in (tack.f32, tack.f64):
            continue
        on_points = isinstance(f.space, H1) or (isinstance(f.space, Values)
                                                and f.space.on == "points")
        on_cells = isinstance(f.space, Constant) or (isinstance(f.space, Values)
                                                     and f.space.on == "cells")
        if on_points:
            out = _like(f.values, count)
            if count:
                _interpolate_edges(arrays.materialize(f.values), edges, weights, out, count)
            fields[name] = Field(H1(surface) if isinstance(f.space, H1)
                                 else Values(surface, "points"), out)
        elif on_cells:
            fields[name] = Field(Values(surface, "cells"), _take(f.values, triangle_cells))
    return DataSet(surface, Field(H1(surface), points), fields=fields)


# ── Contour of polyhedra: López's polygon tracing, face by face ─────
#
# docs/design/polyhedra.md, section 4, after VTK's vtkPolyhedronContour and
# the face-based form its design note proposes. Inside means at or above the
# isovalue. Walking a face outward for its cell, a step from outside to
# inside is a *key* crossing; each is paired with the next crossing round
# that face. With consistent winding, every crossing of a cell is a key
# crossing on exactly one of its faces and the next of another, so a cell's
# pairs form cycles: its iso-polygons. Crossings are named by their edge's
# (low, high) point ids, so a cell's faces agree on them without deriving the
# mesh's edges; the crossings are then merged across cells into one
# iso-vertex per edge, numbered in (low, high) order -- the order of the
# mesh's edge ids, had they been derived.

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
            n = cells.side_size(c, k)
            keyed = 0
            for j in range(n):
                va = values[cells.side_point(c, k, j)]
                vb = values[cells.side_point(c, k, j + 1 if j + 1 < n else 0)]
                if va < iso and vb >= iso:
                    keyed += 1
            counts[cells.entry(c, k)] = keyed


@tack.func
def _edge_key(a, b):
    return (tack.u64(min(a, b)) << tack.u64(32)) | tack.u64(max(a, b))


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
            n = cells.side_size(c, k)
            at = starts[cells.entry(c, k)]
            for j in range(n):
                va = values[cells.side_point(c, k, j)]
                vb = values[cells.side_point(c, k, j + 1 if j + 1 < n else 0)]
                if va < iso and vb >= iso:
                    nxt = j
                    for m in range(1, n):
                        j2 = j + m if j + m < n else j + m - n
                        wa = values[cells.side_point(c, k, j2)]
                        wb = values[cells.side_point(c, k, j2 + 1 if j2 + 1 < n else 0)]
                        if (1 if wa >= iso else 0) != (1 if wb >= iso else 0):
                            nxt = j2
                            break
                    pair_from[at] = _edge_key(
                        cells.side_point(c, k, j), cells.side_point(c, k, j + 1 if j + 1 < n else 0))
                    pair_to[at] = _edge_key(
                        cells.side_point(c, k, nxt),
                        cells.side_point(c, k, nxt + 1 if nxt + 1 < n else 0))
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
                        if pair_from[first + i] == target:
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
        a = tack.i32(key >> tack.u64(32))
        b = tack.i32(key & tack.u64(0xFFFFFFFF))
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


@tack.kernel
def _close_starts(starts, count, total):
    for i in range(1):
        starts[count] = total


def _polyhedral_contour(data, field, isovalue):
    """``contour`` of a polyhedral topology: iso-polygons by López's tracing, as a
    ``PolygonalTopology`` -- polygons, not triangles."""
    from tack.data.polyhedra import PolygonalTopology, SizeBuckets

    t = data.topology
    if t.dimension != 3:
        raise NotImplementedError("contour lines of polygons are not built yet")
    if not (isinstance(field.space, H1) or field.space is Values(data, "points")):
        raise TypeError("a polyhedral topology contours point data")
    values = arrays.materialize(field.values)
    entries = int(t.cell_faces.shape[0])
    counts = tack.zeros(tack.i32, (entries,))
    for group in data.launch_groups("cells", []):
        if group.count:
            _lopez_counts(data.domain_view("cells", group), values, isovalue, counts)
    pair_offsets = tack.field(tack.i32, shape=(entries + 1,))
    pairs = exclusive_scan(counts, pair_offsets, entries) if entries else 0
    _close_starts(pair_offsets, entries, pairs)
    pair_from = tack.field(tack.u64, shape=(pairs,))
    pair_to = tack.field(tack.u64, shape=(pairs,))
    if pairs:
        for group in data.launch_groups("cells", []):
            if group.count:
                _lopez_pairs(data.domain_view("cells", group), values, isovalue,
                             pair_offsets, pair_from, pair_to)
    n = data.num_cells
    poly_counts = tack.zeros(tack.i32, (n,))
    poly_offsets = tack.field(tack.i32, shape=(n + 1,))
    bad = tack.zeros(tack.i32, (1,))
    buckets = SizeBuckets(t)
    traced = [g for g in data.launch_groups("cells", [], keys=[buckets]) if g.count]
    dummy = tack.field(tack.i32, shape=(1,))
    if pairs:
        for group in traced:
            _lopez_trace(data.domain_view("cells", group), pair_offsets, pair_from, pair_to,
                         0, poly_counts, poly_offsets, dummy, dummy, dummy, bad)
    polygons = exclusive_scan(poly_counts, poly_offsets, n) if n and pairs else 0
    _close_starts(poly_offsets, n, polygons)
    poly_starts = tack.field(tack.i32, shape=(polygons + 1,))
    poly_cells = tack.field(tack.i32, shape=(polygons,))
    vertices = tack.field(tack.i32, shape=(pairs,))
    if polygons:
        for group in traced:
            _lopez_trace(data.domain_view("cells", group), pair_offsets, pair_from, pair_to,
                         1, poly_counts, poly_offsets, poly_starts, poly_cells, vertices, bad)
    _close_starts(poly_starts, polygons, pairs)
    if bad[0]:
        raise ValueError(f"{bad[0]} iso-polygons do not close: the cells' faces are not "
                         "wound consistently (check_winding)")

    # Each crossing is named by its edge in every cell around that edge: one
    # iso-vertex per run of equal names.
    order, _ = bucket_order(pair_from, data.num_points)
    offsets, count = _run_offsets(gather(pair_from, order), pairs) if pairs else (None, 0)
    positions = tack.Vector.field(3, data.dtype, shape=(count,))
    ends = tack.Vector.field(2, tack.i32, shape=(count,))
    weights = tack.field(arrays.dtype_of(field.values), shape=(count,))
    vertex_of_pair = tack.field(tack.i32, shape=(pairs,))
    if count:
        _iso_vertices(pair_from, order, offsets, values, isovalue,
                      arrays.materialize(data.geometry.values), weights, positions, ends,
                      vertex_of_pair, count)
    if pairs:
        _renumber_vertices(vertices, vertex_of_pair)
    surface = PolygonalTopology(poly_starts, vertices, num_points=count)
    fields = {}
    for name, f in data.fields.items():
        if name == "shape" or arrays.dtype_of(f.values) not in (tack.f32, tack.f64):
            continue
        if isinstance(f.space, H1) or f.space is Values(data, "points"):
            out = _like(f.values, count)
            if count:
                _interpolate_edges(arrays.materialize(f.values), ends, weights, out, count)
            fields[name] = Field(H1(surface) if isinstance(f.space, H1)
                                 else Values(surface, "points"), out)
        elif isinstance(f.space, Constant) or f.space is Values(data, "cells"):
            fields[name] = Field(Values(surface, "cells"), _take(f.values, poly_cells))
    return DataSet(surface, Field(H1(surface), positions), fields=fields)


# ── Slice ───────────────────────────────────────────────────────────

@tack.kernel
def _plane_distances(points, ox, oy, oz, nx, ny, nz, out, n):
    for p in range(n):
        out[p] = (points[p] - tack.Vector([ox, oy, oz])).dot(tack.Vector([nx, ny, nz]))


def slice_plane(data, origin, normal, merge_points=True):
    """The cut of the 3D cells by the plane through ``origin`` with ``normal`` (any
    length), as triangles: ``contour`` at zero of each position's signed distance
    to the plane along ``normal``. The distance is computed in the geometry's own
    space -- per point, or per cell corner for an ``L2`` geometry -- so the cut is
    the cells' own."""
    geometry = data.geometry
    n = arrays.size_of(geometry.values)
    distance = tack.field(data.dtype, shape=(n,))
    if n:
        ox, oy, oz = (float(v) for v in origin)
        nx, ny, nz = (float(v) for v in normal)
        _plane_distances(arrays.materialize(geometry.values), ox, oy, oz, nx, ny, nz,
                         distance, n)
    return contour(data, Field(geometry.space, distance), 0.0, merge_points=merge_points)


# ── Threshold ───────────────────────────────────────────────────────

@tack.kernel
def _keep_by_corners(cells, u, lower, upper, all_points, keep, sizes):
    for c in cells:
        inside = 0
        for j in range(cells.NUM_POINTS):
            v = u.corner_value(c, j)
            if lower <= v and v <= upper:
                inside += 1
        ok = 0
        if all_points == 1:
            if inside == cells.NUM_POINTS:
                ok = 1
        elif inside > 0:
            ok = 1
        e = cells.entity_id(c)
        keep[e] = ok
        sizes[e] = ok * cells.NUM_POINTS


@tack.kernel
def _kept_cells(cells, keep, slots, starts, types, offsets, connectivity, sources, used):
    for c in cells:
        cell = cells.entity_id(c)
        if keep[cell] == 1:
            slot = slots[cell]
            at = starts[cell]
            types[slot] = tack.u8(cells.ID)
            offsets[slot] = at
            sources[slot] = cell
            for j in range(cells.NUM_POINTS):
                p = cells.point_id(c, j)
                connectivity[at + j] = p
                tack.atomic_max(used, p, 1)


@tack.kernel
def _close_offsets(offsets, count, length):
    for i in range(1):
        offsets[count] = length


@tack.kernel
def _renumber(connectivity, new_ids):
    for k in range(connectivity.shape[0]):
        connectivity[k] = new_ids[connectivity[k]]


@tack.kernel
def _keep_rows(values, used, new_ids, out, n):
    for p in range(n):
        if used[p] == 1:
            out[new_ids[p]] = values[p]


@tack.kernel
def _copy_blocks(values, src_offsets, sources, dst_offsets, out, count):
    for i in range(count):
        a = src_offsets[sources[i]]
        b = dst_offsets[i]
        for k in range(dst_offsets[i + 1] - b):
            out[b + k] = values[a + k]


def threshold(data, field, lower, upper, all_points=True):
    """The cells whose ``field`` lies in ``[lower, upper]``, with only the points they use.

    ``field`` is a scalar field or its name. A cell field (``Constant``, values
    on cells) keeps a cell by its value; a field with a basis (``H1``, ``L2``
    of any orders) or values on points by its values at the cell's corners:
    all of them in range, or with ``all_points=False`` any (VTK's
    ``AllScalars``). The result's cells keep their shapes and order; its points
    are the used ones, renumbered in order. Every field it can carry comes
    along: point fields gathered to the kept points, cell fields to the kept
    cells, ``L2`` fields (and an ``L2`` geometry) block by block, keeping each
    cell's order. Fields on faces and edges, which the new topology derives
    afresh, are left behind.

    On a polyhedral topology the kept cells keep their faces, stored once and
    numbered in their old order, so values on faces come along. A face whose
    side-0 cell is dropped is turned around for its remaining cell, and its
    oriented values (``Values(..., "faces", oriented=True)``) are negated. A
    polygonal topology's kept polygons keep their loops; its edge values, whose
    numbering is derived afresh, are left behind.
    """
    field = _field(data, field)
    if arrays.width_of(field.values):
        raise TypeError("threshold needs a scalar field")
    space = field.space
    if isinstance(space, Values) and space.on in ("points", "cells"):
        field = Field((H1 if space.on == "points" else Constant)(data), field.values)
        space = field.space
    if not space.interpolated or space.on not in ("cells", "points"):
        raise TypeError(f"threshold needs a field on the points or cells, not {space!r}")
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_threshold(data, field, lower, upper, all_points)

    n = data.num_cells
    keep = tack.zeros(tack.i32, (n,))
    sizes = tack.zeros(tack.i32, (n,))
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _keep_by_corners(data.domain_view("cells", group), field.view(group), lower,
                             upper, 1 if all_points else 0, keep, sizes)
    slots = tack.field(tack.i32, shape=(n,))
    starts = tack.field(tack.i32, shape=(n,))
    count = exclusive_scan(keep, slots, n) if n else 0
    length = exclusive_scan(sizes, starts, n) if n else 0

    num_points = data.num_points
    types = tack.field(tack.u8, shape=(count,))
    offsets = tack.zeros(tack.i32, (count + 1,))
    connectivity = tack.field(tack.i32, shape=(length,))
    sources = tack.field(tack.i32, shape=(count,))
    used = tack.zeros(tack.i32, (num_points,))
    for group in data.topology.groups():
        if group.count:
            _kept_cells(group.view(), keep, slots, starts, types, offsets, connectivity,
                        sources, used)
    _close_offsets(offsets, count, length)
    new_ids = tack.field(tack.i32, shape=(num_points,))
    kept_points = exclusive_scan(used, new_ids, num_points) if num_points else 0
    if length:
        _renumber(connectivity, new_ids)
    kept = UnstructuredTopology(types, offsets, connectivity, num_points=kept_points)

    def carry(f):
        values = f.values
        if isinstance(f.space, H1) or (isinstance(f.space, Values) and f.space.on == "points"):
            out = _like(values, kept_points)
            if kept_points:
                _keep_rows(arrays.materialize(values), used, new_ids, out, num_points)
            return Field(H1(kept) if isinstance(f.space, H1) else Values(kept, "points"), out)
        if isinstance(f.space, Constant) or (isinstance(f.space, Values)
                                             and f.space.on == "cells"):
            return Field(type(f.space)(kept) if isinstance(f.space, Constant)
                         else Values(kept, "cells"), _take(values, sources))
        if isinstance(f.space, L2):
            if f.space.varies:
                orders = np.asarray(arrays.to_host(f.space.cell_keys()))
                out_space = L2(kept, order=orders[sources.to_numpy()] if count
                               else np.zeros(0, int))
            else:
                out_space = L2(kept, order=f.space.order)
            out = _like(values, out_space.size)
            if count:
                _copy_blocks(arrays.materialize(values), f.space.offsets, sources,
                             out_space.offsets, out, count)
            return Field(out_space, out)
        return None

    geometry = carry(data.geometry)
    fields = {}
    for name, f in data.fields.items():
        if name != "shape":
            out = carry(f)
            if out is not None:
                fields[name] = out
    return DataSet(kept, geometry, fields=fields)


# ── Threshold of polyhedra and polygons ─────────────────────────────

@tack.kernel
def _polyhedral_keep(cells, values, lower, upper, by_cells, all_points, keep):
    for c in cells:
        e = cells.entity_id(c)
        ok = 0
        if by_cells == 1:
            v = values[e]
            if lower <= v and v <= upper:
                ok = 1
        else:
            inside = 0
            n = cells.num_points(c)
            for j in range(n):
                v = values[cells.point_id(c, j)]
                if lower <= v and v <= upper:
                    inside += 1
            if all_points == 1:
                if inside == n:
                    ok = 1
            elif inside > 0:
                ok = 1
        keep[e] = ok


@tack.kernel
def _kept_faces(cell_offsets, cell_faces, cell_face_sides, keep, used, owned, n_cells):
    for c in range(n_cells):
        if keep[c] == 1:
            for e in range(cell_offsets[c], cell_offsets[c + 1]):
                f = cell_faces[e]
                tack.atomic_max(used, f, 1)
                if tack.i32(cell_face_sides[e]) == 0:
                    owned[f] = 1


@tack.kernel
def _kept_face_sizes(face_offsets, used, sizes):
    for f in range(used.shape[0]):
        sizes[f] = face_offsets[f + 1] - face_offsets[f] if used[f] == 1 else 0


@tack.kernel
def _copy_faces(face_offsets, face_points, used, owned, new_face, starts, out_points,
                flipped, point_used):
    # A kept face whose side-0 cell is gone is turned around: its only cell is
    # now its side 0, and its winding must point out of it.
    for f in range(used.shape[0]):
        if used[f] == 1:
            at = starts[f]
            n = face_offsets[f + 1] - face_offsets[f]
            turn = 1 if owned[f] == 0 else 0
            flipped[new_face[f]] = turn
            for j in range(n):
                p = face_points[face_offsets[f] + (n - 1 - j if turn == 1 else j)]
                out_points[at + j] = p
                tack.atomic_max(point_used, p, 1)


@tack.kernel
def _kept_entry_counts(cell_offsets, keep, counts, n_cells):
    for c in range(n_cells):
        counts[c] = cell_offsets[c + 1] - cell_offsets[c] if keep[c] == 1 else 0


@tack.kernel
def _copy_entries(cell_offsets, cell_faces, cell_face_sides, keep, new_face, flipped, starts,
                  out_faces, out_sides, n_cells):
    for c in range(n_cells):
        if keep[c] == 1:
            at = starts[c]
            for e in range(cell_offsets[c], cell_offsets[c + 1]):
                g = new_face[cell_faces[e]]
                out_faces[at] = g
                out_sides[at] = tack.u8(0 if flipped[g] == 1 else tack.i32(cell_face_sides[e]))
                at += 1


@tack.kernel
def _renumber_points(points, new_ids):
    for i in range(points.shape[0]):
        points[i] = new_ids[points[i]]


@tack.kernel
def _negate_flipped(values, flipped):
    for f in range(flipped.shape[0]):
        if flipped[f] == 1:
            values[f] = -values[f]


@tack.kernel
def _kept_ids(flags, slots, out):
    for i in range(flags.shape[0]):
        if flags[i] == 1:
            out[slots[i]] = i


def _compact(flags):
    """``(new ids, kept ids, count)`` of a 0/1 field: each kept index's new number,
    and the kept indices in order."""
    n = flags.shape[0]
    slots = tack.field(tack.i32, shape=(n,))
    count = exclusive_scan(flags, slots, n) if n else 0
    kept = tack.field(tack.i32, shape=(count,))
    if count:
        _kept_ids(flags, slots, kept)
    return slots, kept, count


def _polyhedral_threshold(data, field, lower, upper, all_points):
    """``threshold`` of a polyhedral or polygonal topology: whole cells, their faces
    kept once and numbered in their old order, points compacted."""
    from tack.data.polyhedra import PolygonalTopology, PolyhedralTopology

    t = data.topology
    by_cells = isinstance(field.space, Constant)
    values = arrays.materialize(field.values)
    n = data.num_cells
    keep = tack.zeros(tack.i32, (n,))
    for group in data.launch_groups("cells", []):
        if group.count:
            _polyhedral_keep(data.domain_view("cells", group), values, lower, upper,
                             1 if by_cells else 0, 1 if all_points else 0, keep)
    _, kept_cells, kept_n = _compact(keep)
    point_used = tack.zeros(tack.i32, (data.num_points,))

    if t.dimension == 2:
        loop_offsets, loop_points = t.loops()
        sizes = tack.zeros(tack.i32, (n,))
        if n:
            _kept_entry_counts(loop_offsets, keep, sizes, n)
        starts = tack.field(tack.i32, shape=(n + 1,))
        total = exclusive_scan(sizes, starts, n) if n else 0
        out_loops = tack.field(tack.i32, shape=(total,))
        if total:
            _copy_loops(loop_offsets, loop_points, keep, starts, out_loops, point_used, n)
        point_new, kept_points, kept_np = _compact(point_used)
        if total:
            _renumber_points(out_loops, point_new)
        offsets = _gather_offsets(starts, kept_cells, total)
        out = PolygonalTopology(offsets, out_loops, num_points=kept_np)
        flipped = None
    else:
        nf = t.num_faces
        used = tack.zeros(tack.i32, (nf,))
        owned = tack.zeros(tack.i32, (nf,))
        if n:
            _kept_faces(t.cell_offsets, t.cell_faces, t.cell_face_sides, keep, used, owned, n)
        new_face, kept_faces, kept_nf = _compact(used)
        sizes = tack.field(tack.i32, shape=(nf,))
        if nf:
            _kept_face_sizes(t.face_offsets, used, sizes)
        starts = tack.field(tack.i32, shape=(nf + 1,))
        total = exclusive_scan(sizes, starts, nf) if nf else 0
        face_points = tack.field(tack.i32, shape=(total,))
        flipped = tack.zeros(tack.i32, (kept_nf,))
        if nf:
            _copy_faces(t.face_offsets, t.face_points, used, owned, new_face, starts,
                        face_points, flipped, point_used)
        face_offsets = _gather_offsets(starts, kept_faces, total)
        counts = tack.zeros(tack.i32, (n,))
        if n:
            _kept_entry_counts(t.cell_offsets, keep, counts, n)
        entry_starts = tack.field(tack.i32, shape=(n + 1,))
        entries = exclusive_scan(counts, entry_starts, n) if n else 0
        cell_faces = tack.field(tack.i32, shape=(entries,))
        cell_sides = tack.field(tack.u8, shape=(entries,))
        if entries:
            _copy_entries(t.cell_offsets, t.cell_faces, t.cell_face_sides, keep, new_face,
                          flipped, entry_starts, cell_faces, cell_sides, n)
        cell_offsets = _gather_offsets(entry_starts, kept_cells, entries)
        point_new, kept_points, kept_np = _compact(point_used)
        if total:
            _renumber_points(face_points, point_new)
        out = PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, cell_sides,
                                 num_points=kept_np)

    def carry(f):
        space = f.space
        if (isinstance(space, H1) and space.order == 1) or space is Values(data, "points"):
            values = _take(f.values, kept_points)
            return Field(H1(out) if isinstance(space, H1) else Values(out, "points"), values)
        if isinstance(space, Constant) or space is Values(data, "cells"):
            return Field(Constant(out) if isinstance(space, Constant) else Values(out, "cells"),
                         _take(f.values, kept_cells))
        if isinstance(space, Values) and space.on == "faces" and flipped is not None:
            values = _take(f.values, kept_faces)
            if space.oriented and kept_nf:
                _negate_flipped(values, flipped)
            return Field(Values(out, "faces", oriented=space.oriented), values)
        return None

    fields = {}
    for name, f in data.fields.items():
        if name != "shape" and carry(f) is not None:
            fields[name] = carry(f)
    return DataSet(out, carry(data.geometry), fields=fields)


@tack.kernel
def _copy_loops(loop_offsets, loop_points, keep, starts, out, point_used, n_cells):
    for c in range(n_cells):
        if keep[c] == 1:
            at = starts[c]
            for j in range(loop_offsets[c], loop_offsets[c + 1]):
                out[at] = loop_points[j]
                tack.atomic_max(point_used, loop_points[j], 1)
                at += 1


@tack.kernel
def _offsets_of(starts, kept, out, total):
    for i in range(kept.shape[0]):
        out[i] = starts[kept[i]]
        if i == 0:
            out[kept.shape[0]] = total


def _gather_offsets(starts, kept, total):
    """CSR offsets of the kept rows: each kept row's start in the compacted array, and
    ``total`` at the end."""
    out = tack.field(tack.i32, shape=(kept.shape[0] + 1,))
    if kept.shape[0]:
        _offsets_of(starts, kept, out, total)
    else:
        out.from_numpy(np.zeros(1, np.int32))
    return out


# ── External faces ──────────────────────────────────────────────────

def external_faces(data, name="boundary"):
    """The faces of the 3D cells that only one cell has, as a surface dataset: the
    boundary face set (kept in ``data.sets[name]``) through ``extract_surface``.
    Each face faces out of its cell and carries that cell's cell fields; point
    fields stay on the same points."""
    boundary_faces(data, name)
    return extract_surface(data, name)
