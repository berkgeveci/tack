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
from tack.data import arrays, ids, shapes
from tack.data.algorithms import boundary_faces, extract_surface
from tack.data.buckets import bucket_order, run_offsets
from tack.data.carry import Interpolate, Pieces, Same, Take, _point_values, carry
from tack.data.dataset import DataSet, Field
from tack.data.implicit import Plane
from tack.data.spaces import H1, Constant, Values
from tack.data.topology import UnstructuredTopology, _as_field

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


@tack.kernel
def _close_starts(starts, count, total):
    for i in range(1):
        starts[count] = total


def _polyhedral_contour(data, field, isovalue, fields=None):
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
    idt = data.id_dtype
    # A face's crossings are at most its points; every face is in two cells at most.
    odt = ids.for_output(data, 2 * int(t.face_points.shape[0]))
    counts = tack.zeros(tack.i32, (entries,))
    for group in data.launch_groups("cells", []):
        if group.count:
            _lopez_counts(data.domain_view("cells", group), values, isovalue, counts)
    pair_offsets = tack.field(odt, shape=(entries + 1,))
    pairs = exclusive_scan(counts, pair_offsets, entries) if entries else 0
    _close_starts(pair_offsets, entries, pairs)
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
    _close_starts(poly_offsets, n, polygons)
    poly_starts = tack.field(odt, shape=(polygons + 1,))
    poly_cells = tack.field(idt, shape=(polygons,))
    vertices = tack.field(odt, shape=(pairs,))
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


# ── Slice ───────────────────────────────────────────────────────────

def slice_plane(data, origin, normal, merge_points=True, fields=None):
    """The cut of the 3D cells by the plane through ``origin`` with ``normal`` (any
    length), as triangles: ``contour`` at zero of each position's signed distance
    to the plane along ``normal``. The distance is computed in the geometry's own
    space -- per point, or per cell corner for an ``L2`` geometry -- so the cut is
    the cells' own. ``slice`` with a ``Plane``."""
    return slice(data, Plane(origin, normal), merge_points=merge_points, fields=fields)


# ── Implicit functions, extraction and masks ────────────────────────

def slice(data, function, merge_points=True, fields=None):
    """Where ``function`` (``tack.data.implicit``) is zero: a contour of its values on
    the geometry, carrying fields as ``contour`` does. ``slice_plane`` is this
    with a ``Plane``."""
    from tack.data.algorithms import implicit_values

    return contour(data, implicit_values(data, function), 0.0, merge_points=merge_points,
                   fields=fields)


def extract_geometry(data, function, inside=True, boundary=False, fields=None):
    """The whole cells inside ``function``'s region (all their points at or below zero),
    or, with ``inside=False``, outside it (all above); with ``boundary``, also the
    cells it cuts (some point on the kept side). Cells are not split -- that is
    clip. Points are compacted and fields carried, as by ``threshold``."""
    from tack.data.algorithms import implicit_values

    values = implicit_values(data, function)
    lower, upper = (-np.inf, 0.0) if inside else (np.nextafter(0.0, 1.0), np.inf)
    return threshold(data, values, lower, upper, all_points=not boundary, fields=fields)


@tack.kernel
def _flag_ids(ids, flags):
    for i in range(ids.shape[0]):
        flags[ids[i]] = 1


def extract_cells(data, cells, fields=None):
    """The cells with the given ids (a host array or an integer field), with only the
    points they use; cells keep their order, not the order given."""
    if not hasattr(cells, "to_numpy"):
        values = np.asarray(cells).reshape(-1)
        if values.size and not np.issubdtype(values.dtype, np.integer):
            raise TypeError("cell ids must be integers")
        cells = values
    chosen = ids.as_ids(cells, data.id_dtype)
    keep = tack.zeros(tack.i32, (data.num_cells,))
    if chosen.shape[0]:
        bad = chosen.to_numpy()
        if bad.min() < 0 or bad.max() >= data.num_cells:
            raise IndexError(f"cell ids must lie in [0, {data.num_cells})")
        _flag_ids(chosen, keep)
    return _keep_cells(data, keep, fields)


@tack.kernel
def _every(flags, stride):
    for i in range(flags.shape[0]):
        flags[i] = 1 if i % stride == 0 else 0


def mask(data, stride, fields=None):
    """Every ``stride``-th cell (0, stride, 2 * stride, ...), as Viskores' ``Mask``."""
    if int(stride) < 1:
        raise ValueError(f"stride must be at least 1, not {stride!r}")
    keep = tack.field(tack.i32, shape=(data.num_cells,))
    if data.num_cells:
        _every(keep, int(stride))
    return _keep_cells(data, keep, fields)


# Points as vertex cells: what extract_points, threshold_points and
# mask_points make, as Viskores' do.

@tack.kernel
def _vertex_cells(types, offsets, connectivity):
    for i in range(connectivity.shape[0]):
        types[i] = tack.u8(shapes.VERTEX)
        offsets[i] = i
        connectivity[i] = i
        if i == 0:
            offsets[connectivity.shape[0]] = connectivity.shape[0]


def _keep_points(data, flags, fields):
    """The points whose ``flags`` (i32, one per point) is 1, each a vertex cell; point
    fields come along, cell fields do not."""
    _, kept, count = _compact(flags, data.id_dtype)
    types = tack.field(tack.u8, shape=(count,))
    offsets = tack.zeros(data.id_dtype, (count + 1,))
    connectivity = tack.field(data.id_dtype, shape=(count,))
    if count:
        _vertex_cells(types, offsets, connectivity)
    vertices = UnstructuredTopology(types, offsets, connectivity, num_points=count)
    return carry(data, vertices, points=Take(kept), fields=fields)


@tack.kernel
def _flag_range(values, lower, upper, flags):
    for i in range(flags.shape[0]):
        v = values[i]
        flags[i] = 1 if lower <= v and v <= upper else 0


def _point_flags(data, values, lower, upper):
    flags = tack.field(tack.i32, shape=(data.num_points,))
    if data.num_points:
        _flag_range(arrays.materialize(values), lower, upper, flags)
    return flags


def extract_points(data, function, inside=True, fields=None):
    """The points inside ``function``'s region (at or below zero), or outside it
    (above), as vertex cells. Needs an ``H1`` geometry: one position per point."""
    from tack.data.algorithms import implicit_values

    if not isinstance(data.geometry.space, H1):
        raise TypeError("extract_points needs an H1 geometry: one position per point")
    values = _point_values(implicit_values(data, function).values, data.num_points)
    lower, upper = (-np.inf, 0.0) if inside else (np.nextafter(0.0, 1.0), np.inf)
    return _keep_points(data, _point_flags(data, values, lower, upper), fields)


def threshold_points(data, field, lower, upper, fields=None):
    """The points whose ``field`` (point data: ``H1``, values on points) lies in
    ``[lower, upper]``, as vertex cells."""
    field = _field(data, field)
    space = field.space
    if not (isinstance(space, H1) or (isinstance(space, Values) and space.on == "points")):
        raise TypeError(f"threshold_points needs point data, not {space!r}")
    if arrays.width_of(field.values):
        raise TypeError("threshold_points needs a scalar field")
    values = _point_values(field.values, data.num_points)
    return _keep_points(data, _point_flags(data, values, lower, upper), fields)


def mask_points(data, stride, fields=None):
    """Every ``stride``-th point (0, stride, 2 * stride, ...) as vertex cells, as
    Viskores' ``MaskPoints``."""
    if int(stride) < 1:
        raise ValueError(f"stride must be at least 1, not {stride!r}")
    flags = tack.field(tack.i32, shape=(data.num_points,))
    if data.num_points:
        _every(flags, int(stride))
    return _keep_points(data, flags, fields)


# ── Refinement: tetrahedra, triangles, shrink, point cloud ───────────
#
# Viskores' tables (worklet/internal/TriangulateTables.h): a hexahedron in 5
# tetrahedra, a wedge in 3, a pyramid in 2; a quad in 2 triangles. A voxel's
# and a pixel's corners run x-fastest, so they take the hexahedron's and the
# quad's tables through their corner order. The 5-tetrahedron split does not
# meet a neighbour's on every shared face, as in Viskores. One of Viskores'
# wedge tetrahedra is inside out under the VTK corner order both use; it is
# turned here, the split otherwise the same.

_HEX_TETS = [[0, 1, 3, 4], [1, 4, 5, 6], [1, 4, 6, 3], [1, 3, 6, 2], [3, 6, 7, 4]]
_VOXEL_ORDER = [0, 1, 3, 2, 4, 5, 7, 6]
_PIECES = {
    "tetrahedra": {
        int(shapes.TETRA): [[0, 1, 2, 3]],
        int(shapes.HEXAHEDRON): _HEX_TETS,
        int(shapes.VOXEL): [[_VOXEL_ORDER[k] for k in tet] for tet in _HEX_TETS],
        # Viskores' (3, 4, 5, 2) is inside out: the same tetrahedron, turned.
        int(shapes.WEDGE): [[0, 1, 2, 4], [3, 5, 4, 2], [0, 2, 3, 4]],
        int(shapes.PYRAMID): [[0, 1, 2, 4], [0, 2, 3, 4]],
    },
    "triangles": {
        int(shapes.TRIANGLE): [[0, 1, 2]],
        int(shapes.QUAD): [[0, 1, 2], [0, 2, 3]],
        int(shapes.PIXEL): [[0, 1, 3], [0, 3, 2]],
    },
}


@tack.kernel
def _piece_counts(cells, pieces, counts):
    for c in cells:
        counts[cells.entity_id(c)] = pieces


@tack.kernel
def _split_cells(cells, table, pieces, width, starts, connectivity, sources):
    for c in cells:
        e = cells.entity_id(c)
        at = starts[e]
        for t in range(pieces):
            for v in range(width):
                connectivity[(at + t) * width + v] = cells.point_id(c, table[t * width + v])
            sources[at + t] = e


@tack.kernel
def _fan_counts(loop_offsets, counts):
    for c in range(counts.shape[0]):
        counts[c] = max(loop_offsets[c + 1] - loop_offsets[c] - 2, 0)


@tack.kernel
def _fans(loop_offsets, loop_points, starts, connectivity, sources):
    for c in range(loop_offsets.shape[0] - 1):
        first = loop_offsets[c]
        at = starts[c]
        for k in range(loop_offsets[c + 1] - first - 2):
            connectivity[3 * (at + k)] = loop_points[first]
            connectivity[3 * (at + k) + 1] = loop_points[first + k + 1]
            connectivity[3 * (at + k) + 2] = loop_points[first + k + 2]
            sources[at + k] = c


@tack.kernel
def _uniform_cells(types, offsets, shape, width):
    for i in range(types.shape[0]):
        types[i] = tack.u8(shape)
        offsets[i] = i * width
        if i == 0:
            offsets[types.shape[0]] = types.shape[0] * width


def _pieces_topology(data, connectivity, sources, count, shape, width, fields):
    types = tack.field(tack.u8, shape=(count,))
    offsets = tack.zeros(connectivity.dtype, (count + 1,))
    if count:
        _uniform_cells(types, offsets, int(shape), width)
    out = UnstructuredTopology(types, offsets, connectivity, num_points=data.num_points)
    return carry(data, out, points=Same(), cells=Pieces(sources), fields=fields)


def _split(data, kind, fields):
    tables = _PIECES[kind]
    width, shape = (4, shapes.TETRA) if kind == "tetrahedra" else (3, shapes.TRIANGLE)
    n = data.num_cells
    groups = [g for g in data.topology.groups() if g.count and int(g.shape.ID) in tables]
    counts = tack.zeros(tack.i32, (n,))
    for group in groups:
        _piece_counts(group.view(), len(tables[int(group.shape.ID)]), counts)
    most = max((len(tables[int(g.shape.ID)]) for g in groups), default=0)
    odt = ids.for_output(data, most * width * n)
    starts = tack.field(odt, shape=(n,))
    count = exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.field(odt, shape=(count * width,))
    sources = tack.field(data.id_dtype, shape=(count,))
    for group in groups:
        table = np.asarray(tables[int(group.shape.ID)], np.int32)
        _split_cells(group.view(), _as_field(table.reshape(-1), tack.i32), len(table), width,
                     starts, connectivity, sources)
    return _pieces_topology(data, connectivity, sources, count, shape, width, fields)


def tetrahedralize(data, fields=None):
    """Every 3D cell as tetrahedra, by Viskores' tables (a hexahedron or voxel in 5,
    a wedge in 3, a pyramid in 2); other cells are left out. The points are the
    same; cell fields go to each cell's tetrahedra."""
    if not getattr(data.topology, "reference_cells", True):
        raise NotImplementedError("tetrahedralize takes cells of the linear shapes")
    return _split(data, "tetrahedra", fields)


def triangulate(data, fields=None):
    """Every 2D cell as triangles: a quad or pixel in 2, a polygon in a fan from its
    first point; other cells are left out. The points are the same; cell fields
    go to each cell's triangles."""
    t = data.topology
    if getattr(t, "reference_cells", True):
        return _split(data, "triangles", fields)
    if t.dimension != 2:
        raise NotImplementedError("triangulate takes 2D cells: polygons, not polyhedra")
    loop_offsets, loop_points = t.loops()
    n = data.num_cells
    odt = ids.for_output(data, 3 * int(loop_points.shape[0]))
    counts = tack.field(tack.i32, shape=(n,))
    starts = tack.field(odt, shape=(n,))
    if n:
        _fan_counts(loop_offsets, counts)
    count = exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.field(odt, shape=(3 * count,))
    sources = tack.field(data.id_dtype, shape=(count,))
    if count:
        _fans(loop_offsets, loop_points, starts, connectivity, sources)
    return _pieces_topology(data, connectivity, sources, count, shapes.TRIANGLE, 3, fields)


@tack.kernel
def _shrink_corners(positions, offsets, factor, n_cells):
    for c in range(n_cells):
        first = offsets[c]
        end = offsets[c + 1]
        center = positions[first]
        for k in range(first + 1, end):
            center += positions[k]
        center = center / (end - first)
        for k in range(first, end):
            positions[k] = center + factor * (positions[k] - center)


def shrink(data, factor=0.5, fields=None):
    """Every cell shrunk toward its centroid (the mean of its corners) by ``factor``,
    as Viskores' Shrink and VTK's vtkShrinkFilter. Viskores gives each cell its
    own copies of its points; here that is an ``L2`` geometry on the same
    topology, and point data becomes ``L2`` the same way (each cell's own copy
    of its corners' values). Everything else is kept, the topology's faces and
    edges with it."""
    from tack.data.algorithms import discontinuous

    if not 0 <= factor <= 1:
        raise ValueError(f"factor must lie in [0, 1], not {factor!r}")
    corners = discontinuous(data, Field(H1(data), data.geometry.values)
                            if isinstance(data.geometry.space, H1) else data.geometry)
    if data.num_cells:
        _shrink_corners(corners.values, corners.space.offsets, float(factor), data.num_cells)
    from tack.data.carry import selected

    out = {}
    for name in selected(data, fields):
        f = data.fields[name]
        space = f.space
        if isinstance(space, Values) and space.on == "points":
            f, space = Field(H1(data), f.values), H1(data)
        out[name] = discontinuous(data, f) if isinstance(space, H1) else f
    return DataSet(data.topology, corners, fields=out, sets=dict(data.sets))


def point_cloud(data, fields=None):
    """Every point as a vertex cell, with the point data (Viskores'
    ConvertToPointCloud); cell data is left behind."""
    flags = tack.full(tack.i32, (data.num_points,), 1) if data.num_points else \
        tack.field(tack.i32, shape=(0,))
    return _keep_points(data, flags, fields)


# ── Threshold ───────────────────────────────────────────────────────

@tack.kernel
def _keep_by_corners(cells, u, lower, upper, all_points, keep):
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
        keep[cells.entity_id(c)] = ok


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


def threshold(data, field, lower, upper, all_points=True, fields=None):
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
        return _polyhedral_threshold(data, field, lower, upper, all_points, fields)

    keep = tack.zeros(tack.i32, (data.num_cells,))
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _keep_by_corners(data.domain_view("cells", group), field.view(group), lower,
                             upper, 1 if all_points else 0, keep)
    return _keep_cells(data, keep, fields)


@tack.kernel
def _kept_sizes(cells, keep, sizes):
    for c in cells:
        e = cells.entity_id(c)
        sizes[e] = keep[e] * cells.NUM_POINTS


def _keep_cells(data, keep, fields=None):
    """The cells whose ``keep`` flag (i32, one per cell) is 1, with only the points
    they use, renumbered in order; fields carried by ``carry``. Polyhedral and
    polygonal topologies keep their cells' faces too."""
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_keep_cells(data, keep, fields)
    n = data.num_cells
    sizes = tack.zeros(tack.i32, (n,))
    for group in data.topology.groups():
        if group.count:
            _kept_sizes(group.view(), keep, sizes)
    idt = data.id_dtype
    slots = tack.field(idt, shape=(n,))
    starts = tack.field(idt, shape=(n,))
    count = exclusive_scan(keep, slots, n) if n else 0
    length = exclusive_scan(sizes, starts, n) if n else 0

    num_points = data.num_points
    types = tack.field(tack.u8, shape=(count,))
    offsets = tack.zeros(idt, (count + 1,))
    connectivity = tack.field(idt, shape=(length,))
    sources = tack.field(idt, shape=(count,))
    used = tack.zeros(tack.i32, (num_points,))
    for group in data.topology.groups():
        if group.count:
            _kept_cells(group.view(), keep, slots, starts, types, offsets, connectivity,
                        sources, used)
    _close_offsets(offsets, count, length)
    new_ids = tack.field(idt, shape=(num_points,))
    kept_points = exclusive_scan(used, new_ids, num_points) if num_points else 0
    if length:
        _renumber(connectivity, new_ids)
    kept = UnstructuredTopology(types, offsets, connectivity, num_points=kept_points)
    point_ids = tack.field(idt, shape=(kept_points,))
    if kept_points:
        _kept_ids(used, new_ids, point_ids)
    return carry(data, kept, points=Take(point_ids), cells=Take(sources), fields=fields)


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
def _kept_ids(flags, slots, out):
    for i in range(flags.shape[0]):
        if flags[i] == 1:
            out[slots[i]] = i


def _compact(flags, dtype):
    """``(new ids, kept ids, count)`` of a 0/1 field: each kept index's new number,
    and the kept indices in order, both ``dtype``."""
    n = flags.shape[0]
    slots = tack.field(dtype, shape=(n,))
    count = exclusive_scan(flags, slots, n) if n else 0
    kept = tack.field(dtype, shape=(count,))
    if count:
        _kept_ids(flags, slots, kept)
    return slots, kept, count


def _polyhedral_threshold(data, field, lower, upper, all_points, fields=None):
    """``threshold`` of a polyhedral or polygonal topology: whole cells, their faces
    kept once and numbered in their old order, points compacted."""

    t = data.topology
    by_cells = isinstance(field.space, Constant)
    values = arrays.materialize(field.values)
    n = data.num_cells
    keep = tack.zeros(tack.i32, (n,))
    for group in data.launch_groups("cells", []):
        if group.count:
            _polyhedral_keep(data.domain_view("cells", group), values, lower, upper,
                             1 if by_cells else 0, 1 if all_points else 0, keep)
    return _keep_cells(data, keep, fields)


def _polyhedral_keep_cells(data, keep, fields=None):
    """``_keep_cells`` of a polyhedral or polygonal topology: whole cells, their faces
    kept once and numbered in their old order, points compacted."""
    from tack.data.polyhedra import PolygonalTopology, PolyhedralTopology

    t = data.topology
    n = data.num_cells
    idt = data.id_dtype
    _, kept_cells, kept_n = _compact(keep, idt)
    point_used = tack.zeros(tack.i32, (data.num_points,))

    if t.dimension == 2:
        loop_offsets, loop_points = t.loops()
        sizes = tack.zeros(tack.i32, (n,))
        if n:
            _kept_entry_counts(loop_offsets, keep, sizes, n)
        starts = tack.field(idt, shape=(n + 1,))
        total = exclusive_scan(sizes, starts, n) if n else 0
        out_loops = tack.field(idt, shape=(total,))
        if total:
            _copy_loops(loop_offsets, loop_points, keep, starts, out_loops, point_used, n)
        point_new, kept_points, kept_np = _compact(point_used, idt)
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
        new_face, kept_faces, kept_nf = _compact(used, idt)
        sizes = tack.field(tack.i32, shape=(nf,))
        if nf:
            _kept_face_sizes(t.face_offsets, used, sizes)
        starts = tack.field(idt, shape=(nf + 1,))
        total = exclusive_scan(sizes, starts, nf) if nf else 0
        face_points = tack.field(idt, shape=(total,))
        flipped = tack.zeros(tack.i32, (kept_nf,))
        if nf:
            _copy_faces(t.face_offsets, t.face_points, used, owned, new_face, starts,
                        face_points, flipped, point_used)
        face_offsets = _gather_offsets(starts, kept_faces, total)
        counts = tack.zeros(tack.i32, (n,))
        if n:
            _kept_entry_counts(t.cell_offsets, keep, counts, n)
        entry_starts = tack.field(idt, shape=(n + 1,))
        entries = exclusive_scan(counts, entry_starts, n) if n else 0
        cell_faces = tack.field(idt, shape=(entries,))
        cell_sides = tack.field(tack.u8, shape=(entries,))
        if entries:
            _copy_entries(t.cell_offsets, t.cell_faces, t.cell_face_sides, keep, new_face,
                          flipped, entry_starts, cell_faces, cell_sides, n)
        cell_offsets = _gather_offsets(entry_starts, kept_cells, entries)
        point_new, kept_points, kept_np = _compact(point_used, idt)
        if total:
            _renumber_points(face_points, point_new)
        out = PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, cell_sides,
                                 num_points=kept_np)

    faces = Take(kept_faces, turned=flipped) if flipped is not None else None
    return carry(data, out, points=Take(kept_points), cells=Take(kept_cells), faces=faces,
                 fields=fields)


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
    out = tack.field(starts.dtype, shape=(kept.shape[0] + 1,))
    if kept.shape[0]:
        _offsets_of(starts, kept, out, total)
    else:
        out.fill(0)
    return out


# ── External faces ──────────────────────────────────────────────────

def external_faces(data, name="boundary", fields=None):
    """The faces of the 3D cells that only one cell has, as a surface dataset: the
    boundary face set (kept in ``data.sets[name]``) through ``extract_surface``.
    Each face faces out of its cell and carries that cell's cell fields; point
    fields stay on the same points."""
    boundary_faces(data, name)
    return extract_surface(data, name, fields=fields)
