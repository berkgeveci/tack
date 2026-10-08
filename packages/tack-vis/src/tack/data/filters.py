"""Filters over datasets: each takes a ``DataSet`` and returns a new one.

- ``cell_centers``: one vertex per cell at its parametric center, its cell
  data becoming point data (``vtkCellCenters``).
- ``point_data_to_cell_data``: each cell averages its points' values
  (``vtkPointDataToCellData``).
- ``cell_data_to_point_data``: each point averages the values of the cells
  that use it (``vtkCellDataToPointData``), through the point-to-cell
  links: every (point, cell) incidence sorted by point, so each point's
  cells are one run, reduced serially -- reproducible, unlike atomics.
- ``external_faces``: the faces of the 3D cells that belong to only one
  cell, as triangles and quads in the cells' outward point order
  (``vtkGeometryFilter`` on 3D cells). Each face is keyed by its sorted
  point ids, two stable sorts bring a face's copies together, and the
  faces occurring once are counted, scanned and scattered.
- ``contour``: the triangles where point data crosses an isovalue in the
  3D cells, by VTK's case tables (``vtkContourFilter``). Each cell counts
  its triangles, a scan places them, and each writes its points,
  interpolated along the edges they lie on; the points are then merged by
  edge, so the surface is connected, and point data interpolated onto them.

Each runs a kernel per shape present (``for_each_shape``), written once
against the cell views' methods. Data arrays must be floating point; a
vector field averages per component.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import _run_offsets, argsort, gather, sort_by_key
from tack.data import shapes
from tack.data.cell_set import ExplicitCellSet, SingleTypeCellSet
from tack.data.dataset import DataSet, for_each_shape

__all__ = ["cell_centers", "cell_data_to_point_data", "contour", "external_faces",
           "point_data_to_cell_data", "point_links"]


def _like(values, n):
    """A new field of ``n`` elements shaped like ``values``'s: scalar or vector, same dtype."""
    width = getattr(values, "_vector_n", None)
    if width:
        return tack.Vector.field(width, values.dtype, shape=(n,))
    return tack.field(values.dtype, shape=(n,))


def _check_floating(name, values):
    if values.dtype not in (tack.f32, tack.f64):
        raise TypeError(f"array {name!r} is {values.dtype.name}; averaging needs f32 or f64")


def _views(data):
    """The dataset's cell views, with each one's first position in a per-view layout."""
    views = data.cells.views(data.points)
    starts = np.concatenate([[0], np.cumsum([v.num_cells for v in views])]).astype(int)
    return list(zip(views, starts[:-1].tolist()))


# ── Cell centers ────────────────────────────────────────────────────

@tack.kernel
def _centers(cells, out):
    for c in cells:
        pc = cells.parametric_center()
        x = tack.Vector([0.0, 0.0, 0.0])
        for j in range(cells.NUM_POINTS):
            x += cells.shape_function(j, pc) * cells.point(c, j)
        out[cells.cell_id(c)] = x


@tack.kernel
def _vertex_rows(ids):
    for i in range(ids.shape[0]):
        ids[i, 0] = i


def cell_centers(data):
    """One vertex per cell, at the position of its parametric center.

    The cell data becomes the new dataset's point data, as in VTK.
    """
    n = data.num_cells
    points = tack.Vector.field(3, data.points.dtype, shape=(n,))
    if n:
        for_each_shape(_centers, data, points)
    rows = tack.field(tack.i32, shape=(n, 1))
    if n:
        _vertex_rows(rows)
    return DataSet(points, SingleTypeCellSet(shapes.Vertex, rows),
                   point_data=dict(data.cell_data))


# ── Point data to cell data ─────────────────────────────────────────

@tack.kernel
def _average_points(cells, values, out):
    for c in cells:
        total = values[cells.point_id(c, 0)]
        for j in range(1, cells.NUM_POINTS):
            total += values[cells.point_id(c, j)]
        out[cells.cell_id(c)] = total / cells.NUM_POINTS


def point_data_to_cell_data(data, names=None):
    """Each cell gets the average of its points' values, for each point array.

    ``names`` picks the arrays (all of them by default). The result has the
    same points, cells and point data, and the averages as its cell data.
    """
    cell_data = {}
    for name in names if names is not None else list(data.point_data):
        values = data.point_data[name]
        _check_floating(name, values)
        out = _like(values, data.num_cells)
        if data.num_cells:
            for_each_shape(_average_points, data, values, out)
        cell_data[name] = out
    return DataSet(data.points, data.cells, point_data=dict(data.point_data),
                   cell_data=cell_data)


# ── Point-to-cell links and cell data to point data ─────────────────

@tack.kernel
def _incidences(cells, start, point_keys, cell_values):
    for c in cells:
        cell = cells.cell_id(c)
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            point_keys[first + j] = cells.point_id(c, j)
            cell_values[first + j] = cell


def point_links(data):
    """The cells using each point: ``(points, cells, offsets, count)``.

    ``points`` holds the ``count`` distinct point ids that some cell uses,
    ascending; the cells using ``points[r]`` are ``cells[offsets[r]:offsets[r + 1]]``.
    Built by sorting every (point, cell) incidence by point, once per cell
    set (topology does not change), and kept on it.
    """
    cached = getattr(data.cells, "_point_links", None)
    if cached is not None:
        return cached
    groups = _views(data)
    total = sum(view.num_cells * view.NUM_POINTS for view, _ in groups)
    point_keys = tack.field(tack.i32, shape=(total,))
    cell_values = tack.field(tack.i32, shape=(total,))
    first = 0
    for view, _ in groups:
        if view.num_cells:
            _incidences(view, first, point_keys, cell_values)
        first += view.num_cells * view.NUM_POINTS
    keys, cells = sort_by_key(point_keys, cell_values)
    offsets, count = _run_offsets(keys, total)
    points = gather(keys, offsets, count) if count else tack.field(tack.i32, shape=(0,))
    links = (points, cells, offsets, count)
    data.cells._point_links = links
    return links


@tack.kernel
def _average_cells(points, cells, offsets, values, out, count):
    for r in range(count):
        start = offsets[r]
        end = offsets[r + 1]
        total = values[cells[start]]
        for i in range(start + 1, end):
            total += values[cells[i]]
        out[points[r]] = total / (end - start)


def cell_data_to_point_data(data, names=None):
    """Each point gets the average of the values of the cells using it.

    ``names`` picks the cell arrays (all by default). Points no cell uses
    get zero. The result has the same points, cells and cell data, and the
    averages as its point data. The sums run in a fixed order, so the
    result is the same on every run.
    """
    points, cells, offsets, count = point_links(data)
    point_data = {}
    for name in names if names is not None else list(data.cell_data):
        values = data.cell_data[name]
        _check_floating(name, values)
        out = _like(values, data.num_points)
        out.fill(0.0)
        if count:
            _average_cells(points, cells, offsets, values, out, count)
        point_data[name] = out
    return DataSet(data.points, data.cells, point_data=point_data,
                   cell_data=dict(data.cell_data))


# ── External faces ──────────────────────────────────────────────────

_NO_POINT = tack.constant(0x7FFFFFFF, tack.i32)


@tack.func
def _order(a, b):
    return min(a, b), max(a, b)


@tack.kernel
def _faces(cells, start, rows, kinds, owners):
    for c in cells:
        cell = cells.cell_id(c)
        first = start + cells.index(c) * cells.NUM_FACES
        for f in range(cells.NUM_FACES):
            k = first + f
            quad = cells.face_num_points(f) == 4
            p0 = cells.point_id(c, cells.face_point(f, 0))
            p1 = cells.point_id(c, cells.face_point(f, 1))
            p2 = cells.point_id(c, cells.face_point(f, 2))
            p3 = cells.point_id(c, cells.face_point(f, 3)) if quad else _NO_POINT
            rows[k] = [p0, p1, p2, p3]
            kinds[k] = cells.face_shape(f)
            owners[k] = cell


@tack.kernel
def _face_keys(rows, hi, lo):
    # A face's key is its point ids sorted, whichever cell lists it.
    for k in range(hi.shape[0]):
        row = rows[k]
        a, b = _order(row[0], row[1])
        c, d = _order(row[2], row[3])
        a, c = _order(a, c)
        b, d = _order(b, d)
        b, c = _order(b, c)
        hi[k] = (tack.u64(a) << tack.u64(32)) | tack.u64(b)
        lo[k] = (tack.u64(c) << tack.u64(32)) | tack.u64(d)


@tack.kernel
def _flag_single(hi, lo, order, flags, sizes, kinds, n):
    for i in range(n):
        same_before = i > 0 and hi[i] == hi[i - 1] and lo[i] == lo[i - 1]
        same_after = i < n - 1 and hi[i] == hi[i + 1] and lo[i] == lo[i + 1]
        single = 0 if same_before or same_after else 1
        flags[i] = single
        sizes[i] = single * (3 if kinds[order[i]] == shapes.TRIANGLE else 4)


@tack.kernel
def _write_faces(order, flags, slots, starts, rows, kinds, owners,
                 types, offsets, connectivity, cell_ids, n, total):
    for i in range(n):
        if flags[i] == 1:
            face = order[i]
            slot = slots[i]
            at = starts[i]
            row = rows[face]
            kind = kinds[face]
            offsets[slot] = at
            cell_ids[slot] = owners[face]
            if kind == shapes.TRIANGLE:
                types[slot] = tack.u8(shapes.TRIANGLE)
                connectivity[at] = row[0]
                connectivity[at + 1] = row[1]
                connectivity[at + 2] = row[2]
            else:
                types[slot] = tack.u8(shapes.QUAD)
                connectivity[at] = row[0]
                connectivity[at + 1] = row[1]
                # A voxel's faces are pixels, x-fastest: as a quad, 0 1 3 2.
                if kind == shapes.PIXEL:
                    connectivity[at + 2] = row[3]
                    connectivity[at + 3] = row[2]
                else:
                    connectivity[at + 2] = row[2]
                    connectivity[at + 3] = row[3]
        if i == 0:
            offsets[slots[n - 1] + flags[n - 1]] = total


def external_faces(data):
    """The faces of the 3D cells that only one cell has, as a surface dataset.

    Returns a dataset with the same points and an ``ExplicitCellSet`` of
    triangles and quads, each in its cell's face order, which faces out of
    the cell (a voxel's pixel faces become quads). Each face carries its
    cell's cell data. Cells of lower dimension contribute nothing.
    """
    groups = [(view, start) for view, start in _views(data) if view.NUM_FACES]
    total = sum(view.num_cells * view.NUM_FACES for view, _ in groups)
    hi = tack.field(tack.u64, shape=(total,))
    lo = tack.field(tack.u64, shape=(total,))
    rows = tack.Vector.field(4, tack.i32, shape=(total,))
    kinds = tack.field(tack.i32, shape=(total,))
    owners = tack.field(tack.i32, shape=(total,))
    first = 0
    for view, _ in groups:
        if view.num_cells:
            _faces(view, first, rows, kinds, owners)
        first += view.num_cells * view.NUM_FACES

    if total:
        _face_keys(rows, hi, lo)
        # Two stable sorts order the faces by (hi, lo).
        by_lo = argsort(lo)
        hi_by_lo = gather(hi, by_lo)
        by_hi = argsort(hi_by_lo)
        order = gather(by_lo, by_hi)
        hi_sorted = gather(hi_by_lo, by_hi)
        lo_sorted = gather(lo, order)
        flags = tack.field(tack.i32, shape=(total,))
        sizes = tack.field(tack.i32, shape=(total,))
        _flag_single(hi_sorted, lo_sorted, order, flags, sizes, kinds, total)
        slots = tack.field(tack.i32, shape=(total,))
        starts = tack.field(tack.i32, shape=(total,))
        count = exclusive_scan(flags, slots, total)
        length = exclusive_scan(sizes, starts, total)
    else:
        count = length = 0

    types = tack.field(tack.u8, shape=(count,))
    offsets = tack.field(tack.i32, shape=(count + 1,))
    connectivity = tack.field(tack.i32, shape=(length,))
    cell_ids = tack.field(tack.i32, shape=(count,))
    if total:
        _write_faces(order, flags, slots, starts, rows, kinds, owners,
                     types, offsets, connectivity, cell_ids, total, length)
    else:
        offsets.fill(0)
    cell_data = {name: _take(values, cell_ids, count)
                 for name, values in data.cell_data.items()}
    return DataSet(data.points, ExplicitCellSet(types, offsets, connectivity),
                   point_data=dict(data.point_data), cell_data=cell_data)


@tack.kernel
def _gather_rows(values, ids, out):
    for i in range(ids.shape[0]):
        out[i] = values[ids[i]]


def _take(values, ids, count):
    """``values[ids]`` for a scalar or vector field."""
    out = _like(values, count)
    if count:
        _gather_rows(values, ids, out)
    return out


# ── Contour ─────────────────────────────────────────────────────────

# A cell's contour case: bit j is 1 when point j is at or above the isovalue.
# Computed in each kernel: a cell view cannot be passed to a device function.

@tack.kernel
def _contour_counts(cells, values, isovalue, counts):
    for c in cells:
        case = 0
        for j in range(cells.NUM_POINTS):
            if values[cells.point_id(c, j)] >= isovalue:
                case |= 1 << j
        counts[cells.cell_id(c)] = cells.contour_count(case)


@tack.kernel
def _contour_points(cells, values, isovalue, starts, points, edges, weights):
    for c in cells:
        case = 0
        for j in range(cells.NUM_POINTS):
            if values[cells.point_id(c, j)] >= isovalue:
                case |= 1 << j
        first = 3 * starts[cells.cell_id(c)]
        for k in range(3 * cells.contour_count(case)):
            e = cells.contour_edge(case, k)
            ja = cells.edge_point(e, 0)
            jb = cells.edge_point(e, 1)
            # Always from the lower point id, so the cells sharing an edge
            # compute the same point, bit for bit.
            if cells.point_id(c, jb) < cells.point_id(c, ja):
                ja, jb = jb, ja
            a = cells.point_id(c, ja)
            b = cells.point_id(c, jb)
            w = (isovalue - values[a]) / (values[b] - values[a])
            xa = cells.point(c, ja)
            points[first + k] = xa + w * (cells.point(c, jb) - xa)
            edges[first + k] = [a, b]
            weights[first + k] = w


@tack.kernel
def _edge_keys(edges, keys):
    for i in range(keys.shape[0]):
        ab = edges[i]
        keys[i] = (tack.u64(ab[0]) << tack.u64(32)) | tack.u64(ab[1])


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
def _interpolate_edges(values, edges, weights, out):
    for i in range(out.shape[0]):
        ab = edges[i]
        va = values[ab[0]]
        out[i] = va + weights[i] * (values[ab[1]] - va)


def contour(data, values, isovalue, merge_points=True):
    """The surface where point data ``values`` crosses ``isovalue``, as triangles.

    ``values`` names a point data array, or is a scalar field with one value
    per point. Each 3D cell contributes the triangles of VTK's case tables
    for it (``vtkContourFilter``): a point's bit is 1 when its value is at
    least the isovalue, and the triangles face away from the larger values.
    Cells of lower dimension contribute nothing.

    Points are interpolated along cell edges and, with ``merge_points``,
    merged by edge, so triangles sharing an edge share its point; otherwise
    every triangle has its own three. The result's point data is the
    input's, interpolated onto the new points.
    """
    if isinstance(values, str):
        values = data.point_data[values]
    _check_floating("values", values)
    if getattr(values, "_vector_n", None):
        raise TypeError("contour needs one value per point, not a vector field")
    groups = [(view, start) for view, start in _views(data) if view.CONTOUR_TRIANGLES]
    counts = tack.zeros(tack.i32, (data.num_cells,))
    for view, _ in groups:
        if view.num_cells:
            _contour_counts(view, values, isovalue, counts)
    starts = tack.field(tack.i32, shape=(data.num_cells,))
    triangles = exclusive_scan(counts, starts, data.num_cells) if data.num_cells else 0

    n = 3 * triangles
    dtype = data.points.dtype
    points = tack.Vector.field(3, dtype, shape=(n,))
    edges = tack.Vector.field(2, tack.i32, shape=(n,))
    weights = tack.field(values.dtype, shape=(n,))
    for view, _ in groups:
        if view.num_cells:
            _contour_points(view, values, isovalue, starts, points, edges, weights)

    if merge_points and n:
        keys = tack.field(tack.u64, shape=(n,))
        _edge_keys(edges, keys)
        order = argsort(keys)
        offsets, count = _run_offsets(gather(keys, order), n)
        merged = tack.Vector.field(3, dtype, shape=(count,))
        merged_edges = tack.Vector.field(2, tack.i32, shape=(count,))
        merged_weights = tack.field(values.dtype, shape=(count,))
        _first_of_runs(order, offsets, points, edges, weights, merged, merged_edges,
                       merged_weights, count)
        run_of = tack.field(tack.i32, shape=(n,))
        _run_of_each(order, offsets, run_of, count)
        rows = run_of
        points, edges, weights = merged, merged_edges, merged_weights
    else:
        rows = tack.arange(n, tack.i32)

    connectivity = tack.field(tack.i32, shape=(triangles, 3))
    if triangles:
        _copy_rows(rows, connectivity)
    point_data = {}
    for name, array in data.point_data.items():
        if array.dtype in (tack.f32, tack.f64):
            out = _like(array, points.shape[0] // 3)
            if out.shape[0]:
                _interpolate_edges(array, edges, weights, out)
            point_data[name] = out
    return DataSet(points, SingleTypeCellSet(shapes.Triangle, connectivity),
                   point_data=point_data)


@tack.kernel
def _copy_rows(flat, rows):
    for t in range(rows.shape[0]):
        rows[t, 0] = flat[3 * t]
        rows[t, 1] = flat[3 * t + 1]
        rows[t, 2] = flat[3 * t + 2]

