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
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import _run_offsets, argsort, gather
from tack.data import arrays, shapes
from tack.data.algorithms import _like, _take, boundary_faces, extract_surface
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
        order = argsort(keys)
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


# ── External faces ──────────────────────────────────────────────────

def external_faces(data, name="boundary"):
    """The faces of the 3D cells that only one cell has, as a surface dataset: the
    boundary face set (kept in ``data.sets[name]``) through ``extract_surface``.
    Each face faces out of its cell and carries that cell's cell fields; point
    fields stay on the same points."""
    boundary_faces(data, name)
    return extract_surface(data, name)
