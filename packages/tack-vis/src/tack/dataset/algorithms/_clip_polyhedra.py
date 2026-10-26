"""``clip`` of a polyhedral topology: each cut cell becomes the polyhedron of its kept
part -- its faces clipped, plus its iso-polygons (López's, as ``contour`` traces
them) as caps -- so faces two cells share stay one face, and the output is a
conforming ``PolyhedralTopology``."""

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids
from tack.dataset.algorithms._common import _close, _crossing
from tack.dataset.buckets import bucket_order, run_offsets
from tack.dataset.carry import Interpolate, Take, carry


@tack.kernel
def _face_sizes(face_offsets, face_points, values, iso, sizes, kept):
    # Each face's clipped polygon: its kept points, and a crossing on each edge
    # strictly between the sides (a point at the value is kept, so an edge from
    # it has no other crossing). Below is not at or above, so a NaN is below,
    # as contour's tracing and the shape path's cases have it. Under three
    # points, the face is gone.
    for f in range(sizes.shape[0]):
        start = face_offsets[f]
        n = face_offsets[f + 1] - start
        k = 0
        for j in range(n):
            va = values[face_points[start + j]]
            vb = values[face_points[start + (j + 1 if j + 1 < n else 0)]]
            if va >= iso:
                k += 1
            if (va > iso and not vb >= iso) or (not va >= iso and vb > iso):
                k += 1
        sizes[f] = k if k >= 3 else 0
        kept[f] = 1 if k >= 3 else 0


@tack.kernel
def _face_keys(face_offsets, face_points, values, iso, sizes, starts, keys):
    # The clipped polygon's points, named as contour names crossings: (a, a) for a
    # kept point, (low, high) for an edge's crossing.
    for f in range(sizes.shape[0]):
        if sizes[f] > 0:
            start = face_offsets[f]
            n = face_offsets[f + 1] - start
            at = starts[f]
            for j in range(n):
                a = face_points[start + j]
                b = face_points[start + (j + 1 if j + 1 < n else 0)]
                va = values[a]
                vb = values[b]
                if va >= iso:
                    keys[at] = [a, a]
                    at += 1
                if (va > iso and not vb >= iso) or (not va >= iso and vb > iso):
                    keys[at] = [min(a, b), max(a, b)]
                    at += 1


@tack.kernel
def _cap_ranges(poly_cells, polygons, first, last):
    # Polygons come in cell order: each cell's run of them.
    for p in range(polygons):
        c = poly_cells[p]
        if p == 0 or poly_cells[p - 1] != c:
            first[c] = p
        if p == polygons - 1 or poly_cells[p + 1] != c:
            last[c] = p + 1


@tack.kernel
def _cell_counts(cell_offsets, cell_faces, face_offsets, face_points, values, iso, sizes,
                 first, last, counts, kept):
    # A cell is kept when one of its points is strictly on the kept side (a cell
    # only touching the value has no volume there); its faces are its clipped
    # faces left and its caps.
    for c in range(counts.shape[0]):
        inside = 0
        faces = 0
        for i in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[i]
            if sizes[f] > 0:
                faces += 1
            for j in range(face_offsets[f], face_offsets[f + 1]):
                if values[face_points[j]] > iso:
                    inside = 1
        counts[c] = faces + last[c] - first[c] if inside == 1 else 0
        kept[c] = inside


@tack.kernel
def _point_runs(keys, order, offsets, values, iso, ends, weights, point_of, count):
    # One output point per run of equal names: a kept point (weight 0) or an
    # edge's crossing, interpolated from its lower point id as contour does.
    for r in range(count):
        key = keys[order[offsets[r]]]
        a = key[0]
        b = key[1]
        ends[r] = [a, b]
        weights[r] = 0.0 if a == b else _crossing(iso, values[a], values[b])
        for i in range(offsets[r], offsets[r + 1]):
            point_of[order[i]] = r


@tack.kernel
def _side_zero_kept(cell_offsets, cell_faces, cell_face_sides, kept, out, n_cells):
    # Whether each face's side-0 cell (the one it is wound out of) is kept.
    for c in range(n_cells):
        for i in range(cell_offsets[c], cell_offsets[c + 1]):
            if cell_face_sides[i] == 0:
                out[cell_faces[i]] = kept[c]


@tack.kernel
def _faces_out(sizes, kept, starts, slots, point_of, side_zero, out_offsets, out_points,
               n_faces):
    # The clipped faces left, in face order, their points renumbered; a face whose
    # side-0 cell is gone is walked backwards, out of its other cell, which now
    # has it on side 0.
    for f in range(n_faces):
        if kept[f] == 1:
            s = slots[f]
            out_offsets[s] = starts[f]
            n = sizes[f]
            for j in range(n):
                k = j if side_zero[f] == 1 else n - 1 - j
                out_points[starts[f] + j] = point_of[starts[f] + k]


@tack.kernel
def _caps_out(poly_starts, vertices, point_of, face_keys, first_face, first_point,
              out_offsets, out_points, polygons):
    # Each iso-polygon as a cap face after the clipped faces, wound as López traced
    # it: toward the lower values, out of the kept part.
    for p in range(polygons):
        out_offsets[first_face + p] = first_point + poly_starts[p]
        for j in range(poly_starts[p], poly_starts[p + 1]):
            out_points[first_point + j] = point_of[face_keys + vertices[j]]


@tack.kernel
def _cells_out(cell_offsets, cell_faces, cell_face_sides, sizes, face_slots, side_zero, counts,
               kept, cell_slots, starts, first, last, first_cap, out_faces, out_sides, sources,
               n_cells):
    for c in range(n_cells):
        if kept[c] == 1:
            s = cell_slots[c]
            sources[s] = c
            at = starts[c]
            for i in range(cell_offsets[c], cell_offsets[c + 1]):
                f = cell_faces[i]
                if sizes[f] > 0:
                    out_faces[at] = face_slots[f]
                    out_sides[at] = cell_face_sides[i] if side_zero[f] == 1 else 0
                    at += 1
            for p in range(first[c], last[c]):
                out_faces[at] = first_cap + p
                out_sides[at] = 0
                at += 1


@tack.kernel
def _kept_offsets(counts, kept, starts, slots, out, n_cells):
    for c in range(n_cells):
        if kept[c] == 1:
            out[slots[c]] = starts[c]


def clip_polyhedra(data, values, iso, inside, fields):
    """``clip`` of ``data``'s polyhedral topology by point ``values`` (a materialized
    scalar array) at ``iso``: the part at or above it, or with ``inside`` at or
    below it."""
    from tack.dataset.algorithms.contour import _lopez_polygons
    from tack.dataset.polyhedra import PolyhedralTopology

    t = data.topology
    if inside:
        negated = tack.empty_like(values)
        if arrays.size_of(values):
            _negate(values, negated, float("inf"))
        values, iso = negated, -iso
    idt = data.id_dtype
    weight_dtype = arrays.dtype_of(values)
    levels = tack.array([iso], weight_dtype)
    (poly_starts, vertices, poly_cells, _, polygons, cap_ends, _, _,
     cap_count) = _lopez_polygons(data, values, levels, 1, weight_dtype)

    n_faces, n_cells = int(t.face_offsets.shape[0]) - 1, data.num_cells
    sizes = tack.empty((n_faces,), tack.i32)
    face_kept = tack.empty((n_faces,), tack.i32)
    if n_faces:
        _face_sizes(t.face_offsets, t.face_points, values, iso, sizes, face_kept)
    odt = ids.for_output(data, 2 * int(t.face_points.shape[0]) + cap_count + n_faces)
    starts = tack.empty((n_faces,), odt)
    face_keys = exclusive_scan(sizes, starts, n_faces) if n_faces else 0
    face_slots = tack.empty((n_faces,), odt)
    kept_faces = exclusive_scan(face_kept, face_slots, n_faces) if n_faces else 0

    # Every point's name: the clipped faces' points, then the caps' iso-vertices.
    keys = tack.empty((face_keys + cap_count,), tack.vec(2, idt))
    if face_keys:
        _face_keys(t.face_offsets, t.face_points, values, iso, sizes, starts, keys)
    if cap_count:
        _place(cap_ends, keys, face_keys, cap_count)
    total = face_keys + cap_count
    point_of = tack.empty((total,), odt)
    count = 0
    if total:
        order, _ = bucket_order(keys, data.num_points, odt)
        runs, count = run_offsets(keys, order)
    ends = tack.empty((count,), tack.vec(2, idt))
    weights = tack.empty((count,), weight_dtype)
    if count:
        _point_runs(keys, order, runs, values, iso, ends, weights, point_of, count)

    # Cells: each kept one's faces left and caps.
    first = tack.zeros((n_cells,), odt)
    last = tack.zeros((n_cells,), odt)
    if polygons:
        _cap_ranges(poly_cells, polygons, first, last)
    counts = tack.empty((n_cells,), tack.i32)
    kept = tack.empty((n_cells,), tack.i32)
    if n_cells:
        _cell_counts(t.cell_offsets, t.cell_faces, t.face_offsets, t.face_points, values, iso,
                     sizes, first, last, counts, kept)
    side_zero = tack.zeros((n_faces,), tack.i32)
    if n_cells:
        _side_zero_kept(t.cell_offsets, t.cell_faces, t.cell_face_sides, kept, side_zero,
                        n_cells)

    # Faces: the clipped ones left, then the caps.
    cap_points = int(poly_starts.numpy()[-1]) if polygons else 0
    out_offsets = tack.empty((kept_faces + polygons + 1,), odt)
    out_points = tack.empty((face_keys + cap_points,), odt)
    if n_faces:
        _faces_out(sizes, face_kept, starts, face_slots, point_of, side_zero, out_offsets,
                   out_points, n_faces)
    if polygons:
        _caps_out(poly_starts, vertices, point_of, face_keys, kept_faces, face_keys,
                  out_offsets, out_points, polygons)
    _close(out_offsets, kept_faces + polygons, face_keys + cap_points)

    cell_starts = tack.empty((n_cells,), odt)
    entries = exclusive_scan(counts, cell_starts, n_cells) if n_cells else 0
    cell_slots = tack.empty((n_cells,), odt)
    kept_cells = exclusive_scan(kept, cell_slots, n_cells) if n_cells else 0
    out_cell_offsets = tack.empty((kept_cells + 1,), odt)
    out_faces = tack.empty((entries,), odt)
    out_sides = tack.empty((entries,), tack.u8)
    sources = tack.empty((kept_cells,), idt)
    if n_cells:
        _kept_offsets(counts, kept, cell_starts, cell_slots, out_cell_offsets, n_cells)
        _cells_out(t.cell_offsets, t.cell_faces, t.cell_face_sides, sizes, face_slots,
                   side_zero, counts, kept, cell_slots, cell_starts, first, last, kept_faces,
                   out_faces, out_sides, sources, n_cells)
    _close(out_cell_offsets, kept_cells, entries)
    topology = PolyhedralTopology(out_offsets, out_points, out_cell_offsets, out_faces,
                                  out_sides, num_points=count, id_dtype=odt)
    return carry(data, topology, points=Interpolate(ends, weights), cells=Take(sources),
                 fields=fields)


@tack.kernel
def _negate(values, out, infinity):
    # A NaN is below every level, so after negating it must be above them all:
    # the inside part keeps it, as the shape path's inside clip does.
    for i in range(out.shape[0]):
        v = values[i]
        out[i] = -v if v == v else infinity


@tack.kernel
def _place(src, dst, offset, n):
    for i in range(n):
        dst[offset + i] = src[i]
