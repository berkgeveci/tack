"""Keeping cells or points: the compaction behind ``threshold``,
``extract_cells``, ``extract_geometry``, ``mask`` and the point extractions,
which keeps the used points, renumbers them and carries fields -- and on a
polyhedral topology keeps the kept cells' faces once."""

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.arrays import CountingArray
from tack.dataset import ids, shapes
from tack.dataset.algorithms._common import _close, _compact, _kept_ids
from tack.dataset.carry import Take, carry
from tack.dataset.dataset import DataSet
from tack.dataset.topology import UnstructuredTopology, uniform_cells


@tack.kernel
def _kept_sizes(cells, keep, sizes):
    for c in cells:
        e = cells.entity_id(c)
        sizes[e] = keep[e] * cells.NUM_POINTS


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
def _kept_uniform_cells(cells, keep, slots, starts, connectivity, sources, used):
    # As _kept_cells, for a topology of one shape, whose types and offsets the
    # output takes as implicit arrays.
    for c in cells:
        cell = cells.entity_id(c)
        if keep[cell] == 1:
            slot = slots[cell]
            at = starts[cell]
            sources[slot] = cell
            for j in range(cells.NUM_POINTS):
                p = cells.point_id(c, j)
                connectivity[at + j] = p
                tack.atomic_max(used, p, 1)


@tack.kernel
def _renumber(connectivity, new_ids):
    for k in range(connectivity.shape[0]):
        connectivity[k] = new_ids[connectivity[k]]


@tack.kernel
def _flip(flags):
    for i in range(flags.shape[0]):
        flags[i] = 1 - flags[i]


def _complement(flags):
    """Turn a 0/1 field's selection inside out, in place."""
    if flags.shape[0]:
        _flip(flags)


@tack.kernel
def _or_flags(extra, used):
    for i in range(used.shape[0]):
        if extra[i] == 1:
            used[i] = 1


def _keep_cells(data, keep, fields=None, keep_points=None):
    """The cells whose ``keep`` flag (i32, one per cell) is 1, with only the points
    they use, renumbered in order -- and the points whose ``keep_points`` flag is
    1, used or not; fields carried by ``carry``. Polyhedral topologies keep
    their cells' faces too; polygonal and polyline ones their loops and chains.
    Hidden and refined cells (``HIDDENCELL``, ``REFINEDCELL``) are never kept."""
    from tack.dataset.ghosts import without_hidden

    keep = without_hidden(data, keep)
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_keep_cells(data, keep, fields, keep_points)
    n = data.num_cells
    sizes = tack.empty((n,), tack.i32)
    for group in data.topology.groups():
        if group.count:
            _kept_sizes(group.view(), keep, sizes)
    idt = data.id_dtype
    slots = tack.empty((n,), idt)
    starts = tack.empty((n,), idt)
    count = exclusive_scan(keep, slots, n) if n else 0
    length = exclusive_scan(sizes, starts, n) if n else 0

    num_points = data.num_points
    connectivity = tack.empty((length,), idt)
    sources = tack.empty((count,), idt)
    used = tack.zeros((num_points,), tack.i32)
    present = [g for g in data.topology.groups() if g.count]
    if len(present) == 1:
        # One shape in, one shape out: the output's types and offsets are implicit.
        types, offsets = uniform_cells(int(present[0].shape.ID), count, idt)
        _kept_uniform_cells(present[0].view(), keep, slots, starts, connectivity, sources,
                            used)
    else:
        types = tack.empty((count,), tack.u8)
        offsets = tack.empty((count + 1,), idt)
        for group in present:
            _kept_cells(group.view(), keep, slots, starts, types, offsets, connectivity,
                        sources, used)
        _close(offsets, count, length)
    if keep_points is not None and num_points:
        _or_flags(keep_points, used)
    new_ids = tack.empty((num_points,), idt)
    kept_points = exclusive_scan(used, new_ids, num_points) if num_points else 0
    if length:
        _renumber(connectivity, new_ids)
    kept = UnstructuredTopology(types, offsets, connectivity, num_points=kept_points)
    point_ids = tack.empty((kept_points,), idt)
    if kept_points:
        _kept_ids(used, new_ids, point_ids)
    return carry(data, kept, points=Take(point_ids), cells=Take(sources), fields=fields)


def _polyhedral_keep_cells(data, keep, fields=None, keep_points=None):
    """``_keep_cells`` of a polyhedral or polygonal topology: whole cells, their faces
    kept once and numbered in their old order, points compacted."""
    from tack.dataset.polyhedra import PolygonalTopology, PolyhedralTopology, PolylineTopology

    t = data.topology
    n = data.num_cells
    idt = data.id_dtype
    _, kept_cells, kept_n = _compact(keep, idt)
    point_used = tack.zeros((data.num_points,), tack.i32)

    if t.dimension < 3:
        # Polygons' loops or polylines' chains, copied whole.
        loop_offsets, loop_points = t.loops() if t.dimension == 2 else t.lines()
        sizes = tack.empty((n,), tack.i32)
        if n:
            _kept_entry_counts(loop_offsets, keep, sizes, n)
        starts = tack.empty((n + 1,), idt)
        total = exclusive_scan(sizes, starts, n) if n else 0
        out_loops = tack.empty((total,), idt)
        if total:
            _copy_loops(loop_offsets, loop_points, keep, starts, out_loops, point_used, n)
        if keep_points is not None and data.num_points:
            _or_flags(keep_points, point_used)
        point_new, kept_points, kept_np = _compact(point_used, idt)
        if total:
            _renumber_points(out_loops, point_new)
        offsets = _gather_offsets(starts, kept_cells, total)
        kind = PolygonalTopology if t.dimension == 2 else PolylineTopology
        out = kind(offsets, out_loops, num_points=kept_np)
        flipped = None
    else:
        nf = t.num_faces
        used = tack.zeros((nf,), tack.i32)
        owned = tack.zeros((nf,), tack.i32)
        if n:
            _kept_faces(t.cell_offsets, t.cell_faces, t.cell_face_sides, keep, used, owned, n)
        new_face, kept_faces, kept_nf = _compact(used, idt)
        sizes = tack.empty((nf,), tack.i32)
        if nf:
            _kept_face_sizes(t.face_offsets, used, sizes)
        starts = tack.empty((nf + 1,), idt)
        total = exclusive_scan(sizes, starts, nf) if nf else 0
        face_points = tack.empty((total,), idt)
        flipped = tack.zeros((kept_nf,), tack.i32)
        if nf:
            _copy_faces(t.face_offsets, t.face_points, used, owned, new_face, starts,
                        face_points, flipped, point_used)
        face_offsets = _gather_offsets(starts, kept_faces, total)
        counts = tack.empty((n,), tack.i32)
        if n:
            _kept_entry_counts(t.cell_offsets, keep, counts, n)
        entry_starts = tack.empty((n + 1,), idt)
        entries = exclusive_scan(counts, entry_starts, n) if n else 0
        cell_faces = tack.empty((entries,), idt)
        cell_sides = tack.empty((entries,), tack.u8)
        if entries:
            _copy_entries(t.cell_offsets, t.cell_faces, t.cell_face_sides, keep, new_face,
                          flipped, entry_starts, cell_faces, cell_sides, n)
        cell_offsets = _gather_offsets(entry_starts, kept_cells, entries)
        if keep_points is not None and data.num_points:
            _or_flags(keep_points, point_used)
        point_new, kept_points, kept_np = _compact(point_used, idt)
        if total:
            _renumber_points(face_points, point_new)
        out = PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, cell_sides,
                                 num_points=kept_np)

    faces = Take(kept_faces, turned=flipped) if flipped is not None else None
    return carry(data, out, points=Take(kept_points), cells=Take(kept_cells), faces=faces,
                 fields=fields)


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
def _copy_loops(loop_offsets, loop_points, keep, starts, out, point_used, n_cells):
    for c in range(n_cells):
        if keep[c] == 1:
            at = starts[c]
            for j in range(loop_offsets[c], loop_offsets[c + 1]):
                out[at] = loop_points[j]
                tack.atomic_max(point_used, loop_points[j], 1)
                at += 1


@tack.kernel
def _offsets_of(starts, kept, out, total, n):
    # The count comes as a scalar: a field's length used inside the loop would
    # be compiled in, and every output size would compile anew.
    for i in range(n):
        out[i] = starts[kept[i]]
        if i == 0:
            out[n] = total


def _gather_offsets(starts, kept, total):
    """CSR offsets of the kept rows: each kept row's start in the compacted array, and
    ``total`` at the end."""
    out = tack.empty((kept.shape[0] + 1,), starts.dtype)
    if kept.shape[0]:
        _offsets_of(starts, kept, out, total, int(kept.shape[0]))
    else:
        out.fill(0)
    return out


def _vertex_cells(count, id_dtype):
    """``count`` vertex cells, cell ``i`` at point ``i``: types, offsets and
    connectivity all implicit, nothing stored."""
    types, offsets = uniform_cells(shapes.VERTEX, count, id_dtype)
    return types, offsets, CountingArray(count, dtype=id_dtype)


def _keep_points(data, flags, fields):
    """The points whose ``flags`` (i32, one per point) is 1, each a vertex cell; point
    fields come along, cell fields do not."""
    _, kept, count = _compact(flags, data.id_dtype)
    vertices = UnstructuredTopology(*_vertex_cells(count, data.id_dtype), num_points=count)
    return carry(data, vertices, points=Take(kept), fields=fields)


@tack.kernel
def _flag_range(values, lower, upper, flags):
    for i in range(flags.shape[0]):
        v = values[i]
        flags[i] = 1 if lower <= v and v <= upper else 0


def _point_flags(data, values, lower, upper):
    flags = tack.empty((data.num_points,), tack.i32)
    if data.num_points:
        _flag_range(arrays.materialize(values), lower, upper, flags)
    return flags


@tack.kernel
def _every(flags, stride):
    for i in range(flags.shape[0]):
        flags[i] = 1 if i % stride == 0 else 0


def _vertices(points, like=None):
    """Vertex cells at ``points`` (an ``(n, 3)`` host array or an array of 3-vectors),
    with ``like``'s id type (or wider if needed), origin and spatial unit."""
    from tack.dataset.topology import UnstructuredTopology

    n = int(points.shape[0])
    idt = ids.for_output(like, n) if like is not None else ids.choose(None, n)
    return DataSet(UnstructuredTopology(*_vertex_cells(n, idt), num_points=n), points,
                   dtype=like.dtype if like is not None else tack.f32,
                   origin=like.origin if like is not None else None,
                   unit=like.unit if like is not None else None)
