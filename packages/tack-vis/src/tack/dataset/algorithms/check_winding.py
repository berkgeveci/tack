"""``check_winding``: the cells whose faces do not wind consistently, as a cell ``Set``."""


import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset.buckets import bucket_order, run_offsets
from tack.dataset.polyhedra import _cell_sizes
from tack.dataset.sets import Set
from tack.runtime.dispatch import on_inputs_device

__all__ = ['check_winding']


@tack.kernel
def _directed_edges(cell_offsets, cell_faces, cell_face_sides, face_offsets, face_edge,
                    face_edge_sign, starts, keys, directions, n_cells):
    # Each cell walks each face outward: forward on side 0, backward on side 1,
    # so each face edge runs along its stored direction or against it.
    for c in range(n_cells):
        at = starts[c]
        for e in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[e]
            flip = 1 - 2 * tack.i32(cell_face_sides[e])
            for j in range(face_offsets[f], face_offsets[f + 1]):
                keys[at] = [c, face_edge[j]]
                directions[at] = flip * face_edge_sign[j]
                at += 1


@tack.kernel
def _unbalanced(keys, directions, order, offsets, bad_cells, n_runs):
    # Each run, (cell, edge) or (cell, point), read through the sorted order.
    for r in range(n_runs):
        begin = offsets[r]
        end = offsets[r + 1]
        total = 0
        for i in range(begin, end):
            total += directions[order[i]]
        if end - begin != 2 or total != 0:
            bad_cells[keys[order[begin]][0]] = 1


@tack.kernel
def _loop_ends(cell_offsets, cell_faces, cell_face_sides, face_points, keys, ends, n_cells):
    # Each polygon's edges walked its way: every point must start one edge and
    # end another.
    for c in range(n_cells):
        for e in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[e]
            s = tack.i32(cell_face_sides[e])
            tail = face_points[2 * f + s]
            head = face_points[2 * f + 1 - s]
            keys[2 * e] = [c, tail]
            ends[2 * e] = 1
            keys[2 * e + 1] = [c, head]
            ends[2 * e + 1] = -1


def _check_loops(t):
    total = int(t.cell_faces.shape[0])
    bad = tack.zeros((t.num_cells,), tack.i32)
    if total:
        keys = tack.empty((2 * total,), tack.vec(2, t.id_dtype))
        ends = tack.empty((2 * total,), tack.i32)
        _loop_ends(t.cell_offsets, t.cell_faces, t.cell_face_sides, t.face_points, keys, ends,
                   t.num_cells)
        order, _ = bucket_order(keys, t.num_cells, t.id_dtype)
        offsets, runs = run_offsets(keys, order)
        _unbalanced(keys, ends, order, offsets, bad, runs)
    return Set._from_flags(t, "cells", bad)


@on_inputs_device
def check_winding(data):
    """The cells whose faces do not wind consistently, as a cell ``Set``.

    Walking every face outward (``side_point``), a closed, consistently wound cell
    walks each of its edges exactly twice, once each way. A cell that does not --
    a face with the wrong side, an open cell, an edge of three faces -- is listed.
    Combinatorial: geometry plays no part. For polygons: the cells whose edges,
    walked their way, do not close into a loop. Polylines have no winding, and
    shape-based cells take their faces from their shape's table, so are refused.
    """
    t = getattr(data, "topology", data)
    if getattr(t, "reference_cells", True):
        raise TypeError("check_winding checks cells that list their faces (polyhedra and "
                        "polygons); a shape's faces come from its table, wound one way")
    if t.dimension == 1:
        raise ValueError("polylines have no winding to check")
    if t.dimension == 2:
        return _check_loops(t)
    edges = t.edges()
    n = t.num_cells
    sizes = tack.empty((n,), tack.i32)
    starts = tack.empty((n,), t.id_dtype)
    if n:
        _cell_sizes(t.cell_offsets, t.cell_faces, t.face_offsets, sizes, n)
    total = exclusive_scan(sizes, starts, n) if n else 0
    bad = tack.zeros((n,), tack.i32)
    if total:
        keys = tack.empty((total,), tack.vec(2, t.id_dtype))
        directions = tack.empty((total,), tack.i32)
        _directed_edges(t.cell_offsets, t.cell_faces, t.cell_face_sides, t.face_offsets,
                        edges.face_edge, edges.face_edge_sign, starts, keys, directions, n)
        order, _ = bucket_order(keys, n, t.id_dtype)
        offsets, runs = run_offsets(keys, order)
        _unbalanced(keys, directions, order, offsets, bad, runs)
    return Set._from_flags(t, "cells", bad)
