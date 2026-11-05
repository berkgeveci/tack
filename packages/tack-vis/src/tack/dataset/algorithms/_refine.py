# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Splitting cells into pieces, behind ``tetrahedralize`` and ``triangulate``.

Tetrahedra conform, as VTK's ``vtkDataSetTriangleFilter``'s do: two cells
sharing a quadrilateral face split it along the same diagonal, so the pieces
meet face to face. A structured grid's hexahedra take 5 tetrahedra, the two
mirror splits alternating with the parity of (i, j, k); any other cell is the
cone, from its corner of smallest point id, over the faces that do not hold
that corner, each quadrilateral face split from its own corner of smallest
point id -- a rule both cells of a face apply alike, in whatever order they
come (a hexahedron in 6, a wedge in 3, a pyramid in 2). The cone fills a
convex cell. Triangles of 2D cells need no rule: cells share edges, which
are not split. A collapsed cell (repeated points) keeps only its pieces with
distinct points: the others measure nothing, and their repeated faces would
meet more than two cells."""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.arrays import CountingArray
from tack.dataset import ids, shapes
from tack.dataset.algorithms._common import _compact, _table
from tack.dataset.carry import Pieces, Same, carry
from tack.dataset.topology import UnstructuredTopology, uniform_cells

# A structured hexahedron's two splits into 5 (VTK's corner order): a tetrahedron of
# alternate corners in the middle, one at each other corner. Neighbours take the
# other split, so a shared face has one diagonal.
_EVEN_TETS = [[0, 2, 5, 7], [1, 0, 2, 5], [3, 0, 2, 7], [4, 0, 5, 7], [6, 2, 5, 7]]
_ODD_TETS = [[1, 3, 4, 6], [0, 1, 3, 4], [2, 1, 3, 6], [5, 1, 4, 6], [7, 3, 4, 6]]

# Tetrahedra of the cone from the smallest corner, per shape.
_CONE_TETS = {int(shapes.TETRA): 1, int(shapes.HEXAHEDRON): 6, int(shapes.VOXEL): 6,
              int(shapes.WEDGE): 3, int(shapes.PYRAMID): 2}


_PIECES = {
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


@tack.func
def _emit(cells, c, connectivity, at, p, a, b, d):
    # Tetrahedron (p, a, b, d) of corners of cell c into slot at, turned if needed
    # so its volume is positive.
    x = cells.corner_position(c, p)
    u = cells.corner_position(c, a) - x
    v = cells.corner_position(c, b) - x
    w = cells.corner_position(c, d) - x
    turn = u.dot(v.cross(w)) < 0.0
    connectivity[4 * at] = cells.point_id(c, p)
    connectivity[4 * at + 1] = cells.point_id(c, a)
    connectivity[4 * at + 2] = cells.point_id(c, d if turn else b)
    connectivity[4 * at + 3] = cells.point_id(c, b if turn else d)


@tack.kernel
def _parity_tets(cells, even, odd, shift, starts, connectivity, sources):
    # A structured grid's hexahedra, 5 tetrahedra each, the split by the parity of
    # (i + j + k) in the whole grid: the topology's offset's parity is ``shift``.
    for c in cells:
        e = cells.entity_id(c)
        at = starts[e]
        flip = (c[0] + c[1] + c[2] + shift) % 2
        for t in range(5):
            r = 4 * t
            if flip == 0:
                _emit(cells, c, connectivity, at + t, even[r], even[r + 1], even[r + 2],
                      even[r + 3])
            else:
                _emit(cells, c, connectivity, at + t, odd[r], odd[r + 1], odd[r + 2],
                      odd[r + 3])
            sources[at + t] = e


@tack.kernel
def _cone_tets(cells, rank, starts, connectivity, sources):
    # Each cell as the cone from its corner of smallest rank (its point's id, or
    # global id) over the faces without it, a quadrilateral face split from its
    # own corner of smallest rank.
    for c in cells:
        e = cells.entity_id(c)
        at = starts[e]
        apex = 0
        for j in range(1, cells.NUM_POINTS):
            if rank[cells.point_id(c, j)] < rank[cells.point_id(c, apex)]:
                apex = j
        t = 0
        if cells.NUM_FACES == 4 and cells.NUM_POINTS == 4:
            _emit(cells, c, connectivity, at, 0, 1, 2, 3)        # a tetrahedron is one
            sources[at] = e
            t = 1
        else:
            for f in range(cells.NUM_FACES):
                n = 4 if cells.face_corner(f, 3) >= 0 else 3
                holds = 0
                for j in range(n):
                    if cells.face_corner(f, j) == apex:
                        holds = 1
                if holds == 0:
                    if n == 3:
                        _emit(cells, c, connectivity, at + t, apex, cells.face_corner(f, 0),
                              cells.face_corner(f, 1), cells.face_corner(f, 2))
                        sources[at + t] = e
                        t += 1
                    else:
                        m = 0
                        for j in range(1, 4):
                            if (rank[cells.point_id(c, cells.face_corner(f, j))]
                                    < rank[cells.point_id(c, cells.face_corner(f, m))]):
                                m = j
                        q0 = cells.face_corner(f, m)
                        q1 = cells.face_corner(f, (m + 1) % 4)
                        q2 = cells.face_corner(f, (m + 2) % 4)
                        q3 = cells.face_corner(f, (m + 3) % 4)
                        _emit(cells, c, connectivity, at + t, apex, q0, q1, q2)
                        _emit(cells, c, connectivity, at + t + 1, apex, q0, q2, q3)
                        sources[at + t] = e
                        sources[at + t + 1] = e
                        t += 2


@tack.kernel
def _fan_counts(loop_offsets, counts):
    for c in range(counts.shape[0]):
        counts[c] = max(loop_offsets[c + 1] - loop_offsets[c] - 2, 0)


@tack.func
def _plane_uv(p, drop):
    """``p``'s coordinates in the plane that drops axis ``drop``, the other two in
    cyclic order (so the plane's normal is the dropped axis, sign kept)."""
    u = p[1]
    v = p[2]
    if drop == 1:
        u = p[2]
        v = p[0]
    elif drop == 2:
        u = p[0]
        v = p[1]
    return u, v


@tack.func
def _turn(x, loop_points, first, a, b, c, drop):
    """Twice the signed area of the triangle of loop corners ``a``, ``b``, ``c`` in the
    plane (``_plane_uv``)."""
    au, av = _plane_uv(x[loop_points[first + a]], drop)
    bu, bv = _plane_uv(x[loop_points[first + b]], drop)
    cu, cv = _plane_uv(x[loop_points[first + c]], drop)
    return (bu - au) * (cv - av) - (bv - av) * (cu - au)


@tack.kernel
def _ears(loop_offsets, loop_points, x, starts, after, before, connectivity, sources):
    # Each polygon by ear cutting in its plane -- the plane of its Newell normal's
    # largest component -- so a concave one is covered once (a fan from its first
    # corner covers part of it twice and part not at all). An ear is a corner
    # turning as the polygon does whose triangle holds no other corner; cutting it
    # leaves a polygon of one corner fewer. ``after`` and ``before`` link the
    # corners left, within the polygon's own range. Where no ear is left (a
    # degenerate remainder) the rest is a fan, so every polygon of n corners gives
    # n - 2 triangles, each wound as the polygon.
    for c in range(len(loop_offsets) - 1):
        first = loop_offsets[c]
        n = loop_offsets[c + 1] - first
        at = starts[c]
        if n >= 3:
            nx = 0.0
            ny = 0.0
            nz = 0.0
            for k in range(n):
                p = x[loop_points[first + k]]
                q = x[loop_points[first + (k + 1) % n]]
                nx += (p[1] - q[1]) * (p[2] + q[2])
                ny += (p[2] - q[2]) * (p[0] + q[0])
                nz += (p[0] - q[0]) * (p[1] + q[1])
            drop = 2
            sense = nz
            if abs(nx) >= abs(ny) and abs(nx) >= abs(nz):
                drop = 0
                sense = nx
            elif abs(ny) >= abs(nz):
                drop = 1
                sense = ny
            sign = 1.0
            if sense < 0.0:
                sign = -1.0
            for k in range(n):
                after[first + k] = (k + 1) % n
                before[first + k] = (k + n - 1) % n
            left = n
            i = 0
            misses = 0
            t = 0
            while left > 3 and misses < left:
                a = before[first + i]
                b = after[first + i]
                ear = sign * _turn(x, loop_points, first, a, i, b, drop) > 0.0
                if ear:
                    j = after[first + b]
                    while ear and j != a:
                        inside = (sign * _turn(x, loop_points, first, a, i, j, drop) > 0.0
                                  and sign * _turn(x, loop_points, first, i, b, j, drop) > 0.0
                                  and sign * _turn(x, loop_points, first, b, a, j, drop) > 0.0)
                        if inside:
                            ear = False
                        j = after[first + j]
                if ear:
                    connectivity[3 * (at + t)] = loop_points[first + a]
                    connectivity[3 * (at + t) + 1] = loop_points[first + i]
                    connectivity[3 * (at + t) + 2] = loop_points[first + b]
                    sources[at + t] = c
                    t += 1
                    after[first + a] = b
                    before[first + b] = a
                    left -= 1
                    i = a
                    misses = 0
                else:
                    i = b
                    misses += 1
            # The last triangle, or a fan of a degenerate remainder from corner i.
            j = after[first + i]
            while t < n - 2:
                connectivity[3 * (at + t)] = loop_points[first + i]
                connectivity[3 * (at + t) + 1] = loop_points[first + j]
                connectivity[3 * (at + t) + 2] = loop_points[first + after[first + j]]
                sources[at + t] = c
                t += 1
                j = after[first + j]


@tack.kernel
def _distinct(connectivity, width, flags):
    # 1 where a piece's points are distinct: a collapsed cell's (a hexahedron with
    # its top face pinched to a point) repeat one and measure nothing.
    for t in range(flags.shape[0]):
        whole = 1
        for a in range(width):
            for b in range(a + 1, width):
                if connectivity[t * width + a] == connectivity[t * width + b]:
                    whole = 0
        flags[t] = whole


@tack.kernel
def _kept_pieces(kept, width, connectivity, sources, out_connectivity, out_sources):
    for i in range(kept.shape[0]):
        t = kept[i]
        for v in range(width):
            out_connectivity[i * width + v] = connectivity[t * width + v]
        out_sources[i] = sources[t]


def _without_repeats(connectivity, sources, count, width):
    """The pieces whose points are distinct. VTK keeps the others, of no volume,
    but their faces repeat and would meet more than two cells."""
    if not count:
        return connectivity, sources, count
    flags = tack.empty((count,), tack.i32)
    _distinct(connectivity, width, flags)
    _, kept, whole = _compact(flags, connectivity.dtype)
    if whole == count:
        return connectivity, sources, count
    out_connectivity = tack.empty((whole * width,), connectivity.dtype)
    out_sources = tack.empty((whole,), sources.dtype)
    if whole:
        _kept_pieces(kept, width, connectivity, sources, out_connectivity, out_sources)
    return out_connectivity, out_sources, whole


def _pieces_topology(data, connectivity, sources, count, shape, width, fields):
    # All tetrahedra, or all triangles: the types and offsets are implicit.
    connectivity, sources, count = _without_repeats(connectivity, sources, count, width)
    types, offsets = uniform_cells(int(shape), count, connectivity.dtype, width)
    out = UnstructuredTopology(types, offsets, connectivity, num_points=data.num_points)
    return carry(data, out, points=Same(), cells=Pieces(sources), fields=fields)


def _tetrahedra(data, fields):
    """Every 3D shape cell as conforming tetrahedra (the module's docstring)."""
    from tack.dataset.topology import StructuredTopology

    n = data.num_cells
    structured = isinstance(data.topology, StructuredTopology)
    if structured and data.topology.dimension != 3:
        groups, per = [], {}
    elif structured:
        groups = [g for g in data.topology.groups() if g.count]
        per = {int(shapes.HEXAHEDRON): 5}
    else:
        groups = [g for g in data.topology.groups()
                  if g.count and int(g.shape.ID) in _CONE_TETS]
        per = _CONE_TETS
    counts = tack.zeros((n,), tack.i32)
    for group in groups:
        _piece_counts(group.view(), per[int(group.shape.ID)], counts)
    odt = ids.for_output(data, 24 * n)
    starts = tack.empty((n,), odt)
    count = exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.empty((4 * count,), odt)
    sources = tack.empty((count,), data.id_dtype)
    # Corners rank by global id where there are global ids, so pieces of a
    # partitioned dataset split the faces they share alike; by point id otherwise.
    ids_ = data.global_ids
    rank = ids_.points if ids_ is not None and ids_.points is not None else \
        CountingArray(data.num_points, dtype=tack.i64)
    shift = sum(data.topology.offset) % 2 if structured else 0
    for group in groups:
        view = data.domain_view("cells", group)        # with the corners' positions
        if structured:
            _parity_tets(view, _table("even tets", _EVEN_TETS), _table("odd tets", _ODD_TETS),
                         shift, starts, connectivity, sources)
        else:
            _cone_tets(view, rank, starts, connectivity, sources)
    return _pieces_topology(data, connectivity, sources, count, shapes.TETRA, 4, fields)


def _split(data, kind, fields):
    if kind == "tetrahedra":
        return _tetrahedra(data, fields)
    tables = _PIECES[kind]
    width, shape = (4, shapes.TETRA) if kind == "tetrahedra" else (3, shapes.TRIANGLE)
    n = data.num_cells
    groups = [g for g in data.topology.groups() if g.count and int(g.shape.ID) in tables]
    counts = tack.zeros((n,), tack.i32)
    for group in groups:
        _piece_counts(group.view(), len(tables[int(group.shape.ID)]), counts)
    most = max((len(tables[int(g.shape.ID)]) for g in groups), default=0)
    odt = ids.for_output(data, most * width * n)
    starts = tack.empty((n,), odt)
    count = exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.empty((count * width,), odt)
    sources = tack.empty((count,), data.id_dtype)
    for group in groups:
        table = np.asarray(tables[int(group.shape.ID)], np.int32)
        _split_cells(group.view(), _table(("pieces", kind, int(group.shape.ID)), table),
                     len(table), width, starts, connectivity, sources)
    return _pieces_topology(data, connectivity, sources, count, shape, width, fields)
