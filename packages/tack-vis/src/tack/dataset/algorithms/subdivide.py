"""``subdivide``: a dataset of curved or high-order cells as linear sub-cells -- each
cell's lattice of ``level`` steps along each axis -- with its fields evaluated at
the lattice points. ``contour`` and ``slice`` cut high-order data this way, on
only the cells that can hold the cut (``_refined``).

A lattice point on a cell's edge, face or corner is shared with the cells around
it: its id is a node of the equispaced ``H1`` space of order ``level``, numbered
by entity, so every cell names it alike. Each is evaluated once (a stable sort of
the ids), so a continuous field and the geometry have one value there, to the
bit, and cuts through neighbouring cells meet without cracks. No atomics, no
tolerance.

The same holds across pieces. A point on an edge or face is computed from that
entity's own values (a continuous field's trace there), in a frame its corners'
positions fix, so every piece holding any cell around it gets the same bits; a
corner is the stored corner; the rest (an interior point, an ``L2`` field) by the
cell every piece chooses alike, the lowest global id or the first centroid by
position. With ghost marks, each lattice point is marked as ``carry`` marks a
point it makes (``_lattice_marks``) and each sub-cell takes its parent's, so each
piece's cut is its share of the whole's.
"""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import argsort
from tack.arrays import ConstantArray, size_of
from tack.dataset import bases, ids, shapes, views
from tack.dataset.dataset import Field, for_each
from tack.dataset.ghosts import _before
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.dataset.topology import UnstructuredTopology, uniform_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['subdivide']


# ── Which cells ─────────────────────────────────────────────────────

@tack.kernel
def _all_cells(cells, flags):
    for c in cells:
        flags[cells.entity_id(c)] = 1


@tack.kernel
def _straddling(cells, u, levels, nlevels, flags):
    # A cell whose field's Bernstein bound holds a level: it may be cut.
    for c in cells:
        lo, hi = u.value_range(c)
        hit = 0
        for v in range(nlevels):
            if lo <= levels[v] and levels[v] <= hi:
                hit = 1
        flags[cells.entity_id(c)] = hit


@tack.kernel
def _by_range(cells, u, lower, upper, whole, cut):
    # A cell whose field's Bernstein bound lies in [lower, upper] is kept whole, one
    # whose bound misses it dropped, and the rest may be cut.
    for c in cells:
        lo, hi = u.value_range(c)
        e = cells.entity_id(c)
        w = 0
        k = 0
        if lower <= lo and hi <= upper:
            w = 1
        elif not (hi < lower or upper < lo):
            k = 1
        whole[e] = w
        cut[e] = k


@tack.kernel
def _flatness(cells, u, flags, out):
    # What linearizing each flagged cell at its corners loses, at most.
    for c in cells:
        e = cells.entity_id(c)
        out[e] = u.flatness(c) if flags[e] == 1 else u.flatness(c) - u.flatness(c)


#: The most lattice steps a tolerance chooses: 32^3 sub-cells a hexahedron.
MAX_LEVEL = 32


@tack.kernel
def _place_chosen(flags, slots, slot_of, chosen):
    for e in range(flags.shape[0]):
        if flags[e] == 1:
            slot_of[e] = slots[e]
            chosen[slots[e]] = e
        else:
            slot_of[e] = -1


# ── The lattice ─────────────────────────────────────────────────────

@tack.kernel
def _lattice_keys(cells, t, slot_of, per_cell, node_of, keys):
    # Each refined cell's lattice points' ids: the nodes of the equispaced H1 space
    # of the lattice's order (node_of: a lattice node's number in that space's
    # cell order, which at order 2 is the quadratic shapes').
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                keys[slot * per_cell + n] = t.dof_index(c, node_of[n])


@tack.kernel
def _corner_keys(cells, slot_of, corner_of, keys):
    # Level 1: the lattice is the corners.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(cells.NUM_POINTS):
                keys[slot * cells.NUM_POINTS + n] = cells.point_id(c, corner_of[n])


@tack.kernel
def _run_starts(keys, order, starts):
    # starts[i]: whether sorted position i begins a run of one lattice point's ids.
    for i in range(order.shape[0]):
        starts[i] = 1 if i == 0 or keys[order[i - 1]] != keys[order[i]] else 0


@tack.kernel
def _ranks(order, starts, numbers, rank):
    # The runs before position i, counting its own if it starts one, less one.
    for i in range(order.shape[0]):
        rank[order[i]] = numbers[i] + starts[i] - 1


@tack.func
def _ahead(a, b):
    # Whether cell rank a comes before b: by x, then y, then z of a cell's
    # centroid (or a global id in x alone), the same in every piece.
    return a[0] < b[0] or (a[0] == b[0] and (a[1] < b[1] or (a[1] == b[1] and a[2] < b[2])))


@tack.kernel
def _evaluators(order, starts, per_cell, cell_rank, first):
    # Each lattice point's evaluating cell: among the refined cells holding it,
    # the first by cell_rank (a global id, or the centroid by position), which
    # every piece holding those cells chooses alike -- so a shared point, and a
    # cut through it, are the same bits in every piece.
    total = order.shape[0]
    for i in range(total):
        if starts[i] == 1:
            best = order[i]
            j = i + 1
            while j < total and starts[j] == 0:
                k = order[j]
                if _ahead(cell_rank[k // per_cell], cell_rank[best // per_cell]):
                    best = k
                j += 1
            first[best] = 1


@tack.kernel
def _centroid_ranks(cells, slot_of, out):
    # A refined cell's rank: its centroid, the sum of its corners in its own
    # order, which every copy of the cell holds alike.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            x = cells.corner_position(c, 0)
            for j in range(1, cells.NUM_POINTS):
                x += cells.corner_position(c, j)
            out[slot] = x


@tack.kernel
def _id_ranks(gids, chosen, out):
    # A global id as two parts below 2^20 each, exact in f32 too (Metal has no f64).
    for slot in range(out.shape[0]):
        gid = gids[chosen[slot]]
        high = gid // 1048576
        out[slot] = [high, gid - high * 1048576, 0]


@tack.kernel
def _lattice_marks(cells, slot_of, per_cell, node_kind, node_entity, first, rank,
                   point_marks, cell_marks, out):
    # A lattice point's ghost mark, as carry gives a point it makes: a corner its
    # point's; inside an input edge or quad face, its first corner by position's
    # (DUPLICATEPOINT alone); inside a cell, the cell's (DUPLICATECELL as
    # DUPLICATEPOINT, HIDDENCELL as HIDDENPOINT).
    for c in cells:
        e = cells.entity_id(c)
        slot = slot_of[e]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                if first[k] == 1:
                    kind = node_kind[n]
                    m = node_entity[n]
                    if kind == 0:
                        out[rank[k]] = point_marks[cells.point_id(c, m)]
                    elif kind == 3:
                        cm = cell_marks[e]
                        out[rank[k]] = (cm & 1) | (2 if (cm & 32) != 0 else 0)
                    else:
                        s = cells.edge_corner(m, 0)
                        count = 2
                        if kind == 2:
                            s = cells.face_corner(m, 0)
                            count = cells.face_num_points(m)
                        for j in range(1, count):
                            t = cells.edge_corner(m, j) if kind == 1 else cells.face_corner(m, j)
                            if _before(cells.corner_position(c, t), cells.corner_position(c, s)):
                                s = t
                        out[rank[k]] = point_marks[cells.point_id(c, s)] & 1


@tack.kernel
def _seam_marks(cells, slot_of, per_cell, node_kind, node_entity, first, rank, elsewhere,
                out):
    # A lattice corner on a point another part owns: 1.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                if first[k] == 1 and node_kind[n] == 0:
                    if elsewhere[cells.point_id(c, node_entity[n])] == 1:
                        out[rank[k]] = 1


@tack.kernel
def _corners_of(cells, flags, out):
    # Each flagged cell's corners: 1 (cells sharing a point write the same 1).
    for c in cells:
        if flags[cells.entity_id(c)] == 1:
            for j in range(cells.NUM_POINTS):
                out[cells.point_id(c, j)] = 1


@tack.func
def _lattice_point(n, dimension, steps, ijk):
    """Lattice node ``n``'s parametric coordinates: ``steps[i]`` (``i / level`` in the
    dataset's float type) for each of its integer coordinates (``ijk``,
    ``bases.lattice_ijk``: a tensor lattice's or a simplex's)."""
    pc = tack.vec3(steps[ijk[n * dimension]], 0.0, 0.0)
    if dimension >= 2:
        pc[1] = steps[ijk[n * dimension + 1]]
    if dimension == 3:
        pc[2] = steps[ijk[n * dimension + 2]]
    return pc


@tack.kernel
def _positions(cells, slot_of, per_cell, ijk, steps, corner_at, first, rank, out):
    # A lattice corner is the cell's corner, its stored position to the bit.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                if first[k] == 1:
                    if corner_at[n] >= 0:
                        out[rank[k]] = cells.corner_position(c, corner_at[n])
                    else:
                        out[rank[k]] = cells.position(c, _lattice_point(n, cells.DIMENSION,
                                                                       steps, ijk))


@tack.kernel
def _values(cells, u, slot_of, per_cell, ijk, steps, corner_at, first, rank, out):
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                if first[k] == 1:
                    if corner_at[n] >= 0:
                        out[rank[k]] = u.corner_value(c, corner_at[n])
                    else:
                        out[rank[k]] = u.value(c, _lattice_point(n, cells.DIMENSION, steps, ijk))


# ── Shared lattice points from their edge's or face's own data ───────
#
# A continuous field's trace on an edge depends on that edge's values alone, and on
# a face on the face's: so a lattice point on an edge or face is computed from
# them, in a frame its corners' positions fix (an edge from its first corner by
# position; a face from its first corner, toward the first of that corner's two
# neighbours), the same in every cell and every piece holding it, whatever cells
# a piece holds -- as a linear contour's crossings are since #188.


@tack.func
def _lagrange(t, j, nodes, weights, q):
    # The 1D Lagrange polynomial of node j (of q, at nodes) at t, barycentric.
    v = weights[j]
    for m in range(q):
        if m != j:
            v = v * (t - nodes[m])
    return v


@tack.func
def _axis(t, j, nodes, weights, q, bern):
    # The 1D basis polynomial j (of q) at t: Bernstein's, weights[j] t^j (1 - t)^(q-1-j),
    # with bern, else the Lagrange polynomial through nodes.
    if bern == 1:
        v = weights[j]
        s = 1.0 - t
        for _ in range(j):
            v = v * t
        for _ in range(q - 1 - j):
            v = v * s
        return v
    return _lagrange(t, j, nodes, weights, q)


@tack.func
def _unit(k):
    # Face corner k (in the face's own order) on the unit square.
    return (1 if k == 1 or k == 2 else 0), (1 if k == 2 or k == 3 else 0)


@tack.func
def _face_frame(cells, c, f):
    # (A, B, D): the face's first corner by position, the first of its two
    # neighbours (the frame's first axis), the other (its second).
    a = 0
    for k in range(1, 4):
        if _before(cells.corner_position(c, cells.face_corner(f, k)),
                   cells.corner_position(c, cells.face_corner(f, a))):
            a = k
    b = (a + 1) % 4
    d = (a + 3) % 4
    if _before(cells.corner_position(c, cells.face_corner(f, d)),
               cells.corner_position(c, cells.face_corner(f, b))):
        b, d = d, b
    return a, b, d


@tack.func
def _edge_step(cells, c, e, a, level):
    # The lattice point's step from its edge's first corner by position, and
    # whether that corner is the edge's corner 1 (its order reversed).
    reversed_ = 0
    if _before(cells.corner_position(c, cells.edge_corner(e, 1)),
               cells.corner_position(c, cells.edge_corner(e, 0))):
        reversed_ = 1
    return (level - a if reversed_ == 1 else a), reversed_


@tack.func
def _face_steps(cells, c, f, a, b, level):
    # The lattice point (a, b), from the face's corner 0, in the face's frame.
    ka, kb, kd = _face_frame(cells, c, f)
    ax, ay = _unit(ka)
    bx, by = _unit(kb)
    dx, dy = _unit(kd)
    ua = a - ax * level
    ub = b - ay * level
    return ua * (bx - ax) + ub * (by - ay), ua * (dx - ax) + ub * (dy - ay), ka, kb, kd


@tack.kernel
def _trace_values(cells, u, slot_of, per_cell, level, steps, node_kind, node_entity, node_a,
                  node_b, first, rank, edge_nodes, face_nodes, view_of, nodes, weights, q, bern,
                  out):
    # An H1 field at the lattice points on edges and faces, from their own values.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                kind = node_kind[n]
                if first[k] == 1 and (kind == 1 or kind == 2):
                    m = node_entity[n]
                    if kind == 1:
                        s, flip = _edge_step(cells, c, m, node_a[n], level)
                        t = steps[s]
                        j0 = q - 1 if flip == 1 else 0
                        total = u.dof(c, view_of[edge_nodes[m * q + j0]]) * \
                            _axis(t, 0, nodes, weights, q, bern)
                        for jp in range(1, q):
                            j = q - 1 - jp if flip == 1 else jp
                            total += u.dof(c, view_of[edge_nodes[m * q + j]]) * \
                                _axis(t, jp, nodes, weights, q, bern)
                        out[rank[k]] = total
                    else:
                        sa, sb, ka, kb, kd = _face_steps(cells, c, m, node_a[n], node_b[n],
                                                         level)
                        ta = steps[sa]
                        tb = steps[sb]
                        ax, ay = _unit(ka)
                        bx, by = _unit(kb)
                        dx, dy = _unit(kd)
                        p = q - 1
                        total = u.dof(c, view_of[face_nodes[m * q * q + ax * p + q * (ay * p)]]) \
                            * (_axis(ta, 0, nodes, weights, q, bern)
                               * _axis(tb, 0, nodes, weights, q, bern))
                        for i in range(1, q * q):
                            jp = i % q
                            kp = i // q
                            fa = ax * p + jp * (bx - ax) + kp * (dx - ax)
                            fb = ay * p + jp * (by - ay) + kp * (dy - ay)
                            total += u.dof(c, view_of[face_nodes[m * q * q + fa + q * fb]]) \
                                * (_axis(ta, jp, nodes, weights, q, bern)
                                   * _axis(tb, kp, nodes, weights, q, bern))
                        out[rank[k]] = total


@tack.kernel
def _trace_straight(cells, slot_of, per_cell, level, steps, node_kind, node_entity, node_a,
                    node_b, first, rank, out):
    # A straight geometry at the lattice points on edges and faces: along the edge
    # from its first corner, bilinear over the face in its frame.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                kind = node_kind[n]
                if first[k] == 1 and (kind == 1 or kind == 2):
                    m = node_entity[n]
                    if kind == 1:
                        s, flip = _edge_step(cells, c, m, node_a[n], level)
                        xa = cells.corner_position(c, cells.edge_corner(m, 1 if flip == 1 else 0))
                        xb = cells.corner_position(c, cells.edge_corner(m, 0 if flip == 1 else 1))
                        out[rank[k]] = xa + steps[s] * (xb - xa)
                    else:
                        sa, sb, ka, kb, kd = _face_steps(cells, c, m, node_a[n], node_b[n],
                                                         level)
                        ta = steps[sa]
                        tb = steps[sb]
                        xa = cells.corner_position(c, cells.face_corner(m, ka))
                        xb = cells.corner_position(c, cells.face_corner(m, kb))
                        xc = cells.corner_position(c, cells.face_corner(m, (ka + 2) % 4))
                        xd = cells.corner_position(c, cells.face_corner(m, kd))
                        out[rank[k]] = (xa + ta * (1.0 - tb) * (xb - xa) + ta * tb * (xc - xa)
                                        + (1.0 - ta) * tb * (xd - xa))


@tack.kernel
def _trace_curved(cells, slot_of, per_cell, level, steps, node_kind, node_entity, node_a,
                  node_b, first, rank, edge_nodes, face_nodes, q, bern, out):
    # A curved (tensor) geometry the same way, from its own nodes on the entity.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                kind = node_kind[n]
                if first[k] == 1 and (kind == 1 or kind == 2):
                    m = node_entity[n]
                    if kind == 1:
                        s, flip = _edge_step(cells, c, m, node_a[n], level)
                        t = steps[s]
                        j0 = q - 1 if flip == 1 else 0
                        x = cells.get(cells.dof_index(c, edge_nodes[m * q + j0])) * \
                            _axis(t, 0, cells.basis_nodes, cells.basis_weights, q, bern)
                        for jp in range(1, q):
                            j = q - 1 - jp if flip == 1 else jp
                            x += cells.get(cells.dof_index(c, edge_nodes[m * q + j])) * \
                                _axis(t, jp, cells.basis_nodes, cells.basis_weights, q, bern)
                        out[rank[k]] = x
                    else:
                        sa, sb, ka, kb, kd = _face_steps(cells, c, m, node_a[n], node_b[n],
                                                         level)
                        ta = steps[sa]
                        tb = steps[sb]
                        ax, ay = _unit(ka)
                        bx, by = _unit(kb)
                        dx, dy = _unit(kd)
                        p = q - 1
                        x = cells.get(cells.dof_index(c, face_nodes[m * q * q + ax * p
                                                                    + q * (ay * p)])) \
                            * (_axis(ta, 0, cells.basis_nodes, cells.basis_weights, q, bern)
                               * _axis(tb, 0, cells.basis_nodes, cells.basis_weights, q, bern))
                        for i in range(1, q * q):
                            jp = i % q
                            kp = i // q
                            fa = ax * p + jp * (bx - ax) + kp * (dx - ax)
                            fb = ay * p + jp * (by - ay) + kp * (dy - ay)
                            x += cells.get(cells.dof_index(c, face_nodes[m * q * q + fa + q * fb])) \
                                * (_axis(ta, jp, cells.basis_nodes, cells.basis_weights, q, bern)
                                   * _axis(tb, kp, cells.basis_nodes, cells.basis_weights, q, bern))
                        out[rank[k]] = x


# ── Simplices: Freudenthal's sub-cells and triangle faces' traces ─────


@tack.func
def _silvester(t, a, p):
    # F[a](t) = prod_{s < a} (p t - s) / (s + 1), in t's type.
    v = t - t + 1.0
    scale = t - t + p
    for s in range(a):
        v = v * (scale * t - s) / (s + 1)
    return v


@tack.func
def _triangle_term(la, lb, lc, ba, bb, bc, p, bern):
    # A triangle face's basis function (ba, bb, bc) at barycentric (la, lb, lc):
    # Bernstein's p! / (ba! bb! bc!) la^ba lb^bb lc^bc with bern, else Silvester's.
    if bern == 1:
        v = la - la + 1.0
        for s in range(2, p + 1):
            v = v * s
        for k in range(1, ba + 1):
            v = v * la / k
        for k in range(1, bb + 1):
            v = v * lb / k
        for k in range(1, bc + 1):
            v = v * lc / k
        return v
    return _silvester(la, ba, p) * _silvester(lb, bb, p) * _silvester(lc, bc, p)


@tack.func
def _triangle_order(cells, c, f):
    # A triangle face's corners (its own 0, 1, 2) ordered by position: (A, B, C).
    a = 0
    b = 1
    cc = 2
    if _before(cells.corner_position(c, cells.face_corner(f, b)),
               cells.corner_position(c, cells.face_corner(f, a))):
        a, b = b, a
    if _before(cells.corner_position(c, cells.face_corner(f, cc)),
               cells.corner_position(c, cells.face_corner(f, b))):
        b, cc = cc, b
        if _before(cells.corner_position(c, cells.face_corner(f, b)),
                   cells.corner_position(c, cells.face_corner(f, a))):
            a, b = b, a
    return a, b, cc


@tack.func
def _face_weights(ka, kb, kc, a, b, level):
    # The lattice point (a, b) from the face's corner 0 (toward 1 and 2) as steps on
    # the ordered corners A, B, C.
    w = level - a - b
    wa = w if ka == 0 else (a if ka == 1 else b)
    wb = w if kb == 0 else (a if kb == 1 else b)
    wc = w if kc == 0 else (a if kc == 1 else b)
    return wa, wb, wc


@tack.kernel
def _trace_simplex_values(cells, u, slot_of, per_cell, level, steps, node_kind, node_entity,
                          node_a, node_b, first, rank, edge_nodes, face_nodes, stride, view_of,
                          nodes, weights, q, bern, out):
    # An H1 field at the lattice points on a simplex's edges and triangle faces, from
    # their own values: an edge as _trace_values does it, a face in its corners'
    # order by position, the sum in that order.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                kind = node_kind[n]
                if first[k] == 1 and (kind == 1 or kind == 2):
                    m = node_entity[n]
                    if kind == 1:
                        s, flip = _edge_step(cells, c, m, node_a[n], level)
                        t = steps[s]
                        j0 = q - 1 if flip == 1 else 0
                        total = u.dof(c, view_of[edge_nodes[m * q + j0]]) * \
                            _axis(t, 0, nodes, weights, q, bern)
                        for jp in range(1, q):
                            j = q - 1 - jp if flip == 1 else jp
                            total += u.dof(c, view_of[edge_nodes[m * q + j]]) * \
                                _axis(t, jp, nodes, weights, q, bern)
                        out[rank[k]] = total
                    elif cells.face_num_points(m) == 4:
                        # A wedge's quad face, as _trace_values evaluates one.
                        sa, sb, ka, kb, kd = _face_steps(cells, c, m, node_a[n], node_b[n],
                                                         level)
                        ta = steps[sa]
                        tb = steps[sb]
                        ax, ay = _unit(ka)
                        bx, by = _unit(kb)
                        dx, dy = _unit(kd)
                        p = q - 1
                        total = u.dof(c, view_of[face_nodes[m * stride + ax * p + q * (ay * p)]]) \
                            * (_axis(ta, 0, nodes, weights, q, bern)
                               * _axis(tb, 0, nodes, weights, q, bern))
                        for i in range(1, q * q):
                            jp = i % q
                            kp = i // q
                            fa = ax * p + jp * (bx - ax) + kp * (dx - ax)
                            fb = ay * p + jp * (by - ay) + kp * (dy - ay)
                            total += u.dof(c, view_of[face_nodes[m * stride + fa + q * fb]]) \
                                * (_axis(ta, jp, nodes, weights, q, bern)
                                   * _axis(tb, kp, nodes, weights, q, bern))
                        out[rank[k]] = total
                    else:
                        p = q - 1
                        per_face = stride
                        ka, kb, kc = _triangle_order(cells, c, m)
                        wa, wb, wc = _face_weights(ka, kb, kc, node_a[n], node_b[n], level)
                        la = steps[wa]
                        lb = steps[wb]
                        lc = steps[wc]
                        total = la - la
                        started = 0
                        # The field's face nodes in the ordered frame: (ba, bb, bc) steps on A,
                        # B, C; the node's own face index from its steps on corners 1 and 2.
                        for bc in range(q):
                            for bb in range(q - bc):
                                ba = p - bb - bc
                                s1 = ba if ka == 1 else (bb if kb == 1 else bc)
                                s2 = ba if ka == 2 else (bb if kb == 2 else bc)
                                t = s2 * (p + 1) - (s2 * (s2 - 1)) // 2 + s1
                                term = u.dof(c, view_of[face_nodes[m * per_face + t]]) * (
                                    _triangle_term(la, lb, lc, ba, bb, bc, p, bern))
                                if started == 0:
                                    total = term
                                    started = 1
                                else:
                                    total += term
                        out[rank[k]] = total


@tack.kernel
def _trace_simplex_curved(cells, slot_of, per_cell, level, steps, node_kind, node_entity,
                          node_a, node_b, first, rank, edge_nodes, face_nodes, stride, view_of,
                          nodes, weights, q, bern, out):
    # A curved simplex geometry (order 2's quadratic, or 3 and up) at the lattice
    # points on edges and triangle faces, from its own nodes there, as
    # _trace_simplex_values evaluates a field.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                kind = node_kind[n]
                if first[k] == 1 and (kind == 1 or kind == 2):
                    m = node_entity[n]
                    if kind == 1:
                        s, flip = _edge_step(cells, c, m, node_a[n], level)
                        t = steps[s]
                        j0 = q - 1 if flip == 1 else 0
                        x = cells.get(cells.dof_index(c, view_of[edge_nodes[m * q + j0]])) * \
                            _axis(t, 0, nodes, weights, q, bern)
                        for jp in range(1, q):
                            j = q - 1 - jp if flip == 1 else jp
                            x += cells.get(cells.dof_index(c, view_of[edge_nodes[m * q + j]])) * \
                                _axis(t, jp, nodes, weights, q, bern)
                        out[rank[k]] = x
                    elif cells.face_num_points(m) == 4:
                        sa, sb, ka, kb, kd = _face_steps(cells, c, m, node_a[n], node_b[n],
                                                         level)
                        ta = steps[sa]
                        tb = steps[sb]
                        ax, ay = _unit(ka)
                        bx, by = _unit(kb)
                        dx, dy = _unit(kd)
                        p = q - 1
                        x = cells.get(cells.dof_index(c, view_of[face_nodes[
                            m * stride + ax * p + q * (ay * p)]])) * (
                            _axis(ta, 0, nodes, weights, q, bern) * _axis(tb, 0, nodes, weights, q, bern))
                        for i in range(1, q * q):
                            jp = i % q
                            kp = i // q
                            fa = ax * p + jp * (bx - ax) + kp * (dx - ax)
                            fb = ay * p + jp * (by - ay) + kp * (dy - ay)
                            x += cells.get(cells.dof_index(c, view_of[face_nodes[
                                m * stride + fa + q * fb]])) * (
                                _axis(ta, jp, nodes, weights, q, bern)
                                * _axis(tb, kp, nodes, weights, q, bern))
                        out[rank[k]] = x
                    else:
                        p = q - 1
                        per_face = stride
                        ka, kb, kc = _triangle_order(cells, c, m)
                        wa, wb, wc = _face_weights(ka, kb, kc, node_a[n], node_b[n], level)
                        la = steps[wa]
                        lb = steps[wb]
                        lc = steps[wc]
                        x = cells.corner_position(c, 0)
                        started = 0
                        for bc in range(q):
                            for bb in range(q - bc):
                                ba = p - bb - bc
                                s1 = ba if ka == 1 else (bb if kb == 1 else bc)
                                s2 = ba if ka == 2 else (bb if kb == 2 else bc)
                                t = s2 * (p + 1) - (s2 * (s2 - 1)) // 2 + s1
                                term = cells.get(cells.dof_index(
                                    c, view_of[face_nodes[m * per_face + t]])) * (
                                    _triangle_term(la, lb, lc, ba, bb, bc, p, bern))
                                if started == 0:
                                    x = term
                                    started = 1
                                else:
                                    x += term
                        out[rank[k]] = x


@tack.kernel
def _trace_simplex_straight(cells, slot_of, per_cell, level, steps, node_kind, node_entity,
                            node_a, node_b, first, rank, out):
    # A straight geometry at the lattice points on a simplex's edges and triangle faces:
    # along the edge from its first corner, over the face in its corners' order.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                kind = node_kind[n]
                if first[k] == 1 and (kind == 1 or kind == 2):
                    m = node_entity[n]
                    if kind == 1:
                        s, flip = _edge_step(cells, c, m, node_a[n], level)
                        xa = cells.corner_position(c, cells.edge_corner(m, 1 if flip == 1 else 0))
                        xb = cells.corner_position(c, cells.edge_corner(m, 0 if flip == 1 else 1))
                        out[rank[k]] = xa + steps[s] * (xb - xa)
                    elif cells.face_num_points(m) == 4:
                        sa, sb, ka, kb, kd = _face_steps(cells, c, m, node_a[n], node_b[n],
                                                         level)
                        ta = steps[sa]
                        tb = steps[sb]
                        xa = cells.corner_position(c, cells.face_corner(m, ka))
                        xb = cells.corner_position(c, cells.face_corner(m, kb))
                        xc = cells.corner_position(c, cells.face_corner(m, (ka + 2) % 4))
                        xd = cells.corner_position(c, cells.face_corner(m, kd))
                        out[rank[k]] = (xa + ta * (1.0 - tb) * (xb - xa) + ta * tb * (xc - xa)
                                        + (1.0 - ta) * tb * (xd - xa))
                    else:
                        ka, kb, kc = _triangle_order(cells, c, m)
                        wa, wb, wc = _face_weights(ka, kb, kc, node_a[n], node_b[n], level)
                        xa = cells.corner_position(c, cells.face_corner(m, ka))
                        out[rank[k]] = (xa + steps[wb] * (cells.corner_position(
                            c, cells.face_corner(m, kb)) - xa) + steps[wc] * (
                            cells.corner_position(c, cells.face_corner(m, kc)) - xa))


@tack.kernel
def _sub_simplices(cells, slot_of, per_cell, subs, sub, table, rank, conn, parent):
    # Each refined simplex's level^d sub-simplices (bases.simplex_subdivision).
    for c in cells:
        e = cells.entity_id(c)
        slot = slot_of[e]
        if slot >= 0:
            for k in range(subs):
                parent[slot * subs + k] = e
                for v in range(sub):
                    conn[(slot * subs + k) * sub + v] = rank[slot * per_cell + table[k * sub + v]]


@tack.kernel
def _tensor_values(cells, u, slot_of, per_cell, ijk, steps, corner_at, first, rank, out):
    # _values for a scalar field in a tensor basis: the cell's values gathered once,
    # not once a lattice point, then the same sum (the same bits).
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            coeffs = tack.local_array_like(u.basis_nodes, (u.ORDER + 1) ** u.DIMENSION)
            u.dofs_into(c, coeffs)
            for n in range(per_cell):
                k = slot * per_cell + n
                if first[k] == 1:
                    if corner_at[n] >= 0:
                        out[rank[k]] = u.corner_value(c, corner_at[n])
                    else:
                        out[rank[k]] = u.value_from(coeffs, _lattice_point(
                            n, cells.DIMENSION, steps, ijk))


@tack.kernel
def _sub_cells(cells, slot_of, per_cell, level, rank, sub, conn, parent):
    # Each refined cell's level^d sub-cells, in its frame: corners in VTK's order
    # of the cell's shape (a hexahedron's, a quad's, a line's).
    for c in cells:
        e = cells.entity_id(c)
        slot = slot_of[e]
        if slot >= 0:
            q = level + 1
            subs = 1
            for _ in range(cells.DIMENSION):
                subs *= level
            for k in range(subs):
                i = k % level
                j = (k // level) % level
                m = k // (level * level)
                out = (slot * subs + k) * sub
                parent[slot * subs + k] = e
                for corner in range(sub):
                    a = i + ((corner ^ (corner >> 1)) & 1)
                    b = j + ((corner >> 1) & 1)
                    h = m + ((corner >> 2) & 1)
                    if cells.DIMENSION == 1:
                        a = i + corner
                        b = 0
                        h = 0
                    node = a + q * (b + q * h)
                    conn[out + corner] = rank[slot * per_cell + node]


# ── The refined dataset ─────────────────────────────────────────────

def _lattice_tables(shape, level):
    """The lattice's nodes: kind, entity and place there (``bases.dof_tables``) as device
    arrays."""
    kind, entity, a, b, _ = bases.dof_tables(shape, level)
    return tuple(tack.array(np.asarray(x, np.int32)) for x in (kind, entity, a, b))


def _entity_tables(shape, order):
    """Each edge's and quad face's nodes of ``order`` (``bases.entity_nodes``), flat, as
    device arrays (a placeholder where the shape has none)."""
    edges, faces = bases.entity_nodes(shape, order)
    return tuple(tack.array(x.reshape(-1) if x.size else np.zeros(1, np.int32))
                 for x in (edges, faces))


def _face_stride(shape, order):
    """The length of each face's row in ``bases.entity_nodes``' faces (1 for none)."""
    faces = bases.entity_nodes(shape, order)[1]
    return max(int(faces.shape[1]) if faces.ndim == 2 else 0, 1)


def _typed(values, like):
    """A host table as a device array of ``like``'s type."""
    out = tack.empty((len(values),), like.dtype)
    tack.copyto(out, np.asarray(values).astype(like.dtype.numpy_dtype))
    return out


def _cell_ranks(data, slot_of, chosen, count):
    """Each refined cell's rank for choosing evaluators: its global id where the data
    has them, else its centroid; either the same in every piece."""
    from tack.runtime.dispatch import backend_of

    wide = backend_of(data.device).supports_f64
    out = tack.empty((count,), tack.vec(3, tack.f64 if wide else tack.f32))
    gids = data.global_ids.cells if data.global_ids is not None else None
    if gids is not None:
        _id_ranks(gids, chosen, out)
    else:
        for group in data.launch_groups("cells", []):
            if group.count:
                _centroid_ranks(data.domain_view("cells", group), slot_of, out)
    return out


def _cell_family(data, what):
    """``(dimension, family)``: the cells' dimension and family, ``"tensor"``
    (segments, quads, pixels, hexahedra, voxels), ``"simplex"`` (triangles,
    tetrahedra) or ``"wedge"``. One family and one dimension per mesh."""
    groups = [g for g in data.topology.groups() if g.count]
    for group in groups:
        if not bases.has_bases(group.shape):
            raise NotImplementedError(
                f"{what} is on segments, quads, pixels, hexahedra, voxels, triangles, "
                f"tetrahedra and wedges; not yet on {group.shape.__name__} cells")
    families = {_family(g.shape) for g in groups}
    dims = {g.shape.DIMENSION for g in groups}
    if len(dims) > 1 or len(families) > 1:
        raise NotImplementedError(f"{what} takes cells of one dimension, all simplices, all "
                                  "wedges or all tensor shapes")
    return (dims.pop() if dims else 3), (families.pop() if families else "tensor")


def _family(shape):
    if bases.simplex_shape(shape):
        return "simplex"
    return "wedge" if bases.wedge_shape(shape) else "tensor"


def _order(field):
    space = field.space
    if isinstance(space, (H1, L2)) and not space.varies:
        return space.order
    return 1


def default_level(data, *fields):
    """The lattice steps a cell is cut into by default: the largest order of the
    geometry and ``fields`` (1 when every one is linear)."""
    return max([_order(data.geometry)] + [_order(f) for f in fields])


def _refined(data, flags, level, fields, elsewhere=None, cut=None):
    """The cells ``flags`` marks, cut into ``level`` steps along each axis: an
    unstructured linear dataset of their sub-cells, its geometry and ``fields`` (a
    dict of names to fields) evaluated at the lattice points (``H1`` and ``L2`` fields of any order;
    an ``L2`` field is evaluated from one cell at a shared point, the one every piece
    chooses alike, so its jumps are not kept), cell fields taken from each sub-cell's parent.
    ``elsewhere`` (i32 per point, optional) flags points another part owns: the
    refined dataset's point field ``SEAM`` is 1 at the lattice corners on them, 0
    elsewhere. ``cut``, ``(name, levels)``, makes a cut of field ``name`` at
    ``levels`` put its crossings at the field's roots (``_crossings``).
    Returns ``(refined, parent)``, ``parent`` each sub-cell's cell."""
    from math import comb

    dimension, cell_family = _cell_family(data, "subdivision")
    simplex = cell_family != "tensor"       # simplices and wedges: table-driven sub-cells
    topology = data.topology
    n = data.num_cells
    slots = tack.empty((n,), tack.i64)
    count = exclusive_scan(flags, slots, n) if n else 0
    slot_of = tack.empty((n,), tack.i64)
    chosen = tack.empty((max(count, 1),), tack.i64)
    if n:
        _place_chosen(flags, slots, slot_of, chosen)
    per_cell = {"simplex": comb(level + dimension, dimension),
                "wedge": comb(level + 2, 2) * (level + 1)}.get(cell_family,
                                                               (level + 1) ** dimension)
    subs = level ** dimension
    sub = {"simplex": dimension + 1, "wedge": 6}.get(cell_family, 2 ** dimension)
    total = count * per_cell
    keys = tack.empty((total,), tack.i64)
    if count:
        if level == 1:
            corner_of = tack.array(np.asarray(bases.dof_tables(
                next(g.shape for g in topology.groups() if g.count), 1)[1], np.int32))
            for group in topology.groups():
                if group.count:
                    _corner_keys(group.view(), slot_of, corner_of, keys)
        else:
            space = H1(data, order=level, basis="equispaced")
            lattice = Field(space, ConstantArray(0.0, space.size, dtype=data.dtype))
            for group in data.launch_groups("cells", [lattice]):
                if group.count:
                    node_of = tack.array(bases.view_order(group.shape, level))
                    _lattice_keys(data.domain_view("cells", group), lattice.view(group),
                                  slot_of, per_cell, node_of, keys)
    by_key = argsort(keys) if total else tack.empty((0,), tack.i64)
    first = tack.zeros((total,), tack.i32)
    numbers = tack.empty((total,), tack.i64)
    rank = tack.empty((total,), tack.i64)
    starts = tack.empty((total,), tack.i32)
    ranks = None
    points = 0
    if total:
        _run_starts(keys, by_key, starts)
        points = exclusive_scan(starts, numbers, total)
        _ranks(by_key, starts, numbers, rank)
        ranks = _cell_ranks(data, slot_of, chosen, count)
        _evaluators(by_key, starts, per_cell, ranks, first)
    geometry = tack.empty((points,), tack.vec(3, data.dtype))
    idt = ids.at_least(data.id_dtype, max(points, count * subs * sub))
    conn = tack.empty((count * subs * sub,), idt)
    parent = tack.empty((count * subs,), data.id_dtype)
    steps = tack.empty((level + 1,), data.dtype)
    tack.copyto(steps, (np.arange(level + 1) / level).astype(data.dtype.numpy_dtype))
    corner_at, ijk = {}, {}
    for group in topology.groups():
        if group.count:
            kind, entity, _, _, _ = bases.dof_tables(group.shape, level)
            corner_at[group.root] = tack.array(np.where(kind == bases.CORNER, entity, -1)
                                               .astype(np.int32))
            ijk[group.root] = tack.array(np.ascontiguousarray(
                bases.lattice_ijk(group.shape, level)).reshape(-1))
    if count:
        for group in data.launch_groups("cells", []):
            if group.count:
                cells = data.domain_view("cells", group)
                _positions(cells, slot_of, per_cell, ijk[group.root], steps,
                           corner_at[group.root], first, rank, geometry)
                lattice = _lattice_tables(group.shape, level)
                order = data.geometry.space.order if isinstance(data.geometry.space, H1) else 0
                if simplex:
                    if order == 1:
                        _trace_simplex_straight(cells, slot_of, per_cell, level, steps,
                                                *lattice, first, rank, geometry)
                    elif order >= 2:
                        edge_nodes, face_nodes = _entity_tables(group.shape, order)
                        family = data.geometry.space.basis or "equispaced"
                        _trace_simplex_curved(
                            cells, slot_of, per_cell, level, steps, *lattice, first, rank,
                            edge_nodes, face_nodes, _face_stride(group.shape, order),
                            tack.array(bases.view_index(group.shape, order, family)),
                            _typed(bases.nodes(family, order), steps),
                            _typed(bases.weights(family, order), steps), order + 1,
                            int(family == bases.BERNSTEIN), geometry)
                    table = tack.array(np.ascontiguousarray(
                        bases.wedge_subdivision(level) if cell_family == "wedge" else
                        bases.simplex_subdivision(group.shape, level)).reshape(-1))
                    _sub_simplices(cells, slot_of, per_cell, subs, sub, table, rank, conn,
                                   parent)
                    continue
                if order == 1:
                    _trace_straight(cells, slot_of, per_cell, level, steps, *lattice, first, rank,
                                    geometry)
                elif order >= 2:
                    edge_nodes, face_nodes = _entity_tables(group.shape, order)
                    _trace_curved(cells, slot_of, per_cell, level, steps, *lattice, first, rank,
                                  edge_nodes, face_nodes, order + 1,
                                  int(data.geometry.space.basis == bases.BERNSTEIN), geometry)
                _sub_cells(cells, slot_of, per_cell, level, rank, sub, conn, parent)
    shape = (shapes.WEDGE if cell_family == "wedge" else
             {3: shapes.TETRA, 2: shapes.TRIANGLE}[dimension] if simplex else
             {3: shapes.HEXAHEDRON, 2: shapes.QUAD, 1: shapes.LINE}[dimension])
    types, offsets = uniform_cells(int(shape), count * subs, idt)
    linear = UnstructuredTopology(types, offsets, conn, num_points=points, id_dtype=idt)
    point_fields = {}
    for name, field in fields.items():
        if not isinstance(field.space, (H1, L2)) or field.space.varies:
            continue
        out = tack.empty((points,), _element(field.values))
        if count:
            for group in data.launch_groups("cells", [field]):
                if group.count:
                    cells, u = data.domain_view("cells", group), field.view(group)
                    # A scalar in a tensor basis gathers each cell's values once.
                    kernel = (_tensor_values if isinstance(u, views._TensorBasis)
                              and not arrays.width_of(field.values)
                              and out.dtype == u.basis_nodes.dtype else _values)
                    kernel(cells, u, slot_of, per_cell, ijk[group.root], steps,
                           corner_at[group.root], first, rank, out)
                    if isinstance(field.space, H1) and simplex:
                        # A continuous field's shared points from their entities' values.
                        order = field.space.order
                        edge_nodes, face_nodes = _entity_tables(group.shape, order)
                        family = field.space.basis or "equispaced"
                        _trace_simplex_values(
                            cells, u, slot_of, per_cell, level, steps,
                            *_lattice_tables(group.shape, level), first, rank, edge_nodes,
                            face_nodes, _face_stride(group.shape, order),
                            tack.array(bases.view_index(group.shape, order, family)),
                            _typed(bases.nodes(family, order), steps),
                            _typed(bases.weights(family, order), steps), order + 1,
                            int(family == bases.BERNSTEIN), out)
                    elif isinstance(field.space, H1):
                        # A continuous field's shared points from their entities' values.
                        order = field.space.order
                        edge_nodes, face_nodes = _entity_tables(group.shape, order)
                        family = field.space.basis or "gauss_lobatto"
                        _trace_values(cells, u, slot_of, per_cell, level, steps,
                                      *_lattice_tables(group.shape, level), first, rank,
                                      edge_nodes, face_nodes,
                                      tack.array(bases.view_index(group.shape, order, family)),
                                      _typed(bases.nodes(family, order), steps),
                                      _typed(bases.weights(family, order), steps), order + 1,
                                      int(family == bases.BERNSTEIN), out)
        point_fields[name] = out
    from tack.dataset.dataset import DataSet

    refined = DataSet(linear, geometry, **data._frame())
    if data._ghosts_lost:
        refined._ghosts_lost = data._ghosts_lost
    elif data.ghosts is not None:
        refined.ghosts = _refined_marks(data, slot_of, per_cell, level, first, rank, parent,
                                        points)
    if elsewhere is not None:
        refined.fields[SEAM] = _seam(data, refined, elsewhere, slot_of, per_cell, level,
                                     first, rank)
    for name, values in point_fields.items():
        refined.fields[name] = Field(H1(refined), values, fields[name].unit)
    for name, field in fields.items():
        if isinstance(field.space, Constant) or (isinstance(field.space, Values)
                                                 and field.space.on == "cells"):
            refined.fields[name] = _taken(field, parent, refined)
    if cut is not None and total and cut[0] in point_fields:
        from tack.dataset.algorithms._crossings import Crossings

        name, levels = cut
        sources = {id(refined.geometry): data.geometry}
        sources.update({id(refined.fields[k]): fields[k] for k in point_fields})
        refined._crossings = Crossings(data, level, by_key, starts, numbers, points,
                                       per_cell, chosen, ranks, fields[name],
                                       refined.fields[name].values, levels, sources)
    return refined, parent


def _element(values):
    """``values``' element type: a scalar's, or a vector's."""
    width = arrays.width_of(values)
    dtype = arrays.dtype_of(values)
    return dtype if not width or width == 1 else tack.vec(width, dtype)


def _refined_marks(data, slot_of, per_cell, level, first, rank, parent, points):
    """The refined dataset's ghost marks: a sub-cell its parent's, a lattice point by
    the rules ``carry`` gives points it makes (``_lattice_marks``)."""
    from tack.dataset.ghosts import Ghosts

    ghosts = data.ghosts
    cell_marks = ghosts.cells if ghosts.cells is not None else \
        tack.zeros((data.num_cells,), tack.u8)
    cells_out = tack.empty((size_of(parent),), tack.u8)
    if size_of(parent):
        _take_values(cell_marks, parent, cells_out)
    points_out = None
    if ghosts.points is not None:
        points_out = tack.zeros((points,), tack.u8)
        for group in data.launch_groups("cells", []):
            if group.count:
                kind, entity, _, _, _ = bases.dof_tables(group.shape, level)
                _lattice_marks(data.domain_view("cells", group), slot_of, per_cell,
                               tack.array(kind), tack.array(entity), first, rank,
                               ghosts.points, cell_marks, points_out)
    return Ghosts(cells=cells_out, points=points_out)


#: The refined dataset's field of corners another part owns (``_refined``'s ``elsewhere``).
SEAM = "__seam__"


def _seam(data, refined, elsewhere, slot_of, per_cell, level, first, rank):
    """A point field of ``refined``: 1 at the lattice corners on points ``elsewhere``
    flags, 0 elsewhere (in the geometry's type, so a filter carries it)."""
    out = tack.zeros((refined.num_points,), data.dtype)
    for group in data.launch_groups("cells", []):
        if group.count:
            kind, entity, _, _, _ = bases.dof_tables(group.shape, level)
            _seam_marks(data.domain_view("cells", group), slot_of, per_cell,
                        tack.array(np.asarray(kind, np.int32)),
                        tack.array(np.asarray(entity, np.int32)), first, rank, elsewhere, out)
    return Field(H1(refined), out)


@tack.kernel
def _duplicates(seam, marks):
    # A point the seam field gives 1, a corner of a whole cell: DUPLICATEPOINT.
    for i in range(marks.shape[0]):
        if seam[i] == 1:
            marks[i] = marks[i] | 1


def _seam_marked(part):
    """``part`` without its ``SEAM`` field, the points it gives 1 marked
    ``DUPLICATEPOINT`` (beside any marks it has)."""
    from tack.dataset.ghosts import Ghosts

    seam = part.fields.pop(SEAM)
    if part._ghosts_lost or not part.num_points:
        return part
    ghosts = part.ghosts
    marks = tack.zeros((part.num_points,), tack.u8)
    if ghosts is not None and ghosts.points is not None:
        tack.copyto(marks, ghosts.points)
    _duplicates(arrays.materialize(seam.values), marks)
    part.ghosts = Ghosts(cells=ghosts.cells if ghosts is not None else None, points=marks)
    return part


@tack.kernel
def _take_values(source, parent, out):
    for i in range(out.shape[0]):
        out[i] = source[parent[i]]


def _taken(field, parent, refined):
    """A cell field of the parents, one value a sub-cell."""
    source = arrays.materialize(field.values)
    out = tack.empty((size_of(parent),), _element(source))
    if size_of(parent):
        _take_values(source, parent, out)
    space = Constant(refined) if isinstance(field.space, Constant) else Values(refined, "cells")
    return Field(space, out, field.unit)


@on_inputs_device
def subdivide(data, level=None, fields=None):
    """``data``'s cells cut into ``level`` steps along each parametric axis -- linear
    hexahedra, quads or lines on each cell's lattice -- and its fields evaluated at
    the lattice points: a curved or high-order dataset as a linear one, as VTK's
    ``vtkCellGridToUnstructuredGrid`` gives one (for export, rendering, or a filter
    that takes only linear cells). ``level`` is by default the largest order of the
    geometry and fields. A lattice point shared by cells is one point, its value
    computed once. ``H1`` and ``L2`` fields become point data (an ``L2`` field's
    value at a shared point from one of its cells); cell fields go to every
    sub-cell of their cell. Segments, quads, pixels, hexahedra and voxels."""
    names = list(data.fields) if fields is None else list(fields)
    level = default_level(data, *(data.fields[name] for name in names)) if level is None \
        else int(level)
    if level < 1:
        raise ValueError(f"subdivide's level is at least 1, not {level}")
    flags = tack.empty((data.num_cells,), tack.i32)
    if data.num_cells:
        for_each(_all_cells, data, "cells", flags)
    refined, _ = _refined(data, flags, level, {name: data.fields[name] for name in names})
    return refined


def level_for(data, field, flags, tolerance):
    """The fewest lattice steps that cut ``field``'s flagged cells within ``tolerance``
    (in the field's values): each cell's Bernstein coefficients' largest distance
    from the corners' multilinear interpolation, ``D``, bounds what linearizing it
    loses, and a lattice of ``s`` steps divides that by about ``s^2`` (exactly for a
    quadratic), so ``s = ceil(sqrt(D / tolerance))``, at most ``MAX_LEVEL``."""
    tolerance = float(tolerance)
    if not tolerance > 0:
        raise ValueError(f"tolerance is a positive number, not {tolerance}")
    out = tack.empty((data.num_cells,), arrays.dtype_of(field.values))
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _flatness(data.domain_view("cells", group), field.view(group), flags, out)
    worst = float(out.max()) if data.num_cells else 0.0
    return int(min(MAX_LEVEL, max(1, np.ceil(np.sqrt(worst / tolerance)))))


def subdivides(data, field):
    """Whether cutting ``field`` on ``data`` takes subdivision: a curved geometry or a
    field above order 1, on tensor cells (segments, quads, pixels, hexahedra,
    voxels)."""
    nonlinear = _order(data.geometry) > 1 or (isinstance(field.space, (H1, L2))
                                              and not field.space.varies
                                              and (field.space.order > 1
                                                   or getattr(field.space, "tensor", False)))
    groups = [g for g in data.topology.groups() if g.count]
    families = {_family(g.shape) for g in groups if bases.has_bases(g.shape)}
    return nonlinear and len(families) <= 1 and all(bases.has_bases(g.shape) for g in groups)


def subdivided_contour(data, field, value, levels, nlevels, subdivision, cut, tolerance=None,
                       exact=True):
    """``cut`` (contour with its other arguments bound) of the cells whose ``field``
    may hold one of ``levels``, cut into ``subdivision`` lattice steps (by default
    the largest order of the field and geometry, or with ``tolerance`` the fewest
    whose estimated error is within it): only those cells are refined, and every
    field carried evaluated at their lattice points; with ``exact``, at the field's
    roots along the sub-edges where it crosses (``_crossings``)."""
    if subdivision is not None and tolerance is not None:
        raise ValueError("give subdivision or tolerance, not both")
    flags = tack.empty((data.num_cells,), tack.i32)
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _straddling(data.domain_view("cells", group), field.view(group), levels, nlevels,
                        flags)
    if tolerance is not None:
        level = level_for(data, field, flags, tolerance)
    else:
        level = default_level(data, field) if subdivision is None else int(subdivision)
    if level < 1:
        raise ValueError(f"subdivision is at least 1, not {level}")
    name = next((n for n, f in data.fields.items() if f is field), "__cut__")
    carried = dict(data.fields)
    carried[name] = field
    refined, _ = _refined(data, flags, level, carried, cut=(name, levels) if exact else None)
    out = cut(refined, name)
    if name == "__cut__":
        out.fields.pop(name, None)
    return out


def split_by_range(data, field, lower, upper, subdivision, tolerance, fields, cut,
                   levels=None):
    """``clip``'s and ``threshold``'s high-order path: ``PartitionedDataSet([whole,
    part])``. ``whole`` is the cells whose ``field`` lies in ``[lower, upper]``
    over the whole cell by its Bernstein bound, kept as they are (curved, their
    fields of any order); ``part`` is ``cut`` (the linear filter, given the refined
    dataset and the field's name there) of the cells whose bound overlaps the
    range's ends, cut into ``subdivision`` lattice steps (by default the largest
    order of the field and geometry; with ``tolerance``, the fewest whose
    estimated error is within it). ``fields`` names the fields carried (``None``
    every one). With ``levels`` (the values a clip cuts at, an array of the field's
    type), the cut's crossings are at the field's roots (``_crossings``). The two share the whole cells' corners on the seam, which the whole
    part owns: the cut part's copies are marked ``DUPLICATEPOINT``, so reductions
    over the two count each once."""
    from tack.dataset.algorithms._extract import _keep_cells
    from tack.dataset.partitioned import PartitionedDataSet

    if subdivision is not None and tolerance is not None:
        raise ValueError("give subdivision or tolerance, not both")
    n = data.num_cells
    whole = tack.empty((n,), tack.i32)
    flags = tack.empty((n,), tack.i32)
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _by_range(data.domain_view("cells", group), field.view(group), float(lower),
                      float(upper), whole, flags)
    if tolerance is not None:
        level = level_for(data, field, flags, tolerance)
    else:
        level = default_level(data, field) if subdivision is None else int(subdivision)
    if level < 1:
        raise ValueError(f"subdivision is at least 1, not {level}")
    name = next((k for k, f in data.fields.items() if f is field), "__cut__")
    names = list(data.fields) if fields is None else list(fields)
    carried = {k: data.fields[k] for k in names}
    carried[name] = field
    # The whole cells' corners are the whole part's: the cut part's copies are ghosts.
    elsewhere = tack.zeros((data.num_points,), tack.i32)
    for group in data.launch_groups("cells", []):
        if group.count:
            _corners_of(data.domain_view("cells", group), whole, elsewhere)
    refined, _ = _refined(data, flags, level, carried, elsewhere,
                          cut=None if levels is None else (name, levels))
    part = cut(refined, name, None if fields is None else [*names, SEAM])
    if name not in names:
        part.fields.pop(name, None)
    part = _seam_marked(part)
    return PartitionedDataSet([_keep_cells(data, whole, fields), part])
