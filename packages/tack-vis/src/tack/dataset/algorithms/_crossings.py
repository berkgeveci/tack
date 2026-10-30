"""Exact crossings: where a cut through sub-cells crosses a sub-edge, the field's own
root along it, not the crossing of its lattice values' straight line.

A cut of high-order data runs a linear filter on the refined dataset
(``subdivide._refined``), whose crossings ``carry`` makes as ``Interpolate`` points
between two lattice points. The refined dataset holds a ``Crossings`` object, and
``carry`` asks it to place them (``located``): each crossing's sub-edge is found in
a cell holding both its lattice points, on the smallest entity of that cell holding
both -- one of its edges, one of its faces, or its inside -- and the root of the
field less the cut's value is found along it by a fixed number of Illinois steps,
from the lattice values the linear filter used, so the root is where they put a
crossing and the cut keeps its cells. ``carry`` then takes the geometry and every
field there instead of between the lattice values.

As subdivision computes a lattice point, a root on an edge or face is computed from
that entity's own values, in a frame its corners' positions fix
(``subdivide._edge_step``, ``_face_steps``, ``_triangle_order``), and along the
sub-edge from the end the filter puts first by position: the same bits in every
cell and every piece holding it. Inside a cell, the cell's own basis. ``L2``
fields, which have no traces, are evaluated in the holding cell that comes first
by rank (``_cell_ranks``), as subdivision chooses a lattice point's.
"""

import numpy as np

import tack
from tack import arrays
from tack.dataset import bases
from tack.dataset.algorithms.subdivide import (
    _ahead,
    _axis,
    _edge_step,
    _face_steps,
    _face_weights,
    _lattice_point,
    _triangle_order,
    _triangle_term,
    _unit,
)
from tack.dataset.buckets import bucket_order
from tack.dataset.ghosts import _before
from tack.dataset.spaces import H1, L2

#: Illinois steps a root takes at most: regula falsi from the bracket (its first
#: step the linear crossing) that halves the value kept at an end twice running, so
#: it converges superlinearly and stops when a step makes no progress; the count is
#: fixed so the bits are.
ROOT_STEPS = tack.constant(32)




# ── A sub-edge's entity and its ends in the entity's frame ───────────

@tack.func
def _ends(cells, c, na, nb, on_edges, on_faces, ne, nf, level, steps, ijk):
    # The smallest entity of cell c holding lattice nodes na and nb: kind 0 an edge,
    # 1 a face, 2 the inside; m its local number; A, B the ends in its frame (an
    # edge's t from its first corner by position; a quad face's (ta, tb) in
    # _face_steps' frame; a triangle's steps on its corners ordered by position);
    # PA, PB the ends' parametric coordinates; k1, k2, k3 the frame's corners (an
    # edge's reversal in k1); corners the face's corner count.
    pa = _lattice_point(na, cells.DIMENSION, steps, ijk)
    pb = _lattice_point(nb, cells.DIMENSION, steps, ijk)
    a = pa
    b = pb
    kind = 2
    m = 0
    k1 = 0
    k2 = 0
    k3 = 0
    corners = 0
    for e in range(ne):
        if kind == 2 and on_edges[na * ne + e] >= 0 and on_edges[nb * ne + e] >= 0:
            kind = 0
            m = e
    for f in range(nf):
        if kind == 2 and on_faces[(na * nf + f) * 2] >= 0 and on_faces[(nb * nf + f) * 2] >= 0:
            kind = 1
            m = f
    if kind == 0:
        sa, flip = _edge_step(cells, c, m, on_edges[na * ne + m], level)
        sb, _ = _edge_step(cells, c, m, on_edges[nb * ne + m], level)
        a = tack.vec3(steps[sa], 0.0, 0.0)
        b = tack.vec3(steps[sb], 0.0, 0.0)
        k1 = flip
    elif kind == 1:
        corners = cells.face_num_points(m)
        ia = on_faces[(na * nf + m) * 2]
        ja = on_faces[(na * nf + m) * 2 + 1]
        ib = on_faces[(nb * nf + m) * 2]
        jb = on_faces[(nb * nf + m) * 2 + 1]
        if corners == 4:
            sa0, sb0, ka, kb, kd = _face_steps(cells, c, m, ia, ja, level)
            sa1, sb1, _, _, _ = _face_steps(cells, c, m, ib, jb, level)
            a = tack.vec3(steps[sa0], steps[sb0], 0.0)
            b = tack.vec3(steps[sa1], steps[sb1], 0.0)
            k1 = ka
            k2 = kb
            k3 = kd
        else:
            ka, kb, kc = _triangle_order(cells, c, m)
            wa0, wb0, wc0 = _face_weights(ka, kb, kc, ia, ja, level)
            wa1, wb1, wc1 = _face_weights(ka, kb, kc, ib, jb, level)
            a = tack.vec3(steps[wa0], steps[wb0], steps[wc0])
            b = tack.vec3(steps[wa1], steps[wb1], steps[wc1])
            k1 = ka
            k2 = kb
            k3 = kc
    return kind, m, a, b, pa, pb, k1, k2, k3, corners


@tack.func
def _oriented(cells, c, na, nb, on_edges, on_faces, ne, nf, level, steps, ijk):
    # _ends with the sub-edge walked from the end first in the entity's frame (by
    # its coordinates, which its corners' positions fix), whichever end a filter
    # names first; and whether that is the filter's second end.
    kind, m, a, b, pa, pb, k1, k2, k3, corners = _ends(cells, c, na, nb, on_edges, on_faces,
                                                        ne, nf, level, steps, ijk)
    turned = 0
    if _before(b, a) if kind != 2 else _before(pb, pa):
        turned = 1
        a, b = b, a
        pa, pb = pb, pa
    return kind, m, a, b, pa, pb, k1, k2, k3, corners, turned


# ── A trace's terms ──────────────────────────────────────────────────

@tack.device_class
class _Nodes:
    """One basis's nodes on a shape's edges and faces (``bases.entity_nodes``,
    through ``view_of`` to the view's numbering), and its 1D nodes and barycentric
    weights (Bernstein's binomials, with ``bern``), for evaluating a trace."""

    def __init__(self, edge_nodes, face_nodes, view_of, nodes, weights, q, stride, bern):
        self.edge_nodes = edge_nodes
        self.face_nodes = face_nodes
        self.view_of = view_of
        self.nodes = nodes
        self.weights = weights
        self.q = q
        self.stride = stride
        self.bern = bern

    @tack.func
    def count(self, kind, corners):
        # How many terms the trace on that entity has.
        n = self.q
        if kind == 1:
            n = self.q * self.q if corners == 4 else (self.q * (self.q + 1)) // 2
        return n

    @tack.func
    def term(self, kind, m, x, k1, k2, k3, corners, i):
        # Term i of the trace at x (in the entity's frame): (the node's index in the
        # view, its weight), in the frame's order, so every cell sums alike.
        q = self.q
        p = q - 1
        index = 0
        w = x[0] - x[0]
        if kind == 0:
            j = q - 1 - i if k1 == 1 else i
            index = self.view_of[self.edge_nodes[m * q + j]]
            w = _axis(x[0], i, self.nodes, self.weights, q, self.bern)
        elif corners == 4:
            jp = i % q
            kp = i // q
            ax, ay = _unit(k1)
            bx, by = _unit(k2)
            dx, dy = _unit(k3)
            fa = ax * p + jp * (bx - ax) + kp * (dx - ax)
            fb = ay * p + jp * (by - ay) + kp * (dy - ay)
            index = self.view_of[self.face_nodes[m * self.stride + fa + q * fb]]
            w = _axis(x[0], jp, self.nodes, self.weights, q, self.bern) * \
                _axis(x[1], kp, self.nodes, self.weights, q, self.bern)
        else:
            bc = 0
            r = i
            while r >= q - bc:
                r -= q - bc
                bc += 1
            bb = r
            ba = p - bb - bc
            s1 = ba if k1 == 1 else (bb if k2 == 1 else bc)
            s2 = ba if k1 == 2 else (bb if k2 == 2 else bc)
            t = s2 * (p + 1) - (s2 * (s2 - 1)) // 2 + s1
            index = self.view_of[self.face_nodes[m * self.stride + t]]
            w = _triangle_term(x[0], x[1], x[2], ba, bb, bc, p, self.bern)
        return index, w


@tack.func
def _field_at(u, c, nodes, traced, kind, m, x, pc, k1, k2, k3, corners):
    # A field at a point of cell c: by its trace on an edge or face (traced), else by
    # the cell's basis at pc.
    if traced == 1 and kind != 2:
        index, w = nodes.term(kind, m, x, k1, k2, k3, corners, 0)
        total = u.dof(c, index) * w
        for i in range(1, nodes.count(kind, corners)):
            index, w = nodes.term(kind, m, x, k1, k2, k3, corners, i)
            total += u.dof(c, index) * w
        return total
    return u.value(c, pc)


@tack.func
def _curved_at(cells, c, nodes, kind, m, x, pc, k1, k2, k3, corners):
    # A curved geometry at a point of cell c, by its own nodes on the entity.
    if kind != 2:
        index, w = nodes.term(kind, m, x, k1, k2, k3, corners, 0)
        total = cells.get(cells.dof_index(c, index)) * w
        for i in range(1, nodes.count(kind, corners)):
            index, w = nodes.term(kind, m, x, k1, k2, k3, corners, i)
            total += cells.get(cells.dof_index(c, index)) * w
        return total
    return cells.position(c, pc)


@tack.func
def _straight_at(cells, c, kind, m, x, pc, k1, k2, k3, corners):
    # A straight geometry at a point of cell c: along an edge from its first corner,
    # bilinear over a quad face in its frame, barycentric over a triangle.
    if kind == 0:
        xa = cells.corner_position(c, cells.edge_corner(m, 1 if k1 == 1 else 0))
        xb = cells.corner_position(c, cells.edge_corner(m, 0 if k1 == 1 else 1))
        return xa + x[0] * (xb - xa)
    if kind == 1 and corners == 4:
        ta = x[0]
        tb = x[1]
        xa = cells.corner_position(c, cells.face_corner(m, k1))
        xb = cells.corner_position(c, cells.face_corner(m, k2))
        xc = cells.corner_position(c, cells.face_corner(m, (k1 + 2) % 4))
        xd = cells.corner_position(c, cells.face_corner(m, k3))
        return (xa + ta * (1.0 - tb) * (xb - xa) + ta * tb * (xc - xa)
                + (1.0 - ta) * tb * (xd - xa))
    if kind == 1:
        xa = cells.corner_position(c, cells.face_corner(m, k1))
        return (xa + x[1] * (cells.corner_position(c, cells.face_corner(m, k2)) - xa)
                + x[2] * (cells.corner_position(c, cells.face_corner(m, k3)) - xa))
    return cells.position(c, pc)


# ── Kernels ─────────────────────────────────────────────────────────

@tack.kernel
def _run_firsts(starts, numbers, points, out):
    # Where each refined point's run of (cell, node) entries starts in key order, and
    # past the last.
    total = starts.shape[0]
    for i in range(total):
        if starts[i] == 1:
            out[numbers[i]] = i
        if i == total - 1:
            out[points] = total


@tack.kernel
def _holding(ends, firsts, by_key, per_cell, chosen, cell_rank, cell, na, nb):
    # Each crossing's cell holding both its ends, the first by rank, and the ends'
    # nodes there.
    for i in range(ends.shape[0]):
        a = ends[i][0]
        b = ends[i][1]
        best = -1
        for ia in range(firsts[a], firsts[a + 1]):
            ka = by_key[ia]
            slot = ka // per_cell
            for ib in range(firsts[b], firsts[b + 1]):
                kb = by_key[ib]
                if kb // per_cell == slot:
                    if best < 0 or _ahead(cell_rank[slot], cell_rank[best]):
                        best = slot
                        na[i] = ka - slot * per_cell
                        nb[i] = kb - slot * per_cell
        cell[i] = chosen[best]


@tack.kernel
def _roots(cells, u, nodes, traced, offsets, order, na, nb, ends, weights, lattice, levels,
           nlevels, on_edges, on_faces, ne, nf, level, steps, ijk, out):
    # Each crossing's root along its sub-edge, walked from its first end in the
    # entity's frame (_oriented): Illinois steps on the field less the cut's value,
    # bracketed by the lattice values the linear filter crossed between. -1 for a
    # crossing at a lattice point, which keeps it.
    for c in cells:
        e = cells.entity_id(c)
        for r in range(offsets[e], offsets[e + 1]):
            i = order[r]
            w = weights[i]
            s = w - w - 1.0
            if w > 0.0 and w < 1.0:
                kind, m, a, b, pa, pb, k1, k2, k3, corners, turned = _oriented(
                    cells, c, na[i], nb[i], on_edges, on_faces, ne, nf, level, steps, ijk)
                va = lattice[ends[i][1 if turned == 1 else 0]]
                vb = lattice[ends[i][0 if turned == 1 else 1]]
                guess = lattice[ends[i][0]] + w * (lattice[ends[i][1]] - lattice[ends[i][0]])
                value = levels[0]
                for j in range(1, nlevels):
                    if abs(levels[j] - guess) < abs(value - guess):
                        value = levels[j]
                s = 1.0 - w if turned == 1 else w
                lo = s - s
                hi = lo + 1.0
                glo = va - value
                ghi = vb - value
                side = 0
                if glo * ghi < 0.0:
                    for _ in range(ROOT_STEPS):
                        mid = (lo * ghi - hi * glo) / (ghi - glo)
                        if mid <= lo or mid >= hi:
                            break
                        s = mid
                        gm = _field_at(u, c, nodes, traced, kind, m, a + s * (b - a),
                                       pa + s * (pb - pa), k1, k2, k3, corners) - value
                        if gm == 0.0:
                            break
                        if (gm < 0.0) == (glo < 0.0):
                            lo = s
                            glo = gm
                            if side == -1:
                                ghi = ghi * 0.5
                            side = -1
                        else:
                            hi = s
                            ghi = gm
                            if side == 1:
                                glo = glo * 0.5
                            side = 1
            out[i] = s


@tack.kernel
def _field_rows(cells, u, nodes, traced, offsets, order, na, nb, roots, on_edges, on_faces,
                ne, nf, level, steps, ijk, out):
    # A field at each crossing's root (a crossing at a lattice point keeps its value).
    for c in cells:
        e = cells.entity_id(c)
        for r in range(offsets[e], offsets[e + 1]):
            i = order[r]
            s = roots[i]
            if s >= 0.0:
                kind, m, a, b, pa, pb, k1, k2, k3, corners, _ = _oriented(
                    cells, c, na[i], nb[i], on_edges, on_faces, ne, nf, level, steps, ijk)
                out[i] = _field_at(u, c, nodes, traced, kind, m, a + s * (b - a),
                                   pa + s * (pb - pa), k1, k2, k3, corners)


@tack.kernel
def _curved_rows(cells, nodes, offsets, order, na, nb, roots, on_edges, on_faces, ne, nf,
                 level, steps, ijk, out):
    for c in cells:
        e = cells.entity_id(c)
        for r in range(offsets[e], offsets[e + 1]):
            i = order[r]
            s = roots[i]
            if s >= 0.0:
                kind, m, a, b, pa, pb, k1, k2, k3, corners, _ = _oriented(
                    cells, c, na[i], nb[i], on_edges, on_faces, ne, nf, level, steps, ijk)
                out[i] = _curved_at(cells, c, nodes, kind, m, a + s * (b - a),
                                    pa + s * (pb - pa), k1, k2, k3, corners)


@tack.kernel
def _straight_rows(cells, offsets, order, na, nb, roots, on_edges, on_faces, ne, nf, level,
                   steps, ijk, out):
    for c in cells:
        e = cells.entity_id(c)
        for r in range(offsets[e], offsets[e + 1]):
            i = order[r]
            s = roots[i]
            if s >= 0.0:
                kind, m, a, b, pa, pb, k1, k2, k3, corners, _ = _oriented(
                    cells, c, na[i], nb[i], on_edges, on_faces, ne, nf, level, steps, ijk)
                out[i] = _straight_at(cells, c, kind, m, a + s * (b - a), pa + s * (pb - pa),
                                      k1, k2, k3, corners)


# ── The hook ────────────────────────────────────────────────────────

def _nodes_of(space, shape, dtype, geometry=False):
    """``_Nodes`` for an ``H1`` space of order ``p`` on ``shape`` cells (its basis; by
    default the simplices' and wedges' equispaced, the tensor shapes'
    Gauss-Lobatto). A tensor geometry's view takes lexicographic nodes, as
    ``_trace_curved`` reads it."""
    from tack.dataset.algorithms.subdivide import _entity_tables, _face_stride, _typed

    order = space.order
    edge_nodes, face_nodes = _entity_tables(shape, order)
    family = space.basis or ("gauss_lobatto" if bases.tensor_shape(shape) else "equispaced")
    like = tack.empty((1,), dtype)
    view_of = np.arange((order + 1) ** shape.DIMENSION, dtype=np.int32) if (
        geometry and bases.tensor_shape(shape)) else bases.view_index(
        shape, order, space._any_order(shape))
    return _Nodes(edge_nodes, face_nodes, tack.array(np.asarray(view_of, np.int32)),
                  _typed(bases.nodes(family, order), like),
                  _typed(bases.weights(family, order), like), order + 1,
                  _face_stride(shape, order), int(family == bases.BERNSTEIN))


class Crossings:
    """What a refined dataset needs to put its cuts' crossings at their roots:
    ``data`` (the dataset refined) and ``level``; where each refined point's
    (cell, node) entries are in key order (``by_key``, from ``starts`` and
    ``numbers``), ``per_cell``, the refined cells (``chosen``) and their ranks;
    the field cut (``cut``), its values at the refined points (``lattice``) and the
    values it is cut at (``levels``); and each refined field's source
    (``sources``: id of the refined field or geometry to the field refined)."""

    def __init__(self, data, level, by_key, starts, numbers, points, per_cell, chosen,
                 ranks, cut, lattice, levels, sources):
        self.data = data
        self.level = level
        self.by_key = by_key
        self.firsts = tack.empty((points + 1,), tack.i64)
        if points:
            _run_firsts(starts, numbers, points, self.firsts)
        self.per_cell = per_cell
        self.chosen = chosen
        self.ranks = ranks
        self.cut = cut
        self.lattice = lattice
        self.levels = levels
        self.sources = sources

    def located(self, interpolate):
        """``interpolate`` with its crossings' roots: an ``Interpolate`` whose
        ``exact`` evaluates a refined field's source at them."""
        from tack.dataset.carry import Interpolate

        return Interpolate(interpolate.ends, interpolate.weights, interpolate.averages,
                           interpolate.average_cells, exact=_Located(self, interpolate))


class _Located:
    """One cut's crossings placed: each one's holding cell, its ends' nodes there and
    its root, found once and used for every field."""

    def __init__(self, crossings, interpolate):
        x = crossings
        data = x.data
        n = int(interpolate.weights.shape[0])
        self.crossings = x
        self.count = n
        if not n:
            return
        self.cell = tack.empty((n,), data.id_dtype)
        self.na = tack.empty((n,), tack.i32)
        self.nb = tack.empty((n,), tack.i32)
        # At the wider of the geometry's and the field's precision: an f64 field's
        # root on an f32 mesh is an f64 root.
        wide = tack.f64 in (data.dtype, arrays.dtype_of(x.cut.values))
        self.roots = tack.empty((n,), tack.f64 if wide else tack.f32)
        _holding(interpolate.ends, x.firsts, x.by_key, x.per_cell, x.chosen, x.ranks,
                 self.cell, self.na, self.nb)
        self.order, self.offsets = bucket_order(self.cell, data.num_cells)
        cut = x.cut
        for group in data.launch_groups("cells", [cut]):
            if group.count:
                cells, tables = self._tables(group)
                _roots(cells, cut.view(group), *self._basis(cut, group), self.offsets,
                       self.order, self.na, self.nb, interpolate.ends, interpolate.weights,
                       x.lattice, x.levels, int(x.levels.shape[0]), *tables, self.roots)

    def _tables(self, group):
        data = self.crossings.data
        level = self.crossings.level
        on_edges, on_faces = bases.entity_coordinates(group.shape, level)
        steps = tack.empty((level + 1,), data.dtype)
        tack.copyto(steps, (np.arange(level + 1) / level).astype(data.dtype.numpy_dtype))
        return data.domain_view("cells", group), (
            tack.array(np.ascontiguousarray(on_edges).reshape(-1)),
            tack.array(np.ascontiguousarray(on_faces).reshape(-1)), int(on_edges.shape[1]),
            int(on_faces.shape[1]), level, steps,
            tack.array(np.ascontiguousarray(bases.lattice_ijk(group.shape, level)).reshape(-1)))

    def _basis(self, field, group):
        """``(nodes, traced)`` for a field: its traces' nodes, and whether it has
        traces (an ``H1`` field of one order; an ``L2`` field is evaluated in its
        cell, and given an order-1 ``H1``'s nodes it never reads)."""
        space = field.space
        dtype = arrays.dtype_of(field.values)
        if isinstance(space, H1):
            return _nodes_of(space, group.shape, dtype), 1
        return _nodes_of(H1(space.topology), group.shape, dtype), 0

    def __call__(self, field, out):
        """Write ``field``'s source (``field`` a refined field, or the refined
        geometry) at the roots into ``out``'s first rows; nothing for a field with
        no source."""
        source = self.crossings.sources.get(id(field))
        if source is None or not self.count:
            return
        data = self.crossings.data
        space = source.space
        if not isinstance(space, (H1, L2)) or space.varies:
            return
        is_geometry = source is data.geometry
        if is_geometry and not isinstance(space, H1):
            return
        for group in data.launch_groups("cells", [source]):
            if not group.count:
                continue
            cells, tables = self._tables(group)
            if is_geometry and space.order == 1:
                _straight_rows(cells, self.offsets, self.order, self.na, self.nb,
                               self.roots, *tables, out)
            elif is_geometry:
                nodes = _nodes_of(space, group.shape, arrays.dtype_of(source.values),
                                  geometry=True)
                _curved_rows(cells, nodes, self.offsets, self.order, self.na, self.nb,
                             self.roots, *tables, out)
            else:
                _field_rows(cells, source.view(group), *self._basis(source, group),
                            self.offsets, self.order, self.na, self.nb, self.roots, *tables,
                            out)
