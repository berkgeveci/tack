"""Vector spaces of the lowest order on hexahedra, voxels and tetrahedra: ``HDiv``, one value
per face, and ``HCurl``, one value per edge.

``HDiv`` (Raviart-Thomas) holds each face's integrated flux along the face's stored
normal, out of its side 0 -- the convention of ``Values(data, "faces", oriented=True)``,
``upwind_flux`` and ``net_flux`` -- and interpolates it inside each cell so that the
flux through every face is its value. ``HCurl`` (Nedelec, first kind) holds each
edge's tangential integral from its low point id to its high one, and interpolates
it so that the integral along every edge is its value. Both reproduce constant
fields exactly, and ``HDiv`` the fields whose x-component depends on x alone, and so
on: ``(a + b x, c + d y, e + f z)``.

A field view gives ``parametric_value(c, pc)``, the field pulled back to the
reference cell. ``hdiv_value`` and ``hcurl_value`` map it into the world through the
cell's geometry, as ``world_gradient`` maps ``parametric_gradient``: the contravariant
Piola map ``J v / det J``, which keeps fluxes, and the covariant ``J^-T v``, which
keeps tangential integrals.
"""

import numpy as np

import tack
from tack.dataset.spaces import Space

__all__ = ["HCurl", "HDiv", "hcurl_curl", "hcurl_value", "hdiv_div", "hdiv_value"]


def _is_hexahedral(shape):
    return getattr(shape, "DIMENSION", 0) == 3 and shape.NUM_FACES == 6 and shape.NUM_EDGES == 12


def _is_tetrahedral(shape):
    return getattr(shape, "DIMENSION", 0) == 3 and shape.NUM_POINTS == 4 and shape.NUM_FACES == 4


def _check_shape(space, group):
    shape = group.shape
    if not (_is_hexahedral(shape) or _is_tetrahedral(shape)):
        raise NotImplementedError(f"{type(space).__name__} is on hexahedra, voxels and tetrahedra, "
                                  f"not {shape.__name__}")


class _HDivField:
    """An HDiv field's value in the reference cell: each face's outward flux times its
    reference function, which has unit flux through that face and none through the others."""

    @tack.func
    def parametric_value(self, c, pc):
        v = tack.vec3(0.0, 0.0, 0.0)
        for k in range(6):
            center = tack.vec3(0.0, 0.0, 0.0)
            for i in range(4):
                center += self.parametric_point(self.face_corner(k, i))
            # The reference outward normal: +-1 along the face's axis, 0 elsewhere.
            n = 2.0 * (center * 0.25) - 1.0
            g = n.dot(pc) + (0.0 if n.sum() > 0.0 else 1.0)
            flux = self.get(self.face_id(c, k)) * (1.0 if self.face_side(c, k) == 0 else -1.0)
            v += n * (g * flux)
        return v

    @tack.func
    def parametric_divergence(self, c, pc):
        """Each face's function has divergence 1 in the reference cell."""
        d = 0.0 * self.get(0)
        for k in range(6):
            d += self.get(self.face_id(c, k)) * (1.0 if self.face_side(c, k) == 0 else -1.0)
        return d


def gauss_nodes(n):
    """n Gauss-Legendre points on [0, 1], increasing."""
    x, _ = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0)


def lobatto_nodes(n):
    """n Gauss-Lobatto points on [0, 1], increasing, with both ends."""
    inner = np.polynomial.legendre.Legendre.basis(n - 1).deriv().roots() if n > 2 else np.array([])
    return 0.5 * (np.concatenate([[-1.0], np.sort(inner.real), [1.0]]) + 1.0)


@tack.func
def _lagrange(nodes, n, i, x):
    """The i-th of the n Lagrange polynomials on nodes, at x."""
    r = 1.0 + 0.0 * x
    for m in range(n):
        if m != i:
            r = r * (x - nodes[m]) / (nodes[i] - nodes[m])
    return r


@tack.func
def _lagrange_derivative(nodes, n, i, x):
    s = 0.0 * x
    for m in range(n):
        if m != i:
            t = 1.0 / (nodes[i] - nodes[m])
            for q in range(n):
                if q != i and q != m:
                    t = t * (x - nodes[q]) / (nodes[i] - nodes[q])
            s = s + t
    return s


class _HDivHexOrderField:
    """HDiv of order P on a hexahedron (Raviart-Thomas): component a is a tensor product of
    P + 1 Gauss-Lobatto nodes along axis a and P Gauss nodes across it. The nodes on a face
    hold the normal flux density there, along the face's stored normal and in the face's own
    frame, shared by the cells on both sides; the others are the cell's own."""

    @tack.func
    def _face_of(self, a, side):
        """The local face whose center has coordinate side (0 or 1) along axis a."""
        found = 0
        for k in range(6):
            center = tack.vec3(0.0, 0.0, 0.0)
            for i in range(4):
                center += self.parametric_point(self.face_corner(k, i))
            if (0.25 * center[a] - float(side)) * (0.25 * center[a] - float(side)) < 0.01:
                found = k
        return found

    @tack.func
    def _face_dof(self, c, lf, a, j, k):
        """The value stored for local face lf at the cell's Gauss node (j, k) across axis a."""
        b = (a + 1) % 3
        cc = (a + 2) % 3
        p0 = tack.vec3(0.0, 0.0, 0.0)
        p1 = tack.vec3(0.0, 0.0, 0.0)
        p3 = tack.vec3(0.0, 0.0, 0.0)
        for i in range(4):
            where = self.face_position(c, lf, i)
            point = self.parametric_point(self.face_corner(lf, i))
            if where == 0:
                p0 = point
            if where == 1:
                p1 = point
            if where == 3:
                p3 = point
        u = p1 - p0
        w = p3 - p0
        n = self.P
        m = 0
        q = 0
        if u[b] * u[b] > 0.25:
            m = j if u[b] > 0.0 else n - 1 - j
            q = k if w[cc] > 0.0 else n - 1 - k
        else:
            m = k if u[cc] > 0.0 else n - 1 - k
            q = j if w[b] > 0.0 else n - 1 - j
        return self.get(self.face_id(c, lf) * n * n + m + n * q)

    @tack.func
    def _component(self, c, pc, a, derivative):
        """Component a at pc, or with derivative its derivative along axis a."""
        n = self.P
        b = (a + 1) % 3
        cc = (a + 2) % 3
        total = 0.0 * pc[0]
        for side in range(2):
            lf = self._face_of(a, side)
            sign = (1.0 if self.face_side(c, lf) == 0 else -1.0) * (1.0 if side == 1 else -1.0)
            la = (_lagrange_derivative(self.lobatto, n + 1, side * n, pc[a]) if derivative
                  else _lagrange(self.lobatto, n + 1, side * n, pc[a]))
            for j in range(n):
                gb = _lagrange(self.gauss, n, j, pc[b])
                for k in range(n):
                    gc = _lagrange(self.gauss, n, k, pc[cc])
                    total += sign * self._face_dof(c, lf, a, j, k) * la * gb * gc
        base = self.face_block + self.entity_id(c) * 3 * (n - 1) * n * n + a * (n - 1) * n * n
        for i in range(1, n):
            la = (_lagrange_derivative(self.lobatto, n + 1, i, pc[a]) if derivative
                  else _lagrange(self.lobatto, n + 1, i, pc[a]))
            for j in range(n):
                gb = _lagrange(self.gauss, n, j, pc[b])
                for k in range(n):
                    gc = _lagrange(self.gauss, n, k, pc[cc])
                    total += self.get(base + (i - 1) * n * n + j * n + k) * la * gb * gc
        return total

    @tack.func
    def parametric_value(self, c, pc):
        return tack.vec3(self._component(c, pc, 0, False), self._component(c, pc, 1, False),
                         self._component(c, pc, 2, False))

    @tack.func
    def parametric_divergence(self, c, pc):
        return self._component(c, pc, 0, True) + self._component(c, pc, 1, True) + self._component(c, pc, 2, True)


@tack.func
def _local_face(view, a, side):
    """The local face whose center has coordinate side (0 or 1) along axis a."""
    found = 0
    for k in range(6):
        center = tack.vec3(0.0, 0.0, 0.0)
        for i in range(4):
            center += view.parametric_point(view.face_corner(k, i))
        if (0.25 * center[a] - float(side)) * (0.25 * center[a] - float(side)) < 0.01:
            found = k
    return found


@tack.func
def _face_axis(view, c, lf, which):
    """The direction, in the cell's reference coordinates, from the local face's own corner
    0 to its own corner ``which`` (1: the face's own u axis, 3: its w axis)."""
    p0 = tack.vec3(0.0, 0.0, 0.0)
    pw = tack.vec3(0.0, 0.0, 0.0)
    for i in range(4):
        where = view.face_position(c, lf, i)
        point = view.parametric_point(view.face_corner(lf, i))
        if where == 0:
            p0 = point
        if where == which:
            pw = point
    return pw - p0


@tack.func
def _local_edge(view, a, cb, cc_):
    """The local edge along axis a at coordinates cb and cc_ (0 or 1) on the other two."""
    b = (a + 1) % 3
    cc = (a + 2) % 3
    found = 0
    for e in range(12):
        p0 = view.parametric_point(view.edge_corner(e, 0))
        d = view.parametric_point(view.edge_corner(e, 1)) - p0
        along = d[a] * d[a] > 0.25
        if along and (p0[b] - float(cb)) * (p0[b] - float(cb)) < 0.01 and \
                (p0[cc] - float(cc_)) * (p0[cc] - float(cc_)) < 0.01:
            found = e
    return found


@tack.func
def _node_basis(gauss, lobatto, n, i, j, k, x, y, z, dx, dy, dz):
    """G_i(x) L_j(y) L_k(z), or its derivative along the axes flagged by dx, dy, dz."""
    gx = _lagrange_derivative(gauss, n, i, x) if dx else _lagrange(gauss, n, i, x)
    ly = _lagrange_derivative(lobatto, n + 1, j, y) if dy else _lagrange(lobatto, n + 1, j, y)
    lz = _lagrange_derivative(lobatto, n + 1, k, z) if dz else _lagrange(lobatto, n + 1, k, z)
    return gx * ly * lz


class _HCurlHexOrderField:
    """HCurl of order P on a hexahedron (Nedelec, first kind): component a is a tensor
    product of P Gauss nodes along axis a and P + 1 Gauss-Lobatto nodes across it. A node
    on an edge holds the tangential component along the edge's own direction (low point id
    to high); a node inside a face, one of the two tangential components in the face's own
    frame; the others are the cell's own."""

    @tack.func
    def _dof(self, c, a, i, j, k):
        """The reference component a carried by node (i, j, k): i along a, j and k across."""
        n = self.P
        b = (a + 1) % 3
        cc = (a + 2) % 3
        jb = j == 0 or j == n
        kb = k == 0 or k == n
        value = 0.0 * self.get(0)
        if jb and kb:
            e = _local_edge(self, a, j // n, k // n)
            d = self.parametric_point(self.edge_corner(e, 1)) - self.parametric_point(self.edge_corner(e, 0))
            forward = (d[a] > 0.0) == (self.edge_sign(c, e) > 0)
            m = i if forward else n - 1 - i
            value = (1.0 if forward else -1.0) * self.get(self.edge_id(c, e) * n + m)
        elif jb or kb:
            normal = b if jb else cc
            side = (j if jb else k) // n
            across = cc if jb else b
            t = k if jb else j
            lf = _local_face(self, normal, side)
            u = _face_axis(self, c, lf, 1)
            w = _face_axis(self, c, lf, 3)
            base = self.face_base + self.face_id(c, lf) * 2 * n * (n - 1)
            if u[a] * u[a] > 0.25:
                m = i if u[a] > 0.0 else n - 1 - i
                q = t if w[across] > 0.0 else n - t
                value = (1.0 if u[a] > 0.0 else -1.0) * self.get(base + m + n * (q - 1))
            else:
                m = i if w[a] > 0.0 else n - 1 - i
                q = t if u[across] > 0.0 else n - t
                value = (1.0 if w[a] > 0.0 else -1.0) * self.get(base + n * (n - 1) + m + n * (q - 1))
        else:
            index = (self.cell_base + self.entity_id(c) * 3 * n * (n - 1) * (n - 1) + a * n * (n - 1) * (n - 1)
                     + i * (n - 1) * (n - 1) + (j - 1) * (n - 1) + (k - 1))
            value = self.get(index)
        return value

    @tack.func
    def _component(self, c, pc, a, along):
        """Component a at pc, or its derivative along axis ``along`` (-1: the value)."""
        n = self.P
        b = (a + 1) % 3
        cc = (a + 2) % 3
        total = 0.0 * pc[0]
        for i in range(n):
            for j in range(n + 1):
                for k in range(n + 1):
                    total += self._dof(c, a, i, j, k) * _node_basis(
                        self.gauss, self.lobatto, n, i, j, k, pc[a], pc[b], pc[cc],
                        along == a, along == b, along == cc)
        return total

    @tack.func
    def parametric_value(self, c, pc):
        return tack.vec3(self._component(c, pc, 0, -1), self._component(c, pc, 1, -1),
                         self._component(c, pc, 2, -1))

    @tack.func
    def parametric_curl(self, c, pc):
        return tack.vec3(self._component(c, pc, 2, 1) - self._component(c, pc, 1, 2),
                         self._component(c, pc, 0, 2) - self._component(c, pc, 2, 0),
                         self._component(c, pc, 1, 0) - self._component(c, pc, 0, 1))


class _HDivTetField:
    """HDiv on a tetrahedron: face k's function is 2 (pc - v), v the vertex opposite the face,
    with unit outward flux through face k and none through the others (Raviart-Thomas)."""

    @tack.func
    def parametric_value(self, c, pc):
        corners = tack.vec3(0.0, 0.0, 0.0)
        for j in range(4):
            corners += self.parametric_point(j)
        v = tack.vec3(0.0, 0.0, 0.0)
        for k in range(4):
            opposite = corners
            for i in range(3):
                opposite -= self.parametric_point(self.face_corner(k, i))
            flux = self.get(self.face_id(c, k)) * (1.0 if self.face_side(c, k) == 0 else -1.0)
            v += (pc - opposite) * (2.0 * flux)
        return v

    @tack.func
    def parametric_divergence(self, c, pc):
        """Each face's function has divergence 6, the reference volume's reciprocal."""
        d = 0.0 * self.get(0)
        for k in range(4):
            d += self.get(self.face_id(c, k)) * (1.0 if self.face_side(c, k) == 0 else -1.0)
        return 6.0 * d


@tack.func
def _barycentric(p, pc):
    """The barycentric coordinate of reference vertex p at pc."""
    return 1.0 - pc.sum() if p.sum() < 0.5 else p.dot(pc)


@tack.func
def _barycentric_gradient(p):
    return tack.vec3(-1.0, -1.0, -1.0) if p.sum() < 0.5 else p


class _HCurlTetField:
    """HCurl on a tetrahedron: edge a -> b's function is l_a grad l_b - l_b grad l_a, in
    barycentric coordinates, with unit tangential integral along that edge (Nedelec)."""

    @tack.func
    def parametric_value(self, c, pc):
        v = tack.vec3(0.0, 0.0, 0.0)
        for e in range(6):
            pa = self.parametric_point(self.edge_corner(e, 0))
            pb = self.parametric_point(self.edge_corner(e, 1))
            la = _barycentric(pa, pc)
            lb = _barycentric(pb, pc)
            ga = _barycentric_gradient(pa)
            gb = _barycentric_gradient(pb)
            integral = self.get(self.edge_id(c, e)) * float(self.edge_sign(c, e))
            v += (la * gb - lb * ga) * integral
        return v

    @tack.func
    def parametric_curl(self, c, pc):
        v = tack.vec3(0.0, 0.0, 0.0)
        for e in range(6):
            pa = self.parametric_point(self.edge_corner(e, 0))
            pb = self.parametric_point(self.edge_corner(e, 1))
            la = _barycentric(pa, pc)
            lb = _barycentric(pb, pc)
            ga = _barycentric_gradient(pa)
            gb = _barycentric_gradient(pb)
            integral = self.get(self.edge_id(c, e)) * float(self.edge_sign(c, e))
            v += 2.0 * ga.cross(gb) * integral
        return v


class _HCurlField:
    """An HCurl field's value in the reference cell: each edge's tangential integral times
    its reference function, which has unit integral along that edge and none along the others."""

    @tack.func
    def parametric_value(self, c, pc):
        v = tack.vec3(0.0, 0.0, 0.0)
        for e in range(12):
            p0 = self.parametric_point(self.edge_corner(e, 0))
            d = self.parametric_point(self.edge_corner(e, 1)) - p0
            w = 1.0
            for a in range(3):
                across = p0[a] * pc[a] + (1.0 - p0[a]) * (1.0 - pc[a])
                w = w * (1.0 if d[a] * d[a] > 0.25 else across)
            integral = self.get(self.edge_id(c, e)) * float(self.edge_sign(c, e))
            v += d * (w * integral)
        return v

    @tack.func
    def parametric_curl(self, c, pc):
        """The curl in the reference cell. An edge's function is its direction d times a
        weight w across the other two axes, so its curl is grad(w) x d."""
        v = tack.vec3(0.0, 0.0, 0.0)
        for e in range(12):
            p0 = self.parametric_point(self.edge_corner(e, 0))
            d = self.parametric_point(self.edge_corner(e, 1)) - p0
            # w's factor on each axis across the edge, and 1 along it.
            f = tack.vec3(1.0 if d[0] * d[0] > 0.25 else p0[0] * pc[0] + (1.0 - p0[0]) * (1.0 - pc[0]),
                          1.0 if d[1] * d[1] > 0.25 else p0[1] * pc[1] + (1.0 - p0[1]) * (1.0 - pc[1]),
                          1.0 if d[2] * d[2] > 0.25 else p0[2] * pc[2] + (1.0 - p0[2]) * (1.0 - pc[2]))
            s = 2.0 * p0 - 1.0
            g = tack.vec3(0.0 if d[0] * d[0] > 0.25 else s[0] * f[1] * f[2],
                          0.0 if d[1] * d[1] > 0.25 else f[0] * s[1] * f[2],
                          0.0 if d[2] * d[2] > 0.25 else f[0] * f[1] * s[2])
            integral = self.get(self.edge_id(c, e)) * float(self.edge_sign(c, e))
            v += g.cross(d) * integral
        return v


class HDiv(Space):
    """H(div), Raviart-Thomas. At ``order`` 1 (on hexahedra, voxels and tetrahedra): one
    value per face, its integrated flux along the face's stored normal (out of side 0).
    Above (on hexahedra and voxels), with p the order: p squared values per face, the normal
    flux density at the Gauss points of the face's own frame, then 3 (p - 1) p squared per
    cell, the reference components at their nodes (``gauss_nodes``, ``lobatto_nodes``).
    Read with ``hdiv_value``."""

    on = "cells"
    mixin = _HDivField

    def mixin_for(self, key, shape=None):
        if shape is not None and _is_tetrahedral(shape):
            if self.order > 1:
                raise NotImplementedError("HDiv above order 1 is on hexahedra and voxels")
            return _HDivTetField
        return _HDivField if self.order == 1 else _HDivHexOrderField

    @classmethod
    def _params(cls, order=1):
        if int(order) < 1:
            raise ValueError(f"HDiv's order is 1 or more, not {order}")
        return {"order": int(order)}

    @property
    def size(self):
        p = self.order
        faces = self.topology.faces().num_faces
        return faces if p == 1 else faces * p * p + self.topology.num_cells * 3 * (p - 1) * p * p

    def extra_mixins(self, group):
        from tack.dataset import views

        _check_shape(self, group)
        return (views._FaceIncidence,)

    def attributes(self, group=None, dtype=None):
        if group is None:
            return {}
        faces = self.topology.faces()
        attributes = {"side_face": faces.side_face, "side_slot": faces.side_slot,
                      "side_orientation": faces.side_orientation,
                      "face_start": faces.group_starts[group.root]}
        p = self.order
        if p > 1:
            real = np.float64 if dtype is tack.f64 else np.float32
            attributes.update(P=p, face_block=faces.num_faces * p * p,
                              gauss=tack.array(gauss_nodes(p).astype(real)),
                              lobatto=tack.array(lobatto_nodes(p + 1).astype(real)))
        return attributes


class HCurl(Space):
    """H(curl), Nedelec of the first kind. At ``order`` 1 (on hexahedra, voxels and
    tetrahedra): one value per edge, its tangential integral from the edge's low point id to
    its high one. Above (on hexahedra and voxels), with p the order: p values per edge, the
    tangential component along the edge at its Gauss points; 2 p (p - 1) per face, the two
    tangential components of the face's own frame; then 3 p (p - 1) squared per cell, the
    reference components at their nodes. Read with ``hcurl_value``."""

    on = "cells"
    mixin = _HCurlField

    def mixin_for(self, key, shape=None):
        if shape is not None and _is_tetrahedral(shape):
            if self.order > 1:
                raise NotImplementedError("HCurl above order 1 is on hexahedra and voxels")
            return _HCurlTetField
        return _HCurlField if self.order == 1 else _HCurlHexOrderField

    @classmethod
    def _params(cls, order=1):
        if int(order) < 1:
            raise ValueError(f"HCurl's order is 1 or more, not {order}")
        return {"order": int(order)}

    @property
    def size(self):
        p = self.order
        edges = self.topology.edges().num_edges
        if p == 1:
            return edges
        return (edges * p + self.topology.faces().num_faces * 2 * p * (p - 1)
                + self.topology.num_cells * 3 * p * (p - 1) * (p - 1))

    def extra_mixins(self, group):
        from tack.dataset import views

        _check_shape(self, group)
        if self.order == 1:
            return (views._EdgeIncidence,)
        return (views._EdgeIncidence, views._FaceIncidence)

    def attributes(self, group=None, dtype=None):
        if group is None:
            return {}
        edges = self.topology.edges()
        attributes = {"side_edge": edges.side_edge, "side_sign": edges.side_sign,
                      "edge_start": edges.group_starts[group.root]}
        p = self.order
        if p > 1:
            faces = self.topology.faces()
            real = np.float64 if dtype is tack.f64 else np.float32
            attributes.update(side_face=faces.side_face, side_slot=faces.side_slot,
                              side_orientation=faces.side_orientation,
                              face_start=faces.group_starts[group.root], P=p,
                              face_base=edges.num_edges * p,
                              cell_base=edges.num_edges * p + faces.num_faces * 2 * p * (p - 1),
                              gauss=tack.array(gauss_nodes(p).astype(real)),
                              lobatto=tack.array(lobatto_nodes(p + 1).astype(real)))
        return attributes


@tack.func
def hdiv_div(cells, u, c, pc):
    """An HDiv field's divergence in the world: div_ref / det J."""
    return u.parametric_divergence(c, pc) / cells.geometry_jacobian(c, pc).determinant()


@tack.func
def hdiv_value(cells, u, c, pc):
    """An HDiv field's value in the world: J v / det J, the map that keeps fluxes."""
    J = cells.geometry_jacobian(c, pc)
    return (J @ u.parametric_value(c, pc)) / J.determinant()


@tack.func
def hcurl_curl(cells, u, c, pc):
    """An HCurl field's curl in the world: J curl_ref / det J."""
    J = cells.geometry_jacobian(c, pc)
    return (J @ u.parametric_curl(c, pc)) / J.determinant()


@tack.func
def hcurl_value(cells, u, c, pc):
    """An HCurl field's value in the world: J^-T v, the map that keeps tangential integrals."""
    return cells.geometry_jacobian(c, pc).inverse().transpose() @ u.parametric_value(c, pc)

