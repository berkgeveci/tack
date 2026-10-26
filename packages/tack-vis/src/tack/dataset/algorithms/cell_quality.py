"""``cell_quality``: how well shaped each cell is, by measures of its corners' Jacobians
and its edges, for every shape -- order-2 geometry, wedges, pyramids, polygons and
polyhedra included."""

import itertools
import math

import numpy as np

import tack
from tack.dataset import shapes
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values
from tack.runtime.dispatch import get_backend, on_inputs_device

__all__ = ['cell_quality']

#: The measures, in the order they are computed.
MEASURES = ("scaled_jacobian", "jacobian_ratio", "condition", "edge_ratio", "min_edge",
            "max_edge")

_S2, _S3 = math.sqrt(2.0), math.sqrt(3.0)

# Each shape with a reference element: its corners' parametric coordinates (VTK's
# order), its edges, and its ideal cell -- equilateral, edges all 1 where the shape
# allows -- against which a measure is 1.
_REFERENCE = {
    shapes.Triangle: ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1), (1, 2), (2, 0)],
                      [(0, 0, 0), (1, 0, 0), (0.5, _S3 / 2, 0)]),
    shapes.Quad: ([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                  [(0, 1), (1, 2), (2, 3), (3, 0)],
                  [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
    shapes.Pixel: ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
                   [(0, 1), (1, 3), (3, 2), (2, 0)],
                   [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]),
    shapes.Tetra: ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                   [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
                   [(0, 0, 0), (1, 0, 0), (0.5, _S3 / 2, 0),
                    (0.5, _S3 / 6, math.sqrt(2.0 / 3.0))]),
    shapes.Hexahedron: ([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1),
                         (1, 1, 1), (0, 1, 1)],
                        [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                         (0, 4), (1, 5), (2, 6), (3, 7)], None),
    shapes.Voxel: ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1),
                    (0, 1, 1), (1, 1, 1)],
                   [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)], None),
    shapes.Wedge: ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)],
                   [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
                   [(0, 0, 0), (1, 0, 0), (0.5, _S3 / 2, 0), (0, 0, 1), (1, 0, 1),
                    (0.5, _S3 / 2, 1)]),
}

# The pyramid: no Jacobian at its apex, so its four tetrahedra over both base
# diagonals, as Verdict measures it, against the ideal pyramid's (equilateral
# faces on a unit square).
_PYRAMID_TETS = ((0, 1, 2, 4), (0, 2, 3, 4), (0, 1, 3, 4), (1, 2, 3, 4))
_PYRAMID_IDEAL = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1 / _S2)])
_PYRAMID_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4))

# The regular tetrahedron's corner matrix (edges from its first point): what a
# polyhedron's fan tetrahedra are measured against.
_REGULAR = np.array([(1, 0, 0), (0.5, _S3 / 2, 0), (0.5, _S3 / 6, math.sqrt(2.0 / 3.0))]).T
_EQUILATERAL = np.array([(1, 0.5), (0, _S3 / 2)])     # a triangle's, in its plane


def _samples(shape):
    """A shape's sample points -- its corners, and a hexahedron's center, as Verdict
    samples them -- each with its parametric point, the parametric directions of
    its edges there (columns), the ideal cell's normalization of the scaled
    Jacobian, the inverse of the ideal corner matrix (in the cell's plane for 2D
    shapes), and whether it is a corner."""
    pcs, edges, ideal = _REFERENCE[shape]
    pcs = np.asarray(pcs, np.float64)
    dim = shape.DIMENSION
    ideal = pcs if ideal is None else np.asarray(ideal, np.float64)
    out = []
    for k in range(len(pcs)):
        neighbours = [b if a == k else a for a, b in edges if k in (a, b)]
        for order in itertools.permutations(neighbours, dim):
            w = (ideal[list(order)] - ideal[k]).T
            turn = np.cross(w[:, 0], w[:, 1])[2] if dim == 2 else np.linalg.det(w)
            if turn > 1e-9:
                break
        directions = (pcs[list(order)] - pcs[k]).T
        out.append((pcs[k], directions, w, True))
    if shape in (shapes.Hexahedron, shapes.Voxel):
        # The principal axes at the center: Verdict's ninth sample of a hexahedron.
        out.append((np.full(3, 0.5), np.eye(3), np.eye(3), False))
    rows = []
    for pc, directions, w, corner in out:
        d = np.zeros((3, 3))
        d[:, :dim] = directions
        if dim == 2:
            plane = np.array([w[:2, 0], w[:2, 1]]).T
            scale = (plane[0, 0] * plane[1, 1] - plane[0, 1] * plane[1, 0]) / (
                np.linalg.norm(plane[:, 0]) * np.linalg.norm(plane[:, 1]))
            inverse = np.zeros((3, 3))
            inverse[:2, :2] = np.linalg.inv(plane)
        else:
            scale = np.linalg.det(w) / np.prod(np.linalg.norm(w, axis=0))
            inverse = np.linalg.inv(w)
        rows.append((pc, d, 1.0 / scale, inverse, corner))
    return rows, edges


_TABLES = {}


def _tables(shape, dtype):
    """A shape's sample tables as device arrays (``_samples``), made once per shape
    and type on the default device."""
    key = (shape, dtype, get_backend())
    if key not in _TABLES:
        rows, edges = _samples(shape)
        f = dtype.numpy_dtype
        _TABLES[key] = (
            tack.array(np.concatenate([r[0] for r in rows]).astype(f)),
            tack.array(np.concatenate([r[1].T.reshape(-1) for r in rows]).astype(f)),
            tack.array(np.array([r[2] for r in rows]).astype(f)),
            tack.array(np.concatenate([r[3].reshape(-1) for r in rows]).astype(f)),
            tack.array(np.array([1 if r[4] else 0 for r in rows], np.int32)),
            tack.array(np.array(edges, np.int32).reshape(-1)))
    return _TABLES[key]


@tack.func
def _column(directions, k, j, jacobian):
    # The world direction of a sample's j-th edge: the Jacobian there times its
    # parametric direction.
    base = 9 * k + 3 * j
    return jacobian @ tack.vec3(directions[base], directions[base + 1], directions[base + 2])


@tack.func
def _matrix(m, k):
    b = 9 * k
    return tack.mat3([m[b], m[b + 1], m[b + 2]], [m[b + 3], m[b + 4], m[b + 5]],
                     [m[b + 6], m[b + 7], m[b + 8]])


@tack.kernel
def _reference_quality(cells, dim, pcs, directions, norms, inverses, corner, big, sj, ratio,
                       cond):
    for c in cells:
        e = cells.entity_id(c)
        worst = big
        least = big
        most = -big
        conditioned = 1.0
        normal = tack.vec3(0.0, 0.0, 0.0)
        if dim == 2:
            # A surface cell is measured about its normal at its center, Verdict's.
            middle = cells.geometry_jacobian(c, cells.parametric_center())
            normal = middle @ tack.vec3(1.0, 0.0, 0.0)
            normal = normal.cross(middle @ tack.vec3(0.0, 1.0, 0.0))
            normal = normal / normal.norm()
        for k in range(norms.shape[0]):
            j = cells.geometry_jacobian(c, tack.vec3(pcs[3 * k], pcs[3 * k + 1], pcs[3 * k + 2]))
            a0 = _column(directions, k, 0, j)
            a1 = _column(directions, k, 1, j)
            det = 0.0
            lengths = a0.norm() * a1.norm()
            a2 = a0
            if dim == 3:
                a2 = _column(directions, k, 2, j)
                det = a0.dot(a1.cross(a2))
                lengths = lengths * a2.norm()
            else:
                det = a0.cross(a1).dot(normal)
            scaled = det / lengths * norms[k] if lengths > 0.0 else -big
            worst = min(worst, scaled)
            if corner[k] == 1:
                least = min(least, det)
                most = max(most, det)
            # The condition number of T = A W^-1, A the corner's edges and W the
            # ideal cell's: |T| |T^-1| / dim (Frobenius norms), infinite where T
            # is singular or inverted.
            w = _matrix(inverses, k)
            if dim == 3:
                t = tack.mat3([a0[0], a1[0], a2[0]], [a0[1], a1[1], a2[1]],
                              [a0[2], a1[2], a2[2]]) @ w
                d = t.determinant()
                if d > 0.0:
                    inverse = t.inverse()
                    frob = 0.0
                    frob_inv = 0.0
                    for r in range(3):
                        for s in range(3):
                            frob += t[r, s] * t[r, s]
                            frob_inv += inverse[r, s] * inverse[r, s]
                    conditioned = max(conditioned, tack.sqrt(frob * frob_inv) / 3.0)
                else:
                    conditioned = big
            else:
                # T's columns as vectors (a warped quad's corner is not in the
                # cell's plane): |T|^2 over twice its area about the normal,
                # Verdict's.
                t0 = a0 * w[0, 0] + a1 * w[1, 0]
                t1 = a0 * w[0, 1] + a1 * w[1, 1]
                d = t0.cross(t1).dot(normal)
                if d > 0.0:
                    conditioned = max(conditioned, (t0.dot(t0) + t1.dot(t1)) / (2.0 * d))
                else:
                    conditioned = big
        sj[e] = worst
        cond[e] = conditioned
        ratio[e] = least / most if most > 0.0 else -big


@tack.func
def _tet_scaled(p0, p1, p2, p3):
    # Verdict's tetrahedron scaled Jacobian: the volume's determinant over the
    # largest product of a corner's three edges, times sqrt(2) (1 when regular).
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2
    e3 = p3 - p0
    e4 = p3 - p1
    e5 = p3 - p2
    det = e3.dot(e2.cross(e0))
    l0 = e0.norm()
    l1 = e1.norm()
    l2 = e2.norm()
    l3 = e3.norm()
    l4 = e4.norm()
    l5 = e5.norm()
    most = max(l0 * l2 * l3, l0 * l1 * l4, l1 * l2 * l5, l3 * l4 * l5)
    return det * 1.4142135623730951 / most if most > 0.0 else 0.0, det


@tack.func
def _tet_condition(p0, p1, p2, p3, w, big):
    # |T| |T^-1| / 3 for T = A W^-1, A the edges from p0 and W the reference's.
    a0 = p1 - p0
    a1 = p2 - p0
    a2 = p3 - p0
    t = tack.mat3([a0[0], a1[0], a2[0]], [a0[1], a1[1], a2[1]], [a0[2], a1[2], a2[2]]) @ w
    d = t.determinant()
    out = big
    if d > 0.0:
        inverse = t.inverse()
        frob = 0.0
        frob_inv = 0.0
        for r in range(3):
            for s in range(3):
                frob += t[r, s] * t[r, s]
                frob_inv += inverse[r, s] * inverse[r, s]
        out = tack.sqrt(frob * frob_inv) / 3.0
    return out


@tack.kernel
def _pyramid_quality(cells, norms, inverses, big, sj, ratio, cond):
    for c in cells:
        e = cells.entity_id(c)
        worst = big
        least = big
        most = -big
        conditioned = 1.0
        for k in range(4):
            a = 0
            b = 1
            d_ = 2
            if k == 1:
                b = 2
                d_ = 3
            elif k == 2:
                d_ = 3
            elif k == 3:
                a = 1
                b = 2
                d_ = 3
            p0 = cells.corner_position(c, a)
            p1 = cells.corner_position(c, b)
            p2 = cells.corner_position(c, d_)
            p3 = cells.corner_position(c, 4)
            if k == 2:
                # (0, 1, 3, apex) turns the other way: measured as (0, 3, 1, apex).
                p1 = cells.corner_position(c, 3)
                p2 = cells.corner_position(c, 1)
            scaled, det = _tet_scaled(p0, p1, p2, p3)
            worst = min(worst, scaled * norms[k])
            least = min(least, det)
            most = max(most, det)
            conditioned = max(conditioned, _tet_condition(p0, p1, p2, p3, _matrix(inverses, k),
                                                          big))
        sj[e] = worst
        cond[e] = conditioned
        ratio[e] = least / most if most > 0.0 else -big


@tack.kernel
def _polyhedron_quality(cells, w, big, sj, ratio, cond):
    for c in cells:
        # The fan tetrahedra polyhedra are evaluated in: each face fanned from its
        # points' mean, the faces from the cell's points' mean.
        e = cells.entity_id(c)
        center = tack.vec3(0.0, 0.0, 0.0)
        for j in range(cells.num_points(c)):
            center += cells.corner_position(c, j)
        center = center / cells.num_points(c)
        worst = big
        least = big
        most = -big
        conditioned = 1.0
        for k in range(cells.num_faces(c)):
            size = cells.side_num_points(c, k)
            apex = tack.vec3(0.0, 0.0, 0.0)
            for j in range(size):
                apex += cells.side_position(c, k, j)
            apex = apex / size
            for j in range(size):
                a = cells.side_position(c, k, j)
                b = cells.side_position(c, k, j + 1 if j + 1 < size else 0)
                # Faces are wound out of the cell: (center, a, b, apex) turns positively.
                scaled, det = _tet_scaled(center, a, b, apex)
                worst = min(worst, scaled)
                least = min(least, det)
                most = max(most, det)
                conditioned = max(conditioned, _tet_condition(center, a, b, apex,
                                                              _matrix(w, 0), big))
        sj[e] = worst
        cond[e] = conditioned
        ratio[e] = least / most if most > 0.0 else -big


@tack.kernel
def _polygon_quality(cells, big, sj, ratio, cond):
    for c in cells:
        # The fan triangles from the polygon's points' mean, about Newell's normal.
        e = cells.entity_id(c)
        origin = cells.side_position(c, 0, 0)
        normal = tack.vec3(0.0, 0.0, 0.0)
        center = tack.vec3(0.0, 0.0, 0.0)
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - origin
            normal += a.cross(cells.side_position(c, k, 1) - origin)
            center += a
        normal = normal / normal.norm()
        center = origin + center / cells.num_faces(c)
        worst = big
        least = big
        most = -big
        conditioned = 1.0
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - center
            b = cells.side_position(c, k, 1) - center
            ab = b - a
            det = a.cross(b).dot(normal)
            most_pair = max(max(a.norm() * b.norm(), b.norm() * ab.norm()), ab.norm() * a.norm())
            # Verdict's triangle: twice the area over the largest product of two
            # edges, times 2/sqrt(3) (1 when equilateral).
            scaled = det / most_pair * 1.1547005383792515 if most_pair > 0.0 else -big
            worst = min(worst, scaled)
            least = min(least, det)
            most = max(most, det)
            # The equilateral triangle's condition number: (|a|^2 + |b|^2 + |b-a|^2)
            # over 2 sqrt(3) times the area -- twice it is det.
            if det > 0.0:
                conditioned = max(conditioned, (a.dot(a) + b.dot(b) + ab.dot(ab))
                                  / (1.7320508075688772 * det))
            else:
                conditioned = big
        sj[e] = worst
        cond[e] = conditioned
        ratio[e] = least / most if most > 0.0 else -big


@tack.kernel
def _edges_by_table(cells, edges, lo, hi):
    for c in cells:
        e = cells.entity_id(c)
        shortest = 0.0
        longest = 0.0
        for k in range(edges.shape[0] // 2):
            length = (cells.corner_position(c, edges[2 * k + 1])
                      - cells.corner_position(c, edges[2 * k])).norm()
            if k == 0:
                shortest = length
                longest = length
            shortest = min(shortest, length)
            longest = max(longest, length)
        lo[e] = shortest
        hi[e] = longest


@tack.kernel
def _edges_by_sides(cells, lo, hi):
    for c in cells:
        e = cells.entity_id(c)
        shortest = 0.0
        longest = 0.0
        first = 1
        for k in range(cells.num_faces(c)):
            size = cells.side_num_points(c, k)
            for j in range(size):
                length = (cells.side_position(c, k, j + 1 if j + 1 < size else 0)
                          - cells.side_position(c, k, j)).norm()
                if first == 1:
                    shortest = length
                    longest = length
                    first = 0
                shortest = min(shortest, length)
                longest = max(longest, length)
        lo[e] = shortest
        hi[e] = longest


@tack.kernel
def _edges_of_lines(cells, lo, hi):
    for c in cells:
        e = cells.entity_id(c)
        shortest = 0.0
        longest = 0.0
        for j in range(cells.line_num_points(c) - 1):
            length = (cells.point_position(cells.line_point_id(c, j + 1))
                      - cells.point_position(cells.line_point_id(c, j))).norm()
            if j == 0:
                shortest = length
                longest = length
            shortest = min(shortest, length)
            longest = max(longest, length)
        lo[e] = shortest
        hi[e] = longest


@tack.kernel
def _no_jacobian(cells, value, sj, ratio, cond):
    for c in cells:
        e = cells.entity_id(c)
        sj[e] = value
        ratio[e] = value
        cond[e] = value


@tack.kernel
def _ratio(lo, hi, big, nan, out):
    for e in range(out.shape[0]):
        r = big
        if lo[e] > 0.0:
            r = hi[e] / lo[e]
        elif hi[e] == 0.0:
            r = nan                    # no edges of any length: a vertex
        out[e] = r


def _pyramid_tables(dtype):
    f = dtype.numpy_dtype
    norms, inverses = [], []
    for tet in _PYRAMID_TETS:
        p = _PYRAMID_IDEAL[list(tet)]
        if tet == (0, 1, 3, 4):
            p = _PYRAMID_IDEAL[[0, 3, 1, 4]]
        a = (p[1:] - p[0]).T
        lengths = [np.linalg.norm(p[i] - p[j]) for i, j in ((0, 1), (1, 2), (2, 0), (0, 3),
                                                             (1, 3), (2, 3))]
        most = max(lengths[0] * lengths[2] * lengths[3], lengths[0] * lengths[1] * lengths[4],
                   lengths[1] * lengths[2] * lengths[5], lengths[3] * lengths[4] * lengths[5])
        norms.append(most / (np.linalg.det(a) * _S2))
        inverses.append(np.linalg.inv(a).reshape(-1))
    return tack.array(np.array(norms).astype(f)), tack.array(np.concatenate(inverses).astype(f))


@on_inputs_device
def cell_quality(data, measures=None):
    """How well shaped each cell is: a dict of cell fields, one per measure named in
    ``measures`` (all of ``MEASURES`` by default). Each is 1 on the ideal cell of its
    shape (equilateral; a cube, a regular tetrahedron, an equilateral wedge and
    pyramid), as Verdict (VTK's ``vtkMeshQuality``) defines it where it does:

    - ``scaled_jacobian``: the least, over the cell's corners (and a hexahedron's
      center), of the determinant of its edges there, each scaled to unit length,
      relative to the ideal cell's: 1 ideal, 0 degenerate, negative inverted.
    - ``jacobian_ratio``: the least corner determinant over the greatest; 1 for a
      cell whose Jacobian is constant (every simplex), negative where one corner
      is inverted.
    - ``condition``: the greatest, over the corners, condition number of the
      corner's edges against the ideal cell's (``|T| |T^-1| / d``, ``T = A W^-1``,
      Frobenius norms): 1 ideal, growing as the cell distorts, infinite (the
      largest float) where a corner is degenerate or inverted.
    - ``edge_ratio``: the longest edge over the shortest.
    - ``min_edge``, ``max_edge``: the shortest and longest edge, in the spatial
      unit.

    The corners' edges are the geometry's Jacobian there applied to the reference
    cell's edge directions, so an order-2 (curved) cell is measured along its
    curved edges at its corners. A pyramid, which has no Jacobian at its apex, is
    measured as Verdict does, by its four tetrahedra over both base diagonals.
    Polyhedra and polygons, which have no reference cell, are measured by the fan
    tetrahedra and triangles Tack evaluates them in, each against the regular
    tetrahedron or equilateral triangle: a valid star-shaped polyhedron has a
    positive scaled Jacobian, though an ideal-looking one (a cube as a polyhedron)
    scores below 1. Lines and vertices have no Jacobian measures (NaN); a line's
    edges are its segments. Hidden cells are measured like the rest.

    Matched to VTK 9.7.1's ``vtkMeshQuality`` to rounding for the scaled Jacobian
    of every linear shape, the condition number of triangles, quads, tetrahedra
    and hexahedra, and the edge ratio of triangles, quads, tetrahedra and wedges.
    VTK's wedge condition number and hexahedron edge ratio differ (see the
    tests)."""
    names = list(MEASURES if measures is None else
                 [measures] if isinstance(measures, str) else measures)
    unknown = [m for m in names if m not in MEASURES]
    if unknown:
        raise ValueError(f"no measure {', '.join(map(repr, unknown))}: the measures are "
                         f"{', '.join(MEASURES)}")
    n = data.num_cells
    dtype = data.dtype
    big = float(np.finfo(dtype.numpy_dtype).max)
    out = {m: tack.empty((n,), dtype) for m in ("scaled_jacobian", "jacobian_ratio",
                                                "condition", "min_edge", "max_edge")}
    if data.topology.dimension > 1 and not getattr(data.topology, "reference_cells", True):
        data.topology.faces()                      # polyhedra and polygons walk theirs
    jacobian = {"scaled_jacobian", "jacobian_ratio", "condition"} & set(names)
    for group in data.launch_groups("cells", []):
        if not group.count:
            continue
        view = data.domain_view("cells", group)
        shape = group.shape
        sj, ratio, cond = out["scaled_jacobian"], out["jacobian_ratio"], out["condition"]
        lo, hi = out["min_edge"], out["max_edge"]
        if shape in _REFERENCE:
            if jacobian:
                pcs, directions, norms, inverses, corner, _ = _tables(shape, dtype)
                _reference_quality(view, shape.DIMENSION, pcs, directions, norms, inverses,
                                   corner, big, sj, ratio, cond)
            _edges_by_table(view, _tables(shape, dtype)[5], lo, hi)
        elif shape is shapes.Pyramid:
            if jacobian:
                norms, inverses = _pyramid_tables(dtype)
                _pyramid_quality(view, norms, inverses, big, sj, ratio, cond)
            edges = tack.array(np.array(_PYRAMID_EDGES, np.int32).reshape(-1))
            _edges_by_table(view, edges, lo, hi)
        elif shape is shapes.Polyhedron:
            if jacobian:
                w = tack.array(np.linalg.inv(_REGULAR).astype(dtype.numpy_dtype).reshape(-1))
                _polyhedron_quality(view, w, big, sj, ratio, cond)
            _edges_by_sides(view, lo, hi)
        elif shape is shapes.Polygon:
            if jacobian:
                _polygon_quality(view, big, sj, ratio, cond)
            _edges_by_sides(view, lo, hi)
        else:
            # Lines, polylines, vertices: no Jacobian measures.
            _no_jacobian(view, float("nan"), sj, ratio, cond)
            if shape is shapes.PolyLine:
                _edges_of_lines(view, lo, hi)
            else:
                _edges_by_table(view, tack.array(np.array([0, 1] if shape is shapes.Line
                                                          else [0, 0], np.int32)), lo, hi)
    if "edge_ratio" in names:
        out["edge_ratio"] = tack.empty((n,), dtype)
        if n:
            _ratio(out["min_edge"], out["max_edge"], big, float("nan"), out["edge_ratio"])
    space = Values(data, "cells")
    units = {"min_edge": data.unit, "max_edge": data.unit}
    return {m: Field(space, out[m], units.get(m, "dimensionless")) for m in names}
