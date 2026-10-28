"""The template views kernels iterate: one entity kind, one shape, at a time.

A view's class combines mixins, made once per combination and reused, so
every compiled kernel is specialized to it (``docs/vis-guide/design/dataset-api.md``
§4):

- an entity *kind*: the cells of an unstructured or structured topology,
  derived faces, or derived edges. Each kind declares what a loop over it
  runs through (``for i in view``), ``point_id(i, j)`` and ``entity_id(i)``,
  the entity's global id, where its data lives;
- a *shape* from ``tack.dataset.shapes``: the cell's, or the face's
  (``Triangle``/``Quad``), or ``Line`` for edges, with their shape functions;
- optionally *geometry*, from the dataset's geometry field: ``corner_position(i, j)``,
  corner ``j``'s position, and ``position(i, pc)`` and
  ``geometry_jacobian(i, pc)`` through the shape's functions;
- optionally, for cells, *incidence*: ``face_id(c, f)``, ``face_side(c, f)``,
  ``edge_id(c, e)``, ``edge_sign(c, e)``;
- or, for a field view, a *space*: ``dof(i, j)`` and ``value(i, pc)`` for
  spaces with a basis, ``at(i)`` for values on entities.

Geometry and spaces read values through a *storage* mixin, ``get(k)``, so
a field's values may be a device array or an implicit array
(``tack.arrays``), which the view holds as a nested template. Values shared by points are read through
``point_value(i, j)``, which an *addressing* mixin defines: by the flat
point id, or -- for a structured grid's cells over a storage that has
``get_ijk`` -- by the corner's (i, j, k), with no division.

A field is a separate kernel argument: its view is built for the same
group as the domain's, so ``u.value(c, pc)`` in ``for c in cells`` reads
the cell ``c`` the loop is at.
"""

import tack
from tack.dataset.shapes import HOLD_TOLERANCE, NEWTON_DIVERGED, NEWTON_TOLERANCE

# ── Entity kinds ────────────────────────────────────────────────────


class _Cells:
    """The cells of one shape, by an explicit row of point ids each."""

    __tack_iterate__ = "num_cells"

    def __init__(self, connectivity, ids, num_cells):
        self.connectivity = connectivity      # (num_cells, NUM_POINTS) i32
        self.ids = ids                        # (num_cells,) i32: global cell ids
        self.num_cells = num_cells

    @tack.func
    def point_id(self, c, j):
        return self.connectivity[c, j]


    @tack.func
    def entity_id(self, c):
        return self.ids[c]

    @tack.func
    def index(self, c):
        return c


class _SelectedCells(_Cells):
    """Some of a group's cells: those a field's spaces put in one subgroup.

    The rows, ids and positions of every subgroup of a group lie in one set
    of arrays, sorted by subgroup; this one is ``num_cells`` of them from
    ``first``. ``index(c)`` is the cell's position in the whole group, so
    the group's incidence still applies.
    """

    def __init__(self, connectivity, ids, num_cells, positions, first):
        super().__init__(connectivity, ids, num_cells)
        self.positions = positions            # (all subgroups,) i32: index in the group
        self.first = first

    @tack.func
    def point_id(self, c, j):
        return self.connectivity[self.first + c, j]


    @tack.func
    def entity_id(self, c):
        return self.ids[self.first + c]

    @tack.func
    def index(self, c):
        return self.positions[self.first + c]


class _Points:
    """A dataset's points as entities, for a domain over a space's nodes: point
    ``i`` is itself, one point of its own (a ``Vertex``)."""

    __tack_iterate__ = "num_points"

    def __init__(self, num_points):
        self.num_points = num_points

    @tack.func
    def entity_id(self, i):
        return i

    @tack.func
    def index(self, i):
        return i

    @tack.func
    def point_id(self, i, j):
        return i


class _StructuredCells:
    """The cells of a structured grid of ``nx * ny * nz`` points, x fastest."""

    def __init__(self, nx, ny, nz, num_cells):
        self.nx = nx
        self.ny = ny
        self.cx = nx - 1
        self.cy = max(ny - 1, 1)
        self.cz = max(nz - 1, 1)
        self.num_cells = num_cells

    @tack.func
    def index(self, c):
        return self.entity_id(c)


class _StructuredLines(_StructuredCells):
    __tack_iterate__ = "num_cells"

    @tack.func
    def point_id(self, c, j):
        return c + j

    @tack.func
    def point_index(self, c, j):
        return [c + j, 0, 0]

    @tack.func
    def entity_id(self, c):
        return c


class _StructuredQuads(_StructuredCells):
    """Cells by ``c = (i, j)``."""

    __tack_iterate__ = ("cx", "cy")

    @tack.func
    def point_id(self, c, j):
        a, b = self._corner(j)
        return (c[0] + a) + (c[1] + b) * self.nx

    @tack.func
    def point_index(self, c, j):
        a, b = self._corner(j)
        return [c[0] + a, c[1] + b, 0]

    @tack.func
    def entity_id(self, c):
        return c[0] + self.cx * c[1]


class _StructuredHexahedra(_StructuredCells):
    """Cells by ``c = (i, j, k)``."""

    __tack_iterate__ = ("cx", "cy", "cz")

    @tack.func
    def point_id(self, c, j):
        a, b, d = self._corner(j)
        return (c[0] + a) + ((c[1] + b) + (c[2] + d) * self.ny) * self.nx

    @tack.func
    def point_index(self, c, j):
        a, b, d = self._corner(j)
        return [c[0] + a, c[1] + b, c[2] + d]

    @tack.func
    def entity_id(self, c):
        return c[0] + self.cx * (c[1] + self.cy * c[2])


class _Faces:
    """Derived faces of one shape: points in side 0's outward order, and two sides."""

    __tack_iterate__ = "num_faces"

    def __init__(self, rows, ids, sides, num_faces):
        self.rows = rows                      # (n,) 4-vectors of point ids
        self.ids = ids                        # (n,) global face ids
        self.sides = sides                    # (n,) [cell0, local0, cell1, local1]
        self.num_faces = num_faces

    @tack.func
    def point_id(self, f, j):
        return self.rows[f][j]

    @tack.func
    def entity_id(self, f):
        return self.ids[f]


    @tack.func
    def num_points(self, f):
        """The face's number of points: the shape's, a constant."""
        return self.NUM_POINTS

    @tack.func
    def num_sides(self, f):
        return 2 if self.sides[f][2] >= 0 else 1

    @tack.func
    def side_cell(self, f, k):
        """The cell on side ``k`` (0 or 1) of face ``f``; -1 if there is none."""
        return self.sides[f][2 * k]

    @tack.func
    def side_local(self, f, k):
        """Face ``f``'s local face index in the cell on side ``k``."""
        return self.sides[f][2 * k + 1]


class _CellsById:
    """Cells that are their ids, with no rows to gather: a polyhedral, polygonal or
    polyline topology's. A subgroup selects their ids (``_selected``, the view
    class that does)."""


class _PolyhedralCells(_CellsById):
    """The cells of a polyhedral (or polygonal) topology, each a list of facets -- faces
    of a polyhedron, edges of a polygon -- every count at run time.

    ``cell_faces[cell_offsets[c] + k]`` is cell ``c``'s facet ``k``, and
    ``cell_face_sides`` the same entry's side: 0 when the facet's stored
    direction (a face's winding, an edge's direction) points out of ``c``, 1
    when into it. ``point_offsets``/``point_ids`` are the cell's points
    (derived: its facets' points, sorted and unique). Every method reaches the
    arrays through ``cell(c)``, so a subgroup can renumber its cells.
    """

    __tack_iterate__ = "num_cells"

    def __init__(self, cell_offsets, cell_faces, cell_face_sides, face_offsets, face_points,
                 point_offsets, point_ids, num_cells):
        self.cell_offsets = cell_offsets
        self.cell_faces = cell_faces
        self.cell_face_sides = cell_face_sides
        self.face_offsets = face_offsets
        self.face_points = face_points
        self.point_offsets = point_offsets
        self.point_ids = point_ids
        self.num_cells = num_cells

    @tack.func
    def cell(self, c):
        """The topology's cell for loop index ``c``: itself, here."""
        return c

    @tack.func
    def entity_id(self, c):
        return self.cell(c)


    @tack.func
    def index(self, c):
        return self.cell(c)

    @tack.func
    def num_faces(self, c):
        g = self.cell(c)
        return self.cell_offsets[g + 1] - self.cell_offsets[g]

    @tack.func
    def entry(self, c, k):
        """The index of the cell's facet ``k`` among all cell -> facet entries: where
        per-(cell, facet) data -- a count, a side -- lives."""
        return self.cell_offsets[self.cell(c)] + k

    @tack.func
    def face_id(self, c, k):
        return self.cell_faces[self.cell_offsets[self.cell(c)] + k]

    @tack.func
    def face_side(self, c, k):
        """0 if the cell's facet ``k`` points out of it (its side 0), 1 if into it."""
        return tack.i32(self.cell_face_sides[self.cell_offsets[self.cell(c)] + k])

    @tack.func
    def side_point_id(self, c, k, j):
        """Point ``j`` of the cell's facet ``k``, going round it outward for this cell:
        the stored order on side 0, the reverse on side 1 -- which for an edge swaps
        its ends, and for a face starts the reversed cycle at its last point."""
        f = self.face_id(c, k)
        n = self.face_offsets[f + 1] - self.face_offsets[f]
        at = j if self.face_side(c, k) == 0 else n - 1 - j
        return self.face_points[self.face_offsets[f] + at]

    @tack.func
    def side_num_points(self, c, k):
        """The number of points of the cell's facet ``k``."""
        f = self.face_id(c, k)
        return self.face_offsets[f + 1] - self.face_offsets[f]

    @tack.func
    def num_points(self, c):
        g = self.cell(c)
        return self.point_offsets[g + 1] - self.point_offsets[g]

    @tack.func
    def point_id(self, c, j):
        return self.point_ids[self.point_offsets[self.cell(c)] + j]

    @tack.func
    def tet_face(self, c, t):
        """The facet ``k`` and its edge ``j`` (from point ``j`` to the next) of the
        cell's sub-tetrahedron ``t``: sub-tetrahedra are numbered facet by facet,
        edge by edge, as ``side_point_id`` walks them."""
        k = 0
        j = t
        while j >= self.side_num_points(c, k):
            j -= self.side_num_points(c, k)
            k += 1
        return k, j


class _PolyhedralWalk:
    """A polyhedral cell's facet points' positions, from the geometry: the walk's
    other half, in a mixin of its own because it needs the geometry, which only
    domain views have (a bare view, such as the one subgroups key by, does not)."""

    @tack.func
    def side_position(self, c, k, j):
        """The position of ``side_point_id(c, k, j)``."""
        return self.point_position(self.side_point_id(c, k, j))


@tack.func
def _columns(a, b, c):
    """The 3x3 matrix with columns ``a``, ``b``, ``c``."""
    return tack.mat3([a[0], b[0], c[0]], [a[1], b[1], c[1]], [a[2], b[2], c[2]])


@tack.func
def _closest_on_triangle(p, a, b, c):
    """The barycentric coordinates (of ``a``, ``b``, ``c``) of the triangle's point
    nearest ``p``, by the regions of Ericson's *Real-Time Collision Detection*
    (5.1.5)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    bp = p - b
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    cp = p - c
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    w = tack.vec3(0.0, 0.0, 0.0)
    if d1 <= 0.0 and d2 <= 0.0:
        w = [1.0, 0.0, 0.0]
    elif d3 >= 0.0 and d4 <= d3:
        w = [0.0, 1.0, 0.0]
    elif d6 >= 0.0 and d5 <= d6:
        w = [0.0, 0.0, 1.0]
    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        w = [1.0 - v, v, 0.0]
    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        w = [1.0 - v, 0.0, v]
    elif va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        w = [0.0, 1.0 - v, v]
    else:
        inverse = 1.0 / (va + vb + vc)
        w = [1.0 - (vb + vc) * inverse, vb * inverse, vc * inverse]
    return w


class _PolyhedralLocation:
    """Where a point is in a polyhedron, which has no parametric coordinates: in which
    of its *sub-tetrahedra*, and where in that one (``docs/vis-guide/design/polyhedra.md``,
    section 13).

    Each face is fanned from its *face point*, the mean of its points, and the
    fan's triangles from the *cell point*, the mean of the cell's distinct
    points: sub-tetrahedron ``t`` (``tet_face``) is (cell point, face point,
    ``a``, ``b``) for the edge ``ab`` of facet ``k``. A location is ``t`` and
    the coordinates ``pc`` of the point in it, the weights of the face point,
    ``a`` and ``b`` (the cell point's is what is left): a reference tetrahedron
    with the cell point at its origin. A field view evaluates there
    (``tet_value``), without positions; its gradient there is
    ``tet_jacobian(c, t)``'s inverse transpose times ``tet_parametric_gradient``.
    """

    @tack.func
    def cell_point(self, c):
        """The mean of the cell's distinct points."""
        n = self.num_points(c)
        x = self.point_position(self.point_id(c, 0))
        for j in range(1, n):
            x += self.point_position(self.point_id(c, j))
        return x / n

    @tack.func
    def face_point(self, c, k):
        """The mean of the points of the cell's facet ``k``."""
        n = self.side_num_points(c, k)
        x = self.side_position(c, k, 0)
        for j in range(1, n):
            x += self.side_position(c, k, j)
        return x / n

    @tack.func
    def tet_jacobian(self, c, t):
        """The 3x3 derivative of position by ``pc`` in sub-tetrahedron ``t``: columns
        from the cell point to the face point, ``a`` and ``b``."""
        k, j = self.tet_face(c, t)
        n = self.side_num_points(c, k)
        nxt = j + 1 if j + 1 < n else 0
        p = self.cell_point(c)
        return _columns(self.face_point(c, k) - p, self.side_position(c, k, j) - p,
                        self.side_position(c, k, nxt) - p)

    @tack.func
    def locate(self, c, x, slack):
        """``(t, pc, inside)``: the first sub-tetrahedron holding ``x`` and its
        coordinates there, ``inside`` 1; failing that, the one whose face
        triangle is nearest ``x``, if within ``slack``, and the coordinates of
        ``x`` extrapolated in it (so a linear field stays exact), ``inside`` 0;
        else ``t`` is -1. Callers rank a cell that holds the point ahead of one
        it is only near, as the shape path does.

        Inverted sub-tetrahedra (of a cell not star-shaped from its cell point)
        hold nothing; ``CellLocator`` refuses such cells."""
        t, pc, inside = self.locate_from(c, self.cell_point(c), x, slack)
        return t, pc, inside

    @tack.func
    def locate_from(self, c, p, x, slack):
        """``locate``, given the cell point ``p``: for a caller testing many points
        against one cell. The search stops at the first sub-tetrahedron holding
        ``x``; coordinates are ratios of triple products (Cramer's rule)."""
        d = x - p
        tet = -1
        pc = tack.vec3(0.0, 0.0, 0.0)
        near = -1
        near_pc = tack.vec3(0.0, 0.0, 0.0)
        best = slack * slack
        base = 0
        nf = self.num_faces(c)
        k = 0
        while k < nf and tet < 0:
            n = self.side_num_points(c, k)
            f = self.face_point(c, k) - p
            first = self.side_position(c, k, 0) - p
            a = first
            j = 0
            while j < n and tet < 0:
                b = first
                if j + 1 < n:
                    b = self.side_position(c, k, j + 1) - p
                ab = a.cross(b)
                det = f.dot(ab)
                if det > 0.0:
                    w = tack.vec3(d.dot(ab), f.dot(d.cross(b)), f.dot(a.cross(d))) / det
                    if min(w[0], w[1], w[2]) >= -HOLD_TOLERANCE and w[0] + w[1] + w[2] <= 1.0 + HOLD_TOLERANCE:
                        tet = base + j
                        pc = w
                    else:
                        # Outside: near if its face triangle is, and then
                        # extrapolated, as the shapes' coordinates are.
                        r = _closest_on_triangle(d, f, a, b)
                        e = r[0] * f + r[1] * a + r[2] * b - d
                        if e.dot(e) <= best:
                            best = e.dot(e)
                            near = base + j
                            near_pc = w
                a = b
                j += 1
            base += n
            k += 1
        inside = 1 if tet >= 0 else 0
        if tet < 0 and near >= 0:
            tet = near
            pc = near_pc
        return tet, pc, inside


class _PolygonLocation:
    """Where a point is in a polygon, a 2D cell with no parametric coordinates: in
    which of the triangles fanning it from its cell point (the mean of its
    points), and where. Triangle ``t = 2 * k`` is (cell point, ``a``, ``b``) for
    the polygon's edge ``k``: numbered so that ``tet_value`` reads it as the
    sub-tetrahedron (cell point, face point, ``a``, ``b``) of edge ``k`` with
    the face point's weight 0. ``pc`` is ``(0, w_a, w_b)``, the weights of the
    point's projection onto the triangle's plane, extrapolated outside it, so a
    linear field stays exact. As for a shape's surface cell, the point must be
    within ``slack`` of the triangle, and is held when its projection falls in
    it."""

    @tack.func
    def cell_point(self, c):
        """The mean of the cell's distinct points."""
        n = self.num_points(c)
        x = self.point_position(self.point_id(c, 0))
        for j in range(1, n):
            x += self.point_position(self.point_id(c, j))
        return x / n

    @tack.func
    def locate(self, c, x, slack):
        """``(t, pc, inside)``, as ``_PolyhedralLocation.locate``."""
        t, pc, inside = self.locate_from(c, self.cell_point(c), x, slack)
        return t, pc, inside

    @tack.func
    def locate_from(self, c, p, x, slack):
        """``locate``, given the cell point ``p``: a triangle holding the point's
        projection, within ``slack``, and failing that the nearest within it."""
        d = x - p
        tet = -1
        pc = tack.vec3(0.0, 0.0, 0.0)
        held = 0
        best = slack * slack
        for k in range(self.num_faces(c)):
            a = self.side_position(c, k, 0) - p
            b = self.side_position(c, k, 1) - p
            n = a.cross(b)
            nn = n.dot(n)
            if nn > 0.0:
                wa = d.cross(b).dot(n) / nn
                wb = a.cross(d).dot(n) / nn
                inside = 1 if wa >= -HOLD_TOLERANCE and wb >= -HOLD_TOLERANCE and wa + wb <= 1.0 + HOLD_TOLERANCE else 0
                r = _closest_on_triangle(d, a - a, a, b)
                e = r[1] * a + r[2] * b - d
                gap = e.dot(e)
                if gap <= slack * slack and (inside > held or (inside == held and gap < best)):
                    held = inside
                    best = gap
                    tet = 2 * k
                    pc = tack.vec3(0.0, wa, wb)
        return tet, pc, held


class _PolylineLocation:
    """Where a point is on a polyline: on which segment, ``t`` (from the chain's
    point ``t`` to ``t + 1``), and where, ``pc = (0, 1 - s, s)`` with ``s`` the
    projection's place along it, extrapolated past its ends so a linear field
    stays exact. The point must be within ``slack`` of the segment, and is held
    when its projection falls on it."""

    @tack.func
    def cell_point(self, c):
        """The mean of the cell's distinct points."""
        n = self.num_points(c)
        x = self.point_position(self.point_id(c, 0))
        for j in range(1, n):
            x += self.point_position(self.point_id(c, j))
        return x / n

    @tack.func
    def locate(self, c, x, slack):
        t, pc, inside = self.locate_from(c, self.cell_point(c), x, slack)
        return t, pc, inside

    @tack.func
    def locate_from(self, c, p, x, slack):
        tet = -1
        pc = tack.vec3(0.0, 0.0, 0.0)
        held = 0
        best = slack * slack
        for j in range(self.line_num_points(c) - 1):
            a = self.point_position(self.line_point_id(c, j))
            ab = self.point_position(self.line_point_id(c, j + 1)) - a
            length = ab.dot(ab)
            if length > 0.0:
                s = (x - a).dot(ab) / length
                e = a + ab * min(max(s, 0.0), 1.0) - x
                gap = e.dot(e)
                inside = 1 if s >= -HOLD_TOLERANCE and s <= 1.0 + HOLD_TOLERANCE else 0
                if gap <= slack * slack and (inside > held or (inside == held and gap < best)):
                    held = inside
                    best = gap
                    tet = j
                    pc = tack.vec3(0.0, 1.0 - s, s)
        return tet, pc, held


class _SelectedPolyhedra(_PolyhedralCells):
    """Some of a polyhedral topology's cells: a subgroup's, ``num_cells`` of the
    ``selected`` cell ids from ``first``. No rows to gather -- a polyhedral cell is
    its id -- so ``cell(c)`` just looks it up."""

    def __init__(self, cell_offsets, cell_faces, cell_face_sides, face_offsets, face_points,
                 point_offsets, point_ids, num_cells, selected, first):
        super().__init__(cell_offsets, cell_faces, cell_face_sides, face_offsets, face_points,
                         point_offsets, point_ids, num_cells)
        self.selected = selected
        self.first = first

    @tack.func
    def cell(self, c):
        return self.selected[self.first + c]


_PolyhedralCells._selected = _SelectedPolyhedra


class _PolylineCells(_CellsById):
    """The cells of a polyline topology: each a chain of points,
    ``line_points[line_offsets[c]:line_offsets[c + 1]]``, at least two. Its facets
    are its two ends -- facet 0 its first point, facet 1 its last -- whose ids
    and sides come with the face incidence (``_PolylineIncidence``) once the
    faces are derived. ``point_offsets``/``point_ids`` are the cell's points,
    sorted and unique, as a polyhedral cell's are; ``line_point`` walks the
    chain in order."""

    __tack_iterate__ = "num_cells"

    def __init__(self, line_offsets, line_points, point_offsets, point_ids, num_cells):
        self.line_offsets = line_offsets
        self.line_points = line_points
        self.point_offsets = point_offsets
        self.point_ids = point_ids
        self.num_cells = num_cells

    @tack.func
    def cell(self, c):
        """The topology's cell for loop index ``c``: itself, here."""
        return c

    @tack.func
    def entity_id(self, c):
        return self.cell(c)


    @tack.func
    def index(self, c):
        return self.cell(c)

    @tack.func
    def line_num_points(self, c):
        """The number of points in the cell's chain."""
        g = self.cell(c)
        return self.line_offsets[g + 1] - self.line_offsets[g]

    @tack.func
    def line_point_id(self, c, j):
        """Point ``j`` of the cell's chain, in order."""
        return self.line_points[self.line_offsets[self.cell(c)] + j]

    @tack.func
    def num_faces(self, c):
        """Two facets: the chain's first point and its last."""
        return 2

    @tack.func
    def side_num_points(self, c, k):
        return 1

    @tack.func
    def side_point_id(self, c, k, j):
        """Facet ``k``'s point: the chain's first (``k == 0``) or last."""
        g = self.cell(c)
        return self.line_points[self.line_offsets[g] if k == 0 else self.line_offsets[g + 1] - 1]

    @tack.func
    def num_points(self, c):
        g = self.cell(c)
        return self.point_offsets[g + 1] - self.point_offsets[g]

    @tack.func
    def point_id(self, c, j):
        return self.point_ids[self.point_offsets[self.cell(c)] + j]


class _SelectedPolylines(_PolylineCells):
    """Some of a polyline topology's cells, as ``_SelectedPolyhedra`` selects polyhedra."""

    def __init__(self, line_offsets, line_points, point_offsets, point_ids, num_cells, selected,
                 first):
        super().__init__(line_offsets, line_points, point_offsets, point_ids, num_cells)
        self.selected = selected
        self.first = first

    @tack.func
    def cell(self, c):
        return self.selected[self.first + c]


_PolylineCells._selected = _SelectedPolylines


class _PolylineIncidence:
    """A polyline's two facets as global faces: ``cell_faces``/``cell_face_sides``,
    two entries per cell (its first point, then its last)."""

    @tack.func
    def face_id(self, c, k):
        return self.cell_faces[2 * self.cell(c) + k]

    @tack.func
    def face_side(self, c, k):
        """0 if the facet's normal points out of this cell (its side 0), else 1."""
        return tack.i32(self.cell_face_sides[2 * self.cell(c) + k])


class _PolygonFaces:
    """A polyhedral topology's faces: each a polygon of ``num_points(f)`` points, in its
    stored order (wound out of its side 0), with the shape path's two-sided record."""

    __tack_iterate__ = "num_faces"

    def __init__(self, face_offsets, face_points, ids, sides, num_faces):
        self.face_offsets = face_offsets
        self.face_points = face_points
        self.ids = ids                        # (n,) global face ids
        self.sides = sides                    # (all faces,) [cell0, local0, cell1, local1]
        self.num_faces = num_faces

    @tack.func
    def entity_id(self, f):
        return self.ids[f]


    @tack.func
    def num_points(self, f):
        g = self.ids[f]
        return self.face_offsets[g + 1] - self.face_offsets[g]

    @tack.func
    def point_id(self, f, j):
        return self.face_points[self.face_offsets[self.ids[f]] + j]

    @tack.func
    def num_sides(self, f):
        return 2 if self.sides[self.ids[f]][2] >= 0 else 1

    @tack.func
    def side_cell(self, f, k):
        return self.sides[self.ids[f]][2 * k]

    @tack.func
    def side_local(self, f, k):
        return self.sides[self.ids[f]][2 * k + 1]


class _Edges:
    """Derived edges: their (low, high) point ids."""

    __tack_iterate__ = "num_edges"

    def __init__(self, rows, ids, num_edges):
        self.rows = rows                      # (n,) 2-vectors of point ids
        self.ids = ids
        self.num_edges = num_edges

    @tack.func
    def point_id(self, e, j):
        return self.rows[e][j]

    @tack.func
    def entity_id(self, e):
        return self.ids[e]


# ── Quadratic DOFs ──────────────────────────────────────────────────


class _QuadraticDofs:
    """Where an order-2 H1 space keeps cell ``c``'s value ``k``, in the shape's
    quadratic node order (``shapes.Shape.quadratic_node``): the corners' at
    their point ids; an edge's at ``edge_base`` plus its edge id; a quad face's
    at ``face_dofs[face]`` (``quadratic_face_dof``, from ``quadratic_faces``);
    the interior's at ``cell_dofs[cell]``. Edge and face ids come from the
    cell's incidence, so a view of these needs it."""

    @tack.func
    def dof_index(self, c, k):
        if k < self.NUM_POINTS:
            return self.point_id(c, k)
        e = k - self.NUM_POINTS
        if e < self.NUM_EDGES:
            return self.edge_base + self.edge_id(c, e)
        i = e - self.NUM_EDGES
        if i < self.QUADRATIC_FACES:
            return self.quadratic_face_dof(c, i)
        return self.cell_dofs[self.entity_id(c)]


class _QuadraticFaces:
    """A shape with quadratic face nodes (a hexahedron's quads, a wedge's): its
    face ``i``'s value at ``face_dofs[face]``, through the cell's face incidence."""

    @tack.func
    def quadratic_face_dof(self, c, i):
        return self.face_dofs[self.face_id(c, self.NUM_FACES - self.QUADRATIC_FACES + i)]


class _NoQuadraticFaces:
    """A shape without them (a tetrahedron, a triangle, a quad): no face incidence,
    so an order-2 space of these cells needs no faces derived (a surface with
    an edge of three faces has none)."""

    @tack.func
    def quadratic_face_dof(self, c, i):
        return self.cell_dofs[self.entity_id(c)]       # not reached: QUADRATIC_FACES is 0


def quadratic_faces(shape):
    """The mixin that finds ``shape``'s quadratic face values."""
    return _QuadraticFaces if shape.QUADRATIC_FACES else _NoQuadraticFaces


# ── Geometry ────────────────────────────────────────────────────────


#: Newton steps for a curved cell: more than the corners' 10, which a polynomial
#: map of higher degree can take from the center.
CURVED_NEWTON_ITERATIONS = tack.constant(20, tack.i32)


class _Geometry:
    """The geometry field, read through an entity: ``corner_position(i, j)``, which a storage
    mixin below defines, and what follows from it at parametric coordinates.

    At order 1 the geometry's values are the corners' positions, so the shape's
    own functions interpolate them. A higher-order geometry would supply its
    own basis here; nothing that calls ``position`` or ``geometry_jacobian``
    would change.
    """

    @tack.func
    def gather_points(self, i, pts):
        for j in range(self.NUM_POINTS):
            x = self.corner_position(i, j)
            pts[3 * j] = x[0]
            pts[3 * j + 1] = x[1]
            pts[3 * j + 2] = x[2]

    @tack.func
    def position(self, i, pc):
        """The world position at parametric coordinates ``pc``."""
        x = tack.vec3(0.0, 0.0, 0.0)
        for j in range(self.NUM_POINTS):
            x += self.shape_function(j, pc) * self.corner_position(i, j)
        return x

    #: Whether the geometry is curved (of order 2 or more), so that locating a point
    #: takes ``curved_to_parametric`` and a cell's bounds ``control_box``.
    CURVED = 0

    @tack.func
    def control_box(self, i):
        """``(lo, hi)``, a box holding the cell: here its corners'."""
        a = self.corner_position(i, 0)
        b = a
        for j in range(1, self.NUM_POINTS):
            p = self.corner_position(i, j)
            a = min(a, p)
            b = max(b, p)
        return a, b

    @tack.func
    def curved_to_parametric(self, i, x):
        """``world_to_parametric`` on the geometry itself: Newton's method from the
        parametric center on ``position`` and ``geometry_jacobian``, the shape's
        step and VTK's stopping rules, so a curved cell is inverted as it is,
        not as its corners make it. Returns ``(pc, converged)``."""
        pc = self.parametric_center()
        converged = 0
        for _ in range(CURVED_NEWTON_ITERATIONS):
            step = self._newton_step(self.geometry_jacobian(i, pc), x - self.position(i, pc))
            pc += step
            if not (abs(pc[0]) < NEWTON_DIVERGED and abs(pc[1]) < NEWTON_DIVERGED
                    and abs(pc[2]) < NEWTON_DIVERGED):
                break
            if (abs(step[0]) <= NEWTON_TOLERANCE and abs(step[1]) <= NEWTON_TOLERANCE
                    and abs(step[2]) <= NEWTON_TOLERANCE):
                converged = 1
                break
        return pc, converged

    @tack.func
    def geometry_jacobian(self, i, pc):
        """The 3x3 derivative of world position by parametric coordinates at ``pc``:
        column ``k`` is ``dx/d(pc[k])``, zero past the shape's dimension. Formed from
        the corners relative to the first (the shape's gradients sum to zero, so
        it is the same Jacobian): a corner's difference from its neighbour is exact
        in the geometry's precision, where sums of absolute positions lost a thin
        cell's short edges to rounding (in f32, the ONERA M6's boundary-layer
        wedges' gradients off by up to 93 times)."""
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        p0 = self.corner_position(i, 0)
        for j in range(1, self.NUM_POINTS):
            m += (self.corner_position(i, j) - p0).outer_product(self.shape_gradient(j, pc))
        return m


class _PointGeometry:
    """Positions of a cell or face with no reference element -- a polyhedron, a
    polygon: ``corner_position(i, j)`` only, the positions of its points. Nothing maps
    parametric coordinates, so there is no ``position`` or ``geometry_jacobian``."""

    @tack.func
    def corner_position(self, i, j):
        return self.point_value(i, j)

    @tack.func
    def point_position(self, p):
        """The position of point ``p``, by its global id."""
        return self.get(p)


class _H1Geometry(_Geometry):
    """H1 geometry: one position per point, shared by the cells around it."""

    @tack.func
    def corner_position(self, i, j):
        return self.point_value(i, j)


class _L2Geometry(_Geometry):
    """L2 geometry: each cell's own corner positions, at ``point_offsets[cell]``. Cells
    may pull apart while the topology still has them share points."""

    @tack.func
    def corner_position(self, c, j):
        return self.get(self.point_offsets[self.entity_id(c)] + j)


class _H1Order2Geometry(_QuadraticDofs, _Geometry):
    """Curved geometry: an order-2 H1 field of positions. ``corner_position(c, j)`` is still
    corner ``j``'s; positions and Jacobians inside go through the quadratic
    functions. Its edge and face ids come from the domain view's incidence.
    Tetrahedra, triangles and wedges (hexahedra and quads take the tensor
    geometry, ``_H1TensorGeometry``, of order 2)."""

    CURVED = 1

    @tack.func
    def _edge_between(self, a, b):
        found = 0
        for e in range(self.NUM_EDGES):
            ea = self.edge_corner(e, 0)
            eb = self.edge_corner(e, 1)
            if (ea == a and eb == b) or (ea == b and eb == a):
                found = e
        return found

    @tack.func
    def control_box(self, c):
        """The box of the cell's Bernstein control points, which hold it: its corners,
        each edge's ``2 m - (x0 + x1) / 2`` (``m`` its middle node), and each quad
        face's ``4 f - (its edges' middles) + (its corners) / 4``."""
        a = self.corner_position(c, 0)
        b = a
        for j in range(1, self.NUM_POINTS):
            p = self.corner_position(c, j)
            a = min(a, p)
            b = max(b, p)
        for e in range(self.NUM_EDGES):
            p = (2.0 * self.get(self.dof_index(c, self.NUM_POINTS + e))
                 - 0.5 * (self.corner_position(c, self.edge_corner(e, 0))
                          + self.corner_position(c, self.edge_corner(e, 1))))
            a = min(a, p)
            b = max(b, p)
        for i in range(self.QUADRATIC_FACES):
            f = self.NUM_FACES - self.QUADRATIC_FACES + i
            p = 4.0 * self.get(self.dof_index(c, self.NUM_POINTS + self.NUM_EDGES + i))
            for k in range(4):
                ca = self.face_corner(f, k)
                cb = self.face_corner(f, (k + 1) % 4)
                p += (0.25 * self.corner_position(c, ca)
                      - self.get(self.dof_index(c, self.NUM_POINTS + self._edge_between(ca, cb))))
            a = min(a, p)
            b = max(b, p)
        return a, b

    @tack.func
    def corner_position(self, c, j):
        return self.get(self.point_id(c, j))

    @tack.func
    def position(self, c, pc):
        x = tack.vec3(0.0, 0.0, 0.0)
        for k in range(self.NUM_QUADRATIC):
            x += self.quadratic_function(k, pc) * self.get(self.dof_index(c, k))
        return x

    @tack.func
    def geometry_jacobian(self, c, pc):
        # Relative to the first node, as the linear geometry's (the quadratic
        # functions' gradients sum to zero too).
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        p0 = self.get(self.dof_index(c, 0))
        for k in range(1, self.NUM_QUADRATIC):
            m += (self.get(self.dof_index(c, k)) - p0).outer_product(
                self.quadratic_gradient(k, pc))
        return m


class _SideZeroGeometry(_Geometry):
    """A face's positions from its side 0's cell: the face as that cell sees it.
    For an L2 geometry, whose cells have their own corners, a face has no
    positions of its own; ``face_points`` holds each side's, per (face, side,
    point), as a ``SideTraces`` space lays them out."""

    @tack.func
    def corner_position(self, f, j):
        return self.face_points[self.entity_id(f) * 8 + j]


# ── Incidence (cells) ───────────────────────────────────────────────


class _FaceIncidence:
    """A cell's faces: ``side_face``/``side_slot`` hold one entry per (cell, local face)."""

    @tack.func
    def face_id(self, c, f):
        return self.side_face[self.face_start + self.index(c) * self.NUM_FACES + f]

    @tack.func
    def face_side(self, c, f):
        """0 if this cell is the face's side 0 (the face's normal points out of it), else 1."""
        return self.side_slot[self.face_start + self.index(c) * self.NUM_FACES + f]

    @tack.func
    def face_orientation(self, c, f):
        """``2 * r + reflected``: the cell's first point of local face ``f`` is point
        ``r`` of the face's own row, going round it the same way or (``reflected``)
        the other way."""
        return self.side_orientation[self.face_start + self.index(c) * self.NUM_FACES + f]


    @tack.func
    def face_position(self, c, f, k):
        """Where point ``k`` of the cell's local face ``f`` is in the face's own row."""
        o = self.face_orientation(c, f)
        n = self.face_num_points(f)
        r = o >> 1
        return (r + k) % n if (o & 1) == 0 else (r - k + n) % n


class _FaceWalk:
    """A shape-based cell's faces as a polyhedral cell's are walked -- the entity
    methods both paths share, so face-based algorithms run on either: the counts
    are the shape's constants, the points its face tables'. Needs the cell's face
    incidence and geometry, so only domain views have it."""

    @tack.func
    def num_faces(self, c):
        """The cell's number of faces: the shape's, a constant."""
        return self.NUM_FACES

    @tack.func
    def side_num_points(self, c, k):
        """The number of points of the cell's face ``k``."""
        return self.face_num_points(k)

    @tack.func
    def side_point_id(self, c, k, j):
        """Point ``j`` of the cell's face ``k``, going round it outward for this cell."""
        return self.point_id(c, self.face_corner(k, j))

    @tack.func
    def side_position(self, c, k, j):
        """The position of ``side_point_id(c, k, j)``."""
        return self.corner_position(c, self.face_corner(k, j))


class _EdgeIncidence:
    """A cell's edges: ``side_edge``/``side_sign`` hold one entry per (cell, local edge)."""

    @tack.func
    def edge_id(self, c, e):
        return self.side_edge[self.edge_start + self.index(c) * self.NUM_EDGES + e]

    @tack.func
    def edge_sign(self, c, e):
        """+1 if the cell's local edge ``e`` runs from the edge's low point id to its high one."""
        return self.side_sign[self.edge_start + self.index(c) * self.NUM_EDGES + e]


# ── Storage: what get(k) reads ──────────────────────────────────────


class _ExplicitStorage:
    """A device array: value ``k`` is stored at ``values[k]``."""

    @tack.func
    def get(self, k):
        return self.values[k]


class _ArrayStorage:
    """An implicit array (``tack.arrays``), a template the view holds: value ``k``
    is ``array.get(k)``, however the array is composed."""

    @tack.func
    def get(self, k):
        return self.array.get(k)


class _ArrayIJKStorage(_ArrayStorage):
    """An implicit array of grid points, which also reads the point at ``(i, j, k)``."""

    @tack.func
    def get_ijk(self, at):
        return self.array.get_ijk(at)


# ── Addressing: which value a point is ──────────────────────────────


class _PointAddress:
    """Point ``j`` of entity ``i`` is value ``point_id(i, j)``."""

    @tack.func
    def point_value(self, i, j):
        return self.get(self.point_id(i, j))


class _StructuredPointAddress:
    """For a structured grid's cells: the storage reads the corner's (i, j, k) directly."""

    @tack.func
    def point_value(self, c, j):
        return self.get_ijk(self.point_index(c, j))


def point_address(kind, storage, ijk=True):
    """The addressing mixin for entities of ``kind`` over ``storage``; ``ijk`` False
    when the storage's (i, j, k) are not the cells' (see ``_space_parts``)."""
    if ijk and issubclass(kind, _StructuredCells) and hasattr(storage, "get_ijk"):
        return _StructuredPointAddress
    return _PointAddress


# ── Spaces (field views) ────────────────────────────────────────────


class _Interpolated:
    """What the spaces with a basis share: ``value`` and ``parametric_gradient`` from
    ``dof(i, j)``, which each space defines, and the shape's functions."""

    @tack.func
    def value(self, i, pc):
        total = self.dof(i, 0) * self.shape_function(0, pc)
        for j in range(1, self.NUM_POINTS):
            total += self.dof(i, j) * self.shape_function(j, pc)
        return total

    @tack.func
    def parametric_gradient(self, i, pc):
        """The derivative by parametric coordinates at ``pc``, a 3-vector: from the
        values relative to the first (the shape's gradients sum to zero), so a
        large value varying little across a thin cell keeps its differences."""
        g = tack.vec3(0.0, 0.0, 0.0)
        u0 = self.dof(i, 0)
        for j in range(1, self.NUM_POINTS):
            g += (self.dof(i, j) - u0) * self.shape_gradient(j, pc)
        return g

    @tack.func
    def corner_value(self, i, j):
        """The value at the cell's corner ``j``: at order 1 the values are the
        corners', so it is value ``j`` itself."""
        return self.dof(i, j)


class _H1Field(_Interpolated):
    """H1, order 1, Shared: one value per point, interpolated by the shape functions."""

    @tack.func
    def dof(self, i, j):
        return self.point_value(i, j)


class _H1Order2Field(_QuadraticDofs, _Interpolated):
    """H1, order 2, Shared: values at the corners, the middles of edges and quad
    faces, and a hexahedron's center, interpolated by the shape's quadratic
    Lagrange functions."""

    ORDER = 2

    @tack.func
    def dof(self, c, k):
        return self.get(self.dof_index(c, k))

    @tack.func
    def value(self, c, pc):
        total = self.dof(c, 0) * self.quadratic_function(0, pc)
        for k in range(1, self.NUM_QUADRATIC):
            total += self.dof(c, k) * self.quadratic_function(k, pc)
        return total

    @tack.func
    def parametric_gradient(self, c, pc):
        g = tack.vec3(0.0, 0.0, 0.0)
        u0 = self.dof(c, 0)
        for k in range(1, self.NUM_QUADRATIC):
            g += (self.dof(c, k) - u0) * self.quadratic_gradient(k, pc)
        return g


class _L2Field(_Interpolated):
    """L2, order 1, PerCell: each cell's own corner values, at ``offsets[cell]``."""

    ORDER = 1

    @tack.func
    def dof(self, i, j):
        return self.get(self.offsets[self.entity_id(i)] + j)


class _L2Order0Field:
    """L2, order 0, PerCell: one value per cell, at ``offsets[cell]`` -- the
    cells of order 0 in a variable-order L2 field."""

    ORDER = 0

    @tack.func
    def dof(self, i, j):
        return self.get(self.offsets[self.entity_id(i)])

    @tack.func
    def value(self, i, pc):
        return self.get(self.offsets[self.entity_id(i)])

    @tack.func
    def parametric_gradient(self, i, pc):
        return tack.vec3(0.0, 0.0, 0.0)

    @tack.func
    def corner_value(self, i, j):
        return self.get(self.offsets[self.entity_id(i)])


# ── Tensor bases of any order (tack.dataset.bases) ──────────────────


class _TensorPolynomials:
    """A Lagrange basis of order ``ORDER`` on a tensor shape: the products of the 1D
    polynomials through ``basis_nodes``, evaluated in barycentric form with
    ``basis_weights`` (both in the field's float type). Node ``n`` is ``(i, j, k)``,
    ``n = i + q * (j + q * k)`` with ``q = ORDER + 1``."""

    @tack.func
    def lagrange_values(self, t, out):
        """``out[j]``: the 1D Lagrange polynomial of node ``j`` at ``t``."""
        for j in range(self.ORDER + 1):
            v = self.basis_weights[j]
            for m in range(self.ORDER + 1):
                if m != j:
                    v = v * (t - self.basis_nodes[m])
            out[j] = v

    @tack.func
    def lagrange_derivatives(self, t, out):
        """``out[j]``: its derivative at ``t``, by the product rule (no division, so
        exact at the nodes too)."""
        for j in range(self.ORDER + 1):
            d = self.basis_weights[j] - self.basis_weights[j]
            for k in range(self.ORDER + 1):
                if k != j:
                    v = self.basis_weights[j]
                    for m in range(self.ORDER + 1):
                        if m != j and m != k:
                            v = v * (t - self.basis_nodes[m])
                    d = d + v
            out[j] = d

    @tack.func
    def node_parametric(self, n):
        """Node ``n``'s parametric coordinates."""
        q = self.ORDER + 1
        pc = tack.vec3(self.basis_nodes[n % q], 0.0, 0.0)
        if self.DIMENSION >= 2:
            pc[1] = self.basis_nodes[(n // q) % q]
        if self.DIMENSION == 3:
            pc[2] = self.basis_nodes[n // (q * q)]
        return pc

    @tack.func
    def node_weight(self, n, lx, ly, lz):
        q = self.ORDER + 1
        w = lx[n % q]
        if self.DIMENSION >= 2:
            w = w * ly[(n // q) % q]
        if self.DIMENSION == 3:
            w = w * lz[n // (q * q)]
        return w

    @tack.func
    def node_gradient(self, n, lx, ly, lz, dx, dy, dz):
        q = self.ORDER + 1
        i = n % q
        j = (n // q) % q
        k = n // (q * q)
        g = tack.vec3(dx[i], 0.0, 0.0)
        if self.DIMENSION == 2:
            g = tack.vec3(dx[i] * ly[j], lx[i] * dy[j], 0.0)
        if self.DIMENSION == 3:
            g = tack.vec3(dx[i] * ly[j] * lz[k], lx[i] * dy[j] * lz[k], lx[i] * ly[j] * dz[k])
        return g



class _TensorBasis(_TensorPolynomials):
    """A field's value and parametric gradient in a tensor basis, from ``dof(c, n)``,
    the space's."""


    @tack.func
    def value(self, c, pc):
        lx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        ly = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        lz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        self.lagrange_values(pc[0], lx)
        self.lagrange_values(pc[1], ly)
        self.lagrange_values(pc[2], lz)
        total = self.dof(c, 0) * self.node_weight(0, lx, ly, lz)
        for n in range(1, (self.ORDER + 1) ** self.DIMENSION):
            total += self.dof(c, n) * self.node_weight(n, lx, ly, lz)
        return total

    @tack.func
    def parametric_gradient(self, c, pc):
        """From differences to node 0's value (the basis' gradients sum to zero), so
        a field of large, nearly equal values keeps its small differences."""
        lx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        ly = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        lz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dy = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        self.lagrange_values(pc[0], lx)
        self.lagrange_values(pc[1], ly)
        self.lagrange_values(pc[2], lz)
        self.lagrange_derivatives(pc[0], dx)
        self.lagrange_derivatives(pc[1], dy)
        self.lagrange_derivatives(pc[2], dz)
        first = self.dof(c, 0)
        g = (self.dof(c, 1) - first) * self.node_gradient(1, lx, ly, lz, dx, dy, dz)
        for n in range(2, (self.ORDER + 1) ** self.DIMENSION):
            g += (self.dof(c, n) - first) * self.node_gradient(n, lx, ly, lz, dx, dy, dz)
        return g


class _TensorDofs:
    """Where an ``H1`` space of order ``ORDER`` (3 or more) on tensor shapes keeps
    cell ``c``'s node ``n`` (``bases.dof_tables``: ``node_kind``,
    ``node_entity``, ``node_a``, ``node_b``): a corner's at its point; ``ORDER -
    1`` per edge from ``edge_base``, in order from the edge's lower point id;
    ``(ORDER - 1)^2`` per quad face from ``face_dofs[face]``, lexicographic in the
    frame of the face's own row; the interior's from ``cell_dofs[cell]``,
    lexicographic in the cell's frame. So every cell reading a shared node finds
    the same value."""

    @tack.func
    def dof_index(self, c, n):
        kind = self.node_kind[n]
        e = self.node_entity[n]
        if kind == 0:
            return self.point_id(c, e)
        if kind == 1:
            return self.tensor_edge_dof(c, e, self.node_a[n])
        if kind == 2:
            return self.tensor_face_dof(c, e, self.node_a[n], self.node_b[n])
        return self.cell_dofs[self.entity_id(c)] + self.node_a[n]


class _TensorEdges:
    """A shape with edges: an edge node's value, in order from the edge's lower point id."""

    @tack.func
    def tensor_edge_dof(self, c, e, a):
        s = a if self.edge_sign(c, e) > 0 else self.ORDER - a
        return self.edge_base + self.edge_id(c, e) * (self.ORDER - 1) + (s - 1)


class _NoTensorEdges:
    """A segment, whose nodes are its points' and its interior's."""

    @tack.func
    def tensor_edge_dof(self, c, e, a):
        return self.cell_dofs[self.entity_id(c)]          # not reached


class _TensorFaces:
    """A solid's quad face node ``(a, b)`` (from the cell's local corner 0 of the face,
    toward its corners 1 and 3), placed in the frame of the face's own row --
    the same for both cells sharing it -- by the face's orientation in the cell."""

    @tack.func
    def tensor_face_dof(self, c, f, a, b):
        p = self.ORDER
        r0 = self.face_position(c, f, 0)
        r1 = self.face_position(c, f, 1)
        r3 = self.face_position(c, f, 3)
        # Row corner m sits at (p, 0) for 1, (p, p) for 2, (0, p) for 3.
        x0 = p if (r0 == 1 or r0 == 2) else 0
        y0 = p if (r0 == 2 or r0 == 3) else 0
        x1 = p if (r1 == 1 or r1 == 2) else 0
        y1 = p if (r1 == 2 or r1 == 3) else 0
        x3 = p if (r3 == 1 or r3 == 2) else 0
        y3 = p if (r3 == 2 or r3 == 3) else 0
        u = x0 + a * ((x1 - x0) // p) + b * ((x3 - x0) // p)
        v = y0 + a * ((y1 - y0) // p) + b * ((y3 - y0) // p)
        return self.face_dofs[self.face_id(c, f)] + (v - 1) * (p - 1) + (u - 1)


class _NoTensorFaces:
    """A segment or a 2D cell: its edges are its sides, and it has no face nodes."""

    @tack.func
    def tensor_face_dof(self, c, f, a, b):
        return self.cell_dofs[self.entity_id(c)]          # not reached


class _H1TensorGeometry(_TensorDofs, _TensorPolynomials, _Geometry):
    """Curved geometry of order 2 or more on tensor shapes: an ``H1`` field of positions
    of that order. Positions and Jacobians are formed from differences to node 0's
    position, as order 1's are, so f32 keeps thin cells."""

    CURVED = 1

    @tack.func
    def control_box(self, c):
        """The box of the cell's Bernstein control points, which hold it: the nodes'
        positions (less node 0's) taken to Bernstein coefficients one axis at a
        time by ``basis_bernstein``, the 1D matrix."""
        q = self.ORDER + 1
        count = (self.ORDER + 1) ** self.DIMENSION
        cur = tack.local_array_like(self.basis_nodes, 3 * (self.ORDER + 1) ** self.DIMENSION)
        nxt = tack.local_array_like(self.basis_nodes, 3 * (self.ORDER + 1) ** self.DIMENSION)
        first = self.get(self.dof_index(c, 0))
        for n in range(count):
            v = self.get(self.dof_index(c, n)) - first
            cur[3 * n] = v[0]
            cur[3 * n + 1] = v[1]
            cur[3 * n + 2] = v[2]
        stride = 1
        for _ in range(self.DIMENSION):
            for n in range(count):
                i = (n // stride) % q
                base = n - i * stride
                s0 = cur[0] - cur[0]
                s1 = s0
                s2 = s0
                for m in range(q):
                    w = self.basis_bernstein[i * q + m]
                    src = 3 * (base + m * stride)
                    s0 += w * cur[src]
                    s1 += w * cur[src + 1]
                    s2 += w * cur[src + 2]
                nxt[3 * n] = s0
                nxt[3 * n + 1] = s1
                nxt[3 * n + 2] = s2
            for n in range(3 * count):
                cur[n] = nxt[n]
            stride *= q
        a = tack.vec3(cur[0], cur[1], cur[2])
        b = a
        for n in range(1, count):
            p = tack.vec3(cur[3 * n], cur[3 * n + 1], cur[3 * n + 2])
            a = min(a, p)
            b = max(b, p)
        return first + a, first + b

    @tack.func
    def corner_position(self, c, j):
        return self.get(self.point_id(c, j))

    @tack.func
    def position(self, c, pc):
        lx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        ly = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        lz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        self.lagrange_values(pc[0], lx)
        self.lagrange_values(pc[1], ly)
        self.lagrange_values(pc[2], lz)
        first = self.get(self.dof_index(c, 0))
        x = first
        for n in range(1, (self.ORDER + 1) ** self.DIMENSION):
            x += (self.get(self.dof_index(c, n)) - first) * self.node_weight(n, lx, ly, lz)
        return x

    @tack.func
    def geometry_jacobian(self, c, pc):
        lx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        ly = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        lz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dy = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        self.lagrange_values(pc[0], lx)
        self.lagrange_values(pc[1], ly)
        self.lagrange_values(pc[2], lz)
        self.lagrange_derivatives(pc[0], dx)
        self.lagrange_derivatives(pc[1], dy)
        self.lagrange_derivatives(pc[2], dz)
        first = self.get(self.dof_index(c, 0))
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for n in range(1, (self.ORDER + 1) ** self.DIMENSION):
            m += (self.get(self.dof_index(c, n)) - first).outer_product(
                self.node_gradient(n, lx, ly, lz, dx, dy, dz))
        return m


class _H1TensorField(_TensorDofs, _TensorBasis):
    """H1 of order 3 or more on tensor shapes, Shared."""

    @tack.func
    def dof(self, c, n):
        return self.get(self.dof_index(c, n))

    @tack.func
    def corner_value(self, c, j):
        return self.dof(c, self.corner_node[j])


class _L2TensorField(_TensorBasis):
    """L2 on tensor shapes at order 2 or more, or on Gauss-Legendre nodes: each cell's
    ``(ORDER + 1)^d`` values from ``offsets[cell]``, in node order."""

    @tack.func
    def dof(self, c, n):
        return self.get(self.offsets[self.entity_id(c)] + n)

    @tack.func
    def corner_value(self, c, j):
        return self.value(c, self.parametric_point(j))


def tensor_mixins(shape):
    """The edge and face mixins a tensor ``H1`` view of ``shape`` cells needs."""
    edges = _TensorEdges if shape.NUM_EDGES and shape.DIMENSION >= 2 else _NoTensorEdges
    faces = _TensorFaces if shape.DIMENSION == 3 else _NoTensorFaces
    return edges, faces


_by_order = {}


def of_order(mixin, order):
    """``mixin`` with ``ORDER`` set: one class per (mixin, order), so views of one
    order share their compiled kernels."""
    key = (mixin, int(order))
    if key not in _by_order:
        _by_order[key] = type(f"{mixin.__name__}{order}", (mixin,), {"ORDER": int(order)})
    return _by_order[key]


class _ConstantField:
    """Constant (L2, order 0): one value per cell."""

    @tack.func
    def dof(self, i, j):
        return self.get(self.entity_id(i))

    @tack.func
    def value(self, i, pc):
        return self.get(self.entity_id(i))

    @tack.func
    def parametric_gradient(self, i, pc):
        return tack.vec3(0.0, 0.0, 0.0)

    @tack.func
    def corner_value(self, i, j):
        return self.get(self.entity_id(i))


class _PolyhedralPointField:
    """Point data (H1, order 1) in a polyhedron: linear on its sub-tetrahedra
    (``_PolyhedralLocation``), whose cell and face points take the means of the
    cell's and the face's points' values, as their positions are the means of
    the points' positions -- so a linear field is exact."""

    @tack.func
    def corner_value(self, c, j):
        """The value at the cell's point ``j``, of ``num_points(c)``."""
        return self.point_value(c, j)

    @tack.func
    def cell_point_value(self, c):
        n = self.num_points(c)
        v = self.point_value(c, 0)
        for j in range(1, n):
            v += self.point_value(c, j)
        return v / n

    @tack.func
    def face_point_value(self, c, k):
        n = self.side_num_points(c, k)
        v = self.get(self.side_point_id(c, k, 0))
        for j in range(1, n):
            v += self.get(self.side_point_id(c, k, j))
        return v / n

    @tack.func
    def tet_value(self, c, t, pc):
        """The value at ``pc`` in sub-tetrahedron ``t``."""
        k, j = self.tet_face(c, t)
        n = self.side_num_points(c, k)
        nxt = j + 1 if j + 1 < n else 0
        u0 = self.cell_point_value(c)
        return (u0 + pc[0] * (self.face_point_value(c, k) - u0)
                + pc[1] * (self.get(self.side_point_id(c, k, j)) - u0)
                + pc[2] * (self.get(self.side_point_id(c, k, nxt)) - u0))

    @tack.func
    def tet_parametric_gradient(self, c, t):
        """A scalar field's derivative by ``pc`` in sub-tetrahedron ``t``."""
        k, j = self.tet_face(c, t)
        n = self.side_num_points(c, k)
        nxt = j + 1 if j + 1 < n else 0
        u0 = self.cell_point_value(c)
        return tack.vec3(self.face_point_value(c, k) - u0,
                         self.get(self.side_point_id(c, k, j)) - u0,
                         self.get(self.side_point_id(c, k, nxt)) - u0)


class _PolylinePointField:
    """Point data (H1, order 1) on a polyline: linear along each segment
    (``_PolylineLocation``)."""

    @tack.func
    def corner_value(self, c, j):
        return self.point_value(c, j)

    @tack.func
    def tet_value(self, c, t, pc):
        """The value at ``pc`` on segment ``t``."""
        return (pc[1] * self.get(self.line_point_id(c, t))
                + pc[2] * self.get(self.line_point_id(c, t + 1)))


class _PolyhedralCellField(_ConstantField):
    """Cell data (Constant) in a polyhedron: its value in every sub-tetrahedron."""

    @tack.func
    def tet_value(self, c, t, pc):
        return self.get(self.entity_id(c))

    @tack.func
    def tet_parametric_gradient(self, c, t):
        return tack.vec3(0.0, 0.0, 0.0)


class _SideTracesField:
    """Values on each side of each face: for face ``f``, side ``s`` (0 or 1) and the
    face's point ``j`` (in the face's own row), value ``(f * 2 + s) * 4 + j``.
    ``value(f, s, pc)`` interpolates them with the face shape's functions, so
    each side's trace is evaluated where the other's is."""

    @tack.func
    def trace(self, f, s, j):
        return self.get((self.entity_id(f) * 2 + s) * 4 + j)

    @tack.func
    def value(self, f, s, pc):
        total = self.trace(f, s, 0) * self.shape_function(0, pc)
        for j in range(1, self.NUM_POINTS):
            total += self.trace(f, s, j) * self.shape_function(j, pc)
        return total


class _ValuesField:
    """Values on entities, no basis: one value per entity of the field's kind."""

    @tack.func
    def at(self, i):
        return self.get(self.entity_id(i))


# ── A space's nodes (for_each over a space) ─────────────────────────
#
# A node view is an entity view -- points, edges, quad faces or cells -- with a
# dof mixin (``node_dof(i, j)``: where node ``j`` of entity ``i`` keeps its
# value; ``node_count(i)`` of them) and a position mixin (``node_position(i,
# j)``: where the node is). Shared nodes are one per
# entity; L2's are per cell.


class _PointDof:
    """H1's (or values on points') node at a point."""

    @tack.func
    def node_count(self, i):
        return 1

    @tack.func
    def node_dof(self, i, j):
        return self.entity_id(i)


class _EdgeDof:
    """H1 order 2's node at an edge's middle."""

    @tack.func
    def node_count(self, i):
        return 1

    @tack.func
    def node_dof(self, i, j):
        return self.node_edge_base + self.entity_id(i)


class _FaceDof:
    """H1 order 2's node at a quad face's center."""

    @tack.func
    def node_count(self, i):
        return 1

    @tack.func
    def node_dof(self, i, j):
        return self.node_face_dofs[self.entity_id(i)]


class _InteriorDof:
    """H1 order 2's node inside a hexahedron or quad."""

    @tack.func
    def node_count(self, i):
        return 1

    @tack.func
    def node_dof(self, i, j):
        return self.node_cell_dofs[self.entity_id(i)]


class _CornerDof:
    """L2 order 1's nodes: each cell's own corners."""

    @tack.func
    def node_count(self, i):
        return self.NUM_POINTS

    @tack.func
    def node_dof(self, i, j):
        return self.node_offsets[self.entity_id(i)] + j


class _OffsetDof:
    """L2 order 0's node: one per cell, at its offset."""

    @tack.func
    def node_count(self, i):
        return 1

    @tack.func
    def node_dof(self, i, j):
        return self.node_offsets[self.entity_id(i)]


class _CellDof:
    """Constant's (or values on cells') node: one per cell."""

    @tack.func
    def node_count(self, i):
        return 1

    @tack.func
    def node_dof(self, i, j):
        return self.entity_id(i)


class _MeanPosition:
    """A shared node of a straight (order-1) geometry: the mean of its entity's
    points -- a point itself, an edge's middle, the bilinear center of a quad,
    the trilinear center of a hexahedron."""

    @tack.func
    def node_position(self, i, j):
        x = self.corner_position(i, 0)
        for k in range(1, self.NUM_POINTS):
            x += self.corner_position(i, k)
        return x / self.NUM_POINTS


class _GeometryAtDof:
    """A shared node of an order-2 geometry: the geometry's own value there, the
    two spaces sharing the layout."""

    @tack.func
    def node_position(self, i, j):
        return self.get(self.node_dof(i, j))


class _CornerPosition:
    """A cell's own corner node: where the geometry puts the corner."""

    @tack.func
    def node_position(self, i, j):
        return self.corner_position(i, j)


class _CenterPosition:
    """A cell's one node: where the geometry maps the parametric center."""

    @tack.func
    def node_position(self, i, j):
        return self.position(i, self.parametric_center())


class _TensorEdgeNodes:
    """``H1`` of order ``ORDER`` (3 or more): an edge's ``ORDER - 1`` nodes, in order from
    its lower point id, at the space's 1D nodes (``node_points``) along it -- the
    edge's straight line between its points, or the geometry's own value there
    (``_GeometryAtDof``) when the geometry shares the space's layout."""

    @tack.func
    def node_count(self, i):
        return self.ORDER - 1

    @tack.func
    def node_dof(self, i, j):
        return self.node_edge_base + self.entity_id(i) * (self.ORDER - 1) + j

    @tack.func
    def node_place(self, i, j):
        return tack.vec3(self.node_points[j + 1], 0.0, 0.0)


class _TensorFaceNodes:
    """``H1`` of order ``ORDER`` (3 or more): a quad face's ``(ORDER - 1)^2`` nodes,
    lexicographic in the frame of its own row (from its point 0, toward its points
    1 and 3)."""

    @tack.func
    def node_count(self, i):
        return (self.ORDER - 1) * (self.ORDER - 1)

    @tack.func
    def node_dof(self, i, j):
        return self.node_face_dofs[self.entity_id(i)] + j

    @tack.func
    def node_place(self, i, j):
        return tack.vec3(self.node_points[j % (self.ORDER - 1) + 1],
                         self.node_points[j // (self.ORDER - 1) + 1], 0.0)


class _StraightNodePosition:
    """An edge's or quad face's node where its straight (order-1) geometry puts it:
    linear between an edge's points, bilinear over a face's, at ``node_place``."""

    @tack.func
    def node_position(self, i, j):
        t = self.node_place(i, j)
        if self.NUM_POINTS == 2:
            return self.corner_position(i, 0) + t[0] * (self.corner_position(i, 1)
                                                        - self.corner_position(i, 0))
        x0 = self.corner_position(i, 0)
        return (x0 + t[0] * (1.0 - t[1]) * (self.corner_position(i, 1) - x0)
                + t[0] * t[1] * (self.corner_position(i, 2) - x0)
                + (1.0 - t[0]) * t[1] * (self.corner_position(i, 3) - x0))


class _TensorInteriorNodes:
    """``H1`` of order ``ORDER`` (3 or more): a cell's ``(ORDER - 1)^d`` interior nodes,
    lexicographic in its frame, where its geometry maps them."""

    @tack.func
    def node_count(self, i):
        n = 1
        for _ in range(self.DIMENSION):
            n *= self.ORDER - 1
        return n

    @tack.func
    def node_dof(self, i, j):
        return self.node_cell_dofs[self.entity_id(i)] + j

    @tack.func
    def node_position(self, i, j):
        q = self.ORDER - 1
        pc = tack.vec3(self.node_points[j % q + 1], 0.0, 0.0)
        if self.DIMENSION >= 2:
            pc[1] = self.node_points[(j // q) % q + 1]
        if self.DIMENSION == 3:
            pc[2] = self.node_points[j // (q * q) + 1]
        return self.position(i, pc)


class _TensorCellNodes:
    """``L2`` on a tensor basis: a cell's ``(ORDER + 1)^d`` nodes, in node order, where
    its geometry maps them."""

    @tack.func
    def node_count(self, i):
        n = 1
        for _ in range(self.DIMENSION):
            n *= self.ORDER + 1
        return n

    @tack.func
    def node_dof(self, i, j):
        return self.node_offsets[self.entity_id(i)] + j

    @tack.func
    def node_position(self, i, j):
        q = self.ORDER + 1
        pc = tack.vec3(self.node_points[j % q], 0.0, 0.0)
        if self.DIMENSION >= 2:
            pc[1] = self.node_points[(j // q) % q]
        if self.DIMENSION == 3:
            pc[2] = self.node_points[j // (q * q)]
        return self.position(i, pc)


# ── Groups and view classes ─────────────────────────────────────────

_view_classes = {}


def view_class(kind, shape, *mixins):
    """The template class combining ``mixins``, ``kind`` and ``shape``, made once."""
    key = (kind, shape, mixins)
    if key not in _view_classes:
        _check_names(*mixins, kind, shape)
        name = "".join(m.__name__.strip("_") for m in mixins) + kind.__name__ + shape.__name__
        _view_classes[key] = tack.device_class(type(name, (*mixins, kind, shape), {}))
    return _view_classes[key]


def _check_names(*bases):
    """Refuse a name two of ``bases`` define differently: stacked, one would hide
    the other unseen. A name both inherit from one class is the same name."""
    seen = {}
    for base in bases:
        names = {}
        for klass in reversed(base.__mro__[:-1]):
            names.update(vars(klass))
        for name, value in names.items():
            if name.startswith("__") and name.endswith("__"):
                continue
            first = seen.setdefault(name, (base, value))
            if first[1] is not value:
                raise TypeError(f"{first[0].__name__} and {base.__name__} both define "
                                f"{name!r}: one would hide the other in a view")


class DomainGroup:
    """One shape's entities of one kind: ``kind(*args)`` with ``count`` of them,
    starting at ``start`` in the topology's per-group layout.

    A *subgroup* -- the cells of a group that the spaces of a launch's fields
    put together -- has ``parent``, the topology's group it came from (whose
    incidence it shares), and ``keys``, each varying space's key for its
    cells (for a variable-order space, the order).
    """

    def __init__(self, kind, shape, args, count, start, parent=None, keys=None):
        self.kind = kind
        self.shape = shape
        self.args = args
        self.count = count
        self.start = start
        self.parent = parent
        self.keys = keys or {}

    @property
    def root(self):
        """The topology's group: this one, or the one this subgroup came from."""
        return self.parent or self

    def view(self, *mixins, **attributes):
        """An instance of the view class with ``mixins``, holding ``attributes``."""
        cls = view_class(self.kind, self.shape, *mixins)
        view = cls(*self.args)
        for name, value in attributes.items():
            # The mixins share one namespace: an attribute named like a method
            # would hide it, and the kernel would fail far from here.
            if hasattr(cls, name):
                raise AttributeError(f"{cls.__name__}.{name} is already defined; "
                                     "an attribute of that name would hide it")
            setattr(view, name, value)
        return view


def storage(values):
    """The storage mixin that reads ``values``, and the attributes it needs."""
    from tack.arrays import _DeviceArray, _Implicit

    if isinstance(values, _Implicit):
        mixin = _ArrayIJKStorage if hasattr(values, "get_ijk") else _ArrayStorage
        return mixin, {"array": values}
    if isinstance(values, _DeviceArray):
        return _ExplicitStorage, {"values": values}
    raise TypeError(f"field values must be a Tack array or an implicit array, "
                    f"not {type(values).__name__}")
