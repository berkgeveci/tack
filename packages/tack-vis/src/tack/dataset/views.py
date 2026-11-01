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
from tack.dataset.ghosts import _before
from tack.dataset.shapes import HOLD_TOLERANCE, NEWTON_DIVERGED, NEWTON_TOLERANCE, _symmetric

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

#: Newton steps a curved cell takes past VTK's stopping rule (a step under
#: ``NEWTON_TOLERANCE``): Newton's error squares each step, so two take the
#: answer from about 1e-4 to rounding (curved probes agreed with MFEM to 1e-8).
CURVED_NEWTON_POLISH = tack.constant(2, tack.i32)


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

    @tack.func
    def geometry_curvature(self, i, pc, g):
        """``g . d2x/d(pc)2``: the position's second derivatives by parametric coordinates
        along the world vector ``g``, a symmetric 3x3 matrix (zero where the map is
        affine), which a field's world Hessian takes from its parametric one."""
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        p0 = self.corner_position(i, 0)
        for j in range(1, self.NUM_POINTS):
            m += (self.corner_position(i, j) - p0).dot(g) * self.shape_hessian(j, pc)
        return m


class _CornerBox:
    """A straight geometry's cell bounds, for the cell locator: its corners' box.
    (A curved geometry's mixin defines ``control_box`` and ``CURVED`` itself.)"""

    #: Whether the geometry is curved (of order 2 or more), so that locating a point
    #: takes ``curved_to_parametric`` and a cell's bounds its control points.
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



class _CurvedNewton:
    """Locating a point in a cell of any geometry with a reference element, for the cell
    locator: its own map inverted (cells' domain views only)."""

    @tack.func
    def curved_to_parametric(self, i, x):
        """``world_to_parametric`` on the geometry itself: Newton's method from the
        parametric center on ``position`` and ``geometry_jacobian``, the shape's
        step and VTK's stopping rules, then ``CURVED_NEWTON_POLISH`` steps more, so
        a curved cell is inverted as it is, not as its corners make it, to
        rounding. Returns ``(pc, converged)``."""
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
        if converged == 1:
            for _ in range(CURVED_NEWTON_POLISH):
                pc += self._newton_step(self.geometry_jacobian(i, pc), x - self.position(i, pc))
        return pc, converged


@tack.func
def _newton_from(view, x, rx, ry, rz, first):
    """``_CurvedNewton``'s loop on a map the view evaluates from gathered nodes
    (``view.gathered_map``): the same steps and stopping rules."""
    pc = view.parametric_center()
    converged = 0
    for _ in range(CURVED_NEWTON_ITERATIONS):
        jac, at = view.gathered_map(rx, ry, rz, first, pc)
        step = view._newton_step(jac, x - at)
        pc += step
        if not (abs(pc[0]) < NEWTON_DIVERGED and abs(pc[1]) < NEWTON_DIVERGED
                and abs(pc[2]) < NEWTON_DIVERGED):
            break
        if (abs(step[0]) <= NEWTON_TOLERANCE and abs(step[1]) <= NEWTON_TOLERANCE
                and abs(step[2]) <= NEWTON_TOLERANCE):
            converged = 1
            break
    if converged == 1:
        for _ in range(CURVED_NEWTON_POLISH):
            jac, at = view.gathered_map(rx, ry, rz, first, pc)
            pc += view._newton_step(jac, x - at)
    return pc, converged


class _TensorNewton:
    """``_CurvedNewton`` for a curved tensor geometry (``_H1TensorGeometry``): the
    cell's nodes read once, not at every step through ``dof_index`` (twice a step,
    for ``position`` and ``geometry_jacobian``: on Metal most of a curved probe's
    time); the map formed from them as those two form it, so the same bits."""

    @tack.func
    def curved_to_parametric(self, i, x):
        count = (self.ORDER + 1) ** self.DIMENSION
        rx = tack.local_array_like(self.basis_nodes, (self.ORDER + 1) ** self.DIMENSION)
        ry = tack.local_array_like(self.basis_nodes, (self.ORDER + 1) ** self.DIMENSION)
        rz = tack.local_array_like(self.basis_nodes, (self.ORDER + 1) ** self.DIMENSION)
        first = self.get(self.dof_index(i, 0))
        for n in range(1, count):
            d = self.get(self.dof_index(i, n)) - first
            rx[n] = d[0]
            ry[n] = d[1]
            rz[n] = d[2]
        return _newton_from(self, x, rx, ry, rz, first)

    @tack.func
    def gathered_map(self, rx, ry, rz, first, pc):
        """``(geometry_jacobian, position)`` at ``pc`` from the gathered differences."""
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
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        at = first
        for n in range(1, (self.ORDER + 1) ** self.DIMENSION):
            d = tack.vec3(rx[n], ry[n], rz[n])
            at += d * self.node_weight(n, lx, ly, lz)
            m += d.outer_product(self.node_gradient(n, lx, ly, lz, dx, dy, dz))
        return m, at


class _QuadraticNewton:
    """``_CurvedNewton`` for an order-2 geometry on simplices and wedges
    (``_H1Order2Geometry``): its nodes read once, the map formed as ``position``
    and ``geometry_jacobian`` form it."""

    @tack.func
    def curved_to_parametric(self, i, x):
        px = tack.local_array_like(self.values, 27)
        py = tack.local_array_like(self.values, 27)
        pz = tack.local_array_like(self.values, 27)
        for k in range(self.NUM_QUADRATIC):
            p = self.get(self.dof_index(i, k))
            px[k] = p[0]
            py[k] = p[1]
            pz[k] = p[2]
        return _newton_from(self, x, px, py, pz, x - x)

    @tack.func
    def gathered_map(self, px, py, pz, first, pc):
        at = tack.vec3(0.0, 0.0, 0.0)
        for k in range(self.NUM_QUADRATIC):
            at += self.quadratic_function(k, pc) * tack.vec3(px[k], py[k], pz[k])
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        p0 = tack.vec3(px[0], py[0], pz[0])
        for k in range(1, self.NUM_QUADRATIC):
            m += (tack.vec3(px[k], py[k], pz[k]) - p0).outer_product(
                self.quadratic_gradient(k, pc))
        return m, at


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

    @tack.func
    def geometry_curvature(self, c, pc, g):
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        p0 = self.get(self.dof_index(c, 0))
        for k in range(1, self.NUM_QUADRATIC):
            m += (self.get(self.dof_index(c, k)) - p0).dot(g) * self.quadratic_hessian(k, pc)
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
    ``dof(i, j)``, which each space defines, and the shape's functions.

    Every field view of a shape with a reference element numbers a cell's nodes
    alike: ``NODES`` of them (a class constant, set where the view class is made
    when it depends on the shape, ``_node_count``), node ``k`` at
    ``node_parametric(k)``, its value ``dof(c, k)`` stored at ``dof_index(c, k)``."""

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
    def parametric_hessian(self, i, pc):
        """The second derivatives by parametric coordinates at ``pc``, a symmetric 3x3
        matrix, from the values relative to the first as the gradient is."""
        h = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        u0 = self.dof(i, 0)
        for j in range(1, self.NUM_POINTS):
            h += (self.dof(i, j) - u0) * self.shape_hessian(j, pc)
        return h

    @tack.func
    def parametric_jet(self, c, pc):
        """The value, parametric gradient and parametric Hessian at ``pc``, from one pass over
        the cell's values (each read once), relative to the first as the gradient's."""
        value = pc[0] - pc[0]
        grad = tack.vec3(0.0, 0.0, 0.0)
        hess = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        first = value
        for j in range(self.NUM_POINTS):
            v = self.dof(c, j)
            if j == 0:
                first = v
            d = v - first
            value += d * self.shape_function(j, pc)
            grad += d * self.shape_gradient(j, pc)
            hess += d * self.shape_hessian(j, pc)
        return first + value, grad, hess

    @tack.func
    def corner_value(self, i, j):
        """The value at the cell's corner ``j``: at order 1 the values are the
        corners', so it is value ``j`` itself."""
        return self.dof(i, j)

    @tack.func
    def flatness(self, i):
        """What linearizing the cell at its corners loses: nothing at order 1."""
        return abs(self.dof(i, 0) - self.dof(i, 0))

    @tack.func
    def value_range(self, i):
        """``(lo, hi)`` holding the field over the cell: at order 1, its corner values
        (each shape's functions are its Bernstein basis there)."""
        lo = self.dof(i, 0)
        hi = lo
        for j in range(1, self.NUM_POINTS):
            v = self.dof(i, j)
            lo = min(lo, v)
            hi = max(hi, v)
        return lo, hi


class _H1Field(_Interpolated):
    """H1, order 1, Shared: one value per point, interpolated by the shape functions.
    Its nodes are the cell's corners."""

    @tack.func
    def dof_index(self, i, j):
        return self.point_id(i, j)

    @tack.func
    def dof(self, i, j):
        return self.point_value(i, j)

    @tack.func
    def node_parametric(self, j):
        return self.parametric_point(j)


class _Order2Range:
    """An order-2 field's Bernstein coefficients over a cell, from its nodes: the
    corners' values; each edge's ``2 m - (v0 + v1) / 2`` (``m`` its middle); each quad
    face's ``4 f - (its edges' middles) + (its corners) / 4``; a hexahedron's
    interior ``8 c - 2 (its faces' centers) + (its edges' middles) / 2 - (its corners)
    / 8``, a quad's ``4 c - (its edges' middles) + (its corners) / 4``. The field lies
    between the least and the greatest (``value_range``); each one's distance from
    the corners' multilinear interpolation where it sits bounds what linearizing
    the cell at its corners loses (``flatness``)."""

    @tack.func
    def _between(self, a, b):
        found = 0
        for e in range(self.NUM_EDGES):
            ea = self.edge_corner(e, 0)
            eb = self.edge_corner(e, 1)
            if (ea == a and eb == b) or (ea == b and eb == a):
                found = e
        return found

    @tack.func
    def _corner_mean(self, c):
        total = self.dof(c, 0)
        for j in range(1, self.NUM_POINTS):
            total += self.dof(c, j)
        return total / self.NUM_POINTS

    @tack.func
    def _edge_sum(self, c):
        total = self.dof(c, self.NUM_POINTS)
        for e in range(1, self.NUM_EDGES):
            total += self.dof(c, self.NUM_POINTS + e)
        return total

    @tack.func
    def _interior_control(self, c):
        """The interior's coefficient (a hexahedron's or a quad's center node)."""
        center = self.dof(c, self.NUM_POINTS + self.NUM_EDGES + self.QUADRATIC_FACES)
        corners = self._corner_mean(c) * self.NUM_POINTS
        v = 4.0 * center - self._edge_sum(c) + 0.25 * corners
        if self.DIMENSION == 3:
            faces = self.dof(c, self.NUM_POINTS + self.NUM_EDGES)
            for i in range(1, self.QUADRATIC_FACES):
                faces += self.dof(c, self.NUM_POINTS + self.NUM_EDGES + i)
            v = 8.0 * center - 2.0 * faces + 0.5 * self._edge_sum(c) - 0.125 * corners
        return v

    @tack.func
    def _face_control(self, c, i):
        f = self.NUM_FACES - self.QUADRATIC_FACES + i
        v = 4.0 * self.dof(c, self.NUM_POINTS + self.NUM_EDGES + i)
        for k in range(4):
            ca = self.face_corner(f, k)
            cb = self.face_corner(f, (k + 1) % 4)
            v += 0.25 * self.dof(c, ca) - self.dof(c, self.NUM_POINTS + self._between(ca, cb))
        return v

    @tack.func
    def _face_corner_mean(self, c, i):
        f = self.NUM_FACES - self.QUADRATIC_FACES + i
        total = self.dof(c, self.face_corner(f, 0))
        for k in range(1, 4):
            total += self.dof(c, self.face_corner(f, k))
        return 0.25 * total

    @tack.func
    def _edge_control(self, c, e):
        return (2.0 * self.dof(c, self.NUM_POINTS + e)
                - 0.5 * (self.dof(c, self.edge_corner(e, 0)) + self.dof(c, self.edge_corner(e, 1))))

    @tack.func
    def value_range(self, c):
        lo = self.dof(c, 0)
        hi = lo
        for j in range(1, self.NUM_POINTS):
            v = self.dof(c, j)
            lo = min(lo, v)
            hi = max(hi, v)
        for e in range(self.NUM_EDGES):
            v = self._edge_control(c, e)
            lo = min(lo, v)
            hi = max(hi, v)
        for i in range(self.QUADRATIC_FACES):
            v = self._face_control(c, i)
            lo = min(lo, v)
            hi = max(hi, v)
        if self.QUADRATIC_INTERIOR == 1:
            v = self._interior_control(c)
            lo = min(lo, v)
            hi = max(hi, v)
        return lo, hi

    @tack.func
    def flatness(self, c):
        """The largest distance of a coefficient from the corners' multilinear
        interpolation at its place: an edge's from its corners' mean, a face's
        from its four, the interior's from all."""
        worst = abs(self.dof(c, 0) - self.dof(c, 0))
        for e in range(self.NUM_EDGES):
            d = self._edge_control(c, e) - 0.5 * (self.dof(c, self.edge_corner(e, 0))
                                                   + self.dof(c, self.edge_corner(e, 1)))
            worst = max(worst, abs(d))
        for i in range(self.QUADRATIC_FACES):
            worst = max(worst, abs(self._face_control(c, i) - self._face_corner_mean(c, i)))
        if self.QUADRATIC_INTERIOR == 1:
            worst = max(worst, abs(self._interior_control(c) - self._corner_mean(c)))
        return worst


class _H1Order2Field(_QuadraticDofs, _Order2Range, _Interpolated):
    """H1, order 2, Shared: values at the corners, the middles of edges and quad
    faces, and a hexahedron's center, interpolated by the shape's quadratic
    Lagrange functions."""

    ORDER = 2

    @tack.func
    def dof(self, c, k):
        return self.get(self.dof_index(c, k))

    @tack.func
    def node_parametric(self, k):
        return self.quadratic_node(k)

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

    @tack.func
    def parametric_hessian(self, c, pc):
        h = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        u0 = self.dof(c, 0)
        for k in range(1, self.NUM_QUADRATIC):
            h += (self.dof(c, k) - u0) * self.quadratic_hessian(k, pc)
        return h

    @tack.func
    def parametric_jet(self, c, pc):
        """The value, parametric gradient and parametric Hessian at ``pc``, from one pass over
        the cell's values (each read once), relative to the first as the gradient's."""
        value = pc[0] - pc[0]
        grad = tack.vec3(0.0, 0.0, 0.0)
        hess = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        first = value
        for k in range(self.NUM_QUADRATIC):
            v = self.dof(c, k)
            if k == 0:
                first = v
            d = v - first
            value += d * self.quadratic_function(k, pc)
            grad += d * self.quadratic_gradient(k, pc)
            hess += d * self.quadratic_hessian(k, pc)
        return first + value, grad, hess


class _SerendipityField(_H1Order2Field):
    """H1, order 2, serendipity: read as the order-2 field whose face and interior
    nodes the views compute (``_serendipity.full_values``), but its ``NODES`` are
    the ones it stores, the corners and edges, which come first."""


class _L2Field(_Interpolated):
    """L2, order 1, PerCell: each cell's own corner values, at ``offsets[cell]``."""

    ORDER = 1

    @tack.func
    def dof_index(self, i, j):
        return self.offsets[self.entity_id(i)] + j

    @tack.func
    def dof(self, i, j):
        return self.get(self.dof_index(i, j))

    @tack.func
    def node_parametric(self, j):
        return self.parametric_point(j)


class _L2Order0Field:
    """L2, order 0, PerCell: one value per cell, at ``offsets[cell]`` -- the
    cells of order 0 in a variable-order L2 field."""

    ORDER = 0
    NODES = 1

    @tack.func
    def dof_index(self, i, j):
        return self.offsets[self.entity_id(i)]

    @tack.func
    def node_parametric(self, j):
        """The one node's: the cell's parametric center."""
        return self.parametric_center()

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

    @tack.func
    def parametric_hessian(self, i, pc):
        return tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    @tack.func
    def parametric_jet(self, i, pc):
        return self.value(i, pc), tack.vec3(0.0, 0.0, 0.0), tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    @tack.func
    def value_range(self, i):
        v = self.dof(i, 0)                                # the cell's one value
        return v, v

    @tack.func
    def flatness(self, i):
        return abs(self.dof(i, 0) - self.dof(i, 0))


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
    def lagrange_second_derivatives(self, t, out):
        """``out[j]``: its second derivative at ``t``, the products leaving out two
        factors, each ordered pair once."""
        for j in range(self.ORDER + 1):
            d = self.basis_weights[j] - self.basis_weights[j]
            for k in range(self.ORDER + 1):
                for l in range(self.ORDER + 1):
                    if k != j and l != j and l != k:
                        v = self.basis_weights[j]
                        for m in range(self.ORDER + 1):
                            if m != j and m != k and m != l:
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

    @tack.func
    def node_hessian(self, n, lx, ly, lz, dx, dy, dz, sx, sy, sz):
        q = self.ORDER + 1
        i = n % q
        j = (n // q) % q
        k = n // (q * q)
        h = _symmetric(sx[i], 0.0, 0.0, 0.0, 0.0, 0.0)
        if self.DIMENSION == 2:
            h = _symmetric(sx[i] * ly[j], lx[i] * sy[j], 0.0, dx[i] * dy[j], 0.0, 0.0)
        if self.DIMENSION == 3:
            h = _symmetric(sx[i] * ly[j] * lz[k], lx[i] * sy[j] * lz[k], lx[i] * ly[j] * sz[k],
                           dx[i] * dy[j] * lz[k], dx[i] * ly[j] * dz[k], lx[i] * dy[j] * dz[k])
        return h

    @tack.func
    def tensor_factors(self, pc, lx, ly, lz, dx, dy, dz, sx, sy, sz):
        """The 1D polynomials' values, derivatives and second derivatives at ``pc``."""
        self.lagrange_values(pc[0], lx)
        self.lagrange_values(pc[1], ly)
        self.lagrange_values(pc[2], lz)
        self.lagrange_derivatives(pc[0], dx)
        self.lagrange_derivatives(pc[1], dy)
        self.lagrange_derivatives(pc[2], dz)
        self.lagrange_second_derivatives(pc[0], sx)
        self.lagrange_second_derivatives(pc[1], sy)
        self.lagrange_second_derivatives(pc[2], sz)



class _BernsteinPolynomials:
    """The Bernstein basis of order ``ORDER`` on a tensor shape, in place of
    ``_TensorPolynomials``' Lagrange polynomials (mixed in before them): the 1D
    polynomial ``j`` is ``basis_weights[j] t^j (1 - t)^(p - j)``, ``basis_weights``
    the binomial coefficients, and a field's values are its coefficients. Its
    ``basis_nodes`` are the Greville points ``j / p``, which ``node_parametric``
    gives; its ``basis_bernstein`` is the identity, so its bounds are its
    coefficients'."""

    @tack.func
    def lagrange_values(self, t, out):
        """``out[j]``: Bernstein polynomial ``j`` at ``t``."""
        s = 1.0 - t
        for j in range(self.ORDER + 1):
            v = self.basis_weights[j]
            for _ in range(j):
                v = v * t
            for _ in range(self.ORDER - j):
                v = v * s
            out[j] = v

    @tack.func
    def lagrange_derivatives(self, t, out):
        """``out[j]``: its derivative at ``t``, ``C(p, j) (j t^(j-1) (1-t)^(p-j) - (p-j)
        t^j (1-t)^(p-j-1))``, each term a product (exact at the ends too)."""
        s = 1.0 - t
        p = self.ORDER
        for j in range(p + 1):
            up = self.basis_weights[j] - self.basis_weights[j]
            down = up
            if j > 0:
                up = self.basis_weights[j] * j
                for _ in range(j - 1):
                    up = up * t
                for _ in range(p - j):
                    up = up * s
            if j < p:
                down = self.basis_weights[j] * (p - j)
                for _ in range(j):
                    down = down * t
                for _ in range(p - j - 1):
                    down = down * s
            out[j] = up - down

    @tack.func
    def lagrange_second_derivatives(self, t, out):
        """``out[j]``: its second derivative, ``C(p, j)`` times ``j (j-1) t^(j-2)
        (1-t)^(p-j) - 2 j (p-j) t^(j-1) (1-t)^(p-j-1) + (p-j) (p-j-1) t^j
        (1-t)^(p-j-2)``, each term a product."""
        s = 1.0 - t
        p = self.ORDER
        for j in range(p + 1):
            zero = self.basis_weights[j] - self.basis_weights[j]
            a = zero
            b = zero
            c = zero
            if j > 1:
                a = self.basis_weights[j] * (j * (j - 1))
                for _ in range(j - 2):
                    a = a * t
                for _ in range(p - j):
                    a = a * s
            if j > 0 and j < p:
                b = self.basis_weights[j] * (2 * j * (p - j))
                for _ in range(j - 1):
                    b = b * t
                for _ in range(p - j - 1):
                    b = b * s
            if j < p - 1:
                c = self.basis_weights[j] * ((p - j) * (p - j - 1))
                for _ in range(j):
                    c = c * t
                for _ in range(p - j - 2):
                    c = c * s
            out[j] = a - b + c


class _TensorRange:
    """A tensor field's Bernstein coefficients over a cell: its values (less node 0's)
    taken one axis at a time by ``basis_bernstein``. The field lies between the
    least and the greatest (``value_range``); each one's distance from the corner
    coefficients' multilinear interpolation where it sits bounds what linearizing
    the cell at its corners loses (``flatness``)."""

    @tack.func
    def _bernstein(self, c, cur, nxt):
        """``cur``: the coefficients less node 0's value, in node order."""
        q = self.ORDER + 1
        count = (self.ORDER + 1) ** self.DIMENSION
        first = self.dof(c, 0)
        for n in range(count):
            cur[n] = self.dof(c, n) - first
        stride = 1
        for _ in range(self.DIMENSION):
            for n in range(count):
                i = (n // stride) % q
                base = n - i * stride
                total = cur[0] - cur[0]
                for m in range(q):
                    total += self.basis_bernstein[i * q + m] * cur[base + m * stride]
                nxt[n] = total
            for n in range(count):
                cur[n] = nxt[n]
            stride *= q

    @tack.func
    def value_range(self, c):
        cur = tack.local_array_like(self.basis_nodes, (self.ORDER + 1) ** self.DIMENSION)
        nxt = tack.local_array_like(self.basis_nodes, (self.ORDER + 1) ** self.DIMENSION)
        self._bernstein(c, cur, nxt)
        lo = cur[0]
        hi = lo
        for n in range(1, (self.ORDER + 1) ** self.DIMENSION):
            lo = min(lo, cur[n])
            hi = max(hi, cur[n])
        first = self.dof(c, 0)
        return first + lo, first + hi

    @tack.func
    def flatness(self, c):
        cur = tack.local_array_like(self.basis_nodes, (self.ORDER + 1) ** self.DIMENSION)
        nxt = tack.local_array_like(self.basis_nodes, (self.ORDER + 1) ** self.DIMENSION)
        self._bernstein(c, cur, nxt)
        p = self.ORDER
        q = self.ORDER + 1
        worst = cur[0] - cur[0]
        for n in range((self.ORDER + 1) ** self.DIMENSION):
            i = n % q
            j = (n // q) % q
            k = n // (q * q)
            # The corner coefficients' multilinear interpolation at (i, j, k) / p.
            linear = cur[0] - cur[0]
            for corner in range(2 ** self.DIMENSION):
                a = corner & 1
                b = (corner >> 1) & 1
                h = (corner >> 2) & 1
                w = (i if a == 1 else p - i) / p
                if self.DIMENSION >= 2:
                    w *= (j if b == 1 else p - j) / p
                if self.DIMENSION == 3:
                    w *= (k if h == 1 else p - k) / p
                linear += w * cur[a * p + q * (b * p + q * h * p)]
            worst = max(worst, abs(cur[n] - linear))
        return worst


class _TensorBasis(_TensorRange, _TensorPolynomials):
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
    def dofs_into(self, c, coeffs):
        """Cell ``c``'s values, in node order, into ``coeffs``: for work that evaluates
        a cell at many points (``value_from``), which then reads them once, not
        once a point (an order-6 ``H1`` value's ``dof_index`` per node was most of
        a subdivided contour's time)."""
        for n in range((self.ORDER + 1) ** self.DIMENSION):
            coeffs[n] = self.dof(c, n)

    @tack.func
    def value_from(self, coeffs, pc):
        """``value`` from a cell's values gathered by ``dofs_into``: the same sum in the
        same order, so the same bits."""
        lx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        ly = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        lz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        self.lagrange_values(pc[0], lx)
        self.lagrange_values(pc[1], ly)
        self.lagrange_values(pc[2], lz)
        total = coeffs[0] * self.node_weight(0, lx, ly, lz)
        for n in range(1, (self.ORDER + 1) ** self.DIMENSION):
            total += coeffs[n] * self.node_weight(n, lx, ly, lz)
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

    @tack.func
    def parametric_hessian(self, c, pc):
        """The second derivatives by parametric coordinates, from differences to node 0's
        value as the gradient."""
        lx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        ly = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        lz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dy = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        sx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        sy = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        sz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        self.tensor_factors(pc, lx, ly, lz, dx, dy, dz, sx, sy, sz)
        first = self.dof(c, 0)
        h = (self.dof(c, 1) - first) * self.node_hessian(1, lx, ly, lz, dx, dy, dz, sx, sy, sz)
        for n in range(2, (self.ORDER + 1) ** self.DIMENSION):
            h += (self.dof(c, n) - first) * self.node_hessian(n, lx, ly, lz, dx, dy, dz, sx, sy, sz)
        return h

    @tack.func
    def parametric_jet(self, c, pc):
        """The value, parametric gradient and parametric Hessian at ``pc``, from one pass over
        the cell's values (each read once), relative to the first as the gradient's."""
        lx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        ly = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        lz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dy = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        sx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        sy = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        sz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        self.tensor_factors(pc, lx, ly, lz, dx, dy, dz, sx, sy, sz)
        value = pc[0] - pc[0]
        grad = tack.vec3(0.0, 0.0, 0.0)
        hess = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        first = value
        for n in range((self.ORDER + 1) ** self.DIMENSION):
            v = self.dof(c, n)
            if n == 0:
                first = v
            d = v - first
            value += d * self.node_weight(n, lx, ly, lz)
            grad += d * self.node_gradient(n, lx, ly, lz, dx, dy, dz)
            hess += d * self.node_hessian(n, lx, ly, lz, dx, dy, dz, sx, sy, sz)
        return first + value, grad, hess


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


class _TensorGeometry(_TensorPolynomials, _Geometry):
    """A curved tensor geometry's positions, Jacobians and control box from its nodes'
    values, read through ``dof_index(c, n)`` (node ``n`` lexicographic): an ``H1``
    field of positions (``_H1TensorGeometry``) or an ``L2`` one, each cell its own
    (``_L2TensorGeometry``). Positions and Jacobians are formed from differences to
    node 0's position, as order 1's are, so f32 keeps thin cells."""

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

    @tack.func
    def geometry_curvature(self, c, pc, g):
        lx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        ly = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        lz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dy = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        dz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        sx = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        sy = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        sz = tack.local_array_like(self.basis_nodes, self.ORDER + 1)
        self.tensor_factors(pc, lx, ly, lz, dx, dy, dz, sx, sy, sz)
        first = self.get(self.dof_index(c, 0))
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for n in range(1, (self.ORDER + 1) ** self.DIMENSION):
            m += (self.get(self.dof_index(c, n)) - first).dot(g) * self.node_hessian(
                n, lx, ly, lz, dx, dy, dz, sx, sy, sz)
        return m


class _H1TensorGeometry(_TensorDofs, _TensorGeometry):
    """Curved geometry of order 2 or more on tensor shapes: an ``H1`` field of positions
    of that order, its corners the points'."""

    @tack.func
    def corner_position(self, c, j):
        return self.get(self.point_id(c, j))


class _L2TensorGeometry(_TensorGeometry):
    """A curved geometry in ``L2`` on tensor shapes (a mesh displaced by a DG field, a
    periodic mesh of order 2 or more): each cell's own ``(ORDER + 1)^d`` positions
    from ``point_offsets[cell]`` (the space's offsets, named as the geometry's are
    beside the domain view's); its corners where its map puts them."""

    @tack.func
    def dof_index(self, c, n):
        return self.point_offsets[self.entity_id(c)] + n

    @tack.func
    def corner_position(self, c, j):
        return self.position(c, self.parametric_point(j))


class _L2CellsTensorGeometry(_L2TensorGeometry):
    """``_L2TensorGeometry`` laid out cell-fastest: value ``n`` of cell ``c`` at ``n *
    layout_cells + c``."""

    @tack.func
    def dof_index(self, c, n):
        return n * self.layout_cells + self.entity_id(c)


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
    ``(ORDER + 1)^d`` values from ``offsets[cell]``, in node order (layout
    ``"blocks"``)."""

    @tack.func
    def dof_index(self, c, n):
        return self.offsets[self.entity_id(c)] + n

    @tack.func
    def dof(self, c, n):
        return self.get(self.dof_index(c, n))

    @tack.func
    def corner_value(self, c, j):
        return self.value(c, self.parametric_point(j))


class _L2CellsField(_TensorBasis):
    """L2 on a tensor basis, cell-fastest (layout ``"cells"``): value ``n`` of cell
    ``c`` at ``n * layout_cells + c``, so threads on neighbouring cells read
    neighbouring values."""

    @tack.func
    def dof_index(self, c, n):
        return n * self.layout_cells + self.entity_id(c)

    @tack.func
    def dof(self, c, n):
        return self.get(self.dof_index(c, n))

    @tack.func
    def corner_value(self, c, j):
        return self.value(c, self.parametric_point(j))


# ── Simplices of any order (equispaced nodes) ───────────────────────


class _SimplexPolynomials:
    """The Lagrange basis of order ``ORDER`` on a simplex's equispaced nodes
    (``bases.simplex_lattice``, ``NODES`` of them, ``simplex_ijk`` their integer
    coordinates), by Silvester's product formula: node ``alpha`` (its barycentric
    multi-index, ``(p - i - j - k, i, j, k)``) has the function ``prod_m
    F[alpha_m](lambda_m)``, ``F[a](t) = prod_{s < a} (p t - s) / (s + 1)``."""

    @tack.func
    def simplex_factors(self, pc, f, g):
        """``f[m * (p + 1) + a]``: ``F[a]`` at barycentric coordinate ``m``; ``g`` its
        derivative by that coordinate."""
        q = self.ORDER + 1
        for m in range(self.DIMENSION + 1):
            t = 1.0 - pc[0] - pc[1] - pc[2] if m == 0 else pc[m - 1]
            # In t's type throughout: an integer quotient would be f32.
            order = t - t + self.ORDER
            f[m * q] = t - t + 1.0
            g[m * q] = t - t
            for a in range(1, q):
                step = (order * t - (a - 1)) / a
                g[m * q + a] = g[m * q + a - 1] * step + f[m * q + a - 1] * (order / a)
                f[m * q + a] = f[m * q + a - 1] * step

    @tack.func
    def simplex_alpha(self, n, m):
        """Node ``n``'s barycentric index ``m``."""
        a = self.ORDER
        for k in range(self.DIMENSION):
            a -= self.simplex_ijk[n * self.DIMENSION + k]
        if m > 0:
            a = self.simplex_ijk[n * self.DIMENSION + m - 1]
        return a

    @tack.func
    def node_parametric(self, n):
        """Node ``n``'s parametric coordinates, ``(i, j, k) / ORDER`` in the basis' float
        type."""
        zero = self.simplex_bernstein[0] - self.simplex_bernstein[0]
        pc = tack.vec3(0.0, 0.0, 0.0)
        for k in range(self.DIMENSION):
            pc[k] = (zero + self.simplex_ijk[n * self.DIMENSION + k]) / self.ORDER
        return pc

    @tack.func
    def simplex_weight(self, n, f):
        q = self.ORDER + 1
        w = f[self.simplex_alpha(n, 0)]
        for m in range(1, self.DIMENSION + 1):
            w = w * f[m * q + self.simplex_alpha(n, m)]
        return w

    @tack.func
    def simplex_gradient(self, n, f, g):
        """Node ``n``'s function's parametric gradient: by the product rule, through
        ``lambda_0 = 1 - sum(pc)``."""
        q = self.ORDER + 1
        a0 = self.simplex_alpha(n, 0)
        grad = tack.vec3(0.0, 0.0, 0.0)
        for k in range(self.DIMENSION):
            # d/d pc_k: the factor of lambda_{k+1} differentiated, less lambda_0's.
            up = g[(k + 1) * q + self.simplex_alpha(n, k + 1)] * f[a0]
            down = g[a0] * f[(k + 1) * q + self.simplex_alpha(n, k + 1)]
            for m in range(1, self.DIMENSION + 1):
                if m != k + 1:
                    other = f[m * q + self.simplex_alpha(n, m)]
                    up = up * other
                    down = down * other
            grad[k] = up - down
        return grad

    @tack.func
    def simplex_factors2(self, pc, f, g, h):
        """``simplex_factors``, and ``h`` the factors' second derivatives."""
        q = self.ORDER + 1
        for m in range(self.DIMENSION + 1):
            t = 1.0 - pc[0] - pc[1] - pc[2] if m == 0 else pc[m - 1]
            order = t - t + self.ORDER
            f[m * q] = t - t + 1.0
            g[m * q] = t - t
            h[m * q] = t - t
            for a in range(1, q):
                step = (order * t - (a - 1)) / a
                h[m * q + a] = h[m * q + a - 1] * step + 2.0 * g[m * q + a - 1] * (order / a)
                g[m * q + a] = g[m * q + a - 1] * step + f[m * q + a - 1] * (order / a)
                f[m * q + a] = f[m * q + a - 1] * step

    @tack.func
    def _simplex_mixed(self, n, f, g, h, i, j):
        """Node ``n``'s product differentiated by barycentric coordinates ``i`` and ``j``."""
        q = self.ORDER + 1
        v = f[0] - f[0] + 1.0
        for m in range(self.DIMENSION + 1):
            k = m * q + self.simplex_alpha(n, m)
            factor = f[k]
            if m == i or m == j:
                factor = g[k]
            if m == i and m == j:
                factor = h[k]
            v = v * factor
        return v

    @tack.func
    def simplex_hessian(self, n, f, g, h):
        """Node ``n``'s function's second derivatives by parametric coordinates, through
        ``lambda_0 = 1 - sum(pc)``: ``d2/dpc_k dpc_l`` is the mixed derivative by
        ``lambda_{k+1}`` and ``lambda_{l+1}``, less both with ``lambda_0``, plus
        ``lambda_0``'s second."""
        both = self._simplex_mixed(n, f, g, h, 0, 0)
        out = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for k in range(self.DIMENSION):
            ek = tack.vec3(1.0 if k == 0 else 0.0, 1.0 if k == 1 else 0.0, 1.0 if k == 2 else 0.0)
            down = self._simplex_mixed(n, f, g, h, k + 1, 0)
            for l in range(self.DIMENSION):
                el = tack.vec3(1.0 if l == 0 else 0.0, 1.0 if l == 1 else 0.0, 1.0 if l == 2 else 0.0)
                v = (self._simplex_mixed(n, f, g, h, k + 1, l + 1) - down
                     - self._simplex_mixed(n, f, g, h, 0, l + 1) + both)
                out += v * ek.outer_product(el)
        return out


class _SimplexBernstein:
    """The Bernstein basis of order ``ORDER`` on a simplex, in place of
    ``_SimplexPolynomials``' Lagrange factors (mixed in before them): node ``alpha``'s
    function is ``p! / alpha! * prod_m lambda_m^alpha_m``, the factors ``t^a / a!``,
    corner 0's scaled by ``p!``. Its nodes, ``alpha / p``, are its Greville points."""

    @tack.func
    def simplex_factors(self, pc, f, g):
        q = self.ORDER + 1
        for m in range(self.DIMENSION + 1):
            t = 1.0 - pc[0] - pc[1] - pc[2] if m == 0 else pc[m - 1]
            scale = t - t + 1.0
            if m == 0:
                for s in range(2, q):
                    scale = scale * s
            f[m * q] = scale
            g[m * q] = t - t
            for a in range(1, q):
                g[m * q + a] = (g[m * q + a - 1] * t + f[m * q + a - 1]) / a
                f[m * q + a] = f[m * q + a - 1] * t / a

    @tack.func
    def simplex_factors2(self, pc, f, g, h):
        q = self.ORDER + 1
        for m in range(self.DIMENSION + 1):
            t = 1.0 - pc[0] - pc[1] - pc[2] if m == 0 else pc[m - 1]
            scale = t - t + 1.0
            if m == 0:
                for s in range(2, q):
                    scale = scale * s
            f[m * q] = scale
            g[m * q] = t - t
            h[m * q] = t - t
            for a in range(1, q):
                h[m * q + a] = (h[m * q + a - 1] * t + 2.0 * g[m * q + a - 1]) / a
                g[m * q + a] = (g[m * q + a - 1] * t + f[m * q + a - 1]) / a
                f[m * q + a] = f[m * q + a - 1] * t / a


class _SimplexBasis(_SimplexPolynomials):
    """A field's value, parametric gradient and Bernstein bounds on a simplex of any
    order, from ``dof(c, n)``, the space's."""

    @tack.func
    def value(self, c, pc):
        f = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        g = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        self.simplex_factors(pc, f, g)
        total = self.dof(c, 0) * self.simplex_weight(0, f)
        for n in range(1, self.NODES):
            total += self.dof(c, n) * self.simplex_weight(n, f)
        return total

    @tack.func
    def parametric_gradient(self, c, pc):
        """From differences to node 0's value (the basis' gradients sum to zero)."""
        f = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        g = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        self.simplex_factors(pc, f, g)
        first = self.dof(c, 0)
        grad = (self.dof(c, 1) - first) * self.simplex_gradient(1, f, g)
        for n in range(2, self.NODES):
            grad += (self.dof(c, n) - first) * self.simplex_gradient(n, f, g)
        return grad

    @tack.func
    def parametric_hessian(self, c, pc):
        f = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        g = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        h = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        self.simplex_factors2(pc, f, g, h)
        first = self.dof(c, 0)
        out = (self.dof(c, 1) - first) * self.simplex_hessian(1, f, g, h)
        for n in range(2, self.NODES):
            out += (self.dof(c, n) - first) * self.simplex_hessian(n, f, g, h)
        return out

    @tack.func
    def parametric_jet(self, c, pc):
        """The value, parametric gradient and parametric Hessian at ``pc``, from one pass over
        the cell's values (each read once), relative to the first as the gradient's."""
        f = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        g = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        h = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        self.simplex_factors2(pc, f, g, h)
        value = pc[0] - pc[0]
        grad = tack.vec3(0.0, 0.0, 0.0)
        hess = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        first = value
        for n in range(self.NODES):
            v = self.dof(c, n)
            if n == 0:
                first = v
            d = v - first
            value += d * self.simplex_weight(n, f)
            grad += d * self.simplex_gradient(n, f, g)
            hess += d * self.simplex_hessian(n, f, g, h)
        return first + value, grad, hess

    @tack.func
    def corner_value(self, c, j):
        return self.dof(c, self.corner_node[j])

    @tack.func
    def _simplex_bernstein(self, c, out):
        """``out``: the cell's Bernstein coefficients less node 0's value."""
        first = self.dof(c, 0)
        for i in range(self.NODES):
            total = first - first
            for n in range(1, self.NODES):
                total += self.simplex_bernstein[i * self.NODES + n] * (self.dof(c, n) - first)
            out[i] = total

    @tack.func
    def value_range(self, c):
        coeff = tack.local_array_like(self.simplex_bernstein, self.NODES)
        self._simplex_bernstein(c, coeff)
        lo = coeff[0]
        hi = lo
        for i in range(1, self.NODES):
            lo = min(lo, coeff[i])
            hi = max(hi, coeff[i])
        first = self.dof(c, 0)
        return first + lo, first + hi

    @tack.func
    def flatness(self, c):
        """The greatest distance of a Bernstein coefficient from the corner
        coefficients' linear interpolation where it sits."""
        coeff = tack.local_array_like(self.simplex_bernstein, self.NODES)
        self._simplex_bernstein(c, coeff)
        worst = coeff[0] - coeff[0]
        for n in range(self.NODES):
            linear = coeff[0] - coeff[0]
            for m in range(self.DIMENSION + 1):
                linear += coeff[self.corner_node[m]] * self.simplex_alpha(n, m) / self.ORDER
            worst = max(worst, abs(coeff[n] - linear))
        return worst


class _SimplexDofs:
    """Where an ``H1`` space of order ``ORDER`` (3 or more) on simplices keeps cell
    ``c``'s node ``n`` (``bases.dof_tables``): a corner's at its point; ``ORDER - 1``
    per edge (``_TensorEdges``, from the edge's lower point id); ``(ORDER - 1)
    (ORDER - 2) / 2`` per triangle face from ``face_dofs[face]``, in the frame of
    the face's own row; the interior's from ``cell_dofs[cell]``, in node order."""

    @tack.func
    def dof_index(self, c, n):
        kind = self.node_kind[n]
        e = self.node_entity[n]
        if kind == 0:
            return self.point_id(c, e)
        if kind == 1:
            return self.tensor_edge_dof(c, e, self.node_a[n])
        if kind == 2:
            return self.simplex_face_dof(c, e, self.node_a[n], self.node_b[n])
        return self.cell_dofs[self.entity_id(c)] + self.node_a[n]


class _SimplexFaces:
    """A tetrahedron's triangle face node ``(a, b)`` (steps from the cell's local
    corner 0 of the face toward its corners 1 and 2), placed in the frame of the
    face's own row -- the same for both cells sharing it -- through the face's
    orientation in the cell, and numbered there ``U`` fastest (``U, V >= 1``)."""

    @tack.func
    def simplex_face_dof(self, c, f, a, b):
        p = self.ORDER
        r0 = self.face_position(c, f, 0)
        r1 = self.face_position(c, f, 1)
        u = (p - a - b) if r0 == 1 else (a if r1 == 1 else b)
        v = (p - a - b) if r0 == 2 else (a if r1 == 2 else b)
        return (self.face_dofs[self.face_id(c, f)] + (v - 1) * (p - 1) - ((v - 1) * v) // 2
                + (u - 1))


class _NoSimplexFaces:
    """A triangle: its edges are its sides, and it has no face nodes."""

    @tack.func
    def simplex_face_dof(self, c, f, a, b):
        return self.cell_dofs[self.entity_id(c)]          # not reached


class _H1SimplexField(_SimplexDofs, _SimplexBasis):
    """H1 of order 3 or more on simplices, Shared."""

    @tack.func
    def dof(self, c, n):
        return self.get(self.dof_index(c, n))


class _L2SimplexField(_SimplexBasis):
    """L2 of order 2 or more on simplices: each cell's ``NODES`` values from
    ``offsets[cell]``, in node order."""

    @tack.func
    def dof_index(self, c, n):
        return self.offsets[self.entity_id(c)] + n

    @tack.func
    def dof(self, c, n):
        return self.get(self.dof_index(c, n))


class _SimplexGeometry(_SimplexPolynomials, _Geometry):
    """A curved geometry on simplices (equispaced, or Bernstein), from its nodes'
    positions wherever the space keeps them (``dof_index``). Positions and
    Jacobians from differences to node 0's position, as the tensor geometry's, so
    f32 keeps thin cells."""

    CURVED = 1

    @tack.func
    def control_box(self, c):
        """The box of the cell's Bernstein control points (``simplex_bernstein`` taking
        the nodes' positions, less node 0's, to them), which hold it."""
        first = self.get(self.dof_index(c, 0))
        a = first - first
        b = a
        for i in range(self.NODES):
            s = first - first
            for n in range(1, self.NODES):
                s += self.simplex_bernstein[i * self.NODES + n] * (
                    self.get(self.dof_index(c, n)) - first)
            a = min(a, s)
            b = max(b, s)
        return first + a, first + b

    @tack.func
    def position(self, c, pc):
        f = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        g = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        self.simplex_factors(pc, f, g)
        first = self.get(self.dof_index(c, 0))
        x = first
        for n in range(1, self.NODES):
            x += (self.get(self.dof_index(c, n)) - first) * self.simplex_weight(n, f)
        return x

    @tack.func
    def geometry_jacobian(self, c, pc):
        f = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        g = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        self.simplex_factors(pc, f, g)
        first = self.get(self.dof_index(c, 0))
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for n in range(1, self.NODES):
            m += (self.get(self.dof_index(c, n)) - first).outer_product(
                self.simplex_gradient(n, f, g))
        return m

    @tack.func
    def geometry_curvature(self, c, pc, gw):
        f = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        g = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        h = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        self.simplex_factors2(pc, f, g, h)
        first = self.get(self.dof_index(c, 0))
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for n in range(1, self.NODES):
            m += (self.get(self.dof_index(c, n)) - first).dot(gw) * self.simplex_hessian(n, f, g, h)
        return m


class _H1SimplexGeometry(_SimplexDofs, _SimplexGeometry):
    """Curved geometry of order 3 or more on simplices: an ``H1`` field of positions of
    that order, its corners the points'."""

    @tack.func
    def corner_position(self, c, j):
        return self.get(self.point_id(c, j))


class _L2SimplexGeometry(_SimplexGeometry):
    """A curved geometry in ``L2`` on simplices: each cell's own ``NODES`` positions
    from ``point_offsets[cell]``; its corners where its map puts them."""

    @tack.func
    def dof_index(self, c, n):
        return self.point_offsets[self.entity_id(c)] + n

    @tack.func
    def corner_position(self, c, j):
        return self.position(c, self.parametric_point(j))


class _SimplexNewton:
    """``_CurvedNewton`` for a curved simplex geometry (``_H1SimplexGeometry``): the
    cell's nodes read once, the map formed from them as ``position`` and
    ``geometry_jacobian`` form it."""

    @tack.func
    def curved_to_parametric(self, i, x):
        rx = tack.local_array_like(self.simplex_bernstein, self.NODES)
        ry = tack.local_array_like(self.simplex_bernstein, self.NODES)
        rz = tack.local_array_like(self.simplex_bernstein, self.NODES)
        first = self.get(self.dof_index(i, 0))
        for n in range(1, self.NODES):
            d = self.get(self.dof_index(i, n)) - first
            rx[n] = d[0]
            ry[n] = d[1]
            rz[n] = d[2]
        return _newton_from(self, x, rx, ry, rz, first)

    @tack.func
    def gathered_map(self, rx, ry, rz, first, pc):
        f = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        g = tack.local_array_like(self.simplex_bernstein, (self.DIMENSION + 1) * (self.ORDER + 1))
        self.simplex_factors(pc, f, g)
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        at = first
        for n in range(1, self.NODES):
            d = tack.vec3(rx[n], ry[n], rz[n])
            at += d * self.simplex_weight(n, f)
            m += d.outer_product(self.simplex_gradient(n, f, g))
        return m, at


# ── Wedges of any order: a triangle times a segment (equispaced) ────


class _WedgeBernstein:
    """The wedge's Bernstein basis, in place of ``_WedgePolynomials``' Lagrange factors:
    the triangle's ``p! / (a! b! c!) x^a y^b z^c`` times the segment's ``p! / (k! (p -
    k)!) ...``: the factors ``t^a / a!``, the triangle's first and the segment's
    first scaled by ``p!``."""

    @tack.func
    def wedge_factors(self, pc, f, g):
        q = self.ORDER + 1
        for m in range(5):
            t = 1.0 - pc[0] - pc[1]
            if m == 1:
                t = pc[0]
            elif m == 2:
                t = pc[1]
            elif m == 3:
                t = 1.0 - pc[2]
            elif m == 4:
                t = pc[2]
            scale = t - t + 1.0
            if m == 0 or m == 3:
                for s in range(2, q):
                    scale = scale * s
            f[m * q] = scale
            g[m * q] = t - t
            for a in range(1, q):
                g[m * q + a] = (g[m * q + a - 1] * t + f[m * q + a - 1]) / a
                f[m * q + a] = f[m * q + a - 1] * t / a


class _WedgePolynomials:
    """The wedge's Lagrange basis of order ``ORDER`` (``bases.wedge_lattice``: node ``(i, j,
    k)``, ``wedge_ijk``): Silvester's product over five barycentric coordinates, the
    triangle's ``1 - x - y``, ``x``, ``y`` with indices ``(p - i - j, i, j)`` and the
    segment's ``1 - z``, ``z`` with ``(p - k, k)``."""

    @tack.func
    def wedge_factors(self, pc, f, g):
        """``f[m * (p + 1) + a]``: ``F[a]`` at barycentric coordinate ``m`` (0-2 the
        triangle's, 3-4 the segment's); ``g`` its derivative by it."""
        q = self.ORDER + 1
        for m in range(5):
            t = 1.0 - pc[0] - pc[1]
            if m == 1:
                t = pc[0]
            elif m == 2:
                t = pc[1]
            elif m == 3:
                t = 1.0 - pc[2]
            elif m == 4:
                t = pc[2]
            order = t - t + self.ORDER
            f[m * q] = t - t + 1.0
            g[m * q] = t - t
            for a in range(1, q):
                step = (order * t - (a - 1)) / a
                g[m * q + a] = g[m * q + a - 1] * step + f[m * q + a - 1] * (order / a)
                f[m * q + a] = f[m * q + a - 1] * step

    @tack.func
    def wedge_alpha(self, n, m):
        """Node ``n``'s barycentric index ``m`` (0-4)."""
        i = self.wedge_ijk[3 * n]
        j = self.wedge_ijk[3 * n + 1]
        k = self.wedge_ijk[3 * n + 2]
        a = self.ORDER - i - j
        if m == 1:
            a = i
        elif m == 2:
            a = j
        elif m == 3:
            a = self.ORDER - k
        elif m == 4:
            a = k
        return a

    @tack.func
    def wedge_weight(self, n, f):
        q = self.ORDER + 1
        w = f[self.wedge_alpha(n, 0)]
        for m in range(1, 5):
            w = w * f[m * q + self.wedge_alpha(n, m)]
        return w

    @tack.func
    def wedge_gradient(self, n, f, g):
        q = self.ORDER + 1
        f0 = f[self.wedge_alpha(n, 0)]
        f1 = f[q + self.wedge_alpha(n, 1)]
        f2 = f[2 * q + self.wedge_alpha(n, 2)]
        f3 = f[3 * q + self.wedge_alpha(n, 3)]
        f4 = f[4 * q + self.wedge_alpha(n, 4)]
        g0 = g[self.wedge_alpha(n, 0)]
        g1 = g[q + self.wedge_alpha(n, 1)]
        g2 = g[2 * q + self.wedge_alpha(n, 2)]
        g3 = g[3 * q + self.wedge_alpha(n, 3)]
        g4 = g[4 * q + self.wedge_alpha(n, 4)]
        return tack.vec3((g1 * f0 - g0 * f1) * f2 * f3 * f4, (g2 * f0 - g0 * f2) * f1 * f3 * f4,
                         (g4 * f3 - g3 * f4) * f0 * f1 * f2)

    @tack.func
    def node_parametric(self, n):
        zero = self.wedge_bernstein[0] - self.wedge_bernstein[0]
        return tack.vec3((zero + self.wedge_ijk[3 * n]) / self.ORDER,
                         (zero + self.wedge_ijk[3 * n + 1]) / self.ORDER,
                         (zero + self.wedge_ijk[3 * n + 2]) / self.ORDER)


class _WedgeBasis(_WedgePolynomials):
    """A field's value, parametric gradient and Bernstein bounds on a wedge of any order,
    from ``dof(c, n)``, the space's."""

    @tack.func
    def value(self, c, pc):
        f = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        g = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        self.wedge_factors(pc, f, g)
        total = self.dof(c, 0) * self.wedge_weight(0, f)
        for n in range(1, self.NODES):
            total += self.dof(c, n) * self.wedge_weight(n, f)
        return total

    @tack.func
    def parametric_gradient(self, c, pc):
        f = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        g = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        self.wedge_factors(pc, f, g)
        first = self.dof(c, 0)
        grad = (self.dof(c, 1) - first) * self.wedge_gradient(1, f, g)
        for n in range(2, self.NODES):
            grad += (self.dof(c, n) - first) * self.wedge_gradient(n, f, g)
        return grad

    @tack.func
    def parametric_hessian(self, c, pc):
        """The second derivatives by parametric coordinates, a symmetric 3x3 matrix: the
        parametric gradient by fourth-order central differences (steps of 1/4),
        exact for these polynomials up to order 4."""
        h = 0.25
        d = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for k in range(3):
            e = tack.vec3(1.0 if k == 0 else 0.0, 1.0 if k == 1 else 0.0, 1.0 if k == 2 else 0.0)
            dk = tack.vec3(0.0, 0.0, 0.0)
            for s in range(4):
                offset = -2.0 if s == 0 else (-1.0 if s == 1 else (1.0 if s == 2 else 2.0))
                weight = 1.0 if s == 0 else (-8.0 if s == 1 else (8.0 if s == 2 else -1.0))
                dk += weight * self.parametric_gradient(c, pc + (offset * h) * e)
            d += (dk / (12.0 * h)).outer_product(e)
        return 0.5 * (d + d.transpose())

    @tack.func
    def parametric_jet(self, c, pc):
        return self.value(c, pc), self.parametric_gradient(c, pc), self.parametric_hessian(c, pc)

    @tack.func
    def corner_value(self, c, j):
        return self.dof(c, self.corner_node[j])

    @tack.func
    def _wedge_bernstein(self, c, out):
        first = self.dof(c, 0)
        for i in range(self.NODES):
            total = first - first
            for n in range(1, self.NODES):
                total += self.wedge_bernstein[i * self.NODES + n] * (self.dof(c, n) - first)
            out[i] = total

    @tack.func
    def value_range(self, c):
        coeff = tack.local_array_like(self.wedge_bernstein, self.NODES)
        self._wedge_bernstein(c, coeff)
        lo = coeff[0]
        hi = lo
        for i in range(1, self.NODES):
            lo = min(lo, coeff[i])
            hi = max(hi, coeff[i])
        first = self.dof(c, 0)
        return first + lo, first + hi

    @tack.func
    def flatness(self, c):
        """The greatest distance of a Bernstein coefficient from the corner coefficients'
        interpolation (the wedge's linear functions) where it sits."""
        coeff = tack.local_array_like(self.wedge_bernstein, self.NODES)
        self._wedge_bernstein(c, coeff)
        worst = coeff[0] - coeff[0]
        for n in range(self.NODES):
            linear = coeff[0] - coeff[0]
            for m in range(6):
                layer = self.wedge_alpha(n, 4) if m >= 3 else self.wedge_alpha(n, 3)
                linear += (coeff[self.corner_node[m]] * self.wedge_alpha(n, m % 3) * layer
                           / (self.ORDER * self.ORDER))
            worst = max(worst, abs(coeff[n] - linear))
        return worst


class _WedgeDofs:
    """Where an ``H1`` space of order ``ORDER`` (3 or more) on wedges keeps cell ``c``'s
    node ``n``: as on simplices, a triangle face's through ``_SimplexFaces``, a quad
    face's through ``_TensorFaces``."""

    @tack.func
    def dof_index(self, c, n):
        kind = self.node_kind[n]
        e = self.node_entity[n]
        if kind == 0:
            return self.point_id(c, e)
        if kind == 1:
            return self.tensor_edge_dof(c, e, self.node_a[n])
        if kind == 2:
            if self.face_num_points(e) == 3:
                return self.simplex_face_dof(c, e, self.node_a[n], self.node_b[n])
            return self.tensor_face_dof(c, e, self.node_a[n], self.node_b[n])
        return self.cell_dofs[self.entity_id(c)] + self.node_a[n]


class _H1WedgeField(_WedgeDofs, _WedgeBasis):
    """H1 of order 3 or more on wedges, Shared."""

    @tack.func
    def dof(self, c, n):
        return self.get(self.dof_index(c, n))


class _L2WedgeField(_WedgeBasis):
    """L2 of order 2 or more on wedges: each cell's ``NODES`` values from
    ``offsets[cell]``, in node order."""

    @tack.func
    def dof_index(self, c, n):
        return self.offsets[self.entity_id(c)] + n

    @tack.func
    def dof(self, c, n):
        return self.get(self.dof_index(c, n))


class _WedgeGeometry(_WedgePolynomials, _Geometry):
    """A curved geometry on wedges, from differences to node 0, its nodes wherever
    the space keeps them (``dof_index``)."""

    CURVED = 1

    @tack.func
    def control_box(self, c):
        first = self.get(self.dof_index(c, 0))
        a = first - first
        b = a
        for i in range(self.NODES):
            s = first - first
            for n in range(1, self.NODES):
                s += self.wedge_bernstein[i * self.NODES + n] * (
                    self.get(self.dof_index(c, n)) - first)
            a = min(a, s)
            b = max(b, s)
        return first + a, first + b

    @tack.func
    def position(self, c, pc):
        f = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        g = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        self.wedge_factors(pc, f, g)
        first = self.get(self.dof_index(c, 0))
        x = first
        for n in range(1, self.NODES):
            x += (self.get(self.dof_index(c, n)) - first) * self.wedge_weight(n, f)
        return x

    @tack.func
    def geometry_jacobian(self, c, pc):
        f = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        g = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        self.wedge_factors(pc, f, g)
        first = self.get(self.dof_index(c, 0))
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for n in range(1, self.NODES):
            m += (self.get(self.dof_index(c, n)) - first).outer_product(
                self.wedge_gradient(n, f, g))
        return m

    @tack.func
    def geometry_curvature(self, c, pc, g):
        """``g . d2x/d(pc)2``, as the other geometries': the Jacobian's transpose times
        ``g`` by fourth-order central differences (steps of 1/4)."""
        h = 0.25
        d = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for k in range(3):
            e = tack.vec3(1.0 if k == 0 else 0.0, 1.0 if k == 1 else 0.0, 1.0 if k == 2 else 0.0)
            dk = tack.vec3(0.0, 0.0, 0.0)
            for s in range(4):
                offset = -2.0 if s == 0 else (-1.0 if s == 1 else (1.0 if s == 2 else 2.0))
                weight = 1.0 if s == 0 else (-8.0 if s == 1 else (8.0 if s == 2 else -1.0))
                dk += weight * (self.geometry_jacobian(c, pc + (offset * h) * e).transpose() @ g)
            d += (dk / (12.0 * h)).outer_product(e)
        return 0.5 * (d + d.transpose())


class _H1WedgeGeometry(_WedgeDofs, _WedgeGeometry):
    """Curved geometry of order 3 or more on wedges, its corners the points'."""

    @tack.func
    def corner_position(self, c, j):
        return self.get(self.point_id(c, j))


class _L2WedgeGeometry(_WedgeGeometry):
    """A curved geometry in ``L2`` on wedges: each cell's own ``NODES`` positions from
    ``point_offsets[cell]``; its corners where its map puts them."""

    @tack.func
    def dof_index(self, c, n):
        return self.point_offsets[self.entity_id(c)] + n

    @tack.func
    def corner_position(self, c, j):
        return self.position(c, self.parametric_point(j))


class _WedgeNewton:
    """``_CurvedNewton`` for a curved wedge geometry: its nodes read once."""

    @tack.func
    def curved_to_parametric(self, i, x):
        rx = tack.local_array_like(self.wedge_bernstein, self.NODES)
        ry = tack.local_array_like(self.wedge_bernstein, self.NODES)
        rz = tack.local_array_like(self.wedge_bernstein, self.NODES)
        first = self.get(self.dof_index(i, 0))
        for n in range(1, self.NODES):
            d = self.get(self.dof_index(i, n)) - first
            rx[n] = d[0]
            ry[n] = d[1]
            rz[n] = d[2]
        return _newton_from(self, x, rx, ry, rz, first)

    @tack.func
    def gathered_map(self, rx, ry, rz, first, pc):
        f = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        g = tack.local_array_like(self.wedge_bernstein, 5 * (self.ORDER + 1))
        self.wedge_factors(pc, f, g)
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        at = first
        for n in range(1, self.NODES):
            d = tack.vec3(rx[n], ry[n], rz[n])
            at += d * self.wedge_weight(n, f)
            m += d.outer_product(self.wedge_gradient(n, f, g))
        return m, at


class _TableInteriorNodes:
    """``H1`` of order ``NODE_ORDER`` (3 or more): a cell's interior nodes from a table of their
    lattice coordinates (``node_interior``, three a node), where its geometry maps
    them (wedges)."""

    @tack.func
    def node_count(self, i):
        return self.node_interior.shape[0] // 3

    @tack.func
    def node_dof(self, i, j):
        return self.node_cell_dofs[self.entity_id(i)] + j

    @tack.func
    def node_position(self, i, j):
        pc = tack.vec3(self.node_points[self.node_interior[3 * j]],
                       self.node_points[self.node_interior[3 * j + 1]],
                       self.node_points[self.node_interior[3 * j + 2]])
        return self.position(i, pc)


def wedge_mixins(shape):
    """The edge and face mixins a wedge ``H1`` view needs: both face kinds'."""
    return _TensorEdges, _SimplexFaces, _TensorFaces


def of_wedge_order(mixin, order, bernstein=False):
    """``mixin`` with ``ORDER`` and ``NODES`` (the wedge's node count) set; ``bernstein``
    the Bernstein basis (``_WedgeBernstein``)."""
    from math import comb

    key = (mixin, int(order), "wedge", bool(bernstein))
    if key not in _by_order:
        parents = (_WedgeBernstein, mixin) if bernstein else (mixin,)
        _by_order[key] = type(f"{mixin.__name__}{order}w" + ("Bernstein" if bernstein else ""),
                              parents, {"ORDER": int(order),
                                        "NODES": comb(int(order) + 2, 2) * (int(order) + 1)})
    return _by_order[key]


def simplex_mixins(shape):
    """The edge and face mixins a simplex ``H1`` view of ``shape`` cells needs."""
    return _TensorEdges, (_SimplexFaces if shape.DIMENSION == 3 else _NoSimplexFaces)


def of_simplex_order(mixin, order, dimension, bernstein=False):
    """``mixin`` with ``ORDER`` and ``NODES`` (the simplex's node count) set;
    ``bernstein`` the Bernstein basis (``_SimplexBernstein``)."""
    from math import comb

    key = (mixin, int(order), int(dimension), bool(bernstein))
    if key not in _by_order:
        parents = (_SimplexBernstein, mixin) if bernstein else (mixin,)
        _by_order[key] = type(f"{mixin.__name__}{order}d{dimension}"
                              + ("Bernstein" if bernstein else ""), parents,
                              {"ORDER": int(order), "NODES": comb(int(order) + dimension,
                                                                  dimension)})
    return _by_order[key]


def of_node_order(mixin, order, count=None):
    """A node mixin (``node_views``) with ``NODE_ORDER`` set, and ``NODE_COUNT`` (the
    nodes a cell has) when given: names of their own, since a node view is the
    cell view of a geometry that may be curved and of another order, whose
    ``ORDER`` and ``NODES`` they would hide."""
    key = (mixin, int(order), "nodes", count)
    if key not in _by_order:
        attributes = {"NODE_ORDER": int(order)}
        if count is not None:
            attributes["NODE_COUNT"] = int(count)
        _by_order[key] = type(f"{mixin.__name__}{order}n{count or ''}", (mixin,), attributes)
    return _by_order[key]


def tensor_mixins(shape):
    """The edge and face mixins a tensor ``H1`` view of ``shape`` cells needs."""
    edges = _TensorEdges if shape.NUM_EDGES and shape.DIMENSION >= 2 else _NoTensorEdges
    faces = _TensorFaces if shape.DIMENSION == 3 else _NoTensorFaces
    return edges, faces


_by_order = {}


class _TableDofs:
    """An ``H1`` space numbered by an element-to-DOF table (``H1(..., dofs=table)``):
    cell ``c``'s node ``n``, in Tack's node order for its shape (``bases.lattice_ijk``),
    is value ``dof_table[c * dof_count + n]``. Mixed in before the space's view,
    it stands in for numbering by entity: each cell lists its shared values in its
    own local order, so no orientation is applied."""

    @tack.func
    def dof_index(self, c, n):
        return self.dof_table[self.entity_id(c) * self.dof_count + n]


def with_table(mixin):
    """``mixin`` read through an element-to-DOF table (``_TableDofs``)."""
    key = (mixin, "table")
    if key not in _by_order:
        _by_order[key] = type(f"{mixin.__name__}Table", (_TableDofs, mixin), {})
    return _by_order[key]


def of_order(mixin, order, bernstein=False):
    """``mixin`` with ``ORDER`` set: one class per (mixin, order), so views of one
    order share their compiled kernels. ``bernstein`` puts the Bernstein
    polynomials in place of the Lagrange ones (``_BernsteinPolynomials``)."""
    key = (mixin, int(order), bool(bernstein))
    if key not in _by_order:
        name = f"{mixin.__name__}{order}" + ("Bernstein" if bernstein else "")
        parents = (_BernsteinPolynomials, mixin) if bernstein else (mixin,)
        _by_order[key] = type(name, parents, {"ORDER": int(order)})
    return _by_order[key]


class _CellValue:
    """One value per cell, at the cell's id: what ``Constant`` reads on every shape."""

    NODES = 1

    @tack.func
    def dof_index(self, i, j):
        return self.entity_id(i)

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

    @tack.func
    def parametric_hessian(self, i, pc):
        return tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    @tack.func
    def parametric_jet(self, i, pc):
        return self.value(i, pc), tack.vec3(0.0, 0.0, 0.0), tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    @tack.func
    def value_range(self, i):
        v = self.dof(i, 0)                                # the cell's one value
        return v, v

    @tack.func
    def flatness(self, i):
        return abs(self.dof(i, 0) - self.dof(i, 0))


class _ConstantField(_CellValue):
    """Constant (L2, order 0): one value per cell, its one node at the cell's
    parametric center."""

    @tack.func
    def node_parametric(self, j):
        return self.parametric_center()


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


class _PolyhedralCellField(_CellValue):
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


@tack.func
def _square_corner(k):
    # Corner k of a quad's loop on the unit square.
    return (1 if k == 1 or k == 2 else 0), (1 if k == 2 or k == 3 else 0)


@tack.func
def _square_frame(view, i):
    # A quad's frame fixed by its corners' positions, the same in every piece
    # whatever its stored order (which follows the piece's point ids): its first
    # corner by position, the first by position of that corner's two neighbours
    # (the first axis), and the other (the second).
    a = 0
    for k in range(1, 4):
        if _before(view.corner_position(i, k), view.corner_position(i, a)):
            a = k
    b = (a + 1) % 4
    d = (a + 3) % 4
    if _before(view.corner_position(i, d), view.corner_position(i, b)):
        b, d = d, b
    return a, b, d


class _MeanPosition:
    """A shared node of a straight (order-1) geometry: the mean of its entity's
    points -- a point itself, an edge's middle, the bilinear center of a quad,
    the trilinear center of a hexahedron. A quad's points are summed in its frame
    (``_square_frame``), so every piece holding it gets the same bits."""

    @tack.func
    def node_position(self, i, j):
        if self.NUM_POINTS == 4:
            a, b, d = _square_frame(self, i)
            return (((self.corner_position(i, a) + self.corner_position(i, b))
                     + self.corner_position(i, (a + 2) % 4)) + self.corner_position(i, d)) / 4
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

    @tack.func
    def node_parametric_point(self, i, j):
        return self.parametric_point(j)


class _CenterPosition:
    """A cell's one node: where the geometry maps the parametric center."""

    @tack.func
    def node_position(self, i, j):
        return self.position(i, self.parametric_center())

    @tack.func
    def node_parametric_point(self, i, j):
        return self.parametric_center()


class _TensorEdgeNodes:
    """``H1`` of order ``NODE_ORDER`` (3 or more): an edge's ``NODE_ORDER - 1`` nodes, in order from
    its lower point id, at the space's 1D nodes (``node_points``) along it -- the
    edge's straight line between its points, or the geometry's own value there
    (``_GeometryAtDof``) when the geometry shares the space's layout."""

    @tack.func
    def node_count(self, i):
        return self.NODE_ORDER - 1

    @tack.func
    def node_dof(self, i, j):
        return self.node_edge_base + self.entity_id(i) * (self.NODE_ORDER - 1) + j

    @tack.func
    def node_place(self, i, j):
        return tack.vec3(self.node_points[j + 1], 0.0, 0.0)

    @tack.func
    def node_steps(self, i, j):
        # The node's place in steps of the 1D nodes, from the edge's point 0.
        return j + 1, 0


class _TensorFaceNodes:
    """``H1`` of order ``NODE_ORDER`` (3 or more): a quad face's ``(NODE_ORDER - 1)^2`` nodes,
    lexicographic in the frame of its own row (from its point 0, toward its points
    1 and 3)."""

    @tack.func
    def node_count(self, i):
        return (self.NODE_ORDER - 1) * (self.NODE_ORDER - 1)

    @tack.func
    def node_dof(self, i, j):
        return self.node_face_dofs[self.entity_id(i)] + j

    @tack.func
    def node_place(self, i, j):
        return tack.vec3(self.node_points[j % (self.NODE_ORDER - 1) + 1],
                         self.node_points[j // (self.NODE_ORDER - 1) + 1], 0.0)

    @tack.func
    def node_steps(self, i, j):
        # The node's place in steps of the 1D nodes, from the face's point 0.
        return j % (self.NODE_ORDER - 1) + 1, j // (self.NODE_ORDER - 1) + 1


class _SimplexFaceNodes:
    """``H1`` of order ``NODE_ORDER`` (3 or more): a triangle face's ``(NODE_ORDER - 1)(NODE_ORDER - 2) /
    2`` nodes, in the frame of its own row (``U`` toward its point 1, ``V`` toward
    its point 2, ``U`` fastest)."""

    @tack.func
    def node_count(self, i):
        return (self.NODE_ORDER - 1) * (self.NODE_ORDER - 2) // 2

    @tack.func
    def node_dof(self, i, j):
        return self.node_face_dofs[self.entity_id(i)] + j

    @tack.func
    def node_steps(self, i, j):
        # Rows of V = 1, 2, ...: row V holds NODE_ORDER - 1 - V nodes.
        v = 1
        rest = j
        while rest >= self.NODE_ORDER - 1 - v:
            rest -= self.NODE_ORDER - 1 - v
            v += 1
        return rest + 1, v


class _SimplexInteriorNodes:
    """``H1`` of order ``NODE_ORDER`` (3 or more): a triangle's or tetrahedron's nodes strictly
    inside it, in node order (``bases.simplex_lattice``), where its geometry maps
    them."""

    @tack.func
    def node_count(self, i):
        n = 1
        for k in range(self.DIMENSION):
            n = n * (self.NODE_ORDER - 1 - k) // (k + 1)
        return n

    @tack.func
    def node_dof(self, i, j):
        return self.node_cell_dofs[self.entity_id(i)] + j

    @tack.func
    def node_position(self, i, j):
        # The j-th lattice point (i, j, k >= 1, sum <= NODE_ORDER - 1), the first axis fastest.
        p = self.NODE_ORDER
        found = 0
        pc = tack.vec3(0.0, 0.0, 0.0)
        for kk in range(1 if self.DIMENSION == 3 else 0, p if self.DIMENSION == 3 else 1):
            for jj in range(1, p - kk):
                for ii in range(1, p - jj - kk):
                    if found == j:
                        pc = tack.vec3(self.node_points[ii], self.node_points[jj],
                                       self.node_points[kk] if self.DIMENSION == 3 else 0.0)
                    found += 1
        return self.position(i, pc)


class _SimplexCellNodes:
    """``L2`` on a simplex of order ``NODE_ORDER``: a cell's ``NODE_COUNT`` nodes, in node order
    (``node_ijk``, ``bases.simplex_lattice``), where its geometry maps them."""

    @tack.func
    def node_count(self, i):
        return self.NODE_COUNT

    @tack.func
    def node_dof(self, i, j):
        return self.node_offsets[self.entity_id(i)] + j

    @tack.func
    def node_parametric_point(self, i, j):
        pc = tack.vec3(self.node_points[self.node_ijk[j * self.DIMENSION]],
                       self.node_points[self.node_ijk[j * self.DIMENSION + 1]], 0.0)
        if self.DIMENSION == 3:
            pc[2] = self.node_points[self.node_ijk[j * self.DIMENSION + 2]]
        return pc

    @tack.func
    def node_position(self, i, j):
        return self.position(i, self.node_parametric_point(i, j))


@tack.func
def _triangle_frame(view, i):
    # A triangle's corners ordered by position (_before), the same in every piece
    # whatever its stored order: (first, second, third).
    a = 0
    b = 1
    c = 2
    if _before(view.corner_position(i, b), view.corner_position(i, a)):
        a, b = b, a
    if _before(view.corner_position(i, c), view.corner_position(i, b)):
        b, c = c, b
        if _before(view.corner_position(i, b), view.corner_position(i, a)):
            a, b = b, a
    return a, b, c


class _StraightNodePosition:
    """An edge's or quad face's node where its straight (order-1) geometry puts it:
    linear between an edge's points, bilinear over a face's. Computed from the edge's
    first point by position, or in the face's frame (``_square_frame``), with the
    1D nodes read there, so every piece holding the entity gets the same bits
    whatever order its point ids store it in."""

    @tack.func
    def node_position(self, i, j):
        sa, sb = self.node_steps(i, j)
        if self.NUM_POINTS == 3:
            # A triangle face: the node's barycentric steps over its row's points,
            # from the first point by position toward the other two.
            a, b, c = _triangle_frame(self, i)
            w0 = self.NODE_ORDER - sa - sb
            wb = sa if b == 1 else (sb if b == 2 else w0)
            wc = sa if c == 1 else (sb if c == 2 else w0)
            xa = self.corner_position(i, a)
            return (xa + self.node_points[wb] * (self.corner_position(i, b) - xa)
                    + self.node_points[wc] * (self.corner_position(i, c) - xa))
        if self.NUM_POINTS == 2:
            x0 = self.corner_position(i, 0)
            x1 = self.corner_position(i, 1)
            if _before(x1, x0):
                return x1 + self.node_points[self.NODE_ORDER - sa] * (x0 - x1)
            return x0 + self.node_points[sa] * (x1 - x0)
        a, b, d = _square_frame(self, i)
        ax, ay = _square_corner(a)
        bx, by = _square_corner(b)
        dx, dy = _square_corner(d)
        ua = sa - ax * self.NODE_ORDER
        ub = sb - ay * self.NODE_ORDER
        tu = self.node_points[ua * (bx - ax) + ub * (by - ay)]
        tv = self.node_points[ua * (dx - ax) + ub * (dy - ay)]
        xa = self.corner_position(i, a)
        return (xa + tu * (1.0 - tv) * (self.corner_position(i, b) - xa)
                + tu * tv * (self.corner_position(i, (a + 2) % 4) - xa)
                + (1.0 - tu) * tv * (self.corner_position(i, d) - xa))


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
def _sorted_corners(p0, p1, p2):
    """Three corners' order by position (``_before``): the canonical frame of a
    triangle face, the same from every cell holding it."""
    # Each corner's rank: how many corners come before it.
    r0 = tack.i32(_before(p1, p0)) + tack.i32(_before(p2, p0))
    r1 = tack.i32(_before(p0, p1)) + tack.i32(_before(p2, p1))
    s0 = 0 if r0 == 0 else (1 if r1 == 0 else 2)
    s1 = 0 if r0 == 1 else (1 if r1 == 1 else 2)
    s2 = 0 if r0 == 2 else (1 if r1 == 2 else 2)
    return s0, s1, s2


class _CurvedTriangleNodePosition:
    """A triangle face's node where a curved geometry of another order puts it: the
    geometry's trace on the face (``trace``, copied by ``dataset._entity_traces`` in
    the face's canonical frame, its corners by position), evaluated at the node's
    barycentric coordinates in that frame -- Lagrange on equispaced nodes in
    Silvester's form, or Bernstein -- so every piece holding the face gets the same
    bits. The field's nodes are equispaced (``node_points``)."""

    @tack.func
    def node_position(self, i, j):
        sa, sb = self.node_steps(i, j)
        p = self.NODE_ORDER
        s0, s1, s2 = _sorted_corners(self.corner_position(i, 0), self.corner_position(i, 1),
                                     self.corner_position(i, 2))
        # The node's steps toward the row's corners 0, 1 and 2, then in the frame.
        w0 = p - sa - sb
        k0 = w0 if s0 == 0 else (sa if s0 == 1 else sb)
        k1 = w0 if s1 == 0 else (sa if s1 == 1 else sb)
        k2 = w0 if s2 == 0 else (sa if s2 == 1 else sb)
        l0 = self.node_points[k0]
        l1 = self.node_points[k1]
        l2 = self.node_points[k2]
        q = self.trace_q
        g = q - 1
        base = self.entity_id(i) * q * q
        x = self.trace[base] * 0.0
        for jj in range(q):
            for ii in range(q - jj):
                w = (_triangle_factor(ii, g, l1, self.trace_bernstein)
                     * _triangle_factor(jj, g, l2, self.trace_bernstein)
                     * _triangle_factor(g - ii - jj, g, l0, self.trace_bernstein))
                if self.trace_bernstein == 1:
                    for s in range(2, g + 1):
                        w = w * s                     # g! / (i! j! k!): the factors hold 1/m!
                x += self.trace[base + ii + q * jj] * w
        return x


@tack.func
def _triangle_factor(m, g, x, bernstein):
    """One barycentric factor of a triangle basis of order ``g``: Silvester's
    ``prod_{s < m} (g x - s) / (s + 1)`` (Lagrange, equispaced), or ``x^m / m!``
    (Bernstein, the ``g!`` applied by the caller)."""
    out = x - x + 1.0
    for s in range(m):
        if bernstein == 1:
            out = out * x / (s + 1)
        else:
            out = out * (g * x - s) / (s + 1)
    return out


class _CurvedNodePosition:
    """An edge's or quad face's node where a curved geometry of another order puts
    it: the geometry's trace on the entity at the node, from the geometry's values
    there (``trace``: ``trace_q`` a side per entity, copied from the cells by
    ``dataset._entity_traces`` in the frame the entity's corners' positions fix),
    with the field's 1D nodes read in that frame as ``_StraightNodePosition``
    reads them, so every piece holding the entity gets the same bits."""

    @tack.func
    def node_position(self, i, j):
        sa, sb = self.node_steps(i, j)
        q = self.trace_q
        if self.NUM_POINTS == 2:
            t = self.node_points[sa]
            if _before(self.corner_position(i, 1), self.corner_position(i, 0)):
                t = self.node_points[self.NODE_ORDER - sa]
            base = self.entity_id(i) * q
            x = self.trace[base] * _axis(t, 0, self.trace_nodes, self.trace_weights, q,
                                         self.trace_bernstein)
            for k in range(1, q):
                x += self.trace[base + k] * _axis(t, k, self.trace_nodes, self.trace_weights,
                                                  q, self.trace_bernstein)
            return x
        a, b, d = _square_frame(self, i)
        ax, ay = _square_corner(a)
        bx, by = _square_corner(b)
        dx, dy = _square_corner(d)
        ua = sa - ax * self.NODE_ORDER
        ub = sb - ay * self.NODE_ORDER
        tu = self.node_points[ua * (bx - ax) + ub * (by - ay)]
        tv = self.node_points[ua * (dx - ax) + ub * (dy - ay)]
        base = self.entity_id(i) * q * q
        x = self.trace[base] * (
            _axis(tu, 0, self.trace_nodes, self.trace_weights, q, self.trace_bernstein)
            * _axis(tv, 0, self.trace_nodes, self.trace_weights, q, self.trace_bernstein))
        for k in range(1, q * q):
            x += self.trace[base + k] * (
                _axis(tu, k % q, self.trace_nodes, self.trace_weights, q, self.trace_bernstein)
                * _axis(tv, k // q, self.trace_nodes, self.trace_weights, q,
                        self.trace_bernstein))
        return x


class _TensorInteriorNodes:
    """``H1`` of order ``NODE_ORDER`` (3 or more): a cell's ``(NODE_ORDER - 1)^d`` interior nodes,
    lexicographic in its frame, where its geometry maps them."""

    @tack.func
    def node_count(self, i):
        n = 1
        for _ in range(self.DIMENSION):
            n *= self.NODE_ORDER - 1
        return n

    @tack.func
    def node_dof(self, i, j):
        return self.node_cell_dofs[self.entity_id(i)] + j

    @tack.func
    def node_position(self, i, j):
        q = self.NODE_ORDER - 1
        pc = tack.vec3(self.node_points[j % q + 1], 0.0, 0.0)
        if self.DIMENSION >= 2:
            pc[1] = self.node_points[(j // q) % q + 1]
        if self.DIMENSION == 3:
            pc[2] = self.node_points[j // (q * q) + 1]
        return self.position(i, pc)


class _TensorCellNodes:
    """``L2`` on a tensor basis: a cell's ``(NODE_ORDER + 1)^d`` nodes, in node order, where
    its geometry maps them (layout ``"blocks"``; ``_TensorCellNodesCells`` is
    cell-fastest)."""

    @tack.func
    def node_count(self, i):
        n = 1
        for _ in range(self.DIMENSION):
            n *= self.NODE_ORDER + 1
        return n

    @tack.func
    def node_dof(self, i, j):
        return self.node_offsets[self.entity_id(i)] + j

    @tack.func
    def node_parametric_point(self, i, j):
        q = self.NODE_ORDER + 1
        pc = tack.vec3(self.node_points[j % q], 0.0, 0.0)
        if self.DIMENSION >= 2:
            pc[1] = self.node_points[(j // q) % q]
        if self.DIMENSION == 3:
            pc[2] = self.node_points[j // (q * q)]
        return pc

    @tack.func
    def node_position(self, i, j):
        return self.position(i, self.node_parametric_point(i, j))


class _TensorCellNodesCells(_TensorCellNodes):
    """``L2`` on a tensor basis, cell-fastest: node ``j`` of cell ``i`` at ``j *
    node_cells + i``."""

    @tack.func
    def node_dof(self, i, j):
        return j * self.node_cells + self.entity_id(i)


# ── Groups and view classes ─────────────────────────────────────────

_view_classes = {}


def view_class(kind, shape, *mixins):
    """The template class combining ``mixins``, ``kind`` and ``shape``, made once."""
    key = (kind, shape, mixins)
    if key not in _view_classes:
        _check_names(*mixins, kind, shape)
        name = "".join(m.__name__.strip("_") for m in mixins) + kind.__name__ + shape.__name__
        _view_classes[key] = tack.device_class(type(name, (*mixins, kind, shape),
                                                    _node_count(mixins, shape)))
    return _view_classes[key]


def _node_count(mixins, shape):
    """``{"NODES": n}`` for a field view whose node count follows its shape (the
    corners at order 1, the quadratic nodes at order 2, ``(ORDER + 1)^d`` in a
    tensor basis, the corners and edges in serendipity), else nothing: the simplex and wedge views and order 0 set
    their own."""
    if any(hasattr(m, "NODES") for m in mixins) or not hasattr(shape, "NUM_POINTS"):
        return {}
    for mixin in mixins:
        for klass in mixin.__mro__:
            if klass is _SerendipityField:
                return {"NODES": shape.NUM_POINTS + shape.NUM_EDGES}
            if klass is _H1Order2Field:
                return {"NODES": shape.NUM_QUADRATIC}
            if klass is _TensorBasis:
                return {"NODES": (mixin.ORDER + 1) ** shape.DIMENSION}
            if klass is _Interpolated:
                return {"NODES": shape.NUM_POINTS}
    return {}


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
