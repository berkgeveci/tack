"""Cell shapes: VTK's linear cells as template classes.

Each shape is a ``@tack.device_class`` class whose ``@tack.func`` methods
are its parametric functions. A kernel takes a shape object as a template
argument and calls its methods directly, so every compiled variant is
specialized to one shape and contains no dispatch on the shape::

    @tack.kernel
    def centers(cell, conn, xyz, out):
        for c in range(out.shape[0]):
            pc = cell.parametric_center()
            x = tack.vec3(0.0, 0.0, 0.0)
            for j in range(cell.NUM_POINTS):
                x += cell.shape_function(j, pc) * xyz[conn[c, j]]
            out[c] = x

    centers(shapes.Hexahedron(), conn, xyz, out)

A mesh of several shapes runs such a kernel once per shape present, over
that shape's cells.

Every shape carries VTK's cell type id (``Hexahedron.ID`` is 12, as in
``vtkCellType.h``), and VTK's point order, parametric coordinates, edges
and faces, so VTK serves as the reference without any remapping:

=========== ==== ====== === ==============================================
class       ID   points dim parametric domain
=========== ==== ====== === ==============================================
Vertex      1    1      0   the point (0, 0, 0)
Line        3    2      1   r in [0, 1]
Triangle    5    3      2   r, s >= 0, r + s <= 1
Pixel       8    4      2   r, s in [0, 1]; points in x-fastest order
Quad        9    4      2   r, s in [0, 1]; points counterclockwise
Tetra       10   4      3   r, s, t >= 0, r + s + t <= 1
Voxel       11   8      3   r, s, t in [0, 1]; points in x-fastest order
Hexahedron  12   8      3   r, s, t in [0, 1]; each face counterclockwise
Wedge       13   6      3   a triangle in (r, s) times t in [0, 1]
Pyramid     14   5      3   r, s, t in [0, 1]; the apex is (0, 0, 1)
=========== ==== ====== === ==============================================

Class constants: ``ID``, ``NUM_POINTS``, ``DIMENSION``, ``NUM_EDGES`` and
``NUM_FACES``. Faces are codimension-1: a 3D shape's faces are polygons, a 2D
shape's are its edges (lines, in the order going round the cell), and a line
has none. VTK counts a 2D cell's faces as 0; Tack does not, so a 2D mesh has
faces, as a polygonal one does.

Methods, all callable only from kernels and device functions:

- ``parametric_point(j)``, ``parametric_center()``: 3-vectors.
- ``shape_function(j, pc)``, ``shape_gradient(j, pc)``: VTK's
  ``InterpolateFunctions`` and ``InterpolateDerivs`` for point ``j`` at
  parametric coordinates ``pc``, a 3-vector; the gradient is
  ``(d/dr, d/ds, d/dt)``.
- ``is_inside(pc, tol)``: 1 if ``pc`` is in the parametric domain widened
  by ``tol``, else 0.
- ``edge_corner(e, k)``, ``face_corner(f, k)``, ``face_shape(f)``,
  ``face_num_points(f)``: VTK's ``GetEdge``/``GetFace`` as indices into the
  cell's points, and the face's shape id.
- ``interpolate(values, pc)``, ``interpolate_point(pts, pc)``,
  ``jacobian(pts, pc)``, ``world_to_parametric(pts, x)``,
  ``nearest_point(pts, x, pc)``: the geometry of a cell whose points are
  given, defined once for every shape.

Coordinates a shape does not use are ignored: a quad's functions do not
read ``t``. A point index must satisfy ``0 <= j < NUM_POINTS``, and edge
and face indices likewise; others give unspecified values.

The geometric methods take the cell's points from a local array the
caller gathered, three coordinates per point, and per-point values from
another::

    pts = tack.local_array(tack.f32, 3 * cell.NUM_POINTS)
    for j in range(cell.NUM_POINTS):
        pts[3 * j], pts[3 * j + 1], pts[3 * j + 2] = xyz[conn[c, j]]
    pc, found = cell.world_to_parametric(pts, x)
    if found and cell.is_inside(pc, 1e-4):
        ...
"""

import tack
from tack.dataset import _contour_tables as _ct

VERTEX = tack.constant(1, tack.i32)
LINE = tack.constant(3, tack.i32)
TRIANGLE = tack.constant(5, tack.i32)
PIXEL = tack.constant(8, tack.i32)
QUAD = tack.constant(9, tack.i32)
TETRA = tack.constant(10, tack.i32)
VOXEL = tack.constant(11, tack.i32)
HEXAHEDRON = tack.constant(12, tack.i32)
WEDGE = tack.constant(13, tack.i32)
PYRAMID = tack.constant(14, tack.i32)

#: The most points any of these shapes has, for sizing local arrays.
MAX_CELL_POINTS = tack.constant(8, tack.i32)

#: Newton steps in ``world_to_parametric``, as in VTK.
NEWTON_ITERATIONS = tack.constant(10, tack.i32)
#: A step no longer than this, in every parametric coordinate, ends the
#: iteration. It is applied first, so the error left is of order its square.
NEWTON_TOLERANCE = tack.constant(1e-4)
#: A parametric coordinate past this ends the iteration as diverged.
NEWTON_DIVERGED = tack.constant(1e6)
#: How far outside its parametric domain (or a fanned cell's sub-triangle or
#: sub-tetrahedron) a point may lie and still count as held by the cell. A
#: point on a face, edge or vertex lies on the boundary of every cell that has
#: it, and without a margin which of them held it was the sign of rounding
#: noise: the cell-side and point-side searches disagreed at shared mesh
#: points (every shape but the voxel on the CPU, and on CUDA, where NVRTC
#: fuses the same inlined functions differently in the two kernels). With it
#: they all hold the point, and the smaller id wins. Far above that noise
#: (Newton leaves about its tolerance squared), and far below the locator's
#: own tolerance: a point this close to a neighbour is extrapolated by at most
#: this much in it.
HOLD_TOLERANCE = tack.constant(1e-5)
#: How far inside its parametric domain a point must lie for a cell to be the
#: only one holding it (``_MeshSearch.find_from``): a neighbour across a face
#: holds it only if its own 1e-5 reaches that far, which takes a neighbour a
#: hundred times as deep across the face.
WELL_INSIDE = tack.constant(1e-3)
#: A point this close to a pyramid's apex, relative to the pyramid's size, is
#: taken to be at it (``Pyramid.world_to_parametric``): there every ``(r, s)``
#: at ``t = 1`` is the same point, Newton's Jacobian is singular, and it did
#: not converge within 1e-8 of the apex.
APEX_SNAP = tack.constant(1e-6)
#: Within this much of a pyramid's size of its apex, a point Newton cannot place
#: (outside, beside the apex: the parametric map runs off as (offset) / (1 - t))
#: is put just past the apex, ``APEX_OUTSIDE`` above it in ``t``: past the hold
#: margin, so a cell that holds the point ranks first, and near if the faces'
#: nearest point is within the locator's tolerance (0.1% of a cell's diagonal,
#: which this reach covers), its values the apex's.
APEX_REACH = tack.constant(2e-3)
APEX_OUTSIDE = tack.constant(2.0 * float(HOLD_TOLERANCE))

# Edges as point pairs and faces as four points (-1 pads a triangle),
# flattened. Each is VTK's GetEdge/GetFace order.
_TRIANGLE_EDGES = tack.constant((0, 1, 1, 2, 2, 0), tack.i32)
_PIXEL_EDGES = tack.constant((0, 1, 1, 3, 2, 3, 0, 2), tack.i32)
_QUAD_EDGES = tack.constant((0, 1, 1, 2, 2, 3, 3, 0), tack.i32)
# A pixel's edges as faces, going round it (its points are x-fastest).
_PIXEL_FACES = tack.constant((0, 1, 1, 3, 3, 2, 2, 0), tack.i32)
_TETRA_EDGES = tack.constant((0, 1, 1, 2, 2, 0, 0, 3, 1, 3, 2, 3), tack.i32)
_VOXEL_EDGES = tack.constant(
    (0, 1, 1, 3, 2, 3, 0, 2, 4, 5, 5, 7, 6, 7, 4, 6, 0, 4, 1, 5, 2, 6, 3, 7),
    tack.i32)
_HEXAHEDRON_EDGES = tack.constant(
    (0, 1, 1, 2, 3, 2, 0, 3, 4, 5, 5, 6, 7, 6, 4, 7, 0, 4, 1, 5, 3, 7, 2, 6),
    tack.i32)
_WEDGE_EDGES = tack.constant(
    (0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3, 0, 3, 1, 4, 2, 5), tack.i32)
_PYRAMID_EDGES = tack.constant(
    (0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 4, 2, 4, 3, 4), tack.i32)

_TETRA_FACES = tack.constant(
    (0, 1, 3, -1, 1, 2, 3, -1, 2, 0, 3, -1, 0, 2, 1, -1), tack.i32)
# A voxel's faces as VTK's pixels, each with its last two corners swapped so it
# goes round: as quads.
_VOXEL_FACES = tack.constant(
    (2, 0, 4, 6, 1, 3, 7, 5, 0, 1, 5, 4, 3, 2, 6, 7, 1, 0, 2, 3, 4, 5, 7, 6),
    tack.i32)
_HEXAHEDRON_FACES = tack.constant(
    (0, 4, 7, 3, 1, 2, 6, 5, 0, 1, 5, 4, 3, 7, 6, 2, 0, 3, 2, 1, 4, 5, 6, 7),
    tack.i32)
_WEDGE_FACES = tack.constant(
    (0, 2, 1, -1, 3, 4, 5, -1, 0, 1, 4, 3, 1, 2, 5, 4, 2, 0, 3, 5), tack.i32)
_PYRAMID_FACES = tack.constant(
    (0, 3, 2, 1, 0, 1, 4, -1, 1, 2, 4, -1, 2, 3, 4, -1, 3, 0, 4, -1), tack.i32)


@tack.func
def _q1d(a, x):
    """The 1D quadratic Lagrange function of the node at ``a`` (0, 0.5 or 1), at ``x``."""
    if a < 0.25:
        return 2.0 * (x - 0.5) * (x - 1.0)
    if a > 0.75:
        return 2.0 * x * (x - 0.5)
    return 4.0 * x * (1.0 - x)


@tack.func
def _q1d_derivative(a, x):
    if a < 0.25:
        return 4.0 * x - 3.0
    if a > 0.75:
        return 4.0 * x - 1.0
    return 4.0 - 8.0 * x


@tack.func
def _p2(m, x):
    """One barycentric coordinate's factor of a simplex's quadratic Lagrange function:
    the node's coordinate is ``m`` (0, 0.5 or 1) and the point's ``x``. Their
    product is ``x (2x - 1)`` at a corner and ``4 x_i x_j`` at an edge's middle."""
    if m > 0.75:
        return x * (2.0 * x - 1.0)
    if m > 0.25:
        return 2.0 * x
    return 1.0 + 0.0 * x


@tack.func
def _p2_derivative(m, x):
    if m > 0.75:
        return 4.0 * x - 1.0
    if m > 0.25:
        return 2.0 + 0.0 * x
    return 0.0 * x


@tack.func
def _linear(a, x):
    """The 1D linear shape function of the end at ``a`` (0 or 1), at ``x``."""
    return x if a == 1 else 1.0 - x


@tack.func
def _nearest_on_triangle(a, b, c, p):
    """The point of triangle ``abc`` nearest ``p``, by the region of the triangle's
    plane that ``p`` projects into (Ericson, Real-Time Collision Detection, 5.1.5)."""
    ab = b - a
    ac = c - a
    d1 = ab.dot(p - a)
    d2 = ac.dot(p - a)
    d3 = ab.dot(p - b)
    d4 = ac.dot(p - b)
    d5 = ab.dot(p - c)
    d6 = ac.dot(p - c)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    out = a
    if d1 <= 0.0 and d2 <= 0.0:
        out = a
    elif d3 >= 0.0 and d4 <= d3:
        out = b
    elif d6 >= 0.0 and d5 <= d6:
        out = c
    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        out = a + (d1 / (d1 - d3)) * ab
    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        out = a + (d2 / (d2 - d6)) * ac
    elif va <= 0.0 and d4 >= d3 and d5 >= d6:
        out = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    else:
        total = va + vb + vc
        out = a + (vb / total) * ab + (vc / total) * ac
    return out


@tack.func
def _point(pts, j):
    """Point ``j`` of a local array of x, y, z triples, as a 3-vector."""
    return tack.vec3(pts[3 * j], pts[3 * j + 1], pts[3 * j + 2])


# ── The algorithms every shape shares ───────────────────────────────

@tack.device_class
class Shape:
    """What every shape has in common: the geometry of a cell given its points.

    A subclass supplies the class constants and ``shape_function``,
    ``shape_gradient``, ``parametric_point``, ``parametric_center`` and
    ``is_inside``, and the Newton step for its dimension (``_newton_step``).
    """

    NUM_EDGES = 0
    NUM_FACES = 0
    #: The most triangles one cell of the shape contours to; 0 for shapes
    #: below three dimensions, which contour to nothing here.
    CONTOUR_TRIANGLES = 0
    #: Nodes of the quadratic (order-2) Lagrange element: the corners, then a
    #: node in the middle of each edge (in edge order), of each of the last
    #: QUADRATIC_FACES faces, and QUADRATIC_INTERIOR in the cell. 0 for a
    #: shape with no quadratic element here.
    NUM_QUADRATIC = 0
    QUADRATIC_FACES = 0
    QUADRATIC_INTERIOR = 0

    @tack.func
    def quadratic_node(self, k):
        """The parametric coordinates of quadratic node ``k``."""
        if k < self.NUM_POINTS:
            return self.parametric_point(k)
        e = k - self.NUM_POINTS
        if e < self.NUM_EDGES:
            return 0.5 * (self.parametric_point(self.edge_corner(e, 0))
                          + self.parametric_point(self.edge_corner(e, 1)))
        i = e - self.NUM_EDGES
        if i < self.QUADRATIC_FACES:
            f = self.NUM_FACES - self.QUADRATIC_FACES + i
            return 0.25 * (self.parametric_point(self.face_corner(f, 0))
                           + self.parametric_point(self.face_corner(f, 1))
                           + self.parametric_point(self.face_corner(f, 2))
                           + self.parametric_point(self.face_corner(f, 3)))
        return self.parametric_center()

    @tack.func
    def contour_count(self, case):
        """How many triangles case ``case`` makes: bit ``j`` is 1 when point ``j``
        is at or above the isovalue (VTK's tables)."""
        return 0

    @tack.func
    def contour_edge(self, case, k):
        """The edge that point ``k % 3`` of triangle ``k // 3`` of case ``case`` lies on."""
        return -1

    @tack.func
    def edge_corner(self, e, k):
        """Point ``k`` (0 or 1) of edge ``e``, as an index into the cell's points."""
        return -1

    @tack.func
    def face_corner(self, f, k):
        """Point ``k`` of face ``f``, as an index into the cell's points.

        A face's points are in VTK's order, which orients its normal out of
        the cell (except for a voxel, whose faces are pixels in x-fastest
        order).
        """
        return -1

    @tack.func
    def face_shape(self, f):
        """The shape id of face ``f``: ``TRIANGLE`` or ``QUAD`` (a voxel's faces too, going
        round); ``LINE`` for a 2D shape's."""
        return 0

    @tack.func
    def face_num_points(self, f):
        """The number of points of face ``f``: 2, 3 or 4."""
        shape = self.face_shape(f)
        return 2 if shape == LINE else (3 if shape == TRIANGLE else 4)

    @tack.func
    def interpolate(self, values, pc):
        """The value at ``pc`` of the cell whose points carry ``values``, a local array."""
        total = 0.0
        for j in range(self.NUM_POINTS):
            total += self.shape_function(j, pc) * values[j]
        return total

    @tack.func
    def interpolate_point(self, pts, pc):
        """The world position at ``pc`` of the cell whose points are ``pts``.

        ``pts`` is a local array holding each point's x, y and z in turn.
        """
        x = tack.vec3(0.0, 0.0, 0.0)
        for j in range(self.NUM_POINTS):
            x += self.shape_function(j, pc) * _point(pts, j)
        return x

    @tack.func
    def jacobian(self, pts, pc):
        """The 3x3 derivative of world position by parametric coordinates at ``pc``.

        Column ``k`` is ``dx/d(pc[k])``; columns past the shape's dimension
        are zero. Its determinant is positive for a well-formed 3D cell in
        VTK's point order.
        """
        m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for j in range(self.NUM_POINTS):
            m += _point(pts, j).outer_product(self.shape_gradient(j, pc))
        return m

    @tack.func
    def nearest_point(self, pts, x, pc):
        """The point of the cell nearest world point ``x``, given ``pc`` from
        ``world_to_parametric(pts, x)``. This default, for a line and a vertex,
        is the position at ``pc`` clamped into the parametric domain, which for
        them is exact. A solid's is ``x`` itself inside and otherwise the nearest
        point of its faces; a triangle's and a quad's the nearest point of the
        cell itself (``_Solid``, ``Triangle``, ``_Quadrilateral``)."""
        return self.interpolate_point(pts, self._clamp(pc))

    @tack.func
    def world_to_parametric(self, pts, x):
        """The parametric coordinates of world point ``x`` in the cell whose points are ``pts``.

        Returns ``(pc, converged)``. ``pc`` is found by Newton's method from
        the parametric center, as VTK's ``EvaluatePosition`` does: at most
        ``NEWTON_ITERATIONS`` steps, ending when a step is shorter than
        ``NEWTON_TOLERANCE`` in every coordinate (``converged`` is then 1)
        or when ``pc`` diverges past ``NEWTON_DIVERGED`` or stops being
        finite (``converged`` stays 0). ``pc`` is returned for points outside
        the cell as well; ``is_inside`` decides.

        For a triangle, pixel or quad the result is the point of the cell's
        surface nearest ``x`` along its normal, for a line the nearest point
        of the line through it, and for a vertex ``(0, 0, 0)``.

        The residual is taken relative to the cell's first point, so cells
        far from the origin keep their precision in f32.
        """
        pc, converged = self._newton(pts, x)
        return pc, converged

    @tack.func
    def _newton(self, pts, x):
        """``world_to_parametric``'s Newton iteration. A shape that answers some
        points itself (the pyramid, at its apex) overrides
        ``world_to_parametric`` and calls this for the rest."""
        pc = self.parametric_center()
        converged = 0
        p0 = _point(pts, 0)
        target = x - p0
        for _ in range(NEWTON_ITERATIONS):
            # Position and Jacobian at pc, relative to the first point.
            here = tack.vec3(0.0, 0.0, 0.0)
            m = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            for j in range(self.NUM_POINTS):
                p = _point(pts, j) - p0
                here += self.shape_function(j, pc) * p
                m += p.outer_product(self.shape_gradient(j, pc))
            step = self._newton_step(m, target - here)
            pc += step
            if not (abs(pc[0]) < NEWTON_DIVERGED and abs(pc[1]) < NEWTON_DIVERGED
                    and abs(pc[2]) < NEWTON_DIVERGED):
                break
            if (abs(step[0]) <= NEWTON_TOLERANCE and abs(step[1]) <= NEWTON_TOLERANCE
                    and abs(step[2]) <= NEWTON_TOLERANCE):
                converged = 1
                break
        return pc, converged


@tack.device_class
class _Solid(Shape):
    DIMENSION = 3

    @tack.func
    def nearest_point(self, pts, x, pc):
        """``x`` itself inside the cell; outside, the nearest point of its nearest
        face, a quad face taken as two triangles -- exact when the face is
        planar, as VTK finds it since 45f0f69a7e (2026-07) for a hexahedron. The
        position at clamped parametric coordinates, used before (TA-39), is
        exact only for an axis-aligned box."""
        out = x
        if self.is_inside(pc, 0.0) == 1:
            out = self.interpolate_point(pts, pc)
        else:
            nearest = -1.0
            for f in range(self.NUM_FACES):
                a = _point(pts, self.face_corner(f, 0))
                for k in range(1, self.face_num_points(f) - 1):
                    q = _nearest_on_triangle(a, _point(pts, self.face_corner(f, k)),
                                             _point(pts, self.face_corner(f, k + 1)), x)
                    d = (q - x).dot(q - x)
                    if nearest < 0.0 or d < nearest:
                        nearest = d
                        out = q
        return out

    @tack.func
    def _newton_step(self, m, residual):
        return m.inverse() @ residual


@tack.device_class
class _Surface(Shape):
    DIMENSION = 2

    @tack.func
    def face_shape(self, f):
        return LINE

    @tack.func
    def face_corner(self, f, k):
        """Point ``k`` of the cell's edge ``f``, going round the cell: its faces."""
        return self.edge_corner(f, k)

    @tack.func
    def _newton_step(self, m, residual):
        # Complete the frame with the unit normal; the step along it is the
        # distance off the surface, and is dropped.
        n = tack.vec3(m[0, 0], m[1, 0], m[2, 0]).cross(
            tack.vec3(m[0, 1], m[1, 1], m[2, 1])).normalized()
        m[0, 2] = n[0]
        m[1, 2] = n[1]
        m[2, 2] = n[2]
        step = m.inverse() @ residual
        step[2] = 0.0
        return step


# ── Shapes ──────────────────────────────────────────────────────────

@tack.device_class
class Vertex(Shape):
    """VTK_VERTEX: one point. Every ``pc`` is inside; nearness is a matter of distance."""

    ID = VERTEX
    NUM_POINTS = 1
    DIMENSION = 0

    @tack.func
    def parametric_point(self, j):
        return tack.vec3(0.0, 0.0, 0.0)

    @tack.func
    def parametric_center(self):
        return tack.vec3(0.0, 0.0, 0.0)

    @tack.func
    def shape_function(self, j, pc):
        return 1.0

    @tack.func
    def shape_gradient(self, j, pc):
        return tack.vec3(0.0, 0.0, 0.0)

    @tack.func
    def is_inside(self, pc, tol):
        return 1

    @tack.func
    def _clamp(self, pc):
        return pc

    @tack.func
    def _newton(self, pts, x):
        # No iteration: every point's coordinates are (0, 0, 0). Overriding
        # _newton rather than world_to_parametric leaves Shape's Newton loop,
        # which needs a _newton_step the vertex lacks, out of its template.
        return tack.vec3(0.0, 0.0, 0.0), 1


@tack.device_class
class Line(Shape):
    """VTK_LINE: two points, ``r`` from point 0 to point 1."""

    ID = LINE
    NUM_POINTS = 2
    DIMENSION = 1

    @tack.func
    def parametric_point(self, j):
        return tack.vec3(tack.f32(j), 0.0, 0.0)

    @tack.func
    def parametric_center(self):
        return tack.vec3(0.5, 0.0, 0.0)

    @tack.func
    def shape_function(self, j, pc):
        return _linear(j, pc[0])

    @tack.func
    def shape_gradient(self, j, pc):
        return tack.vec3(2.0 * j - 1.0, 0.0, 0.0)

    @tack.func
    def is_inside(self, pc, tol):
        return -tol <= pc[0] and pc[0] <= 1.0 + tol

    @tack.func
    def _clamp(self, pc):
        return tack.vec3(min(max(pc[0], 0.0), 1.0), pc[1], pc[2])

    @tack.func
    def _newton_step(self, m, residual):
        a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
        return tack.vec3(a.dot(residual) / a.dot(a), 0.0, 0.0)


@tack.device_class
class Triangle(_Surface):
    """VTK_TRIANGLE: shape functions ``1 - r - s``, ``r`` and ``s``."""

    ID = TRIANGLE
    NUM_POINTS = 3
    NUM_EDGES = 3
    NUM_FACES = 3

    @tack.func
    def parametric_point(self, j):
        return tack.vec3(1.0 if j == 1 else 0.0, 1.0 if j == 2 else 0.0, 0.0)

    @tack.func
    def parametric_center(self):
        return tack.vec3(1.0 / 3.0, 1.0 / 3.0, 0.0)

    @tack.func
    def shape_function(self, j, pc):
        if j == 0:
            return 1.0 - pc[0] - pc[1]
        return pc[0] if j == 1 else pc[1]

    @tack.func
    def shape_gradient(self, j, pc):
        if j == 0:
            return tack.vec3(-1.0, -1.0, 0.0)
        return self.parametric_point(j)

    #: The six-node triangle, a tetrahedron's face: corners and edge middles.
    NUM_QUADRATIC = 6

    @tack.func
    def quadratic_function(self, k, pc):
        n = self.quadratic_node(k)
        return (_p2(1.0 - n[0] - n[1], 1.0 - pc[0] - pc[1])
                * _p2(n[0], pc[0]) * _p2(n[1], pc[1]))

    @tack.func
    def quadratic_gradient(self, k, pc):
        n = self.quadratic_node(k)
        m0 = 1.0 - n[0] - n[1]
        x0 = 1.0 - pc[0] - pc[1]
        f0 = _p2(m0, x0)
        f1 = _p2(n[0], pc[0])
        f2 = _p2(n[1], pc[1])
        d0 = _p2_derivative(m0, x0) * f1 * f2               # by x0, which falls with r, s
        return tack.vec3(f0 * _p2_derivative(n[0], pc[0]) * f2 - d0,
                         f0 * f1 * _p2_derivative(n[1], pc[1]) - d0, 0.0)

    @tack.func
    def is_inside(self, pc, tol):
        return -tol <= pc[0] and -tol <= pc[1] and pc[0] + pc[1] <= 1.0 + tol

    @tack.func
    def _clamp(self, pc):
        r = max(pc[0], 0.0)
        s = max(pc[1], 0.0)
        scale = 1.0 / max(r + s, 1.0)
        return tack.vec3(r * scale, s * scale, pc[2])

    @tack.func
    def nearest_point(self, pts, x, pc):
        return _nearest_on_triangle(_point(pts, 0), _point(pts, 1), _point(pts, 2), x)

    @tack.func
    def edge_corner(self, e, k):
        return _TRIANGLE_EDGES[2 * e + k]


@tack.device_class
class _Quadrilateral(_Surface):
    """Bilinear shape functions over the unit square; ``_corner`` orders the points."""

    NUM_POINTS = 4
    NUM_EDGES = 4
    NUM_FACES = 4

    @tack.func
    def parametric_point(self, j):
        a, b = self._corner(j)
        return tack.vec3(tack.f32(a), tack.f32(b), 0.0)

    @tack.func
    def parametric_center(self):
        return tack.vec3(0.5, 0.5, 0.0)

    @tack.func
    def shape_function(self, j, pc):
        a, b = self._corner(j)
        return _linear(a, pc[0]) * _linear(b, pc[1])

    @tack.func
    def shape_gradient(self, j, pc):
        a, b = self._corner(j)
        # d/dx of _linear(a, x) is +1 at the end a = 1 and -1 at a = 0.
        return tack.vec3((2.0 * a - 1.0) * _linear(b, pc[1]),
                         _linear(a, pc[0]) * (2.0 * b - 1.0), 0.0)

    #: The nine-node biquadratic quad, a hexahedron's face: corners, edge
    #: middles and the center.
    NUM_QUADRATIC = 9
    QUADRATIC_INTERIOR = 1

    @tack.func
    def quadratic_function(self, k, pc):
        n = self.quadratic_node(k)
        return _q1d(n[0], pc[0]) * _q1d(n[1], pc[1])

    @tack.func
    def quadratic_gradient(self, k, pc):
        n = self.quadratic_node(k)
        return tack.vec3(_q1d_derivative(n[0], pc[0]) * _q1d(n[1], pc[1]),
                         _q1d(n[0], pc[0]) * _q1d_derivative(n[1], pc[1]), 0.0)

    @tack.func
    def is_inside(self, pc, tol):
        lo = -tol
        hi = 1.0 + tol
        return lo <= pc[0] and pc[0] <= hi and lo <= pc[1] and pc[1] <= hi

    @tack.func
    def _clamp(self, pc):
        return tack.vec3(min(max(pc[0], 0.0), 1.0), min(max(pc[1], 0.0), 1.0), pc[2])

    @tack.func
    def nearest_point(self, pts, x, pc):
        """The nearest point of the quad's two triangles, split at its first
        corner: exact when the quad is planar. Its clamped position, used before
        (TA-39), is exact only for a rectangle."""
        a = _point(pts, self.face_corner(0, 0))
        b = _point(pts, self.face_corner(1, 0))
        c = _point(pts, self.face_corner(2, 0))
        d = _point(pts, self.face_corner(3, 0))
        p = _nearest_on_triangle(a, b, c, x)
        q = _nearest_on_triangle(a, c, d, x)
        return p if (p - x).dot(p - x) <= (q - x).dot(q - x) else q


@tack.device_class
class Pixel(_Quadrilateral):
    """VTK_PIXEL: an axis-aligned quad, points in x-fastest order."""

    ID = PIXEL

    @tack.func
    def _corner(self, j):
        return j & 1, (j >> 1) & 1

    @tack.func
    def edge_corner(self, e, k):
        return _PIXEL_EDGES[2 * e + k]

    @tack.func
    def face_corner(self, f, k):
        return _PIXEL_FACES[2 * f + k]


@tack.device_class
class Quad(_Quadrilateral):
    """VTK_QUAD: four points counterclockwise."""

    ID = QUAD

    @tack.func
    def _corner(self, j):
        return (j ^ (j >> 1)) & 1, (j >> 1) & 1

    @tack.func
    def edge_corner(self, e, k):
        return _QUAD_EDGES[2 * e + k]


@tack.device_class
class Tetra(_Solid):
    """VTK_TETRA: shape functions ``1 - r - s - t``, ``r``, ``s`` and ``t``."""

    ID = TETRA
    NUM_POINTS = 4
    NUM_EDGES = 6
    NUM_FACES = 4

    CONTOUR_TRIANGLES = _ct.TETRA_MAX_TRIANGLES

    @tack.func
    def contour_count(self, case):
        return _ct.TETRA_CASES[case]

    @tack.func
    def contour_edge(self, case, k):
        return _ct.TETRA_EDGES[case * (3 * self.CONTOUR_TRIANGLES) + k]

    @tack.func
    def parametric_point(self, j):
        return tack.vec3(1.0 if j == 1 else 0.0, 1.0 if j == 2 else 0.0,
                         1.0 if j == 3 else 0.0)

    @tack.func
    def parametric_center(self):
        return tack.vec3(0.25, 0.25, 0.25)

    @tack.func
    def shape_function(self, j, pc):
        if j == 0:
            return 1.0 - pc[0] - pc[1] - pc[2]
        return pc[j - 1]

    @tack.func
    def shape_gradient(self, j, pc):
        if j == 0:
            return tack.vec3(-1.0, -1.0, -1.0)
        return self.parametric_point(j)

    NUM_QUADRATIC = 10

    @tack.func
    def quadratic_function(self, k, pc):
        n = self.quadratic_node(k)
        return (_p2(1.0 - n[0] - n[1] - n[2], 1.0 - pc[0] - pc[1] - pc[2])
                * _p2(n[0], pc[0]) * _p2(n[1], pc[1]) * _p2(n[2], pc[2]))

    @tack.func
    def quadratic_gradient(self, k, pc):
        n = self.quadratic_node(k)
        m0 = 1.0 - n[0] - n[1] - n[2]
        x0 = 1.0 - pc[0] - pc[1] - pc[2]
        f0 = _p2(m0, x0)
        f1 = _p2(n[0], pc[0])
        f2 = _p2(n[1], pc[1])
        f3 = _p2(n[2], pc[2])
        d0 = _p2_derivative(m0, x0) * f1 * f2 * f3          # by x0, which falls with r, s, t
        return tack.vec3(f0 * _p2_derivative(n[0], pc[0]) * f2 * f3 - d0,
                         f0 * f1 * _p2_derivative(n[1], pc[1]) * f3 - d0,
                         f0 * f1 * f2 * _p2_derivative(n[2], pc[2]) - d0)

    @tack.func
    def is_inside(self, pc, tol):
        return (-tol <= pc[0] and -tol <= pc[1] and -tol <= pc[2]
                and pc[0] + pc[1] + pc[2] <= 1.0 + tol)

    @tack.func
    def _clamp(self, pc):
        r = max(pc[0], 0.0)
        s = max(pc[1], 0.0)
        t = max(pc[2], 0.0)
        scale = 1.0 / max(r + s + t, 1.0)
        return tack.vec3(r * scale, s * scale, t * scale)

    @tack.func
    def edge_corner(self, e, k):
        return _TETRA_EDGES[2 * e + k]

    @tack.func
    def face_corner(self, f, k):
        return _TETRA_FACES[4 * f + k]

    @tack.func
    def face_shape(self, f):
        return TRIANGLE


@tack.device_class
class _Box(_Solid):
    """Trilinear shape functions over the unit cube; ``_corner`` orders the points."""

    NUM_POINTS = 8
    NUM_EDGES = 12
    NUM_FACES = 6

    @tack.func
    def parametric_point(self, j):
        a, b, c = self._corner(j)
        return tack.vec3(tack.f32(a), tack.f32(b), tack.f32(c))

    @tack.func
    def parametric_center(self):
        return tack.vec3(0.5, 0.5, 0.5)

    @tack.func
    def shape_function(self, j, pc):
        a, b, c = self._corner(j)
        return _linear(a, pc[0]) * _linear(b, pc[1]) * _linear(c, pc[2])

    @tack.func
    def shape_gradient(self, j, pc):
        a, b, c = self._corner(j)
        fa = _linear(a, pc[0])
        fb = _linear(b, pc[1])
        fc = _linear(c, pc[2])
        return tack.vec3((2.0 * a - 1.0) * fb * fc, fa * (2.0 * b - 1.0) * fc,
                         fa * fb * (2.0 * c - 1.0))

    NUM_QUADRATIC = 27
    QUADRATIC_FACES = 6
    QUADRATIC_INTERIOR = 1

    @tack.func
    def quadratic_function(self, k, pc):
        n = self.quadratic_node(k)
        return _q1d(n[0], pc[0]) * _q1d(n[1], pc[1]) * _q1d(n[2], pc[2])

    @tack.func
    def quadratic_gradient(self, k, pc):
        n = self.quadratic_node(k)
        fa = _q1d(n[0], pc[0])
        fb = _q1d(n[1], pc[1])
        fc = _q1d(n[2], pc[2])
        return tack.vec3(_q1d_derivative(n[0], pc[0]) * fb * fc,
                         fa * _q1d_derivative(n[1], pc[1]) * fc,
                         fa * fb * _q1d_derivative(n[2], pc[2]))

    @tack.func
    def is_inside(self, pc, tol):
        lo = -tol
        hi = 1.0 + tol
        return (lo <= pc[0] and pc[0] <= hi and lo <= pc[1] and pc[1] <= hi
                and lo <= pc[2] and pc[2] <= hi)

    @tack.func
    def _clamp(self, pc):
        return tack.vec3(min(max(pc[0], 0.0), 1.0), min(max(pc[1], 0.0), 1.0),
                         min(max(pc[2], 0.0), 1.0))


@tack.device_class
class Voxel(_Box):
    """VTK_VOXEL: an axis-aligned hexahedron, points in x-fastest order."""

    ID = VOXEL

    CONTOUR_TRIANGLES = _ct.VOXEL_MAX_TRIANGLES

    @tack.func
    def contour_count(self, case):
        return _ct.VOXEL_CASES[case]

    @tack.func
    def contour_edge(self, case, k):
        return _ct.VOXEL_EDGES[case * (3 * self.CONTOUR_TRIANGLES) + k]

    @tack.func
    def _corner(self, j):
        return j & 1, (j >> 1) & 1, (j >> 2) & 1

    @tack.func
    def edge_corner(self, e, k):
        return _VOXEL_EDGES[2 * e + k]

    @tack.func
    def face_corner(self, f, k):
        return _VOXEL_FACES[4 * f + k]

    @tack.func
    def face_shape(self, f):
        return QUAD


@tack.device_class
class Hexahedron(_Box):
    """VTK_HEXAHEDRON: the bottom face counterclockwise, then the top above it."""

    ID = HEXAHEDRON

    CONTOUR_TRIANGLES = _ct.HEXAHEDRON_MAX_TRIANGLES

    @tack.func
    def contour_count(self, case):
        return _ct.HEXAHEDRON_CASES[case]

    @tack.func
    def contour_edge(self, case, k):
        return _ct.HEXAHEDRON_EDGES[case * (3 * self.CONTOUR_TRIANGLES) + k]

    @tack.func
    def _corner(self, j):
        return (j ^ (j >> 1)) & 1, (j >> 1) & 1, (j >> 2) & 1

    @tack.func
    def edge_corner(self, e, k):
        return _HEXAHEDRON_EDGES[2 * e + k]

    @tack.func
    def face_corner(self, f, k):
        return _HEXAHEDRON_FACES[4 * f + k]

    @tack.func
    def face_shape(self, f):
        return QUAD


@tack.device_class
class Wedge(_Solid):
    """VTK_WEDGE: a triangle's shape functions in (r, s) times ``1 - t`` or ``t``."""

    ID = WEDGE
    NUM_POINTS = 6
    NUM_EDGES = 9
    NUM_FACES = 5

    CONTOUR_TRIANGLES = _ct.WEDGE_MAX_TRIANGLES

    @tack.func
    def contour_count(self, case):
        return _ct.WEDGE_CASES[case]

    @tack.func
    def contour_edge(self, case, k):
        return _ct.WEDGE_EDGES[case * (3 * self.CONTOUR_TRIANGLES) + k]

    @tack.func
    def parametric_point(self, j):
        k = j % 3
        return tack.vec3(1.0 if k == 1 else 0.0, 1.0 if k == 2 else 0.0,
                         tack.f32(j // 3))

    @tack.func
    def parametric_center(self):
        return tack.vec3(1.0 / 3.0, 1.0 / 3.0, 0.5)

    @tack.func
    def _triangle(self, k, pc):
        """The triangle's shape function ``k`` and its r and s derivatives."""
        if k == 0:
            return 1.0 - pc[0] - pc[1], -1.0, -1.0
        if k == 1:
            return pc[0], 1.0, 0.0
        return pc[1], 0.0, 1.0

    @tack.func
    def shape_function(self, j, pc):
        w, _, _ = self._triangle(j % 3, pc)
        return w * _linear(j // 3, pc[2])

    @tack.func
    def shape_gradient(self, j, pc):
        w, wr, ws = self._triangle(j % 3, pc)
        c = j // 3
        fc = _linear(c, pc[2])
        return tack.vec3(wr * fc, ws * fc, w * (2.0 * c - 1.0))

    NUM_QUADRATIC = 18
    QUADRATIC_FACES = 3

    @tack.func
    def quadratic_function(self, k, pc):
        n = self.quadratic_node(k)
        return (_p2(1.0 - n[0] - n[1], 1.0 - pc[0] - pc[1]) * _p2(n[0], pc[0])
                * _p2(n[1], pc[1]) * _q1d(n[2], pc[2]))

    @tack.func
    def quadratic_gradient(self, k, pc):
        n = self.quadratic_node(k)
        m0 = 1.0 - n[0] - n[1]
        x0 = 1.0 - pc[0] - pc[1]
        f0 = _p2(m0, x0)
        f1 = _p2(n[0], pc[0])
        f2 = _p2(n[1], pc[1])
        ft = _q1d(n[2], pc[2])
        d0 = _p2_derivative(m0, x0) * f1 * f2
        return tack.vec3((f0 * _p2_derivative(n[0], pc[0]) * f2 - d0) * ft,
                         (f0 * f1 * _p2_derivative(n[1], pc[1]) - d0) * ft,
                         f0 * f1 * f2 * _q1d_derivative(n[2], pc[2]))

    @tack.func
    def is_inside(self, pc, tol):
        return (-tol <= pc[0] and -tol <= pc[1] and pc[0] + pc[1] <= 1.0 + tol
                and -tol <= pc[2] and pc[2] <= 1.0 + tol)

    @tack.func
    def _clamp(self, pc):
        r = max(pc[0], 0.0)
        s = max(pc[1], 0.0)
        scale = 1.0 / max(r + s, 1.0)
        return tack.vec3(r * scale, s * scale, min(max(pc[2], 0.0), 1.0))

    @tack.func
    def edge_corner(self, e, k):
        return _WEDGE_EDGES[2 * e + k]

    @tack.func
    def face_corner(self, f, k):
        return _WEDGE_FACES[4 * f + k]

    @tack.func
    def face_shape(self, f):
        return TRIANGLE if f < 2 else QUAD


@tack.device_class
class Pyramid(_Solid):
    """VTK_PYRAMID: the quad's shape functions times ``1 - t`` at the base, ``t`` at the apex."""

    ID = PYRAMID
    NUM_POINTS = 5
    NUM_EDGES = 8
    NUM_FACES = 5

    CONTOUR_TRIANGLES = _ct.PYRAMID_MAX_TRIANGLES

    @tack.func
    def contour_count(self, case):
        return _ct.PYRAMID_CASES[case]

    @tack.func
    def contour_edge(self, case, k):
        return _ct.PYRAMID_EDGES[case * (3 * self.CONTOUR_TRIANGLES) + k]

    @tack.func
    def _corner(self, j):
        return (j ^ (j >> 1)) & 1, (j >> 1) & 1

    @tack.func
    def parametric_point(self, j):
        if j == 4:
            return tack.vec3(0.0, 0.0, 1.0)
        a, b = self._corner(j)
        return tack.vec3(tack.f32(a), tack.f32(b), 0.0)

    @tack.func
    def parametric_center(self):
        return tack.vec3(0.4, 0.4, 0.2)

    @tack.func
    def shape_function(self, j, pc):
        if j == 4:
            return pc[2]
        a, b = self._corner(j)
        return _linear(a, pc[0]) * _linear(b, pc[1]) * (1.0 - pc[2])

    @tack.func
    def shape_gradient(self, j, pc):
        if j == 4:
            return tack.vec3(0.0, 0.0, 1.0)
        a, b = self._corner(j)
        fa = _linear(a, pc[0])
        fb = _linear(b, pc[1])
        return tack.vec3((2.0 * a - 1.0) * fb * (1.0 - pc[2]),
                         fa * (2.0 * b - 1.0) * (1.0 - pc[2]), -fa * fb)

    @tack.func
    def world_to_parametric(self, pts, x):
        """As ``Shape.world_to_parametric``, but at the apex, where every
        ``(r, s)`` at ``t = 1`` is the same point: the map's Jacobian is singular
        there, Newton did not converge, and no cell held a mesh point at a
        pyramid's apex. Within ``APEX_SNAP`` of the pyramid's size of the apex,
        ``x`` is taken to be at it: ``(0.5, 0.5, 1)``, converged, where every
        base corner's shape function is 0 and the apex's 1. A point Newton cannot
        place within ``APEX_REACH`` of the apex is outside beside it: just past
        the apex (``APEX_OUTSIDE``), converged, held by no margin."""
        apex = _point(pts, 4)
        size = (_point(pts, 0) - apex).norm()
        for j in range(1, 4):
            size = max(size, (_point(pts, j) - apex).norm())
        d = x - apex
        reach = APEX_SNAP * size
        pc = tack.vec3(0.5, 0.5, 1.0)
        converged = 1
        if d.dot(d) > reach * reach:
            pc, converged = self._newton(pts, x)
            near = APEX_REACH * size
            if converged == 0 and d.dot(d) <= near * near:
                pc = tack.vec3(0.5, 0.5, 1.0 + APEX_OUTSIDE)
                converged = 1
        return pc, converged

    @tack.func
    def is_inside(self, pc, tol):
        lo = -tol
        hi = 1.0 + tol
        return (lo <= pc[0] and pc[0] <= hi and lo <= pc[1] and pc[1] <= hi
                and lo <= pc[2] and pc[2] <= hi)

    @tack.func
    def _clamp(self, pc):
        return tack.vec3(min(max(pc[0], 0.0), 1.0), min(max(pc[1], 0.0), 1.0),
                         min(max(pc[2], 0.0), 1.0))

    @tack.func
    def edge_corner(self, e, k):
        return _PYRAMID_EDGES[2 * e + k]

    @tack.func
    def face_corner(self, f, k):
        return _PYRAMID_FACES[4 * f + k]

    @tack.func
    def face_shape(self, f):
        return QUAD if f == 0 else TRIANGLE


#: The ten shape classes, in id order.
SHAPES = (Vertex, Line, Triangle, Pixel, Quad, Tetra, Voxel, Hexahedron, Wedge, Pyramid)

_BY_ID = {int(cls.ID): cls for cls in SHAPES}

POLY_LINE = tack.constant(4, tack.i32)
POLYGON = tack.constant(7, tack.i32)
POLYHEDRON = tack.constant(42, tack.i32)


class _NoReference:
    """A cell or face without a reference element: an id and a dimension, and the
    counts host-side code reads (none fixed). Not a ``Shape``, whose methods all
    assume fixed points and shape functions."""

    NUM_EDGES = 0
    NUM_FACES = 0
    CONTOUR_TRIANGLES = 0
    NUM_QUADRATIC = 0


@tack.device_class
class PolyLine(_NoReference):
    """VTK_POLY_LINE: a cell of a polyline topology, a chain of any number of points
    (two or more). No fixed counts and no shape functions; its facets are its two
    end points, and a cell view gives its chain at run time."""

    ID = POLY_LINE
    DIMENSION = 1


@tack.device_class
class Polygon(_NoReference):
    """VTK_POLYGON: a face of a polyhedral topology, of any number of points. It has no
    fixed counts and no shape functions; a face view gives its size at run time
    (``face_size``). Not in ``SHAPES``: no shape-based topology holds polygons."""

    ID = POLYGON
    DIMENSION = 2


@tack.device_class
class Polyhedron(_NoReference):
    """VTK_POLYHEDRON: a cell of a polyhedral topology, given by its faces. No fixed
    counts, no reference element, no shape functions: a cell view gives its faces,
    and their points in its outward order, at run time."""

    ID = POLYHEDRON
    DIMENSION = 3


def shape_class(type_id):
    """The shape class for a VTK cell type id; raises ``ValueError`` for any other id."""
    try:
        return _BY_ID[int(type_id)]
    except KeyError:
        raise ValueError(
            f"cell type {type_id} is not one of the linear shapes "
            f"{sorted(_BY_ID)}") from None
