"""Cells and fields reached by cell id, for work that runs from the points.

The cell-side search (``CellLocator``'s, through each group's view) cannot
look a cell up by id: a view addresses its cells its own way. Work that runs
from the points -- a particle asking for its cell, a probe point -- needs the
opposite, and this module gives it: the topology's cells by id (``_FlatCells``,
``_GridCells``, ``_PolyhedralMesh``), a search over a locator's cell bins that
ranks its candidates as the locator does (``_MeshSearch.find``: holding the
point ahead of coming within the tolerance, then the smaller id), and fields
evaluated by cell id (``_PointValues``, ``_CellValues``: ``value(mesh, e, kind,
pc, tet)``). Each shape's code is compiled in only when the mesh has that
shape (``_kinds``).

What it serves: an order-1 ``H1`` geometry over unstructured cells of the
shapes in ``_SHAPES``, 2D and 3D structured grids, and polyhedra; point data
(``H1`` order 1, values on points) and cell data (``Constant``, values on
cells). Anything else keeps the cell-side path.
"""

import tack
from tack import arrays
from tack.dataset import shapes, views
from tack.dataset.spaces import H1, Constant, Values

__all__ = []

_SHAPES = (shapes.Tetra, shapes.Hexahedron, shapes.Wedge, shapes.Pyramid, shapes.Voxel,
           shapes.Triangle, shapes.Quad, shapes.Pixel)

# The kinds bit for polyhedra, past the shapes'.
_POLYHEDRA = 256


@tack.func
def _bin(x, origin, inverse, n, slack):
    """The bin along one axis of coordinate ``x``, the grid widened by ``slack``
    on both sides (its edge bins take the margin); -1 outside that."""
    f = (x - origin) * inverse
    reach = slack * inverse
    i = tack.i64(-1)
    if f >= -reach and f <= n + reach:
        i = min(max(tack.i64(floor(f)), 0), n - 1)
    return i


# ── Cells by id ─────────────────────────────────────────────────────

class _NoPolyhedra:
    """For meshes of shapes: the methods the search names for cells with no
    reference element, never called (their branch is compiled out)."""

    NO_REFERENCE = 0

    @tack.func
    def locate(self, e, p, slack):
        return -1, p - p, 0

    @tack.func
    def tet_value_of(self, field, e, t, pc):
        return field.values[0]


class _FannedCells:
    """A polyhedral or polygonal topology's cells by id, with the positions they
    read: located by their views' code, point data evaluated in a fan's piece."""

    NO_REFERENCE = 1

    def __init__(self, topology, positions):
        point_offsets, point_ids = topology.cell_points()
        super().__init__(topology.cell_offsets, topology.cell_faces, topology.cell_face_sides,
                         topology.face_offsets, topology.face_points, point_offsets,
                         point_ids, topology.num_cells)
        self.positions = positions

    @tack.func
    def point_position(self, p):
        return self.positions[p]

    @tack.func
    def tet_value_of(self, field, e, t, pc):
        """Point data at ``pc`` in sub-tetrahedron ``t``: linear there, the cell and
        face points taking the means of their points' values."""
        k, j = self.tet_face(e, t)
        n = self.side_num_points(e, k)
        nxt = j + 1 if j + 1 < n else 0
        count = self.num_points(e)
        v0 = field.values[self.point_id(e, 0)]
        for i in range(1, count):
            v0 += field.values[self.point_id(e, i)]
        v0 = v0 / count
        vf = field.values[self.side_point_id(e, k, 0)]
        for i in range(1, n):
            vf += field.values[self.side_point_id(e, k, i)]
        vf = vf / n
        return (v0 + pc[0] * (vf - v0)
                + pc[1] * (field.values[self.side_point_id(e, k, j)] - v0)
                + pc[2] * (field.values[self.side_point_id(e, k, nxt)] - v0))


@tack.device_class
class _PolyhedralMesh(_FannedCells, views._PolyhedralLocation, views._PolyhedralWalk,
                      views._PolyhedralCells):
    """A polyhedral topology's cells by id, located in their sub-tetrahedra
    (``views._PolyhedralLocation``)."""

    @tack.func
    def cell_type(self, e):
        return shapes.POLYHEDRON


@tack.device_class
class _PolygonMesh(_FannedCells, views._PolygonLocation, views._PolyhedralWalk,
                   views._PolyhedralCells):
    """A polygonal topology's cells by id, located in the triangles fanning them
    (``views._PolygonLocation``)."""

    @tack.func
    def cell_type(self, e):
        return shapes.POLYGON


@tack.device_class
class _PolylineMesh(views._PolylineLocation, views._PolyhedralWalk, views._PolylineCells):
    """A polyline topology's cells by id, located on their segments
    (``views._PolylineLocation``), with the positions they read."""

    NO_REFERENCE = 1

    def __init__(self, topology, positions):
        point_offsets, point_ids = topology.cell_points()
        super().__init__(topology.line_offsets, topology.line_points, point_offsets,
                         point_ids, topology.num_cells)
        self.positions = positions

    @tack.func
    def point_position(self, p):
        return self.positions[p]

    @tack.func
    def cell_type(self, e):
        return shapes.POLY_LINE

    @tack.func
    def tet_value_of(self, field, e, t, pc):
        """Point data at ``pc`` on segment ``t``: linear along it."""
        return (pc[1] * field.values[self.line_point_id(e, t)]
                + pc[2] * field.values[self.line_point_id(e, t + 1)])


@tack.device_class
class _FlatCells(_NoPolyhedra):
    """An unstructured topology's cells by id: type and points from its arrays."""

    def __init__(self, types, offsets, connectivity):
        self.types = types
        self.offsets = offsets
        self.connectivity = connectivity

    @tack.func
    def cell_type(self, e):
        return tack.i32(self.types[e])

    @tack.func
    def num_points(self, e):
        return self.offsets[e + 1] - self.offsets[e]

    @tack.func
    def point_id(self, e, j):
        return self.connectivity[self.offsets[e] + j]


@tack.device_class
class _GridCells(_NoPolyhedra):
    """A structured grid's cells by id: hexahedra (quads in 2D), x fastest, corners
    in VTK's order."""

    def __init__(self, point_dims):
        dims = tuple(point_dims) + (1,) * (3 - len(point_dims))
        self.nx, self.ny = dims[0], dims[1]
        self.cx, self.cy = dims[0] - 1, max(dims[1] - 1, 1)
        self.dimension = len(point_dims)

    @tack.func
    def cell_type(self, e):
        return shapes.HEXAHEDRON if self.dimension == 3 else shapes.QUAD

    @tack.func
    def num_points(self, e):
        return 8 if self.dimension == 3 else 4

    @tack.func
    def point_id(self, e, j):
        i = e % self.cx
        rest = e // self.cx
        k = rest // self.cy
        return (i + ((j ^ (j >> 1)) & 1)) + ((rest - k * self.cy + ((j >> 1) & 1))
                                             + (k + (j >> 2)) * self.ny) * self.nx


_kinds_classes = {}


def _kinds(present):
    """A template whose class constant ``PRESENT`` has a bit for each of ``_SHAPES``
    the mesh has (and ``_POLYHEDRA``), so a kernel compiles only those shapes' code."""
    if present not in _kinds_classes:
        _kinds_classes[present] = tack.device_class(
            type(f"_Kinds{present}", (), {"PRESENT": present}))
    return _kinds_classes[present]()


def _as_vectors(values, dtype):
    """``values`` (3-vectors) as an array of the dataset's precision."""
    if values.scalar_dtype == dtype:
        return values
    out = tack.empty((arrays.size_of(values),), tack.vec(3, dtype))
    if arrays.size_of(values):
        tack.copyto(out, values.numpy().astype(dtype.numpy_dtype))
    return out


def cells_by_id(data, positions, strict=True):
    """``(cells, present)``: ``data``'s cells by id and the bits of the shapes it
    has, or ``None`` if this module cannot serve its topology. Unless
    ``strict``, unstructured cells of other shapes are allowed and never found
    (as advection has always taken them)."""
    from tack.dataset.polyhedra import PolygonalTopology, PolyhedralTopology, PolylineTopology
    from tack.dataset.topology import StructuredTopology, UnstructuredTopology

    t = data.topology
    present = 0
    for group in t.groups() if hasattr(t, "groups") else ():
        if not group.count:
            continue
        if group.shape in _SHAPES:
            present |= 1 << _SHAPES.index(group.shape)
        elif strict and isinstance(t, UnstructuredTopology):
            return None
    if isinstance(t, UnstructuredTopology):
        return _FlatCells(t.types, t.offsets, t.connectivity), present
    if isinstance(t, StructuredTopology) and len(t.point_dims) >= 2:
        return _GridCells(t.point_dims), present
    if isinstance(t, PolygonalTopology):
        return _PolygonMesh(t, positions), _POLYHEDRA if t.num_cells else 0
    if isinstance(t, PolyhedralTopology):
        return _PolyhedralMesh(t, positions), _POLYHEDRA if t.num_cells else 0
    if isinstance(t, PolylineTopology):
        return _PolylineMesh(t, positions), _POLYHEDRA if t.num_cells else 0
    return None


# ── The search ──────────────────────────────────────────────────────

@tack.device_class
class _MeshSearch:
    """A point's cell by the locator's cell bins (``CellLocator.cell_bins``), ranked
    as the locator ranks them, and the shapes' functions by cell id."""

    def __init__(self, cells, present, positions, locator, tolerance):
        self.cells = cells
        self.kinds = _kinds(present)
        self.tetra = shapes.Tetra()
        self.hexa = shapes.Hexahedron()
        self.wedge = shapes.Wedge()
        self.pyramid = shapes.Pyramid()
        self.voxel = shapes.Voxel()
        self.triangle = shapes.Triangle()
        self.quad = shapes.Quad()
        self.pixel = shapes.Pixel()
        self.positions = positions
        self.lo = locator.lo
        self.hi = locator.hi
        self.bin_offsets, self.bin_cells = locator.cell_bins()
        (self.ox, self.oy, self.oz, self.ix, self.iy, self.iz,
         self.nx, self.ny, self.nz) = locator._grid()
        self.slack = float(locator.slack)
        self.tolerance = float(tolerance)
        self.ncells = locator.data.num_cells

    @tack.func
    def find(self, p):
        """``(e, kind, pc, tet)``: the cell holding ``p`` -- the smallest id among
        those holding it, else among those within their tolerance -- its type,
        and ``p``'s coordinates there (in a polyhedron, its sub-tetrahedron and
        the coordinates in that); ``e`` is ``ncells`` for none."""
        best = self.ncells
        best_rank = 2 * self.ncells
        best_type = 0
        best_pc = p - p
        best_tet = -1
        pts = tack.local_array_like(self.lo, 24)
        bi = _bin(p[0], self.ox, self.ix, self.nx, self.slack)
        bj = _bin(p[1], self.oy, self.iy, self.ny, self.slack)
        bk = _bin(p[2], self.oz, self.iz, self.nz, self.slack)
        if bi >= 0 and bj >= 0 and bk >= 0:
            b = bi + self.nx * (bj + self.ny * bk)
            for m in range(self.bin_offsets[b], self.bin_offsets[b + 1]):
                e = self.bin_cells[m]
                a = self.lo[e]
                c = self.hi[e]
                s = self.tolerance * (c - a).norm()
                if (e < best_rank and p[0] >= a[0] - s and p[0] <= c[0] + s
                        and p[1] >= a[1] - s and p[1] <= c[1] + s
                        and p[2] >= a[2] - s and p[2] <= c[2] + s):
                    kind = self.cells.cell_type(e)
                    if self.cells.NO_REFERENCE == 1:
                        # No parametric coordinates: a piece of the cell's fan (a
                        # polyline's segment) and the point's place in it, within
                        # the cell's tolerance.
                        tet, tpc, held = self.cells.locate(e, p, s)
                        if tet >= 0:
                            rank = e if held == 1 else e + self.ncells
                            if rank < best_rank:
                                best_rank = rank
                                best = e
                                best_type = kind
                                best_tet = tet
                                best_pc = tpc
                        continue
                    for j in range(self.cells.num_points(e)):
                        y = self.positions[self.cells.point_id(e, j)]
                        pts[3 * j] = y[0]
                        pts[3 * j + 1] = y[1]
                        pts[3 * j + 2] = y[2]
                    pc = p - p
                    converged = 0
                    inside = 0
                    near = p - p
                    surface = 0
                    if (self.kinds.PRESENT & 1) != 0 and kind == shapes.TETRA:
                        pc, converged = self.tetra.world_to_parametric(pts, p)
                        inside = self.tetra.is_inside(pc, shapes.HOLD_TOLERANCE)
                        near = self.tetra.nearest_point(pts, p, pc)
                    elif (self.kinds.PRESENT & 2) != 0 and kind == shapes.HEXAHEDRON:
                        pc, converged = self.hexa.world_to_parametric(pts, p)
                        inside = self.hexa.is_inside(pc, shapes.HOLD_TOLERANCE)
                        near = self.hexa.nearest_point(pts, p, pc)
                    elif (self.kinds.PRESENT & 4) != 0 and kind == shapes.WEDGE:
                        pc, converged = self.wedge.world_to_parametric(pts, p)
                        inside = self.wedge.is_inside(pc, shapes.HOLD_TOLERANCE)
                        near = self.wedge.nearest_point(pts, p, pc)
                    elif (self.kinds.PRESENT & 8) != 0 and kind == shapes.PYRAMID:
                        pc, converged = self.pyramid.world_to_parametric(pts, p)
                        inside = self.pyramid.is_inside(pc, shapes.HOLD_TOLERANCE)
                        near = self.pyramid.nearest_point(pts, p, pc)
                    elif (self.kinds.PRESENT & 16) != 0 and kind == shapes.VOXEL:
                        pc, converged = self.voxel.world_to_parametric(pts, p)
                        inside = self.voxel.is_inside(pc, shapes.HOLD_TOLERANCE)
                        near = self.voxel.nearest_point(pts, p, pc)
                    elif (self.kinds.PRESENT & 32) != 0 and kind == shapes.TRIANGLE:
                        pc, converged = self.triangle.world_to_parametric(pts, p)
                        inside = self.triangle.is_inside(pc, shapes.HOLD_TOLERANCE)
                        surface = 1
                        near = self.triangle.nearest_point(pts, p, pc)
                    elif (self.kinds.PRESENT & 64) != 0 and kind == shapes.QUAD:
                        pc, converged = self.quad.world_to_parametric(pts, p)
                        inside = self.quad.is_inside(pc, shapes.HOLD_TOLERANCE)
                        surface = 1
                        near = self.quad.nearest_point(pts, p, pc)
                    elif (self.kinds.PRESENT & 128) != 0 and kind == shapes.PIXEL:
                        pc, converged = self.pixel.world_to_parametric(pts, p)
                        inside = self.pixel.is_inside(pc, shapes.HOLD_TOLERANCE)
                        surface = 1
                        near = self.pixel.nearest_point(pts, p, pc)
                    # As the locator: inside a solid, or within the tolerance of
                    # the cell's nearest point; holding it (a surface: over it)
                    # ranks first.
                    d = near - p
                    if converged == 1 and ((inside == 1 and surface == 0)
                                           or d.dot(d) <= s * s):
                        rank = e if inside == 1 else e + self.ncells
                        if rank < best_rank:
                            best_rank = rank
                            best = e
                            best_type = kind
                            best_pc = pc
        return best, best_type, best_pc, best_tet

    @tack.func
    def weight(self, kind, j, pc):
        """Shape function ``j`` of a cell of type ``kind`` at ``pc``."""
        w = 0.0
        if (self.kinds.PRESENT & 1) != 0 and kind == shapes.TETRA:
            w = self.tetra.shape_function(j, pc)
        elif (self.kinds.PRESENT & 2) != 0 and kind == shapes.HEXAHEDRON:
            w = self.hexa.shape_function(j, pc)
        elif (self.kinds.PRESENT & 4) != 0 and kind == shapes.WEDGE:
            w = self.wedge.shape_function(j, pc)
        elif (self.kinds.PRESENT & 8) != 0 and kind == shapes.PYRAMID:
            w = self.pyramid.shape_function(j, pc)
        elif (self.kinds.PRESENT & 16) != 0 and kind == shapes.VOXEL:
            w = self.voxel.shape_function(j, pc)
        elif (self.kinds.PRESENT & 32) != 0 and kind == shapes.TRIANGLE:
            w = self.triangle.shape_function(j, pc)
        elif (self.kinds.PRESENT & 64) != 0 and kind == shapes.QUAD:
            w = self.quad.shape_function(j, pc)
        elif (self.kinds.PRESENT & 128) != 0 and kind == shapes.PIXEL:
            w = self.pixel.shape_function(j, pc)
        return w


def mesh_search(data, locator, positions=None, strict=True):
    """A ``_MeshSearch`` over ``data`` with ``locator``'s bins, or ``None`` where
    this module cannot serve the dataset (see the module's docstring)."""
    from tack.dataset.locator import _TOLERANCE

    geometry = data.geometry
    # No cells is served too: every point is outside.
    if not isinstance(geometry.space, H1) or geometry.space.order != 1:
        return None
    if positions is None:
        positions = _as_vectors(arrays.materialize(geometry.values), data.dtype)
    found = cells_by_id(data, positions, strict=strict)
    if found is None:
        return None
    cells, present = found
    return _MeshSearch(cells, present, positions, locator, _TOLERANCE)


# ── Fields by cell id ───────────────────────────────────────────────

@tack.device_class
class _PointValues:
    """Point data by cell id: interpolated by the cell's shape functions, or in a
    polyhedron linear on its sub-tetrahedron."""

    def __init__(self, values):
        self.values = values

    @tack.func
    def value(self, mesh, e, kind, pc, tet):
        v = self.values[mesh.cells.point_id(e, 0)]
        if mesh.cells.NO_REFERENCE == 1:
            v = mesh.cells.tet_value_of(self, e, tet, pc)
        else:
            v = v * mesh.weight(kind, 0, pc)
            for j in range(1, mesh.cells.num_points(e)):
                v += self.values[mesh.cells.point_id(e, j)] * mesh.weight(kind, j, pc)
        return v


@tack.device_class
class _CellValues:
    """Cell data by cell id: constant in each cell."""

    def __init__(self, values):
        self.values = values

    @tack.func
    def value(self, mesh, e, kind, pc, tet):
        return self.values[e]


def field_by_id(data, field):
    """``field`` read by cell id (``_PointValues`` or ``_CellValues`` over its
    materialized values), or ``None`` for a space this module does not serve."""
    space = field.space
    if (isinstance(space, H1) and space.order == 1) or space is Values(data, "points"):
        return _PointValues(arrays.materialize(field.values))
    if isinstance(space, Constant) or space is Values(data, "cells"):
        return _CellValues(arrays.materialize(field.values))
    return None

