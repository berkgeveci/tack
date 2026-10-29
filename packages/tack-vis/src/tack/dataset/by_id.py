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

What it serves: an ``H1`` geometry over unstructured cells of the shapes in
``_SHAPES``, 2D and 3D structured grids, and polyhedra (order 1 there); point
data (``H1`` order 1, values on points), cell data (``Constant``, values on
cells), and any other interpolated space of one layout per shape. A curved
geometry (``H1`` above order 1) and the fields of other spaces are read
through each shape group's own view (``_ByShape``: a cell's place in its group,
``local``), so curved cells are inverted (``curved_to_parametric``) and
high-order fields evaluated as the cell-side path does it. Anything else keeps
the cell-side path.
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


@tack.func
def _span(low, high, origin, inverse, n):
    """``(f0, f1, i0, i1)``: an interval's ends in grid units along one axis, and
    the bins it overlaps (clamped to the grid; empty, ``i1 < i0``, past it)."""
    f0 = (low - origin) * inverse
    f1 = (high - origin) * inverse
    return f0, f1, max(tack.i64(floor(f0)), 0), min(tack.i64(floor(f1)), n - 1)


@tack.func
def _leaf(f, i, m):
    """The leaf along one axis, of ``m`` in coarse bin ``i``, of a coordinate at
    ``f`` in coarse grid units, clamped to the bin's leaves. The build and the
    search both place by this from the same ``f`` (``(x - origin) * inverse``):
    rounding and ``floor`` keep order, so a point inside a cell's widened box
    lands in one of the leaves the cell was entered in."""
    return min(max(tack.i64(floor((f - i) * m)), 0), m - 1)


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


# ── Cells and fields through each shape group's view ─────────────────

@tack.kernel
def _group_places(cells, local):
    # Each cell's index in its group (a structured grid's (i, j, k)), by cell id.
    for c in cells:
        local[cells.entity_id(c)] = c


def _by_shape(data, groups, view_of):
    """``(views, local, present)``: a view for each of ``_SHAPES`` (``view_of(group)``,
    the first group's standing in for a shape the mesh lacks, which a kernel never
    compiles), each cell's index in its group by id, and the shapes' bits; or
    ``None`` where a shape has two groups (an order per cell) or is not one of
    ``_SHAPES``, or there are no cells."""
    groups = [g for g in groups if g.count]
    kinds = [g.shape for g in groups]
    if not groups or len(set(kinds)) != len(kinds) or any(k not in _SHAPES for k in kinds):
        return None
    from tack.dataset.topology import StructuredTopology

    t = data.topology
    # A structured grid's views take a cell as its index vector.
    local = tack.empty((data.num_cells,), tack.vec(len(t.point_dims), tack.i64)
                       if isinstance(t, StructuredTopology) else tack.i64)
    views = [None] * len(_SHAPES)
    present = 0
    for group in groups:
        _group_places(group.view(), local)
        i = _SHAPES.index(group.shape)
        views[i] = view_of(group)
        present |= 1 << i
    first = next(v for v in views if v is not None)
    return [v if v is not None else first for v in views], local, present


class _ByShape:
    """A view per shape (``_SHAPES``' order), and each cell's index in its group, so a
    method of the view a cell's group has can be called by cell id."""

    def __init__(self, views, local, present):
        (self.tetra, self.hexa, self.wedge, self.pyramid, self.voxel, self.triangle,
         self.quad, self.pixel) = views
        self.local = local
        self.kinds = _kinds(present)


@tack.func
def _locate_curved(view, c, p, tolerance):
    # A curved cell inverted as it is (round 1's Newton on its own map), whether
    # it holds the point within ``tolerance``, and its nearest point there.
    pc, ok = view.curved_to_parametric(c, p)
    inside = 0
    near = p
    if ok == 1:
        inside = view.is_inside(pc, tolerance)
        near = view.position(c, view._clamp(pc))
    return pc, ok, inside, near


@tack.device_class
class _CurvedCells(_ByShape):
    """A curved geometry's cells by id, through their groups' domain views."""

    @tack.func
    def locate(self, e, kind, p, tolerance):
        """``(pc, converged, inside, near)`` of ``p`` in cell ``e`` of type ``kind``."""
        c = self.local[e]
        pc = p - p
        ok = 0
        inside = 0
        near = p
        if (self.kinds.PRESENT & 1) != 0 and kind == shapes.TETRA:
            pc, ok, inside, near = _locate_curved(self.tetra, c, p, tolerance)
        elif (self.kinds.PRESENT & 2) != 0 and kind == shapes.HEXAHEDRON:
            pc, ok, inside, near = _locate_curved(self.hexa, c, p, tolerance)
        elif (self.kinds.PRESENT & 4) != 0 and kind == shapes.WEDGE:
            pc, ok, inside, near = _locate_curved(self.wedge, c, p, tolerance)
        elif (self.kinds.PRESENT & 8) != 0 and kind == shapes.PYRAMID:
            pc, ok, inside, near = _locate_curved(self.pyramid, c, p, tolerance)
        elif (self.kinds.PRESENT & 16) != 0 and kind == shapes.VOXEL:
            pc, ok, inside, near = _locate_curved(self.voxel, c, p, tolerance)
        elif (self.kinds.PRESENT & 32) != 0 and kind == shapes.TRIANGLE:
            pc, ok, inside, near = _locate_curved(self.triangle, c, p, tolerance)
        elif (self.kinds.PRESENT & 64) != 0 and kind == shapes.QUAD:
            pc, ok, inside, near = _locate_curved(self.quad, c, p, tolerance)
        elif (self.kinds.PRESENT & 128) != 0 and kind == shapes.PIXEL:
            pc, ok, inside, near = _locate_curved(self.pixel, c, p, tolerance)
        return pc, ok, inside, near


@tack.device_class
class _OneCurvedShape:
    """``_CurvedCells`` for a mesh of one shape (most are), its one view called
    directly: the eight-way dispatch, every branch a curved cell's Newton and
    basis, took minutes to compile even with all but one never reached."""

    def __init__(self, view, local):
        self.view = view
        self.local = local

    @tack.func
    def locate(self, e, kind, p, tolerance):
        return _locate_curved(self.view, self.local[e], p, tolerance)


@tack.device_class
class _NoCurves:
    """A straight geometry: ``_MeshSearch`` never calls this (``CURVED`` 0)."""

    @tack.func
    def locate(self, e, kind, p, tolerance):
        return p - p, 0, 0, p


def curved_cells(data):
    """A curved geometry's ``_CurvedCells``, or ``None`` where ``_by_shape`` cannot
    serve the topology."""
    found = _by_shape(data, data.launch_groups("cells", []),
                      lambda group: data.domain_view("cells", group))
    if found is None:
        return None
    return _OneCurvedShape(*_one_shape(found)) if _single(found) else _CurvedCells(*found)


def _single(found):
    """Whether ``_by_shape``'s mesh has one shape."""
    return bin(found[2]).count("1") == 1


def _one_shape(found):
    """``(view, local)`` of a one-shape mesh's ``_by_shape``."""
    views, local, present = found
    return views[present.bit_length() - 1], local


# ── The search ──────────────────────────────────────────────────────

@tack.device_class
class _MeshSearch:
    """A point's cell by the locator's cell bins (``CellLocator.cell_bins``, of one
    level or two), ranked as the locator ranks them, and the shapes' functions
    by cell id."""

    #: ``find_from`` tries the cell it is given before the bins; 0 searches every
    #: time (the tests compare the two, bit for bit).
    PREVIOUS_CELL = 1

    #: 1 (an instance's, compiled in): ``find_from`` tries the given cell's face
    #: neighbours (``neighbors``, ``CellLocator.cell_neighbors``) before the bins.
    NEIGHBORS = 0
    MAX_FACES = 6

    #: 1 (an instance's, compiled in): a curved geometry, whose cells ``curved``
    #: locates through their groups' views (``_CurvedCells``).
    CURVED = 0

    def __init__(self, cells, present, positions, locator, tolerance, neighbors=None,
                 curved=None):
        self.cells = cells
        self.curved = curved if curved is not None else _NoCurves()
        if curved is not None:
            self.CURVED = 1
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
        self.leaf_dims, self.leaf_starts = locator.leaves()
        (self.ox, self.oy, self.oz, self.ix, self.iy, self.iz,
         self.nx, self.ny, self.nz) = locator._coarse_grid()
        self.slack = float(locator.slack)
        self.tolerance = float(tolerance)
        self.ncells = locator.data.num_cells
        self.neighbors = neighbors if neighbors is not None else tack.zeros((1,), tack.i32)
        if neighbors is not None:
            self.NEIGHBORS = 1

    @tack.func
    def find(self, p):
        """``(e, kind, pc, tet)``: the cell holding ``p`` -- the smallest id among
        those holding it, else among those within their tolerance -- its type,
        and ``p``'s coordinates there (in a polyhedron, its sub-tetrahedron and
        the coordinates in that); ``e`` is ``ncells`` for none."""
        e, kind, pc, tet, held = self.find_ranked(p)
        return e, kind, pc, tet

    @tack.func
    def find_from(self, p, hint):
        """``find``, trying cell ``hint`` first (``ncells`` for none): a particle's
        next location is mostly in the cell it was last found in. The hint is
        taken only where it holds ``p`` by ``WELL_INSIDE``, far more than
        ``HOLD_TOLERANCE``: no other cell of a mesh whose cells do not overlap
        holds ``p`` then, so the search would give the same cell, and the same
        coordinates, computed alike. Elsewhere, and for pyramids (whose
        coordinates collapse at the apex), surfaces and cells without a
        reference element, the bins are searched.

        With ``NEIGHBORS``, the hint's face neighbours are tried next, by the same
        rule: a point that left its cell is mostly in one across a face (78% of
        the misses on the ONERA M6, 43-49% on the aneurysm), which a few cells'
        tests find where the bins test 50-100."""
        e = self.ncells
        kind = 0
        pc = p - p
        tet = -1
        if self.PREVIOUS_CELL == 1 and self.cells.NO_REFERENCE == 0 and hint < self.ncells:
            e, kind, pc = self._well_inside(p, hint)
            if self.NEIGHBORS == 1 and e == self.ncells:
                for k in range(self.MAX_FACES):
                    f = self.neighbors[self.MAX_FACES * hint + k]
                    if f >= 0:
                        e, kind, pc = self._well_inside(p, f)
                        if e < self.ncells:
                            break
        if e == self.ncells:
            e, kind, pc, tet = self.find(p)
        return e, kind, pc, tet

    @tack.func
    def _well_inside(self, p, e):
        """``(e, kind, pc)`` if solid cell ``e`` holds ``p`` by ``WELL_INSIDE``,
        else ``(ncells, 0, 0)``."""
        pts = tack.local_array_like(self.lo, 24)
        kind = self.cells.cell_type(e)
        pc = p - p
        converged = 0
        inside = 0
        if self.CURVED == 1:
            if (kind == shapes.TETRA or kind == shapes.HEXAHEDRON or kind == shapes.WEDGE
                    or kind == shapes.VOXEL):
                pc, converged, inside, near = self.curved.locate(e, kind, p,
                                                                 -shapes.WELL_INSIDE)
        else:
            for j in range(self.cells.num_points(e)):
                y = self.positions[self.cells.point_id(e, j)]
                pts[3 * j] = y[0]
                pts[3 * j + 1] = y[1]
                pts[3 * j + 2] = y[2]
        if self.CURVED == 1:
            pass
        elif (self.kinds.PRESENT & 1) != 0 and kind == shapes.TETRA:
            pc, converged = self.tetra.world_to_parametric(pts, p)
            inside = self.tetra.is_inside(pc, -shapes.WELL_INSIDE)
        elif (self.kinds.PRESENT & 2) != 0 and kind == shapes.HEXAHEDRON:
            pc, converged = self.hexa.world_to_parametric(pts, p)
            inside = self.hexa.is_inside(pc, -shapes.WELL_INSIDE)
        elif (self.kinds.PRESENT & 4) != 0 and kind == shapes.WEDGE:
            pc, converged = self.wedge.world_to_parametric(pts, p)
            inside = self.wedge.is_inside(pc, -shapes.WELL_INSIDE)
        elif (self.kinds.PRESENT & 16) != 0 and kind == shapes.VOXEL:
            pc, converged = self.voxel.world_to_parametric(pts, p)
            inside = self.voxel.is_inside(pc, -shapes.WELL_INSIDE)
        found = self.ncells
        if converged == 1 and inside == 1:
            found = e
        else:
            kind = 0
            pc = p - p
        return found, kind, pc

    @tack.func
    def find_ranked(self, p):
        """``find``'s ``(e, kind, pc, tet)`` and ``held``: 1 if the cell holds ``p``,
        0 if it only comes within its tolerance (or there is none)."""
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
            # The coarse bin, then the leaf in it (CellLocator.cell_bins).
            c = bi + self.nx * (bj + self.ny * bk)
            lx = self.leaf_dims[3 * c]
            ly = self.leaf_dims[3 * c + 1]
            li = _leaf((p[0] - self.ox) * self.ix, bi, lx)
            lj = _leaf((p[1] - self.oy) * self.iy, bj, ly)
            lk = _leaf((p[2] - self.oz) * self.iz, bk, self.leaf_dims[3 * c + 2])
            b = self.leaf_starts[c] + li + lx * (lj + ly * lk)
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
                    pc = p - p
                    converged = 0
                    inside = 0
                    near = p - p
                    surface = 0
                    if self.CURVED == 1:
                        # As the cell-side search inverts a curved cell.
                        pc, converged, inside, near = self.curved.locate(
                            e, kind, p, shapes.HOLD_TOLERANCE)
                        if kind == shapes.TRIANGLE or kind == shapes.QUAD or kind == shapes.PIXEL:
                            surface = 1
                    else:
                        for j in range(self.cells.num_points(e)):
                            y = self.positions[self.cells.point_id(e, j)]
                            pts[3 * j] = y[0]
                            pts[3 * j + 1] = y[1]
                            pts[3 * j + 2] = y[2]
                    if self.CURVED == 1:
                        pass
                    elif (self.kinds.PRESENT & 1) != 0 and kind == shapes.TETRA:
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
        return best, best_type, best_pc, best_tet, 1 if best_rank < self.ncells else 0

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


def mesh_search(data, locator, positions=None, strict=True, neighbors=False):
    """A ``_MeshSearch`` over ``data`` with ``locator``'s bins, or ``None`` where
    this module cannot serve the dataset (see the module's docstring); with
    ``neighbors``, walking to a hint's face neighbours too (where the locator
    has them, ``CellLocator.cell_neighbors``)."""
    from tack.dataset.locator import _TOLERANCE

    geometry = data.geometry
    # No cells is served too: every point is outside.
    if not isinstance(geometry.space, H1):
        return None
    curved = None
    if geometry.space.order != 1 and data.num_cells:
        curved = curved_cells(data)
        if curved is None:
            return None
    if positions is None:
        # A curved geometry's points come first among its values: the corners.
        positions = _as_vectors(arrays.materialize(geometry.values), data.dtype)
    found = cells_by_id(data, positions, strict=strict)
    if found is None or (curved is not None and not isinstance(found[0], (_FlatCells,
                                                                          _GridCells))):
        return None
    cells, present = found
    return _MeshSearch(cells, present, positions, locator, _TOLERANCE,
                       locator.cell_neighbors() if neighbors else None, curved)


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


@tack.device_class
class _OneShapeValues:
    """``_ViewValues`` for a mesh of one shape: its one field view, called directly."""

    def __init__(self, view, local):
        self.view = view
        self.local = local

    @tack.func
    def value(self, mesh, e, kind, pc, tet):
        return self.view.value(self.local[e], pc)


@tack.device_class
class _ViewValues(_ByShape):
    """A field of any interpolated space by cell id, through its groups' field views:
    ``value`` as the cell-side path evaluates it."""

    @tack.func
    def value(self, mesh, e, kind, pc, tet):
        c = self.local[e]
        v = self.tetra.value(c, pc)
        if (self.kinds.PRESENT & 2) != 0 and kind == shapes.HEXAHEDRON:
            v = self.hexa.value(c, pc)
        elif (self.kinds.PRESENT & 4) != 0 and kind == shapes.WEDGE:
            v = self.wedge.value(c, pc)
        elif (self.kinds.PRESENT & 8) != 0 and kind == shapes.PYRAMID:
            v = self.pyramid.value(c, pc)
        elif (self.kinds.PRESENT & 16) != 0 and kind == shapes.VOXEL:
            v = self.voxel.value(c, pc)
        elif (self.kinds.PRESENT & 32) != 0 and kind == shapes.TRIANGLE:
            v = self.triangle.value(c, pc)
        elif (self.kinds.PRESENT & 64) != 0 and kind == shapes.QUAD:
            v = self.quad.value(c, pc)
        elif (self.kinds.PRESENT & 128) != 0 and kind == shapes.PIXEL:
            v = self.pixel.value(c, pc)
        return v


def field_by_id(data, field):
    """``field`` read by cell id (``_PointValues`` or ``_CellValues`` over its
    materialized values; ``_ViewValues`` for another interpolated space of one
    layout per shape), or ``None`` for a space this module does not serve."""
    space = field.space
    if (isinstance(space, H1) and space.order == 1) or space is Values(data, "points"):
        return _PointValues(arrays.materialize(field.values))
    if isinstance(space, Constant) or space is Values(data, "cells"):
        return _CellValues(arrays.materialize(field.values))
    if getattr(space, "interpolated", False) and not space.varies and data.num_cells:
        from tack.dataset.topology import StructuredTopology, UnstructuredTopology

        if isinstance(data.topology, (UnstructuredTopology, StructuredTopology)):
            found = _by_shape(data, data.launch_groups("cells", [field]), field.view)
            if found is not None:
                return (_OneShapeValues(*_one_shape(found)) if _single(found)
                        else _ViewValues(*found))
    return None

