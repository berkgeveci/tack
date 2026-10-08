"""The template views kernels iterate: one entity kind, one shape, at a time.

A view's class combines mixins, made once per combination and reused, so
every compiled kernel is specialized to it (``docs/design/dataset-api.md``
§4):

- an entity *kind*: the cells of an unstructured or structured topology,
  derived faces, or derived edges. Each kind declares what a loop over it
  runs through (``for i in view``), ``point_id(i, j)`` and ``entity_id(i)``,
  the entity's global id, where its data lives;
- a *shape* from ``tack.data.shapes``: the cell's, or the face's
  (``Triangle``/``Quad``), or ``Line`` for edges, with their shape functions;
- optionally *geometry*, from the dataset's geometry field: ``point(i, j)``,
  corner ``j``'s position, and ``position(i, pc)`` and
  ``geometry_jacobian(i, pc)`` through the shape's functions;
- optionally, for cells, *incidence*: ``face_id(c, f)``, ``face_side(c, f)``,
  ``edge_id(c, e)``, ``edge_sign(c, e)``;
- or, for a field view, a *space*: ``dof(i, j)`` and ``value(i, pc)`` for
  spaces with a basis, ``at(i)`` for values on entities.

A field is a separate kernel argument: its view is built for the same
group as the domain's, so ``u.value(c, pc)`` in ``for c in cells`` reads
the cell ``c`` the loop is at.
"""

import tack

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
    def cell_id(self, c):
        return self.ids[c]

    @tack.func
    def entity_id(self, c):
        return self.ids[c]

    @tack.func
    def index(self, c):
        return c


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
        return self.cell_id(c)

    @tack.func
    def entity_id(self, c):
        return self.cell_id(c)


class _StructuredLines(_StructuredCells):
    __tack_iterate__ = "num_cells"

    @tack.func
    def point_id(self, c, j):
        return c + j

    @tack.func
    def point_index(self, c, j):
        return [c + j, 0, 0]

    @tack.func
    def cell_id(self, c):
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
    def cell_id(self, c):
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
    def cell_id(self, c):
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
    def face_id(self, f):
        return self.ids[f]

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


# ── Geometry ────────────────────────────────────────────────────────


class _Geometry:
    """The geometry field, read through an entity: ``point(i, j)``, which a storage
    mixin below defines, and what follows from it at parametric coordinates.

    At order 1 the geometry's values are the corners' positions, so the shape's
    own functions interpolate them. A higher-order geometry would supply its
    own basis here; nothing that calls ``position`` or ``geometry_jacobian``
    would change.
    """

    @tack.func
    def gather_points(self, i, pts):
        for j in range(self.NUM_POINTS):
            x = self.point(i, j)
            pts[3 * j] = x[0]
            pts[3 * j + 1] = x[1]
            pts[3 * j + 2] = x[2]

    @tack.func
    def position(self, i, pc):
        """The world position at parametric coordinates ``pc``."""
        x = tack.Vector([0.0, 0.0, 0.0])
        for j in range(self.NUM_POINTS):
            x += self.shape_function(j, pc) * self.point(i, j)
        return x

    @tack.func
    def geometry_jacobian(self, i, pc):
        """The 3x3 derivative of world position by parametric coordinates at ``pc``:
        column ``k`` is ``dx/d(pc[k])``, zero past the shape's dimension."""
        m = tack.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for j in range(self.NUM_POINTS):
            m += self.point(i, j).outer_product(self.shape_gradient(j, pc))
        return m


class _H1Geometry(_Geometry):
    """H1 geometry: one position per point id, shared by the cells around it."""

    @tack.func
    def point(self, i, j):
        return self.points[self.point_id(i, j)]


class _L2Geometry(_Geometry):
    """L2 geometry: each cell's own corner positions, at ``point_offsets[cell]``. Cells
    may pull apart while the topology still has them share points."""

    @tack.func
    def point(self, c, j):
        return self.points[self.point_offsets[self.entity_id(c)] + j]


class _RectilinearPoints(_Geometry):
    """Rectilinear coordinates, for any entity: the flat point id is split into (i, j, k)."""

    @tack.func
    def point(self, i, j):
        p = self.point_id(i, j)
        rest = p // self.px
        return [self.xs[p - rest * self.px], self.ys[rest % self.py], self.zs[rest // self.py]]


class _RectilinearStructured(_Geometry):
    """Rectilinear coordinates for structured cells: (i, j, k) comes from the cell's own."""

    @tack.func
    def point(self, c, j):
        at = self.point_index(c, j)
        return [self.xs[at[0]], self.ys[at[1]], self.zs[at[2]]]


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


class _EdgeIncidence:
    """A cell's edges: ``side_edge``/``side_sign`` hold one entry per (cell, local edge)."""

    @tack.func
    def edge_id(self, c, e):
        return self.side_edge[self.edge_start + self.index(c) * self.NUM_EDGES + e]

    @tack.func
    def edge_sign(self, c, e):
        """+1 if the cell's local edge ``e`` runs from the edge's low point id to its high one."""
        return self.side_sign[self.edge_start + self.index(c) * self.NUM_EDGES + e]


# ── Spaces (field views) ────────────────────────────────────────────


class _H1Field:
    """H1, order 1, Shared: one value per point, interpolated by the shape functions."""

    @tack.func
    def dof(self, i, j):
        return self.values[self.point_id(i, j)]

    @tack.func
    def value(self, i, pc):
        total = self.dof(i, 0) * self.shape_function(0, pc)
        for j in range(1, self.NUM_POINTS):
            total += self.dof(i, j) * self.shape_function(j, pc)
        return total

    @tack.func
    def parametric_gradient(self, i, pc):
        """The derivative by parametric coordinates at ``pc``, a 3-vector."""
        g = tack.Vector([0.0, 0.0, 0.0])
        for j in range(self.NUM_POINTS):
            g += self.dof(i, j) * self.shape_gradient(j, pc)
        return g


class _ConstantField:
    """Constant (L2, order 0): one value per cell."""

    @tack.func
    def dof(self, i, j):
        return self.values[self.entity_id(i)]

    @tack.func
    def value(self, i, pc):
        return self.values[self.entity_id(i)]

    @tack.func
    def parametric_gradient(self, i, pc):
        return tack.Vector([0.0, 0.0, 0.0])


class _L2Field:
    """L2, order 1, PerCell: each cell's own corner values, at ``offsets[cell]``."""

    @tack.func
    def dof(self, i, j):
        return self.values[self.offsets[self.entity_id(i)] + j]

    @tack.func
    def value(self, i, pc):
        total = self.dof(i, 0) * self.shape_function(0, pc)
        for j in range(1, self.NUM_POINTS):
            total += self.dof(i, j) * self.shape_function(j, pc)
        return total

    @tack.func
    def parametric_gradient(self, i, pc):
        g = tack.Vector([0.0, 0.0, 0.0])
        for j in range(self.NUM_POINTS):
            g += self.dof(i, j) * self.shape_gradient(j, pc)
        return g


class _ValuesField:
    """Values on entities, no basis: one value per entity of the field's kind."""

    @tack.func
    def at(self, i):
        return self.values[self.entity_id(i)]


# ── Groups and view classes ─────────────────────────────────────────

_view_classes = {}


def view_class(kind, shape, *mixins):
    """The template class combining ``mixins``, ``kind`` and ``shape``, made once."""
    key = (kind, shape, mixins)
    if key not in _view_classes:
        name = "".join(m.__name__.strip("_") for m in mixins) + kind.__name__ + shape.__name__
        _view_classes[key] = tack.data_oriented(type(name, (*mixins, kind, shape), {}))
    return _view_classes[key]


class DomainGroup:
    """One shape's entities of one kind: ``kind(*args)`` with ``count`` of them,
    starting at ``start`` in the topology's per-group layout."""

    def __init__(self, kind, shape, args, count, start):
        self.kind = kind
        self.shape = shape
        self.args = args
        self.count = count
        self.start = start

    def view(self, *mixins, **attributes):
        """An instance of the view class with ``mixins``, holding ``attributes``."""
        view = view_class(self.kind, self.shape, *mixins)(*self.args)
        for name, value in attributes.items():
            setattr(view, name, value)
        return view
