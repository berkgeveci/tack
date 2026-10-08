"""Cell sets: which points each cell joins, and of which shape.

Three kinds, each a host object holding device fields:

- ``ExplicitCellSet(types, offsets, connectivity)``: any mix of the linear
  shapes, as a ``vtkUnstructuredGrid`` stores them -- a ``u8`` type per
  cell, and each cell's point ids at ``connectivity[offsets[c]:offsets[c + 1]]``.
- ``SingleTypeCellSet(shape, connectivity)``: cells of one shape, one row
  of point ids per cell.
- ``StructuredCellSet(point_dims)``: the cells of a structured grid of
  ``nx``, ``nx * ny`` or ``nx * ny * nz`` points, with connectivity implied
  by the dimensions: lines, quads or hexahedra in ``vtkStructuredGrid``'s
  order.

Kernels never see a cell set. ``tack.data.for_each_shape`` hands a kernel
the cells of one shape at a time, as a *cell view*: one template object
that is both the shape (its methods and constants from
``tack.data.shapes``) and that shape's cells. A kernel loops over it,
``for c in cells:``, and treats ``c`` as a handle the view's methods
interpret: a cell number for explicit cells, the vector ``(i, j)`` or
``(i, j, k)`` for structured quads and hexahedra, which then launch in
their grid's shape and never divide to find their indices. A view has

- ``point_id(c, j)``: point ``j`` of cell ``c``;
- ``cell_id(c)``: cell ``c``'s index in the whole cell set, where its
  cell data and results belong;
- ``num_cells``: how many cells this launch covers;
- ``index(c)``: cell ``c``'s position among them, ``0`` to ``num_cells - 1``,
  for output laid out per view;

and, when made from a ``DataSet``, its point coordinates:

- ``point(c, j)``: the position of point ``j`` of cell ``c``, a 3-vector;
- ``gather_points(c, pts)``: all of cell ``c``'s positions into a local
  array, three coordinates per point, as the shapes' geometric methods
  take them.

An explicit cell set groups its cells by shape on the device the first
time it is used (a stable sort of the types), and keeps the groups.
"""

import numpy as np

import tack
from tack.algorithms.sort import argsort
from tack.data import shapes
from tack.lang.field import Field

__all__ = ["ExplicitCellSet", "SingleTypeCellSet", "StructuredCellSet"]


def _as_field(values, dtype):
    """A device field of ``dtype`` from a NumPy array or a field of that dtype."""
    if isinstance(values, Field):
        if values.dtype != dtype:
            raise TypeError(f"expected a {dtype.name} field, got {values.dtype.name}")
        return values
    values = np.ascontiguousarray(values)
    out = tack.field(dtype, shape=values.shape)
    if values.size:
        out.from_numpy(values.astype(dtype.numpy_dtype))
    return out


# ── Cell views ──────────────────────────────────────────────────────

class _Cells:
    """The cells of one shape, by an explicit row of point ids each."""

    __tack_iterate__ = "num_cells"

    def __init__(self, connectivity, ids, num_cells):
        self.connectivity = connectivity      # (num_cells, NUM_POINTS) i32
        self.ids = ids                        # (num_cells,) i32
        self.num_cells = num_cells

    @tack.func
    def point_id(self, c, j):
        return self.connectivity[c, j]

    @tack.func
    def cell_id(self, c):
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


class _Points:
    """Positions for a view: ``point(c, j)``, which a coordinates mixin defines."""

    @tack.func
    def gather_points(self, c, pts):
        for j in range(self.NUM_POINTS):
            x = self.point(c, j)
            pts[3 * j] = x[0]
            pts[3 * j + 1] = x[1]
            pts[3 * j + 2] = x[2]


class _ExplicitPoints(_Points):
    """A field of positions, one per point id."""

    @tack.func
    def point(self, c, j):
        return self.points[self.point_id(c, j)]


class _RectilinearPoints(_Points):
    """Rectilinear coordinates, for any cells: the flat point id is split into (i, j, k)."""

    @tack.func
    def point(self, c, j):
        p = self.point_id(c, j)
        rest = p // self.px
        return [self.xs[p - rest * self.px], self.ys[rest % self.py], self.zs[rest // self.py]]


class _RectilinearStructured(_Points):
    """Rectilinear coordinates for structured cells: (i, j, k) comes from the cell's own."""

    @tack.func
    def point(self, c, j):
        at = self.point_index(c, j)
        return [self.xs[at[0]], self.ys[at[1]], self.zs[at[2]]]


_view_classes = {}


def _view_class(cells, shape, coordinates):
    """The template class for ``shape``'s cells, made once and reused.

    Templates are keyed by class, so making it once per combination is what
    lets every launch for that shape reuse one compiled variant.
    ``coordinates`` is the positions mixin, or None for a cell set alone.
    """
    key = (cells, shape, coordinates)
    if key not in _view_classes:
        bases = ((coordinates,) if coordinates else ()) + (cells, shape)
        name = (f"{shape.__name__}{coordinates.__name__ if coordinates else ''}"
                f"{cells.__name__}")
        _view_classes[key] = tack.data_oriented(type(name, bases, {}))
    return _view_classes[key]


def _make_view(cells, shape, points, *args):
    """A view of ``cells``, with ``points`` -- a field or coordinates object -- if given."""
    from tack.data.coordinates import RectilinearCoordinates

    if points is None:
        return _view_class(cells, shape, None)(*args)
    if isinstance(points, RectilinearCoordinates):
        structured = issubclass(cells, _StructuredCells)
        view = _view_class(cells, shape,
                           _RectilinearStructured if structured else _RectilinearPoints)(*args)
        view.xs, view.ys, view.zs = points.x, points.y, points.z
        view.px, view.py = points.dims[0], points.dims[1]
        return view
    view = _view_class(cells, shape, _ExplicitPoints)(*args)
    view.points = points
    return view


# ── Cell sets ───────────────────────────────────────────────────────

@tack.kernel
def _histogram_types(types, keys, counts):
    for c in range(types.shape[0]):
        t = tack.i32(types[c])
        keys[c] = t
        tack.atomic_add(counts, t, 1)


@tack.kernel
def _gather_shape(order, start, offsets, connectivity, ids, rows, bad):
    for c in range(rows.shape[0]):
        cell = order[start + c]
        ids[c] = cell
        begin = offsets[cell]
        count = offsets[cell + 1] - begin
        if count != rows.shape[1]:
            tack.atomic_add(bad, 0, 1)
        for j in range(rows.shape[1]):
            rows[c, j] = connectivity[begin + min(j, count - 1)] if count > 0 else 0


class ExplicitCellSet:
    """Cells of any of the linear shapes, stored as a ``vtkUnstructuredGrid`` does.

    ``types`` holds a VTK cell type id per cell (``u8``), ``offsets`` the
    ``num_cells + 1`` starts of each cell's point ids in ``connectivity``.
    Each may be a NumPy array or a Tack field (``u8``, ``i32`` and ``i32``);
    arrays are copied to the device.
    """

    def __init__(self, types, offsets, connectivity):
        self.types = _as_field(types, tack.u8)
        self.offsets = _as_field(offsets, tack.i32)
        self.connectivity = _as_field(connectivity, tack.i32)
        self.num_cells = self.types.shape[0]
        if self.offsets.shape != (self.num_cells + 1,):
            raise ValueError(f"offsets must have num_cells + 1 = {self.num_cells + 1} "
                             f"entries, not {self.offsets.shape[0]}")
        self._groups = None

    def groups(self):
        """``[(shape class, cell ids, connectivity rows), ...]``, one per shape present.

        Computed on the device the first time and kept. Raises
        ``ValueError`` for a type that is not one of the linear shapes, or a
        cell whose point count does not match its shape.
        """
        if self._groups is not None:
            return self._groups
        n = self.num_cells
        groups = []
        if n:
            keys = tack.field(tack.i32, shape=(n,))
            counts = tack.zeros(tack.i32, (256,))
            _histogram_types(self.types, keys, counts)
            counts = counts.to_numpy()
            present = np.flatnonzero(counts)
            unknown = [int(t) for t in present if int(t) not in shapes._BY_ID]
            if unknown:
                raise ValueError(f"cell types {unknown} are not linear shapes "
                                 f"{sorted(shapes._BY_ID)}")
            order = argsort(keys)
            bad = tack.zeros(tack.i32, (1,))
            starts = np.concatenate([[0], np.cumsum(counts)])
            for t in present:
                shape = shapes.shape_class(t)
                count = int(counts[t])
                ids = tack.field(tack.i32, shape=(count,))
                rows = tack.field(tack.i32, shape=(count, shape.NUM_POINTS))
                _gather_shape(order, int(starts[t]), self.offsets, self.connectivity,
                              ids, rows, bad)
                groups.append((shape, ids, rows))
            if bad[0]:
                raise ValueError(f"{bad[0]} cells have a point count that does not "
                                 "match their type")
        self._groups = groups
        return groups

    def shapes(self):
        """The shape classes present, in id order."""
        return [shape for shape, _, _ in self.groups()]

    def views(self, points=None):
        """One cell view per shape present (see the module docstring)."""
        return [_make_view(_Cells, shape, points, rows, ids, ids.shape[0])
                for shape, ids, rows in self.groups()]


class SingleTypeCellSet:
    """Cells of one shape: ``connectivity`` has one row of point ids per cell."""

    def __init__(self, shape, connectivity):
        self.shape = shape
        connectivity = _as_field(connectivity, tack.i32)
        if len(connectivity.shape) != 2 or connectivity.shape[1] != shape.NUM_POINTS:
            raise ValueError(f"connectivity must be (num_cells, {shape.NUM_POINTS}) "
                             f"for {shape.__name__}, not {connectivity.shape}")
        self.connectivity = connectivity
        self.num_cells = connectivity.shape[0]
        self._ids = None

    def shapes(self):
        return [self.shape] if self.num_cells else []

    def views(self, points=None):
        if not self.num_cells:
            return []
        if self._ids is None:
            self._ids = tack.arange(self.num_cells, tack.i32)
        return [_make_view(_Cells, self.shape, points, self.connectivity, self._ids,
                           self.num_cells)]


class StructuredCellSet:
    """The cells of a structured grid of ``point_dims`` points, x fastest.

    One, two or three dimensions give lines, quads or hexahedra, numbered
    and ordered as ``vtkStructuredGrid`` numbers them. Every dimension must
    have at least two points.
    """

    _CELLS = {1: (_StructuredLines, shapes.Line), 2: (_StructuredQuads, shapes.Quad),
              3: (_StructuredHexahedra, shapes.Hexahedron)}

    def __init__(self, point_dims):
        point_dims = tuple(int(d) for d in point_dims)
        if not 1 <= len(point_dims) <= 3 or min(point_dims) < 2:
            raise ValueError(f"point_dims must be 1 to 3 sizes of at least 2, not {point_dims}")
        self.point_dims = point_dims
        self.num_cells = int(np.prod([d - 1 for d in point_dims]))
        self.shape = self._CELLS[len(point_dims)][1]

    def shapes(self):
        return [self.shape]

    def views(self, points=None):
        cells, shape = self._CELLS[len(self.point_dims)]
        dims = self.point_dims + (1,) * (3 - len(self.point_dims))
        return [_make_view(cells, shape, points, *dims, self.num_cells)]
