"""Ghost marks: which of a dataset's cells and points another piece owns.

A piece of a partitioned dataset holds copies of what its neighbours own, so
that work near its edge sees whole neighbourhoods. Every cell and point that
several pieces hold has one owner; every other copy is a ghost, marked with
VTK's ``vtkGhostType`` bits in ``data.ghosts`` (a ``Ghosts``, kept apart from
the fields as the geometry is). ``vtk_to_dataset`` and ``dataset_to_vtk`` map
the marks to and from VTK's arrays, ``split`` marks the points its pieces
share, and reductions (``statistics``, ``entropy``) count only what a piece
owns, so each value counts once across all pieces.

Filters carry the marks (``carry_marks``, which ``carry`` calls): what a
filter makes inherits its owner from what it came from -- an output cell from
its source cell, a kept point from itself, a point on an input edge from that
edge, a point inside a cell from the cell, a surface face from the cell it is
a side of. An edge or a face, which several cells share, is owned where its
smallest point is, smallest by position: every piece holding it computes the
same answer, and with a ghost layer the piece owning that point holds it. So
the owned part of a filter's output, over all pieces, is the whole's output.
Filters skip hidden cells (``visible``), and ``remove_ghosts`` takes the
duplicates out at the end.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.arrays import materialize, size_of
from tack.runtime.dispatch import as_device, backend_of, get_backend, same_device

__all__ = ["Ghosts"]


def keep_marks(source, out):
    """``out``, a dataset on ``source``'s cells and points (the same topology), with
    ``source``'s ghost marks -- and its record of having lost them."""
    out.ghosts = source.ghosts
    out._ghosts_lost = source._ghosts_lost
    out.global_ids = source.global_ids
    out.origin = source.origin
    return out


def drop_marks(source, out, reason=True):
    """``out``, made from ``source``'s cells and points by a filter that cannot give
    it ghost marks: if ``source`` had marks, or had lost them, record that
    ``out`` lost them (``reason``, a sentence saying why, or ``True``), so
    reductions refuse it rather than count shared cells and points once per
    piece."""
    if source.ghosts is not None or source._ghosts_lost:
        out._ghosts_lost = source._ghosts_lost or reason
    out.origin = source.origin
    return out


def _marks(values, what):
    """``values`` as a one-dimensional u8 array, or ``None``."""
    if values is None:
        return None
    if isinstance(values, tack.Array):
        if values.dtype is not tack.u8 or len(values.shape) != 1:
            raise TypeError(f"ghost marks on {what} are a one-dimensional u8 array")
        return values
    host = np.asarray(values)
    if host.ndim != 1:
        raise TypeError(f"ghost marks on {what} are one-dimensional")
    return tack.array(host.astype(np.uint8), tack.u8)


class Ghosts:
    """Ghost marks of a dataset's cells and points: one u8 per cell (``cells``) and
    per point (``points``), either of which may be ``None``, holding VTK's
    ``vtkGhostType`` bits.

    A cell or point marked ``DUPLICATECELL`` or ``DUPLICATEPOINT`` is a copy of one
    another piece owns. ``HIDDENCELL`` and ``HIDDENPOINT`` mark data that is not
    valid there (a blanked cell). ``REFINEDCELL`` marks a cell a finer AMR
    level covers, whose data are valid but superseded, and ``REFINEDPOINT`` (a
    Tack bit; VTK has no such point bit) a point at whose location a finer
    level has one. Filters that make cells leave hidden and refined cells out
    (``MASKED_CELLS``); reductions leave out every skipped bit. The other bits
    are carried, not read."""

    DUPLICATECELL = 1
    HIGHCONNECTIVITYCELL = 2
    LOWCONNECTIVITYCELL = 4
    REFINEDCELL = 8
    EXTERIORCELL = 16
    HIDDENCELL = 32
    DUPLICATEPOINT = 1
    HIDDENPOINT = 2
    REFINEDPOINT = 8

    #: What a filter that makes cells never processes: data not valid there,
    #: and data a finer AMR level supersedes (VTK's masked cells).
    MASKED_CELLS = HIDDENCELL | REFINEDCELL
    #: What a reduction leaves out: copies of what another piece owns, what is
    #: not valid there, and what a finer level supersedes.
    SKIPPED_CELLS = DUPLICATECELL | HIDDENCELL | REFINEDCELL
    SKIPPED_POINTS = DUPLICATEPOINT | HIDDENPOINT | REFINEDPOINT

    def __init__(self, cells=None, points=None):
        self.cells = _marks(cells, "cells")
        self.points = _marks(points, "points")
        _ = self._device                    # refuses marks on two devices

    @property
    def _device(self):
        return same_device("Ghosts", (self.__dict__.get("cells"), self.__dict__.get("points")))

    @property
    def device(self):
        """The `tack.Device` the marks are on."""
        return as_device(self._device or get_backend())

    def to_device(self, device):
        """These marks on ``device``."""
        target = as_device(backend_of(device))
        return Ghosts(None if self.cells is None else self.cells.to_device(target),
                      None if self.points is None else self.points.to_device(target))

    def _check(self, topology):
        for what, marks, n in (("cells", self.cells, topology.num_cells),
                               ("points", self.points, topology.num_points)):
            if marks is not None and marks.shape[0] != n:
                raise ValueError(f"{marks.shape[0]} ghost marks for {n} {what}")

    def __repr__(self):
        parts = [f"{what}={'marked' if marks is not None else None}"
                 for what, marks in (("cells", self.cells), ("points", self.points))]
        return f"Ghosts({', '.join(parts)})"


@tack.kernel
def _keep(marks, skipped, flags):
    for i in range(flags.shape[0]):
        flags[i] = 1 if (marks[i] & skipped) == 0 else 0


@tack.kernel
def _keep_blocks(marks, skipped, offsets, flags):
    # One flag per value of an L2 field, the values of cell c from offsets[c].
    for c in range(marks.shape[0]):
        keep = 1 if (marks[c] & skipped) == 0 else 0
        for k in range(offsets[c], offsets[c + 1]):
            flags[k] = keep


@tack.kernel
def _kept(flags, slots, ids):
    for i in range(flags.shape[0]):
        if flags[i] == 1:
            ids[slots[i]] = i


def owned_values(data, field):
    """The values of ``field`` (a ``Field`` of ``data``) that ``data`` owns: all of
    them when it has no marks for where they live, else those of cells and
    points not marked a duplicate or hidden, materialized. Fields on faces or
    edges, and order-2 fields, have no marks of their own yet; on a dataset
    with ghost marks they are refused rather than counted twice."""
    from tack.dataset.carry import _take
    from tack.dataset.spaces import H1, L2, Constant, Values

    if data._ghosts_lost:
        why = data._ghosts_lost if isinstance(data._ghosts_lost, str) else (
            "the filter that made it does not carry the marks")
        raise NotImplementedError(
            f"this dataset is a filter's output from data with ghost marks, and {why}: "
            "its shared cells and points would count once per piece. Reduce the input "
            "instead, or set the output's ghosts")
    values = materialize(field.values)
    ghosts = data.ghosts
    if ghosts is None or (ghosts.cells is None and ghosts.points is None):
        return values
    space = field.space
    n = size_of(values)
    on_points = (isinstance(space, H1) and space.order == 1) or (
        isinstance(space, Values) and space.on == "points")
    on_cells = isinstance(space, (Constant, L2)) or (
        isinstance(space, Values) and space.on == "cells")
    if on_points:
        marks, skipped = ghosts.points, Ghosts.SKIPPED_POINTS
    elif on_cells:
        marks, skipped = ghosts.cells, Ghosts.SKIPPED_CELLS
    else:
        raise NotImplementedError(f"which values of a field on {space!r} a piece owns is "
                                  "not decided yet; this dataset has ghost marks")
    if marks is None:
        return values
    flags = tack.empty((n,), tack.i32)
    if isinstance(space, L2):
        if marks.shape[0]:
            _keep_blocks(marks, skipped, space.offsets, flags)
    elif n:
        _keep(marks, skipped, flags)
    idt = data.id_dtype
    slots = tack.empty((n,), idt)
    count = exclusive_scan(flags, slots, n) if n else 0
    if count == n:
        return values
    ids = tack.empty((count,), idt)
    if count:
        _kept(flags, slots, ids)
    return _take(values, ids)


# ── What filters make: the marks carried ────────────────────────────

@tack.kernel
def _any_bit(marks, bit, found):
    for i in range(marks.shape[0]):
        if (marks[i] & bit) != 0:
            tack.atomic_max(found, 0, 1)


def _has(marks, bit):
    """Whether any of ``marks`` (or ``None``) has ``bit``."""
    if marks is None or marks.shape[0] == 0:
        return False
    found = tack.zeros((1,), tack.i32)
    _any_bit(marks, bit, found)
    return int(found[0]) == 1


@tack.func
def _before(p, q):
    # Whether position p comes before q: by x, then y, then z. Every copy of a
    # point has the same position, so every piece orders the same points alike.
    return p[0] < q[0] or (p[0] == q[0] and (p[1] < q[1] or (p[1] == q[1] and p[2] < q[2])))


@tack.kernel
def _edge_point_marks(marks, positions, ends, out):
    # A point on input edge (a, b) is owned where the edge's smaller point is.
    for i in range(ends.shape[0]):
        ab = ends[i]
        s = ab[0]
        if _before(positions[ab[1]], positions[ab[0]]):
            s = ab[1]
        out[i] = marks[s] & 1


@tack.kernel
def _cell_point_marks(cell_marks, cells, out, first):
    # A point inside a cell is owned with it: DUPLICATECELL gives DUPLICATEPOINT,
    # HIDDENCELL HIDDENPOINT.
    for k in range(cells.shape[0]):
        m = cell_marks[cells[k]]
        out[first + k] = (m & 1) | (2 if (m & 32) != 0 else 0)


@tack.kernel
def _row_marks(offsets, members, point_marks, positions, out):
    # A cell made from what several input cells share (a face, an edge), or with no
    # source cell, is owned where its smallest point is.
    for c in range(out.shape[0]):
        begin = offsets[c]
        s = members[begin]
        for k in range(begin + 1, offsets[c + 1]):
            p = members[k]
            if _before(positions[p], positions[s]):
                s = p
        out[c] = point_marks[s] & 1


@tack.kernel
def _face_cell_marks(face_ids, sides, cell_marks, by_points, out):
    # A face with one side here is that cell's (a surface face); a face with two is
    # shared, and owned where its smallest point is.
    for i in range(out.shape[0]):
        s = sides[face_ids[i]]
        if s[2] < 0:
            out[i] = cell_marks[s[0]]
        else:
            out[i] = by_points[i]


def _rows(topology):
    """Each cell's points of a filter's output topology, CSR: ``(offsets, members)``."""
    from tack.dataset.polyhedra import PolygonalTopology, PolyhedralTopology, PolylineTopology

    if isinstance(topology, PolygonalTopology):
        return topology.loop_offsets, topology.loop_points
    if isinstance(topology, PolylineTopology):
        return topology.line_offsets, topology.line_points
    if isinstance(topology, PolyhedralTopology):
        return topology.cell_points()
    return topology.offsets, topology.connectivity


def _owned_by_points(out, point_marks):
    """Each of ``out``'s cells' mark by its smallest point, or ``None`` without point
    marks."""
    if point_marks is None:
        return None
    marks = tack.empty((out.num_cells,), tack.u8)
    if out.num_cells:
        offsets, members = _rows(out.topology)
        _row_marks(offsets, members, point_marks, out.geometry.values, marks)
    return marks


def _or_zeros(marks, n):
    return marks if marks is not None else tack.zeros((n,), tack.u8)


def _faces_marks(source, out, faces_to_cells, point_marks):
    """The marks of ``out``'s cells, which are ``source``'s faces."""
    from tack.arrays import CountingArray
    from tack.dataset.carry import Take

    faces = source.topology.faces()
    face_ids = faces_to_cells.ids if isinstance(faces_to_cells, Take) else CountingArray(
        faces.num_faces, dtype=source.id_dtype)
    marks = tack.empty((out.num_cells,), tack.u8)
    if out.num_cells:
        _face_cell_marks(face_ids, faces.sides,
                         _or_zeros(source.ghosts.cells, source.num_cells),
                         _or_zeros(_owned_by_points(out, point_marks), out.num_cells), marks)
    return marks


def _interpolated_marks(source, points):
    """The marks of an ``Interpolate`` map's points: by their edges, then the
    averages' by their cells."""
    ghosts = source.ghosts
    n = points.weights.shape[0]
    extra = points.averages[0].shape[0] - 1 if points.averages is not None else 0
    marks = tack.zeros((n + extra,), tack.u8)
    if n and ghosts.points is not None:
        _edge_point_marks(ghosts.points, source.geometry.values, points.ends, marks)
    if extra and ghosts.cells is not None:
        _cell_point_marks(ghosts.cells, points.average_cells, marks, n)
    return marks


def _unplaceable(source, points, cells, faces_to_cells, edges_to_cells):
    """Why ``source``'s marks cannot place what these maps make or keep, or ``None``
    when they can. What depends on the cells around it -- a point on an edge,
    a face, an edge, a point kept because a kept cell uses it -- needs the piece
    owning it to hold those cells: a ghost layer."""
    from tack.dataset.carry import Interpolate, Same, Take
    from tack.dataset.spaces import H1

    used_points = isinstance(points, Take) and cells is not None and not isinstance(cells, Same)
    if not (used_points or isinstance(points, Interpolate) or faces_to_cells is not None
            or edges_to_cells is not None):
        return None
    ghosts = source.ghosts
    duplicate_cells = _has(ghosts.cells, Ghosts.DUPLICATECELL)
    if _has(ghosts.points, Ghosts.DUPLICATEPOINT) and not duplicate_cells:
        return ("its input has no ghost cells, so a piece cannot tell what its neighbours "
                "make or keep from the cells, faces and edges they share: split with "
                "ghost_layers=1, or give the pieces ghost cells (vtkGhostCellsGenerator)")
    if used_points and not isinstance(points, Interpolate) and faces_to_cells is None \
            and edges_to_cells is None:
        return None                     # kept points keep their marks
    if not isinstance(source.geometry.space, H1):
        return "its owners on edges and faces go by position, and its geometry has no points"
    if duplicate_cells and ghosts.points is None:
        return "its input has ghost cells but no point marks, which owners on edges need"
    if isinstance(points, Interpolate) and points.averages is not None \
            and points.average_cells is None:
        return "its points inside cells do not say which cells"
    return None


def carry_marks(source, out, points=None, cells=None, faces_to_cells=None,
                edges_to_cells=None):
    """``out``, made from ``source`` through ``carry``'s maps, with ghost marks: an
    output cell's from its source cell, a kept point's from itself, a point on
    an input edge from the edge's smallest point, a point inside a cell from the
    cell; a cell that is an input face from its one cell, or from its smallest
    point when two cells share it; a cell that is an input edge, or that has no
    source cell, from its smallest point. Smallest goes by position.

    Placing what is made on edges and faces needs pieces with ghost cells: a
    piece without them cannot tell what its neighbours make there. Without
    them, or without point positions, ``out`` records that it lost the marks,
    so reductions refuse it."""
    from tack.dataset.carry import Interpolate, Same, Take, _take

    if source._ghosts_lost:
        out._ghosts_lost = source._ghosts_lost
        return out
    ghosts = source.ghosts
    if ghosts is None or (ghosts.cells is None and ghosts.points is None):
        return out
    if isinstance(points, Same) and isinstance(cells, Same):
        return keep_marks(source, out)
    reason = _unplaceable(source, points, cells, faces_to_cells, edges_to_cells)
    if reason is not None:
        return drop_marks(source, out, reason)
    point_marks = None
    if isinstance(points, Same):
        point_marks = ghosts.points
    elif isinstance(points, Take):
        point_marks = None if ghosts.points is None else _take(ghosts.points, points.ids)
    elif isinstance(points, Interpolate):
        point_marks = _interpolated_marks(source, points)
    if faces_to_cells is not None:
        cell_marks = _faces_marks(source, out, faces_to_cells, point_marks)
    elif edges_to_cells is not None or cells is None:
        cell_marks = _owned_by_points(out, point_marks)
    elif isinstance(cells, Same):
        cell_marks = ghosts.cells
    else:
        cell_marks = None if ghosts.cells is None else _take(ghosts.cells, cells.ids)
    if point_marks is not None or cell_marks is not None:
        out.ghosts = Ghosts(cells=cell_marks, points=point_marks)
    return out


# ── Hidden cells ────────────────────────────────────────────────────

@tack.kernel
def _clear_marked(marks, bits, keep, out):
    for c in range(out.shape[0]):
        out[c] = keep[c] if (marks[c] & bits) == 0 else 0


def without_hidden(data, keep):
    """``keep`` (an i32 flag per cell of ``data``) with ``data``'s hidden and refined
    cells (``MASKED_CELLS``) cleared: ``keep`` itself when it has none."""
    ghosts = data.ghosts
    if ghosts is None or not _has(ghosts.cells, Ghosts.MASKED_CELLS):
        return keep
    out = tack.empty((data.num_cells,), tack.i32)
    _clear_marked(ghosts.cells, Ghosts.MASKED_CELLS, keep, out)
    return out


def visible(data, *arguments):
    """``data`` without its hidden cells (``HIDDENCELL``: data not valid there) and
    refined cells (``REFINEDCELL``: an AMR cell a finer level covers), which
    filters that make cells never process: ``data`` itself when it has none,
    else its other cells as ``extract_cells`` keeps them, marks and fields
    carried. Given field ``arguments`` too (``Field``s of
    ``data`` or names; anything else passes through), returns ``(data,
    *arguments)``, each ``Field`` carried onto the cells kept."""
    from tack.dataset.dataset import DataSet, Field

    ghosts = data.ghosts
    if ghosts is None or not _has(ghosts.cells, Ghosts.MASKED_CELLS):
        return (data, *arguments) if arguments else data
    from tack.dataset.algorithms._extract import _keep_cells

    names = {f"__tack_argument{k}__": a for k, a in enumerate(arguments)
             if isinstance(a, Field) and a.space.topology is data.topology}
    source = data
    if names:
        source = DataSet(data.topology, data.geometry, fields={**data.fields, **names},
                         sets=dict(data.sets), ghosts=data.ghosts, origin=data.origin)
    keep = tack.empty((data.num_cells,), tack.i32)
    keep.fill(1)
    out = _keep_cells(source, keep)
    if not arguments:
        return out
    carried = {id(a): out.fields.pop(name) for name, a in names.items() if name in out.fields}
    return (out, *(carried.get(id(a), a) for a in arguments))


def refuse_hidden(data, what):
    """Refuse ``data`` with hidden or refined cells where ``what`` (a phrase) cannot
    leave them out."""
    ghosts = data.ghosts
    if ghosts is not None and _has(ghosts.cells, Ghosts.MASKED_CELLS):
        raise NotImplementedError(f"{what} on a dataset with hidden or refined cells "
                                  "(HIDDENCELL, REFINEDCELL) is not supported yet")
