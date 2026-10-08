"""Topology: cells, and the faces and edges derived from them.

Two kinds of topology, as in ``docs/design/dataset-api.md`` §3.1:

- ``UnstructuredTopology(types, offsets, connectivity)``: any mix of the
  linear shapes, stored as a ``vtkUnstructuredGrid`` stores them.
- ``StructuredTopology(point_dims)``: the cells of a structured grid,
  connectivity implied by the dimensions.

Both give their cells as *groups*, one per shape present, which a kernel
iterates one at a time (``tack.data.for_each``). Faces and edges are not
stored; ``faces()`` and ``edges()`` derive them from any topology by the
same kernels, through the groups' cell views: every (cell, local face) is
keyed by its sorted point ids, two stable sorts bring a face's copies
together, and each run of equal keys is one face. Its first copy is *side
0* -- the face's points are listed in that cell's outward order -- and a
second copy, if any, is side 1. Edges are runs of (low, high) point-id
pairs; each (cell, local edge) records whether it runs low to high.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import _run_offsets, argsort, gather
from tack.data import shapes
from tack.data.views import (
    DomainGroup,
    _Cells,
    _Edges,
    _Faces,
    _StructuredHexahedra,
    _StructuredLines,
    _StructuredQuads,
)
from tack.lang.field import Field

__all__ = ["Edges", "Faces", "StructuredTopology", "UnstructuredTopology"]


def _as_field(values, dtype):
    if isinstance(values, Field):
        if values.dtype != dtype:
            raise TypeError(f"expected a {dtype.name} field, got {values.dtype.name}")
        return values
    values = np.ascontiguousarray(values)
    out = tack.field(dtype, shape=values.shape)
    if values.size:
        out.from_numpy(values.astype(dtype.numpy_dtype))
    return out


# ── Cell groups ─────────────────────────────────────────────────────

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


class _Topology:
    """What both kinds share: the derived faces and edges, made once and kept."""

    _faces = None
    _edges = None

    def faces(self):
        """The faces of the 3D cells (``Faces``), derived on first use and kept."""
        if self._faces is None:
            self._faces = Faces._derive(self)
        return self._faces

    def edges(self):
        """The edges of every cell (``Edges``), derived on first use and kept."""
        if self._edges is None:
            self._edges = Edges._derive(self)
        return self._edges

    def groups(self):
        """The cells as one ``DomainGroup`` per shape present, with their start in a
        per-group layout (``start``: the count of cells in earlier groups)."""
        raise NotImplementedError


class UnstructuredTopology(_Topology):
    """Cells of any of the linear shapes: a VTK type per cell (``u8``), and each
    cell's point ids at ``connectivity[offsets[c]:offsets[c + 1]]``, over
    ``num_points`` points -- by default one past the highest id used, but a
    dataset may have points no cell uses, as VTK's may."""

    def __init__(self, types, offsets, connectivity, num_points=None):
        self.types = _as_field(types, tack.u8)
        self.offsets = _as_field(offsets, tack.i32)
        self.connectivity = _as_field(connectivity, tack.i32)
        self.num_cells = self.types.shape[0]
        if self.offsets.shape != (self.num_cells + 1,):
            raise ValueError(f"offsets must have num_cells + 1 = {self.num_cells + 1} "
                             f"entries, not {self.offsets.shape[0]}")
        if num_points is None:
            used = self.connectivity.to_numpy()
            num_points = int(used.max()) + 1 if used.size else 0
        self.num_points = int(num_points)
        self._groups = None

    def groups(self):
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
                raise ValueError(f"cell types {unknown} are not linear shapes")
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
                groups.append(DomainGroup(_Cells, shape, (rows, ids, count), count,
                                          int(starts[t])))
            if bad[0]:
                raise ValueError(f"{bad[0]} cells have a point count that does not "
                                 "match their type")
        self._groups = groups
        return groups


class StructuredTopology(_Topology):
    """The cells of a structured grid of ``point_dims`` points, x fastest: lines, quads
    or hexahedra in ``vtkStructuredGrid``'s order, iterated by (i, j, k)."""

    _KINDS = {1: (_StructuredLines, shapes.Line), 2: (_StructuredQuads, shapes.Quad),
              3: (_StructuredHexahedra, shapes.Hexahedron)}

    def __init__(self, point_dims):
        point_dims = tuple(int(d) for d in point_dims)
        if not 1 <= len(point_dims) <= 3 or min(point_dims) < 2:
            raise ValueError(f"point_dims must be 1 to 3 sizes of at least 2, not {point_dims}")
        self.point_dims = point_dims
        self.num_cells = int(np.prod([d - 1 for d in point_dims]))
        self.num_points = int(np.prod(point_dims))
        kind, shape = self._KINDS[len(point_dims)]
        dims = point_dims + (1,) * (3 - len(point_dims))
        # One group, kept: derived incidence is laid out per group.
        self._groups = [DomainGroup(kind, shape, (*dims, self.num_cells), self.num_cells, 0)]

    def groups(self):
        return self._groups


# ── Derived faces ───────────────────────────────────────────────────

_NO_POINT = tack.constant(0x7FFFFFFF, tack.i32)


@tack.func
def _order(a, b):
    return min(a, b), max(a, b)


@tack.kernel
def _emit_faces(cells, start, rows, kinds, owners):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_FACES
        for f in range(cells.NUM_FACES):
            quad = cells.face_num_points(f) == 4
            p0 = cells.point_id(c, cells.face_point(f, 0))
            p1 = cells.point_id(c, cells.face_point(f, 1))
            p2 = cells.point_id(c, cells.face_point(f, 2))
            p3 = cells.point_id(c, cells.face_point(f, 3)) if quad else _NO_POINT
            kind = cells.face_shape(f)
            if kind == shapes.PIXEL:
                # A voxel's faces are pixels, x-fastest: as a quad, 0 1 3 2.
                rows[first + f] = [p0, p1, p3, p2]
                kinds[first + f] = shapes.QUAD
            else:
                rows[first + f] = [p0, p1, p2, p3]
                kinds[first + f] = kind
            owners[first + f] = [cells.cell_id(c), f]


@tack.kernel
def _face_keys(rows, hi, lo):
    for k in range(hi.shape[0]):
        row = rows[k]
        a, b = _order(row[0], row[1])
        c, d = _order(row[2], row[3])
        a, c = _order(a, c)
        b, d = _order(b, d)
        b, c = _order(b, c)
        hi[k] = (tack.u64(a) << tack.u64(32)) | tack.u64(b)
        lo[k] = (tack.u64(c) << tack.u64(32)) | tack.u64(d)


@tack.kernel
def _faces_from_runs(order, offsets, rows, kinds, owners, face_rows, face_kinds, face_sides,
                     side_face, side_slot, too_many, count):
    for r in range(count):
        begin = offsets[r]
        end = offsets[r + 1]
        first = order[begin]
        face_rows[r] = rows[first]
        face_kinds[r] = kinds[first]
        o0 = owners[first]
        o1 = owners[order[begin + 1]] if end - begin > 1 else [-1, -1]
        face_sides[r] = [o0[0], o0[1], o1[0], o1[1]]
        if end - begin > 2:
            tack.atomic_add(too_many, 0, 1)
        for i in range(begin, end):
            side_face[order[i]] = r
            side_slot[order[i]] = i - begin


@tack.kernel
def _side_orientations(rows, side_face, face_rows, orientation, bad):
    # Where this side's first face point sits in the face's row (side 0's
    # order), and whether the side lists the points the other way round.
    for s in range(side_face.shape[0]):
        mine = rows[s]
        face = face_rows[side_face[s]]
        n = 3 if mine[3] == _NO_POINT else 4
        r = 0
        for j in range(n):
            if face[j] == mine[0]:
                r = j
        after = r + 1 if r + 1 < n else 0
        before = r - 1 if r > 0 else n - 1
        if face[after] == mine[1]:
            orientation[s] = 2 * r
        elif face[before] == mine[1]:
            orientation[s] = 2 * r + 1
        else:
            orientation[s] = 0
            tack.atomic_add(bad, 0, 1)


@tack.kernel
def _select_kind(face_kinds, kind, flags):
    for f in range(face_kinds.shape[0]):
        flags[f] = 1 if face_kinds[f] == kind else 0


@tack.kernel
def _gather_kind(flags, slots, face_rows, face_sides, ids, rows, sides):
    for f in range(flags.shape[0]):
        if flags[f] == 1:
            s = slots[f]
            ids[s] = f
            rows[s] = face_rows[f]
            sides[s] = face_sides[f]


def _lexicographic_order(hi, lo):
    """The permutation sorting (hi, lo) pairs, stable: two stable sorts, low key first."""
    by_lo = argsort(lo)
    by_hi = argsort(gather(hi, by_lo))
    return gather(by_lo, by_hi)


class Faces:
    """The faces of a topology's 3D cells, as global entities.

    ``num_faces``; per face (by face id): ``kinds`` (``TRIANGLE`` or ``QUAD``),
    ``rows`` (its point ids in side 0's outward order, a 4-vector padded with
    ``0x7FFFFFFF``), ``sides`` (``[cell0, local0, cell1, local1]``, the second
    pair ``-1`` on the boundary). Per (cell, local face), in the topology's
    group layout: ``side_face`` (the face id), ``side_slot`` (0 if the cell
    is the face's side 0, else 1) and ``side_orientation``, ``2 * r +
    reflected``: the cell's first point of the face is point ``r`` of the
    face's row, and ``reflected`` is 1 when the cell goes round the face the
    other way (as side 1 does, its outward normal being opposite). ``groups()`` are the faces by shape,
    for ``for_each``; ``boundary()`` the faces with one side.
    """

    @classmethod
    def _derive(cls, topology):
        self = cls()
        cell_groups = [g for g in topology.groups() if g.shape.NUM_FACES]
        starts = np.concatenate([[0], np.cumsum([g.count * g.shape.NUM_FACES
                                                 for g in cell_groups])]).astype(int)
        total = int(starts[-1])
        self.group_starts = {id(g): int(s) for g, s in zip(cell_groups, starts[:-1])}
        rows = tack.Vector.field(4, tack.i32, shape=(total,))
        kinds = tack.field(tack.i32, shape=(total,))
        owners = tack.Vector.field(2, tack.i32, shape=(total,))
        for group, start in zip(cell_groups, starts[:-1]):
            if group.count:
                _emit_faces(group.view(), int(start), rows, kinds, owners)

        self.side_face = tack.field(tack.i32, shape=(total,))
        self.side_slot = tack.field(tack.i32, shape=(total,))
        self.side_orientation = tack.field(tack.i32, shape=(total,))
        if total:
            hi = tack.field(tack.u64, shape=(total,))
            lo = tack.field(tack.u64, shape=(total,))
            _face_keys(rows, hi, lo)
            order = _lexicographic_order(hi, lo)
            offsets, count = _pair_run_offsets(gather(hi, order), gather(lo, order), total)
        else:
            count = 0
        self.num_faces = count
        self.rows = tack.Vector.field(4, tack.i32, shape=(count,))
        self.kinds = tack.field(tack.i32, shape=(count,))
        self.sides = tack.Vector.field(4, tack.i32, shape=(count,))
        if count:
            too_many = tack.zeros(tack.i32, (1,))
            _faces_from_runs(order, offsets, rows, kinds, owners, self.rows, self.kinds,
                             self.sides, self.side_face, self.side_slot, too_many, count)
            if too_many[0]:
                raise ValueError(f"{too_many[0]} faces are shared by more than two cells")
            bad = tack.zeros(tack.i32, (1,))
            _side_orientations(rows, self.side_face, self.rows, self.side_orientation, bad)
            if bad[0]:
                raise ValueError(f"{bad[0]} cell faces do not go round their face's points "
                                 "in either direction")
        self._groups = None
        self._boundary = None
        return self

    def groups(self, subset=None):
        """The faces (or those in ``subset``, an i32 field of face ids) as one
        ``DomainGroup`` per face shape."""
        if subset is None and self._groups is not None:
            return self._groups
        ids_in = subset if subset is not None else tack.arange(self.num_faces, tack.i32)
        n = ids_in.shape[0]
        kinds = _take_scalar(self.kinds, ids_in) if subset is not None else self.kinds
        rows = _take_vector(self.rows, ids_in) if subset is not None else self.rows
        sides = _take_vector(self.sides, ids_in) if subset is not None else self.sides
        groups = []
        for shape in (shapes.Triangle, shapes.Quad):
            flags = tack.field(tack.i32, shape=(n,))
            slots = tack.field(tack.i32, shape=(n,))
            if n:
                _select_kind(kinds, int(shape.ID), flags)
            count = exclusive_scan(flags, slots, n) if n else 0
            positions = tack.field(tack.i32, shape=(count,))
            group_rows = tack.Vector.field(4, tack.i32, shape=(count,))
            group_sides = tack.Vector.field(4, tack.i32, shape=(count,))
            if count:
                _gather_kind(flags, slots, rows, sides, positions, group_rows, group_sides)
            ids = _take_scalar(ids_in, positions) if count else positions
            groups.append(DomainGroup(_Faces, shape, (group_rows, ids, group_sides, count),
                                      count, 0))
        if subset is None:
            self._groups = groups
        return groups

    def boundary(self):
        """The ids of the faces with one side, an i32 field: the boundary side set."""
        if self._boundary is None:
            n = self.num_faces
            flags = tack.field(tack.i32, shape=(n,))
            slots = tack.field(tack.i32, shape=(n,))
            if n:
                _one_sided(self.sides, flags)
            count = exclusive_scan(flags, slots, n) if n else 0
            ids = tack.field(tack.i32, shape=(count,))
            if count:
                _compact_ids(flags, slots, ids)
            self._boundary = ids
        return self._boundary


@tack.kernel
def _flag_pair_runs(hi, lo, flags):
    for i in range(flags.shape[0]):
        same = i > 0 and hi[i] == hi[i - 1] and lo[i] == lo[i - 1]
        flags[i] = 0 if same else 1


@tack.kernel
def _scatter_run_starts(flags, run_ids, offsets, n, nruns):
    for i in range(n):
        if flags[i] == 1:
            offsets[run_ids[i]] = i
        if i == 0:
            offsets[nruns] = n


def _pair_run_offsets(hi, lo, n):
    """Offsets of the runs of equal adjacent (hi, lo) pairs: ``nruns + 1`` entries."""
    flags = tack.field(tack.i32, shape=(n,))
    run_ids = tack.field(tack.i32, shape=(n,))
    _flag_pair_runs(hi, lo, flags)
    nruns = exclusive_scan(flags, run_ids, n)
    offsets = tack.field(tack.i32, shape=(nruns + 1,))
    _scatter_run_starts(flags, run_ids, offsets, n, nruns)
    return offsets, nruns


@tack.kernel
def _one_sided(sides, flags):
    for f in range(flags.shape[0]):
        flags[f] = 1 if sides[f][2] < 0 else 0


@tack.kernel
def _compact_ids(flags, slots, ids):
    for f in range(flags.shape[0]):
        if flags[f] == 1:
            ids[slots[f]] = f


@tack.kernel
def _take_scalar_kernel(values, ids, out):
    for i in range(ids.shape[0]):
        out[i] = values[ids[i]]


def _take_scalar(values, ids):
    out = tack.field(values.dtype, shape=(ids.shape[0],))
    if ids.shape[0]:
        _take_scalar_kernel(values, ids, out)
    return out


def _take_vector(values, ids):
    out = tack.Vector.field(values._vector_n, values.dtype, shape=(ids.shape[0],))
    if ids.shape[0]:
        _take_scalar_kernel(values, ids, out)
    return out


# ── Derived edges ───────────────────────────────────────────────────

@tack.kernel
def _emit_edges(cells, start, keys, signs):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_EDGES
        for e in range(cells.NUM_EDGES):
            a = cells.point_id(c, cells.edge_point(e, 0))
            b = cells.point_id(c, cells.edge_point(e, 1))
            lo, hi = _order(a, b)
            keys[first + e] = (tack.u64(lo) << tack.u64(32)) | tack.u64(hi)
            signs[first + e] = 1 if a < b else -1


@tack.kernel
def _edges_from_runs(order, offsets, keys, edge_rows, side_edge, count):
    for r in range(count):
        key = keys[order[offsets[r]]]
        edge_rows[r] = [tack.i32(key >> tack.u64(32)),
                        tack.i32(key & tack.u64(0xFFFFFFFF))]
        for i in range(offsets[r], offsets[r + 1]):
            side_edge[order[i]] = r


class Edges:
    """The edges of a topology's cells, as global entities.

    ``num_edges``; per edge: ``rows``, its (low, high) point ids, a 2-vector.
    Per (cell, local edge), in the topology's group layout: ``side_edge``
    (the edge id) and ``side_sign`` (+1 if the cell's local edge runs low to
    high, -1 if not). ``groups()`` are the edges, as one ``Line`` group.
    """

    @classmethod
    def _derive(cls, topology):
        self = cls()
        cell_groups = [g for g in topology.groups() if g.shape.NUM_EDGES]
        starts = np.concatenate([[0], np.cumsum([g.count * g.shape.NUM_EDGES
                                                 for g in cell_groups])]).astype(int)
        total = int(starts[-1])
        self.group_starts = {id(g): int(s) for g, s in zip(cell_groups, starts[:-1])}
        keys = tack.field(tack.u64, shape=(total,))
        self.side_sign = tack.field(tack.i32, shape=(total,))
        for group, start in zip(cell_groups, starts[:-1]):
            if group.count:
                _emit_edges(group.view(), int(start), keys, self.side_sign)
        self.side_edge = tack.field(tack.i32, shape=(total,))
        if total:
            order = argsort(keys)
            offsets, count = _run_offsets(gather(keys, order), total)
        else:
            count = 0
        self.num_edges = count
        self.rows = tack.Vector.field(2, tack.i32, shape=(count,))
        if count:
            _edges_from_runs(order, offsets, keys, self.rows, self.side_edge, count)
        return self

    def groups(self):
        ids = tack.arange(self.num_edges, tack.i32)
        return [DomainGroup(_Edges, shapes.Line, (self.rows, ids, self.num_edges),
                            self.num_edges, 0)]
