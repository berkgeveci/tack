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

Every id a topology stores or derives has its ``id_dtype`` (``tack.data.ids``):
``i32``, or ``i64`` for a topology too large for it or asked to use it.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import argsort
from tack.data import ids, shapes
from tack.data.buckets import bucket_order, run_offsets
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


@tack.kernel
def _corner_keys(cells, start, keys):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            keys[first + j] = cells.point_id(c, j)


def corner_layout(topology):
    """Where each group's (cell, corner) entries start in the layout of all of them,
    group by group, a cell's corners together (``index(c)`` within the group)."""
    groups = topology.groups()
    starts = np.concatenate([[0], np.cumsum([g.count * g.shape.NUM_POINTS
                                             for g in groups])]).astype(int)
    return {id(g): int(s) for g, s in zip(groups, starts[:-1])}, int(starts[-1])


class _Topology:
    """What both kinds share: the derived faces and edges, made once and kept."""

    _faces = None
    _edges = None
    _point_links = None

    def point_links(self):
        """Each point's cell corners, derived on first use and kept: ``(offsets,
        entries)``, CSR by point in ``id_dtype``, ``entries`` indexing the (cell,
        corner) layout of ``corner_layout`` in increasing order. Bucketed by point,
        as VTK builds its cell links, rather than sorted."""
        if self._point_links is None:
            start_of, total = corner_layout(self)
            keys = tack.field(self.id_dtype, shape=(total,))
            for group in self.groups():
                if group.count:
                    _corner_keys(group.view(), start_of[id(group)], keys)
            order, offsets = bucket_order(keys, self.num_points, self.id_dtype)
            self._point_links = (offsets, order)
        return self._point_links

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
    dataset may have points no cell uses, as VTK's may.

    ``offsets`` and ``connectivity`` are fields or arrays of ids, kept in
    ``id_dtype``: ``i32`` unless asked for ``tack.i64``, given ``i64`` fields,
    or too large for ``i32`` -- a mesh whose ids, connectivity or derived
    arrays (twelve edges a cell) pass ``2**31 - 1``."""

    def __init__(self, types, offsets, connectivity, num_points=None, id_dtype=None):
        self.types = _as_field(types, tack.u8)
        self.num_cells = self.types.shape[0]
        offsets, connectivity = ids.host_or_field(offsets), ids.host_or_field(connectivity)
        if offsets.shape != (self.num_cells + 1,):
            raise ValueError(f"offsets must have num_cells + 1 = {self.num_cells + 1} "
                             f"entries, not {offsets.shape[0]}")
        if num_points is None:
            used = connectivity.to_numpy() if isinstance(connectivity, Field) else connectivity
            num_points = int(used.max()) + 1 if used.size else 0
        self.num_points = int(num_points)
        extent = max(self.num_points, int(connectivity.shape[0]), 12 * self.num_cells)
        self.id_dtype = ids.choose(id_dtype, extent, given=(offsets, connectivity))
        self.offsets = ids.as_ids(offsets, self.id_dtype, check=False)
        self.connectivity = ids.as_ids(connectivity, self.id_dtype, check=False)
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
            order = argsort(keys, index_dtype=self.id_dtype)
            bad = tack.zeros(tack.i32, (1,))
            starts = np.concatenate([[0], np.cumsum(counts)])
            for t in present:
                shape = shapes.shape_class(t)
                count = int(counts[t])
                cell_ids = tack.field(self.id_dtype, shape=(count,))
                rows = tack.field(self.id_dtype, shape=(count, shape.NUM_POINTS))
                _gather_shape(order, int(starts[t]), self.offsets, self.connectivity,
                              cell_ids, rows, bad)
                groups.append(DomainGroup(_Cells, shape, (rows, cell_ids, count), count,
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

    def __init__(self, point_dims, id_dtype=None):
        point_dims = tuple(int(d) for d in point_dims)
        if not 1 <= len(point_dims) <= 3 or min(point_dims) < 2:
            raise ValueError(f"point_dims must be 1 to 3 sizes of at least 2, not {point_dims}")
        self.point_dims = point_dims
        self.num_cells = int(np.prod([d - 1 for d in point_dims]))
        self.num_points = int(np.prod(point_dims))
        self.id_dtype = ids.choose(id_dtype, max(self.num_points, 12 * self.num_cells))
        kind, shape = self._KINDS[len(point_dims)]
        dims = point_dims + (1,) * (3 - len(point_dims))
        # One group, kept: derived incidence is laid out per group.
        self._groups = [DomainGroup(kind, shape, (*dims, self.num_cells), self.num_cells, 0)]

    def groups(self):
        return self._groups


# ── Derived faces ───────────────────────────────────────────────────

# A triangle's fourth point in a face's row.
_NO_POINT = tack.constant(-1, tack.i32)


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
def _face_keys(rows, keys):
    # The face's points in increasing order, a triangle's pad (-1) kept last.
    for k in range(keys.shape[0]):
        row = rows[k]
        a, b = _order(row[0], row[1])
        if row[3] == _NO_POINT:
            b, c = _order(b, row[2])
            a, b = _order(a, b)
            keys[k] = [a, b, c, row[3]]
        else:
            c, d = _order(row[2], row[3])
            a, c = _order(a, c)
            b, d = _order(b, d)
            b, c = _order(b, c)
            keys[k] = [a, b, c, d]


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


class Faces:
    """The faces of a topology's 3D cells, as global entities.

    ``num_faces``; per face (by face id): ``kinds`` (``TRIANGLE`` or ``QUAD``),
    ``rows`` (its point ids in side 0's outward order, a 4-vector padded with
    -1), ``sides`` (``[cell0, local0, cell1, local1]``, the second
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
        self.id_dtype = idt = topology.id_dtype
        rows = tack.Vector.field(4, idt, shape=(total,))
        kinds = tack.field(tack.i32, shape=(total,))
        owners = tack.Vector.field(2, idt, shape=(total,))
        for group, start in zip(cell_groups, starts[:-1]):
            if group.count:
                _emit_faces(group.view(), int(start), rows, kinds, owners)

        self.side_face = tack.field(idt, shape=(total,))
        self.side_slot = tack.field(tack.i32, shape=(total,))
        self.side_orientation = tack.field(tack.i32, shape=(total,))
        if total:
            keys = tack.Vector.field(4, idt, shape=(total,))
            _face_keys(rows, keys)
            order, _ = bucket_order(keys, topology.num_points, idt)
            offsets, count = run_offsets(keys, order)
        else:
            count = 0
        self.num_faces = count
        self.rows = tack.Vector.field(4, idt, shape=(count,))
        self.kinds = tack.field(tack.i32, shape=(count,))
        self.sides = tack.Vector.field(4, idt, shape=(count,))
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
        """The faces (or those in ``subset``, a field of face ids) as one
        ``DomainGroup`` per face shape."""
        if subset is None and self._groups is not None:
            return self._groups
        idt = self.id_dtype
        ids_in = (ids.as_ids(subset, idt) if subset is not None
                  else tack.arange(self.num_faces, idt))
        n = ids_in.shape[0]
        kinds = _take_scalar(self.kinds, ids_in) if subset is not None else self.kinds
        rows = _take_vector(self.rows, ids_in) if subset is not None else self.rows
        sides = _take_vector(self.sides, ids_in) if subset is not None else self.sides
        groups = []
        for shape in (shapes.Triangle, shapes.Quad):
            flags = tack.field(tack.i32, shape=(n,))
            slots = tack.field(idt, shape=(n,))
            if n:
                _select_kind(kinds, int(shape.ID), flags)
            count = exclusive_scan(flags, slots, n) if n else 0
            positions = tack.field(idt, shape=(count,))
            group_rows = tack.Vector.field(4, idt, shape=(count,))
            group_sides = tack.Vector.field(4, idt, shape=(count,))
            if count:
                _gather_kind(flags, slots, rows, sides, positions, group_rows, group_sides)
            face_ids = _take_scalar(ids_in, positions) if count else positions
            groups.append(DomainGroup(_Faces, shape, (group_rows, face_ids, group_sides, count),
                                      count, 0))
        if subset is None:
            self._groups = groups
        return groups

    def boundary(self):
        """The ids of the faces with one side, a field of ``id_dtype``: the boundary
        side set."""
        if self._boundary is None:
            n = self.num_faces
            flags = tack.field(tack.i32, shape=(n,))
            slots = tack.field(self.id_dtype, shape=(n,))
            if n:
                _one_sided(self.sides, flags)
            count = exclusive_scan(flags, slots, n) if n else 0
            boundary = tack.field(self.id_dtype, shape=(count,))
            if count:
                _compact_ids(flags, slots, boundary)
            self._boundary = boundary
        return self._boundary


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
            keys[first + e] = [lo, hi]
            signs[first + e] = 1 if a < b else -1


@tack.kernel
def _edges_from_runs(order, offsets, keys, edge_rows, side_edge, count):
    for r in range(count):
        edge_rows[r] = keys[order[offsets[r]]]
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
        self.id_dtype = idt = topology.id_dtype
        keys = tack.Vector.field(2, idt, shape=(total,))
        self.side_sign = tack.field(tack.i32, shape=(total,))
        for group, start in zip(cell_groups, starts[:-1]):
            if group.count:
                _emit_edges(group.view(), int(start), keys, self.side_sign)
        self.side_edge = tack.field(idt, shape=(total,))
        if total:
            order, _ = bucket_order(keys, topology.num_points, idt)
            offsets, count = run_offsets(keys, order)
        else:
            count = 0
        self.num_edges = count
        self.rows = tack.Vector.field(2, idt, shape=(count,))
        if count:
            _edges_from_runs(order, offsets, keys, self.rows, self.side_edge, count)
        return self

    def groups(self):
        edge_ids = tack.arange(self.num_edges, self.id_dtype)
        return [DomainGroup(_Edges, shapes.Line, (self.rows, edge_ids, self.num_edges),
                            self.num_edges, 0)]
