"""Polyhedral topologies: every cell a list of faces, every face a polygon stored once.

``docs/design/polyhedra.md``, phase 1. A ``PolyhedralTopology`` holds

- ``face_offsets``/``face_points``: each face's points (CSR), in the order that
  winds the face out of its side 0 -- the cell its normal points away from;
- ``cell_offsets``/``cell_faces``: each cell's faces (CSR);
- ``cell_face_sides``: one u8 per cell -> face entry, 0 if the face is wound out
  of that cell (it is the face's side 0), 1 if into it. This is the shape path's
  ``side_slot`` under the same meaning.

Face numbering is the input's: a producer's face ids (an MPAS edge id) are the
face ids, so face data lines up with the file, and nothing re-derives them.
Everything else is derived on demand by sorting, and kept:

- ``faces()``: the shape path's two-sided record per face, ``[cell0, local0,
  cell1, local1]`` (``-1, -1`` for a missing side 1), and the boundary;
- ``cell_points()``: each cell's points, its face points sorted and unique --
  exactly what VTK keeps as a polyhedron's connectivity;
- ``edges()``: the faces' consecutive point pairs as ``(low, high)`` keys, sorted
  into runs -- the same numbering the shape path gives the same edges.

Orientation comes from the input, never from geometry. ``check_winding`` lists
the cells that do not walk each of their edges once in each direction;
``orient`` turns a face wound into its only cell around. ``as_polyhedra``
converts any shape-based dataset, reusing its derived faces as they stand.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import _run_offsets, argsort, gather, sort_by_key
from tack.data import shapes
from tack.data.topology import _as_field, _Topology
from tack.data.views import DomainGroup, _Edges, _PolygonFaces, _PolyhedralCells

__all__ = ["PolyhedralTopology", "as_polyhedra", "check_winding", "orient"]


# ── Sides ───────────────────────────────────────────────────────────

@tack.kernel
def _count_sides(cell_offsets, cell_faces, cell_face_sides, counts, bad, n_cells, n_faces):
    for c in range(n_cells):
        for e in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[e]
            s = tack.i32(cell_face_sides[e])
            if f < 0 or f >= n_faces or s > 1:
                tack.atomic_add(bad, 0, 1)
            else:
                tack.atomic_add(counts, 2 * f + s, 1)


@tack.kernel
def _scatter_sides(cell_offsets, cell_faces, cell_face_sides, sides, n_cells):
    for c in range(n_cells):
        for e in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[e]
            s = tack.i32(cell_face_sides[e])
            sides[f][2 * s] = c
            sides[f][2 * s + 1] = e - cell_offsets[c]


@tack.kernel
def _side_problems(counts, problems, n_faces):
    # problems: [0] a side used more than once, [1] side 1 without side 0,
    # [2] a face no cell uses.
    for f in range(n_faces):
        a = counts[2 * f]
        b = counts[2 * f + 1]
        if a > 1 or b > 1:
            tack.atomic_add(problems, 0, 1)
        if a == 0 and b == 1:
            tack.atomic_add(problems, 1, 1)
        if a == 0 and b == 0:
            tack.atomic_add(problems, 2, 1)


@tack.kernel
def _one_sided(sides, flags):
    for f in range(flags.shape[0]):
        flags[f] = 1 if sides[f][0] >= 0 and sides[f][2] < 0 else 0


@tack.kernel
def _compact(flags, slots, out):
    for i in range(flags.shape[0]):
        if flags[i] == 1:
            out[slots[i]] = i


def _side_counts(topology):
    """References per (face, side), and how many entries name no face."""
    nf = topology.num_faces
    counts = tack.zeros(tack.i32, (2 * nf,))
    bad = tack.zeros(tack.i32, (1,))
    if topology.num_cells:
        _count_sides(topology.cell_offsets, topology.cell_faces, topology.cell_face_sides,
                     counts, bad, topology.num_cells, nf)
    return counts, int(bad[0])


class PolygonFaces:
    """A polyhedral topology's faces, with the shape path's ``Faces`` interface:
    ``num_faces``, ``sides`` (``[cell0, local0, cell1, local1]`` per face),
    ``boundary()`` and ``groups()``. Per (cell, local face), the topology's own
    ``cell_faces`` and ``cell_face_sides`` are the shape path's ``side_face`` and
    ``side_slot``, in CSR rather than a fixed stride."""

    def __init__(self, topology):
        self.topology = topology
        self.num_faces = nf = topology.num_faces
        counts, bad = _side_counts(topology)
        if bad:
            raise ValueError(f"{bad} cell -> face entries name no face, or a side other "
                             "than 0 or 1")
        problems = tack.zeros(tack.i32, (3,))
        if nf:
            _side_problems(counts, problems, nf)
        twice, inward, _ = (int(problems[i]) for i in range(3))
        if twice:
            raise ValueError(f"{twice} faces are on the same side of two cells: more than "
                             "two cells share a face, or two claim the same side of it")
        if inward:
            raise ValueError(f"{inward} faces are wound into their only cell (side 1 with "
                             "no side 0); tack.data.polyhedra.orient turns them around")
        self.sides = tack.Vector.field(4, tack.i32, shape=(nf,))
        if nf:
            self.sides.from_numpy(np.full((nf, 4), -1, np.int32))
            _scatter_sides(topology.cell_offsets, topology.cell_faces, topology.cell_face_sides,
                           self.sides, topology.num_cells)
        self.side_face = topology.cell_faces
        self.side_slot = topology.cell_face_sides
        self._groups = None
        self._boundary = None

    def groups(self, subset=None):
        """The faces (or those in ``subset``, an i32 field of face ids) as one
        ``Polygon`` group."""
        if subset is None and self._groups is not None:
            return self._groups
        ids = subset if subset is not None else tack.arange(self.num_faces, tack.i32)
        t = self.topology
        groups = [DomainGroup(_PolygonFaces, shapes.Polygon,
                              (t.face_offsets, t.face_points, ids, self.sides, ids.shape[0]),
                              ids.shape[0], 0)]
        if subset is None:
            self._groups = groups
        return groups

    def boundary(self):
        """The ids of the faces with one side, an i32 field."""
        if self._boundary is None:
            nf = self.num_faces
            flags = tack.field(tack.i32, shape=(nf,))
            slots = tack.field(tack.i32, shape=(nf,))
            if nf:
                _one_sided(self.sides, flags)
            count = exclusive_scan(flags, slots, nf) if nf else 0
            ids = tack.field(tack.i32, shape=(count,))
            if count:
                _compact(flags, slots, ids)
            self._boundary = ids
        return self._boundary


# ── Cell points ─────────────────────────────────────────────────────

@tack.kernel
def _cell_sizes(cell_offsets, cell_faces, face_offsets, sizes, n_cells):
    for c in range(n_cells):
        total = 0
        for e in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[e]
            total += face_offsets[f + 1] - face_offsets[f]
        sizes[c] = total


@tack.kernel
def _cell_point_keys(cell_offsets, cell_faces, face_offsets, face_points, starts, keys,
                     n_cells):
    for c in range(n_cells):
        at = starts[c]
        for e in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[e]
            for j in range(face_offsets[f], face_offsets[f + 1]):
                keys[at] = (tack.u64(c) << tack.u64(32)) | tack.u64(face_points[j])
                at += 1


@tack.kernel
def _split_keys(keys, run_offsets, cells, points, counts, n_runs):
    for r in range(n_runs):
        key = keys[run_offsets[r]]
        c = tack.i32(key >> tack.u64(32))
        cells[r] = c
        points[r] = tack.i32(key & tack.u64(0xFFFFFFFF))
        tack.atomic_add(counts, c, 1)


@tack.kernel
def _close(offsets, n, total):
    for i in range(1):
        offsets[n] = total


def _derive_cell_points(t):
    """Each cell's points, sorted and unique: ``(offsets, point_ids)``. The pairs
    ``(cell, point)`` of every cell's face points, as u64 keys, sorted once: runs
    are the distinct pairs, in cell order and point order within a cell."""
    n = t.num_cells
    sizes = tack.field(tack.i32, shape=(n,))
    starts = tack.field(tack.i32, shape=(n,))
    if n:
        _cell_sizes(t.cell_offsets, t.cell_faces, t.face_offsets, sizes, n)
    total = exclusive_scan(sizes, starts, n) if n else 0
    keys = tack.field(tack.u64, shape=(total,))
    if total:
        _cell_point_keys(t.cell_offsets, t.cell_faces, t.face_offsets, t.face_points, starts,
                         keys, n)
        keys, _ = sort_by_key(keys)
    run_offsets, runs = _run_offsets(keys, total)
    cells = tack.field(tack.i32, shape=(runs,))
    points = tack.field(tack.i32, shape=(runs,))
    counts = tack.zeros(tack.i32, (n,))
    if runs:
        _split_keys(keys, run_offsets, cells, points, counts, runs)
    offsets = tack.field(tack.i32, shape=(n + 1,))
    if n:
        exclusive_scan(counts, offsets, n)
    _close(offsets, n, runs)
    return offsets, points


# ── Edges ───────────────────────────────────────────────────────────

@tack.kernel
def _face_edge_keys(face_offsets, face_points, keys, signs, n_faces):
    for f in range(n_faces):
        first = face_offsets[f]
        n = face_offsets[f + 1] - first
        for j in range(n):
            a = face_points[first + j]
            b = face_points[first + (j + 1 if j + 1 < n else 0)]
            lo = min(a, b)
            hi = max(a, b)
            keys[first + j] = (tack.u64(lo) << tack.u64(32)) | tack.u64(hi)
            signs[first + j] = 1 if a < b else -1


@tack.kernel
def _edges_from_runs(order, offsets, keys, rows, face_edge, count):
    for r in range(count):
        key = keys[order[offsets[r]]]
        rows[r] = [tack.i32(key >> tack.u64(32)), tack.i32(key & tack.u64(0xFFFFFFFF))]
        for i in range(offsets[r], offsets[r + 1]):
            face_edge[order[i]] = r


class PolygonEdges:
    """The edges of a polyhedral topology's faces, as global entities: ``num_edges``,
    ``rows`` (``(low, high)`` point ids), and per (face, local edge) -- indexed like
    ``face_points``, edge ``j`` of a face running from its point ``j`` to the next
    -- ``face_edge`` (the edge id) and ``face_edge_sign`` (+1 if the face's edge runs
    low to high). Numbered by ``(low, high)``, as the shape path numbers edges."""

    def __init__(self, topology):
        t = topology
        total = int(t.face_points.shape[0])
        keys = tack.field(tack.u64, shape=(total,))
        self.face_edge_sign = tack.field(tack.i32, shape=(total,))
        self.face_edge = tack.field(tack.i32, shape=(total,))
        if total:
            _face_edge_keys(t.face_offsets, t.face_points, keys, self.face_edge_sign,
                            t.num_faces)
            order = argsort(keys)
            offsets, count = _run_offsets(gather(keys, order), total)
        else:
            count = 0
        self.num_edges = count
        self.rows = tack.Vector.field(2, tack.i32, shape=(count,))
        if count:
            _edges_from_runs(order, offsets, keys, self.rows, self.face_edge, count)

    def groups(self):
        ids = tack.arange(self.num_edges, tack.i32)
        return [DomainGroup(_Edges, shapes.Line, (self.rows, ids, self.num_edges),
                            self.num_edges, 0)]


# ── The topology ────────────────────────────────────────────────────

class PolyhedralTopology(_Topology):
    """Cells given by their faces, faces stored once (``docs/design/polyhedra.md``).

    ``face_offsets``/``face_points`` (CSR, i32) are each face's points, in the order
    that winds it out of its side 0; ``cell_offsets``/``cell_faces`` (CSR, i32) each
    cell's faces; ``cell_face_sides`` (u8, one per ``cell_faces`` entry) 0 where the
    face is wound out of that cell and 1 where into it. ``num_points`` defaults to one
    past the highest point id used.

    Every cell is a ``Polyhedron``: no reference element, so no spaces with a basis
    (``L2``, ``H1`` above order 1) and no parametric coordinates.
    """

    #: No reference element: spaces needing one refuse a polyhedral topology.
    reference_cells = False

    def __init__(self, face_offsets, face_points, cell_offsets, cell_faces, cell_face_sides,
                 num_points=None):
        self.face_offsets = _as_field(face_offsets, tack.i32)
        self.face_points = _as_field(face_points, tack.i32)
        self.cell_offsets = _as_field(cell_offsets, tack.i32)
        self.cell_faces = _as_field(cell_faces, tack.i32)
        self.cell_face_sides = _as_field(cell_face_sides, tack.u8)
        self.num_faces = self.face_offsets.shape[0] - 1
        self.num_cells = self.cell_offsets.shape[0] - 1
        if self.num_faces < 0 or self.num_cells < 0:
            raise ValueError("offsets have one entry more than there are faces or cells")
        if self.cell_face_sides.shape != self.cell_faces.shape:
            raise ValueError("cell_face_sides needs one entry per cell_faces entry")
        if num_points is None:
            used = self.face_points.to_numpy()
            num_points = int(used.max()) + 1 if used.size else 0
        self.num_points = int(num_points)
        self._cell_points = None
        self._groups = None

    def faces(self):
        """The faces' sides and boundary (``PolygonFaces``), derived on first use."""
        if self._faces is None:
            self._faces = PolygonFaces(self)
        return self._faces

    def edges(self):
        """The faces' edges (``PolygonEdges``), derived on first use."""
        if self._edges is None:
            self._edges = PolygonEdges(self)
        return self._edges

    def cell_points(self):
        """Each cell's points, sorted and unique: ``(offsets, point_ids)``, i32 CSR."""
        if self._cell_points is None:
            self._cell_points = _derive_cell_points(self)
        return self._cell_points

    def groups(self):
        """The cells as one ``Polyhedron`` group."""
        if self._groups is None:
            point_offsets, point_ids = self.cell_points()
            self._groups = [DomainGroup(
                _PolyhedralCells, shapes.Polyhedron,
                (self.cell_offsets, self.cell_faces, self.cell_face_sides, self.face_offsets,
                 self.face_points, point_offsets, point_ids, self.num_cells),
                self.num_cells, 0)]
        return self._groups

    def arrays(self):
        """The five defining arrays as host arrays, in constructor order."""
        return (self.face_offsets.to_numpy(), self.face_points.to_numpy(),
                self.cell_offsets.to_numpy(), self.cell_faces.to_numpy(),
                self.cell_face_sides.to_numpy())


# ── Winding ─────────────────────────────────────────────────────────

@tack.kernel
def _directed_edges(cell_offsets, cell_faces, cell_face_sides, face_offsets, face_edge,
                    face_edge_sign, starts, keys, directions, n_cells):
    # Each cell walks each face outward: forward on side 0, backward on side 1,
    # so each face edge runs along its stored direction or against it.
    for c in range(n_cells):
        at = starts[c]
        for e in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[e]
            flip = 1 - 2 * tack.i32(cell_face_sides[e])
            for j in range(face_offsets[f], face_offsets[f + 1]):
                keys[at] = (tack.u64(c) << tack.u64(32)) | tack.u64(face_edge[j])
                directions[at] = flip * face_edge_sign[j]
                at += 1


@tack.kernel
def _unbalanced(keys, directions, run_offsets, bad_cells, n_runs):
    for r in range(n_runs):
        begin = run_offsets[r]
        end = run_offsets[r + 1]
        total = 0
        for i in range(begin, end):
            total += directions[i]
        if end - begin != 2 or total != 0:
            bad_cells[tack.i32(keys[begin] >> tack.u64(32))] = 1


def check_winding(data):
    """The cells whose faces do not wind consistently, a sorted host array of cell ids.

    Walking every face outward (``side_point``), a closed, consistently wound cell
    walks each of its edges exactly twice, once each way. A cell that does not --
    a face with the wrong side, an open cell, an edge of three faces -- is listed.
    Combinatorial: geometry plays no part.
    """
    t = getattr(data, "topology", data)
    edges = t.edges()
    n = t.num_cells
    sizes = tack.field(tack.i32, shape=(n,))
    starts = tack.field(tack.i32, shape=(n,))
    if n:
        _cell_sizes(t.cell_offsets, t.cell_faces, t.face_offsets, sizes, n)
    total = exclusive_scan(sizes, starts, n) if n else 0
    bad = tack.zeros(tack.i32, (n,))
    if total:
        keys = tack.field(tack.u64, shape=(total,))
        directions = tack.field(tack.i32, shape=(total,))
        _directed_edges(t.cell_offsets, t.cell_faces, t.cell_face_sides, t.face_offsets,
                        edges.face_edge, edges.face_edge_sign, starts, keys, directions, n)
        keys, directions = sort_by_key(keys, directions)
        run_offsets, runs = _run_offsets(keys, total)
        _unbalanced(keys, directions, run_offsets, bad, runs)
    return np.flatnonzero(bad.to_numpy())


def orient(topology):
    """A topology whose faces wound into their only cell are turned around: each such
    face's points reversed and its one reference made side 0. Other faces are left as
    they are; ``check_winding`` judges the rest."""
    face_offsets, face_points, cell_offsets, cell_faces, sides = topology.arrays()
    counts, bad = _side_counts(topology)
    if bad:
        raise ValueError(f"{bad} cell -> face entries name no face, or a bad side")
    counts = counts.to_numpy().reshape(-1, 2)
    inward = (counts[:, 0] == 0) & (counts[:, 1] == 1)
    face_points = face_points.copy()
    for f in np.flatnonzero(inward):
        a, b = face_offsets[f], face_offsets[f + 1]
        face_points[a:b] = face_points[a:b][::-1]
    sides = np.where(inward[cell_faces], 0, sides).astype(np.uint8)
    return PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, sides,
                              num_points=topology.num_points)


# ── From a shape-based dataset ──────────────────────────────────────

@tack.kernel
def _face_sizes(kinds, sizes):
    for f in range(kinds.shape[0]):
        sizes[f] = 3 if kinds[f] == shapes.TRIANGLE else 4


@tack.kernel
def _face_rows(rows, starts, sizes, out):
    for f in range(sizes.shape[0]):
        row = rows[f]
        for j in range(sizes[f]):
            out[starts[f] + j] = row[j]


@tack.kernel
def _cell_face_counts(cells, counts):
    for c in cells:
        counts[cells.entity_id(c)] = cells.NUM_FACES


@tack.kernel
def _cell_face_entries(cells, offsets, cell_faces, cell_face_sides):
    for c in cells:
        at = offsets[cells.entity_id(c)]
        for f in range(cells.NUM_FACES):
            cell_faces[at + f] = cells.face_id(c, f)
            cell_face_sides[at + f] = tack.u8(cells.face_side(c, f))


def as_polyhedra(data):
    """A shape-based dataset as a polyhedral one, every cell a polyhedron.

    Its derived faces already hold every face once, in side 0's outward order
    (a voxel's pixel faces as quads), with the side of each cell's reference, so
    they become the polyhedral faces as they stand: face ids, and so fields on
    faces, are unchanged. Edges come out numbered as before. Cells must be 3D.
    ``Constant`` and ``H1`` order-1 fields, values on points, cells, faces and
    edges, and sets carry over; fields needing a reference element are dropped.
    """
    from tack.data.dataset import DataSet, Field
    from tack.data.spaces import H1, Constant, Values

    if not isinstance(data.geometry.space, H1) or data.geometry.space.order != 1:
        raise TypeError("as_polyhedra needs an order-1 H1 geometry: positions per point")
    topology = data.topology
    for group in topology.groups():
        if group.count and not group.shape.NUM_FACES:
            raise ValueError(f"{group.shape.__name__} cells have no faces: a polyhedral "
                             "topology is of 3D cells")
    faces = topology.faces()
    nf = faces.num_faces
    sizes = tack.field(tack.i32, shape=(nf,))
    starts = tack.field(tack.i32, shape=(nf + 1,))
    if nf:
        _face_sizes(faces.kinds, sizes)
    total = exclusive_scan(sizes, starts, nf) if nf else 0
    _close(starts, nf, total)
    face_points = tack.field(tack.i32, shape=(total,))
    if nf:
        _face_rows(faces.rows, starts, sizes, face_points)

    n = data.num_cells
    counts = tack.zeros(tack.i32, (n,))
    for group in topology.groups():
        if group.count:
            _cell_face_counts(group.view(), counts)
    cell_offsets = tack.field(tack.i32, shape=(n + 1,))
    entries = exclusive_scan(counts, cell_offsets, n) if n else 0
    _close(cell_offsets, n, entries)
    cell_faces = tack.field(tack.i32, shape=(entries,))
    cell_face_sides = tack.field(tack.u8, shape=(entries,))
    for group in topology.groups():
        if group.count:
            _cell_face_entries(data.domain_view("cells", group), cell_offsets, cell_faces,
                               cell_face_sides)
    out = PolyhedralTopology(starts, face_points, cell_offsets, cell_faces, cell_face_sides,
                             num_points=data.num_points)

    def carry(field):
        space = field.space
        if isinstance(space, H1) and space.order == 1:
            return Field(H1(out), field.values)
        if isinstance(space, Constant):
            return Field(Constant(out), field.values)
        if isinstance(space, Values):
            return Field(Values(out, space.on), field.values)
        return None

    fields = {}
    for name, field in data.fields.items():
        if name != "shape" and carry(field) is not None:
            fields[name] = carry(field)
    return DataSet(out, Field(H1(out), data.geometry.values), fields=fields,
                   sets=dict(data.sets))
