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

Ids are the topology's ``id_dtype`` (``tack.dataset.ids``), as on the shape path.

Orientation comes from the input, never from geometry. ``check_winding`` lists
the cells that do not walk each of their edges once in each direction;
``orient`` turns a face wound into its only cell around. ``as_polyhedra``
converts any shape-based dataset, reusing its derived faces as they stand.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids, shapes
from tack.dataset.buckets import bucket_order, run_offsets
from tack.dataset.topology import (
    _as_field,
    _checks_ids,
    _csr_problems,
    _id_problems,
    _OnItsArrays,
    _refuse,
    _Topology,
    edge_groups,
)
from tack.dataset.views import DomainGroup, _PolygonFaces, _PolyhedralCells, _PolylineCells
from tack.lang.array import Array
from tack.runtime.dispatch import on_inputs_device

__all__ = ["PolygonalTopology", "PolyhedralTopology", "PolylineTopology", "SizeBuckets",
           "orient"]


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
    counts = tack.zeros((2 * nf,), tack.i32)
    bad = tack.zeros((1,), tack.i32)
    if topology.num_cells:
        _count_sides(topology.cell_offsets, topology.cell_faces, topology.cell_face_sides,
                     counts, bad, topology.num_cells, nf)
    return counts, int(bad[0])


class PolygonFaces(_OnItsArrays):
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
        problems = tack.zeros((3,), tack.i32)
        if nf:
            _side_problems(counts, problems, nf)
        twice, inward, _ = (int(problems[i]) for i in range(3))
        if twice:
            raise ValueError(f"{twice} faces are on the same side of two cells: more than "
                             "two cells share a face, or two claim the same side of it")
        if inward:
            raise ValueError(f"{inward} faces are wound into their only cell (side 1 with "
                             "no side 0); tack.dataset.polyhedra.orient turns them around")
        self.id_dtype = topology.id_dtype
        self.sides = tack.empty((nf,), tack.vec(4, self.id_dtype))
        if nf:
            self.sides.fill(-1)
            _scatter_sides(topology.cell_offsets, topology.cell_faces, topology.cell_face_sides,
                           self.sides, topology.num_cells)
        self.side_face = topology.cell_faces
        self.side_slot = topology.cell_face_sides
        self._groups = None
        self._boundary = None

    @on_inputs_device
    def groups(self, subset=None):
        """The faces (or those in ``subset``, a field of face ids) as one
        ``Polygon`` group."""
        if subset is None and self._groups is not None:
            return self._groups
        face_ids = (ids.as_ids(subset, self.id_dtype) if subset is not None
                    else tack.arange(self.num_faces, dtype=self.id_dtype))
        t = self.topology
        n = face_ids.shape[0]
        groups = [DomainGroup(_PolygonFaces, t.facet_shape,
                              (t.face_offsets, t.face_points, face_ids, self.sides, n), n, 0)]
        if subset is None:
            self._groups = groups
        return groups

    @on_inputs_device
    def boundary(self):
        """The ids of the faces with one side, a field of ``id_dtype``."""
        if self._boundary is None:
            nf = self.num_faces
            flags = tack.empty((nf,), tack.i32)
            slots = tack.empty((nf,), self.id_dtype)
            if nf:
                _one_sided(self.sides, flags)
            count = exclusive_scan(flags, slots, nf) if nf else 0
            boundary = tack.empty((count,), self.id_dtype)
            if count:
                _compact(flags, slots, boundary)
            self._boundary = boundary
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
                keys[at] = [c, face_points[j]]
                at += 1


@tack.kernel
def _split_keys(keys, order, offsets, points, counts, n_runs):
    for r in range(n_runs):
        key = keys[order[offsets[r]]]
        points[r] = key[1]
        tack.atomic_add(counts, key[0], 1)


@tack.kernel
def _close(offsets, n, total):
    for i in range(1):
        offsets[n] = total


def _derive_cell_points(t, cell_offsets=None, cell_faces=None, face_offsets=None,
                        face_points=None):
    """Each cell's points, sorted and unique: ``(offsets, point_ids)``. The pairs
    ``(cell, point)`` of every cell's face points, sorted once: runs are the
    distinct pairs, in cell order and point order within a cell. The CSR arrays
    default to the topology's own."""
    n = t.num_cells
    idt = t.id_dtype
    cell_offsets = t.cell_offsets if cell_offsets is None else cell_offsets
    cell_faces = t.cell_faces if cell_faces is None else cell_faces
    face_offsets = t.face_offsets if face_offsets is None else face_offsets
    face_points = t.face_points if face_points is None else face_points
    sizes = tack.empty((n,), tack.i32)
    starts = tack.empty((n,), idt)
    if n:
        _cell_sizes(cell_offsets, cell_faces, face_offsets, sizes, n)
    total = exclusive_scan(sizes, starts, n) if n else 0
    keys = tack.empty((total,), tack.vec(2, idt))
    if total:
        _cell_point_keys(cell_offsets, cell_faces, face_offsets, face_points, starts,
                         keys, n)
    order, _ = bucket_order(keys, n, idt)
    offsets_of_runs, runs = run_offsets(keys, order)
    points = tack.empty((runs,), idt)
    counts = tack.zeros((n,), tack.i32)
    if runs:
        _split_keys(keys, order, offsets_of_runs, points, counts, runs)
    offsets = tack.empty((n + 1,), idt)
    if n:
        exclusive_scan(counts, offsets, n, total=False)
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
            keys[first + j] = [min(a, b), max(a, b)]
            signs[first + j] = 1 if a < b else -1


@tack.kernel
def _edges_from_runs(order, offsets, keys, rows, face_edge, count):
    for r in range(count):
        rows[r] = keys[order[offsets[r]]]
        for i in range(offsets[r], offsets[r + 1]):
            face_edge[order[i]] = r


class PolygonEdges(_OnItsArrays):
    """The edges of a polyhedral topology's faces, as global entities: ``num_edges``,
    ``rows`` (``(low, high)`` point ids), and per (face, local edge) -- indexed like
    ``face_points``, edge ``j`` of a face running from its point ``j`` to the next
    -- ``face_edge`` (the edge id) and ``face_edge_sign`` (+1 if the face's edge runs
    low to high). Numbered by ``(low, high)``, as the shape path numbers edges."""

    def __init__(self, topology):
        t = topology
        self.id_dtype = idt = t.id_dtype
        total = int(t.face_points.shape[0])
        keys = tack.empty((total,), tack.vec(2, idt))
        self.face_edge_sign = tack.empty((total,), tack.i32)
        self.face_edge = tack.empty((total,), idt)
        if total:
            _face_edge_keys(t.face_offsets, t.face_points, keys, self.face_edge_sign,
                            t.num_faces)
            order, _ = bucket_order(keys, t.num_points, idt)
            offsets, count = run_offsets(keys, order)
        else:
            count = 0
        self.num_edges = count
        self.rows = tack.empty((count,), tack.vec(2, idt))
        if count:
            _edges_from_runs(order, offsets, keys, self.rows, self.face_edge, count)

    @on_inputs_device
    def groups(self, subset=None):
        return edge_groups(self, subset)


# ── The topology ────────────────────────────────────────────────────

class PolyhedralTopology(_Topology):
    """Cells given by their faces, faces stored once (``docs/design/polyhedra.md``).

    ``face_offsets``/``face_points`` (CSR) are each face's points, in the order that
    winds it out of its side 0; ``cell_offsets``/``cell_faces`` (CSR) each cell's
    faces; ``cell_face_sides`` (u8, one per ``cell_faces`` entry) 0 where the face
    is wound out of that cell and 1 where into it. ``num_points`` defaults to one
    past the highest point id used. Ids are kept in ``id_dtype``, as
    ``UnstructuredTopology`` keeps them.

    Every cell is a ``Polyhedron``: no reference element, so no spaces with a basis
    (``L2``, ``H1`` above order 1) and no parametric coordinates.
    """

    #: No reference element: spaces needing one refuse a polyhedral topology.
    reference_cells = False
    #: Cells are polyhedra, facets polygons.
    dimension = 3
    cell_shape = shapes.Polyhedron
    facet_shape = shapes.Polygon

    _STORED = ("face_offsets", "face_points", "cell_offsets", "cell_faces", "cell_face_sides")

    def _moved(self, target):
        return PolyhedralTopology(
            *(getattr(self, name).to_device(target) for name in self._STORED),
            num_points=self.num_points, id_dtype=self.id_dtype)

    @on_inputs_device
    def __init__(self, face_offsets, face_points, cell_offsets, cell_faces, cell_face_sides,
                 num_points=None, id_dtype=None):
        face_offsets, face_points, cell_offsets, cell_faces = (
            ids.host_or_field(a) for a in (face_offsets, face_points, cell_offsets, cell_faces))
        if num_points is None:
            used = face_points.numpy() if isinstance(face_points, Array) else face_points
            num_points = int(used.max()) + 1 if used.size else 0
        # Cell points and directed edges hold each face's points once per side.
        extent = max(int(num_points), 2 * int(face_points.shape[0]),
                     2 * int(face_offsets.shape[0]), int(cell_faces.shape[0]))
        given = (face_offsets, face_points, cell_offsets, cell_faces)
        self.id_dtype = idt = ids.choose(id_dtype, extent, given)
        self.face_offsets = ids.as_ids(face_offsets, idt, check=False)
        self.face_points = ids.as_ids(face_points, idt, check=False)
        self.cell_offsets = ids.as_ids(cell_offsets, idt, check=False)
        self.cell_faces = ids.as_ids(cell_faces, idt, check=False)
        self.cell_face_sides = _as_field(cell_face_sides, tack.u8)
        self.num_faces = self.face_offsets.shape[0] - 1
        self.num_cells = self.cell_offsets.shape[0] - 1
        if self.num_faces < 0 or self.num_cells < 0:
            raise ValueError("offsets have one entry more than there are faces or cells")
        if self.cell_face_sides.shape != self.cell_faces.shape:
            raise ValueError("cell_face_sides needs one entry per cell_faces entry")
        self.num_points = int(num_points)
        self._cell_points = None
        self._groups = None
        if _checks_ids():
            _refuse("polyhedral topology", self._id_problems())

    def _id_problems(self):
        found = _csr_problems(self.face_offsets, self.face_points, self.num_points,
                              "face_offsets", "face_points", "num_points")
        found += _csr_problems(self.cell_offsets, self.cell_faces, self.num_faces,
                               "cell_offsets", "cell_faces", "the number of faces")
        n = self.cell_face_sides.shape[0]
        if n:
            problems = tack.zeros((3,), tack.i32)
            _id_problems(self.cell_face_sides, 2, problems, n)
            if problems[1]:
                found.append(f"{problems[1]} of the {n} entries of cell_face_sides are "
                             "neither 0 nor 1")
        return found

    def validate(self):
        _refuse("polyhedral topology", self._id_problems())
        return self

    @on_inputs_device
    def faces(self):
        """The faces' sides and boundary (``PolygonFaces``), derived on first use."""
        if self._faces is None:
            self._faces = PolygonFaces(self)
        return self._faces

    @on_inputs_device
    def edges(self):
        """The faces' edges (``PolygonEdges``), derived on first use."""
        if self._edges is None:
            self._edges = PolygonEdges(self)
        return self._edges

    def point_links(self):
        """Each point's entries in ``cell_points()``, derived on first use and kept:
        ``(offsets, entries)``, CSR by point, entries in increasing order (so in
        cell order)."""
        if self._point_links is None:
            _, point_ids = self.cell_points()
            order, offsets = bucket_order(point_ids, self.num_points, self.id_dtype)
            self._point_links = (offsets, order)
        return self._point_links

    @on_inputs_device
    def cell_points(self):
        """Each cell's points, sorted and unique: ``(offsets, point_ids)``, CSR."""
        if self._cell_points is None:
            self._cell_points = _derive_cell_points(self)
        return self._cell_points

    @on_inputs_device
    def groups(self):
        """The cells as one ``Polyhedron`` group."""
        if self._groups is None:
            point_offsets, point_ids = self.cell_points()
            self._groups = [DomainGroup(
                _PolyhedralCells, self.cell_shape,
                (self.cell_offsets, self.cell_faces, self.cell_face_sides, self.face_offsets,
                 self.face_points, point_offsets, point_ids, self.num_cells),
                self.num_cells, 0)]
        return self._groups

    @classmethod
    def from_cell_faces(cls, cells, num_points, positions=None, orient=False, id_dtype=None):
        """A topology from each cell's own list of faces (point-id rings), as VTK and many
        readers give them: copies of a face are matched by point set, the first copy
        found is the face -- its cell side 0 -- and a second copy, run the other way,
        is side 1. Two copies run the same way are refused.

        With ``orient``, the copies are made consistent first, on the host:
        within each cell, faces walk each shared edge in opposite directions;
        across cells, the two copies of a shared face run opposite ways (whole
        cells are reversed to make it so); then each connected component is
        turned outward as a whole, by the sign of its total volume -- the one
        geometric test, made once per component on a sum over many cells, which
        thin cells cannot fool as they fool a per-cell test. ``positions`` (one
        row per point) are needed for that last step.
        """
        cells = [[[int(p) for p in ring] for ring in faces] for faces in cells]
        if orient:
            if positions is None:
                raise ValueError("orient needs the points' positions, for the sign of each "
                                 "component's volume")
            cells = _orient_cells(cells, np.asarray(positions, float))
        known, face_points, face_offsets = {}, [], [0]
        cell_offsets, cell_faces, sides = [0], [], []
        same_way = 0
        for faces in cells:
            for ring in faces:
                key = tuple(sorted(ring))
                f = known.get(key)
                if f is None:
                    f = known[key] = len(face_offsets) - 1
                    face_points.extend(ring)
                    face_offsets.append(len(face_points))
                    side = 0
                else:
                    if not _opposite(face_points[face_offsets[f]:face_offsets[f + 1]], ring):
                        same_way += 1
                    side = 1
                cell_faces.append(f)
                sides.append(side)
            cell_offsets.append(len(cell_faces))
        if same_way:
            raise ValueError(f"{same_way} shared faces are wound the same way by both their "
                             "cells; from_cell_faces(..., orient=True) makes them consistent")
        return cls(np.array(face_offsets, np.int64), np.array(face_points, np.int64),
                   np.array(cell_offsets, np.int64), np.array(cell_faces, np.int64),
                   np.array(sides, np.uint8), num_points=num_points, id_dtype=id_dtype)

    def arrays(self):
        """The five defining arrays as host arrays, in constructor order."""
        return (self.face_offsets.numpy(), self.face_points.numpy(),
                self.cell_offsets.numpy(), self.cell_faces.numpy(),
                self.cell_face_sides.numpy())


# ── Size buckets ────────────────────────────────────────────────────

@tack.kernel
def _bucket_keys(sizes, caps, keys, largest, n):
    # The first cap at least half the cell's facet points; past the last, the
    # need is recorded (atomically, but only when something is wrong).
    for c in range(n):
        need = (sizes[c] + 1) // 2
        key = caps.shape[0]
        for i in range(caps.shape[0]):
            if key == caps.shape[0] and caps[i] >= need:
                key = i
        keys[c] = key
        if key == caps.shape[0]:
            tack.atomic_max(largest, 0, need)


class SizeBuckets(_OnItsArrays):
    """Groups a polyhedral topology's cells by size, for kernels that keep per-cell
    scratch: a cell's key is the first cap in ``caps`` at least half its facet
    points (an upper bound on its edges, so on its crossings of any isovalue).
    ``for_each``/``launch_groups`` given the buckets split the cells by them, and
    each launch's view carries its cap as the class constant ``MAX_SCRATCH``, so
    ``tack.local_array(dtype, cells.MAX_SCRATCH)`` has a compile-time size --
    as launches split by shape and order. One per topology and caps.
    """

    varies = True

    def __new__(cls, topology, caps=(16, 32, 64, 128, 256)):
        caps = tuple(int(c) for c in caps)
        kept = topology.__dict__.setdefault("_size_buckets", {})
        if caps not in kept:
            buckets = super().__new__(cls)
            buckets.topology = topology
            buckets.caps = caps
            kept[caps] = buckets
        return kept[caps]

    @on_inputs_device
    def cell_keys(self):
        """Each cell's bucket, an i32 array."""
        if "_keys" not in self.__dict__:
            t = self.topology
            n = t.num_cells
            field = tack.empty((n,), tack.i32)
            if n:
                sizes = tack.empty((n,), tack.i32)
                _cell_sizes(t.cell_offsets, t.cell_faces, t.face_offsets, sizes, n)
                caps = _as_field(np.asarray(self.caps, np.int32), tack.i32)
                largest = tack.zeros((1,), tack.i32)
                _bucket_keys(sizes, caps, field, largest, n)
                if largest[0]:
                    raise ValueError(f"a cell needs scratch for {largest[0]} entries, more "
                                     f"than the largest bucket, {self.caps[-1]}")
            self._keys = field
        return self._keys

    def domain_mixin(self, key):
        """The mixin that gives a launch of bucket ``key`` its ``MAX_SCRATCH``."""
        return _bucket_mixin(self.caps[key])


_bucket_mixins = {}


def _bucket_mixin(cap):
    if cap not in _bucket_mixins:
        _bucket_mixins[cap] = type(f"_Scratch{cap}", (), {"MAX_SCRATCH": cap})
    return _bucket_mixins[cap]


# ── Polygons: the same, one dimension down ──────────────────────────

@tack.kernel
def _loop_edge_keys(loop_offsets, loop_points, keys, directions, n_cells):
    for c in range(n_cells):
        first = loop_offsets[c]
        n = loop_offsets[c + 1] - first
        for j in range(n):
            a = loop_points[first + j]
            b = loop_points[first + (j + 1 if j + 1 < n else 0)]
            keys[first + j] = [min(a, b), max(a, b)]
            directions[first + j] = 1 if a < b else -1


@tack.kernel
def _facets_from_runs(order, offsets, keys, directions, face_points, cell_faces, sides,
                      problems, count):
    # problems: [0] an edge of three or more polygons, [1] two polygons walking
    # an edge the same way.
    for r in range(count):
        begin = offsets[r]
        end = offsets[r + 1]
        first = order[begin]
        key = keys[first]
        lo = key[0]
        hi = key[1]
        # The facet runs the way its first polygon walks it: out of side 0.
        if directions[first] > 0:
            face_points[2 * r] = lo
            face_points[2 * r + 1] = hi
        else:
            face_points[2 * r] = hi
            face_points[2 * r + 1] = lo
        if end - begin > 2:
            tack.atomic_add(problems, 0, 1)
        if end - begin == 2 and directions[order[begin + 1]] == directions[first]:
            tack.atomic_add(problems, 1, 1)
        for i in range(begin, end):
            cell_faces[order[i]] = r
            sides[order[i]] = tack.u8(0 if i == begin else 1)


@tack.kernel
def _every_other(offsets):
    for i in range(offsets.shape[0]):
        offsets[i] = 2 * i


class PolygonalTopology(PolyhedralTopology):
    """Polygons of any number of points, the polyhedral topology one dimension down:
    cells are polygons, their facets edges.

    Built from point loops -- ``loop_points[loop_offsets[c]:loop_offsets[c + 1]]``
    is polygon ``c``'s points in order -- as producers and isosurfaces give them.
    Facet ``k`` of a polygon is its edge from loop point ``k`` to the next. Edges
    are matched by sorting: the first polygon to use an edge is its side 0 and
    the edge runs its way; a second must run it the other way, or the winding is
    inconsistent (or the surface not orientable) and is refused, as is an edge
    of three polygons. ``loops()`` gives the loops back.
    """

    dimension = 2
    cell_shape = shapes.Polygon
    facet_shape = shapes.Line

    def _moved(self, target):
        return PolygonalTopology(self.loop_offsets.to_device(target),
                                 self.loop_points.to_device(target),
                                 num_points=self.num_points, id_dtype=self.id_dtype)

    @on_inputs_device
    def __init__(self, loop_offsets, loop_points, num_points=None, id_dtype=None):
        loop_offsets, loop_points = ids.host_or_field(loop_offsets), ids.host_or_field(loop_points)
        total = int(loop_points.shape[0])
        if num_points is None:
            used = loop_points.numpy() if isinstance(loop_points, Array) else loop_points
            num_points = int(used.max()) + 1 if total else 0
        # Its facets' points are two per edge, and there are as many edges as loop points.
        idt = ids.choose(id_dtype, max(int(num_points), 4 * total + 2),
                         given=(loop_offsets, loop_points))
        loop_offsets = ids.as_ids(loop_offsets, idt, check=False)
        loop_points = ids.as_ids(loop_points, idt, check=False)
        if _checks_ids():
            # Before the facets are derived from the loops, which index by point.
            _refuse("polygonal topology", _csr_problems(
                loop_offsets, loop_points, int(num_points), "loop_offsets", "loop_points",
                "num_points"))
        n = loop_offsets.shape[0] - 1
        keys = tack.empty((total,), tack.vec(2, idt))
        directions = tack.empty((total,), tack.i32)
        cell_faces = tack.empty((total,), idt)
        sides = tack.empty((total,), tack.u8)
        count = 0
        if total:
            _loop_edge_keys(loop_offsets, loop_points, keys, directions, n)
            order, _ = bucket_order(keys, num_points, idt)
            offsets, count = run_offsets(keys, order)
        face_points = tack.empty((2 * count,), idt)
        if count:
            problems = tack.zeros((2,), tack.i32)
            _facets_from_runs(order, offsets, keys, directions, face_points, cell_faces,
                              sides, problems, count)
            if problems[0]:
                raise ValueError(f"{problems[0]} edges are shared by more than two polygons")
            if problems[1]:
                raise ValueError(f"{problems[1]} edges are walked the same way by both their "
                                 "polygons: the winding is inconsistent, or the surface is "
                                 "not orientable")
        face_offsets = tack.empty((count + 1,), idt)
        _every_other(face_offsets)
        super().__init__(face_offsets, face_points, loop_offsets, cell_faces, sides,
                         num_points=num_points, id_dtype=idt)
        self.loop_offsets = loop_offsets
        self.loop_points = loop_points


    def validate(self):
        _refuse("polygonal topology", _csr_problems(
            self.loop_offsets, self.loop_points, self.num_points, "loop_offsets",
            "loop_points", "num_points"))
        return self
    def loops(self):
        """Each polygon's points in order: ``(loop_offsets, loop_points)``."""
        return self.loop_offsets, self.loop_points


# ── Polylines ───────────────────────────────────────────────────────

@tack.kernel
def _line_ends(line_offsets, line_points, ends, short, n_cells):
    # Each chain's first point, then its last; and how many have under two points.
    for c in range(n_cells):
        first = line_offsets[c]
        last = line_offsets[c + 1] - 1
        if last <= first:
            tack.atomic_add(short, 0, 1)
        ends[2 * c] = line_points[first]
        ends[2 * c + 1] = line_points[max(last, first)]


@tack.kernel
def _ends_from_runs(order, offsets, ends, face_points, cell_faces, sides, crowded, count):
    # A facet per point that chains end at: side 0 the first end there (in cell
    # order), side 1 the second; a third is a branch the faces cannot hold.
    for r in range(count):
        begin = offsets[r]
        end = offsets[r + 1]
        face_points[r] = ends[order[begin]]
        if end - begin > 2:
            tack.atomic_add(crowded, 0, 1)
        for i in range(min(end - begin, 2)):
            e = order[begin + i]
            cell_faces[e] = r
            sides[e] = tack.u8(i)


@tack.kernel
def _counting(values):
    for i in range(values.shape[0]):
        values[i] = i


class PolylineTopology(_Topology):
    """Polylines of any number of points, the polyhedral topology two dimensions
    down: each cell a chain of points, ``line_points[line_offsets[c]:line_offsets[c
    + 1]]``, two or more, as streamlines and VTK's ``VTK_POLY_LINE`` give them.

    Its facets (``faces()``, codimension 1) are the chains' ends: a facet per point
    where chains end, its normal along the chain out of its side 0 -- the first
    chain ending there, in cell order -- and into its side 1, the second. A vertex
    has no winding, so any two chains meeting end to end are consistent. Three or
    more ending at one point (a branch) the faces cannot hold, and ``faces()``
    refuses them, as an edge of three polygons is refused; the topology itself,
    and everything that does not need its faces, works. A chain may close (its
    last point its first): its two ends are then one interior facet. ``lines()``
    gives the chains back; a cell's points (``cell_points()``) are its chain's,
    sorted and unique.
    """

    #: No reference element: spaces needing one refuse a polyline topology.
    reference_cells = False
    dimension = 1
    cell_shape = shapes.PolyLine
    facet_shape = shapes.Vertex

    _STORED = ("line_offsets", "line_points")

    def _moved(self, target):
        return PolylineTopology(self.line_offsets.to_device(target),
                                self.line_points.to_device(target),
                                num_points=self.num_points, id_dtype=self.id_dtype)

    @on_inputs_device
    def __init__(self, line_offsets, line_points, num_points=None, id_dtype=None):
        line_offsets, line_points = ids.host_or_field(line_offsets), ids.host_or_field(line_points)
        total = int(line_points.shape[0])
        if num_points is None:
            used = line_points.numpy() if isinstance(line_points, Array) else line_points
            num_points = int(used.max()) + 1 if total else 0
        n = int(line_offsets.shape[0]) - 1
        if n < 0:
            raise ValueError("line_offsets has one entry more than there are polylines")
        self.id_dtype = idt = ids.choose(id_dtype, max(int(num_points), total, 2 * n + 1),
                                         given=(line_offsets, line_points))
        self.line_offsets = ids.as_ids(line_offsets, idt, check=False)
        self.line_points = ids.as_ids(line_points, idt, check=False)
        self.num_cells = n
        self.num_points = int(num_points)
        if _checks_ids():
            # Before the ends are read from the lines.
            self.validate()
        self._ends = tack.empty((2 * n,), idt)
        if n:
            short = tack.zeros((1,), tack.i32)
            _line_ends(self.line_offsets, self.line_points, self._ends, short, n)
            if short[0]:
                raise ValueError(f"{short[0]} polylines have fewer than two points")
        self._cell_points = None
        self._groups = None

    def validate(self):
        _refuse("polyline topology", _csr_problems(
            self.line_offsets, self.line_points, self.num_points, "line_offsets",
            "line_points", "num_points"))
        return self

    def lines(self):
        """Each polyline's points in order: ``(line_offsets, line_points)``."""
        return self.line_offsets, self.line_points

    @on_inputs_device
    def cell_points(self):
        """Each cell's points, sorted and unique: ``(offsets, point_ids)``, CSR."""
        if self._cell_points is None:
            # Each chain as its cell's one "face".
            identity = tack.empty((self.num_cells + 1,), self.id_dtype)
            _counting(identity)
            self._cell_points = _derive_cell_points(self, identity, identity, self.line_offsets,
                                                    self.line_points)
        return self._cell_points

    point_links = PolyhedralTopology.point_links

    @on_inputs_device
    def groups(self):
        """The cells as one ``PolyLine`` group."""
        if self._groups is None:
            point_offsets, point_ids = self.cell_points()
            self._groups = [DomainGroup(
                _PolylineCells, self.cell_shape,
                (self.line_offsets, self.line_points, point_offsets, point_ids, self.num_cells),
                self.num_cells, 0)]
        return self._groups

    @on_inputs_device
    def faces(self):
        """The chains' ends as faces (``PolygonFaces`` of one-point faces), derived on
        first use and kept; refuses a point where more than two chains end."""
        if self._faces is None:
            n, idt = self.num_cells, self.id_dtype
            cell_faces = tack.empty((2 * n,), idt)
            sides = tack.empty((2 * n,), tack.u8)
            count = 0
            if n:
                order, _ = bucket_order(self._ends, self.num_points, idt)
                offsets, count = run_offsets(self._ends, order)
            face_points = tack.empty((count,), idt)
            if count:
                crowded = tack.zeros((1,), tack.i32)
                _ends_from_runs(order, offsets, self._ends, face_points, cell_faces, sides,
                                crowded, count)
                if crowded[0]:
                    raise ValueError(
                        f"more than two polylines end at {crowded[0]} points: the faces of a "
                        "branching network are not defined (split it at the branches)")
            self.face_offsets = tack.empty((count + 1,), idt)
            _counting(self.face_offsets)
            self.face_points = face_points
            self.cell_offsets = tack.empty((n + 1,), idt)
            _every_other(self.cell_offsets)
            self.cell_faces = cell_faces
            self.cell_face_sides = sides
            self.num_faces = count
            self._faces = PolygonFaces(self)
        return self._faces

    @on_inputs_device
    def edges(self):
        """Not derived: a polyline's segments are parts of its cells, and nothing
        needs them as global entities yet."""
        raise NotImplementedError("a polyline topology's edges (its segments) are not "
                                  "derived")


# ── Orienting face lists ────────────────────────────────────────────

def _opposite(stored, walked):
    """Whether ``walked`` goes round the same polygon as ``stored`` the other way."""
    n = len(stored)
    walked = list(walked)
    if n != len(walked) or stored[0] not in walked:
        return False
    i = walked.index(stored[0])
    return [walked[(i - k) % n] for k in range(n)] == list(stored)


def _ring_edges(ring):
    return list(zip(ring, ring[1:] + ring[:1]))


def _orient_cells(cells, positions):
    """Each cell's faces made consistent with each other, the cells with their
    neighbours, and each connected component turned outward by its total volume."""
    from collections import deque

    for c, faces in enumerate(cells):
        # Within the cell: neighbouring faces walk their shared edge oppositely.
        by_edge = {}
        for k, ring in enumerate(faces):
            for a, b in _ring_edges(ring):
                by_edge.setdefault((min(a, b), max(a, b)), []).append(k)
        done = [False] * len(faces)
        for start in range(len(faces)):
            if done[start]:
                continue
            done[start] = True
            queue = deque([start])
            while queue:
                k = queue.popleft()
                for a, b in _ring_edges(faces[k]):
                    for other in by_edge[(min(a, b), max(a, b))]:
                        if other == k:
                            continue
                        walks_same = (a, b) in _ring_edges(faces[other])
                        if not done[other]:
                            if walks_same:
                                faces[other] = faces[other][::-1]
                            done[other] = True
                            queue.append(other)
                        elif walks_same:
                            raise ValueError(f"cell {c}'s faces cannot all be wound one way: "
                                             "it is not a closed, orientable polyhedron")
    # Across cells: the two copies of a shared face run opposite ways.
    by_face = {}
    for c, faces in enumerate(cells):
        for k, ring in enumerate(faces):
            by_face.setdefault(tuple(sorted(ring)), []).append((c, k))
    done = [False] * len(cells)
    for start in range(len(cells)):
        if done[start]:
            continue
        done[start] = True
        component, queue = [start], deque([start])
        while queue:
            c = queue.popleft()
            for ring in cells[c]:
                for d, k in by_face[tuple(sorted(ring))]:
                    if d == c:
                        continue
                    agree = _opposite(ring, cells[d][k])
                    if not done[d]:
                        if not agree:
                            cells[d] = [r[::-1] for r in cells[d]]
                        done[d] = True
                        component.append(d)
                        queue.append(d)
                    elif not agree:
                        raise ValueError("the cells cannot all be wound consistently: the "
                                         "mesh is not orientable")
        # The whole component outward, by the sign of its total volume.
        origin = positions[cells[start][0][0]]
        six = 0.0
        for c in component:
            for ring in cells[c]:
                p = positions[ring] - origin
                for j in range(1, len(ring) - 1):
                    six += p[0].dot(np.cross(p[j], p[j + 1]))
        if six < 0:
            for c in component:
                cells[c] = [r[::-1] for r in cells[c]]
    return cells


# ── Winding ─────────────────────────────────────────────────────────


def orient(topology):
    """A topology whose faces wound into their only cell are turned around: each such
    face's points reversed and its one reference made side 0. Other faces are left as
    they are; ``check_winding`` judges the rest."""
    face_offsets, face_points, cell_offsets, cell_faces, sides = topology.arrays()
    counts, bad = _side_counts(topology)
    if bad:
        raise ValueError(f"{bad} cell -> face entries name no face, or a bad side")
    counts = counts.numpy().reshape(-1, 2)
    inward = (counts[:, 0] == 0) & (counts[:, 1] == 1)
    face_points = face_points.copy()
    for f in np.flatnonzero(inward):
        a, b = face_offsets[f], face_offsets[f + 1]
        face_points[a:b] = face_points[a:b][::-1]
    sides = np.where(inward[cell_faces], 0, sides).astype(np.uint8)
    return PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, sides,
                              num_points=topology.num_points, id_dtype=topology.id_dtype)
