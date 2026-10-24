"""Topology: cells, and the faces and edges derived from them.

Two kinds of topology, as in ``docs/design/dataset-api.md`` §3.1:

- ``UnstructuredTopology(types, offsets, connectivity)``: any mix of the
  linear shapes, stored as a ``vtkUnstructuredGrid`` stores them.
- ``StructuredTopology(point_dims)``: the cells of a structured grid,
  connectivity implied by the dimensions.

Both give their cells as *groups*, one per shape present, which a kernel
iterates one at a time (``tack.dataset.for_each``). Faces and edges are not
stored; ``faces()`` and ``edges()`` derive them from any topology by the
same kernels, through the groups' cell views: every (cell, local face) is
keyed by its sorted point ids, two stable sorts bring a face's copies
together, and each run of equal keys is one face. Its first copy is *side
0* -- the face's points are listed in that cell's outward order -- and a
second copy, if any, is side 1. Edges are runs of (low, high) point-id
pairs; each (cell, local edge) records whether it runs low to high.

Every id a topology stores or derives has its ``id_dtype`` (``tack.dataset.ids``):
``i32``, or ``i64`` for a topology too large for it or asked to use it.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import argsort
from tack.arrays import Cast, ConstantArray, CountingArray, _Implicit, size_of
from tack.dataset import ids, shapes
from tack.dataset.buckets import bucket_order, run_offsets
from tack.dataset.views import (
    DomainGroup,
    _Cells,
    _Edges,
    _Faces,
    _StructuredHexahedra,
    _StructuredLines,
    _StructuredQuads,
)
from tack.lang.array import Array
from tack.runtime.dispatch import as_device, backend_of, get_backend, on_inputs_device, same_device

__all__ = ["Edges", "Faces", "StructuredTopology", "UnstructuredTopology"]


def _as_field(values, dtype):
    if isinstance(values, (Array, _Implicit)):
        if values.dtype != dtype:
            raise TypeError(f"expected a {dtype.name} field, got {values.dtype.name}")
        return values
    values = np.ascontiguousarray(values)
    out = tack.empty(values.shape, dtype)
    if values.size:
        tack.copyto(out, values.astype(dtype.numpy_dtype))
    return out


def uniform_cells(shape_id, count, id_dtype, points_per_cell=None):
    """``(types, offsets)`` of ``count`` cells of one VTK type, each of
    ``points_per_cell`` points (the shape's own count by default), as implicit
    arrays: a ``ConstantArray`` and a ``CountingArray``, computed where they are
    read and stored nowhere -- VTK's constant and affine arrays, as a filter
    whose cells are all one shape (a contour's triangles, vertex cells) gives
    them."""
    k = points_per_cell or shapes.shape_class(shape_id).NUM_POINTS
    return (ConstantArray(shape_id, count, dtype=tack.u8),
            CountingArray(count + 1, start=0, step=k, dtype=id_dtype))


def _ids_of(values, id_dtype):
    """A topology's offsets or connectivity in ``id_dtype``: an implicit array kept
    implicit -- a counting or constant one remade in that type, so it stays what
    it is (and crosses to VTK as an affine or constant array), anything else
    cast -- and a stored one converted."""
    if isinstance(values, _Implicit):
        if values.dtype == id_dtype:
            return values
        if isinstance(values, CountingArray):
            return CountingArray(values.size, start=values.start, step=values.step,
                                 dtype=id_dtype)
        if isinstance(values, ConstantArray):
            return ConstantArray(int(values.value.numpy().reshape(-1)[0]), values.size,
                                 dtype=id_dtype, device=values.device)
        return Cast(values, id_dtype)
    return ids.as_ids(values, id_dtype, check=False)


# ── Cell groups ─────────────────────────────────────────────────────

@tack.kernel
def _histogram_types(types, keys, counts):
    for c in range(keys.shape[0]):              # types may be implicit: no shape
        t = tack.i32(types[c])
        keys[c] = t
        tack.atomic_add(counts, t, 1)


# On the CPU the types are counted by chunks, each into a private table, and the
# tables summed: no atomics. With one atomic counter per type, a mesh of one
# type put every worker on one counter: 34 ms for 1M cells against 2.7 ms on
# one thread, nearly all of a fresh topology's cost. At most _TYPE_CHUNKS
# chunks, so the tables stay small.
_TYPE_CHUNK = 4096
_TYPE_CHUNKS = 1024


@tack.kernel
def _count_types(types, keys, chunk_counts, n, chunk, nchunks):
    # chunk_counts[t * nchunks + c]: the cells of type t in chunk c.
    for c in range(nchunks):
        table = tack.local_array(tack.i32, 256)
        for t in range(256):
            table[t] = 0
        start = c * chunk
        end = min(start + chunk, n)
        for i in range(start, end):
            t = tack.i32(types[i])
            keys[i] = t
            table[t] = table[t] + 1
        for t in range(256):
            chunk_counts[t * nchunks + c] = table[t]


@tack.kernel
def _sum_type_counts(chunk_counts, counts, nchunks):
    for t in range(256):
        total = 0
        for c in range(nchunks):
            total += chunk_counts[t * nchunks + c]
        counts[t] = total


def _type_counts(types, n):
    """Each cell's type as an i32 key, and the count of each of the 256 types."""
    keys = tack.empty((n,), tack.i32)
    counts = tack.zeros((256,), tack.i32)
    if get_backend().supports_workgroups:
        # GPUs run the chunk tables as few threads with spilled tables; their
        # atomics stay.
        _histogram_types(types, keys, counts)
        return keys, counts
    chunk = max(_TYPE_CHUNK, -(-n // _TYPE_CHUNKS))
    nchunks = -(-n // chunk)
    chunk_counts = tack.empty((256 * nchunks,), tack.i32)
    _count_types(types, keys, chunk_counts, n, chunk, nchunks)
    _sum_type_counts(chunk_counts, counts, nchunks)
    return keys, counts


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
    return {g: int(s) for g, s in zip(groups, starts[:-1])}, int(starts[-1])


class _OnItsArrays:
    """For an object whose device is that of the arrays it holds (derived faces
    and edges): ``_device``, what `on_inputs_device` reads, so its methods run
    there."""

    @property
    def _device(self):
        return same_device(type(self).__name__, vars(self).values())


@tack.kernel
def _range_problems(offsets, total, problems, n):
    # Ranges [offsets[c], offsets[c + 1]) that run backwards or leave [0, total]
    # (problems[0]), and offsets that do not start at 0 and end at total ([2]).
    for c in range(n):
        a = offsets[c]
        b = offsets[c + 1]
        if a > b or a < 0 or b > total:
            tack.atomic_add(problems, 0, 1)
        if (c == 0 and a != 0) or (c == n - 1 and b != total):
            tack.atomic_add(problems, 2, 1)


@tack.kernel
def _id_problems(members, limit, problems, n):
    # Entries outside [0, limit) (problems[1]); members may be implicit: no shape.
    for k in range(n):
        m = members[k]
        if m < 0 or m >= limit:
            tack.atomic_add(problems, 1, 1)


def _csr_problems(offsets, members, limit, offsets_name, members_name, limit_name):
    """What is wrong with a CSR pair whose members are ids below ``limit``: a list of
    sentences, empty when nothing is. Reads back three integers."""
    n = size_of(offsets) - 1
    total = size_of(members)
    problems = tack.zeros((3,), tack.i32)
    if n > 0:
        _range_problems(offsets, total, problems, n)
    if total:
        _id_problems(members, limit, problems, total)
    p = problems.numpy()
    found = []
    if p[2] or (n == 0 and total):
        found.append(f"{offsets_name} does not start at 0 and end at {total}, the length of "
                     f"{members_name}")
    if p[0]:
        found.append(f"{p[0]} of the {n} ranges in {offsets_name} run backwards or leave "
                     f"[0, {total}]")
    if p[1]:
        found.append(f"{p[1]} of the {total} entries of {members_name} are outside "
                     f"[0, {limit}) ({limit_name})")
    return found


def _refuse(kind, found):
    if found:
        raise ValueError(f"invalid {kind}: " + "; ".join(found))


def _checks_ids():
    """Whether topologies check their ids when made: in debug mode
    (`tack.set_debug`, ``TACK_DEBUG``), where a bad id would otherwise be read
    or written out of bounds by the first derivation that follows it."""
    from tack.runtime.dispatch import debug_enabled

    return debug_enabled()


class _Topology:
    """What both kinds share: the derived faces and edges, made once and kept.

    A topology is on the device its arrays are on (``device``); everything
    derived from it is made there, its entry points running under that device
    (`on_inputs_device`), and its algorithms run there too.
    """

    _faces = None
    _edges = None
    _point_links = None
    #: The last `CellLocator` made for this topology's cells without one given
    #: (`locator.cached_locator`), for whatever geometry it was made from.
    _locator = None

    #: The arrays it is built from, which say its device.
    _STORED = ()

    def validate(self):
        """Check the arrays it is built from, on their device, reading back a few
        integers: offsets start at 0, never run backwards and end at their
        members' length; every id is within its range (points, faces); the
        cells' types and point counts agree. Raises ``ValueError`` saying what
        is wrong and how often; returns the topology. Debug mode
        (`tack.set_debug`) checks the ids of every topology when it is made.
        A structured topology holds no arrays and has nothing to check."""
        return self

    @property
    def _device(self):
        found = self.__dict__.get("_found_device")
        if found is None:
            found = same_device(type(self).__name__,
                                [getattr(self, name, None) for name in self._STORED])
            if found is not None:           # its arrays never change: keep it
                self.__dict__["_found_device"] = found
        return found

    @property
    def device(self):
        """The `tack.Device` its arrays are on, where it and its datasets compute."""
        return as_device(self._device or get_backend())

    def to_device(self, device):
        """This topology on ``device``: itself if it is already there, else one
        made from its arrays copied there (what it derives is derived afresh)."""
        target = as_device(backend_of(device))
        if target._backend.memory_domain == self.device._backend.memory_domain:
            return self
        return self._moved(target)

    def _moved(self, target):
        """A copy on ``target`` (a `tack.Device`), from its arrays copied there."""
        raise NotImplementedError

    @on_inputs_device
    def point_links(self):
        """Each point's cell corners, derived on first use and kept: ``(offsets,
        entries)``, CSR by point in ``id_dtype``, ``entries`` indexing the (cell,
        corner) layout of ``corner_layout`` in increasing order. Bucketed by point,
        as VTK builds its cell links, rather than sorted."""
        if self._point_links is None:
            start_of, total = corner_layout(self)
            keys = tack.empty((total,), self.id_dtype)
            for group in self.groups():
                if group.count:
                    _corner_keys(group.view(), start_of[group], keys)
            order, offsets = bucket_order(keys, self.num_points, self.id_dtype)
            self._point_links = (offsets, order)
        return self._point_links

    @property
    def dimension(self):
        """The cells' highest dimension: 3 for a mesh with solids, 2 for a surface."""
        return max((g.shape.DIMENSION for g in self.groups() if g.count), default=0)

    @on_inputs_device
    def faces(self):
        """The faces (``Faces``): codimension 1 of the cells of the highest
        dimension -- polygons of a 3D mesh, edges of a 2D one. Derived on first
        use and kept."""
        if self._faces is None:
            if self.dimension == 1:
                # Lines declare no faces (a line's would be its end points),
                # and deriving none made a line mesh's boundary empty.
                raise NotImplementedError(
                    "the faces of a mesh of lines (a 1D grid's, say) are not derived: "
                    "their faces would be points. A PolylineTopology's are its chains' ends")
            self._faces = Faces._derive(self)
        return self._faces

    @on_inputs_device
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
    arrays (twelve edges a cell) pass ``2**31 - 1``.

    Any of the three may be an implicit array (``tack.arrays``), kept as it is:
    the types of cells all of one shape a ``ConstantArray``, the offsets of
    cells all of one size a ``CountingArray`` (``uniform_cells`` makes both),
    the connectivity of cells that each have their own points a
    ``CountingArray`` too. VTK's constant and affine arrays map onto them."""

    _STORED = ("types", "offsets", "connectivity")

    @on_inputs_device
    def __init__(self, types, offsets, connectivity, num_points=None, id_dtype=None):
        self.types = _as_field(types, tack.u8)
        self.num_cells = size_of(self.types)
        offsets, connectivity = (v if isinstance(v, _Implicit) else ids.host_or_field(v)
                                 for v in (offsets, connectivity))
        if size_of(offsets) != self.num_cells + 1:
            raise ValueError(f"offsets must have num_cells + 1 = {self.num_cells + 1} "
                             f"entries, not {size_of(offsets)}")
        if num_points is None:
            used = (connectivity.numpy() if isinstance(connectivity, (Array, _Implicit))
                    else connectivity)
            num_points = int(used.max()) + 1 if used.size else 0
        self.num_points = int(num_points)
        extent = max(self.num_points, size_of(connectivity), 12 * self.num_cells)
        if id_dtype is None and any(isinstance(v, _Implicit) and v.dtype == tack.i64
                                    for v in (offsets, connectivity)):
            id_dtype = tack.i64
        self.id_dtype = ids.choose(id_dtype, extent,
                                   given=[v for v in (offsets, connectivity)
                                          if isinstance(v, Array)])
        self.offsets = _ids_of(offsets, self.id_dtype)
        self.connectivity = _ids_of(connectivity, self.id_dtype)
        self._groups = None
        if _checks_ids():
            _refuse("unstructured topology", self._id_problems())

    def _id_problems(self):
        return _csr_problems(self.offsets, self.connectivity, self.num_points, "offsets",
                             "connectivity", "num_points")

    def validate(self):
        _refuse("unstructured topology", self._id_problems())
        self._groups = None
        self.groups()                    # the types are shapes, of their point counts
        return self

    def _moved(self, target):
        return UnstructuredTopology(self.types.to_device(target),
                                    self.offsets.to_device(target),
                                    self.connectivity.to_device(target),
                                    num_points=self.num_points, id_dtype=self.id_dtype)

    @on_inputs_device
    def groups(self):
        if self._groups is not None:
            return self._groups
        n = self.num_cells
        groups = []
        bad = tack.zeros((1,), tack.i32)
        if n and isinstance(self.types, ConstantArray):
            # One shape: no histogram of the types and no sort; the cells in order.
            t = int(self.types.value.numpy().reshape(-1)[0])
            if t not in shapes._BY_ID:
                raise ValueError(f"cell types [{t}] are not linear shapes")
            groups.append(self._gathered(t, CountingArray(n, dtype=self.id_dtype), 0, n, bad))
        elif n:
            keys, counts = _type_counts(self.types, n)
            counts = counts.numpy()
            present = np.flatnonzero(counts)
            unknown = [int(t) for t in present if int(t) not in shapes._BY_ID]
            if unknown:
                raise ValueError(f"cell types {unknown} are not linear shapes")
            order = argsort(keys, index_dtype=self.id_dtype)
            starts = np.concatenate([[0], np.cumsum(counts)])
            for t in present:
                groups.append(self._gathered(int(t), order, int(starts[t]), int(counts[t]),
                                             bad))
        if bad[0]:
            raise ValueError(f"{bad[0]} cells have a point count that does not "
                             "match their type")
        self._groups = groups
        return groups

    def _gathered(self, t, order, start, count, bad):
        """The group of the ``count`` cells of type ``t``, ``order[start:start +
        count]``: their ids, and their point ids gathered into rows."""
        shape = shapes.shape_class(t)
        cell_ids = tack.empty((count,), self.id_dtype)
        rows = tack.empty((count, shape.NUM_POINTS), self.id_dtype)
        _gather_shape(order, start, self.offsets, self.connectivity, cell_ids, rows, bad)
        return DomainGroup(_Cells, shape, (rows, cell_ids, count), count, start)


class StructuredTopology(_Topology):
    """The cells of a structured grid of ``point_dims`` points, x fastest: lines, quads
    or hexahedra in ``vtkStructuredGrid``'s order, iterated by (i, j, k).

    ``offset`` is the grid index of its first point in a larger grid, one per
    axis (VTK's extent's start; zeros by default): a piece ``split`` cut, or an
    image whose extent starts elsewhere. What depends on a cell's place in the
    whole grid -- ``tetrahedralize``'s alternating splits -- reads it, so
    neighbouring pieces agree."""

    _KINDS = {1: (_StructuredLines, shapes.Line), 2: (_StructuredQuads, shapes.Quad),
              3: (_StructuredHexahedra, shapes.Hexahedron)}

    def __init__(self, point_dims, id_dtype=None, device=None, offset=None):
        point_dims = tuple(int(d) for d in point_dims)
        if not 1 <= len(point_dims) <= 3 or min(point_dims) < 2:
            raise ValueError(f"point_dims must be 1 to 3 sizes of at least 2, not {point_dims}")
        offset = (0,) * len(point_dims) if offset is None else tuple(int(o) for o in offset)
        if len(offset) != len(point_dims):
            raise ValueError(f"offset needs one index per axis ({len(point_dims)}), not "
                             f"{offset}")
        self.point_dims = point_dims
        self.offset = offset
        self.num_cells = int(np.prod([d - 1 for d in point_dims]))
        self.num_points = int(np.prod(point_dims))
        self.id_dtype = ids.choose(id_dtype, max(self.num_points, 12 * self.num_cells))
        kind, shape = self._KINDS[len(point_dims)]
        dims = point_dims + (1,) * (3 - len(point_dims))
        # One group, kept: derived incidence is laid out per group.
        self._groups = [DomainGroup(kind, shape, (*dims, self.num_cells), self.num_cells, 0)]
        # It holds no arrays, so its device is the one it is made for: ``device``,
        # or the default device when it is made.
        self._found_device = backend_of(device)

    def _moved(self, target):
        return StructuredTopology(self.point_dims, id_dtype=self.id_dtype, device=target,
                                  offset=self.offset)

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
            n = cells.face_num_points(f)
            p0 = cells.point_id(c, cells.face_corner(f, 0))
            p1 = cells.point_id(c, cells.face_corner(f, 1))
            p2 = cells.point_id(c, cells.face_corner(f, 2)) if n >= 3 else _NO_POINT
            p3 = cells.point_id(c, cells.face_corner(f, 3)) if n == 4 else _NO_POINT
            rows[first + f] = [p0, p1, p2, p3]
            kinds[first + f] = cells.face_shape(f)
            owners[first + f] = [cells.entity_id(c), f]


@tack.kernel
def _face_keys(rows, keys):
    # The face's points in increasing order, the pads (-1) of a triangle or a
    # line kept last.
    for k in range(keys.shape[0]):
        row = rows[k]
        a, b = _order(row[0], row[1])
        if row[2] == _NO_POINT:
            keys[k] = [a, b, row[2], row[3]]
        elif row[3] == _NO_POINT:
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
        n = 2 if mine[2] == _NO_POINT else (3 if mine[3] == _NO_POINT else 4)
        r = 0
        for j in range(n):
            if face[j] == mine[0]:
                r = j
        after = r + 1 if r + 1 < n else 0
        before = r - 1 if r > 0 else n - 1
        # A line's points are the same forwards and back from its second: the
        # side that starts there is reflected, as side 1 of a 3D face is.
        if face[after] == mine[1] and (n > 2 or r == 0):
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


@tack.func
def _before(i, j, k, nx, ny, a, b, c):
    """How many points before (i, j, k), x fastest, have their own i < a, j < b and
    k < c: on a grid, the entities of one kind that start at earlier points."""
    across = min(nx, a)
    n = min(k, c) * across * min(ny, b)
    if k < c:
        n += min(j, b) * across
        if j < b:
            n += min(i, a)
    return n


@tack.kernel
def _structured_faces_by_cell(cells, nx, ny, nz, solid, face_axis, face_high, face_turned,
                              side_face, side_slot, side_orientation, face_rows, face_kinds,
                              face_sides):
    # A structured grid's faces, one pass over the cells, no sort: each local
    # face's normal axis and side are the shape's (face_axis, face_high), so its
    # face id follows from the cell's (i, j, k), numbered as the general
    # derivation's sort numbers faces (by smallest point, then z-, y-, x-normal
    # in 3D, the edge along x before the edge along y in 2D). Side 0 is the cell
    # below the face along its normal, or the only one on the boundary, and its
    # face is the face's row; side 1's orientation is the shape's (face_turned).
    cx = nx - 1
    cy = ny - 1
    cz = nz - 1
    for c in cells:
        e = cells.entity_id(c)
        ci = e % cx
        rest = e // cx
        cj = rest % max(cy, 1)
        ck = rest // max(cy, 1)
        first = cells.index(c) * cells.NUM_FACES
        for lf in range(cells.NUM_FACES):
            axis = face_axis[lf]
            high = face_high[lf]
            i = ci + (high if axis == 0 else 0)
            j = cj + (high if axis == 1 else 0)
            k = ck + (high if axis == 2 else 0)
            pos = ci if axis == 0 else (cj if axis == 1 else ck)
            f = e - e
            if solid == 1:
                f = (_before(i, j, k, nx, ny, nx, cy, cz) + _before(i, j, k, nx, ny, cx, ny, cz)
                     + _before(i, j, k, nx, ny, cx, cy, nz))
                z_here = 1 if i < cx and j < cy else 0
                y_here = 1 if i < cx and k < cz else 0
                if axis == 0:
                    f += z_here + y_here
                elif axis == 1:
                    f += z_here
            else:
                f = _before(i, j, k, nx, ny, cx, ny, 1) + _before(i, j, k, nx, ny, nx, cy, 1)
                if axis == 0:
                    f += 1 if i < cx else 0
            slot = 1 if high == 0 and pos > 0 else 0
            side_face[first + lf] = f
            side_slot[first + lf] = slot
            side_orientation[first + lf] = face_turned[lf] if slot == 1 else 0
            if slot == 0:
                n = cells.face_num_points(lf)
                p0 = cells.point_id(c, cells.face_corner(lf, 0))
                p1 = cells.point_id(c, cells.face_corner(lf, 1))
                p2 = cells.point_id(c, cells.face_corner(lf, 2)) if n >= 3 else _NO_POINT
                p3 = cells.point_id(c, cells.face_corner(lf, 3)) if n == 4 else _NO_POINT
                face_rows[f] = [p0, p1, p2, p3]
                face_kinds[f] = cells.face_shape(lf)
                face_sides[f][0] = e
                face_sides[f][1] = lf
            else:
                face_sides[f][2] = e
                face_sides[f][3] = lf


# The shapes' corner tables, read on the host for the structured tables.
_HEXAHEDRON_FACE_CORNERS = shapes._HEXAHEDRON_FACES
_HEXAHEDRON_EDGE_CORNERS = shapes._HEXAHEDRON_EDGES
_QUAD_EDGE_CORNERS = shapes._QUAD_EDGES


def _corner_offset(j):
    """A structured cell's corner ``j`` as (di, dj, dk), VTK's order: the formula
    ``_StructuredHexahedra.point_id`` and ``by_id._GridCells`` use."""
    return ((j ^ (j >> 1)) & 1, (j >> 1) & 1, j >> 2)


_FACE_TABLES = {}


def _structured_face_tables(shape):
    """Per local face of a structured cell (``Hexahedron``, or ``Quad`` whose faces
    are its edges): its normal axis, whether it is the cell's high side on it,
    and the orientation its cell has as the face's side 1 -- read once from the
    general derivation of a tiny grid, so the two cannot disagree."""
    if shape not in _FACE_TABLES:
        solid = shape.DIMENSION == 3
        axis, high = [], []
        for lf in range(shape.NUM_FACES):
            corners = ([_HEXAHEDRON_FACE_CORNERS[4 * lf + m] for m in range(4)] if solid
                       else [_QUAD_EDGE_CORNERS[2 * lf + m] for m in range(2)])
            offsets = [_corner_offset(int(cc)) for cc in corners]
            for a in range(3 if solid else 2):
                values = {o[a] for o in offsets}
                if len(values) == 1:
                    axis.append(a)
                    high.append(values.pop())
                    break
        tiny = StructuredTopology((3, 3, 3) if solid else (3, 3))
        faces = Faces._derive_sorted(tiny)
        slots = faces.side_slot.numpy()
        orientations = faces.side_orientation.numpy()
        turned = [0] * shape.NUM_FACES
        for s_, (slot, orientation) in enumerate(zip(slots, orientations)):
            if slot == 1:
                turned[s_ % shape.NUM_FACES] = int(orientation)
        _FACE_TABLES[shape] = (np.array(axis, np.int32), np.array(high, np.int32),
                               np.array(turned, np.int32))
    return _FACE_TABLES[shape]


class Faces(_OnItsArrays):
    """The faces of a topology's cells of the highest dimension, as global entities:
    a 3D mesh's polygons, a 2D mesh's edges.

    ``num_faces``; per face (by face id): ``kinds`` (``TRIANGLE``, ``QUAD`` or
    ``LINE``), ``rows`` (its point ids in side 0's outward order, a 4-vector
    padded with -1), ``sides`` (``[cell0, local0, cell1, local1]``, the second
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
        if isinstance(topology, StructuredTopology) and topology.dimension >= 2:
            return cls._derive_structured(topology)
        return cls._derive_sorted(topology)

    @classmethod
    def _derive_sorted(cls, topology):
        """Any topology's faces: emitted by every cell, matched by a sort of their
        points."""
        self = cls()
        # Codimension 1 of the top dimension: a surface's edges, not also the
        # edges of its polygons' lines; a solid mesh's faces, not its triangles'.
        top = topology.dimension
        cell_groups = [g for g in topology.groups()
                       if g.shape.NUM_FACES and g.shape.DIMENSION == top]
        starts = np.concatenate([[0], np.cumsum([g.count * g.shape.NUM_FACES
                                                 for g in cell_groups])]).astype(int)
        total = int(starts[-1])
        self.group_starts = {g: int(s) for g, s in zip(cell_groups, starts[:-1])}
        self.id_dtype = idt = topology.id_dtype
        rows = tack.empty((total,), tack.vec(4, idt))
        kinds = tack.empty((total,), tack.i32)
        owners = tack.empty((total,), tack.vec(2, idt))
        for group, start in zip(cell_groups, starts[:-1]):
            if group.count:
                _emit_faces(group.view(), int(start), rows, kinds, owners)

        self.side_face = tack.empty((total,), idt)
        self.side_slot = tack.empty((total,), tack.i32)
        self.side_orientation = tack.empty((total,), tack.i32)
        if total:
            keys = tack.empty((total,), tack.vec(4, idt))
            _face_keys(rows, keys)
            order, _ = bucket_order(keys, topology.num_points, idt)
            offsets, count = run_offsets(keys, order)
        else:
            count = 0
        self.num_faces = count
        self.rows = tack.empty((count,), tack.vec(4, idt))
        self.kinds = tack.empty((count,), tack.i32)
        self.sides = tack.empty((count,), tack.vec(4, idt))
        if count:
            too_many = tack.zeros((1,), tack.i32)
            _faces_from_runs(order, offsets, rows, kinds, owners, self.rows, self.kinds,
                             self.sides, self.side_face, self.side_slot, too_many, count)
            if too_many[0]:
                raise ValueError(f"{too_many[0]} faces are shared by more than two cells")
            bad = tack.zeros((1,), tack.i32)
            _side_orientations(rows, self.side_face, self.rows, self.side_orientation, bad)
            if bad[0]:
                raise ValueError(f"{bad[0]} cell faces do not go round their face's points "
                                 "in either direction")
        self._groups = None
        self._boundary = None
        return self

    @classmethod
    def _derive_structured(cls, topology):
        """A structured grid's faces by arithmetic, one pass over the cells with no
        sort, numbered, sided and oriented exactly as the general derivation
        would (``_structured_face_tables``)."""
        self = cls()
        (group,) = topology.groups()
        solid = topology.dimension == 3
        nx, ny, nz = topology.point_dims + (1,) * (3 - len(topology.point_dims))
        cx, cy, cz = nx - 1, ny - 1, nz - 1
        total = group.count * group.shape.NUM_FACES
        self.group_starts = {group: 0}
        self.id_dtype = idt = topology.id_dtype
        count = (nx * cy * cz + cx * ny * cz + cx * cy * nz) if solid else (nx * cy + cx * ny)
        self.num_faces = count
        self.side_face = tack.empty((total,), idt)
        self.side_slot = tack.empty((total,), tack.i32)
        self.side_orientation = tack.empty((total,), tack.i32)
        self.rows = tack.empty((count,), tack.vec(4, idt))
        self.kinds = tack.empty((count,), tack.i32)
        self.sides = tack.empty((count,), tack.vec(4, idt))
        self.sides.fill(-1)
        axis, high, turned = _structured_face_tables(group.shape)
        _structured_faces_by_cell(group.view(), nx, ny, nz, 1 if solid else 0,
                                  tack.array(axis), tack.array(high), tack.array(turned),
                                  self.side_face, self.side_slot, self.side_orientation,
                                  self.rows, self.kinds, self.sides)
        self._groups = None
        self._boundary = None
        return self

    @on_inputs_device
    def groups(self, subset=None):
        """The faces (or those in ``subset``, a field of face ids) as one
        ``DomainGroup`` per face shape."""
        if subset is None and self._groups is not None:
            return self._groups
        idt = self.id_dtype
        ids_in = (ids.as_ids(subset, idt) if subset is not None
                  else tack.arange(self.num_faces, dtype=idt))
        n = ids_in.shape[0]
        kinds = _take_scalar(self.kinds, ids_in) if subset is not None else self.kinds
        rows = _take_vector(self.rows, ids_in) if subset is not None else self.rows
        sides = _take_vector(self.sides, ids_in) if subset is not None else self.sides
        groups = []
        for shape in (shapes.Line, shapes.Triangle, shapes.Quad):
            flags = tack.empty((n,), tack.i32)
            slots = tack.empty((n,), idt)
            if n:
                _select_kind(kinds, int(shape.ID), flags)
            count = exclusive_scan(flags, slots, n) if n else 0
            positions = tack.empty((count,), idt)
            group_rows = tack.empty((count,), tack.vec(4, idt))
            group_sides = tack.empty((count,), tack.vec(4, idt))
            if count:
                _gather_kind(flags, slots, rows, sides, positions, group_rows, group_sides)
            face_ids = _take_scalar(ids_in, positions) if count else positions
            groups.append(DomainGroup(_Faces, shape, (group_rows, face_ids, group_sides, count),
                                      count, 0))
        if subset is None:
            self._groups = groups
        return groups

    @on_inputs_device
    def boundary(self):
        """The ids of the faces with one side, a field of ``id_dtype``: the boundary
        side set."""
        if self._boundary is None:
            n = self.num_faces
            flags = tack.empty((n,), tack.i32)
            slots = tack.empty((n,), self.id_dtype)
            if n:
                _one_sided(self.sides, flags)
            count = exclusive_scan(flags, slots, n) if n else 0
            boundary = tack.empty((count,), self.id_dtype)
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
    out = tack.empty((ids.shape[0],), values.dtype)
    if ids.shape[0]:
        _take_scalar_kernel(values, ids, out)
    return out


def _take_vector(values, ids):
    out = tack.empty((ids.shape[0],), tack.vec(values._vector_n, values.scalar_dtype))
    if ids.shape[0]:
        _take_scalar_kernel(values, ids, out)
    return out


# ── Derived edges ───────────────────────────────────────────────────

@tack.kernel
def _emit_edges(cells, start, keys, signs):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_EDGES
        for e in range(cells.NUM_EDGES):
            a = cells.point_id(c, cells.edge_corner(e, 0))
            b = cells.point_id(c, cells.edge_corner(e, 1))
            lo, hi = _order(a, b)
            keys[first + e] = [lo, hi]
            signs[first + e] = 1 if a < b else -1


@tack.kernel
def _edges_from_runs(order, offsets, keys, edge_rows, side_edge, count):
    for r in range(count):
        edge_rows[r] = keys[order[offsets[r]]]
        for i in range(offsets[r], offsets[r + 1]):
            side_edge[order[i]] = r


@tack.kernel
def _structured_edges_by_cell(cells, nx, ny, nz, edge_axis, edge_low, edge_sign, side_edge,
                              side_sign):
    # A structured grid's edge of each (cell, local edge), one pass over the
    # cells, no sort: each local edge's axis, low corner and direction are the
    # shape's, so its id follows from the cell's (i, j, k), numbered as the
    # general derivation's sort numbers edges (by low point, then along x, y, z).
    cx = nx - 1
    cy = ny - 1
    cz = nz - 1
    for c in cells:
        e = cells.entity_id(c)
        ci = e % cx
        rest = e // cx
        cj = rest % max(cy, 1)
        ck = rest // max(cy, 1)
        first = cells.index(c) * cells.NUM_EDGES
        for le in range(cells.NUM_EDGES):
            low = edge_low[le]
            i = ci + ((low ^ (low >> 1)) & 1)
            j = cj + ((low >> 1) & 1)
            k = ck + (low >> 2)
            g = (_before(i, j, k, nx, ny, cx, ny, nz) + _before(i, j, k, nx, ny, nx, cy, nz)
                 + _before(i, j, k, nx, ny, nx, ny, cz))
            axis = edge_axis[le]
            if axis == 1:
                g += 1 if i < cx else 0
            elif axis == 2:
                g += (1 if i < cx else 0) + (1 if j < cy else 0)
            side_edge[first + le] = g
            side_sign[first + le] = edge_sign[le]


_EDGE_TABLES = {}


def _structured_edge_tables(shape):
    """Per local edge of a structured cell (``Hexahedron`` or ``Quad``): its axis,
    its low corner, and +1 if it runs low to high (its first corner the low
    one), else -1."""
    if shape not in _EDGE_TABLES:
        table = _HEXAHEDRON_EDGE_CORNERS if shape.DIMENSION == 3 else _QUAD_EDGE_CORNERS
        axis, low, sign = [], [], []
        for le in range(shape.NUM_EDGES):
            a_, b_ = int(table[2 * le]), int(table[2 * le + 1])
            oa, ob = _corner_offset(a_), _corner_offset(b_)
            along = next(d for d in range(3) if oa[d] != ob[d])
            axis.append(along)
            low.append(a_ if oa[along] < ob[along] else b_)
            sign.append(1 if oa[along] < ob[along] else -1)
        _EDGE_TABLES[shape] = tuple(np.array(v, np.int32) for v in (axis, low, sign))
    return _EDGE_TABLES[shape]


@tack.kernel
def _structured_edge_rows(nx, ny, nz, rows):
    # Each point's edges, in order, where its count of earlier edges puts them.
    cx = nx - 1
    cy = ny - 1
    cz = nz - 1
    for p in range(nx * ny * nz):
        i = p % nx
        rest = p // nx
        j = rest % ny
        k = rest // ny
        e = (_before(i, j, k, nx, ny, cx, ny, nz) + _before(i, j, k, nx, ny, nx, cy, nz)
             + _before(i, j, k, nx, ny, nx, ny, cz))
        if i < cx:
            rows[e] = [p, p + 1]
            e += 1
        if j < cy:
            rows[e] = [p, p + nx]
            e += 1
        if k < cz:
            rows[e] = [p, p + nx * ny]


class Edges(_OnItsArrays):
    """The edges of a topology's cells, as global entities.

    ``num_edges``; per edge: ``rows``, its (low, high) point ids, a 2-vector.
    Per (cell, local edge), in the topology's group layout: ``side_edge``
    (the edge id) and ``side_sign`` (+1 if the cell's local edge runs low to
    high, -1 if not). ``groups()`` are the edges (or a subset of them), as one
    ``Line`` group.
    """

    @classmethod
    def _derive(cls, topology):
        if isinstance(topology, StructuredTopology) and topology.dimension >= 2:
            return cls._derive_structured(topology)
        self = cls()
        cell_groups = [g for g in topology.groups() if g.shape.NUM_EDGES]
        starts = np.concatenate([[0], np.cumsum([g.count * g.shape.NUM_EDGES
                                                 for g in cell_groups])]).astype(int)
        total = int(starts[-1])
        self.group_starts = {g: int(s) for g, s in zip(cell_groups, starts[:-1])}
        self.id_dtype = idt = topology.id_dtype
        keys = tack.empty((total,), tack.vec(2, idt))
        self.side_sign = tack.empty((total,), tack.i32)
        for group, start in zip(cell_groups, starts[:-1]):
            if group.count:
                _emit_edges(group.view(), int(start), keys, self.side_sign)
        self.side_edge = tack.empty((total,), idt)
        if total:
            order, _ = bucket_order(keys, topology.num_points, idt)
            offsets, count = run_offsets(keys, order)
        else:
            count = 0
        self.num_edges = count
        self.rows = tack.empty((count,), tack.vec(2, idt))
        if count:
            _edges_from_runs(order, offsets, keys, self.rows, self.side_edge, count)
        return self

    @classmethod
    def _derive_structured(cls, topology):
        """A structured grid's edges by arithmetic, one pass over the cells with no
        sort, numbered and signed exactly as the general derivation would."""
        self = cls()
        (group,) = topology.groups()
        nx, ny, nz = topology.point_dims + (1,) * (3 - len(topology.point_dims))
        cx, cy, cz = nx - 1, ny - 1, nz - 1
        total = group.count * group.shape.NUM_EDGES
        self.group_starts = {group: 0}
        self.id_dtype = idt = topology.id_dtype
        self.side_edge = tack.empty((total,), idt)
        self.side_sign = tack.empty((total,), tack.i32)
        axis, low, sign = _structured_edge_tables(group.shape)
        _structured_edges_by_cell(group.view(), nx, ny, nz, tack.array(axis), tack.array(low),
                                  tack.array(sign), self.side_edge, self.side_sign)
        count = cx * ny * nz + nx * cy * nz + nx * ny * cz
        self.num_edges = count
        self.rows = tack.empty((count,), tack.vec(2, idt))
        _structured_edge_rows(nx, ny, nz, self.rows)
        return self

    @on_inputs_device
    def groups(self, subset=None):
        return edge_groups(self, subset)


def edge_groups(edges, subset=None):
    """``edges``' edges (or those in ``subset``, a field of edge ids) as one ``Line``
    ``DomainGroup``."""
    if subset is None:
        n = edges.num_edges
        rows = edges.rows
        edge_ids = CountingArray(n, dtype=edges.id_dtype)        # computed, not stored
    else:
        edge_ids = ids.as_ids(subset, edges.id_dtype)
        n = edge_ids.shape[0]
        rows = _take_vector(edges.rows, edge_ids)
    return [DomainGroup(_Edges, shapes.Line, (rows, edge_ids, n), n, 0)]
