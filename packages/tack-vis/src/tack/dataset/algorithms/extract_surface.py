"""``extract_surface``: the faces of set ``name`` as a surface dataset: triangles and quads on the same points, each in its side 0's outward order. Fields on faces become the surface's cell fields, and so do cell fields (``Constant``, values on cells), each face taking its side 0 cell's value; point fields (``H1``, values on points) stay on the same points. A solid's order-2 field or geometry stays order 2 on its surface, gathered by entity: the faces' points, edges and quad centers are the surface's. The geometry must be ``H1``: an ``L2`` one has no shared points for the faces to stand on."""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids as id_types
from tack.dataset import shapes
from tack.dataset.carry import Pieces, Same, Take, _take, carry, selected
from tack.dataset.dataset import Field
from tack.dataset.sets import Set
from tack.dataset.spaces import H1
from tack.dataset.topology import UnstructuredTopology, _face_row
from tack.runtime.dispatch import on_inputs_device

__all__ = ['extract_surface']


@tack.kernel
def _surface_sizes(ids, kinds, sizes):
    for i in range(ids.shape[0]):
        kind = kinds[ids[i]]
        sizes[i] = 2 if kind == 3 else (3 if kind == 5 else 4)


@tack.kernel
def _surface_cells(ids, kinds, rows, starts, types, offsets, connectivity, length, n):
    for i in range(n):
        f = ids[i]
        at = starts[i]
        types[i] = tack.u8(kinds[f])
        offsets[i] = at
        row = rows[f]
        for j in range(2 if kinds[f] == 3 else (3 if kinds[f] == 5 else 4)):
            connectivity[at + j] = row[j]
        if i == 0:
            offsets[n] = length


@tack.kernel
def _face_cells(ids, sides, out):
    for i in range(ids.shape[0]):
        out[i] = sides[ids[i]][0]


@tack.kernel
def _singletons(records, flags):
    # 1 for a record whose key no neighbour shares: a face of one cell.
    n = flags.shape[0]
    for i in range(n):
        r = records[i]
        alone = 1
        if i > 0:
            p = records[i - 1]
            if p[0] == r[0] and p[1] == r[1] and p[2] == r[2] and p[3] == r[3]:
                alone = 0
        if i + 1 < n:
            q = records[i + 1]
            if q[0] == r[0] and q[1] == r[1] and q[2] == r[2] and q[3] == r[3]:
                alone = 0
        flags[i] = alone


@tack.kernel
def _boundary_slots(records, flags, slots, entry_slot, sizes):
    # Each boundary face's place in the surface (the faces' order, by key), by its
    # (cell, face) entry; its size from its key's pads.
    for i in range(flags.shape[0]):
        if flags[i] == 1:
            r = records[i]
            entry_slot[r[4]] = slots[i]
            sizes[slots[i]] = 2 if r[2] == -1 else (3 if r[3] == -1 else 4)


@tack.kernel
def _boundary_cells(cells, start, entry_slot, starts, types, offsets, connectivity, sources):
    # Each boundary face from its one cell, in the cell's walk (its side 0's).
    for c in cells:
        first = start + cells.index(c) * cells.NUM_FACES
        for f in range(cells.NUM_FACES):
            at_slot = entry_slot[first + f]
            if at_slot >= 0:
                row = _face_row(cells, c, f)
                n = cells.face_num_points(f)
                at = starts[at_slot]
                for j in range(n):
                    connectivity[at + j] = row[j]
                offsets[at_slot] = at
                types[at_slot] = tack.u8(3 if n == 2 else (5 if n == 3 else 9))
                sources[at_slot] = cells.entity_id(c)


def _fresh_boundary(data, fields):
    """``extract_surface``'s default on a mesh whose faces have not been derived:
    the same surface, from a counting sort of the (cell, face) records keeping
    the faces of one cell, without the whole face incidence (rows, sides and
    orientations of every face) nobody asked for -- VTK's external faces
    hashes faces the same way. ``None`` where the full path is needed: faces
    derived already (then they are free), ghost marks (placed through the
    faces), order-2 fields, or a point of more than ``_LARGEST`` faces."""
    from tack.dataset.buckets import _CHUNK, _LARGEST, _largest_counts
    from tack.dataset.topology import (
        _face_bucket_counts,
        _place_faces,
        _sort_face_buckets,
    )

    topology = data.topology
    if (not isinstance(topology, UnstructuredTopology) or topology._faces is not None
            or topology.dimension != 3 or data.ghosts is not None
            or any(isinstance(f.space, H1) and f.space.order != 1
                   for f in data.fields.values())):
        return None
    groups = [g for g in topology.groups() if g.shape.NUM_FACES and g.shape.DIMENSION == 3]
    starts = np.concatenate([[0], np.cumsum([g.count * g.shape.NUM_FACES
                                             for g in groups])]).astype(int)
    total, np_, idt = int(starts[-1]), topology.num_points, topology.id_dtype
    if not total:
        return None
    counts = tack.zeros((np_,), tack.i32)
    for group in groups:
        if group.count:
            _face_bucket_counts(group.view(), counts)
    nchunks = (np_ + _CHUNK - 1) // _CHUNK
    largest = tack.empty((nchunks,), tack.i32)
    _largest_counts(counts, largest, _CHUNK, nchunks, np_)
    if int(largest.numpy().max()) > _LARGEST:
        return None
    bucket_starts = tack.empty((np_ + 1,), idt)
    exclusive_scan(counts, bucket_starts, np_, total=False, close=True)
    counts.fill(0)
    records = tack.empty((total,), tack.vec(5, idt))
    for group, start in zip(groups, starts[:-1]):
        if group.count:
            _place_faces(group.view(), int(start), bucket_starts, counts, records)
    del counts
    _sort_face_buckets(bucket_starts, records, np_)
    flags = tack.empty((total,), tack.i32)
    _singletons(records, flags)
    slots = tack.empty((total,), idt)
    n = exclusive_scan(flags, slots, total)
    entry_slot = tack.empty((total,), idt)
    entry_slot.fill(-1)
    sizes = tack.empty((n,), tack.i32)
    if n:
        _boundary_slots(records, flags, slots, entry_slot, sizes)
    del records, flags, slots
    odt = id_types.for_output(data, 4 * n)
    offsets = tack.empty((n + 1,), odt)
    length = exclusive_scan(sizes, offsets, n, close=True)
    types = tack.empty((n,), tack.u8)
    connectivity = tack.empty((length,), odt)
    cells = tack.empty((n,), idt)
    if n:
        for group, start in zip(groups, starts[:-1]):
            if group.count:
                _boundary_cells(group.view(), int(start), entry_slot, offsets, types, offsets,
                                connectivity, cells)
    surface = UnstructuredTopology(types, offsets, connectivity, num_points=data.num_points)
    return carry(data, surface, points=Same(), cells=Pieces(cells), fields=fields)


@on_inputs_device
def extract_surface(data, faces=None, fields=None):
    """The ``faces`` -- a face ``Set`` (as ``boundary_faces`` returns), a set's name
    in ``data.sets``, or face ids (a field or an array); by default the boundary
    (``boundary_faces``), VTK's external faces -- as a dataset on the same points: a 3D
    mesh's faces as triangles, quads or polygons, each in its side 0's outward
    order; a 2D mesh's edges as line cells; a polyline mesh's ends as vertex
    cells. By default a 2D mesh is its own surface: its cells pass through (its
    hidden ones left out), as VTK's geometry filter passes polygons; its
    boundary edges are ``extract_surface(data, boundary_faces(data))``. Fields on faces become the surface's
    cell fields, and so do cell fields (``Constant``, values on cells), each face
    taking its side 0 cell's value; point fields (``H1``, values on points) stay on
    the same points. A 3D mesh's order-2 ``H1`` fields and geometry stay order 2
    on the surface, the trace of the solid's: each face's corners, edge middles
    and (a quad's) center are the surface cell's, so their values are gathered,
    not evaluated. A 2D mesh's boundary lines take an order-2 field's values at
    the points. The geometry must be ``H1``: an ``L2`` one has no shared points
    for the faces to stand on."""
    from tack.dataset.ghosts import refuse_hidden, visible

    if faces is None:
        data = visible(data)
    else:
        refuse_hidden(data, "extract_surface of chosen faces")
    if not isinstance(data.geometry.space, H1):
        raise TypeError("extract_surface keeps the points, so needs an H1 geometry")
    if faces is None and data.topology.dimension == 2:
        # A surface mesh is its own surface: its polygons pass through, as VTK's
        # geometry filter passes them.
        return carry(data, data.topology, points=Same(), cells=Same(), fields=fields)
    if faces is None:
        fresh = _fresh_boundary(data, fields)
        if fresh is not None:
            return fresh
        from tack.dataset.algorithms.boundary_faces import boundary_faces
        faces = boundary_faces(data)
    ids = _face_ids(data, faces)
    if not getattr(data.topology, "reference_cells", True):
        return _polygon_surface(data, ids, fields)
    faces = data.topology.faces()
    idt = data.id_dtype
    n = ids.shape[0]
    # Up to 4 points a face: past the topology's own bound (12 a cell) for a
    # surface of faces that share no points.
    odt = id_types.for_output(data, 4 * n)
    sizes = tack.empty((n,), tack.i32)
    starts = tack.empty((n,), odt)
    if n:
        _surface_sizes(ids, faces.kinds, sizes)
    length = exclusive_scan(sizes, starts, n) if n else 0
    types = tack.empty((n,), tack.u8)
    offsets = tack.zeros((n + 1,), odt)
    connectivity = tack.empty((length,), odt)
    if n:
        _surface_cells(ids, faces.kinds, faces.rows, starts, types, offsets, connectivity,
                       length, n)
    surface = UnstructuredTopology(types, offsets, connectivity, num_points=data.num_points)
    cells = tack.empty((n,), idt)
    if n:
        _face_cells(ids, faces.sides, cells)
    out = carry(data, surface, points=Same(), cells=Pieces(cells), faces_to_cells=Take(ids),
                fields=fields)
    if data.topology.dimension == 3:
        _keep_order_two(data, out, ids, fields)
    return out


# ── Order 2 ─────────────────────────────────────────────────────────

@tack.kernel
def _point_sources(sources, n):
    for p in range(n):
        sources[p] = p


@tack.kernel
def _edge_sources(rows, solid_rows, sources, base, count):
    # A surface edge's middle is the solid's edge of the same (low, high) points.
    # Edges are numbered in increasing (low, high) order, so bisection finds it.
    for e in range(rows.shape[0]):
        key = rows[e]
        lo = 0
        hi = count
        while lo < hi:
            mid = (lo + hi) // 2
            r = solid_rows[mid]
            if r[0] < key[0] or (r[0] == key[0] and r[1] < key[1]):
                lo = mid + 1
            else:
                hi = mid
        sources[base + e] = base + lo


@tack.kernel
def _center_sources(ids, kinds, face_dofs, cell_dofs, sources):
    # A quad face's center is the surface quad's.
    for c in range(ids.shape[0]):
        f = ids[c]
        if kinds[f] == shapes.QUAD:
            sources[cell_dofs[c]] = face_dofs[f]


def _order_two(field):
    return isinstance(field.space, H1) and field.space.order == 2


def _keep_order_two(data, out, ids, fields):
    """``out``'s fields and geometry that are order 2 in ``data``, made order 2 on the
    surface ``out`` of ``data``'s faces ``ids``: the surface's nodes are the
    solid's, so each value is the solid's at the same node."""
    names = [name for name in selected(data, fields) if _order_two(data.fields[name])]
    curved = _order_two(data.geometry)
    if not (names or curved):
        return
    solid = H1(data, order=2)
    space = H1(out, order=2)
    solid_face_dofs, _, _ = solid._layout()
    _, cell_dofs, size = space._layout()
    surface_edges, solid_edges = out.topology.edges(), data.topology.edges()
    base = data.num_points
    sources = tack.empty((size,), id_types.at_least(data.id_dtype, solid.size))
    if base:
        _point_sources(sources, base)
    if surface_edges.num_edges:
        _edge_sources(surface_edges.rows, solid_edges.rows, sources, base,
                      solid_edges.num_edges)
    if ids.shape[0]:
        _center_sources(ids, data.topology.faces().kinds, solid_face_dofs, cell_dofs, sources)
    for name in names:
        out.fields[name] = Field(space, _take(data.fields[name].values, sources),
                                 data.fields[name].unit)
    if curved:
        out.geometry = Field(space, _take(data.geometry.values, sources), data.unit)


@tack.kernel
def _surface_loop_sizes(ids, face_offsets, sizes):
    for i in range(ids.shape[0]):
        f = ids[i]
        sizes[i] = face_offsets[f + 1] - face_offsets[f]


@tack.kernel
def _surface_loops(ids, face_offsets, face_points, starts, loops):
    for i in range(ids.shape[0]):
        f = ids[i]
        for j in range(face_offsets[f + 1] - face_offsets[f]):
            loops[starts[i] + j] = face_points[face_offsets[f] + j]


def _face_ids(data, faces):
    """``faces`` -- a face set, its name, or face ids -- as a field of the topology's
    ids: a set of this topology's faces, or ids within its face count."""
    if isinstance(faces, str):
        faces = data.sets[faces]
    if isinstance(faces, Set):
        if faces.on != "faces":
            raise ValueError(f"extract_surface takes faces, not a set of {faces.on}")
        if faces.topology is not data.topology:
            raise ValueError("extract_surface takes a set of this dataset's faces, not "
                             "one of another topology")
        return faces.ids
    ids = id_types.as_ids(faces, data.id_dtype)
    id_types.check_bounds(ids, data.topology.faces().num_faces, "face")
    return ids


@tack.kernel
def _facet_types(offsets, types):
    # A polygonal mesh's facets are lines, a polyline mesh's vertices.
    for i in range(types.shape[0]):
        types[i] = tack.u8(shapes.LINE if offsets[i + 1] - offsets[i] == 2 else shapes.VERTEX)


def _polygon_surface(data, ids, fields=None):
    """``extract_surface`` of a polyhedral topology: the set's faces, each a polygon in
    its side 0's outward order, as a ``PolygonalTopology`` on the same points; of a
    polygonal or polyline one, its edges or ends as line or vertex cells."""
    from tack.dataset.polyhedra import PolygonalTopology

    topology = data.topology
    faces = topology.faces()
    idt = data.id_dtype
    n = ids.shape[0]
    sizes = tack.empty((n,), tack.i32)
    offsets = tack.empty((n + 1,), idt)
    if n:
        _surface_loop_sizes(ids, topology.face_offsets, sizes)
    total = exclusive_scan(sizes, offsets, n, close=True)
    loops = tack.empty((total,), idt)
    if n:
        _surface_loops(ids, topology.face_offsets, topology.face_points, offsets, loops)
    if topology.dimension == 3:
        surface = PolygonalTopology(offsets, loops, num_points=data.num_points)
    else:
        types = tack.empty((n,), tack.u8)
        if n:
            _facet_types(offsets, types)
        surface = UnstructuredTopology(types, offsets, loops, num_points=data.num_points)
    cells = tack.empty((n,), idt)
    if n:
        _face_cells(ids, faces.sides, cells)
    return carry(data, surface, points=Same(), cells=Pieces(cells), faces_to_cells=Take(ids),
                 fields=fields)
