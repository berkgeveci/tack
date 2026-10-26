"""``extract_surface``: the faces of set ``name`` as a surface dataset: triangles and quads on the same points, each in its side 0's outward order. Fields on faces become the surface's cell fields, and so do cell fields (``Constant``, values on cells), each face taking its side 0 cell's value; point fields (``H1``, values on points) stay on the same points. A solid's order-2 field or geometry stays order 2 on its surface, gathered by entity: the faces' points, edges and quad centers are the surface's. The geometry must be ``H1``: an ``L2`` one has no shared points for the faces to stand on."""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids as id_types
from tack.dataset import shapes
from tack.dataset.algorithms._common import _close
from tack.dataset.carry import Pieces, Same, Take, _take, carry, selected
from tack.dataset.dataset import Field
from tack.dataset.sets import Set
from tack.dataset.spaces import H1
from tack.dataset.topology import UnstructuredTopology
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
    total = exclusive_scan(sizes, offsets, n) if n else 0
    _close(offsets, n, total)
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
