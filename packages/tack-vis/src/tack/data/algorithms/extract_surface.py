"""``extract_surface``: the faces of set ``name`` as a surface dataset: triangles and quads on the same points, each in its side 0's outward order. Fields on faces become the surface's cell fields, and so do cell fields (``Constant``, values on cells), each face taking its side 0 cell's value; point fields (``H1``, values on points) stay on the same points -- an order-2 field's or geometry's values at the points, the surface being made of linear faces. The geometry must be ``H1``: an ``L2`` one has no shared points for the faces to stand on."""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.data import ids as id_types
from tack.data.algorithms._common import _close
from tack.data.carry import Pieces, Same, Take, carry
from tack.data.spaces import H1
from tack.data.topology import UnstructuredTopology

__all__ = ['extract_surface']


@tack.kernel
def _surface_sizes(ids, kinds, sizes):
    for i in range(ids.shape[0]):
        sizes[i] = 3 if kinds[ids[i]] == 5 else 4


@tack.kernel
def _surface_cells(ids, kinds, rows, starts, types, offsets, connectivity, length):
    for i in range(ids.shape[0]):
        f = ids[i]
        at = starts[i]
        types[i] = tack.u8(kinds[f])
        offsets[i] = at
        row = rows[f]
        for j in range(3 if kinds[f] == 5 else 4):
            connectivity[at + j] = row[j]
        if i == 0:
            offsets[ids.shape[0]] = length


@tack.kernel
def _face_cells(ids, sides, out):
    for i in range(ids.shape[0]):
        out[i] = sides[ids[i]][0]


def extract_surface(data, faces, fields=None):
    """The ``faces`` -- a set's name in ``data.sets``, or face ids (a field or an
    array, as ``boundary_faces`` returns) -- as a surface dataset: triangles and quads on the same
    points, each in its side 0's outward order. Fields on faces become the surface's
    cell fields, and so do cell fields (``Constant``, values on cells), each face
    taking its side 0 cell's value; point fields (``H1``, values on points) stay on
    the same points -- an order-2 field's or geometry's values at the points,
    the surface being made of linear faces. The geometry must be ``H1``: an
    ``L2`` one has no shared points for the faces to stand on."""
    if not isinstance(data.geometry.space, H1):
        raise TypeError("extract_surface keeps the points, so needs an H1 geometry")
    ids = _face_ids(data, faces)
    if not getattr(data.topology, "reference_cells", True):
        return _polygon_surface(data, ids, fields)
    faces = data.topology.faces()
    idt = data.id_dtype
    n = ids.shape[0]
    sizes = tack.field(tack.i32, shape=(n,))
    starts = tack.field(idt, shape=(n,))
    if n:
        _surface_sizes(ids, faces.kinds, sizes)
    length = exclusive_scan(sizes, starts, n) if n else 0
    types = tack.field(tack.u8, shape=(n,))
    offsets = tack.zeros(idt, (n + 1,))
    connectivity = tack.field(idt, shape=(length,))
    if n:
        _surface_cells(ids, faces.kinds, faces.rows, starts, types, offsets, connectivity,
                       length)
    surface = UnstructuredTopology(types, offsets, connectivity, num_points=data.num_points)
    cells = tack.field(idt, shape=(n,))
    if n:
        _face_cells(ids, faces.sides, cells)
    return carry(data, surface, points=Same(), cells=Pieces(cells), faces_to_cells=Take(ids),
                 fields=fields)


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
    """``faces``, a set's name or face ids, as a field of the topology's ids."""
    if isinstance(faces, str):
        return data.sets[faces]
    return id_types.as_ids(faces, data.id_dtype)


def _polygon_surface(data, ids, fields=None):
    """``extract_surface`` of a polyhedral topology: the set's faces, each a polygon in
    its side 0's outward order, as a ``PolygonalTopology`` on the same points."""
    from tack.data.polyhedra import PolygonalTopology

    topology = data.topology
    faces = topology.faces()
    idt = data.id_dtype
    n = ids.shape[0]
    sizes = tack.field(tack.i32, shape=(n,))
    offsets = tack.field(idt, shape=(n + 1,))
    if n:
        _surface_loop_sizes(ids, topology.face_offsets, sizes)
    total = exclusive_scan(sizes, offsets, n) if n else 0
    _close(offsets, n, total)
    loops = tack.field(idt, shape=(total,))
    if n:
        _surface_loops(ids, topology.face_offsets, topology.face_points, offsets, loops)
    surface = PolygonalTopology(offsets, loops, num_points=data.num_points)
    cells = tack.field(idt, shape=(n,))
    if n:
        _face_cells(ids, faces.sides, cells)
    return carry(data, surface, points=Same(), cells=Pieces(cells), faces_to_cells=Take(ids),
                 fields=fields)
