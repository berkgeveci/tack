"""``as_polyhedra``: a shape-based dataset as a polyhedral one, every cell a polyhedron."""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids, shapes
from tack.dataset.algorithms._common import _close
from tack.dataset.carry import Same, carry
from tack.dataset.polyhedra import PolyhedralTopology
from tack.runtime.dispatch import on_inputs_device

__all__ = ['as_polyhedra']


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


@tack.kernel
def _same_way(cells, count):
    # A cell going round its face the way the face's row does (side 0's outward
    # order) while on side 1, or the other way while on side 0: it and the
    # face's other cell go round it alike, one of them inside out.
    for c in cells:
        for f in range(cells.NUM_FACES):
            if cells.face_orientation(c, f) % 2 != cells.face_side(c, f):
                tack.atomic_add(count, 0, 1)


@on_inputs_device
def as_polyhedra(data, fields=None):
    """A shape-based dataset as a polyhedral one, every cell a polyhedron.

    Its derived faces already hold every face once, in side 0's outward order
    (a voxel's pixel faces as quads), with the side of each cell's reference, so
    they become the polyhedral faces as they stand: face ids, and so fields on
    faces, are unchanged. Edges come out numbered as before. Cells must be 3D,
    and neighbours wound alike: an inside-out cell beside an upright one is
    refused, since their shared face would have them on one side.
    ``Constant`` and ``H1`` order-1 fields, values on points, cells, faces and
    edges, and sets carry over; fields needing a reference element are dropped.
    """
    from tack.dataset.spaces import H1

    if not isinstance(data.geometry.space, H1) or data.geometry.space.order != 1:
        raise TypeError("as_polyhedra needs an order-1 H1 geometry: positions per point")
    topology = data.topology
    for group in topology.groups():
        if group.count and group.shape.DIMENSION != 3:
            raise ValueError(f"{group.shape.__name__} cells are not 3D: a polyhedral "
                             "topology is of 3D cells (as_polygons takes 2D ones)")
    faces = topology.faces()
    same_way = tack.zeros((1,), tack.i32)
    for group in topology.groups():
        if group.count:
            _same_way(data.domain_view("cells", group), same_way)
    if same_way[0]:
        raise ValueError(f"{same_way[0]} faces are wound the same way by both their cells "
                         "(an inside-out cell beside an upright one, as cell_geometry's "
                         "negative measures show); a polyhedral topology needs each face's "
                         "cells on opposite sides of it")
    nf = faces.num_faces
    idt = topology.id_dtype
    # The faces' points: 4 a face, past 12 a cell when faces are not shared (a
    # hexahedron has 24), and the polyhedral topology holds twice that.
    fdt = ids.at_least(idt, 8 * nf)
    sizes = tack.empty((nf,), tack.i32)
    starts = tack.empty((nf + 1,), fdt)
    if nf:
        _face_sizes(faces.kinds, sizes)
    total = exclusive_scan(sizes, starts, nf) if nf else 0
    _close(starts, nf, total)
    face_points = tack.empty((total,), fdt)
    if nf:
        _face_rows(faces.rows, starts, sizes, face_points)

    n = data.num_cells
    counts = tack.zeros((n,), tack.i32)
    for group in topology.groups():
        if group.count:
            _cell_face_counts(group.view(), counts)
    cell_offsets = tack.empty((n + 1,), idt)
    entries = exclusive_scan(counts, cell_offsets, n) if n else 0
    _close(cell_offsets, n, entries)
    cell_faces = tack.empty((entries,), idt)
    cell_face_sides = tack.empty((entries,), tack.u8)
    for group in topology.groups():
        if group.count:
            _cell_face_entries(data.domain_view("cells", group), cell_offsets, cell_faces,
                               cell_face_sides)
    out = PolyhedralTopology(starts, face_points, cell_offsets, cell_faces, cell_face_sides,
                             num_points=data.num_points)

    # The same points, cells and faces (in the shape path's numbering), so the
    # sets still apply; edges too, both paths numbering them by (low, high)
    # point ids.
    return carry(data, out, points=Same(), cells=Same(), faces=Same(), edges=Same(),
                 fields=fields)
