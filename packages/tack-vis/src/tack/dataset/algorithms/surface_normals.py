"""``surface_normals``: a surface's polygons wound consistently and its normals at the points and cells -- VTK's vtkPolyDataNormals without splitting."""

import numpy as np

import tack
from tack import arrays
from tack.dataset import shapes
from tack.dataset.algorithms._common import _table, _vectors
from tack.dataset.algorithms._links import _average_links
from tack.dataset.carry import Same, _take, carry
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values
from tack.dataset.topology import UnstructuredTopology, corner_layout
from tack.runtime.dispatch import on_inputs_device

__all__ = ['surface_normals']

#: A 2D shape's corners in order round it, and in the other order from its first.
_ROUND = {shapes.Triangle: (0, 1, 2), shapes.Quad: (0, 1, 2, 3), shapes.Pixel: (0, 1, 3, 2)}
_REVERSED = {shapes.Triangle: (0, 2, 1), shapes.Quad: (0, 3, 2, 1), shapes.Pixel: (0, 2, 1, 3)}


@tack.kernel
def _edge_relations(cells, sides, pairs, relations):
    # Each two-sided edge's cells, and whether they wind it alike (1: one must
    # turn), from its side 1: side 0 walks the edge's row forwards, and a
    # consistent neighbour walks it backwards (reflected).
    for c in cells:
        for k in range(cells.num_faces(c)):
            if cells.face_side(c, k) == 1:
                f = cells.face_id(c, k)
                pairs[f] = [sides[f][0], cells.entity_id(c)]
                relations[f] = 1 - cells.face_orientation(c, k) % 2


@tack.kernel
def _hook(pairs, relations, parents, changed):
    # parents[c] is 2 * parent + parity (whether c winds against its parent).
    # Each edge joins its cells' trees, the larger root under the smaller with
    # the parity the edge implies; an atomic minimum, so a root's parent and
    # parity are written together and concurrent hooks agree on one of them.
    for f in range(pairs.shape[0]):
        a = pairs[f][0]
        if a >= 0:
            b = pairs[f][1]
            va = parents[a]
            vb = parents[b]
            ra = va // 2
            rb = vb // 2
            if ra != rb:
                parity = (va % 2 + vb % 2 + relations[f]) % 2
                tack.atomic_min(parents, max(ra, rb), 2 * min(ra, rb) + parity)
                tack.atomic_max(changed, 0, 1)


@tack.kernel
def _jump(parents):
    # Every cell straight to its root, its parity summed along the way.
    for c in range(parents.shape[0]):
        v = parents[c]
        r = v // 2
        parity = v % 2
        while parents[r] // 2 != r:
            v = parents[r]
            parity = (parity + v % 2) % 2
            r = v // 2
        parents[c] = 2 * r + parity


@tack.kernel
def _contradictions(pairs, relations, parents, count):
    for f in range(pairs.shape[0]):
        a = pairs[f][0]
        if a >= 0:
            b = pairs[f][1]
            if (parents[a] % 2 + parents[b] % 2) % 2 != relations[f]:
                tack.atomic_add(count, 0, 1)


@tack.kernel
def _rewound(cells, parents, offsets, order, connectivity):
    # A cell that winds against its region's first cell, its corners the other
    # way round from its first.
    for c in cells:
        e = cells.entity_id(c)
        if parents[e] % 2 == 1:
            at = offsets[e]
            for j in range(cells.NUM_POINTS):
                connectivity[at + j] = cells.point_id(c, order[j])


@tack.kernel
def _shape_normals(cells, order, out):
    # Newell's normal round the corners, of unit length (zero for no area).
    for c in cells:
        size = order.shape[0]
        origin = cells.corner_position(c, order[0])
        n = tack.vec3(0.0, 0.0, 0.0)
        for k in range(size):
            n += (cells.corner_position(c, order[k]) - origin).cross(
                cells.corner_position(c, order[k + 1 if k + 1 < size else 0]) - origin)
        length = n.norm()
        out[cells.entity_id(c)] = n / length if length > 0.0 else n


@tack.kernel
def _polygon_normals(cells, out):
    for c in cells:
        origin = cells.side_position(c, 0, 0)
        n = tack.vec3(0.0, 0.0, 0.0)
        for k in range(cells.num_faces(c)):
            n += (cells.side_position(c, k, 0) - origin).cross(
                cells.side_position(c, k, 1) - origin)
        length = n.norm()
        out[cells.entity_id(c)] = n / length if length > 0.0 else n


@tack.kernel
def _corner_normals(cells, normals, start, out):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            out[first + j] = normals[cells.entity_id(c)]


@tack.kernel
def _entry_cells(offsets, owners):
    for c in range(offsets.shape[0] - 1):
        for k in range(offsets[c], offsets[c + 1]):
            owners[k] = c


@tack.kernel
def _unit(values):
    for i in range(values.shape[0]):
        length = values[i].norm()
        if length > 0.0:
            values[i] = values[i] / length


def _consistent(data):
    """``data`` with every connected region's polygons wound as its first cell is:
    union-find over the edges with each cell's parity to its root."""
    topology = data.topology
    faces = topology.faces()
    nf, n = faces.num_faces, data.num_cells
    pdt = tack.i64 if 2 * n + 1 > 2 ** 31 - 1 else tack.i32
    pairs = tack.empty((nf,), tack.vec(2, pdt))
    pairs.fill(-1)
    relations = tack.zeros((nf,), tack.i32)
    for group in data.launch_groups("cells", []):
        if group.count:
            _edge_relations(data.domain_view("cells", group), faces.sides, pairs, relations)
    parents = tack.array(2 * np.arange(n, dtype=pdt.numpy_dtype), pdt)
    changed = tack.zeros((1,), tack.i32)
    while nf:
        changed.fill(0)
        _hook(pairs, relations, parents, changed)
        _jump(parents)
        if not int(changed.numpy()[0]):
            break
    bad = tack.zeros((1,), tack.i32)
    if nf:
        _contradictions(pairs, relations, parents, bad)
    if bad[0]:
        raise ValueError(f"the surface is not orientable: {bad[0]} edges cannot be wound "
                         "consistently (a Moebius strip, or polygons folded through an edge)")
    offsets = arrays.materialize(topology.offsets)
    connectivity = tack.empty((arrays.size_of(topology.connectivity),),
                              arrays.dtype_of(topology.connectivity))
    tack.copyto(connectivity, arrays.materialize(topology.connectivity))
    for group in data.launch_groups("cells", []):
        if group.count and group.shape in _REVERSED:
            _rewound(data.domain_view("cells", group), parents, offsets,
                     _table(("reversed", group.shape), _REVERSED[group.shape]), connectivity)
    out = UnstructuredTopology(topology.types, topology.offsets, connectivity,
                               num_points=data.num_points, id_dtype=topology.id_dtype)
    return carry(data, out, points=Same(), cells=Same())


@on_inputs_device
def surface_normals(data, consistent=True):
    """A surface -- triangles, quads, pixels or polygons -- with its normals: the
    unit normal of each cell (Newell's, round its corners) as the cell field
    ``cell_normals``, and at each point the cells' around it summed and made unit
    as the point field ``normals`` (zero at a point of no cells), as VTK's
    ``vtkPolyDataNormals`` computes them without splitting.

    With ``consistent``, the polygons are first wound alike: each connected
    region's as its first (lowest-numbered) cell winds, by union-find over the
    edges on the device, the cells that wind against it turned round (their
    corners reversed from the first), so their normals agree -- VTK's
    consistency, which is on by default there too. A surface no winding makes
    consistent (a Moebius strip) is refused, as is an edge of three or more
    polygons (the mesh's faces refuse it). A polygonal topology is consistent
    already (it refuses to be made otherwise), so it is left as it is. The
    result keeps every field; the normals are dimensionless."""
    from tack.dataset.ghosts import visible

    data = visible(data)
    if data.num_cells and data.topology.dimension != 2:
        raise ValueError(f"surface_normals takes 2D cells (a surface), not a mesh of "
                         f"dimension {data.topology.dimension}: extract_surface a solid "
                         "first")
    if consistent and isinstance(data.topology, UnstructuredTopology) and data.num_cells:
        data = _consistent(data)
    else:
        data = carry(data, data.topology, points=Same(), cells=Same())    # not the input
    n = data.num_cells
    cell_normals = _vectors(n, data.dtype)
    if not getattr(data.topology, "reference_cells", True):
        data.topology.faces()                      # a polygon's walk
    for group in data.launch_groups("cells", []):
        if not group.count:
            continue
        view = data.domain_view("cells", group)
        if group.shape in _ROUND:
            _shape_normals(view, _table(("round", group.shape), _ROUND[group.shape]),
                           cell_normals)
        else:
            _polygon_normals(view, cell_normals)
    points = tack.empty((data.num_points,), tack.vec(3, data.dtype))
    points.fill(0)
    # Each cell's normal at each of its corners, as point_links lays them out
    # (by cell and corner on shapes, by cell_points' entries elsewhere).
    if getattr(data.topology, "reference_cells", True):
        start_of, total = corner_layout(data.topology)
        corners = _vectors(total, data.dtype)
        for group in data.launch_groups("cells", []):
            if group.count:
                _corner_normals(group.view(), cell_normals, start_of[group.root], corners)
    else:
        point_offsets, point_ids = data.topology.cell_points()
        total = point_ids.shape[0]
        owners = tack.empty((total,), data.id_dtype)
        if total:
            _entry_cells(point_offsets, owners)
        corners = _take(cell_normals, owners)
    if total:
        offsets, links = data.topology.point_links()
        _average_links(offsets, links, corners, points)
        _unit(points)
    data.fields["normals"] = Field(Values(data, "points"), points, "dimensionless")
    data.fields["cell_normals"] = Field(Values(data, "cells"), cell_normals, "dimensionless")
    return data
