"""Splitting cells into pieces by Viskores' tables, behind ``tetrahedralize``
and ``triangulate``."""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids, shapes
from tack.dataset.carry import Pieces, Same, carry
from tack.dataset.topology import UnstructuredTopology, _as_field

_HEX_TETS = [[0, 1, 3, 4], [1, 4, 5, 6], [1, 4, 6, 3], [1, 3, 6, 2], [3, 6, 7, 4]]


_VOXEL_ORDER = [0, 1, 3, 2, 4, 5, 7, 6]


_PIECES = {
    "tetrahedra": {
        int(shapes.TETRA): [[0, 1, 2, 3]],
        int(shapes.HEXAHEDRON): _HEX_TETS,
        int(shapes.VOXEL): [[_VOXEL_ORDER[k] for k in tet] for tet in _HEX_TETS],
        # Viskores' (3, 4, 5, 2) is inside out: the same tetrahedron, turned.
        int(shapes.WEDGE): [[0, 1, 2, 4], [3, 5, 4, 2], [0, 2, 3, 4]],
        int(shapes.PYRAMID): [[0, 1, 2, 4], [0, 2, 3, 4]],
    },
    "triangles": {
        int(shapes.TRIANGLE): [[0, 1, 2]],
        int(shapes.QUAD): [[0, 1, 2], [0, 2, 3]],
        int(shapes.PIXEL): [[0, 1, 3], [0, 3, 2]],
    },
}


@tack.kernel
def _piece_counts(cells, pieces, counts):
    for c in cells:
        counts[cells.entity_id(c)] = pieces


@tack.kernel
def _split_cells(cells, table, pieces, width, starts, connectivity, sources):
    for c in cells:
        e = cells.entity_id(c)
        at = starts[e]
        for t in range(pieces):
            for v in range(width):
                connectivity[(at + t) * width + v] = cells.point_id(c, table[t * width + v])
            sources[at + t] = e


@tack.kernel
def _fan_counts(loop_offsets, counts):
    for c in range(counts.shape[0]):
        counts[c] = max(loop_offsets[c + 1] - loop_offsets[c] - 2, 0)


@tack.kernel
def _fans(loop_offsets, loop_points, starts, connectivity, sources):
    for c in range(loop_offsets.shape[0] - 1):
        first = loop_offsets[c]
        at = starts[c]
        for k in range(loop_offsets[c + 1] - first - 2):
            connectivity[3 * (at + k)] = loop_points[first]
            connectivity[3 * (at + k) + 1] = loop_points[first + k + 1]
            connectivity[3 * (at + k) + 2] = loop_points[first + k + 2]
            sources[at + k] = c


@tack.kernel
def _uniform_cells(types, offsets, shape, width, n):
    for i in range(n):
        types[i] = tack.u8(shape)
        offsets[i] = i * width
        if i == 0:
            offsets[n] = n * width


def _pieces_topology(data, connectivity, sources, count, shape, width, fields):
    types = tack.field(tack.u8, shape=(count,))
    offsets = tack.zeros(connectivity.dtype, (count + 1,))
    if count:
        _uniform_cells(types, offsets, int(shape), width, int(types.shape[0]))
    out = UnstructuredTopology(types, offsets, connectivity, num_points=data.num_points)
    return carry(data, out, points=Same(), cells=Pieces(sources), fields=fields)


def _split(data, kind, fields):
    tables = _PIECES[kind]
    width, shape = (4, shapes.TETRA) if kind == "tetrahedra" else (3, shapes.TRIANGLE)
    n = data.num_cells
    groups = [g for g in data.topology.groups() if g.count and int(g.shape.ID) in tables]
    counts = tack.zeros(tack.i32, (n,))
    for group in groups:
        _piece_counts(group.view(), len(tables[int(group.shape.ID)]), counts)
    most = max((len(tables[int(g.shape.ID)]) for g in groups), default=0)
    odt = ids.for_output(data, most * width * n)
    starts = tack.field(odt, shape=(n,))
    count = exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.field(odt, shape=(count * width,))
    sources = tack.field(data.id_dtype, shape=(count,))
    for group in groups:
        table = np.asarray(tables[int(group.shape.ID)], np.int32)
        _split_cells(group.view(), _as_field(table.reshape(-1), tack.i32), len(table), width,
                     starts, connectivity, sources)
    return _pieces_topology(data, connectivity, sources, count, shape, width, fields)
