"""``as_polygons``: a shape-based dataset of 2D cells -- triangles, quads, pixels -- as a polygonal one: each cell's points in order become its loop. Fields carry as ``as_polyhedra``'s do, but values on faces and edges, whose numbering the polygonal topology derives afresh, are dropped."""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset import shapes
from tack.dataset.carry import Same, carry
from tack.dataset.polyhedra import PolygonalTopology
from tack.runtime.dispatch import on_inputs_device

__all__ = ['as_polygons']


@tack.kernel
def _polygon_sizes(cells, sizes):
    for c in cells:
        sizes[cells.entity_id(c)] = cells.NUM_POINTS


@tack.kernel
def _polygon_loops(cells, offsets, loops):
    for c in cells:
        at = offsets[cells.entity_id(c)]
        for k in range(cells.NUM_POINTS):
            # A pixel's points are x-fastest: around it, 0 1 3 2.
            kk = k ^ (k >> 1) if cells.ID == shapes.PIXEL else k
            loops[at + k] = cells.point_id(c, kk)


@on_inputs_device
def as_polygons(data, fields=None):
    """A shape-based dataset of 2D cells -- triangles, quads, pixels -- as a polygonal
    one: each cell's points in order become its loop. Fields carry as
    ``as_polyhedra``'s do, but values on faces and edges, whose numbering the
    polygonal topology derives afresh, are dropped. A polygonal dataset comes
    back as it is (a new dataset on the same topology)."""

    topology = data.topology
    if isinstance(topology, PolygonalTopology):
        return carry(data, topology, points=Same(), cells=Same(), fields=fields)
    for group in topology.groups():
        if group.count and group.shape.DIMENSION != 2:
            raise ValueError(f"{group.shape.__name__} cells are not 2D")
    n = data.num_cells
    idt = topology.id_dtype
    sizes = tack.zeros((n,), tack.i32)
    for group in topology.groups():
        if group.count:
            _polygon_sizes(group.view(), sizes)
    offsets = tack.empty((n + 1,), idt)
    total = exclusive_scan(sizes, offsets, n, close=True)
    loops = tack.empty((total,), idt)
    for group in topology.groups():
        if group.count:
            _polygon_loops(group.view(), offsets, loops)
    out = PolygonalTopology(offsets, loops, num_points=data.num_points)
    # The same points and cells; edge numbering is derived afresh.
    return carry(data, out, points=Same(), cells=Same(), fields=fields)
