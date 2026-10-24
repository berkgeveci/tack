"""``split``: a structured grid cut by index ranges into a ``PartitionedDataSet`` of structured grids, each carrying its block of the points, cells, fields and sets."""

import numpy as np

import tack
from tack.arrays import CartesianProduct, UniformCoordinates, dtype_of
from tack.dataset.carry import Take, _take, carry
from tack.dataset.dataset import Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.partitioned import PartitionedDataSet
from tack.dataset.spaces import H1
from tack.dataset.topology import StructuredTopology
from tack.runtime.dispatch import on_inputs_device

__all__ = ['split']


@tack.kernel
def _block_ids(out, nx, ny, sx, sy, ox, oy, oz):
    # The ids, in a grid of sx * sy * ... entities, of the block of nx * ny * ...
    # starting at (ox, oy, oz), x fastest.
    for n in range(out.shape[0]):
        i = n % nx
        j = (n // nx) % ny
        k = n // (nx * ny)
        out[n] = (oz + k) * (sx * sy) + (oy + j) * sx + (ox + i)


@tack.kernel
def _mark_outside(marks, nx, ny, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z):
    # Marks the entities of a block of nx * ny * ... outside the box lo..hi (local
    # indices, inclusive) as duplicates: DUPLICATECELL and DUPLICATEPOINT are 1.
    for n in range(marks.shape[0]):
        i = n % nx
        j = (n // nx) % ny
        k = n // (nx * ny)
        if i < lo_x or i > hi_x or j < lo_y or j > hi_y or k < lo_z or k > hi_z:
            marks[n] = marks[n] | 1


def _ids(dims, start, block, dtype):
    """The flat ids of ``block`` entities from ``start`` in a grid of ``dims``."""
    dims, block = (tuple(v) + (1,) * (3 - len(v)) for v in (dims, block))
    start = tuple(start) + (0,) * (3 - len(start))
    out = tack.empty((int(np.prod(block)),), dtype)
    _block_ids(out, block[0], block[1], dims[0], dims[1], *start)
    return out


def _counts(pieces, cells):
    """Pieces per axis: the tuple given, or ``pieces`` factored over the axes, each
    prime factor, largest first, cutting the axis whose pieces have most cells."""
    if not isinstance(pieces, (int, np.integer)):
        counts = tuple(int(c) for c in pieces)
        if len(counts) != len(cells):
            raise ValueError(f"split of a grid of {len(cells)} axes takes {len(cells)} "
                             f"counts, not {counts}")
        if min(counts) < 1 or any(c > n for c, n in zip(counts, cells)):
            raise ValueError(f"cannot cut {cells} cells into {counts} pieces: each "
                             "piece needs at least one cell along every axis")
        return counts
    pieces = int(pieces)
    if pieces < 1:
        raise ValueError(f"split needs at least 1 piece, not {pieces}")
    factors, n, p = [], pieces, 2
    while n > 1:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1
    counts = [1] * len(cells)
    for f in sorted(factors, reverse=True):
        fits = [a for a in range(len(cells)) if counts[a] * f <= cells[a]]
        if not fits:
            raise ValueError(f"cannot cut {cells} cells into {pieces} pieces of at least "
                             "one cell each")
        a = max(fits, key=lambda a: cells[a] / counts[a])
        counts[a] *= f
    return tuple(counts)


def _block_geometry(coordinates, axes, start, point_dims, device):
    """The implicit coordinates of a block whose first point is ``start`` along the
    topology's axes, which are the coordinates' axes of more than one point."""
    full = [1, 1, 1]
    offset = [0, 0, 0]
    for axis, s, d in zip(axes, start, point_dims):
        full[axis], offset[axis] = d, s
    if isinstance(coordinates, UniformCoordinates):
        spacing = np.asarray(coordinates._spacing, float)
        origin = np.asarray(coordinates._origin, float) + coordinates._matrix @ (
            np.asarray(offset) * spacing)
        direction = None if not coordinates.ORIENTED else coordinates._matrix
        return UniformCoordinates(full, origin, spacing, direction,
                                  dtype=dtype_of(coordinates), device=device)
    lines = [axis.numpy()[o:o + n] for axis, o, n in
             zip((coordinates.x, coordinates.y, coordinates.z), offset, full)]
    return CartesianProduct(*lines, dtype=dtype_of(coordinates), device=device)


@on_inputs_device
def split(data, pieces, ghost_layers=0):
    """``data``, a structured grid, cut by index ranges into a ``PartitionedDataSet``
    of ``pieces`` structured grids, x fastest.

    ``pieces`` is a count, factored over the axes (each prime factor, largest
    first, cuts the axis whose pieces have most cells), or a tuple of counts,
    one per axis of the topology. Each axis's cells are shared out as evenly
    as they go, every piece getting at least one; neighbouring pieces share
    the points between them. A piece keeps the geometry's kind -- uniform
    coordinates with their origin moved, rectilinear axes cut, explicit
    points taken -- and carries its block of every point, cell and L2 field
    and every set, as filters carry them; fields on faces or edges are left
    behind.

    ``ghost_layers`` adds that many layers of each neighbour's cells around a
    piece, as far as the grid goes, marked duplicates (``DUPLICATECELL`` in
    ``ghosts``), as VTK's ``vtkGhostCellsGenerator`` adds them. Filters need
    one to give each piece's share of what they make on the faces and edges
    pieces share. Every point has one owner: a point of several pieces' own
    cells is the first such piece's, x fastest, and a duplicate
    (``DUPLICATEPOINT``) in the others, as is every point of the ghost
    layers. The input's own ghost marks come along. An unstructured mesh is
    partitioned by VTK (``vtkRedistributeDataSetFilter``) or by its host
    application."""
    topology = data.topology
    if not isinstance(topology, StructuredTopology):
        raise TypeError("split cuts structured grids; partition an unstructured mesh "
                        "with VTK (vtkRedistributeDataSetFilter) or the host application")
    layers = int(ghost_layers)
    if layers < 0 or layers != ghost_layers:
        raise ValueError(f"ghost_layers is a count of layers, not {ghost_layers!r}")
    dims = topology.point_dims
    cells = tuple(d - 1 for d in dims)
    counts = _counts(pieces, cells)
    coordinates = data.geometry.values
    implicit = (isinstance(data.geometry.space, H1)
                and isinstance(coordinates, (UniformCoordinates, CartesianProduct)))
    if implicit:
        # The topology's axes are the coordinates' axes of more than one point.
        axes = [a for a, d in enumerate(coordinates.dims) if d != 1]
    device = data.device
    bounds = [[k * n // c for k in range(c + 1)] for n, c in zip(cells, counts)]
    out = []
    for index in np.ndindex(*reversed(counts)):
        index = tuple(reversed(index))                # x fastest
        own = [(b[i], b[i + 1]) for b, i in zip(bounds, index)]        # its cells
        # With its ghost layers, as far as the grid goes.
        start = tuple(max(lo - layers, 0) for lo, _ in own)
        stop = tuple(min(hi + layers, n) for (_, hi), n in zip(own, cells))
        block_cells = tuple(b - a for a, b in zip(start, stop))
        block_points = tuple(n + 1 for n in block_cells)
        piece = StructuredTopology(block_points, id_dtype=topology.id_dtype, device=device,
                                   offset=[o + s for o, s in zip(topology.offset, start)])
        geometry = None
        if implicit:
            geometry = Field(H1(piece), _block_geometry(coordinates, axes, start,
                                                        block_points, device))
        point_ids = _ids(dims, start, block_points, topology.id_dtype)
        cell_ids = _ids(cells, start, block_cells, topology.id_dtype)
        part = carry(data, piece, points=Take(point_ids), cells=Take(cell_ids),
                     geometry=geometry)
        part.ghosts = _block_ghosts(data.ghosts, point_ids, cell_ids, block_points,
                                    block_cells, start, own)
        out.append(part)
    return PartitionedDataSet(out)


def _block_ghosts(ghosts, point_ids, cell_ids, block_points, block_cells, start, own):
    """A block's ghost marks: the input's, taken, with the cells outside its own
    marked duplicates, and the points outside its own -- the points of its own
    cells but those of a block before it along an axis -- too, so each point
    has one owner."""
    def marked(marks, ids, block, lo, hi):
        out = _take(marks, ids) if marks is not None else tack.zeros((ids.shape[0],), tack.u8)
        block = tuple(block) + (1,) * (3 - len(block))
        lo, hi = (tuple(v) + (0,) * (3 - len(v)) for v in (lo, hi))
        if any(a > 0 for a in lo) or any(b < n - 1 for b, n in zip(hi, block)):
            _mark_outside(out, block[0], block[1], *lo, *hi)
        return out

    # Its own cells, and its own points: those of its cells, less the first layer
    # where a block comes before it.
    cell_lo = tuple(a - s for (a, _), s in zip(own, start))
    cell_hi = tuple(b - 1 - s for (_, b), s in zip(own, start))
    point_lo = tuple(a - s + (1 if a > 0 else 0) for (a, _), s in zip(own, start))
    point_hi = tuple(b - s for (_, b), s in zip(own, start))
    points = marked(None if ghosts is None else ghosts.points, point_ids, block_points,
                    point_lo, point_hi)
    has_ghost_cells = any(c != b - a for c, (a, b) in zip(block_cells, own))
    cells = None
    if (ghosts is not None and ghosts.cells is not None) or has_ghost_cells:
        cells = marked(None if ghosts is None else ghosts.cells, cell_ids, block_cells,
                       cell_lo, cell_hi)
    return Ghosts(cells=cells, points=points)
