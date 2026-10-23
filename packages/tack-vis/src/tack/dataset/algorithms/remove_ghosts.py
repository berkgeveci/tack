"""``remove_ghosts``: a piece without the cells another piece owns or that are hidden, as VTK's vtkRemoveGhosts, for the last step before rendering, writing one file, or code that does not know ghost marks."""

import numpy as np

import tack
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.dataset.algorithms._extract import _keep_cells
from tack.dataset.algorithms.split import _block_geometry, _ids
from tack.dataset.carry import Take, carry
from tack.dataset.dataset import Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.spaces import H1
from tack.dataset.topology import StructuredTopology
from tack.runtime.dispatch import on_inputs_device

__all__ = ['remove_ghosts']


@tack.kernel
def _unmarked(marks, bits, keep):
    for c in range(keep.shape[0]):
        keep[c] = 1 if (marks[c] & bits) == 0 else 0


@on_inputs_device
def remove_ghosts(data, fields=None):
    """``data`` without its ghost cells -- the cells another piece owns
    (``DUPLICATECELL``) and the hidden ones (``HIDDENCELL``) -- and the points no
    remaining cell uses, as VTK's vtkRemoveGhosts: the last step before
    rendering, writing one file, or handing a piece to code that does not know
    ghost marks. Filters keep the ghost layers' output, which the next filter
    near a piece's edge needs, until this.

    The points kept keep their marks: a point a piece shares with another is
    still that one's, so reductions still count it once. A point this piece
    owns stays even when only ghost cells used it (a point a filter made on an
    edge goes to the piece owning the edge's smallest point, whose cells there
    may all be ghosts), so every point keeps its one owner. A structured grid
    whose remaining cells are a block stays a structured grid; otherwise the
    cells are kept as ``extract_cells`` keeps them. ``data`` itself comes back
    when it has no ghost cells."""
    if data._ghosts_lost:
        raise ValueError("this dataset is a filter's output that lost its input's ghost "
                         "marks, so which of its cells are ghosts is not known")
    ghosts = data.ghosts
    if ghosts is None or ghosts.cells is None:
        return data
    keep = tack.empty((data.num_cells,), tack.i32)
    if data.num_cells:
        _unmarked(ghosts.cells, Ghosts.SKIPPED_CELLS, keep)
    owned = None
    if ghosts.points is not None:
        owned = tack.empty((data.num_points,), tack.i32)
        if data.num_points:
            _unmarked(ghosts.points, Ghosts.DUPLICATEPOINT, owned)
    if isinstance(data.topology, StructuredTopology):
        block = _kept_block(data, keep, owned)
        if block is not None:
            return block
    return _keep_cells(data, keep, fields, keep_points=owned)


def _kept_block(data, keep, owned):
    """The kept cells of a structured grid as a structured grid, if they are a
    block holding every point flagged in ``owned``; else ``None``."""
    topology = data.topology
    cells = tuple(d - 1 for d in topology.point_dims)
    flags = keep.numpy().reshape(cells[::-1]).astype(bool)
    if not flags.any():
        return None
    lo, hi = [], []
    for axis in range(len(cells)):
        along = flags.any(axis=tuple(a for a in range(len(cells)) if a != len(cells) - 1 - axis))
        where = np.flatnonzero(along)
        lo.append(int(where[0]))
        hi.append(int(where[-1]) + 1)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))[::-1]
    if not flags[box].all() or flags.sum() != flags[box].size:
        return None
    if owned is not None:
        points = owned.numpy().reshape(topology.point_dims[::-1]).astype(bool)
        inside = tuple(slice(a, b + 1) for a, b in zip(lo, hi))[::-1]
        if points.sum() != points[inside].sum():
            return None
    start = tuple(lo)
    block_cells = tuple(b - a for a, b in zip(lo, hi))
    if block_cells == cells:
        return data
    block_points = tuple(n + 1 for n in block_cells)
    piece = StructuredTopology(block_points, id_dtype=topology.id_dtype, device=data.device)
    coordinates = data.geometry.values
    geometry = None
    if isinstance(data.geometry.space, H1) and isinstance(
            coordinates, (UniformCoordinates, CartesianProduct)):
        axes = [a for a, d in enumerate(coordinates.dims) if d != 1]
        geometry = Field(H1(piece), _block_geometry(coordinates, axes, start, block_points,
                                                    data.device))
    point_ids = _ids(topology.point_dims, start, block_points, topology.id_dtype)
    cell_ids = _ids(cells, start, block_cells, topology.id_dtype)
    return carry(data, piece, points=Take(point_ids), cells=Take(cell_ids), geometry=geometry)
