"""``extract_extent``: a box of a structured grid's points, every ``rate``-th, as a structured grid of the same kind -- VTK's vtkExtractGrid (VOI)."""

import numpy as np

import tack
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.dataset.algorithms.split import _block_geometry
from tack.dataset.carry import Take, carry
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1
from tack.dataset.topology import StructuredTopology
from tack.runtime.dispatch import on_inputs_device

__all__ = ['extract_extent']


@tack.kernel
def _strided_ids(out, nx, ny, sx, sy, ox, oy, oz, rx, ry, rz):
    # The ids, in a grid of sx * sy * ... entities, of the nx * ny * ... ones at
    # (ox + rx * i, oy + ry * j, oz + rz * k), x fastest.
    for n in range(out.shape[0]):
        i = n % nx
        j = (n // nx) % ny
        k = n // (nx * ny)
        out[n] = (oz + rz * k) * (sx * sy) + (oy + ry * j) * sx + (ox + rx * i)


def _ids(dims, start, count, rate, dtype):
    dims, count = (tuple(v) + (1,) * (3 - len(v)) for v in (dims, count))
    start, rate = tuple(start) + (0,) * (3 - len(start)), tuple(rate) + (1,) * (3 - len(rate))
    out = tack.empty((int(np.prod(count)),), dtype)
    if out.shape[0]:
        _strided_ids(out, count[0], count[1], dims[0], dims[1], *start, *rate)
    return out


def _ranges(extent, n):
    """``extent`` as ``n`` (low, high) pairs: pairs, or VTK's flat (i0, i1, j0, ...)."""
    values = np.asarray(extent, dtype=np.int64).reshape(-1)
    if values.size != 2 * n:
        raise ValueError(f"an extent of a grid of {n} axes is {n} (low, high) pairs of "
                         f"point indices, not {extent!r}")
    return [tuple(int(v) for v in values[2 * a:2 * a + 2]) for a in range(n)]


def _grid_axes(data, n):
    """Which of x, y and z a grid of ``n`` axes has, where its coordinates say (an
    image's or a rectilinear grid's keep all three, the flat ones of one point);
    otherwise None."""
    coordinates = data.geometry.values
    if n < 3 and isinstance(coordinates, (UniformCoordinates, CartesianProduct)):
        axes = [a for a, d in enumerate(coordinates.dims) if d != 1]
        if len(axes) == n:
            return axes
    return None


def _vtk_form(data, n, extent, rate):
    """``extent`` and ``rate`` given in VTK's three axes, as VTK writes a flat
    image's (``(i0, i1, j0, j1, 0, 0)``, a rate per axis), reduced to the
    grid's ``n`` axes; anything else as it is."""
    values = np.asarray(extent, dtype=np.int64).reshape(-1)
    axes = _grid_axes(data, n) if values.size == 6 else None
    if axes is None:
        return extent, rate
    extent = [values[2 * a:2 * a + 2] for a in axes]
    if np.ndim(rate) == 1 and len(rate) == 3:
        rate = [rate[a] for a in axes]
    return extent, rate


@on_inputs_device
def extract_extent(data, extent, rate=1, fields=None):
    """The points of ``data``, a structured grid, in the box ``extent`` -- one
    (low, high) pair of point indices per axis, both included, or VTK's flat
    (i0, i1, j0, j1, k0, k1), in the whole grid's indices (as VTK's extents
    count them, from the topology's ``offset``), clipped to the grid -- every
    ``rate``-th along each axis (a number, or one per axis): a structured grid
    of the same kind, an image's coordinates still uniform, a rectilinear
    grid's axes cut, explicit points taken. VTK's ``vtkExtractGrid``. A flat
    image or rectilinear grid also takes VTK's three axes' extent and rates,
    the flat axis's pair and rate ignored.

    An axis of one point is dropped: a k-plane of a 3D grid is a 2D grid,
    whose cells are faces of the 3D grid's. Point fields come along at the
    points kept; each cell takes the cell fields, marks and ids of the input
    cell at its first corner -- the cell beyond a plane, or before it at the
    grid's far side -- as VTK takes them. The output's ``offset`` is its first
    point's index, so its extent is the box's (with ``rate`` 1)."""
    topology = data.topology
    if not isinstance(topology, StructuredTopology):
        raise TypeError("extract_extent takes a structured grid (an image, a rectilinear "
                        "or a curvilinear grid); extract_cells selects any mesh's cells")
    dims = list(topology.point_dims)
    n = len(dims)
    extent, rate = _vtk_form(data, n, extent, rate)
    rates = [int(rate)] * n if np.ndim(rate) == 0 else [int(r) for r in rate]
    if len(rates) != n or min(rates) < 1:
        raise ValueError(f"rate is a step of at least 1, or one per axis, not {rate!r}")
    offset = list(topology.offset)
    starts, counts = [], []
    for (lo, hi), d, o, r in zip(_ranges(extent, n), dims, offset, rates, strict=True):
        lo, hi = max(lo, o), min(hi, o + d - 1)
        if hi < lo:
            raise ValueError(f"the extent {extent!r} misses the grid's points "
                             f"{[(o, o + d - 1) for o, d in zip(offset, dims)]}")
        starts.append(lo - o)
        counts.append((hi - lo) // r + 1)
    kept = [a for a in range(n) if counts[a] > 1]
    if not kept:
        raise ValueError(f"the extent {extent!r} holds one point of the grid: no cells")
    out_dims = [counts[a] for a in kept]
    device = data.device
    out = StructuredTopology(out_dims, id_dtype=topology.id_dtype, device=device,
                             offset=[offset[a] + starts[a] for a in kept])
    # Each output cell's first corner, in the input's cells; along a dropped
    # axis, the cell beyond the point, or before it at the far side.
    cells = [d - 1 for d in dims]
    cell_starts = [min(s, c - 1) if counts[a] == 1 else s
                   for a, (s, c) in enumerate(zip(starts, cells))]
    cell_counts = [max(c - 1, 1) for c in counts]
    point_ids = _ids(dims, starts, counts, rates, topology.id_dtype)
    cell_ids = (_ids(cells, cell_starts, cell_counts, rates, topology.id_dtype)
                if out_dims and min(cells) > 0 else tack.empty((0,), topology.id_dtype))
    coordinates = data.geometry.values
    geometry = None
    if isinstance(data.geometry.space, H1) and isinstance(coordinates,
                                                          (UniformCoordinates, CartesianProduct)):
        geometry = Field(H1(out), _strided_geometry(coordinates, starts, counts, rates, device),
                         data.unit)
    return carry(data, out, points=Take(point_ids), cells=Take(cell_ids), geometry=geometry,
                 fields=fields)


def _strided_geometry(coordinates, starts, counts, rates, device):
    """The implicit coordinates of the points kept: ``split``'s block, its spacing
    or axes taken every ``rate``-th."""
    axes = [a for a, d in enumerate(coordinates.dims) if d != 1]
    if isinstance(coordinates, UniformCoordinates):
        block = _block_geometry(coordinates, axes, starts, counts, device)
        spacing = np.asarray(block._spacing, float).copy()
        for axis, r in zip(axes, rates, strict=True):
            spacing[axis] *= r
        return UniformCoordinates(block.dims, block._origin, spacing,
                                  None if not block.ORIENTED else block._matrix,
                                  dtype=tack.arrays.dtype_of(coordinates), device=device)
    lines = [axis.numpy() for axis in (coordinates.x, coordinates.y, coordinates.z)]
    for axis, s, c, r in zip(axes, starts, counts, rates, strict=True):
        lines[axis] = lines[axis][s:s + (c - 1) * r + 1:r]
    return CartesianProduct(*lines, dtype=tack.arrays.dtype_of(coordinates), device=device)
