"""``voxelize``: data onto an image's voxels -- the fraction of each voxel the data
covers and each field's mean over that part, by sampling every voxel."""

import numpy as np

import tack
from tack import arrays
from tack.arrays import UniformCoordinates
from tack.dataset import units
from tack.dataset.dataset import DataSet, Field
from tack.dataset.spaces import Values

__all__ = ['voxelize']

#: Samples per batch: what one probe takes at a time.
_BATCH = 1 << 22


@tack.kernel
def _samples(coords, v0, cx, cy, kx, ky, kz, hx, hy, hz, dx, dy, dz, per, out):
    # Sample s of voxel v0 + t // per: the voxel's index plus (s + 1/2) / k along
    # each axis the image has cells on (plus 0 on the others), through the
    # image's coordinates, moved from its origin to the data's.
    for t in range(out.shape[0]):
        v = v0 + t // per
        s = t % per
        i = v % cx
        j = (v // cx) % cy
        m = v // (cx * cy)
        a = s % kx
        b = (s // kx) % ky
        c = s // (kx * ky)
        x = coords.get_ijk([i + (a + hx) / kx, j + (b + hy) / ky, m + (c + hz) / kz])
        out[t] = [x[0] + dx, x[1] + dy, x[2] + dz]


@tack.kernel
def _fractions(valid, v0, per, fraction, hidden, n):
    for v in range(n):
        inside = 0
        for s in range(per):
            inside += valid[v * per + s]
        fraction[v0 + v] = inside / per
        hidden[v0 + v] = 0 if inside > 0 else 1


@tack.kernel
def _means(values, valid, v0, per, out, n):
    # Each voxel's mean over its samples inside, in sample order (the same bits
    # every run); 0 where none is.
    for v in range(n):
        total = out[v0 + v] * 0.0
        inside = 0
        for s in range(per):
            if valid[v * per + s] == 1:
                total += values[v * per + s]
                inside += 1
        out[v0 + v] = total / inside if inside > 0 else total


def _like(data):
    """A dataset standing for ``data``'s frame and type: itself, a first piece or box,
    an octree's roots."""
    from tack.dataset.amr import AMRDataSet
    from tack.dataset.octree import OctreeDataSet
    from tack.dataset.partitioned import PartitionedDataSet

    if isinstance(data, OctreeDataSet):
        return data.roots
    if isinstance(data, AMRDataSet):
        return next(box for _, box in data)
    if isinstance(data, PartitionedDataSet):
        return next(iter(data))
    return data


def voxelize(data, grid, fields=None, samples=4):
    """``data`` on the voxels (cells) of ``grid``, an image (``uniform_grid``): the
    image with ``fraction``, the share of each voxel's volume the data covers, and
    each of ``fields``' mean over that share (point and cell fields with a basis,
    all by default), as cell data -- what a simulation's fields are on a uniform
    grid for analysis or learning. A voxel the data does not reach is hidden
    (``HIDDENCELL``), its means 0.

    Each voxel is sampled at ``samples`` points along each axis it spans
    (``samples ** 3`` in 3D, ``** 2`` on a 2D image), evenly, at the centres of
    its sub-voxels; a sample is inside where ``probe`` finds a cell holding it,
    and takes ``probe``'s value there. So ``fraction`` is right to about one
    sample's share, and a mean is the samples' mean, not an integral; more
    samples cost more probes. ``data`` is anything ``probe`` takes: a dataset,
    pieces over MPI ranks, an AMR hierarchy (its finest level at each sample)
    or an octree. The image keeps its fields; the result is on its device, at
    its origin and time. ``data``'s and ``grid``'s spatial units must agree."""
    from tack.dataset.algorithms._extract import _vertices
    from tack.dataset.algorithms.probe import probe
    from tack.dataset.ghosts import Ghosts, keep_marks
    from tack.dataset.topology import StructuredTopology

    coords = grid.geometry.values
    if not isinstance(coords, UniformCoordinates) or \
            not isinstance(grid.topology, StructuredTopology):
        raise TypeError("voxelize goes onto an image's voxels (uniform_grid)")
    k = int(samples)
    if k < 1:
        raise ValueError(f"samples is at least 1 a side, not {samples!r}")
    like = _like(data)
    units.check("voxelize's data and image spatial units", like.unit, grid.unit)
    cells = [max(d - 1, 1) for d in coords.dims]
    spans = [d > 1 for d in coords.dims]
    ks = [k if s else 1 for s in spans]
    halves = [0.5 if s else 0.0 for s in spans]
    per = int(np.prod(ks))
    n = grid.num_cells
    shift = np.asarray(grid.origin, np.float64) - np.asarray(like.origin, np.float64)
    device = grid.device
    with device:
        fraction = tack.empty((n,), like.dtype)
        hidden = tack.empty((n,), tack.u8)
        means = {}
        batch = max(1, _BATCH // per)
        for v0 in range(0, n, batch):
            nb = min(batch, n - v0)
            points = tack.empty((nb * per,), tack.vec3(like.dtype))
            _samples(coords, v0, cells[0], cells[1], ks[0], ks[1], ks[2], *halves, *shift,
                     per, points)
            probed = probe(data, _vertices(points, like), fields=fields)
            valid = probed.fields["valid"].values
            _fractions(valid, v0, per, fraction, hidden, nb)
            for name, field in probed.fields.items():
                if name == "valid" or name in grid.fields:
                    continue
                values = field.values
                if name not in means:
                    width = arrays.width_of(values)
                    dtype = arrays.dtype_of(values)
                    dtype = dtype if dtype.numpy_dtype.kind == "f" else like.dtype
                    means[name] = (tack.zeros((n,), tack.vec(width, dtype) if width else dtype),
                                   field.unit)
                _means(values, valid, v0, per, means[name][0], nb)
    out = dict(grid.fields)
    space = Values(grid, "cells")
    out["fraction"] = Field(space, fraction, "dimensionless")
    for name, (values, unit) in means.items():
        out[name] = Field(space, values, unit)
    result = keep_marks(grid, DataSet(grid.topology, grid.geometry, fields=out,
                                      sets=dict(grid.sets)))
    # Empty voxels hidden, beside whatever marks the image had; on its device.
    marks = hidden.numpy() * np.uint8(Ghosts.HIDDENCELL)
    if grid.ghosts is not None and grid.ghosts.cells is not None:
        marks |= grid.ghosts.cells.numpy()
    result.ghosts = Ghosts(cells=tack.array(marks.astype(np.uint8), device=device),
                           points=None if grid.ghosts is None else grid.ghosts.points)
    return result
