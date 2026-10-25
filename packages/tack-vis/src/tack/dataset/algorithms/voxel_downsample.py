"""``voxel_downsample``: one point per occupied voxel of a uniform grid -- the voxel's centroid, its first point, or its point nearest the centroid -- as VTK's vtkVoxelGrid and PDAL's ``filters.voxeldownsize``. Deterministic: voxels by a stable sort, each reduced in index order."""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.sort import _run_offsets, argsort, gather
from tack.dataset.algorithms._extract import _keep_points, _vertices
from tack.dataset.carry import _point_values, selected
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['voxel_downsample']

_KEEP = ("centroid", "first", "nearest")


@tack.kernel
def _voxel_keys(positions, ox, oy, oz, sx, sy, sz, nx, ny, keys):
    for p in range(keys.shape[0]):
        x = positions[p]
        i = tack.i64(floor((x[0] - ox) / sx))
        j = tack.i64(floor((x[1] - oy) / sy))
        k = tack.i64(floor((x[2] - oz) / sz))
        keys[p] = i + nx * (j + ny * k)


@tack.kernel
def _run_means(values, order, offsets, out):
    # Each voxel's mean, its points summed in index order.
    for r in range(out.shape[0]):
        first = offsets[r]
        total = values[order[first]]
        for s in range(first + 1, offsets[r + 1]):
            total += values[order[s]]
        out[r] = total / (offsets[r + 1] - first)


@tack.kernel
def _run_firsts(order, offsets, flags):
    # A stable sort puts each voxel's lowest index first.
    for r in range(offsets.shape[0] - 1):
        flags[order[offsets[r]]] = 1


@tack.kernel
def _run_nearest(positions, centroids, order, offsets, flags):
    # Each voxel's point nearest its centroid, the lowest index on a tie.
    for r in range(centroids.shape[0]):
        c = centroids[r]
        best = order[offsets[r]]
        d = positions[best] - c
        nearest = d.dot(d)
        for s in range(offsets[r] + 1, offsets[r + 1]):
            p = order[s]
            e = positions[p] - c
            d2 = e.dot(e)
            if d2 < nearest or (d2 == nearest and p < best):
                nearest = d2
                best = p
        flags[best] = 1


@on_inputs_device
def voxel_downsample(data, size, keep="centroid", fields=None):
    """One point per occupied voxel of a grid of ``size`` (a voxel's side, or three
    sides, in the geometry's units) anchored at the points' lower corner, as
    VTK's vtkVoxelGrid and PDAL's ``filters.voxeldownsize``: vertex cells, with
    the input's origin.

    ``keep`` chooses the point: ``"centroid"``, the mean of the voxel's points
    (new points: floating-point point fields averaged, integer ones -- a
    classification, intensities -- the voxel's first point's); ``"first"``,
    the voxel's lowest-numbered point; ``"nearest"``, its point nearest the
    centroid (the lowest-numbered on a tie). The last two keep input points,
    in input order, with their fields and ghost marks. Point fields come along
    (``fields`` choosing, as filters do); cell fields do not. Each voxel is
    reduced in index order, so the result has the same bits on every backend."""
    if keep not in _KEEP:
        raise ValueError(f"keep is one of {', '.join(map(repr, _KEEP))}, not {keep!r}")
    if not isinstance(data.geometry.space, H1):
        raise TypeError("voxel_downsample needs an H1 geometry: one position per point")
    sizes = np.broadcast_to(np.asarray(size, np.float64), (3,)).astype(float)
    if not (np.all(np.isfinite(sizes)) and np.all(sizes > 0)):
        raise ValueError(f"size is a positive voxel side, or three, not {size!r}")
    n = data.num_points
    positions = arrays.materialize(_point_values(data.geometry.values, n))
    if n:
        x = arrays.to_host(positions).reshape(-1, 3).astype(np.float64)
        low, high = x.min(axis=0), x.max(axis=0)
    else:
        low = high = np.zeros(3)
    counts = np.floor((high - low) / sizes).astype(np.int64) + 1
    if float(np.prod(counts.astype(float))) >= 2.0 ** 62:
        raise ValueError(f"{size!r} makes too many voxels over the points' box")
    keys = tack.empty((n,), tack.i64)
    if n:
        _voxel_keys(positions, *map(float, low), *map(float, sizes), int(counts[0]),
                    int(counts[1]), keys)
    order = argsort(keys)
    offsets, nruns = _run_offsets(gather(keys, order), n)
    if keep == "centroid":
        centroids = tack.empty((nruns,), tack.vec(3, data.dtype))
        if nruns:
            _run_means(positions, order, offsets, centroids)
        return _centroids(data, centroids, order, offsets, nruns, fields)
    flags = tack.zeros((n,), tack.i32)
    if nruns and keep == "first":
        _run_firsts(order, offsets, flags)
    elif nruns:
        centroids = tack.empty((nruns,), tack.vec(3, data.dtype))
        _run_means(positions, order, offsets, centroids)
        _run_nearest(positions, centroids, order, offsets, flags)
    return _keep_points(data, flags, fields)


def _centroids(data, centroids, order, offsets, nruns, fields):
    from tack.dataset.ghosts import drop_marks

    out = _vertices(centroids, data)
    firsts = None
    for name in selected(data, fields):
        field = data.fields[name]
        on_points = (isinstance(field.space, H1) and field.space.order == 1) or (
            isinstance(field.space, Values) and field.space.on == "points")
        if not on_points:
            continue
        values = arrays.materialize(_point_values(field.values, data.num_points))
        scalar = getattr(values.dtype, "scalar_dtype", values.dtype)
        if scalar.numpy_dtype.kind == "f":
            averaged = tack.empty((nruns,), values.dtype)
            if nruns:
                _run_means(values, order, offsets, averaged)
            out.fields[name] = Field(H1(out), averaged)
        else:
            if firsts is None:
                firsts = tack.empty((nruns,), order.dtype)
                if nruns:
                    _gather_firsts(order, offsets, firsts)
            out.fields[name] = Field(H1(out), _taken(values, firsts, nruns))
    return drop_marks(data, out, "voxel_downsample's centroids are new points, which no "
                                 "piece owns")


@tack.kernel
def _gather_firsts(order, offsets, firsts):
    for r in range(firsts.shape[0]):
        firsts[r] = order[offsets[r]]


@tack.kernel
def _take(values, ids, out):
    for r in range(out.shape[0]):
        out[r] = values[ids[r]]


def _taken(values, ids, n):
    out = tack.empty((n,), values.dtype)
    if n:
        _take(values, ids, out)
    return out
