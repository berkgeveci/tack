"""``voxel_downsample``: one point per occupied voxel of a uniform grid -- the voxel's centroid, its first point, or its point nearest the centroid -- as VTK's vtkVoxelGrid and PDAL's ``filters.voxeldownsize``. Deterministic: voxels by a stable sort, each reduced in index order."""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.sort import _run_offsets, argsort, gather
from tack.dataset.algorithms._common import _length
from tack.dataset.algorithms._extract import _keep_points, _vertices
from tack.dataset.carry import _point_values, selected
from tack.dataset.dataset import Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.partitioned import PartitionedDataSet, common_origin
from tack.dataset.point_locator import squared_length
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
    # Each voxel's point nearest its centroid, the lowest index on a tie. A
    # voxel of two points has them equidistant: the distances must round alike
    # on every backend (squared_length), or the tie falls differently.
    for r in range(centroids.shape[0]):
        c = centroids[r]
        best = order[offsets[r]]
        nearest = squared_length(positions[best] - c)
        for s in range(offsets[r] + 1, offsets[r + 1]):
            p = order[s]
            d2 = squared_length(positions[p] - c)
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
    reduced in index order, so the result has the same bits from run to run,
    and the voxels and kept points are the same on every backend: the
    nearest point's distances are rounded alike everywhere
    (``squared_length``), so a tie falls the same way.

    Over a ``PartitionedDataSet``, the grid is anchored at the lower corner of
    every piece's points, and each voxel's owned points are brought to one
    block by a sort over the blocks, in order of piece and then index, and
    reduced there. A voxel belongs to the piece of its first point, which keeps
    its centroid; ``"first"`` and ``"nearest"`` keep input points on their own
    pieces. Each piece of the result holds its share of the whole's points."""
    size = _length(data, size, "voxel_downsample's size")
    if keep not in _KEEP:
        raise ValueError(f"keep is one of {', '.join(map(repr, _KEEP))}, not {keep!r}")
    sizes = np.broadcast_to(np.asarray(size, np.float64), (3,)).astype(float)
    if not (np.all(np.isfinite(sizes)) and np.all(sizes > 0)):
        raise ValueError(f"size is a positive voxel side, or three, not {size!r}")
    if isinstance(data, PartitionedDataSet):
        return _downsample_pieces(data, size, sizes, keep, fields)
    if not isinstance(data.geometry.space, H1):
        raise TypeError("voxel_downsample needs an H1 geometry: one position per point")
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
            out.fields[name] = Field(H1(out), averaged, field.unit)
        else:
            if firsts is None:
                firsts = tack.empty((nruns,), order.dtype)
                if nruns:
                    _gather_firsts(order, offsets, firsts)
            out.fields[name] = Field(H1(out), _taken(values, firsts, nruns), field.unit)
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


def _on_points(field):
    return (isinstance(field.space, H1) and field.space.order == 1) or (
        isinstance(field.space, Values) and field.space.on == "points")


def _downsample_pieces(data, size, sizes, keep, fields):
    """``voxel_downsample`` over pieces: the owned points of each voxel sorted to one
    block (``tack.communication.sort_by_key`` by voxel), reduced there, and what
    is kept sent to the piece of the voxel's first point."""
    from tack import communication as tc
    from tack.dataset.carry import _take as take_rows

    common_origin(data, "voxel_downsample of a PartitionedDataSet")
    pieces = list(data)
    blocks = tc.Blocks([piece.device for piece in pieces], data.world)
    names = [selected(piece, fields) for piece in pieces]
    every = blocks.all_gather({g: [name for name in chosen if _on_points(piece.fields[name])]
                               for g, piece, chosen in zip(blocks.local, pieces, names)})
    if any(chosen != every[0] for chosen in every):
        raise ValueError(f"voxel_downsample of a PartitionedDataSet needs the same point "
                         f"fields on every piece, not {every}")
    names = every[0] if every else []
    owned, positions = {}, {}
    for g, piece in zip(blocks.local, pieces):
        if not isinstance(piece.geometry.space, H1):
            raise TypeError("voxel_downsample needs an H1 geometry: one position per point")
        n = piece.num_points
        mine = np.ones(n, bool)
        if piece.ghosts is not None and piece.ghosts.points is not None:
            mine = (piece.ghosts.points.numpy() & Ghosts.DUPLICATEPOINT) == 0
        owned[g] = np.flatnonzero(mine).astype(np.int64)
        with piece.device:
            positions[g] = take_rows(arrays.materialize(_point_values(piece.geometry.values, n)),
                                 tack.array(owned[g], tack.i64))
    hosts = {g: arrays.to_host(positions[g]).reshape(-1, 3).astype(np.float64)
             for g in blocks.local}
    low = blocks.all_reduce({g: hosts[g].min(axis=0) if len(hosts[g]) else np.full(3, np.inf)
                             for g in blocks.local}, "min") if blocks.count else np.zeros(3)
    high = blocks.all_reduce({g: hosts[g].max(axis=0) if len(hosts[g]) else np.full(3, -np.inf)
                              for g in blocks.local}, "max") if blocks.count else np.zeros(3)
    if not np.all(np.isfinite(low)):
        low = high = np.zeros(3)
    counts = np.floor((high - low) / sizes).astype(np.int64) + 1
    if float(np.prod(counts.astype(float))) >= 2.0 ** 62:
        raise ValueError(f"{size!r} makes too many voxels over the points' box")
    keys, values = {}, {}
    for g, piece in zip(blocks.local, pieces):
        m = len(owned[g])
        with piece.device:
            keys[g] = tack.empty((m,), tack.i64)
            if m:
                _voxel_keys(positions[g], *map(float, low), *map(float, sizes),
                            int(counts[0]), int(counts[1]), keys[g])
            at = tack.array(owned[g], tack.i64)
            values[g] = {"x": positions[g], "piece": tack.full((m,), g, tack.i64),
                         "index": at}
            if keep == "centroid":
                for name in names:
                    field = arrays.materialize(_point_values(piece.fields[name].values,
                                                             piece.num_points))
                    if _floating(field):
                        values[g]["f." + name] = take_rows(field, at)
    sorted_keys, sorted_values = tc.sort_by_key(blocks, keys, values)

    outgoing = {}
    for g in blocks.local:
        got = sorted_values[g]
        m = int(sorted_keys[g].size)
        with blocks.device(g):
            offsets, nruns = _run_offsets(sorted_keys[g], m)
            order = tack.arange(m, dtype=tack.i64)
            firsts = tack.empty((nruns,), tack.i64)
            if nruns:
                _gather_firsts(order, offsets, firsts)
            message = {}
            if keep == "centroid":
                rows = firsts
                centroids = tack.empty((nruns,), got["x"].dtype)
                if nruns:
                    _run_means(got["x"], order, offsets, centroids)
                message["x"] = centroids
                for name, field in got.items():
                    if name.startswith("f."):
                        averaged = tack.empty((nruns,), field.dtype)
                        if nruns:
                            _run_means(field, order, offsets, averaged)
                        message[name] = averaged
            elif keep == "first":
                rows = firsts
            else:
                centroids = tack.empty((nruns,), got["x"].dtype)
                flags = tack.zeros((max(m, 1),), tack.i32)
                if nruns:
                    _run_means(got["x"], order, offsets, centroids)
                    _run_nearest(got["x"], centroids, order, offsets, flags)
                rows = tack.array(np.flatnonzero(flags.numpy()[:m]).astype(np.int64),
                                  tack.i64)
            home = arrays.to_host(take_rows(got["piece"], rows)).astype(np.int64) \
                if nruns else np.zeros(0, np.int64)
            index = take_rows(got["index"], rows) if nruns else tack.empty((0,), tack.i64)
            # A centroid goes to the piece of its voxel's first point; a kept
            # point stays where it is.
            outgoing[g] = {}
            for b in np.unique(home).tolist():
                chosen = tack.array(np.flatnonzero(home == b).astype(np.int64), tack.i64)
                outgoing[g][int(b)] = {"index": take_rows(index, chosen),
                                       **{name: take_rows(v, chosen) for name, v in message.items()}}
    incoming = blocks.exchange(outgoing)

    out = []
    for g, piece in zip(blocks.local, pieces):
        received = list(incoming[g].values())          # in order of block: of voxel
        with piece.device:
            if keep != "centroid":
                flags = tack.zeros((piece.num_points,), tack.i32)
                host = flags.numpy()
                for message in received:
                    host[message["index"].numpy()] = 1
                out.append(_keep_points(piece, tack.array(host, tack.i32), fields))
                continue
            firsts = _joined_arrays([m["index"] for m in received], tack.i64)
            centroids = _joined_arrays([m["x"] for m in received],
                                       tack.vec(3, piece.dtype))
            result = _vertices(centroids, piece)
            for name in names:
                field = piece.fields[name]
                values = arrays.materialize(_point_values(field.values, piece.num_points))
                if _floating(values):
                    kept = _joined_arrays([m["f." + name] for m in received], values.dtype)
                else:
                    kept = take_rows(values, firsts) if firsts.size else \
                        tack.empty((0,), values.dtype)
                result.fields[name] = Field(H1(result), kept, field.unit)
            out.append(result)
    return PartitionedDataSet(out, world=data.world)


def _floating(values):
    scalar = getattr(values.dtype, "scalar_dtype", values.dtype)
    return scalar.numpy_dtype.kind == "f"


def _joined_arrays(parts, dtype):
    parts = [p for p in parts if p.size]
    if not parts:
        return tack.empty((0,), dtype)
    return parts[0] if len(parts) == 1 else tack.concat(parts)
