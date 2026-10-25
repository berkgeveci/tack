"""``deposit``: particles onto an image -- into the cell holding each (nearest grid point) or onto that cell's corners by trilinear weights (cloud in cell) -- as counts, sums, means, minima or maxima, as Viskores' ParticleDensityNearestGridPoint and ParticleDensityCloudInCell and yt's deposition. A LiDAR elevation raster (a DSM's highest return, a DTM's lowest) is the maximum or minimum of the heights on a 2D image."""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.sort import _run_offsets, argsort, gather
from tack.arrays import UniformCoordinates
from tack.dataset.carry import _point_values
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['deposit']

_REDUCE = ("count", "sum", "mean", "min", "max")
_METHOD = ("nearest", "cic")


@tack.kernel
def _nearest_cells(positions, ox, oy, oz, ix, iy, iz, cx, cy, cz, targets, weights):
    # The cell holding each particle (the last cell takes its far face); -1 outside.
    for p in range(targets.shape[0]):
        x = positions[p]
        fx = (x[0] - ox) * ix
        fy = (x[1] - oy) * iy
        fz = (x[2] - oz) * iz
        t = tack.i64(-1)
        if fx >= 0.0 and fx <= cx and fy >= 0.0 and fy <= cy and fz >= 0.0 and fz <= cz:
            i = min(tack.i64(floor(fx)), max(cx - 1, 0))
            j = min(tack.i64(floor(fy)), max(cy - 1, 0))
            k = min(tack.i64(floor(fz)), max(cz - 1, 0))
            t = i + max(cx, 1) * (j + max(cy, 1) * k)
        targets[p] = t
        weights[p] = fx * 0.0 + 1.0


@tack.kernel
def _corner_weights(positions, ox, oy, oz, ix, iy, iz, cx, cy, cz, px, py, corners,
                    targets, weights):
    # Each particle's trilinear weights at its cell's corners (2, 4 or 8 of them,
    # the axes the image spans); -1 outside the image.
    for e in range(targets.shape[0]):
        p = e // corners
        c = e % corners
        x = positions[p]
        fx = (x[0] - ox) * ix
        fy = (x[1] - oy) * iy
        fz = (x[2] - oz) * iz
        t = tack.i64(-1)
        w = fx * 0.0
        if fx >= 0.0 and fx <= cx and fy >= 0.0 and fy <= cy and fz >= 0.0 and fz <= cz:
            i = min(tack.i64(floor(fx)), max(cx - 1, 0))
            j = min(tack.i64(floor(fy)), max(cy - 1, 0))
            k = min(tack.i64(floor(fz)), max(cz - 1, 0))
            ux = fx - i
            uy = fy - j
            uz = fz - k
            # Corner bits run over the spanned axes only, x first.
            bit = 0
            di = 0
            dj = 0
            dk = 0
            wx = ux * 0.0 + 1.0
            wy = wx
            wz = wx
            if cx > 0:
                di = (c >> bit) & 1
                bit += 1
                wx = ux if di == 1 else 1.0 - ux
            if cy > 0:
                dj = (c >> bit) & 1
                bit += 1
                wy = uy if dj == 1 else 1.0 - uy
            if cz > 0:
                dk = (c >> bit) & 1
                wz = uz if dk == 1 else 1.0 - uz
            t = (i + di) + px * ((j + dj) + py * (k + dk))
            w = wx * wy * wz
        targets[e] = t
        weights[e] = w


@tack.kernel
def _reduce_runs(order, offsets, sorted_targets, weights, values, per, how, scale, out, start):
    # Each target's entries, in entry order: 0 count, 1 sum, 2 mean, 3 min, 4 max.
    for r in range(start, offsets.shape[0] - 1):
        first = offsets[r]
        t = sorted_targets[first]
        e = order[first]
        w = weights[e]
        v = values[e // per]
        total = v * w
        mass = w
        low = v
        high = v
        for s in range(first + 1, offsets[r + 1]):
            e = order[s]
            w = weights[e]
            v = values[e // per]
            total += v * w
            mass += w
            low = min(low, v)
            high = max(high, v)
        result = mass
        if how == 1:
            result = total
        elif how == 2:
            result = total / mass
        elif how == 3:
            result = low
        elif how == 4:
            result = high
        out[t] = result * scale


@on_inputs_device
def deposit(data, grid, field=None, reduce=None, method="nearest", per_volume=False):
    """``data``'s points deposited on ``grid``, an image (``uniform_grid``, axis-
    aligned): a field on the image.

    ``method="nearest"`` puts each point in the cell holding it (nearest grid
    point, Viskores' ParticleDensityNearestGridPoint): the field is on the
    cells. ``method="cic"`` spreads it over that cell's corners by trilinear
    weights (cloud in cell, Viskores' ParticleDensityCloudInCell, yt): the
    field is on the points. Points outside the image are left out.

    Without ``field``, each cell's (or corner's) count of points, or sum of
    weights. With one -- a point field of ``data`` or its name -- ``reduce``
    takes ``"sum"`` (the default), ``"mean"``, ``"count"``, or, for
    ``"nearest"``, ``"min"`` and ``"max"``: a LiDAR surface model is the
    ``"max"`` of the heights on a 2D image, a terrain model the ``"min"`` of the
    ground's. A cell nothing reaches holds 0 for counts and sums, NaN for the
    rest. ``per_volume`` divides by a cell's volume (area, length on images of
    fewer axes): a number or mass density. Each target sums its entries in
    point order, so the result has the same bits on every backend.

    ``data`` and ``grid`` may have different origins: the points are moved to
    the image's by the difference, taken in double precision."""
    if method not in _METHOD:
        raise ValueError(f"method is one of {', '.join(map(repr, _METHOD))}, not {method!r}")
    reduce = reduce or ("sum" if field is not None else "count")
    if reduce not in _REDUCE:
        raise ValueError(f"reduce is one of {', '.join(map(repr, _REDUCE))}, not {reduce!r}")
    if field is None and reduce != "count":
        raise ValueError(f"reduce={reduce!r} needs a field to reduce")
    if method == "cic" and reduce in ("min", "max"):
        raise ValueError("cloud in cell weighs each point: it sums, counts or averages")
    if not isinstance(data.geometry.space, H1):
        raise TypeError("deposit needs the points' positions: an H1 geometry")
    coordinates = grid.geometry.values
    if not isinstance(coordinates, UniformCoordinates) or coordinates.ORIENTED == 1:
        raise TypeError("deposit goes onto an axis-aligned image (uniform_grid)")
    n = data.num_points
    positions = arrays.materialize(_point_values(data.geometry.values, n))
    if data.origin != grid.origin:
        from tack.dataset.algorithms._moving import _shifted_values

        positions = _shifted_values(positions, data.dtype,
                                    np.asarray(data.origin) - np.asarray(grid.origin))
    dims = np.asarray(coordinates.dims, np.int64)
    cells = np.maximum(dims - 1, 0)
    spacing = np.asarray(coordinates._spacing, np.float64)
    inverse = np.where(cells > 0, 1.0 / spacing, 0.0)
    origin = np.asarray(coordinates._origin, np.float64)
    if field is None:
        values = tack.zeros((max(n, 1),), data.dtype)
    else:
        f = data.fields[field] if isinstance(field, str) else field
        on_points = (isinstance(f.space, H1) and f.space.order == 1) or (
            isinstance(f.space, Values) and f.space.on == "points")
        if not on_points or f.space.topology is not data.topology:
            raise TypeError("deposit reduces a point field of the deposited points")
        values = arrays.materialize(_point_values(f.values, n))
        if getattr(values.dtype, "scalar_dtype", None) is not None:
            raise TypeError("deposit reduces a scalar field: deposit each component")
    corners = 1 if method == "nearest" else 2 ** int((cells > 0).sum())
    entries = n * corners
    targets = tack.empty((entries,), tack.i64)
    weights = tack.empty((entries,), data.dtype)
    grid_args = (*map(float, origin), *map(float, inverse), *map(int, cells))
    if entries and method == "nearest":
        _nearest_cells(positions, *grid_args, targets, weights)
    elif entries:
        _corner_weights(positions, *grid_args, int(dims[0]), int(dims[1]), corners,
                        targets, weights)
    if method == "nearest":
        size = int(np.prod(np.maximum(cells, 1))) if cells.any() else 0
        space = Values(grid, "cells")
    else:
        size = int(np.prod(dims))
        space = Values(grid, "points")
    integral = field is None or reduce == "count"
    if integral and method == "nearest":
        dtype = data.id_dtype
    elif field is None:
        dtype = data.dtype
    else:
        dtype = values.dtype
    out = tack.zeros((size,), dtype) if (reduce in ("count", "sum") or integral) \
        else tack.full((size,), float("nan"), dtype)
    scale = 1.0
    if per_volume:
        scale = 1.0 / float(np.prod(np.where(cells > 0, spacing, 1.0)))
        if integral and method == "nearest":
            out = tack.zeros((size,), data.dtype)
    order = argsort(targets)
    sorted_targets = gather(targets, order)
    offsets, nruns = _run_offsets(sorted_targets, entries)
    if nruns:
        # Runs of -1 (outside) sort first; skip that one.
        start = 1 if int(sorted_targets[0]) < 0 else 0
        how = _REDUCE.index(reduce)
        if nruns > start:
            reduced = values if not integral else weights
            per = corners if not integral else 1
            _reduce_runs(order, offsets, sorted_targets, weights, reduced, per,
                         0 if integral else how, scale, out, start)
    return Field(space, out)
