"""``outliers``: points with too few neighbours within a radius, or too far from their k nearest, marked -- as VTK's vtkRadiusOutlierRemoval and vtkStatisticalOutlierRemoval, and PDAL's ``filters.outlier`` (which marks too)."""

import numpy as np

import tack
from tack import arrays
from tack.dataset.dataset import DataSet, Field
from tack.dataset.partitioned import PartitionedDataSet
from tack.dataset.point_locator import MAX_K, _count_within, cached_point_locator
from tack.dataset.spaces import H1, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['outliers']


@tack.kernel
def _mean_distances(d2s, k, out):
    for q in range(out.shape[0]):
        total = out[q] * 0.0
        found = 0
        for t in range(k):
            d2 = d2s[q * k + t]
            if d2 >= 0.0:
                total += sqrt(d2)
                found += 1
        out[q] = total / max(found, 1)


@tack.kernel
def _below(counts, least, marks):
    for q in range(marks.shape[0]):
        marks[q] = 1 if counts[q] < least else 0


@tack.kernel
def _above(values, limit, marks):
    for q in range(marks.shape[0]):
        marks[q] = 1 if values[q] > limit else 0


@on_inputs_device
def outliers(data, radius=None, min_neighbors=None, k=None, deviations=1.0, periodic=None,
             locator=None):
    """``data`` with each point marked an outlier or not: ``outlier``, a u8 point
    field, 1 for an outlier. Two tests, as VTK's two filters:

    - **radius** (``radius`` and ``min_neighbors``): fewer than ``min_neighbors``
      other points within ``radius``, as vtkRadiusOutlierRemoval; ``neighbors``
      gives each point's count.
    - **statistical** (``k``, at most 32, and ``deviations``): a mean distance
      to its ``k`` nearest other points more than ``deviations`` standard
      deviations above the mean of those means, as vtkStatisticalOutlierRemoval
      and PCL; ``mean_distance`` gives each point's. The mean and deviation are
      taken in double precision on the host.

    The points are marked, not removed, as PDAL marks them (class 7):
    ``threshold_points(out, "outlier", 0, 0)`` keeps the rest. ``periodic`` (a
    box) measures across its faces. ``locator`` reuses a ``PointLocator`` of
    ``data``.

    Over a ``PartitionedDataSet`` of point clouds, each piece is searched with
    the copies of the other pieces' points it needs (``ghost_points``: within
    ``radius``, or its points' ``k`` nearest), and the statistical test's mean
    and deviation are over every piece's points: each point gets the whole's
    answer. The pieces come back without the copies."""
    radial = radius is not None or min_neighbors is not None
    if radial == (k is not None):
        raise ValueError("outliers takes radius and min_neighbors, or k (and deviations)")
    if radial and (radius is None or min_neighbors is None or not radius >= 0):
        raise ValueError("the radius test takes a radius of at least zero and "
                         "min_neighbors")
    if not radial and not (isinstance(k, (int, np.integer)) and 1 <= k <= MAX_K):
        raise ValueError(f"k is a whole number from 1 to {MAX_K}, not {k!r}")
    if isinstance(data, PartitionedDataSet):
        if locator is not None:
            raise ValueError("a PointLocator finds one piece's points; pieces take none")
        return _outliers_pieces(data, radius, min_neighbors, k, deviations, periodic)
    if not isinstance(data.geometry.space, H1):
        raise TypeError("outliers needs an H1 geometry: one position per point")
    locator = _locator(data, locator, periodic)
    n = data.num_points
    marks = tack.zeros((n,), tack.u8)
    out = DataSet(data.topology, data.geometry, fields=dict(data.fields),
                  sets=dict(data.sets), dtype=data.dtype, origin=data.origin, unit=data.unit)
    points = Values(out, "points")
    if radial:
        locator.check_radius(radius)
        counts = tack.empty((n,), data.id_dtype)
        if n:
            _count_within(locator.grid(), locator.positions, float(radius), 1, counts)
            _below(counts, int(min_neighbors), marks)
        out.fields["neighbors"] = Field(points, counts)
    else:
        means = _means(locator, n, int(k), data.dtype)
        if n:
            host = arrays.to_host(means).astype(np.float64)
            limit = host.mean() + float(deviations) * host.std()
            _above(means, float(limit), marks)
        out.fields["mean_distance"] = Field(points, means)
    out.fields["outlier"] = Field(points, marks)
    from tack.dataset.ghosts import keep_marks

    return keep_marks(data, out)


def _means(locator, n, k, dtype):
    """Each point's mean distance to its ``k`` nearest others."""
    means = tack.zeros((n,), dtype)
    if n:
        _, d2s = locator.nearest(k)
        _mean_distances(d2s.reshape((n * k,)), k, means)
    return means


def _outliers_pieces(data, radius, min_neighbors, k, deviations, periodic):
    from tack.communication._kernels import take
    from tack.dataset.algorithms.ghost_points import _ghosted
    from tack.dataset.ghosts import keep_marks

    ghosted = _ghosted(data, radius, k, periodic, None, [], None)
    blocks = ghosted.blocks
    found = {}
    for g, piece in zip(blocks.local, ghosted.pieces):
        n = ghosted.owned[g]
        with piece.device:
            if radius is not None:
                marked = outliers(piece, radius=radius, min_neighbors=min_neighbors)
                found[g] = {name: take(marked.fields[name].values, 0, n)
                            for name in ("neighbors", "outlier")}
            else:
                means = _means(cached_point_locator(piece), piece.num_points, int(k),
                               piece.dtype)
                found[g] = {"mean_distance": take(means, 0, n)}
    if k is not None:
        # The mean and deviation of every piece's means, in two passes over the
        # blocks, in order of block.
        host = {g: arrays.to_host(found[g]["mean_distance"]).astype(np.float64)
                for g in blocks.local}
        count = blocks.all_reduce({g: len(host[g]) for g in blocks.local}, "sum")
        mean = blocks.all_reduce({g: float(host[g].sum()) for g in blocks.local}, "sum") / \
            max(count, 1)
        spread = blocks.all_reduce({g: float(((host[g] - mean) ** 2).sum())
                                    for g in blocks.local}, "sum")
        limit = mean + float(deviations) * float(np.sqrt(spread / max(count, 1)))
        for g, piece in zip(blocks.local, data):
            with piece.device:
                marks = tack.zeros((piece.num_points,), tack.u8)
                if piece.num_points:
                    _above(found[g]["mean_distance"], float(limit), marks)
                found[g]["outlier"] = marks
    out = []
    for g, piece in zip(blocks.local, data):
        with piece.device:
            result = DataSet(piece.topology, piece.geometry, fields=dict(piece.fields),
                             sets=dict(piece.sets), dtype=piece.dtype, origin=piece.origin, unit=piece.unit)
            for name, values in found[g].items():
                result.fields[name] = Field(Values(result, "points"), values)
            out.append(keep_marks(piece, result))
    return PartitionedDataSet(out, world=data.world)


def _locator(data, locator, periodic=None):
    if locator is None:
        return cached_point_locator(data, periodic)
    if not locator.serves(data, periodic):
        raise ValueError("the locator was built for another topology, geometry or periodic "
                         "box, or Tack has written the positions since: make a new one")
    return locator
