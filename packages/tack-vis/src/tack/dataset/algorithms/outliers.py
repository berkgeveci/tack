"""``outliers``: points with too few neighbours within a radius, or too far from their k nearest, marked -- as VTK's vtkRadiusOutlierRemoval and vtkStatisticalOutlierRemoval, and PDAL's ``filters.outlier`` (which marks too)."""

import numpy as np

import tack
from tack import arrays
from tack.dataset.dataset import DataSet, Field
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
def outliers(data, radius=None, min_neighbors=None, k=None, deviations=1.0, locator=None):
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
    ``threshold_points(out, "outlier", 0, 0)`` keeps the rest. ``locator``
    reuses a ``PointLocator`` of ``data``."""
    radial = radius is not None or min_neighbors is not None
    if radial == (k is not None):
        raise ValueError("outliers takes radius and min_neighbors, or k (and deviations)")
    if not isinstance(data.geometry.space, H1):
        raise TypeError("outliers needs an H1 geometry: one position per point")
    locator = _locator(data, locator)
    n = data.num_points
    marks = tack.zeros((n,), tack.u8)
    out = DataSet(data.topology, data.geometry, fields=dict(data.fields),
                  sets=dict(data.sets), dtype=data.dtype, origin=data.origin)
    points = Values(out, "points")
    if radial:
        if radius is None or min_neighbors is None or not radius >= 0:
            raise ValueError("the radius test takes a radius of at least zero and "
                             "min_neighbors")
        counts = tack.empty((n,), data.id_dtype)
        if n:
            _count_within(locator.grid(), locator.positions, float(radius), 1, counts)
            _below(counts, int(min_neighbors), marks)
        out.fields["neighbors"] = Field(points, counts)
    else:
        if not (isinstance(k, (int, np.integer)) and 1 <= k <= MAX_K):
            raise ValueError(f"k is a whole number from 1 to {MAX_K}, not {k!r}")
        means = tack.zeros((n,), data.dtype)
        if n:
            _, d2s = locator.nearest(int(k))
            _mean_distances(d2s.reshape((n * k,)), int(k), means)
            host = arrays.to_host(means).astype(np.float64)
            limit = host.mean() + float(deviations) * host.std()
            _above(means, float(limit), marks)
        out.fields["mean_distance"] = Field(points, means)
    out.fields["outlier"] = Field(points, marks)
    from tack.dataset.ghosts import keep_marks

    return keep_marks(data, out)


def _locator(data, locator):
    if locator is None:
        return cached_point_locator(data)
    if not locator.serves(data):
        raise ValueError("the locator was built for another topology or geometry, or "
                         "Tack has written the positions since: make a new one")
    return locator
