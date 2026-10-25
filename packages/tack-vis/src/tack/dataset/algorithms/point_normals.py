"""``point_normals``: each point's normal from the principal axes of its neighbourhood -- the direction of least spread of its k nearest points, or of those within a radius -- as VTK's vtkPCANormalEstimation and PDAL's ``filters.normal``."""

import numpy as np

import tack
from tack.arrays import CountingArray
from tack.dataset.algorithms.outliers import _locator
from tack.dataset.dataset import DataSet, Field
from tack.dataset.point_locator import MAX_K
from tack.dataset.spaces import H1
from tack.runtime.dispatch import on_inputs_device

__all__ = ['point_normals']


@tack.func
def _smallest_axis(a00, a01, a02, a11, a12, a22):
    """The unit eigenvector of the least eigenvalue of a symmetric 3x3 matrix: the
    eigenvalues in closed form, then the largest cross product of two rows of
    ``A - lambda I``. Zero where the least eigenvalue repeats (points on a
    line, or all at one place): no one direction is least."""
    q = (a00 + a11 + a22) / 3.0
    b00 = a00 - q
    b11 = a11 - q
    b22 = a22 - q
    off = a01 * a01 + a02 * a02 + a12 * a12
    p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * off
    p = sqrt(p2 / 6.0)
    least = q
    if p > 0.0:
        c00 = b00 / p
        c11 = b11 / p
        c22 = b22 / p
        c01 = a01 / p
        c02 = a02 / p
        c12 = a12 / p
        det = (c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02)
               + c02 * (c01 * c12 - c11 * c02))
        r = min(max(det / 2.0, -1.0), 1.0)
        phi = acos(r) / 3.0
        least = q + 2.0 * p * cos(phi + 2.0943951023931953)
    r0 = tack.vec3(a00 - least, a01, a02)
    r1 = tack.vec3(a01, a11 - least, a12)
    r2 = tack.vec3(a02, a12, a22 - least)
    x01 = r0.cross(r1)
    x02 = r0.cross(r2)
    x12 = r1.cross(r2)
    d01 = x01.dot(x01)
    d02 = x02.dot(x02)
    d12 = x12.dot(x12)
    best = x01
    most = d01
    if d02 > most:
        best = x02
        most = d02
    if d12 > most:
        best = x12
        most = d12
    axis = best * 0.0
    if most > 0.0:
        axis = best / sqrt(most)
    return axis


@tack.kernel
def _normals(grid, offsets, ids, vx, vy, vz, toward_point, normals):
    for q in range(normals.shape[0]):
        x = grid.positions[q]
        first = offsets[q]
        count = 0                     # the k nearest end in -1s where points ran out
        for s in range(first, offsets[q + 1]):
            if ids[s] >= 0:
                count += 1
        normal = x * 0.0
        if count >= 3:
            # Centred on the neighbourhood's mean, so the covariance is of offsets.
            mean = grid.offset(ids[first], x)
            for s in range(first + 1, first + count):
                mean += grid.offset(ids[s], x)
            mean = mean / count
            a00 = mean[0] * 0.0
            a01 = a00
            a02 = a00
            a11 = a00
            a12 = a00
            a22 = a00
            for s in range(first, first + count):
                d = grid.offset(ids[s], x) - mean
                a00 += d[0] * d[0]
                a01 += d[0] * d[1]
                a02 += d[0] * d[2]
                a11 += d[1] * d[1]
                a12 += d[1] * d[2]
                a22 += d[2] * d[2]
            # Scaled to its largest entry, so f32 neither overflows nor flushes.
            big = max(max(abs(a00), abs(a11)), max(abs(a22), max(abs(a01),
                                                              max(abs(a02), abs(a12)))))
            if big > 0.0:
                normal = _smallest_axis(a00 / big, a01 / big, a02 / big, a11 / big,
                                        a12 / big, a22 / big)
            toward = tack.vec3(vx, vy, vz)
            if toward_point == 1:
                toward = toward - x
            if normal.dot(toward) < 0.0:
                normal = -normal
        normals[q] = normal


@on_inputs_device
def point_normals(data, k=None, radius=None, viewpoint=None, direction=(0.0, 0.0, 1.0),
                  periodic=None, locator=None):
    """``data`` with ``normals``, each point's unit normal: the direction in which
    its neighbourhood -- its ``k`` nearest points (at most 32), itself among
    them, or the points within ``radius`` -- spreads least, from the
    eigenvectors of their covariance, as VTK's vtkPCANormalEstimation and
    PDAL's ``filters.normal``. A point with fewer than three neighbours, or
    whose neighbours lie on a line, gets a zero normal.

    A normal's sign is chosen to point toward ``viewpoint`` (a world position)
    when one is given, else along ``direction`` (up, by default: the outside
    of terrain). ``periodic`` (a box) takes neighbours across its faces.
    ``locator`` reuses a ``PointLocator`` of ``data``."""
    if (k is None) == (radius is None):
        raise ValueError("point_normals takes k or radius")
    if not isinstance(data.geometry.space, H1):
        raise TypeError("point_normals needs an H1 geometry: one position per point")
    locator = _locator(data, locator, periodic)
    n = data.num_points
    positions = locator.positions
    if k is not None:
        if not (isinstance(k, (int, np.integer)) and 1 <= k <= MAX_K):
            raise ValueError(f"k is a whole number from 1 to {MAX_K}, not {k!r}")
        nearest, _ = locator.nearest(int(k), positions)      # itself included
        # Rows of k already: CSR offsets 0, k, 2k, ..., stored nowhere.
        offsets = CountingArray(n + 1, start=0, step=int(k), dtype=data.id_dtype)
        ids = nearest.reshape((n * k,))
    else:
        if not radius >= 0:
            raise ValueError(f"radius must be at least zero, not {radius!r}")
        offsets, ids = locator.neighbors(float(radius), positions)
    if viewpoint is not None:
        toward = np.asarray(viewpoint, np.float64) - np.asarray(data.origin)
        toward_point = 1
    else:
        toward = np.asarray(direction, np.float64)
        toward_point = 0
    normals = tack.empty((n,), tack.vec(3, data.dtype))
    if n:
        _normals(locator.grid(), offsets, ids, *map(float, toward), toward_point, normals)
    out = DataSet(data.topology, data.geometry, fields=dict(data.fields),
                  sets=dict(data.sets), dtype=data.dtype, origin=data.origin)
    out.fields["normals"] = Field(H1(out), normals)
    from tack.dataset.ghosts import keep_marks

    return keep_marks(data, out)
