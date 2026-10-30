"""``transform``: the geometry under an affine map: a 4x4 matrix (its last row 0 0 0 1) or 3x4, applied to each position as a column (Viskores' PointTransform, VTK's vtkTransformFilter)."""

import numpy as np

import tack
from tack.arrays import UniformCoordinates, materialize, size_of
from tack.dataset.algorithms._common import _vectors
from tack.dataset.algorithms._moving import _moved
from tack.runtime.dispatch import on_inputs_device

__all__ = ['transform']


@tack.kernel
def _affine(positions, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, out):
    for i in range(out.shape[0]):
        p = positions[i]
        out[i] = [m00 * p[0] + m01 * p[1] + m02 * p[2] + m03,
                  m10 * p[0] + m11 * p[1] + m12 * p[2] + m13,
                  m20 * p[0] + m21 * p[1] + m22 * p[2] + m23]


@on_inputs_device
def transform(data, matrix, fields=None):
    """The geometry under an affine map: a 4x4 matrix (its last row 0 0 0 1) or 3x4,
    applied to each position as a column (Viskores' PointTransform, VTK's
    vtkTransformFilter). On a dataset with an origin the map moves the origin,
    in double precision, and only its linear part acts on the geometry's values,
    so a rotation far from zero loses nothing.

    An image stays an image: the map is folded into its origin and direction
    (``UniformCoordinates``; the direction may be any invertible matrix, a shear
    too), in double precision, and no position is stored -- as VTK's image data
    with a direction matrix holds it. A singular map (a projection) gives the
    positions themselves."""
    m = np.asarray(matrix, float)
    if m.shape == (4, 4):
        if not np.allclose(m[3], [0, 0, 0, 1]):
            raise ValueError("a projective matrix (last row not 0 0 0 1) is not affine")
        m = m[:3]
    if m.shape != (3, 4):
        raise ValueError(f"transform takes a 4x4 or 3x4 matrix, not {m.shape}")
    origin = None
    if any(data.origin):
        origin = m[:, :3] @ np.asarray(data.origin) + m[:, 3]
        m = np.hstack([m[:, :3], np.zeros((3, 1))])
    values = data.geometry.values
    # A singular map (a projection) flattens the image: its positions are stored.
    if isinstance(values, UniformCoordinates) and np.linalg.det(m[:, :3]) != 0.0:
        image = UniformCoordinates(values.dims, m[:, :3] @ np.asarray(values._origin) + m[:, 3],
                                   values._spacing, m[:, :3] @ values._matrix,
                                   dtype=data.dtype, device=data.device)
        out = _moved(data, image, fields)
        if origin is not None:
            out.origin = origin
        return out
    n = size_of(values)
    out = _vectors(n, data.dtype)
    if n:
        _affine(materialize(data.geometry.values), *map(float, m.reshape(-1)), out)
    out = _moved(data, out, fields)
    if origin is not None:
        out.origin = origin
    return out
