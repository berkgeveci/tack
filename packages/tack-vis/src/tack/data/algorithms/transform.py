"""``transform``: the geometry under an affine map: a 4x4 matrix (its last row 0 0 0 1) or 3x4, applied to each position as a column (Viskores' PointTransform, VTK's vtkTransformFilter)."""

import numpy as np

import tack
from tack.data.algorithms._common import _vectors
from tack.data.algorithms._moving import _moved
from tack.data.arrays import materialize, size_of

__all__ = ['transform']


@tack.kernel
def _affine(positions, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, out):
    for i in range(out.shape[0]):
        p = positions[i]
        out[i] = [m00 * p[0] + m01 * p[1] + m02 * p[2] + m03,
                  m10 * p[0] + m11 * p[1] + m12 * p[2] + m13,
                  m20 * p[0] + m21 * p[1] + m22 * p[2] + m23]


def transform(data, matrix, fields=None):
    """The geometry under an affine map: a 4x4 matrix (its last row 0 0 0 1) or 3x4,
    applied to each position as a column (Viskores' PointTransform, VTK's
    vtkTransformFilter)."""
    m = np.asarray(matrix, float)
    if m.shape == (4, 4):
        if not np.allclose(m[3], [0, 0, 0, 1]):
            raise ValueError("a projective matrix (last row not 0 0 0 1) is not affine")
        m = m[:3]
    if m.shape != (3, 4):
        raise ValueError(f"transform takes a 4x4 or 3x4 matrix, not {m.shape}")
    n = size_of(data.geometry.values)
    out = _vectors(n, data.dtype)
    if n:
        _affine(materialize(data.geometry.values), *map(float, m.reshape(-1)), out)
    return _moved(data, out, fields)
