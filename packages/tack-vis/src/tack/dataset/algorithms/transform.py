# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``transform``: the geometry under an affine map: a 4x4 matrix (its last row 0 0 0 1) or 3x4, applied to each position as a column (Viskores' PointTransform, VTK's vtkTransformFilter)."""

import numpy as np

import tack
from tack.arrays import UniformCoordinates, size_of
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


@tack.kernel
def _directions(values, a00, a01, a02, a10, a11, a12, a20, a21, a22, unit, out):
    # Each 3-vector under a 3x3 map; with ``unit`` made unit length again (a zero
    # one stays zero).
    for i in range(out.shape[0]):
        v = values[i]
        w = [a00 * v[0] + a01 * v[1] + a02 * v[2],
             a10 * v[0] + a11 * v[1] + a12 * v[2],
             a20 * v[0] + a21 * v[1] + a22 * v[2]]
        if unit != 0:
            length = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
            if length > 0.0:
                w = w / length
        out[i] = w


def _mapped(data, key, linear, unit):
    """Field ``key`` of ``data`` with each 3-vector under ``linear`` (3x3), made unit
    length again if ``unit``: a new field in the same space and unit."""
    from tack.arrays import dtype_of, materialize, width_of
    from tack.dataset.dataset import Field

    field = data.fields[key]
    values = materialize(field.values)
    if width_of(values) != 3:
        raise TypeError(f"transform maps 3-vector fields; {key!r} is not one")
    n = size_of(values)
    kind = dtype_of(values)
    out = _vectors(n, getattr(kind, "scalar_dtype", kind))
    if n:
        _directions(values, *map(float, np.asarray(linear).reshape(-1)), int(unit), out)
    return Field(field.space, out, field.unit)


@on_inputs_device
def transform(data, matrix, fields=None, vectors=(), normals=()):
    """The geometry under an affine map: a 4x4 matrix (its last row 0 0 0 1) or 3x4,
    applied to each position as a column (Viskores' PointTransform, VTK's
    vtkTransformFilter). On a dataset with an origin the map moves the origin,
    in double precision, and only its linear part acts on the geometry's values,
    so a rotation far from zero loses nothing.

    An image stays an image: the map is folded into its origin and direction
    (``UniformCoordinates``; the direction may be any invertible matrix, a shear
    too), in double precision, and no position is stored -- as VTK's image data
    with a direction matrix holds it. A singular map (a projection) gives the
    positions themselves.

    Fields are carried as they are, but ``vectors`` and ``normals`` (field names,
    or one): a vector -- a velocity, a displacement -- turns with the map, its
    linear part applied to it, and a normal takes the inverse transpose of that
    part, made unit length again, so it stays normal to the moved surface. VTK's
    vtkTransformFilter does so to the active vectors and normals, which the VTK
    algorithm ``Transform`` takes by default; Tack keeps no roles, so here they
    are named. Each is a 3-vector field; normals need an invertible map."""
    m = np.asarray(matrix, float)
    if m.shape == (4, 4):
        if not np.allclose(m[3], [0, 0, 0, 1]):
            raise ValueError("a projective matrix (last row not 0 0 0 1) is not affine")
        m = m[:3]
    if m.shape != (3, 4):
        raise ValueError(f"transform takes a 4x4 or 3x4 matrix, not {m.shape}")
    out = _placed(data, m, fields)
    from tack.dataset.dataset import is_field_key

    vectors = [vectors] if is_field_key(vectors) else list(vectors or ())
    normals = [normals] if is_field_key(normals) else list(normals or ())
    linear = m[:, :3]
    for key in vectors:
        out.fields[key] = _mapped(out, key, linear, unit=False)
    if normals:
        if np.linalg.det(linear) == 0.0:
            raise ValueError("a singular map (a projection) has no normals to give")
        inverse_transpose = np.linalg.inv(linear).T
        for key in normals:
            out.fields[key] = _mapped(out, key, inverse_transpose, unit=True)
    return out


def _placed(data, m, fields):
    """``data`` with its geometry under the affine map ``m`` (3x4), its fields
    carried as they are."""
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
        _affine(data.geometry.values, *map(float, m.reshape(-1)), out)
    out = _moved(data, out, fields)
    if origin is not None:
        out.origin = origin
    return out
