"""Field and geometry transforms, and vector analysis: Viskores' field_transform
and vector_analysis filters, in Viskores' conventions.

Functions of fields return a field in the same space (``magnitude``, ``dot``,
``cross``, ``composite``, ``log_values``); ``point_elevation``, ``point_ids`` and
``cell_ids`` make one from the dataset. ``warp``, ``transform``,
``cylindrical`` and ``spherical`` move the geometry and return a dataset on
the same topology, every field kept. All work on any space -- on an ``L2``
geometry each cell's own corners move.
"""

import math

import numpy as np

import tack
from tack.data import arrays
from tack.data.arrays import CountingArray, materialize, size_of, width_of
from tack.data.dataset import DataSet, Field
from tack.data.spaces import Values

__all__ = ["cell_ids", "composite", "cross", "cylindrical", "dot", "log_values", "magnitude",
           "point_elevation", "point_ids", "spherical", "transform", "warp"]


def _vectors(n, dtype, width=3):
    return tack.Vector.field(width, dtype, shape=(n,))


def _scalars(n, dtype):
    return tack.field(dtype, shape=(n,))


def _same_space(*fields):
    space = fields[0].space
    if any(f.space is not space for f in fields[1:]):
        raise ValueError("the fields must be in one space")
    n = size_of(fields[0].values)
    return space, n


# ── Vector analysis ─────────────────────────────────────────────────

@tack.kernel
def _magnitudes(values, out):
    for i in range(out.shape[0]):
        out[i] = values[i].norm()


def magnitude(field):
    """Each vector's length (Viskores' VectorMagnitude)."""
    if not width_of(field.values):
        raise TypeError("magnitude takes a vector field")
    n = size_of(field.values)
    out = _scalars(n, arrays.dtype_of(field.values))
    if n:
        _magnitudes(materialize(field.values), out)
    return Field(field.space, out)


@tack.kernel
def _dots(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i].dot(b[i])


def dot(a, b):
    """``a . b`` at every value of two vector fields of one width and space."""
    space, n = _same_space(a, b)
    if not width_of(a.values) or width_of(a.values) != width_of(b.values):
        raise TypeError("dot takes two vector fields of one width")
    out = _scalars(n, arrays.dtype_of(a.values))
    if n:
        _dots(materialize(a.values), materialize(b.values), out)
    return Field(space, out)


@tack.kernel
def _crosses(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i].cross(b[i])


def cross(a, b):
    """``a x b`` at every value of two 3-vector fields in one space."""
    space, n = _same_space(a, b)
    if width_of(a.values) != 3 or width_of(b.values) != 3:
        raise TypeError("cross takes two 3-vector fields")
    out = _vectors(n, arrays.dtype_of(a.values))
    if n:
        _crosses(materialize(a.values), materialize(b.values), out)
    return Field(space, out)


@tack.kernel
def _set_component(values, component, out):
    for i in range(values.shape[0]):
        out[i][component] = values[i]


def composite(*fields):
    """Scalar fields of one space as the components of one vector field (Viskores'
    CompositeVectors), in the order given."""
    if len(fields) < 2:
        raise ValueError("composite takes at least two scalar fields")
    space, n = _same_space(*fields)
    if any(width_of(f.values) for f in fields):
        raise TypeError("composite takes scalar fields")
    out = _vectors(n, arrays.dtype_of(fields[0].values), len(fields))
    for k, f in enumerate(fields):
        if n:
            _set_component(materialize(f.values), k, out)
    return Field(space, out)


# ── Field transforms ────────────────────────────────────────────────

@tack.kernel
def _logs(values, scale, smallest, out):
    for i in range(out.shape[0]):
        out[i] = log(max(values[i], smallest)) * scale


def log_values(field, base=math.e, min_value=None):
    """The logarithm of a scalar field in ``base`` (e, 2, 10, or any), values below
    ``min_value`` -- by default the smallest positive normal float, as in
    Viskores' LogValues -- taken as ``min_value``."""
    if width_of(field.values):
        raise TypeError("log_values takes a scalar field")
    if not base > 0 or base == 1:
        raise ValueError(f"no logarithm in base {base}")
    dtype = arrays.dtype_of(field.values)
    if min_value is None:
        min_value = float(np.finfo(dtype.numpy_dtype).tiny)
    n = size_of(field.values)
    out = _scalars(n, dtype)
    if n:
        _logs(materialize(field.values), 1.0 / math.log(base), float(min_value), out)
    return Field(field.space, out)


@tack.kernel
def _elevations(positions, lx, ly, lz, dx, dy, dz, length2, low, span, out):
    for i in range(out.shape[0]):
        p = positions[i]
        s = ((p[0] - lx) * dx + (p[1] - ly) * dy + (p[2] - lz) * dz) / length2
        out[i] = low + min(max(s, 0.0), 1.0) * span


def point_elevation(data, low_point=(0, 0, 0), high_point=(0, 0, 1), range=(0.0, 1.0)):
    """Each geometry value's position along the line from ``low_point`` to
    ``high_point``, clamped to it and mapped to ``range`` (VTK's
    vtkElevationFilter, Viskores' PointElevation): a field in the geometry's
    space."""
    low = np.asarray(low_point, float)
    direction = np.asarray(high_point, float) - low
    length2 = float(direction @ direction)
    if length2 == 0:
        raise ValueError("the low and high points must differ")
    n = size_of(data.geometry.values)
    out = _scalars(n, data.dtype)
    if n:
        _elevations(materialize(data.geometry.values), *map(float, low), *map(float, direction),
                    length2, float(range[0]), float(range[1] - range[0]), out)
    return Field(data.geometry.space, out)


def point_ids(data):
    """Each point's id, as values on points: an implicit array, nothing stored."""
    return Field(Values(data, "points"), CountingArray(data.num_points, dtype=data.id_dtype))


def cell_ids(data):
    """Each cell's id, as values on cells: an implicit array, nothing stored."""
    return Field(Values(data, "cells"), CountingArray(data.num_cells, dtype=data.id_dtype))


# ── Moving the geometry ─────────────────────────────────────────────

def _moved(data, positions):
    """``data`` with the geometry's values replaced: same topology, sets and fields."""
    fields = {name: f for name, f in data.fields.items() if name != "shape"}
    return DataSet(data.topology, Field(data.geometry.space, positions), fields=fields,
                   sets=dict(data.sets))


@tack.kernel
def _warp_by_field(positions, direction, factors, scale, out):
    for i in range(out.shape[0]):
        out[i] = positions[i] + direction[i] * (factors[i] * scale)


@tack.kernel
def _warp_constant(positions, dx, dy, dz, factors, scale, out):
    for i in range(out.shape[0]):
        f = factors[i] * scale
        p = positions[i]
        out[i] = [p[0] + dx * f, p[1] + dy * f, p[2] + dz * f]


def warp(data, direction, scale=1.0, scale_by=None):
    """The geometry moved along ``direction`` -- a 3-vector field in the geometry's
    space (or its name), or one 3-vector for every point -- times ``scale``
    and, if given, a scalar field ``scale_by`` in the same space (Viskores'
    Warp; VTK's vtkWarpVector and vtkWarpScalar)."""
    geometry = data.geometry
    n = size_of(geometry.values)
    dtype = data.dtype
    factors = (scale_by if not isinstance(scale_by, str) else data.fields[scale_by])
    if factors is None:
        factors = tack.full(dtype, (n,), 1.0) if n else tack.field(dtype, shape=(0,))
    else:
        if factors.space is not geometry.space or width_of(factors.values):
            raise ValueError("scale_by must be a scalar field in the geometry's space")
        factors = materialize(factors.values)
    out = _vectors(n, dtype)
    if isinstance(direction, (str, Field)):
        direction = data.fields[direction] if isinstance(direction, str) else direction
        if direction.space is not geometry.space or width_of(direction.values) != 3:
            raise ValueError("direction must be a 3-vector field in the geometry's space")
        if n:
            _warp_by_field(materialize(geometry.values), materialize(direction.values),
                           factors, float(scale), out)
    else:
        d = np.asarray(direction, float).reshape(-1)
        if d.size != 3:
            raise ValueError("a constant direction is three numbers")
        if n:
            _warp_constant(materialize(geometry.values), *map(float, d), factors,
                           float(scale), out)
    return _moved(data, out)


@tack.kernel
def _affine(positions, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, out):
    for i in range(out.shape[0]):
        p = positions[i]
        out[i] = [m00 * p[0] + m01 * p[1] + m02 * p[2] + m03,
                  m10 * p[0] + m11 * p[1] + m12 * p[2] + m13,
                  m20 * p[0] + m21 * p[1] + m22 * p[2] + m23]


def transform(data, matrix):
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
    return _moved(data, out)


@tack.kernel
def _to_cylindrical(positions, out):
    for i in range(out.shape[0]):
        p = positions[i]
        r = sqrt(p[0] * p[0] + p[1] * p[1])
        theta = p[0] * 0.0
        if r > 0.0:
            theta = asin(p[1] / r)
            if p[0] < 0.0:
                theta = 3.141592653589793 - theta
        out[i] = [r, theta, p[2]]


@tack.kernel
def _from_cylindrical(positions, out):
    for i in range(out.shape[0]):
        p = positions[i]
        out[i] = [p[0] * cos(p[1]), p[0] * sin(p[1]), p[2]]


@tack.kernel
def _to_spherical(positions, out):
    for i in range(out.shape[0]):
        p = positions[i]
        r = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
        theta = p[0] * 0.0
        if r > 0.0:
            theta = acos(p[2] / r)
        phi = atan2(p[1], p[0])
        if phi < 0.0:
            phi = phi + 6.283185307179586
        out[i] = [r, theta, phi]


@tack.kernel
def _from_spherical(positions, out):
    for i in range(out.shape[0]):
        p = positions[i]
        s = sin(p[1])
        out[i] = [p[0] * s * cos(p[2]), p[0] * s * sin(p[2]), p[0] * cos(p[1])]


def _recoordinate(data, kernel):
    n = size_of(data.geometry.values)
    out = _vectors(n, data.dtype)
    if n:
        kernel(materialize(data.geometry.values), out)
    return _moved(data, out)


def cylindrical(data, inverse=False):
    """The geometry's Cartesian positions as cylindrical ``(r, theta, z)``, or back
    with ``inverse`` -- Viskores' CylindricalCoordinateTransform: ``theta`` from
    ``asin(y / r)``, in ``[-pi/2, 3pi/2)``."""
    return _recoordinate(data, _from_cylindrical if inverse else _to_cylindrical)


def spherical(data, inverse=False):
    """The geometry's Cartesian positions as spherical ``(r, theta, phi)`` -- ``theta``
    from the z axis, ``phi`` in ``[0, 2pi)`` -- or back with ``inverse``, as
    Viskores' SphericalCoordinateTransform."""
    return _recoordinate(data, _from_spherical if inverse else _to_spherical)
