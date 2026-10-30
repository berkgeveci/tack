"""Moving a dataset's geometry, behind ``warp``, ``transform``,
``cylindrical`` and ``spherical``."""

import tack
from tack.arrays import materialize, size_of
from tack.dataset.algorithms._common import _vectors
from tack.dataset.carry import selected
from tack.dataset.dataset import DataSet, Field


def _moved(data, positions, fields=None, space=None):
    """``data`` with the geometry's values replaced (in ``space``, by default the
    geometry's): same topology and sets, and the fields ``fields`` selects (all by
    default)."""
    from tack.dataset.ghosts import keep_marks

    fields = {name: data.fields[name] for name in selected(data, fields)}
    space = data.geometry.space if space is None else space
    return keep_marks(data, DataSet(data.topology, Field(space, positions),
                                    fields=fields, sets=dict(data.sets)))


@tack.kernel
def _shift(positions, out, x, y, z):
    for i in range(out.shape[0]):
        out[i] = positions[i] + [x, y, z]


def _shifted(data, offset):
    """``data``'s geometry values plus ``offset`` (three floats, rounded to the
    geometry's type once): a new array of positions."""
    return _shifted_values(data.geometry.values, data.dtype, offset)


def _shifted_values(values, dtype, offset):
    """Positions (an array of 3-vectors of ``dtype``) plus ``offset``: a new array."""
    n = size_of(values)
    out = _vectors(n, dtype)
    if n:
        _shift(materialize(values), out, *(float(v) for v in offset))
    return out


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


def _recoordinate(data, kernel, fields=None):
    """New coordinates of each world position (``origin`` plus the value, added in
    the geometry's type): the output's origin is zero, and its spatial unit
    unknown, since an angle and lengths are no one unit."""
    values = data.geometry.values
    if any(data.origin):
        values = _shifted(data, data.origin)
    n = size_of(values)
    out = _vectors(n, data.dtype)
    if n:
        kernel(materialize(values), out)
    out = _moved(data, out, fields)
    out.origin = None
    out._geometry = Field(out.geometry.space, out.geometry.values)
    return out
