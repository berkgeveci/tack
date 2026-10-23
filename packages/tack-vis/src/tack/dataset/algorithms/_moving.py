"""Moving a dataset's geometry, behind ``warp``, ``transform``,
``cylindrical`` and ``spherical``."""

import tack
from tack.arrays import materialize, size_of
from tack.dataset.algorithms._common import _vectors
from tack.dataset.carry import selected
from tack.dataset.dataset import DataSet, Field


def _moved(data, positions, fields=None):
    """``data`` with the geometry's values replaced: same topology and sets, and the
    fields ``fields`` selects (all by default)."""
    from tack.dataset.ghosts import keep_marks

    fields = {name: data.fields[name] for name in selected(data, fields)}
    return keep_marks(data, DataSet(data.topology, Field(data.geometry.space, positions),
                                    fields=fields, sets=dict(data.sets)))


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
    n = size_of(data.geometry.values)
    out = _vectors(n, data.dtype)
    if n:
        kernel(materialize(data.geometry.values), out)
    return _moved(data, out, fields)
