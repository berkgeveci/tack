"""``warp``: the geometry moved along ``direction`` -- a 3-vector field in the geometry's space (or its name), or one 3-vector for every point -- times ``scale`` and, if given, a scalar field ``scale_by`` in the same space (Viskores' Warp; VTK's vtkWarpVector and vtkWarpScalar)."""

import numpy as np

import tack
from tack.arrays import materialize, size_of, width_of
from tack.dataset.algorithms._common import _vectors, _with_basis
from tack.dataset.algorithms._moving import _moved
from tack.dataset.dataset import Field
from tack.runtime.dispatch import on_inputs_device

__all__ = ['warp']


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


@on_inputs_device
def warp(data, direction, scale=1.0, scale_by=None, fields=None):
    """The geometry moved along ``direction`` -- a 3-vector field in the geometry's
    space (or its name), or one 3-vector for every point -- times ``scale``
    and, if given, a scalar field ``scale_by`` in the same space (Viskores'
    Warp; VTK's vtkWarpVector and vtkWarpScalar). Values on points are that
    space when the geometry is order-1 ``H1`` (``surface_normals``' normals)."""
    geometry = data.geometry
    n = size_of(geometry.values)
    dtype = data.dtype
    factors = (scale_by if not isinstance(scale_by, str) else data.fields[scale_by])
    if factors is None:
        factors = tack.full((n,), 1.0, dtype) if n else tack.empty((0,), dtype)
    else:
        factors = _with_basis(data, factors)
        if factors.space is not geometry.space or width_of(factors.values):
            raise ValueError("scale_by must be a scalar field in the geometry's space")
        factors = materialize(factors.values)
    out = _vectors(n, dtype)
    if isinstance(direction, (str, Field)):
        direction = data.fields[direction] if isinstance(direction, str) else direction
        direction = _with_basis(data, direction)
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
    return _moved(data, out, fields)
