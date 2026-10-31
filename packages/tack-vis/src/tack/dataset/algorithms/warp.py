"""``warp``: the geometry moved along ``direction`` -- a 3-vector field in the geometry's space (or its name), or one 3-vector for every point -- times ``scale`` and, if given, a scalar field ``scale_by`` in the same space (Viskores' Warp; VTK's vtkWarpVector and vtkWarpScalar)."""

import numpy as np

import tack
from tack.arrays import ConstantArray, size_of, width_of
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
def _warp_in_plane(positions, direction, factors, scale, out):
    # A 2-vector field (a 2D problem's displacement) moves x and y.
    for i in range(out.shape[0]):
        d = direction[i]
        f = factors[i] * scale
        p = positions[i]
        out[i] = [p[0] + d[0] * f, p[1] + d[1] * f, p[2]]


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
    space when the geometry is order-1 ``H1`` (``surface_normals``' normals). A
    2-vector field (a 2D problem's displacement) moves x and y.

    ``direction`` may also be an ``H1`` field of higher order than the geometry
    (a finite-element displacement, as MFEM's examples displace their meshes):
    the geometry is first carried into its space (``project``, exact for the
    lower order), so the output's geometry is curved, in that space; ``scale_by``
    is then in that space too."""
    geometry = data.geometry
    dtype = data.dtype
    if isinstance(direction, (str, Field)):
        direction = data.fields[direction] if isinstance(direction, str) else direction
        direction = _with_basis(data, direction)
        if width_of(direction.values) not in (2, 3):
            raise ValueError("direction must be a 3-vector field in the geometry's space "
                             "(or a 2-vector one, moving x and y)")
        if direction.space is not geometry.space:
            geometry = _lifted(data, geometry, direction.space)
    n = size_of(geometry.values)
    factors = (scale_by if not isinstance(scale_by, str) else data.fields[scale_by])
    if factors is None:
        factors = ConstantArray(1.0, n, dtype=dtype, device=data.device)
    else:
        factors = _with_basis(data, factors)
        if factors.space is not geometry.space or width_of(factors.values):
            raise ValueError("scale_by must be a scalar field in the geometry's space")
        factors = factors.values
    out = _vectors(n, dtype)
    if isinstance(direction, Field):
        if n:
            moving = _warp_by_field if width_of(direction.values) == 3 else _warp_in_plane
            moving(geometry.values, direction.values, factors,
                   float(scale), out)
    else:
        d = np.asarray(direction, float).reshape(-1)
        if d.size != 3:
            raise ValueError("a constant direction is three numbers")
        if n:
            _warp_constant(geometry.values, *map(float, d), factors,
                           float(scale), out)
    return _moved(data, out, fields, space=geometry.space)


def _lifted(data, geometry, space):
    """``geometry`` in ``space``, an ``H1`` space of this topology of no lower order:
    the geometry a displacement in ``space`` moves."""
    from tack.dataset.algorithms.project import project
    from tack.dataset.spaces import H1

    if not (isinstance(space, H1) and isinstance(geometry.space, H1)
            and space.topology is data.topology and space.order >= geometry.space.order):
        raise ValueError("direction must be a 3-vector field in the geometry's space, or "
                         "an H1 field of this topology of no lower order")
    return project(data, geometry, space)
