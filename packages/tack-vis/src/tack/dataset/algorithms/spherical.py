"""``spherical``: the geometry's Cartesian positions as spherical ``(r, theta, phi)`` -- ``theta`` from the z axis, ``phi`` in ``[0, 2pi)`` -- or back with ``inverse``, as Viskores' SphericalCoordinateTransform."""

from tack.dataset.algorithms._moving import _from_spherical, _recoordinate, _to_spherical
from tack.runtime.dispatch import on_inputs_device

__all__ = ['spherical']


@on_inputs_device
def spherical(data, inverse=False, fields=None):
    """The geometry's Cartesian positions as spherical ``(r, theta, phi)`` -- ``theta``
    from the z axis, ``phi`` in ``[0, 2pi)`` -- or back with ``inverse``, as
    Viskores' SphericalCoordinateTransform. The positions are the world's (the
    origin added), and the result's origin is zero and its spatial unit
    unknown, an angle and lengths being no one unit. Fields come along."""
    return _recoordinate(data, _from_spherical if inverse else _to_spherical, fields)
