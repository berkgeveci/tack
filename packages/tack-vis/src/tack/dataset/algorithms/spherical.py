"""``spherical``: the geometry's Cartesian positions as spherical ``(r, theta, phi)`` -- ``theta`` from the z axis, ``phi`` in ``[0, 2pi)`` -- or back with ``inverse``, as Viskores' SphericalCoordinateTransform."""

from tack.dataset.algorithms._moving import _from_spherical, _recoordinate, _to_spherical

__all__ = ['spherical']


def spherical(data, inverse=False, fields=None):
    """The geometry's Cartesian positions as spherical ``(r, theta, phi)`` -- ``theta``
    from the z axis, ``phi`` in ``[0, 2pi)`` -- or back with ``inverse``, as
    Viskores' SphericalCoordinateTransform."""
    return _recoordinate(data, _from_spherical if inverse else _to_spherical, fields)
