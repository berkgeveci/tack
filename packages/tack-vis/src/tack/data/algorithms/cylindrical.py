"""``cylindrical``: the geometry's Cartesian positions as cylindrical ``(r, theta, z)``, or back with ``inverse`` -- Viskores' CylindricalCoordinateTransform: ``theta`` from ``asin(y / r)``, in ``[-pi/2, 3pi/2)``."""

from tack.data.algorithms._moving import _from_cylindrical, _recoordinate, _to_cylindrical

__all__ = ['cylindrical']


def cylindrical(data, inverse=False):
    """The geometry's Cartesian positions as cylindrical ``(r, theta, z)``, or back
    with ``inverse`` -- Viskores' CylindricalCoordinateTransform: ``theta`` from
    ``asin(y / r)``, in ``[-pi/2, 3pi/2)``."""
    return _recoordinate(data, _from_cylindrical if inverse else _to_cylindrical)
