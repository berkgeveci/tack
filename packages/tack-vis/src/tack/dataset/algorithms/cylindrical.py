"""``cylindrical``: the geometry's Cartesian positions as cylindrical ``(r, theta, z)``, or back with ``inverse`` -- Viskores' CylindricalCoordinateTransform: ``theta`` from ``asin(y / r)``, in ``[-pi/2, 3pi/2)``."""

from tack.dataset.algorithms._moving import _from_cylindrical, _recoordinate, _to_cylindrical
from tack.runtime.dispatch import on_inputs_device

__all__ = ['cylindrical']


@on_inputs_device
def cylindrical(data, inverse=False, fields=None):
    """The geometry's Cartesian positions as cylindrical ``(r, theta, z)``, or back
    with ``inverse`` -- Viskores' CylindricalCoordinateTransform: ``theta`` from
    ``asin(y / r)``, in ``[-pi/2, 3pi/2)``. The positions are the world's
    (the origin added), and the result's origin is zero and its spatial unit
    unknown, an angle and lengths being no one unit. Fields come along."""
    return _recoordinate(data, _from_cylindrical if inverse else _to_cylindrical, fields)
