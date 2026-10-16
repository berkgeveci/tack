"""``slice``: where ``function`` (``tack.dataset.implicit``) is zero: a contour of its values on the geometry, carrying fields as ``contour`` does. ``slice_plane`` is this with a ``Plane``."""

from tack.dataset.algorithms.contour import contour
from tack.dataset.algorithms.implicit_values import implicit_values
from tack.dataset.implicit import Plane
from tack.runtime.dispatch import on_inputs_device

__all__ = ['slice', 'slice_plane']


@on_inputs_device
def slice(data, function, merge_points=True, fields=None):
    """Where ``function`` (``tack.dataset.implicit``) is zero: a contour of its values on
    the geometry, carrying fields as ``contour`` does. ``slice_plane`` is this
    with a ``Plane``."""

    return contour(data, implicit_values(data, function), 0.0, merge_points=merge_points,
                   fields=fields)


@on_inputs_device
def slice_plane(data, origin, normal, merge_points=True, fields=None):
    """The cut of the 3D cells by the plane through ``origin`` with ``normal`` (any
    length), as triangles: ``contour`` at zero of each position's signed distance
    to the plane along ``normal``. The distance is computed in the geometry's own
    space -- per point, or per cell corner for an ``L2`` geometry -- so the cut is
    the cells' own. ``slice`` with a ``Plane``."""
    return slice(data, Plane(origin, normal), merge_points=merge_points, fields=fields)
