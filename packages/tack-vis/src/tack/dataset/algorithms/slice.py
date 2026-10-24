"""``slice``: where ``function`` (``tack.dataset.implicit``) is zero: a contour of its values on the geometry, carrying fields as ``contour`` does. ``slice_plane`` is this with a ``Plane``."""

from tack.dataset.algorithms.contour import contour
from tack.dataset.algorithms.implicit_values import implicit_values
from tack.dataset.algorithms.project import project
from tack.dataset.implicit import Plane
from tack.dataset.spaces import H1
from tack.runtime.dispatch import on_inputs_device

__all__ = ['slice', 'slice_plane']


@on_inputs_device
def slice(data, function, merge_points=True, fields=None, value=0.0):
    """Where ``function`` (``tack.dataset.implicit``) is ``value`` (zero by default):
    a contour of its values on the geometry, carrying fields as ``contour``
    does. ``value`` may be a sequence, as ``contour``'s: every level in one
    pass, each output cell's in ``contour_value``. ``slice_plane`` is this
    with a ``Plane``. On a curved geometry (``H1`` above order 1) the cells are cut
    linearized at their corners, as ``contour`` would need its cells subdivided
    for the curve inside them."""
    values = implicit_values(data, function)
    if isinstance(values.space, H1) and values.space.order != 1:
        # The function's values at the points: projected into order-1 H1, a
        # continuous field keeps them. contour refuses order 2, so slicing a
        # curved dataset raised.
        values = project(data, values, H1(data))
    return contour(data, values, value, merge_points=merge_points, fields=fields)


@on_inputs_device
def slice_plane(data, origin, normal, merge_points=True, fields=None, value=0.0):
    """The cut of the 3D cells by the plane through ``origin`` with ``normal`` (any
    length), as triangles: ``contour`` at zero of each position's signed distance
    to the plane along ``normal``. The distance is computed in the geometry's own
    space -- per point, or per cell corner for an ``L2`` geometry -- so the cut is
    the cells' own. ``value``, a signed distance along the unit normal, or a
    sequence of them, gives parallel planes, all in one pass. ``slice`` with a
    ``Plane``."""
    return slice(data, Plane(origin, normal), merge_points=merge_points, fields=fields,
                 value=value)
