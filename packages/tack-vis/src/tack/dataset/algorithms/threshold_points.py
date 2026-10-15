"""``threshold_points``: the points whose ``field`` (point data: ``H1``, values on points) lies in ``[lower, upper]``, as vertex cells."""

from tack import arrays
from tack.dataset.algorithms._common import _field
from tack.dataset.algorithms._extract import _keep_points, _point_flags
from tack.dataset.carry import _point_values
from tack.dataset.spaces import H1, Values

__all__ = ['threshold_points']


def threshold_points(data, field, lower, upper, fields=None):
    """The points whose ``field`` (point data: ``H1``, values on points) lies in
    ``[lower, upper]``, as vertex cells."""
    field = _field(data, field)
    space = field.space
    if not (isinstance(space, H1) or (isinstance(space, Values) and space.on == "points")):
        raise TypeError(f"threshold_points needs point data, not {space!r}")
    if arrays.width_of(field.values):
        raise TypeError("threshold_points needs a scalar field")
    values = _point_values(field.values, data.num_points)
    return _keep_points(data, _point_flags(data, values, lower, upper), fields)
