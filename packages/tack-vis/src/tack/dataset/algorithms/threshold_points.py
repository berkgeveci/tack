"""``threshold_points``: the points whose ``field`` (point data: ``H1``, values on points) lies in ``[lower, upper]``, as vertex cells."""

from tack import arrays
from tack.dataset.algorithms._common import _field, _in_unit_of
from tack.dataset.algorithms._extract import _keep_points, _point_flags
from tack.dataset.carry import _point_values
from tack.dataset.spaces import H1, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['threshold_points']


@on_inputs_device
def threshold_points(data, field, lower, upper, fields=None):
    """The points whose ``field`` (point data: ``H1``, values on points) lies in
    ``[lower, upper]``, as vertex cells.

    The point twin of ``threshold``: the points as vertex cells, with point fields
    carried and cell fields left behind (a vertex cell has no cell of the input)."""
    field = _field(data, field)
    lower, upper = (_in_unit_of(field, v, "threshold_points' bounds") for v in (lower, upper))
    space = field.space
    if not (isinstance(space, H1) or (isinstance(space, Values) and space.on == "points")):
        raise TypeError(f"threshold_points needs point data, not {space!r}")
    if arrays.width_of(field.values):
        raise TypeError("threshold_points needs a scalar field")
    values = _point_values(field.values, data.num_points)
    return _keep_points(data, _point_flags(data, values, lower, upper), fields)
