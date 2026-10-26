"""``point_ids``: each point's id, as values on points: an implicit array, nothing stored."""

from tack.arrays import CountingArray
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['point_ids']


@on_inputs_device
def point_ids(data):
    """Each point's id, as values on points: an implicit array, nothing stored.

    Stored as a field, it reaches every filter's output as the id of the input
    point each kept point is (``cell_ids`` does the same for cells)."""
    return Field(Values(data, "points"), CountingArray(data.num_points, dtype=data.id_dtype))
