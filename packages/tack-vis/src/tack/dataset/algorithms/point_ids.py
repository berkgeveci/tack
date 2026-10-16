"""``point_ids``: each point's id, as values on points: an implicit array, nothing stored."""

from tack.arrays import CountingArray
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values

__all__ = ['point_ids']


def point_ids(data):
    """Each point's id, as values on points: an implicit array, nothing stored."""
    return Field(Values(data, "points"), CountingArray(data.num_points, dtype=data.id_dtype))
