"""``cell_ids``: each cell's id, as values on cells: an implicit array, nothing stored."""

from tack.data.arrays import CountingArray
from tack.data.dataset import Field
from tack.data.spaces import Values

__all__ = ['cell_ids']


def cell_ids(data):
    """Each cell's id, as values on cells: an implicit array, nothing stored."""
    return Field(Values(data, "cells"), CountingArray(data.num_cells, dtype=data.id_dtype))
