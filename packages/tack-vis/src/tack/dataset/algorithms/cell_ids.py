"""``cell_ids``: each cell's id, as values on cells: an implicit array, nothing stored."""

from tack.arrays import CountingArray
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['cell_ids']


@on_inputs_device
def cell_ids(data):
    """Each cell's id, as values on cells: an implicit array, nothing stored.

    Stored as a field (``data.fields["source"] = cell_ids(data)``), it reaches
    every filter's output as the id of the input cell each output cell came
    from -- kept, a piece of, or a face of -- with no option on the filter."""
    return Field(Values(data, "cells"), CountingArray(data.num_cells, dtype=data.id_dtype))
