"""``extract_cells``: the cells with the given ids (a host array, an integer field, or a cell set), with only the points they use; cells keep their order, not the order given."""

import numpy as np

import tack
from tack.dataset import ids
from tack.dataset.algorithms._extract import _keep_cells

__all__ = ['extract_cells']


@tack.kernel
def _flag_ids(ids, flags):
    for i in range(ids.shape[0]):
        flags[ids[i]] = 1


def extract_cells(data, cells, fields=None):
    """The cells with the given ids (a host array, an integer field, or a cell
    ``Set`` or its name in ``data.sets``), with only the points they use; cells
    keep their order, not the order given."""
    from tack.dataset.sets import Set

    if isinstance(cells, str):
        cells = data.sets[cells]
    if isinstance(cells, Set):
        if cells.on != "cells" or cells.topology is not data.topology:
            raise ValueError(f"extract_cells takes a set of this dataset's cells, not a "
                             f"set of {cells.on}" + ("" if cells.topology is data.topology
                                                    else " of another topology"))
        cells = cells.ids
    if not hasattr(cells, "to_numpy"):
        values = np.asarray(cells).reshape(-1)
        if values.size and not np.issubdtype(values.dtype, np.integer):
            raise TypeError("cell ids must be integers")
        cells = values
    chosen = ids.as_ids(cells, data.id_dtype)
    keep = tack.zeros(tack.i32, (data.num_cells,))
    if chosen.shape[0]:
        bad = chosen.to_numpy()
        if bad.min() < 0 or bad.max() >= data.num_cells:
            raise IndexError(f"cell ids must lie in [0, {data.num_cells})")
        _flag_ids(chosen, keep)
    return _keep_cells(data, keep, fields)
