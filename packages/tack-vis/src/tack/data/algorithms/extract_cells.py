"""``extract_cells``: the cells with the given ids (a host array or an integer field), with only the points they use; cells keep their order, not the order given."""

import numpy as np

import tack
from tack.data import ids
from tack.data.algorithms._extract import _keep_cells

__all__ = ['extract_cells']


@tack.kernel
def _flag_ids(ids, flags):
    for i in range(ids.shape[0]):
        flags[ids[i]] = 1


def extract_cells(data, cells, fields=None):
    """The cells with the given ids (a host array or an integer field), with only the
    points they use; cells keep their order, not the order given."""
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
