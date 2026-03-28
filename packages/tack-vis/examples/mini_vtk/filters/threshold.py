"""Threshold filter -- extract cells where a scalar is within a range."""

import numpy as np
import tack
from ..arrays import AOSArray
from ..cellsets import CellSetExplicit
from ..dataset import Dataset


@tack.kernel
def _classify(cell_data, mask, lo, hi, n):
    for i in range(n):
        v = cell_data[i]
        m = 0
        if v >= lo:
            if v <= hi:
                m = 1
        mask[i] = m


@tack.kernel
def _scatter(cell_set: tack.template(), mask, offsets, out_conn, n):
    for c in range(n):
        if mask[c] == 1:
            for v in range(cell_set.points_per_cell):
                out_conn[offsets[c] * cell_set.points_per_cell + v] = cell_set.get_point_id(c, v)


def threshold(dataset, cell_array_name, lo, hi):
    """Extract cells where a cell scalar is in [lo, hi].

    Returns a new Dataset with an explicit cell set containing only the
    qualifying cells. Point arrays are shared (not copied) since point
    indices remain valid.

    Works with any cell set type and any points-per-cell count.
    """
    n_cells = dataset.num_cells
    ppc = dataset.cell_set.points_per_cell
    cell_arr = dataset.get_cell_array(cell_array_name)

    # Pass 1: classify
    mask = tack.field(dtype=tack.i32, shape=(n_cells,))
    _classify(cell_arr.data, mask, lo, hi, n_cells)

    # Exclusive scan for scatter offsets
    offsets = tack.field(dtype=tack.i32, shape=(n_cells,))
    mask_np = mask.to_numpy()
    offsets_np = np.zeros(n_cells, dtype=np.int32)
    offsets_np[1:] = np.cumsum(mask_np[:-1])
    offsets.from_numpy(offsets_np)
    n_out = int(offsets_np[-1] + mask_np[-1])

    if n_out == 0:
        conn = tack.field(dtype=tack.i32, shape=(1,))
        cell_set = CellSetExplicit(conn, ppc)
        return Dataset(dataset.coordinates, cell_set, dataset.num_points, 0)

    # Pass 2: scatter qualifying cells
    out_conn = tack.field(dtype=tack.i32, shape=(n_out * ppc,))
    _scatter(dataset.cell_set, mask, offsets, out_conn, n_cells)

    cell_set = CellSetExplicit(out_conn, ppc)
    result = Dataset(dataset.coordinates, cell_set, dataset.num_points, n_out)

    for name, arr in dataset.point_data.items():
        result.add_point_array(name, arr)

    return result
