"""Threshold filter — extract cells where a scalar is within a range."""

import numpy as np
import pgc
from ..arrays import AOSArray, AOSTupleArray
from ..cellsets import CellSetExplicitHex
from ..dataset import Dataset


@pgc.kernel
def _classify(cell_data, mask, lo, hi, n):
    for i in range(n):
        v = cell_data[i]
        m = 0
        if v >= lo:
            if v <= hi:
                m = 1
        mask[i] = m


@pgc.kernel
def _scatter_hex(cell_set: pgc.template(), mask, offsets, out_conn, n):
    for c in range(n):
        if mask[c] == 1:
            p0, p1, p2, p3, p4, p5, p6, p7 = cell_set.get_cell_points(c)
            base = offsets[c] * 8
            out_conn[base] = p0
            out_conn[base + 1] = p1
            out_conn[base + 2] = p2
            out_conn[base + 3] = p3
            out_conn[base + 4] = p4
            out_conn[base + 5] = p5
            out_conn[base + 6] = p6
            out_conn[base + 7] = p7


def threshold(dataset, cell_array_name, lo, hi):
    """Extract cells where a cell scalar is in [lo, hi].

    Returns a new Dataset with an explicit cell set containing only the
    qualifying cells. Point arrays are shared (not copied) since point
    indices remain valid.

    Args:
        dataset: input Dataset
        cell_array_name: name of the cell data array to threshold on
        lo, hi: scalar range [lo, hi] inclusive

    Returns:
        New Dataset with the extracted cells.
    """
    n_cells = dataset.num_cells
    cell_arr = dataset.get_cell_array(cell_array_name)

    # Pass 1: classify
    mask = pgc.field(dtype=pgc.i32, shape=(n_cells,))
    _classify(cell_arr.data, mask, lo, hi, n_cells)

    # Exclusive scan for scatter offsets
    offsets = pgc.field(dtype=pgc.i32, shape=(n_cells,))
    mask_np = mask.to_numpy()
    offsets_np = np.zeros(n_cells, dtype=np.int32)
    offsets_np[1:] = np.cumsum(mask_np[:-1])
    offsets.from_numpy(offsets_np)
    n_out = int(offsets_np[-1] + mask_np[-1])

    if n_out == 0:
        # Empty result
        conn = pgc.field(dtype=pgc.i32, shape=(1,))
        cell_set = CellSetExplicitHex(conn)
        return Dataset(dataset.coordinates, cell_set, dataset.num_points, 0)

    # Pass 2: scatter qualifying cells
    out_conn = pgc.field(dtype=pgc.i32, shape=(n_out * 8,))
    _scatter_hex(dataset.cell_set, mask, offsets, out_conn, n_cells)

    cell_set = CellSetExplicitHex(out_conn)
    result = Dataset(dataset.coordinates, cell_set, dataset.num_points, n_out)

    # Share point arrays (indices are still valid)
    for name, arr in dataset.point_data.items():
        result.add_point_array(name, arr)

    return result
