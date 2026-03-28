"""Cell average filter -- average point data to cell data."""

import tack
from ..arrays import AOSArray


@tack.kernel
def _cell_avg_kernel(cell_set: tack.template(), point_data, cell_data, n_cells):
    for c in range(n_cells):
        total = 0.0
        for v in range(cell_set.points_per_cell):
            total = total + point_data[cell_set.get_point_id(c, v)]
        cell_data[c] = total / float(cell_set.points_per_cell)


def cell_average(dataset, point_array_name, cell_array_name=None):
    """Average a point scalar to cells using the dataset's cell set.

    Works with any cell set type (structured, explicit) and any
    points-per-cell count -- the kernel compiles to the right loop
    count via the template system.
    """
    if cell_array_name is None:
        cell_array_name = point_array_name

    pt_arr = dataset.get_point_array(point_array_name)
    out_field = tack.field(dtype=tack.f32, shape=(dataset.num_cells,))
    _cell_avg_kernel(dataset.cell_set, pt_arr.data, out_field, dataset.num_cells)
    dataset.add_cell_array(cell_array_name, AOSArray(out_field))
    return dataset
