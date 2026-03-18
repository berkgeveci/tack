"""Cell average filter — average point data to cell data."""

import pgc
from ..arrays import AOSArray


@pgc.kernel
def _cell_avg_kernel(cell_set: pgc.template(), point_data, cell_data, n_cells):
    for c in range(n_cells):
        p0, p1, p2, p3, p4, p5, p6, p7 = cell_set.get_cell_points(c)
        avg = (point_data[p0] + point_data[p1] + point_data[p2] + point_data[p3]
             + point_data[p4] + point_data[p5] + point_data[p6] + point_data[p7]) * 0.125
        cell_data[c] = avg


def cell_average(dataset, point_array_name, cell_array_name=None):
    """Average a point scalar to cells using the dataset's cell set.

    Args:
        dataset: input Dataset
        point_array_name: name of the point data array to average
        cell_array_name: name for the output cell array (defaults to input name)

    Returns:
        The dataset with the new cell array added.
    """
    if cell_array_name is None:
        cell_array_name = point_array_name

    pt_arr = dataset.get_point_array(point_array_name)
    out_field = pgc.field(dtype=pgc.f32, shape=(dataset.num_cells,))
    _cell_avg_kernel(dataset.cell_set, pt_arr.data, out_field, dataset.num_cells)
    dataset.add_cell_array(cell_array_name, AOSArray(out_field))
    return dataset
