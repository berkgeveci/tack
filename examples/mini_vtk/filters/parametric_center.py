"""Parametric center filter — interpolate point data to cell centers."""

import pgc
from ..arrays import AOSArray


@pgc.kernel
def _center_scalar(cell_set: pgc.template(), cell_type: pgc.template(),
                   point_data, center_data, n_cells):
    """Interpolate a scalar field to cell centers using shape functions."""
    for c in range(n_cells):
        pc0 = cell_type.center(0)
        pc1 = cell_type.center(1)
        pc2 = cell_type.center(2)
        val = 0.0
        for v in range(cell_type.num_points):
            w = cell_type.weight(v, pc0, pc1, pc2)
            val = val + w * point_data[cell_set.get_point_id(c, v)]
        center_data[c] = val


@pgc.kernel
def _center_coords(cell_set: pgc.template(), cell_type: pgc.template(),
                   coords: pgc.template(), center_x, center_y, center_z, n_cells):
    """Interpolate coordinates to cell centers using shape functions."""
    for c in range(n_cells):
        pc0 = cell_type.center(0)
        pc1 = cell_type.center(1)
        pc2 = cell_type.center(2)
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for v in range(cell_type.num_points):
            w = cell_type.weight(v, pc0, pc1, pc2)
            pid = cell_set.get_point_id(c, v)
            cx = cx + w * coords.get_value(pid, 0)
            cy = cy + w * coords.get_value(pid, 1)
            cz = cz + w * coords.get_value(pid, 2)
        center_x[c] = cx
        center_y[c] = cy
        center_z[c] = cz


def parametric_center(dataset, cell_type, point_array_name=None, output_name=None):
    """Interpolate point data to cell centers using parametric shape functions.

    If point_array_name is given, interpolates that scalar to cells.
    If point_array_name is None, computes cell center coordinates.

    Args:
        dataset: input Dataset
        cell_type: a cell type instance (Hexahedron(), Tetrahedron(), Wedge())
        point_array_name: name of point scalar to interpolate (or None for coordinates)
        output_name: output cell array name

    Returns:
        The dataset with the new cell array(s) added.
    """
    n = dataset.num_cells

    if point_array_name is not None:
        # Interpolate a scalar field
        if output_name is None:
            output_name = point_array_name + "_center"
        pt_arr = dataset.get_point_array(point_array_name)
        out_field = pgc.field(dtype=pgc.f32, shape=(n,))
        _center_scalar(dataset.cell_set, cell_type, pt_arr.data, out_field, n)
        dataset.add_cell_array(output_name, AOSArray(out_field))
    else:
        # Interpolate coordinates → cell centers
        cx = pgc.field(dtype=pgc.f32, shape=(n,))
        cy = pgc.field(dtype=pgc.f32, shape=(n,))
        cz = pgc.field(dtype=pgc.f32, shape=(n,))
        _center_coords(dataset.cell_set, cell_type, dataset.coordinates,
                       cx, cy, cz, n)
        base = output_name or "center"
        dataset.add_cell_array(base + "_x", AOSArray(cx))
        dataset.add_cell_array(base + "_y", AOSArray(cy))
        dataset.add_cell_array(base + "_z", AOSArray(cz))

    return dataset
