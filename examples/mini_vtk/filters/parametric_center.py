"""Parametric center filter -- interpolate point data to cell centers."""

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
    """Interpolate coordinates to cell centers.

    Uses pgc.local_array to cache weights -- computed once per cell,
    reused for all 3 coordinate components.
    """
    for c in range(n_cells):
        pc0 = cell_type.center(0)
        pc1 = cell_type.center(1)
        pc2 = cell_type.center(2)

        # Cache weights and point IDs in per-thread local arrays
        w = pgc.local_array(pgc.f32, cell_type.num_points)
        pid = pgc.local_array(pgc.i32, cell_type.num_points)
        for v in range(cell_type.num_points):
            w[v] = cell_type.weight(v, pc0, pc1, pc2)
            pid[v] = cell_set.get_point_id(c, v)

        # Interpolate each component using cached weights
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for v in range(cell_type.num_points):
            cx = cx + w[v] * coords.get_value(pid[v], 0)
            cy = cy + w[v] * coords.get_value(pid[v], 1)
            cz = cz + w[v] * coords.get_value(pid[v], 2)
        center_x[c] = cx
        center_y[c] = cy
        center_z[c] = cz


@pgc.kernel
def _center_multi(cell_set: pgc.template(), cell_type: pgc.template(),
                  field1, field2, field3, out1, out2, out3, n_cells):
    """Interpolate 3 scalar fields to cell centers with cached weights.

    Demonstrates pgc.local_array: weights are computed once per cell
    and reused for all 3 fields, avoiding redundant shape function
    evaluations.
    """
    for c in range(n_cells):
        pc0 = cell_type.center(0)
        pc1 = cell_type.center(1)
        pc2 = cell_type.center(2)

        # Cache weights and point IDs once
        w = pgc.local_array(pgc.f32, cell_type.num_points)
        pid = pgc.local_array(pgc.i32, cell_type.num_points)
        for v in range(cell_type.num_points):
            w[v] = cell_type.weight(v, pc0, pc1, pc2)
            pid[v] = cell_set.get_point_id(c, v)

        # Interpolate all 3 fields with the same cached weights
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        for v in range(cell_type.num_points):
            v1 = v1 + w[v] * field1[pid[v]]
            v2 = v2 + w[v] * field2[pid[v]]
            v3 = v3 + w[v] * field3[pid[v]]
        out1[c] = v1
        out2[c] = v2
        out3[c] = v3


def parametric_center(dataset, cell_type, point_array_name=None, output_name=None):
    """Interpolate point data to cell centers using parametric shape functions.

    If point_array_name is given, interpolates that scalar to cells.
    If point_array_name is None, computes cell center coordinates.

    Args:
        dataset: input Dataset
        cell_type: a cell type instance (Hexahedron(), Tetrahedron(), Wedge())
        point_array_name: name of point scalar to interpolate (or None for coordinates)
        output_name: output cell array name
    """
    n = dataset.num_cells

    if point_array_name is not None:
        if output_name is None:
            output_name = point_array_name + "_center"
        pt_arr = dataset.get_point_array(point_array_name)
        out_field = pgc.field(dtype=pgc.f32, shape=(n,))
        _center_scalar(dataset.cell_set, cell_type, pt_arr.data, out_field, n)
        dataset.add_cell_array(output_name, AOSArray(out_field))
    else:
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


def parametric_center_multi(dataset, cell_type, names, output_names=None):
    """Interpolate 3 point scalars to cell centers with cached weights.

    Uses pgc.local_array internally to compute shape function weights
    once per cell and reuse them for all 3 fields.

    Args:
        dataset: input Dataset
        cell_type: cell type instance
        names: list of 3 point array names
        output_names: list of 3 output cell array names (optional)
    """
    if len(names) != 3:
        raise ValueError("parametric_center_multi requires exactly 3 field names")
    if output_names is None:
        output_names = [n + "_center" for n in names]

    n = dataset.num_cells
    f1 = dataset.get_point_array(names[0]).data
    f2 = dataset.get_point_array(names[1]).data
    f3 = dataset.get_point_array(names[2]).data
    o1 = pgc.field(dtype=pgc.f32, shape=(n,))
    o2 = pgc.field(dtype=pgc.f32, shape=(n,))
    o3 = pgc.field(dtype=pgc.f32, shape=(n,))

    _center_multi(dataset.cell_set, cell_type, f1, f2, f3, o1, o2, o3, n)

    dataset.add_cell_array(output_names[0], AOSArray(o1))
    dataset.add_cell_array(output_names[1], AOSArray(o2))
    dataset.add_cell_array(output_names[2], AOSArray(o3))
    return dataset
