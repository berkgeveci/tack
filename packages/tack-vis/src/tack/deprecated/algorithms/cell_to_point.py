"""Cell-to-point data conversion for uniform grids.

Averages cell-centered values to node-centered values. Each node gets
the average of its adjacent cells (up to 8 in 3D, fewer at boundaries).

Public API
----------
cell_to_point(cell_data, nx_cells, ny_cells, nz_cells)
    Convert cell-centered field to point-centered field.
"""

import tack


@tack.kernel
def _cell_to_point(cell_data, point_data, nx_c, ny_c, nz_c, n_points):
    """Average adjacent cell values to each node.

    Input:  cell_data  — shape (nx_c * ny_c * nz_c,)
    Output: point_data — shape ((nx_c+1) * (ny_c+1) * (nz_c+1),)

    Each node averages up to 8 adjacent cells (fewer at boundaries).
    """
    nx_p = nx_c + 1
    ny_p = ny_c + 1
    nxy_c = nx_c * ny_c
    for idx in range(n_points):
        i = idx % nx_p
        j = (idx // nx_p) % ny_p
        k = idx // (nx_p * ny_p)

        total = 0.0
        count = 0

        # Cell (i-1, j-1, k-1)
        if i > 0 and j > 0 and k > 0:
            total = total + cell_data[(k-1) * nxy_c + (j-1) * nx_c + (i-1)]
            count = count + 1
        # Cell (i, j-1, k-1)
        if i < nx_c and j > 0 and k > 0:
            total = total + cell_data[(k-1) * nxy_c + (j-1) * nx_c + i]
            count = count + 1
        # Cell (i-1, j, k-1)
        if i > 0 and j < ny_c and k > 0:
            total = total + cell_data[(k-1) * nxy_c + j * nx_c + (i-1)]
            count = count + 1
        # Cell (i, j, k-1)
        if i < nx_c and j < ny_c and k > 0:
            total = total + cell_data[(k-1) * nxy_c + j * nx_c + i]
            count = count + 1
        # Cell (i-1, j-1, k)
        if i > 0 and j > 0 and k < nz_c:
            total = total + cell_data[k * nxy_c + (j-1) * nx_c + (i-1)]
            count = count + 1
        # Cell (i, j-1, k)
        if i < nx_c and j > 0 and k < nz_c:
            total = total + cell_data[k * nxy_c + (j-1) * nx_c + i]
            count = count + 1
        # Cell (i-1, j, k)
        if i > 0 and j < ny_c and k < nz_c:
            total = total + cell_data[k * nxy_c + j * nx_c + (i-1)]
            count = count + 1
        # Cell (i, j, k)
        if i < nx_c and j < ny_c and k < nz_c:
            total = total + cell_data[k * nxy_c + j * nx_c + i]
            count = count + 1

        point_data[idx] = total / float(count)


def cell_to_point(cell_data, nx_cells, ny_cells, nz_cells):
    """Convert cell-centered data to point-centered data.

    Args:
        cell_data: tack.field, shape (nx_cells * ny_cells * nz_cells,)
            Supports f32 and f64 — output dtype matches input.
        nx_cells, ny_cells, nz_cells: cell dimensions

    Returns:
        tack.field, shape ((nx_cells+1) * (ny_cells+1) * (nz_cells+1),)
    """
    n_points = (nx_cells + 1) * (ny_cells + 1) * (nz_cells + 1)
    point_data = tack.field(dtype=cell_data.dtype, shape=(n_points,))
    _cell_to_point(cell_data, point_data, nx_cells, ny_cells, nz_cells, n_points)
    return point_data
