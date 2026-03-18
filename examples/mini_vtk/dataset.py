"""Dataset -- a mesh container with coordinates, cell sets, and data arrays."""

import numpy as np
import pgc

from .arrays import AOSArray, AOSTupleArray
from .cellsets import CellSetStructured3D


class Dataset:
    """Container for a mesh with associated data arrays.

    Not @pgc.data_oriented -- this is a Python-side container.
    The individual components (coordinates, cell_set, arrays) are
    @pgc.data_oriented and can be passed into kernels as templates.
    """

    def __init__(self, coordinates, cell_set, num_points, num_cells):
        self.coordinates = coordinates
        self.cell_set = cell_set
        self.num_points = num_points
        self.num_cells = num_cells
        self.point_data = {}
        self.cell_data = {}

    def add_point_array(self, name, array):
        self.point_data[name] = array

    def add_cell_array(self, name, array):
        self.cell_data[name] = array

    def get_point_array(self, name):
        return self.point_data[name]

    def get_cell_array(self, name):
        return self.cell_data[name]


def make_rectilinear_dataset(x_np, y_np, z_np):
    """Create a dataset with a structured cell set and AOS coordinates.

    Args:
        x_np, y_np, z_np: 1D numpy arrays of coordinate values along each axis.

    Returns:
        Dataset with AOSTupleArray coordinates and CellSetStructured3D cell set.
    """
    nx, ny, nz = len(x_np) - 1, len(y_np) - 1, len(z_np) - 1
    n_points = (nx + 1) * (ny + 1) * (nz + 1)
    n_cells = nx * ny * nz

    # Build AOS coordinate array: [x0,y0,z0, x1,y1,z1, ...]
    coords_np = np.zeros((n_points, 3), dtype=np.float32)
    idx = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                coords_np[idx, 0] = x_np[i]
                coords_np[idx, 1] = y_np[j]
                coords_np[idx, 2] = z_np[k]
                idx += 1

    flat = coords_np.ravel()
    coord_field = pgc.field(dtype=pgc.f32, shape=(len(flat),))
    coord_field.from_numpy(flat)
    coordinates = AOSTupleArray(coord_field, n_points, 3)

    cell_set = CellSetStructured3D(nx, ny, nz)
    return Dataset(coordinates, cell_set, n_points, n_cells)


def make_explicit_hex_dataset(points_np, connectivity_np):
    """Create a dataset with an explicit hex cell set.

    Args:
        points_np: (N, 3) float32 array of point coordinates.
        connectivity_np: flat int32 array, 8 point IDs per cell.

    Returns:
        Dataset with AOSTupleArray coordinates and CellSetExplicit cell set.
    """
    from .cellsets import CellSetExplicit

    n_points = points_np.shape[0]
    n_cells = len(connectivity_np) // 8

    flat = points_np.astype(np.float32).ravel()
    coord_field = pgc.field(dtype=pgc.f32, shape=(len(flat),))
    coord_field.from_numpy(flat)
    coordinates = AOSTupleArray(coord_field, n_points, 3)

    conn_field = pgc.field(dtype=pgc.i32, shape=(len(connectivity_np),))
    conn_field.from_numpy(connectivity_np.astype(np.int32))
    cell_set = CellSetExplicit(conn_field, 8)

    return Dataset(coordinates, cell_set, n_points, n_cells)
