"""Tests for @tack.data_oriented template parameters."""

import numpy as np

import tack

# Build list of available backends


# --- Structured cell set (scalar attrs only, no field attrs) ---

@tack.data_oriented
class CellSetStructured:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    @tack.func
    def get_cell_point0(self, cell_id):
        cx = cell_id % (self.nx - 1)
        cy = cell_id // (self.nx - 1)
        return cy * self.nx + cx


# --- Explicit cell set (field attrs) ---

@tack.data_oriented
class CellSetExplicit:
    def __init__(self, conn_np):
        self.connectivity = tack.field(dtype=tack.i32, shape=(len(conn_np),))
        self.connectivity.from_numpy(conn_np)

    @tack.func
    def get_cell_point0(self, cell_id):
        return self.connectivity[cell_id * 4]


# --- Kernels that use template objects ---

@tack.kernel
def extract_point0(cell_set, output):
    for i in range(output.shape[0]):
        output[i] = cell_set.get_cell_point0(i)


def test_structured_template(backend):
    """Template with scalar attrs only (nx, ny become constants)."""
    nx, ny = 4, 3
    num_cells = (nx - 1) * (ny - 1)
    output = tack.field(dtype=tack.i32, shape=(num_cells,))

    cs = CellSetStructured(nx, ny)
    extract_point0(cs, output)

    result = output.to_numpy()
    expected = np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_explicit_template(backend):
    """Template with field attrs (connectivity becomes extra parameter)."""
    # Build connectivity for a 4x3 structured grid
    nx, ny = 4, 3
    num_cells = (nx - 1) * (ny - 1)
    conn = np.empty(num_cells * 4, dtype=np.int32)
    for cj in range(ny - 1):
        for ci in range(nx - 1):
            cid = cj * (nx - 1) + ci
            p0 = cj * nx + ci
            conn[cid * 4: cid * 4 + 4] = [p0, p0 + 1, p0 + nx + 1, p0 + nx]

    output = tack.field(dtype=tack.i32, shape=(num_cells,))
    cs = CellSetExplicit(conn)
    extract_point0(cs, output)

    result = output.to_numpy()
    expected = np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_same_kernel_different_templates(backend):
    """Same kernel works with different template types."""
    nx, ny = 4, 3
    num_cells = (nx - 1) * (ny - 1)

    # Build explicit connectivity
    conn = np.empty(num_cells * 4, dtype=np.int32)
    for cj in range(ny - 1):
        for ci in range(nx - 1):
            cid = cj * (nx - 1) + ci
            p0 = cj * nx + ci
            conn[cid * 4: cid * 4 + 4] = [p0, p0 + 1, p0 + nx + 1, p0 + nx]

    output_s = tack.field(dtype=tack.i32, shape=(num_cells,))
    output_e = tack.field(dtype=tack.i32, shape=(num_cells,))

    cs_s = CellSetStructured(nx, ny)
    cs_e = CellSetExplicit(conn)

    extract_point0(cs_s, output_s)
    extract_point0(cs_e, output_e)

    np.testing.assert_array_equal(output_s.to_numpy(), output_e.to_numpy())


def test_different_scalar_values(backend):
    """Same template type with different scalar values produces different code."""
    cs_a = CellSetStructured(4, 3)
    cs_b = CellSetStructured(5, 4)

    output_a = tack.field(dtype=tack.i32, shape=(6,))
    output_b = tack.field(dtype=tack.i32, shape=(12,))

    extract_point0(cs_a, output_a)
    extract_point0(cs_b, output_b)

    # 4x3 grid: point0 of cells
    expected_a = np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
    np.testing.assert_array_equal(output_a.to_numpy(), expected_a)

    # 5x4 grid: point0 of cells
    expected_b = np.array([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13], dtype=np.int32)
    np.testing.assert_array_equal(output_b.to_numpy(), expected_b)
