"""Tests for tack.algorithms.cell_to_point.

Node values are the average of the adjacent cells — up to 8 in the
interior, 4 on a face, 2 on an edge, 1 at a corner.  The boundary
arithmetic is where this kind of stencil goes wrong, so most of these
tests aim at it.
"""

import numpy as np
import pytest
from vis_helpers import upload

import tack
from tack.algorithms.cell_to_point import cell_to_point


def _run(cell_np, nx, ny, nz, dtype=tack.f32, np_dtype=np.float32):
    field = upload(cell_np, dtype=dtype, np_dtype=np_dtype)
    return cell_to_point(field, nx, ny, nz).to_numpy()


def test_output_shape(backend):
    nx, ny, nz = 2, 3, 4
    out = _run(np.zeros(nx * ny * nz), nx, ny, nz)
    assert out.shape == ((nx + 1) * (ny + 1) * (nz + 1),)


def test_constant_field_is_preserved(backend):
    """Averaging any number of equal values gives the same value."""
    nx, ny, nz = 3, 3, 3
    out = _run(np.full(nx * ny * nz, 7.5), nx, ny, nz)
    np.testing.assert_allclose(out, 7.5, rtol=1e-6)


def test_corner_node_takes_the_single_adjacent_cell(backend):
    """The 8 grid corners each touch exactly one cell."""
    nx, ny, nz = 2, 3, 4
    cells = np.arange(nx * ny * nz, dtype=np.float64)
    out = _run(cells, nx, ny, nz)

    nxp, nyp = nx + 1, ny + 1

    def node(i, j, k):
        return out[k * nxp * nyp + j * nxp + i]

    def cell(i, j, k):
        return cells[k * nx * ny + j * nx + i]

    for i, ci in ((0, 0), (nx, nx - 1)):
        for j, cj in ((0, 0), (ny, ny - 1)):
            for k, ck in ((0, 0), (nz, nz - 1)):
                assert node(i, j, k) == pytest.approx(cell(ci, cj, ck))


def test_interior_node_averages_eight_cells(f64_backend):
    """An interior node is the mean of its 8 surrounding cells."""
    nx, ny, nz = 3, 3, 3
    rng = np.random.default_rng(0)
    cells = rng.random(nx * ny * nz)
    out = _run(cells, nx, ny, nz, dtype=tack.f64, np_dtype=np.float64)

    nxp, nyp = nx + 1, ny + 1
    i = j = k = 1
    expected = np.mean([
        cells[(k + dk) * nx * ny + (j + dj) * nx + (i + di)]
        for dk in (-1, 0) for dj in (-1, 0) for di in (-1, 0)
    ])
    got = out[k * nxp * nyp + j * nxp + i]
    assert got == pytest.approx(expected)


def test_matches_a_numpy_reference(f64_backend):
    """Every node, checked against a direct numpy stencil at f64 precision."""
    nx, ny, nz = 3, 4, 2
    rng = np.random.default_rng(7)
    cells = rng.random(nx * ny * nz)
    out = _run(cells, nx, ny, nz, dtype=tack.f64, np_dtype=np.float64)

    grid = cells.reshape(nz, ny, nx)
    expected = np.empty((nz + 1) * (ny + 1) * (nx + 1))
    idx = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                neighbours = [
                    grid[k + dk, j + dj, i + di]
                    for dk in (-1, 0) for dj in (-1, 0) for di in (-1, 0)
                    if 0 <= k + dk < nz and 0 <= j + dj < ny and 0 <= i + di < nx
                ]
                expected[idx] = np.mean(neighbours)
                idx += 1
    np.testing.assert_allclose(out, expected, rtol=1e-14)


def test_linear_ramp_stays_linear_in_the_interior(f64_backend):
    """Averaging a linear cell field reproduces it at interior nodes."""
    nx, ny, nz = 4, 4, 4
    i = np.arange(nx)
    cells = np.broadcast_to(i, (nz, ny, nx)).astype(np.float64).ravel()
    out = _run(cells, nx, ny, nz, dtype=tack.f64, np_dtype=np.float64)

    nxp, nyp = nx + 1, ny + 1
    node = out.reshape(nz + 1, nyp, nxp)
    # Interior node i sits between cell centres i-1 and i → value i - 0.5.
    for ii in range(1, nx):
        np.testing.assert_allclose(node[2, 2, ii], ii - 0.5, rtol=1e-14)


def test_output_dtype_follows_input(f64_backend):
    """f64 input yields an f64 field, not a silent downcast."""
    out = cell_to_point(upload(np.ones(8), dtype=tack.f64,
                               np_dtype=np.float64), 2, 2, 2)
    assert out.dtype is tack.f64
    assert out.to_numpy().dtype == np.float64


def test_single_cell_grid(backend):
    """A 1×1×1 grid: all 8 nodes take that one cell's value."""
    out = _run(np.array([3.25]), 1, 1, 1)
    np.testing.assert_allclose(out, 3.25, rtol=1e-6)
    assert out.size == 8
