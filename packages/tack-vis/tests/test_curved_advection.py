"""Advection and probe on curved cells and high-order fields: ``by_id`` reads a curved
cell (``H1`` geometry above order 1) and a field of any interpolated space
through each shape group's own view (``_ByShape``), so the search from the
points inverts a curved cell as the cell-side search does, and the kernel
evaluates a high-order velocity as probe does."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes
from tack.dataset.topology import uniform_cells


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Compared in f64, on every backend with it."""


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


@tack.func
def _swirl(x):
    return [-(x[1] - 0.6) + 0.1 * x[2] * x[2], (x[0] - 0.6) + 0.2 * x[0] * x[2], 0.1 + 0.3 * x[0] * x[1]]


def _wedges(grid):
    hexes = td.extract_cells(grid, np.arange(grid.num_cells)).topology.connectivity.numpy()
    conn = []
    for h in hexes.reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    topology = td.UnstructuredTopology(types, offsets,
                                       tack.array(np.array(conn, np.int32).reshape(-1)),
                                       num_points=grid.num_points)
    return td.DataSet(topology, grid.positions())


def _mesh(cells, order, curved=True):
    """Cells of ``cells`` over a box of 1.2 by 1.2 by 0.6, the geometry bent at order 2
    (or straight), and ``v``, the swirl at ``order``."""
    grid = td.uniform_grid((7, 7, 4), spacing=(0.2, 0.2, 0.2), dtype=tack.f64)
    data = {"hexahedra": lambda: td.extract_cells(grid, np.arange(grid.num_cells)),
            "grid": lambda: grid,
            "tetrahedra": lambda: td.tetrahedralize(grid),
            "wedges": lambda: _wedges(grid)}[cells]()
    data.fields["v"] = td.interpolate(data, _swirl, td.H1(data, order=order),
                                      dtype=tack.vec3(tack.f64))
    if curved:
        data.geometry = td.interpolate(data, _bend, td.H1(data, order=2),
                                       dtype=tack.vec3(tack.f64))
    return data


_SEEDS = np.array([[0.7, 0.62, 0.15], [0.85, 0.7, 0.25], [0.5, 0.45, 0.2], [0.62, 0.9, 0.3]])


def _rk4_by_probe(data, seeds, h, steps):
    """RK4 steps on the host, each velocity probed: the path ``streamlines`` must
    follow, point for point."""
    def velocity(p):
        out = td.probe(data, p)
        assert out.fields["valid"].values.numpy().all(), "a seed left the mesh"
        return out.fields["v"].values.numpy()

    x = np.array(seeds, np.float64)
    path = [x.copy()]
    for _ in range(steps):
        k1 = velocity(x)
        k2 = velocity(x + (0.5 * h) * k1)
        k3 = velocity(x + (0.5 * h) * k2)
        k4 = velocity(x + h * k3)
        x = x + h * ((k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
        path.append(x.copy())
    return np.stack(path, axis=1)          # (seed, step, 3)


def _paths(out, seeds, steps):
    positions = out.positions()
    offsets = tack.arrays.to_host(out.topology.line_offsets)
    assert len(offsets) == len(seeds) + 1
    paths = [positions[offsets[i]:offsets[i + 1]] for i in range(len(seeds))]
    assert all(len(p) == steps + 1 for p in paths), [len(p) for p in paths]
    return np.stack(paths)


@pytest.mark.parametrize("cells, order, curved", [
    ("hexahedra", 2, True), ("hexahedra", 3, True), ("grid", 3, True),
    ("tetrahedra", 2, True), ("wedges", 2, True),
    ("hexahedra", 3, False), ("grid", 4, False)])
def test_streamlines_follow_the_probed_path(cells, order, curved):
    """Through curved cells, and with a velocity above order 1 on straight ones: each
    step's velocities are where ``probe`` puts them, so the path is the host's RK4
    over probed velocities. Advection refused both: an order-1 geometry and point
    or cell data only."""
    data = _mesh(cells, order, curved)
    h, steps = 0.01, 30
    out = td.streamlines(data, "v", _SEEDS, h, steps)
    np.testing.assert_allclose(_paths(out, _SEEDS, steps), _rk4_by_probe(data, _SEEDS, h, steps),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra", "wedges"])
def test_the_neighbour_walk_changes_nothing_on_curved_cells(cells):
    """``neighbors=True`` and the last cell tried first take a curved cell only where
    it holds the point well inside, by its own map: the paths are the bins' bit
    for bit."""
    from tack.dataset.by_id import _MeshSearch

    data = _mesh(cells, 2)
    h, steps = 0.02, 60
    walked = td.streamlines(data, "v", _SEEDS, h, steps, neighbors=True)
    plain = td.streamlines(data, "v", _SEEDS, h, steps)
    np.testing.assert_array_equal(walked.positions(), plain.positions())
    old = _MeshSearch.PREVIOUS_CELL
    try:
        _MeshSearch.PREVIOUS_CELL = 0
        searched = td.streamlines(data, "v", _SEEDS, h, steps)
    finally:
        _MeshSearch.PREVIOUS_CELL = old
    np.testing.assert_array_equal(searched.positions(), plain.positions())


@pytest.mark.parametrize("cells, order", [("hexahedra", 2), ("hexahedra", 3),
                                          ("tetrahedra", 2), ("wedges", 2)])
def test_probe_finds_curved_cells_alike_from_the_points_and_the_cells(cells, order):
    """A fresh locator's first small query searches from the cells; later ones from
    the points, through ``by_id``. On curved cells the two now give the same cells,
    coordinates and values, to the bit (``by_id`` refused curved geometry, so
    every probe searched from the cells)."""
    from tack.dataset.by_id import mesh_search

    data = _mesh(cells, order)
    locator = td.CellLocator(data)
    assert mesh_search(data, locator) is not None
    centers = td.cell_centers(data).values.numpy()
    points = centers + np.random.default_rng(3).uniform(-0.03, 0.03, centers.shape)
    first = td.probe(data, points, locator=locator)       # from the cells
    again = td.probe(data, points, locator=locator)       # from the points
    for name in ("valid", "v"):
        np.testing.assert_array_equal(again.fields[name].values.numpy(),
                                      first.fields[name].values.numpy())
    assert first.fields["valid"].values.numpy().all()
