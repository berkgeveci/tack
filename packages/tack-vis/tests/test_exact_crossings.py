"""Exact crossings: a cut of high-order data on sub-cells puts each crossing where the
field itself takes the value along the sub-edge (``algorithms/_crossings.py``), not
where the line between its lattice values does, and carries every field there."""

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
def _cubic(x):
    return x[0] * x[0] * x[0] + 0.5 * x[1] * x[2] - 0.3 * x[1] * x[1]


@tack.func
def _other(x):
    return x[0] * x[1] + x[2] * x[2]


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


def _wedges(hexes):
    conn = []
    for h in hexes.topology.connectivity.numpy().reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    return td.DataSet(td.UnstructuredTopology(
        types, offsets, tack.array(np.array(conn, np.int32).reshape(-1)),
        num_points=hexes.num_points), hexes.geometry.values)


def _mesh(cells, order=3, curved=True):
    """Four cells a side over the unit cube (or square), ``u`` a cubic and ``v``
    another polynomial at ``order``, the geometry bent at order 2 (or straight)."""
    two = cells in ("quads", "triangles")
    grid = td.uniform_grid((5,) * (2 if two else 3), spacing=(0.25,) * (2 if two else 3),
                           dtype=tack.f64)
    hexes = td.extract_cells(grid, np.arange(grid.num_cells))
    data = {"hexahedra": lambda: hexes, "quads": lambda: hexes,
            "tetrahedra": lambda: td.tetrahedralize(hexes),
            "triangles": lambda: td.triangulate(hexes),
            "wedges": lambda: _wedges(hexes)}[cells]()
    data = td.DataSet(data.topology, data.geometry)
    nodes = {} if cells in ("hexahedra", "quads") else {"nodes": "equispaced"}
    space = td.H1(data, order=order, **nodes)
    data.fields["u"] = td.interpolate(data, _cubic, space, dtype=tack.f64)
    data.fields["v"] = td.interpolate(data, _other, space, dtype=tack.f64)
    if curved:
        data.geometry = td.interpolate(data, _bend, td.H1(data, order=2),
                                       dtype=tack.vec3(tack.f64))
    return data


def _probed(data, points, name, all_valid=True):
    out = td.probe(data, points)
    valid = out.fields["valid"].values.numpy() == 1
    assert valid.all() or not all_valid
    return out.fields[name].values.numpy()[valid]


@pytest.mark.parametrize("curved", [False, True])
@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra", "wedges", "quads", "triangles"])
def test_crossings_are_on_the_level_and_carry_their_fields(cells, curved):
    """Every vertex is where the field takes the value (probed: to rounding, or to
    the probe's own Newton tolerance in curved cells), and every field carried is
    its value there; the cells are the linear cut's."""
    data = _mesh(cells, curved=curved)
    tolerance = 1e-10 if curved else 1e-13
    for steps in (1, 2):
        exact = td.contour(data, "u", 0.3, subdivision=steps)
        linear = td.contour(data, "u", 0.3, subdivision=steps, exact=False)
        assert exact.num_cells == linear.num_cells > 0
        x = exact.positions()
        assert np.abs(_probed(data, x, "u") - 0.3).max() < tolerance
        np.testing.assert_allclose(exact.fields["v"].values.numpy(), _probed(data, x, "v"),
                                   rtol=0, atol=tolerance)
        # The lattice values' line: off the level, and on a curved mesh's boundary
        # off its cells (those are not probed).
        off = np.abs(_probed(data, linear.positions(), "u", all_valid=False) - 0.3).max()
        assert off > 1e3 * tolerance


@tack.func
def _square_x(x):
    return x[0] * x[0]


def test_a_crossing_at_a_lattice_point_is_that_point():
    """A value the field takes at lattice points is crossed there, at the lattice
    point itself."""
    data = _mesh("hexahedra", order=2, curved=False)
    data.fields["x"] = td.interpolate(data, _square_x, td.H1(data, order=2), dtype=tack.f64)
    surface = td.contour(data, "x", 0.25, subdivision=2)
    assert surface.num_cells and (surface.positions()[:, 0] == 0.5).all()


def test_several_levels_and_slices():
    """Each vertex on its own level. A slice of curved cells at one step a cell: the
    lattice values' line puts its vertices on the plane but on chords between
    corners, off the curved cells; at the roots they are on both."""
    data = _mesh("tetrahedra")
    levels = [0.1, 0.3]
    surface = td.contour(data, "u", levels)
    u = _probed(data, surface.positions(), "u")
    assert np.minimum(np.abs(u - 0.1), np.abs(u - 0.3)).max() < 1e-9      # probe's Newton
    normal = np.array([0.3, -0.2, 1.0])
    unit = normal / np.linalg.norm(normal)
    for cells in ("hexahedra", "wedges"):
        curved = _mesh(cells)
        exact = td.slice_plane(curved, [0.5, 0.5, 0.5], normal, subdivision=1)
        linear = td.slice_plane(curved, [0.5, 0.5, 0.5], normal, subdivision=1, exact=False)
        assert exact.num_cells == linear.num_cells
        assert np.abs((exact.positions() - 0.5) @ unit).max() < 1e-12
        assert td.probe(curved, exact.positions()).fields["valid"].values.numpy().all()
        assert np.abs(np.sort(exact.positions(), 0) - np.sort(linear.positions(), 0)).max() > 1e-3


def test_an_l2_field_is_crossed_in_its_cell():
    """A discontinuous field has no traces: its roots are found with its holding
    cell's values (the polynomial here is continuous, so the cut is on it)."""
    data = _mesh("hexahedra", order=2, curved=False)
    data.fields["w"] = td.interpolate(data, _cubic, td.L2(data, order=3), dtype=tack.f64)
    surface = td.contour(data, "w", 0.3)
    assert surface.num_cells
    assert np.abs(_probed(data, surface.positions(), "w") - 0.3).max() < 1e-12


@pytest.mark.parametrize("cells", ["grid", "tetrahedra"])
def test_pieces_cross_their_shared_edges_alike(cells):
    """With a ghost layer, the pieces' owned vertices are the whole's, bit for bit:
    a root on an edge or face shared by pieces comes from that entity's own values,
    in its frame and from its first end, whichever cell computes it."""

    def bent(d):
        nodes = {} if cells == "grid" else {"nodes": "equispaced"}
        d.fields["u"] = td.interpolate(d, _cubic, td.H1(d, order=3, **nodes), dtype=tack.f64)
        d.geometry = td.interpolate(d, _bend, td.H1(d, order=2), dtype=tack.vec3(tack.f64))
        return d

    def grid():
        g = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25), dtype=tack.f64)
        return g if cells == "grid" else td.tetrahedralize(g)

    def owned(surface):
        x = surface.positions()
        marks = surface.ghosts.points if surface.ghosts is not None else None
        if marks is not None:
            x = x[(marks.numpy() & td.Ghosts.SKIPPED_POINTS) == 0]
        return x

    whole = owned(td.contour(bent(grid()), "u", 0.3))
    if cells == "grid":
        pieces = td.split(grid(), (2, 2, 1), ghost_layers=1)
    else:
        whole_grid = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25), dtype=tack.f64)
        pieces = [td.tetrahedralize(p) for p in td.split(whole_grid, (2, 2, 1), ghost_layers=1)]
    mine = np.concatenate([owned(td.contour(bent(p), "u", 0.3)) for p in pieces])
    assert len(mine) == len({tuple(x) for x in mine})
    assert {tuple(x) for x in mine} == {tuple(x) for x in whole}
