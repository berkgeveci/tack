"""``integrate``: a dataset's measure and its fields' integrals by quadrature through
each cell's own map, held to closed forms -- polynomials over straight and curved
cells of every shape, fields of any order -- and over pieces; and ``cell_geometry``
of curved cells, which it measures."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes
from tack.dataset.algorithms.integrate import quadrature
from tack.dataset.topology import uniform_cells

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    yield f64_backend


@tack.func
def _stretch(x):
    # x scaled by 1 + 0.3 y: the volume element 1 + 0.3 y, a curved map of order 2.
    return [x[0] * (1.0 + 0.3 * x[1]), x[1], x[2]]


@tack.func
def _cubic(x):
    return x[0] * x[0] * x[1] + 2.0 * x[2] * x[2] * x[2] - x[0] * x[1] * x[2]


def _cube(kind, n=3):
    grid = td.uniform_grid((n + 1,) * 3, spacing=(1.0 / n,) * 3, dtype=tack.f64)
    if kind == "hexahedra":
        return td.extract_cells(grid, np.arange(grid.num_cells))
    if kind == "tetrahedra":
        return td.tetrahedralize(grid)
    hexes = td.extract_cells(grid, np.arange(grid.num_cells)).topology.connectivity.numpy()
    conn = []
    for h in hexes.reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    topology = td.UnstructuredTopology(types, offsets,
                                       tack.array(np.array(conn, np.int32).reshape(-1)),
                                       num_points=grid.num_points)
    return td.DataSet(topology, grid.positions(), dtype=tack.f64)


@pytest.mark.parametrize("shape", [shapes.Line, shapes.Quad, shapes.Hexahedron,
                                   shapes.Triangle, shapes.Tetra, shapes.Wedge,
                                   shapes.Pyramid])
def test_quadrature_integrates_monomials(shape):
    """Each rule sums to its reference element's measure and integrates monomials of
    low degree exactly (the pyramid's element is the unit cube)."""
    points, weights = quadrature(shape, 4)
    volume = {shapes.Triangle: 0.5, shapes.Tetra: 1 / 6, shapes.Wedge: 0.5}.get(shape, 1.0)
    assert np.isclose(weights.sum(), volume, rtol=1e-14)
    x, y, z = points.T
    if shape is shapes.Tetra:
        np.testing.assert_allclose((weights * x * y * z).sum(), 1 / 720, rtol=1e-12)
    elif shape is shapes.Triangle:
        np.testing.assert_allclose((weights * x * x * y).sum(), 1 / 60, rtol=1e-12)
    elif shape is shapes.Wedge:
        np.testing.assert_allclose((weights * x * y * z * z).sum(), 1 / 72, rtol=1e-12)


@pytest.mark.parametrize("kind", ["hexahedra", "tetrahedra", "wedges"])
def test_a_cubic_over_straight_and_curved_cubes(kind):
    """The cube's volume and the integral of a cubic field (H1 of order 3, and as
    point data of order 1 its own interpolant's), then the cube stretched by x (1 +
    0.3 y) -- a curved geometry of order 2 -- whose volume is 1.15: exact to
    rounding."""
    data = _cube(kind)
    space = td.H1(data, order=3, basis="equispaced") if kind != "hexahedra" else \
        td.H1(data, order=3)
    data.fields["u"] = td.interpolate(data, _cubic, space)
    out = td.integrate(data)
    assert out.measure == pytest.approx(1.0, rel=1e-13)
    assert out.values["u"] == pytest.approx(1 / 6 + 0.5 - 1 / 8, rel=1e-12)
    curved = td.DataSet(data.topology, data.geometry.values)
    curved.geometry = td.interpolate(curved, _stretch, td.H1(curved, order=2,
                                                             **({} if kind == "hexahedra"
                                                                else {"basis": "equispaced"})),
                                     dtype=tack.vec3(tack.f64))
    assert td.integrate(curved).measure == pytest.approx(1.15, rel=1e-13)
    measures = td.cell_geometry(curved).measure.values.numpy()
    assert measures.sum() == pytest.approx(1.15, rel=1e-13)


def test_a_curved_surface_and_cell_data():
    """A quad surface bent into a cylinder's quarter by an order-4 geometry has nearly
    the quarter's area (the interpolant's, to 1e-6), not the chords'; cell data
    integrates as value times measure."""
    grid = td.uniform_grid((5, 3), spacing=(0.25, 0.5), dtype=tack.f64)

    @tack.func
    def bend(x):
        a = 1.5707963267948966 * x[0]
        return [tack.cos(a), tack.sin(a), x[1]]

    data = td.extract_cells(grid, np.arange(grid.num_cells))
    flat = td.integrate(data).measure
    data.geometry = td.interpolate(data, bend, td.H1(data, order=4), dtype=tack.vec3(tack.f64))
    assert flat == pytest.approx(1.0)
    assert td.integrate(data).measure == pytest.approx(np.pi / 2, rel=1e-5)
    data.fields["c"] = td.Field(td.Constant(data), tack.array(np.arange(data.num_cells) + 1.0))
    measures = td.cell_geometry(data).measure.values.numpy()
    assert td.integrate(data, "c").values["c"] == pytest.approx(
        float((measures * (np.arange(data.num_cells) + 1.0)).sum()), rel=1e-13)


def test_vector_fields_and_pieces():
    """A vector field's integral is a tuple; over pieces with ghost cells each owned
    cell counts once, and the pieces sum to the whole's."""
    grid = td.uniform_grid((7, 7, 7), spacing=(1 / 6,) * 3, dtype=tack.f64)

    @tack.func
    def position(x):
        return [x[0], x[1] * x[1], 1.0]

    grid.fields["v"] = td.interpolate(grid, position, td.H1(grid), dtype=tack.vec3(tack.f64))
    whole = td.integrate(grid, "v")
    assert whole.values["v"] == pytest.approx((0.5, 1 / 3 + 1 / 216, 1.0), rel=1e-12)
    pieces = td.split(grid, (2, 2, 1), ghost_layers=1)
    split = td.integrate(pieces, "v")
    assert split.measure == pytest.approx(whole.measure, rel=1e-13)
    assert split.values["v"] == pytest.approx(whole.values["v"], rel=1e-13)


def test_polyhedra_take_cell_data():
    grid = td.uniform_grid((3, 3, 3), spacing=(0.5,) * 3, dtype=tack.f64)
    poly = td.as_polyhedra(td.extract_cells(grid, np.arange(8)))
    poly.fields["c"] = td.Field(td.Constant(poly), tack.array(np.full(8, 2.0)))
    out = td.integrate(poly, "c")
    assert out.measure == pytest.approx(1.0) and out.values["c"] == pytest.approx(2.0)
    poly.fields["p"] = td.Field(td.H1(poly), tack.array(np.zeros(poly.num_points)))
    with pytest.raises(NotImplementedError, match="cell data only"):
        td.integrate(poly, "p")
