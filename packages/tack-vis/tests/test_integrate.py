# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

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


def _linear_at(x):
    return 1.0 + x[:, 0] + 2.0 * x[:, 1] - 3.0 * x[:, 2]


def test_point_fields_on_polyhedra_and_polygons():
    """A point field is linear on a polyhedron's sub-tetrahedra and a polygon's
    triangles, so a linear field integrates exactly: over the half-unit grid's
    eight polyhedra and over its top's polygons, scalar and vector. They were
    refused."""
    grid = td.uniform_grid((3, 3, 3), spacing=(0.5,) * 3, dtype=tack.f64)
    poly = td.as_polyhedra(td.extract_cells(grid, np.arange(8)))
    x = poly.positions()
    poly.fields["u"] = td.Field(td.H1(poly), tack.array(_linear_at(x)))
    vector = np.c_[_linear_at(x), x[:, 0], np.ones(len(x))]
    poly.fields["v"] = td.Field(td.H1(poly), tack.array(vector, tack.vec3(tack.f64)))
    out = td.integrate(poly, ["u", "v"])
    exact = 1.0 + 0.5 + 2 * 0.5 - 3 * 0.5             # the mean of u over the unit cube
    assert out.measure == pytest.approx(1.0)
    assert out.values["u"] == pytest.approx(exact, abs=1e-14)
    np.testing.assert_allclose(out.values["v"], [exact, 0.5, 1.0], atol=1e-14)
    top = td.as_polygons(td.extract_surface(grid))
    top.fields["u"] = td.Field(td.H1(top), tack.array(_linear_at(top.positions())))
    surface = td.integrate(top, "u")
    # Over the cube's six faces: area 6, and u's mean over them is its mean, 0.5.
    assert surface.measure == pytest.approx(6.0)
    assert surface.values["u"] == pytest.approx(6.0 * exact, abs=1e-13)


def _fan(x, loop):
    """A polygon's fan of triangles from its points' mean: their areas, signed by
    the polygon's Newell normal, and the triangles' corner means of x."""
    p = x[loop]
    m = p.mean(axis=0)
    n = np.sum(np.cross(p, np.roll(p, -1, axis=0)), axis=0)
    cross = np.cross(p - m, np.roll(p, -1, axis=0) - m)
    area = np.linalg.norm(cross, axis=1) / 2 * np.sign(cross @ n)
    return area, (m + p + np.roll(p, -1, axis=0)) / 3


def test_a_polygon_is_measured_as_its_fan():
    """A non-planar polygon is its fan of triangles from its points' mean, as Tack
    interpolates and locates in it: its area is the fan's, more than its projection
    onto its normal (which measured a contour's wrinkled free surface 1% small), and
    a linear field integrates exactly over the fan. A planar polygon whose mean is
    outside it (a U) still measures its own area: the fan's triangles are signed."""
    x = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.6], [0, 1, 0],                 # bent
                  [3, 0, 0], [6, 0, 0], [6, 3, 0], [5, 3, 0], [5, 1, 0], [4, 1, 0],
                  [4, 3, 0], [3, 3, 0]], np.float64)                            # a U
    loops = [[0, 1, 2, 3], list(range(4, 12))]
    top = td.PolygonalTopology([0, 4, 12], loops[0] + loops[1], num_points=len(x))
    data = td.DataSet(top, x, dtype=tack.f64)
    data.fields["u"] = td.Field(td.H1(data), tack.array(x[:, 0] + 2 * x[:, 2]))
    out = td.integrate(data, "u")
    areas = [_fan(x, loop)[0] for loop in loops]
    integral = sum((a * (c[:, 0] + 2 * c[:, 2])).sum() for a, c in (_fan(x, loop) for loop in loops))
    measures = td.cell_geometry(data).measure.values.numpy()
    np.testing.assert_allclose(measures, [a.sum() for a in areas], rtol=1e-13)
    assert out.measure == pytest.approx(sum(a.sum() for a in areas), rel=1e-13)
    assert out.values["u"] == pytest.approx(integral, rel=1e-13)
    projected = np.linalg.norm(np.sum(np.cross(x[loops[0]], np.roll(x[loops[0]], -1, axis=0)), axis=0)) / 2
    assert measures[0] > projected * 1.01
    assert measures[1] == pytest.approx(9.0 - 2.0, rel=1e-13)       # the U: 3 x 3 less 1 x 2


def _patch(q, f, n=64):
    """A bilinear quad's integrals of f(X) |Xu x Xv| (signed by its Newell normal),
    by n Gauss points an axis: the reference for four."""
    a, b, c, d = q
    unit = np.cross(c - a, d - b)
    unit /= np.linalg.norm(unit)
    r, w = np.polynomial.legendre.leggauss(n)
    r, w = (r + 1) / 2, w / 2
    total = 0.0
    for u, wu in zip(r, w):
        for v, wv in zip(r, w):
            cross = np.cross((1 - v) * (b - a) + v * (c - d), (1 - u) * (d - a) + u * (c - b))
            x = (1 - u) * (1 - v) * a + u * (1 - v) * b + u * v * c + (1 - u) * v * d
            total = total + wu * wv * np.linalg.norm(cross) * np.sign(cross @ unit) * f(x)
    return total


def test_a_quad_is_measured_as_its_bilinear_patch():
    """A quad has a reference element, so it is its bilinear patch: a non-planar
    one's area, centroid and integrals are the patch's (by 4 x 4 Gauss points,
    1e-11 from a 64-point rule at this twist), neither its projection
    onto its normal (VTK's, smaller) nor the fan from its mean (larger). A planar
    quad that is not convex measures its own area: the element is signed."""
    x = np.array([[0, 0, 0], [1, 0, 0], [1.1, 0.9, 0.2], [-0.1, 1, 0],            # bent
                  [3, 0, 0], [5, 0, 0], [5, 2, 0], [4.5, 0.5, 0]], np.float64)   # a dart
    types = tack.array(np.full(2, shapes.QUAD, np.uint8))
    offsets = tack.array(np.array([0, 4, 8], np.int32))
    top = td.UnstructuredTopology(types, offsets, tack.array(np.arange(8, dtype=np.int32)),
                                  num_points=8)
    data = td.DataSet(top, x, dtype=tack.f64)
    # u on the bent quad alone: the dart's field integrals weigh by |det J|.
    data.fields["u"] = td.Field(td.H1(data), tack.array(np.r_[x[:4, 0] + 2 * x[:4, 2], np.zeros(4)]))
    out = td.integrate(data, "u")
    geometry = td.cell_geometry(data)
    measures = geometry.measure.values.numpy()
    bent = x[:4]
    area = _patch(bent, lambda p: 1.0)
    assert measures[0] == pytest.approx(area, rel=1e-10)
    np.testing.assert_allclose(geometry.centroid.values.numpy()[0],
                               _patch(bent, lambda p: p) / area, atol=1e-9)
    projected = np.linalg.norm(np.cross(bent[2] - bent[0], bent[3] - bent[1])) / 2
    assert projected < area * (1 - 1e-4) and _fan(x, [0, 1, 2, 3])[0].sum() > area * (1 + 1e-5)
    dart = x[4:, :2]
    shoelace = 0.5 * np.sum(dart[:, 0] * np.roll(dart[:, 1], -1) - np.roll(dart[:, 0], -1) * dart[:, 1])
    assert measures[1] == pytest.approx(shoelace, rel=1e-13)        # 2 x 2 less a notch
    assert out.measure == pytest.approx(area + shoelace, rel=1e-10)
    assert out.values["u"] == pytest.approx(_patch(bent, lambda p: p[0] + 2 * p[2]), rel=1e-10)


def test_only_the_meshs_own_dimension_counts():
    """A tetrahedron with a triangle, a line and a vertex beside it: the volume's
    integrals leave the lower-dimensional cells out, as VTK's vtkIntegrateAttributes
    does. A vertex failed (no quadrature rule)."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [3, 0, 0],
                       [2, 1, 0], [4, 4, 4]], np.float64)
    types = tack.array(np.array([shapes.TETRA, shapes.TRIANGLE, shapes.LINE, shapes.VERTEX],
                                np.uint8))
    offsets = tack.array(np.array([0, 4, 7, 9, 10], np.int32))
    conn = tack.array(np.array([0, 1, 2, 3, 4, 5, 6, 4, 5, 7], np.int32))
    data = td.DataSet(td.UnstructuredTopology(types, offsets, conn, num_points=8),
                      tack.array(points, tack.vec3(tack.f64)))
    data.fields["one"] = td.Field(td.H1(data), tack.array(np.ones(8)))
    out = td.integrate(data, "one")
    assert out.measure == pytest.approx(1 / 6) and out.values["one"] == pytest.approx(1 / 6)


def _slab():
    data = td.uniform_grid((5, 4, 3), spacing=(0.25, 0.5, 1.0))
    x = td.cell_centers(data).values.numpy()[:, 0].astype(np.float32)
    data.fields["x"] = td.Field(td.Constant(data), tack.array(x))
    return data, x


def test_integrate_takes_fields_by_name(backend):
    data, x = _slab()
    doubled = td.Field(td.Constant(data), tack.array(2 * x))
    out = td.integrate(data, {"twice": doubled, "x": "x"})
    by_name = td.integrate(data, "x").values["x"]
    np.testing.assert_allclose(out.values["twice"], 2 * by_name, rtol=1e-6)
    np.testing.assert_allclose(out.values["x"], by_name, rtol=1e-6)
    with pytest.raises(TypeError, match="pass"):
        td.integrate(data, [doubled])
    with pytest.raises(KeyError):
        td.integrate(data, {"y": "nosuch"})


def test_integrate_carries_fields_past_hidden_cells(backend):
    data, x = _slab()
    data.ghosts = td.Ghosts(cells=np.where(x < 0.5, td.Ghosts.HIDDENCELL, 0).astype(np.uint8),
                            points=None)
    given = td.Field(td.Constant(data), tack.array(x))
    out = td.integrate(data, {"given": given, "x": "x"})
    assert out.values["given"] == pytest.approx(out.values["x"], rel=1e-6)
    assert out.measure == pytest.approx(1.5)


def test_integrate_over_pieces_takes_names(backend):
    data, x = _slab()
    pieces = td.PartitionedDataSet([data])
    assert td.integrate(pieces, {"y": "x"}).values["y"] == pytest.approx(
        td.integrate(data, "x").values["x"])
    with pytest.raises(TypeError, match="partition"):
        td.integrate(pieces, {"g": td.Field(td.Constant(data), tack.array(x))})
