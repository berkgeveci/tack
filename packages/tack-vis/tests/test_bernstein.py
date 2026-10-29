"""Bernstein spaces (``basis="bernstein"``) on the tensor shapes: values are the
Bernstein polynomials' coefficients, not values at nodes. Every filter sees the
same polynomials as the equispaced Lagrange space of the order, so a Bernstein
dataset is held to its Lagrange twin, and its coefficients to the convex hull."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import _bernstein, bases


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Compared in f64, on every backend with it."""


@tack.func
def _cubic(x):
    return x[0] * x[0] * x[0] - 2.0 * x[1] * x[2] + 0.5 * x[1] * x[1] * x[0] + 0.3 * x[2]


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


def _cubic_at(p):
    return p[:, 0] ** 3 - 2 * p[:, 1] * p[:, 2] + 0.5 * p[:, 1] ** 2 * p[:, 0] + 0.3 * p[:, 2]


def _mesh(kind, dimension=3):
    grid = td.uniform_grid((4,) * dimension, spacing=(1 / 3,) * dimension, dtype=tack.f64)
    return grid if kind == "grid" else td.extract_cells(grid, np.arange(grid.num_cells))


def _points(dimension=3, n=150):
    p = np.random.default_rng(4).uniform(0.02, 0.98, (n, 3))
    p[:, dimension:] = 0.0
    return p


def test_the_bernstein_polynomials():
    """``weights`` are the binomials, the nodes the Greville points ``j / p``, the
    matrix to Bernstein form the identity; the spaces agree with Lagrange at order
    1 only."""
    from math import comb

    for p in (1, 2, 3, 6):
        np.testing.assert_array_equal(bases.weights("bernstein", p),
                                      [comb(p, j) for j in range(p + 1)])
        np.testing.assert_array_equal(bases.nodes("bernstein", p), bases.nodes("equispaced", p))
        np.testing.assert_array_equal(bases.bernstein("bernstein", p), np.eye(p + 1))
    data = _mesh("grid")
    assert td.H1(data, order=1, basis="bernstein") is td.H1(data)
    assert td.H1(data, order=2, basis="bernstein") is not td.H1(data, order=2)
    assert repr(td.L2(data, order=3, basis="bernstein")) == \
        "L2(order=3, basis='bernstein')"


@pytest.mark.parametrize("kind", ["grid", "cells"])
@pytest.mark.parametrize("order", [2, 3, 5])
def test_interpolation_reproduces_polynomials(kind, order):
    """A cubic interpolated into H1 or L2 of order 3 and up is the cubic, wherever it
    is probed; at order 2 the quadratic part agrees with the Lagrange twin's."""
    data = _mesh(kind)
    points = _points()
    for family in (td.H1, td.L2):
        space = family(data, order=order, basis="bernstein")
        twin = family(data, order=order, basis="equispaced")
        data.fields["u"] = td.interpolate(data, _cubic, space)
        data.fields["v"] = td.interpolate(data, _cubic, twin)
        probed = td.probe(data, points)
        u = probed.fields["u"].values.numpy()
        np.testing.assert_allclose(u, probed.fields["v"].values.numpy(), rtol=0, atol=1e-13)
        if order >= 3:
            np.testing.assert_allclose(u, _cubic_at(points), rtol=0, atol=1e-13)
        np.testing.assert_allclose(td.gradients(data, "u").values.numpy(),
                                   td.gradients(data, "v").values.numpy(), rtol=0, atol=1e-11)


@pytest.mark.parametrize("dimension", [1, 2])
def test_lines_and_quads(dimension):
    data = _mesh("cells", dimension)
    data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=3, basis="bernstein"))
    points = _points(dimension)
    np.testing.assert_allclose(td.probe(data, points).fields["u"].values.numpy(),
                               _cubic_at(points), rtol=0, atol=1e-13)


def test_coefficients_round_trip_and_bound_their_cells():
    """``to_lagrange`` and ``from_lagrange`` invert each other; each cell's field lies
    between its least and greatest coefficients (the convex hull)."""
    data = _mesh("cells")
    space = td.H1(data, order=4, basis="bernstein")
    field = td.interpolate(data, _cubic, space)
    nodal = _bernstein.to_lagrange(data, field)
    assert nodal.space is td.H1(data, order=4, basis="equispaced")
    np.testing.assert_allclose(_bernstein.from_lagrange(data, nodal, space).values.numpy(),
                               field.values.numpy(), rtol=0, atol=1e-13)
    np.testing.assert_allclose(nodal.values.numpy(),
                               td.interpolate(data, _cubic, nodal.space).values.numpy(),
                               rtol=0, atol=1e-13)
    data.fields["u"] = field
    # Each cell's coefficients (projected into L2's, cell by cell) bound its values
    # at its Gauss-Lobatto nodes.
    blocks = td.L2(data, order=4, basis="bernstein")
    per_cell = blocks.blocks(td.project(data, "u", blocks).values.numpy())
    lo, hi = per_cell.min(axis=1), per_cell.max(axis=1)
    lobatto = td.L2(data, order=4)
    values = lobatto.blocks(td.interpolate(data, _cubic, lobatto).values.numpy())
    assert (values >= lo[:, None] - 1e-13).all() and (values <= hi[:, None] + 1e-13).all()
    assert (hi - lo > 1e-3).all()


@pytest.mark.parametrize("curved", [False, True])
def test_filters_see_the_lagrange_twin(curved):
    """Subdividing, contouring (exact or not), clipping and carrying a Bernstein
    dataset -- a curved geometry in Bernstein form too -- give its equispaced
    Lagrange twin's results."""
    out = {}
    for basis in ("equispaced", "bernstein"):
        data = _mesh("cells")
        if curved:
            data.geometry = td.interpolate(data, _bend, td.H1(data, order=2, basis=basis),
                                           dtype=tack.vec3(tack.f64))
        data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=3 if not curved else 2,
                                                              basis=basis))
        whole, part = td.clip(data, "u", 0.3).partitions
        kept = td.extract_cells(data, np.arange(0, data.num_cells, 2))
        assert kept.fields["u"].space.basis == data.fields["u"].space.basis
        out[basis] = (td.subdivide(data, 2), td.contour(data, "u", 0.3),
                      td.contour(data, "u", 0.3, exact=False), part,
                      td.probe(kept, td.cell_centers(kept).values.numpy()))
    for name, a, b in zip(("subdivide", "contour", "linear contour", "clip"),
                          out["equispaced"][:4], out["bernstein"][:4]):
        assert a.num_cells == b.num_cells, name
        np.testing.assert_allclose(np.sort(b.positions(), 0), np.sort(a.positions(), 0),
                                   rtol=0, atol=1e-13, err_msg=name)
        np.testing.assert_allclose(np.sort(b.fields["u"].values.numpy()),
                                   np.sort(a.fields["u"].values.numpy()), rtol=0, atol=1e-13,
                                   err_msg=name)
    np.testing.assert_allclose(out["bernstein"][4].fields["u"].values.numpy(),
                               out["equispaced"][4].fields["u"].values.numpy(), rtol=0,
                               atol=1e-13)


def test_pieces_share_their_coefficients_to_the_bit():
    """A coefficient on an edge or face is formed from that entity's values in its
    corners' frame: pieces with a ghost layer have the whole's coefficients, bit for
    bit, so their exact contours own the whole's vertices."""
    def owned(surface):
        x = surface.positions()
        if surface.ghosts is not None and surface.ghosts.points is not None:
            x = x[(surface.ghosts.points.numpy() & td.Ghosts.SKIPPED_POINTS) == 0]
        return {tuple(p) for p in x}

    def prepared(d):
        d.fields["u"] = td.interpolate(d, _cubic, td.H1(d, order=3, basis="bernstein"))
        return d

    whole = owned(td.contour(prepared(_mesh("grid")), "u", 0.3))
    mine = set()
    count = 0
    for piece in td.split(_mesh("grid"), (2, 2, 1), ghost_layers=1):
        got = owned(td.contour(prepared(piece), "u", 0.3))
        count += len(got)
        mine |= got
    assert count == len(mine) and mine == whole


def test_other_shapes_are_refused():
    tets = td.tetrahedralize(_mesh("grid"))
    with pytest.raises(NotImplementedError, match="Bernstein basis is on segments"):
        td.interpolate(tets, _cubic, td.H1(tets, order=3, basis="bernstein"))
    with pytest.raises(ValueError, match="basis is one of"):
        td.H1(tets, order=3, basis="bezier")


def _vtk_at(grid, name, pcs):
    """VTK's own evaluation of a grid of higher-order cells: each cell's position and
    ``name`` at each of ``pcs`` (Bezier cells weigh their points by Bernstein's
    polynomials)."""
    vtk = pytest.importorskip("vtkmodules.vtkCommonDataModel")
    from vtkmodules.vtkCommonCore import reference

    cell = vtk.vtkGenericCell()
    xs, us = [], []
    array = grid.point_data[name]
    for c in range(grid.number_of_cells):
        grid.GetCell(c, cell)
        for pc in pcs:
            weights = [0.0] * cell.GetNumberOfPoints()
            x = [0.0, 0.0, 0.0]
            cell.EvaluateLocation(reference(0), list(pc), x, weights)
            xs.append(x)
            us.append(sum(w * array.GetValue(cell.GetPointId(k)) for k, w in enumerate(weights)))
    return np.array(xs), np.array(us)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_bezier_cells_are_vtks(dimension):
    """A Bernstein dataset goes to VTK as Bezier cells (``higher_order=True`` chooses
    them), which VTK evaluates as Tack does, and comes back in the same spaces; a
    Lagrange one goes as Bezier cells when asked, the same polynomials."""
    pytest.importorskip("vtkmodules.vtkCommonDataModel")
    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

    data = _mesh("cells", dimension)
    data.geometry = td.interpolate(data, _bend, td.H1(data, order=3, basis="bernstein"),
                                   dtype=tack.vec3(tack.f64))
    data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=3, basis="bernstein"))
    grid = dataset_to_vtk(data, higher_order=True)
    assert grid.GetCellType(0) == {1: 75, 2: 77, 3: 79}[dimension]
    pcs = np.random.default_rng(3).uniform(0.05, 0.95, (3, 3))
    pcs[:, dimension:] = 0.0
    xs, us = _vtk_at(grid, "u", pcs)
    # Probing curved cells inverts them by Newton's method, to about 1e-11.
    np.testing.assert_allclose(td.probe(data, xs).fields["u"].values.numpy(), us, rtol=0,
                               atol=1e-10)
    back = vtk_to_dataset(grid, dtype=tack.f64)
    assert back.fields["u"].space is td.H1(back, order=3, basis="bernstein")
    assert back.geometry.space is td.H1(back, order=3, basis="bernstein")
    np.testing.assert_allclose(td.probe(back, xs).fields["u"].values.numpy(), us, rtol=0,
                               atol=1e-10)
    nodal = td.H1(data, order=3, basis="equispaced")
    lagrange = td.DataSet(data.topology, td.project(data, data.geometry, nodal))
    lagrange.fields["u"] = td.project(data, "u", nodal)
    assert dataset_to_vtk(lagrange, higher_order=True).GetCellType(0) == \
        {1: 68, 2: 70, 3: 72}[dimension]
    bezier = dataset_to_vtk(lagrange, higher_order="bezier")
    xb, ub = _vtk_at(bezier, "u", pcs)
    np.testing.assert_allclose(xb, xs, rtol=0, atol=1e-13)
    np.testing.assert_allclose(ub, us, rtol=0, atol=1e-13)


def test_rational_and_simplex_bezier_cells_are_refused():
    pytest.importorskip("vtkmodules.vtkCommonDataModel")
    from vtkmodules.vtkCommonCore import vtkDoubleArray

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

    data = _mesh("cells", 2)
    data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=2, basis="bernstein"))
    grid = dataset_to_vtk(data, higher_order=True)
    weights = vtkDoubleArray(np.full(grid.number_of_points, 2.0))
    weights.SetName("RationalWeights")
    grid.point_data.SetRationalWeights(weights)
    with pytest.raises(NotImplementedError, match="rational Bezier"):
        vtk_to_dataset(grid)
    tets = td.tetrahedralize(_mesh("grid"))
    tets.fields["u"] = td.interpolate(tets, _cubic, td.H1(tets, order=2))
    with pytest.raises(NotImplementedError, match="Bezier cells are written for"):
        dataset_to_vtk(tets, higher_order="bezier")
    with pytest.raises(ValueError, match="higher_order is"):
        dataset_to_vtk(data, higher_order="spline")
