"""Bases of any order on triangles and tetrahedra (round 2 of the vault's
higher-order proposal): equispaced nodes (VTK's and gmsh's), Silvester's product
basis, values shared by entity with triangle faces placed through their
orientation, Bernstein bounds, and subdivision into Freudenthal's sub-simplices
for the cut filters."""

import itertools
from math import comb, factorial

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import bases, shapes
from tack.dataset.topology import uniform_cells

pytestmark = pytest.mark.quick

_SIMPLICES = [shapes.Triangle, shapes.Tetra]


@tack.func
def _cubic(x):
    return x[0] * x[0] * x[1] - 0.7 * x[1] * x[2] * x[2] + 0.3 * x[0] + x[2] * x[2] * x[2]


def _cubic_at(p):
    return p[:, 0] ** 2 * p[:, 1] - 0.7 * p[:, 1] * p[:, 2] ** 2 + 0.3 * p[:, 0] + p[:, 2] ** 3


def _cubic_gradient(p):
    x, y, z = p.T
    return np.c_[2 * x * y + 0.3, x * x - 0.7 * z * z, -1.4 * y * z + 3 * z * z]


@tack.func
def _bowl(x):
    d = x - tack.vec3(0.6)
    return d.dot(d) + 0.2 * x[0] * x[1] * x[2]


def _bowl_at(p):
    d = p - 0.6
    return (d * d).sum(1) + 0.2 * p[:, 0] * p[:, 1] * p[:, 2]


# ── The bases, on the host ──────────────────────────────────────────

@pytest.mark.parametrize("shape", _SIMPLICES)
@pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
def test_simplex_bases_reproduce_polynomials_of_their_order(shape, order):
    """Silvester's basis on the equispaced nodes: one at its node and zero at the
    others, summing to one, reproducing any polynomial of the order; the Bernstein
    coefficients bound it."""
    lattice = bases.simplex_lattice(shape, order)
    assert len(lattice) == comb(order + shape.DIMENSION, shape.DIMENSION)
    np.testing.assert_allclose(bases.simplex_basis(shape, order, lattice / order),
                               np.eye(len(lattice)), atol=1e-12)
    rng = np.random.default_rng(order)
    points = rng.dirichlet(np.ones(shape.DIMENSION + 1), 200)[:, 1:]
    np.testing.assert_allclose(bases.simplex_basis(shape, order, points).sum(1), 1.0)
    c = np.array([0.7, -1.3, 0.4])[:shape.DIMENSION]
    f = lambda x: (1.0 + x @ c) ** order
    values = f(lattice / order)
    np.testing.assert_allclose(bases.simplex_basis(shape, order, points) @ values, f(points),
                               atol=1e-12)
    coefficients = bases.simplex_bernstein(shape, order) @ values
    assert coefficients.min() <= f(points).min() + 1e-12
    assert coefficients.max() >= f(points).max() - 1e-12


@pytest.mark.parametrize("order", [3, 4, 6])
def test_each_simplex_node_is_a_corner_edge_face_or_interior_node(order):
    p = order
    kind = bases.dof_tables(shapes.Tetra, order)[0]
    assert np.bincount(kind, minlength=4).tolist() == [
        4, 6 * (p - 1), 4 * (p - 1) * (p - 2) // 2, (p - 1) * (p - 2) * (p - 3) // 6]
    kind = bases.dof_tables(shapes.Triangle, order)[0]
    assert np.bincount(kind, minlength=4).tolist() == [3, 3 * (p - 1), 0,
                                                       (p - 1) * (p - 2) // 2]


@pytest.mark.parametrize("shape", _SIMPLICES)
@pytest.mark.parametrize("level", [1, 2, 3, 5])
def test_freudenthal_fills_the_simplex(shape, level):
    """``level^d`` sub-simplices of the lattice, each wound positively, filling the
    cell exactly."""
    rows = bases.simplex_subdivision(shape, level)
    lattice = bases.simplex_lattice(shape, level) / level
    assert len(rows) == level ** shape.DIMENSION
    measures = [np.linalg.det(lattice[r][1:] - lattice[r][0]) for r in rows]
    assert min(measures) > 0
    assert np.isclose(sum(measures), 1.0)        # the cell's, d! times its measure
    assert np.isclose(sum(measures) / factorial(shape.DIMENSION),
                      1.0 / factorial(shape.DIMENSION))


# ── Fields on simplices ─────────────────────────────────────────────

_EVEN = [p for p in itertools.permutations(range(4))
         if sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4)) % 2 == 0]


def _tetrahedra(seed=None):
    """A grid's tetrahedra; with ``seed``, each with its corners renumbered by a
    random even permutation (its orientation kept), so a shared face meets its two
    cells in every one of its orientations."""
    grid = td.uniform_grid((4, 4, 4), spacing=(0.4, 0.4, 0.4), dtype=tack.f64)
    tets = td.tetrahedralize(grid)
    if seed is None:
        return tets
    conn = tets.topology.connectivity.numpy().reshape(-1, 4)
    rng = np.random.default_rng(seed)
    out = np.array([row[list(_EVEN[rng.integers(len(_EVEN))])] for row in conn])
    types, offsets = uniform_cells(int(shapes.TETRA), len(out), tack.i32)
    topology = td.UnstructuredTopology(types, offsets,
                                       tack.array(out.reshape(-1).astype(np.int32)),
                                       num_points=tets.num_points)
    return td.DataSet(topology, tets.geometry.values)


def _triangles():
    return td.triangulate(td.uniform_grid((5, 5), spacing=(0.3, 0.3), dtype=tack.f64))


@pytest.mark.parametrize("order", [3, 4, 6])
def test_h1_and_l2_are_exact_on_tetrahedra_in_any_orientation(f64_backend, order):
    """A cubic in ``H1`` and ``L2`` of order 3 and up: probed anywhere and its cell
    gradients exact, though neighbours see their shared faces in any orientation
    (above order 3 a face holds several values, placed through its frame)."""
    data = _tetrahedra(seed=order)
    points = np.random.default_rng(order).uniform(0.02, 1.18, (300, 3))
    centers = td.cell_centers(data).values.numpy()
    for space in (td.H1(data, order=order, nodes="equispaced"),
                  td.L2(data, order=order, nodes="equispaced")):
        data.fields["u"] = td.interpolate(data, _cubic, space)
        probed = td.probe(data, points)
        assert probed.fields["valid"].values.numpy().all()
        np.testing.assert_allclose(probed.fields["u"].values.numpy(), _cubic_at(points),
                                   atol=1e-12)
        np.testing.assert_allclose(td.gradients(data, "u").values.numpy(),
                                   _cubic_gradient(centers), atol=1e-11)


@pytest.mark.parametrize("order", [3, 5])
def test_triangles_of_any_order(f64_backend, order):
    data = _triangles()
    points = np.c_[np.random.default_rng(1).uniform(0.01, 1.19, (200, 2)), np.zeros(200)]
    for space in (td.H1(data, order=order, nodes="equispaced"),
                  td.L2(data, order=order, nodes="equispaced")):
        data.fields["u"] = td.interpolate(data, _cubic, space)
        probed = td.probe(data, points)
        np.testing.assert_allclose(probed.fields["u"].values.numpy(), _cubic_at(points),
                                   atol=1e-12)


def test_other_simplex_nodes_and_layouts_are_refused(f64_backend):
    """Simplices above order 2 have equispaced nodes this round: Gauss-Lobatto (the
    default) is refused with the remedy, and so is a cell-fastest L2."""
    data = _tetrahedra()
    with pytest.raises(NotImplementedError, match='nodes="equispaced"'):
        _ = td.H1(data, order=3).size
    with pytest.raises(NotImplementedError, match='nodes="equispaced"'):
        _ = td.L2(data, order=4).size
    with pytest.raises(ValueError, match="stored in blocks"):
        _ = td.L2(data, order=3, nodes="equispaced", layout="cells").layout
    assert td.L2(data, order=3, nodes="equispaced").layout == "blocks"


# ── Subdivision and the cut filters ─────────────────────────────────

@pytest.mark.parametrize("level", [1, 2, 3, 5])
def test_subdividing_tetrahedra(f64_backend, level):
    """``level^3`` sub-tetrahedra a cell, the field exact at every lattice point
    (shared ones on edges and faces from their entity's own values)."""
    data = _tetrahedra(seed=3)
    data.fields["u"] = td.interpolate(data, _bowl, td.H1(data, order=3, nodes="equispaced"))
    refined = td.subdivide(data, level, fields=["u"])
    assert refined.num_cells == data.num_cells * level ** 3
    np.testing.assert_allclose(refined.fields["u"].values.numpy(),
                               _bowl_at(refined.positions()), atol=1e-12)
    # One point per lattice node: the corners, (level - 1) an edge, the faces' and
    # the interiors' (the equispaced order-level space's nodes).
    edges = data.topology.edges().num_edges
    faces = data.topology.faces().num_faces
    assert refined.num_points == (data.num_points + edges * (level - 1)
                                  + faces * (level - 1) * (level - 2) // 2
                                  + data.num_cells * comb(level - 1, 3))


def _open_edges(surface):
    from collections import Counter

    edges = Counter()
    for t in surface.topology.connectivity.numpy().reshape(-1, 3):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[(min(a, b), max(a, b))] += 1
    return sum(1 for count in edges.values() if count != 2)


def test_contours_of_simplices_converge_and_close(f64_backend):
    """The level set of an order-3 field on tetrahedra: closed at every subdivision
    (a lattice point shared by cells is computed once), the error falling as the
    steps' square; on triangles, lines."""
    data = _tetrahedra(seed=5)
    data.fields["u"] = td.interpolate(data, _bowl, td.H1(data, order=3, nodes="equispaced"))
    errors = []
    for steps in (3, 6, 12):
        surface = td.contour(data, "u", 0.25, subdivision=steps)
        assert _open_edges(surface) == 0
        errors.append(np.abs(_bowl_at(surface.positions()) - 0.25).max())
    assert errors[1] < errors[0] / 3 and errors[2] < errors[1] / 3
    triangles = _triangles()
    triangles.fields["u"] = td.interpolate(triangles, _bowl,
                                           td.H1(triangles, order=4, nodes="equispaced"))
    lines = td.contour(triangles, "u", 0.5, subdivision=8)
    assert lines.num_cells > 0
    assert np.abs(_bowl_at(lines.positions()) - 0.5).max() < 2e-3


def test_slices_of_tetrahedra_carry_high_order_fields(f64_backend):
    """A plane through tetrahedra holding an order-3 field: the slice lies on the
    plane, and the field carried onto it is the field there (each slice point
    interpolated in a sub-tetrahedron, so within the linearization's error)."""
    data = _tetrahedra(seed=9)
    data.fields["u"] = td.interpolate(data, _bowl, td.H1(data, order=3, nodes="equispaced"))
    cut = td.slice(data, td.implicit.Plane([0.6, 0.6, 0.6], [0.3, -0.5, 1.0]), subdivision=6)
    p = cut.positions()
    assert cut.num_cells > 0
    np.testing.assert_allclose((p - 0.6) @ np.array([0.3, -0.5, 1.0]), 0.0, atol=1e-12)
    assert np.abs(cut.fields["u"].values.numpy() - _bowl_at(p)).max() < 5e-3


def test_a_lattice_point_has_the_same_bits_in_every_piece_holding_it():
    """A lattice point on an edge or triangle face is computed from that entity's
    values in a frame its corners' positions fix, so pieces cut from the whole
    (keeping the field's order) give the whole's bits."""
    data = _tetrahedra(seed=7)
    data.fields["u"] = td.interpolate(data, _bowl, td.H1(data, order=4, nodes="equispaced"))
    whole = td.subdivide(data, 3, fields=["u"])
    at = {tuple(p): u for p, u in zip(whole.positions(), whole.fields["u"].values.numpy())}
    n = data.num_cells
    for kept in (np.arange(0, n, 2), np.arange(n - 1, -1, -3), np.arange(n // 2)):
        piece = td.subdivide(td.extract_cells(data, kept), 3, fields=["u"])
        positions = [tuple(p) for p in piece.positions()]
        assert all(p in at for p in positions)
        np.testing.assert_array_equal(piece.fields["u"].values.numpy(),
                                      [at[p] for p in positions])


# ── Curved simplices ────────────────────────────────────────────────

@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


def _curved(order, seed=11):
    """Tetrahedra bent at ``order`` (order 2's quadratic geometry, or the simplex basis
    from 3), corners shuffled."""
    data = _tetrahedra(seed=seed)
    space = (td.H1(data, order=order, nodes="equispaced") if order > 2
             else td.H1(data, order=2))
    data.geometry = td.interpolate(data, _bend, space, dtype=tack.vec3(tack.f64))
    data.fields["u"] = td.interpolate(data, _bowl, space)        # at the curved nodes
    return data


@pytest.mark.parametrize("order", [2, 3, 4])
def test_curved_tetrahedra_are_located_as_they_are(f64_backend, order):
    """Probing at the geometry's own nodes, many on its curved faces, finds every one:
    the locator bounds a curved simplex by its Bernstein control points and inverts
    its map by Newton's method, from its nodes read once."""
    data = _curved(order)
    nodes = data.geometry.values.numpy()
    probed = td.probe(data, nodes)
    assert probed.fields["valid"].values.numpy().all()
    np.testing.assert_allclose(probed.fields["u"].values.numpy(), _bowl_at(nodes), atol=1e-10)


@pytest.mark.parametrize("order", [3, 4])
def test_subdividing_curved_tetrahedra(f64_backend, order):
    """A curved simplex's lattice points on its edges and faces come from the
    geometry's own nodes there; at the field's own order's nodes the field is
    exact, and its level set closes."""
    data = _curved(order)
    refined = td.subdivide(data, order, fields=["u"])
    assert refined.num_cells == data.num_cells * order ** 3
    np.testing.assert_allclose(refined.fields["u"].values.numpy(),
                               _bowl_at(refined.positions()), atol=1e-12)
    surface = td.contour(data, "u", 0.3, subdivision=2 * order)
    assert surface.num_cells > 0 and _open_edges(surface) == 0


def test_curved_pieces_have_the_wholes_bits():
    """Pieces cut from a curved order-3 mesh by a filter (which keeps its order) give
    the whole's lattice points to the bit."""
    data = _curved(3, seed=13)
    whole = td.subdivide(data, 3, fields=["u"])
    at = {tuple(p): u for p, u in zip(whole.positions(), whole.fields["u"].values.numpy())}
    n = data.num_cells
    for kept in (np.arange(0, n, 2), np.arange(n - 1, -1, -3)):
        piece = td.subdivide(td.extract_cells(data, kept), 3, fields=["u"])
        positions = [tuple(p) for p in piece.positions()]
        assert all(p in at for p in positions)
        np.testing.assert_array_equal(piece.fields["u"].values.numpy(),
                                      [at[p] for p in positions])
