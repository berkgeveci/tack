"""Bases of any order on the tensor shapes (``tack.dataset.bases``): ``H1`` and ``L2``
of order 3 and up on segments, quads, pixels, hexahedra and voxels, on
Gauss-Lobatto, equispaced or (``L2``) Gauss-Legendre nodes.

A space of order ``p`` represents every polynomial of degree ``p`` in each
coordinate exactly on straight cells, so interpolating one and probing it
anywhere gives it back to rounding. On hexahedra each turned by a random one of
the cube's 24 rotations, cells see their shared edges and faces in every
orientation: a cell reading a shared node the wrong way round would show.
"""

import itertools

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import bases, shapes
from tack.dataset.topology import uniform_cells

pytestmark = pytest.mark.quick


@tack.func
def _cubic(x):
    return (x[0] * x[0] * x[0] - 2.0 * x[1] * x[1] * x[2] + x[0] * x[1] * x[2] * x[2]
            + 0.5 * x[2] * x[2] * x[2] + 1.0)


def _cubic_at(p):
    return (p[:, 0] ** 3 - 2 * p[:, 1] ** 2 * p[:, 2] + p[:, 0] * p[:, 1] * p[:, 2] ** 2
            + 0.5 * p[:, 2] ** 3 + 1)


@tack.func
def _identity(x):
    return [x[0], x[1], x[2]]


_CORNERS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])


def _rotations():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            r[range(3), perm] = signs
            if np.linalg.det(r) > 0:
                out.append(r)
    return out


def _rotated(data, seed):
    """``data``'s hexahedra, each with its points renumbered by a random rotation of
    the cube, so neighbours see a shared face in any of its orientations."""
    conn = data.topology.connectivity.numpy().reshape(-1, 8)
    rotations = _rotations()
    rng = np.random.default_rng(seed)
    out = conn.copy()
    for c in range(len(conn)):
        r = rotations[rng.integers(len(rotations))]
        for k in range(8):
            image = ((r @ (2 * _CORNERS[k] - 1)) + 1) // 2
            out[c, k] = conn[c, np.flatnonzero((_CORNERS == image).all(1))[0]]
    types, offsets = uniform_cells(int(shapes.HEXAHEDRON), len(out), tack.i32)
    topology = td.UnstructuredTopology(types, offsets,
                                       tack.array(out.reshape(-1).astype(np.int32)),
                                       num_points=data.num_points)
    return td.DataSet(topology, data.geometry.values)


def _hexes(dtype=tack.f64):
    grid = td.rectilinear_grid([0, 0.5, 1.2, 2], [0, 1, 1.5, 1.9], [0, 0.7, 1.0], dtype=dtype)
    return td.extract_cells(grid, np.arange(grid.num_cells))


def _probe_error(data, space, points):
    data.fields["u"] = td.interpolate(data, _cubic, space)
    got = td.probe(data, points).fields["u"].values.numpy()
    return np.abs(got - _cubic_at(points)).max()


def _points(data, n=200, seed=0):
    x = data.positions()
    return np.random.default_rng(seed).uniform(x.min(0), x.max(0), (n, 3))


# ── Nodes and tables ────────────────────────────────────────────────

@pytest.mark.parametrize("family", bases.FAMILIES)
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 8])
def test_nodes_are_symmetric_and_their_polynomials_reproduce(family, order):
    x = bases.nodes(family, order)
    assert len(x) == order + 1 and np.all(np.diff(x) > 0)
    assert np.all(x + x[::-1] == 1.0)                    # to the last bit
    assert (x[0] == 0.0 and x[-1] == 1.0) == (family in bases.CLOSED)
    w = bases.weights(family, order)
    t = np.linspace(-0.1, 1.1, 13)
    values = np.array([[w[j] * np.prod([s - x[m] for m in range(order + 1) if m != j])
                        for j in range(order + 1)] for s in t])
    for degree in range(order + 1):
        np.testing.assert_allclose(values @ x ** degree, t ** degree, atol=1e-12)


def test_gauss_lobatto_nodes_are_the_known_ones():
    np.testing.assert_allclose(bases.nodes("gauss_lobatto", 4),
                               0.5 + 0.5 * np.array([-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1]),
                               atol=1e-15)
    np.testing.assert_allclose(bases.nodes("gauss_legendre", 1), 0.5 + 0.5 / np.sqrt(3) * np.array([-1, 1]),
                               atol=1e-15)


@pytest.mark.parametrize("shape, counts", [
    (shapes.Line, (2, 0, 0, 2)), (shapes.Quad, (4, 8, 0, 4)), (shapes.Pixel, (4, 8, 0, 4)),
    (shapes.Hexahedron, (8, 24, 24, 8)), (shapes.Voxel, (8, 24, 24, 8))])
def test_each_node_is_a_corner_edge_face_or_interior_node(shape, counts):
    kind, entity, a, b, corner = bases.dof_tables(shape, 3)
    assert tuple(np.bincount(kind, minlength=4)) == counts
    assert sorted(corner) == sorted(np.flatnonzero(kind == bases.CORNER))
    assert np.all((a[kind == bases.EDGE] >= 1) & (a[kind == bases.EDGE] <= 2))
    interior = np.sort(a[kind == bases.INTERIOR])
    np.testing.assert_array_equal(interior, np.arange(len(interior)))


# ── Spaces ──────────────────────────────────────────────────────────

def test_spaces_name_their_nodes_only_where_they_differ():
    data = _hexes()
    assert td.H1(data, order=2, nodes="equispaced") is td.H1(data, order=2)
    assert td.H1(data, order=3) is not td.H1(data, order=3, nodes="equispaced")
    assert repr(td.H1(data, order=3)) == "H1(order=3, nodes='gauss_lobatto')"
    with pytest.raises(ValueError, match="nodes are one of"):
        td.H1(data, order=3, nodes="gauss_legendre")     # H1 needs nodes at the ends
    assert td.L2(data, order=1, nodes="gauss_legendre").tensor
    assert not td.L2(data, order=1).tensor


def test_an_h1_space_holds_each_entitys_nodes_once():
    data = _hexes()
    t = data.topology
    for p in (3, 4):
        size = (data.num_points + t.edges().num_edges * (p - 1)
                + t.faces().num_faces * (p - 1) ** 2 + data.num_cells * (p - 1) ** 3)
        assert td.H1(data, order=p).size == size
        assert td.L2(data, order=p).size == data.num_cells * (p + 1) ** 3


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("nodes", ["gauss_lobatto", "equispaced"])
def test_h1_is_exact_on_turned_hexahedra(backend, order, nodes):
    data = _rotated(_hexes(), seed=order)
    assert _probe_error(data, td.H1(data, order=order, nodes=nodes), _points(data)) < 1e-12


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("nodes", bases.FAMILIES)
def test_l2_is_exact(backend, order, nodes):
    data = _rotated(_hexes(), seed=1)
    assert _probe_error(data, td.L2(data, order=order, nodes=nodes), _points(data)) < 1e-12


@pytest.mark.parametrize("make", [
    lambda: td.uniform_grid((4, 3, 3), spacing=(0.5, 0.6, 0.4), dtype=tack.f64),
    lambda: td.uniform_grid((4, 3, 1), spacing=(0.5, 0.6, 1.0), dtype=tack.f64),
    lambda: td.extract_cells(td.uniform_grid((4, 3, 1), spacing=(0.5, 0.6, 1.0),
                                             dtype=tack.f64), np.arange(6)),
], ids=["structured", "structured 2D", "quads"])
def test_grids_and_quads(backend, make):
    data = make()
    points = _points(data)
    for order in (3, 5):
        assert _probe_error(data, td.H1(data, order=order), points) < 1e-12
        assert _probe_error(data, td.L2(data, order=order, nodes="gauss_legendre"),
                            points) < 1e-12


def test_f32_values_evaluate_in_f32(backend):
    data = _hexes(tack.f32)
    assert _probe_error(data, td.H1(data, order=4), _points(data)) < 1e-5


def test_project_places_the_geometry_at_the_nodes(backend):
    """``project`` of the straight geometry into an order-p space evaluates it at the
    nodes, as ``interpolate`` of the identity does; each shared node once, to the
    same bits however many cells hold it."""
    data = _rotated(_hexes(), seed=3)
    for space in (td.H1(data, order=3), td.H1(data, order=4, nodes="equispaced"),
                  td.L2(data, order=3, nodes="gauss_legendre")):
        projected = td.project(data, data.geometry, space).values.numpy()
        interpolated = td.interpolate(data, _identity, space, dtype=tack.vec3(tack.f64))
        np.testing.assert_allclose(projected, interpolated.values.numpy(), atol=1e-15)


def test_gradients_are_exact(backend):
    data = _rotated(_hexes(), seed=5)
    data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=3))
    centers = td.cell_centers(data).values.numpy()
    x, y, z = centers.T
    want = np.stack([3 * x ** 2 + y * z ** 2, -4 * y * z + x * z ** 2,
                     -2 * y ** 2 + 2 * x * y * z + 1.5 * z ** 2], 1)
    got = td.gradients(data, "u", at="cells").values.numpy()
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_other_shapes_are_refused_above_order_2():
    tets = td.tetrahedralize(_hexes())
    with pytest.raises(NotImplementedError, match="orders above 2 are on segments"):
        _ = td.H1(tets, order=3).size
    with pytest.raises(NotImplementedError, match="on segments, quads"):
        _ = td.L2(tets, order=1, nodes="gauss_legendre").size


# ── Curved geometry ─────────────────────────────────────────────────

@tack.func
def _quadratic(x):
    return x[0] * x[0] - 2.0 * x[1] * x[1] + x[0] * x[1] + 1.0


def _quadratic_at(p):
    return p[:, 0] ** 2 - 2 * p[:, 1] ** 2 + p[:, 0] * p[:, 1] + 1


@tack.func
def _bend(x):
    return [x[0] + 0.15 * x[1] * x[1], x[1] + 0.2 * x[0] * x[0], x[2] + 0.1 * x[0] * x[1]]


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


@pytest.mark.parametrize("cells, order", [("hexahedra", 2), ("hexahedra", 3), ("hexahedra", 4),
                                          ("tetrahedra", 2), ("wedges", 2)])
def test_curved_cells_are_located_as_they_are(backend, cells, order):
    """A curved geometry -- the grid bent by a polynomial of its order -- and a field
    in its own space: probing at the geometry's nodes, many of them on the curved
    faces, finds every one and gives the field back. The
    locator bounds a cell by its Bernstein control points and inverts its own map
    by Newton's method; it bounded and inverted the corners before, at order 2
    as well."""
    grid = td.uniform_grid((3, 3, 3), spacing=(0.5, 0.5, 0.5), dtype=tack.f64)
    data = {"hexahedra": lambda: _rotated(td.extract_cells(grid, np.arange(8)), seed=7),
            "tetrahedra": lambda: td.tetrahedralize(grid),
            "wedges": lambda: _wedges(grid)}[cells]()
    space = td.H1(data, order=order)
    data.geometry = td.interpolate(data, _bend, space, dtype=tack.vec3(tack.f64))
    data.fields["u"] = td.interpolate(data, _quadratic, space)
    nodes = data.geometry.values.numpy()
    probed = td.probe(data, nodes)
    assert probed.fields["valid"].values.numpy().all()
    np.testing.assert_allclose(probed.fields["u"].values.numpy(), _quadratic_at(nodes),
                               atol=1e-8)


@tack.func
def _bulge(x):
    return [x[0] + 0.3 * x[1] * (1.0 - x[1]), x[1], x[2]]


def test_a_curved_cells_box_holds_it(backend):
    """The box of a curved cell's Bernstein control points holds all of it, though the
    cell bulges past the box of its nodes, and every point of it is found: a unit
    cube bent by a quadratic (exact at order 3), positions at a lattice of
    parametric points worked out on the host."""
    from tack.dataset.locator import CellLocator

    data = td.extract_cells(td.uniform_grid((2, 2, 2), dtype=tack.f64), np.arange(1))
    data.geometry = td.interpolate(data, _bulge, td.H1(data, order=3),
                                   dtype=tack.vec3(tack.f64))
    t = np.linspace(0, 1, 9)
    pcs = np.array([(a, b, c) for a in t for b in t for c in t])
    x = pcs.copy()
    x[:, 0] += 0.3 * pcs[:, 1] * (1 - pcs[:, 1])
    nodes = data.geometry.values.numpy()
    assert x[:, 0].max() > nodes[:, 0].max() + 1e-3          # past the nodes' box
    locator = CellLocator(data)
    lo, hi = locator.lo.numpy()[0], locator.hi.numpy()[0]
    assert (x >= lo - 1e-12).all() and (x <= hi + 1e-12).all()
    assert (locator.find(x).cells.numpy() == 0).all()
