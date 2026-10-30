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

@pytest.mark.parametrize("family", [f for f in bases.FAMILIES if f != bases.BERNSTEIN])
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

def test_spaces_name_their_basis_only_where_they_differ():
    data = _hexes()
    assert td.H1(data, order=2, basis="equispaced") is td.H1(data, order=2)
    assert td.H1(data, order=3) is not td.H1(data, order=3, basis="equispaced")
    assert repr(td.H1(data, order=3)) == "H1(order=3, basis='gauss_lobatto')"
    with pytest.raises(ValueError, match="basis is one of"):
        td.H1(data, order=3, basis="gauss_legendre")     # H1 needs nodes at the ends
    assert td.L2(data, order=1, basis="gauss_legendre").tensor
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
def test_h1_is_exact_on_turned_hexahedra(f64_backend, order, nodes):
    data = _rotated(_hexes(), seed=order)
    assert _probe_error(data, td.H1(data, order=order, basis=nodes), _points(data)) < 1e-12


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("nodes", [f for f in bases.FAMILIES if f != bases.SERENDIPITY])
def test_l2_is_exact(f64_backend, order, nodes):
    data = _rotated(_hexes(), seed=1)
    assert _probe_error(data, td.L2(data, order=order, basis=nodes), _points(data)) < 1e-12


@pytest.mark.parametrize("make", [
    lambda: td.uniform_grid((4, 3, 3), spacing=(0.5, 0.6, 0.4), dtype=tack.f64),
    lambda: td.uniform_grid((4, 3, 1), spacing=(0.5, 0.6, 1.0), dtype=tack.f64),
    lambda: td.extract_cells(td.uniform_grid((4, 3, 1), spacing=(0.5, 0.6, 1.0),
                                             dtype=tack.f64), np.arange(6)),
], ids=["structured", "structured 2D", "quads"])
def test_grids_and_quads(f64_backend, make):
    data = make()
    points = _points(data)
    for order in (3, 5):
        assert _probe_error(data, td.H1(data, order=order), points) < 1e-12
        assert _probe_error(data, td.L2(data, order=order, basis="gauss_legendre"),
                            points) < 1e-12


def test_f32_values_evaluate_in_f32(backend):
    data = _hexes(tack.f32)
    assert _probe_error(data, td.H1(data, order=4), _points(data)) < 1e-5


def test_project_places_the_geometry_at_the_nodes(f64_backend):
    """``project`` of the straight geometry into an order-p space evaluates it at the
    nodes, as ``interpolate`` of the identity does; each shared node once, to the
    same bits however many cells hold it."""
    data = _rotated(_hexes(), seed=3)
    for space in (td.H1(data, order=3), td.H1(data, order=4, basis="equispaced"),
                  td.L2(data, order=3, basis="gauss_legendre")):
        projected = td.project(data, data.geometry, space).values.numpy()
        interpolated = td.interpolate(data, _identity, space, dtype=tack.vec3(tack.f64))
        np.testing.assert_allclose(projected, interpolated.values.numpy(), atol=1e-15)


def test_gradients_are_exact(f64_backend):
    data = _rotated(_hexes(), seed=5)
    data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=3))
    centers = td.cell_centers(data).values.numpy()
    x, y, z = centers.T
    want = np.stack([3 * x ** 2 + y * z ** 2, -4 * y * z + x * z ** 2,
                     -2 * y ** 2 + 2 * x * y * z + 1.5 * z ** 2], 1)
    got = td.gradients(data, "u", at="cells").values.numpy()
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_other_shapes_are_refused_above_order_2():
    """Pyramids have no bases above order 2 yet; simplices and wedges have equispaced
    ones (``test_simplex_order.py``), not Gauss-Legendre for L2."""
    types, offsets = uniform_cells(int(shapes.PYRAMID), 1, tack.i32)
    pyramid = td.DataSet(td.UnstructuredTopology(types, offsets, tack.array(
        np.arange(5, dtype=np.int32)), num_points=5), np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], np.float32))
    with pytest.raises(NotImplementedError, match="orders above 2 are on segments"):
        _ = td.H1(pyramid, order=3).size
    tets = td.tetrahedralize(_hexes())
    with pytest.raises(NotImplementedError, match="on segments, quads"):
        _ = td.L2(tets, order=1, basis="gauss_legendre").size


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
def test_curved_cells_are_located_as_they_are(f64_backend, cells, order):
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



@pytest.mark.parametrize("cells, order", [("hexahedra", 2), ("hexahedra", 3), ("hexahedra", 4),
                                          ("tetrahedra", 2), ("wedges", 2)])
def test_filters_keep_curved_geometry_and_high_order_fields(f64_backend, cells, order):
    """A filter that keeps whole cells carries an ``H1`` field above order 1, and a
    curved geometry, at its order and nodes: each output cell's nodes are its
    source cell's, placed through the output's own edges and faces (which a
    reversed subset numbers and orients differently). They kept their corner
    values only: every filter's piece of a curved mesh came out straight. At
    each kept cell's center, which only it holds, the output gives the input's
    value to the bit."""
    grid = td.uniform_grid((4, 4, 4), spacing=(0.4, 0.4, 0.4), dtype=tack.f64)
    data = {"hexahedra": lambda: _rotated(td.extract_cells(grid, np.arange(27)), seed=7),
            "tetrahedra": lambda: td.tetrahedralize(grid),
            "wedges": lambda: _wedges(grid)}[cells]()
    space = td.H1(data, order=order, basis="equispaced") if order > 2 else td.H1(data, order=2)
    data.fields["u"] = td.interpolate(data, _quadratic, space)
    data.geometry = td.interpolate(data, _bend, td.H1(data, order=2), dtype=tack.vec3(tack.f64))
    n = data.num_cells
    for out in (td.extract_cells(data, np.arange(n - 1, -1, -2)),       # reversed: new frames
                td.threshold(data, "u", 0.5, 1.5).partitions[0],      # whole cells
                td.extract_cells(data, np.arange(n))):
        assert out.geometry.space == td.H1(out, order=2)
        assert out.fields["u"].space == td.H1(out, **space.params)
        centers = td.cell_centers(out).values.numpy()
        a, b = td.probe(data, centers), td.probe(out, centers)
        assert b.fields["valid"].values.numpy().all()
        np.testing.assert_array_equal(b.fields["u"].values.numpy(), a.fields["u"].values.numpy())


def test_pieces_of_a_curved_grid_stay_curved(f64_backend):
    """``split`` carries an order-3 field onto each structured piece, ghost layers
    included, and the pieces give the whole's values."""
    grid = td.uniform_grid((5, 5, 5), spacing=(0.3, 0.3, 0.3), dtype=tack.f64)
    grid.fields["u"] = td.interpolate(grid, _quadratic, td.H1(grid, order=3))
    for piece in td.split(grid, (2, 2, 1), ghost_layers=1):
        assert piece.fields["u"].space.order == 3
        centers = td.cell_centers(piece).values.numpy()
        np.testing.assert_array_equal(td.probe(piece, centers).fields["u"].values.numpy(),
                                      td.probe(grid, centers).fields["u"].values.numpy())

@tack.func
def _bulge(x):
    return [x[0] + 0.3 * x[1] * (1.0 - x[1]), x[1], x[2]]


def test_a_curved_cells_box_holds_it(f64_backend):
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


# ── Subdivision ─────────────────────────────────────────────────────

@tack.func
def _sphere(x):
    return ((x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 1.0) * (x[1] - 1.0)
            + (x[2] - 1.0) * (x[2] - 1.0))


def _ball(order, seed=11):
    grid = td.uniform_grid((5, 5, 5), spacing=(0.5, 0.5, 0.5), dtype=tack.f64)
    data = _rotated(td.extract_cells(grid, np.arange(grid.num_cells)), seed=seed)
    data.fields["u"] = td.interpolate(data, _sphere, td.H1(data, order=order))
    data.fields["c"] = td.Field(td.Constant(data),
                                tack.array(np.arange(data.num_cells, dtype=np.float64)))
    return data


def _open_edges(surface):
    from collections import Counter

    edges = Counter()
    for t in surface.topology.connectivity.numpy().reshape(-1, 3):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[(min(a, b), max(a, b))] += 1
    return sum(1 for count in edges.values() if count != 2)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_subdivide_has_each_lattice_point_once(f64_backend, level):
    """``subdivide`` at level ``s``: level^3 hexahedra a cell, one point per node of the
    equispaced order-``s`` space (each shared lattice point once), fields exact at
    the lattice points, cell fields on every sub-cell."""
    data = _rotated(_hexes(), seed=2)
    data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=3))
    data.fields["c"] = td.Field(td.Constant(data),
                                tack.array(np.arange(data.num_cells, dtype=np.float64)))
    out = td.subdivide(data, level)
    points = data.num_points if level == 1 else td.H1(data, order=level).size
    assert (out.num_cells, out.num_points) == (data.num_cells * level ** 3, points)
    x = out.positions()
    np.testing.assert_allclose(out.fields["u"].values.numpy(), _cubic_at(x), atol=1e-12)
    assert np.bincount(out.fields["c"].values.numpy().astype(int)).tolist() == \
        [level ** 3] * data.num_cells
    assert out.num_cells and td.subdivide(data).num_cells == data.num_cells * 27   # order 3


def test_contours_of_high_order_fields_converge_and_close(f64_backend):
    """A sphere's field at order 3 contoured on lattices of 1 to 5 steps: no open
    edge, so no cracks between cells turned every way; the radius error falls as
    the square of the step between lattice values (``exact=False``); level 1 is
    then the corners' linear contour. At the field's roots (the default) every
    vertex is on the sphere, at every level."""
    data = _ball(3)
    errors = []
    for level in (1, 2, 3, 5):
        surface = td.contour(data, "u", 0.49, subdivision=level, exact=False)
        assert _open_edges(surface) == 0
        assert (surface.fields["c"].values.numpy() >= 0).all()          # carried from parents
        errors.append(np.abs(np.linalg.norm(surface.positions() - 1.0, axis=1) - 0.7).max())
        exact = td.contour(data, "u", 0.49, subdivision=level)
        assert _open_edges(exact) == 0 and exact.num_cells == surface.num_cells
        assert np.abs(np.linalg.norm(exact.positions() - 1.0, axis=1) - 0.7).max() < 1e-12
    assert errors == sorted(errors, reverse=True) and errors[-1] < 4e-3
    linear = _ball(1)
    corners = td.contour(linear, "u", 0.49)
    np.testing.assert_array_equal(
        np.sort(td.contour(data, "u", 0.49, subdivision=1, exact=False).positions(), 0),
        np.sort(corners.positions(), 0))
    assert td.contour(data, "u", 0.49).num_cells == td.contour(data, "u", 0.49,
                                                               subdivision=3).num_cells


def test_only_cells_that_can_hold_the_value_are_cut(f64_backend):
    """Culling by the Bernstein bound gives the surface subdividing every cell gives."""
    data = _ball(3)
    culled = td.contour(data, "u", 0.49, subdivision=3, exact=False)
    whole = td.contour(td.subdivide(data, 3), "u", 0.49)
    np.testing.assert_allclose(np.sort(culled.positions(), 0), np.sort(whole.positions(), 0),
                               atol=1e-12)


def test_slices_of_a_curved_geometry_lie_on_their_plane(f64_backend):
    data = _ball(1)
    data.geometry = td.interpolate(data, _bend, td.H1(data, order=3), dtype=tack.vec3(tack.f64))
    normal = np.array([0.3, 0.4, 1.0])
    cut = td.slice_plane(data, [1.0, 1.0, 1.0], normal)
    assert cut.num_cells > td.slice_plane(data, [1.0, 1.0, 1.0], normal, subdivision=1).num_cells
    np.testing.assert_allclose((cut.positions() - 1.0) @ (normal / np.linalg.norm(normal)), 0,
                               atol=1e-12)
    assert _open_edges(cut) > 0                      # a slice is open, where the mesh ends


# ── VTK's Lagrange cells ────────────────────────────────────────────

def _vtk_values_at(grid, name, pcs):
    """VTK's own evaluation of a Lagrange grid: each cell's position and ``name`` at
    each of ``pcs``."""
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


@pytest.mark.parametrize("order", [2, 3])
def test_lagrange_cells_are_vtks(f64_backend, order):
    """``dataset_to_vtk(..., higher_order=True)`` writes VTK Lagrange hexahedra that VTK
    evaluates as Tack does (positions and values at random parametric points), each
    shared node one point; ``vtk_to_dataset`` reads them back into the same spaces."""
    pytest.importorskip("vtkmodules.vtkCommonDataModel")
    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

    data = _rotated(_hexes(), seed=4)
    space = td.H1(data, order=order)
    data.geometry = td.interpolate(data, _bend, space, dtype=tack.vec3(tack.f64))
    data.fields["u"] = td.interpolate(data, _cubic, space)
    data.fields["c"] = td.Field(td.Constant(data),
                                tack.array(np.arange(data.num_cells, dtype=np.float64)))
    grid = dataset_to_vtk(data, higher_order=True)
    assert set(np.asarray(grid.cell_types).tolist()) == {72}
    assert grid.number_of_points == td.H1(data, order=order, basis="equispaced").size
    pcs = np.random.default_rng(order).uniform(0.05, 0.95, (3, 3))
    xs, us = _vtk_values_at(grid, "u", pcs)
    probed = td.probe(data, xs)
    assert probed.fields["valid"].values.numpy().all()
    np.testing.assert_allclose(probed.fields["u"].values.numpy(), us, atol=1e-8)
    back = vtk_to_dataset(grid, dtype=tack.f64)
    assert back.geometry.space is td.H1(back, order=order, basis="equispaced")
    np.testing.assert_allclose(td.probe(back, xs).fields["u"].values.numpy(), us, atol=1e-8)
    np.testing.assert_array_equal(back.fields["c"].values.numpy(), np.arange(data.num_cells))


def test_vtks_own_lagrange_hexahedra(f64_backend):
    """VTK's Lagrange hexahedra of order 3 from ``vtkCellTypeSource``, a cubic of the
    points as point data: Tack reads them as ``H1`` of order 3 on equispaced nodes
    and evaluates them as VTK does."""
    vtk = pytest.importorskip("vtkmodules.vtkFiltersSources")
    from vtkmodules.vtkCommonCore import vtkDoubleArray

    from tack.interop.vtk import vtk_to_dataset

    source = vtk.vtkCellTypeSource(cell_type=72, cell_order=3, blocks_dimensions=(2, 2, 2))
    grid = source()
    x = np.asarray(grid.points.data, np.float64)
    grid.point_data["u"] = vtkDoubleArray(_cubic_at(x))
    data = vtk_to_dataset(grid, dtype=tack.f64)
    assert data.fields["u"].space is td.H1(data, order=3, basis="equispaced")
    assert data.num_cells == 8
    pcs = np.random.default_rng(9).uniform(0.05, 0.95, (4, 3))
    xs, us = _vtk_values_at(grid, "u", pcs)
    # Within the locator's Newton tolerance.
    np.testing.assert_allclose(td.probe(data, xs).fields["u"].values.numpy(), us, atol=1e-7)
    np.testing.assert_allclose(us, _cubic_at(xs), atol=1e-7)       # VTK's own, near exact


@pytest.mark.parametrize("order", [2, 3])
def test_a_tolerance_chooses_the_steps(f64_backend, order):
    """``tolerance=``: the fewest lattice steps whose estimated error (the Bernstein
    coefficients' distance from the corners' interpolation, over the steps
    squared) is within it -- the field on the surface stays within it, and a
    tighter one takes more steps."""
    data = _ball(order)
    counts = []
    for tolerance in (1e-1, 1e-2, 1e-3):
        surface = td.contour(data, "u", 0.49, tolerance=tolerance)
        x = surface.positions()
        assert np.abs(((x - 1.0) ** 2).sum(1) - 0.49).max() <= tolerance
        assert _open_edges(surface) == 0
        counts.append(surface.num_cells)
    assert counts == sorted(counts) and counts[0] < counts[-1]
    with pytest.raises(ValueError, match="not both"):
        td.contour(data, "u", 0.49, subdivision=2, tolerance=1e-2)


def test_an_order_two_bound_holds_the_interior(f64_backend):
    """An order-2 field on hexahedra that is zero at every node but the cells'
    centers: only the interior's Bernstein coefficient sees it, and the cells are
    still cut."""
    grid = td.uniform_grid((3, 3, 3), dtype=tack.f64)
    data = td.extract_cells(grid, np.arange(grid.num_cells))
    space = td.H1(data, order=2)
    values = np.zeros(space.size)
    values[-data.num_cells:] = 1.0                     # the cells' centers come last
    surface = td.contour(data, td.Field(space, tack.array(values)), 0.5)
    assert surface.num_cells > 0 and _open_edges(surface) == 0



def test_everything_in_f32_on_every_backend(backend):
    """The f32 path on every backend, Metal included (no f64): an order-4 field, a
    curved order-3 geometry located at its nodes, a closed contour on sub-cells,
    subdivide and a tolerance."""
    grid = td.uniform_grid((4, 4, 4), spacing=(0.5, 0.5, 0.5))
    data = td.extract_cells(grid, np.arange(grid.num_cells))
    data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=4))
    points = _points(data, 50)
    got = td.probe(data, points).fields["u"].values.numpy()
    assert np.abs(got - _cubic_at(points)).max() < 1e-4
    ball = td.extract_cells(td.uniform_grid((5, 5, 5), spacing=(0.5, 0.5, 0.5)),
                            np.arange(64))
    ball.fields["u"] = td.interpolate(ball, _sphere, td.H1(ball, order=3))
    surface = td.contour(ball, "u", 0.49, subdivision=3)
    assert _open_edges(surface) == 0
    assert np.abs(np.linalg.norm(surface.positions() - 1.0, axis=1) - 0.7).max() < 1e-2
    within = td.contour(ball, "u", 0.49, tolerance=1e-2)
    x = within.positions()
    assert np.abs(((x - 1.0) ** 2).sum(1) - 0.49).max() <= 1e-2
    assert td.subdivide(ball, 2).num_cells == 8 * ball.num_cells
    curved = td.extract_cells(td.uniform_grid((3, 3, 3), spacing=(0.5, 0.5, 0.5)),
                              np.arange(8))
    space = td.H1(curved, order=3)
    curved.geometry = td.interpolate(curved, _bend, space, dtype=tack.vec3())
    curved.fields["u"] = td.interpolate(curved, _quadratic, space)
    nodes = curved.geometry.values.numpy()
    probed = td.probe(curved, nodes)
    assert probed.fields["valid"].values.numpy().all()
    np.testing.assert_allclose(probed.fields["u"].values.numpy(), _quadratic_at(nodes),
                               atol=1e-4)
