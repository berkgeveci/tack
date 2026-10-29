"""``H1(..., dofs=table)``: values numbered by an element-to-DOF table, as MFEM and VTK's
cell grids store them, instead of by entity. A table space holds the same
polynomials as the space numbered by entity, so a field in it is held to its twin
there, through every filter; a filter's output numbers by entity again."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes
from tack.dataset.carry import _node_entries
from tack.dataset.topology import uniform_cells


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Compared in f64, on every backend with it."""


@tack.func
def _cubic(x):
    return x[0] * x[0] * x[0] - 2.0 * x[1] * x[2] + 0.5 * x[1] * x[1] * x[0] + 0.3 * x[2]


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


_DIMENSION = {"lines": 1, "quads": 2, "triangles": 2, "hexahedra": 3, "tetrahedra": 3,
              "wedges": 3}


def _wedges(hexes):
    conn = []
    for h in hexes.topology.connectivity.numpy().reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    return td.DataSet(td.UnstructuredTopology(
        types, offsets, tack.array(np.array(conn, np.int32).reshape(-1)),
        num_points=hexes.num_points), hexes.geometry.values)


def _mesh(cells):
    d = _DIMENSION[cells]
    grid = td.uniform_grid((4,) * d, spacing=(1 / 3,) * d, dtype=tack.f64)
    tensor = td.extract_cells(grid, np.arange(grid.num_cells))
    data = {"tetrahedra": lambda: td.tetrahedralize(tensor),
            "triangles": lambda: td.triangulate(tensor),
            "wedges": lambda: _wedges(tensor)}.get(cells, lambda: tensor)()
    return td.DataSet(data.topology, data.geometry)


def _basis(cells):
    return {} if cells in ("hexahedra", "quads", "lines") else {"basis": "equispaced"}


def _table_of(data, space, seed=0):
    """The table numbering ``space``'s values for each cell's nodes (lexicographic),
    the non-corner values relabelled at random: the same space, numbered otherwise."""
    index, _, _, total = _node_entries(space, tack.zeros((space.size,), tack.f64))
    rows = index.numpy()[:total].reshape(data.num_cells, -1)
    relabel = np.arange(space.size)
    rest = relabel[data.num_points:].copy()
    np.random.default_rng(seed).shuffle(rest)
    relabel[data.num_points:] = rest
    return relabel[rows]


def _twins(cells, order, curved=False, basis=None):
    """``cells`` with ``e`` (a cubic numbered by entity) and ``t`` (the same in a table
    space), the geometry bent in a table space too when ``curved``."""
    data = _mesh(cells)
    kind = {"basis": basis} if basis else _basis(cells)
    if curved:
        bent = td.H1(data, order=2, **kind)
        data.geometry = td.interpolate(data, _bend, td.H1(data, order=2, dofs=_table_of(
            data, bent, seed=1), **kind), dtype=tack.vec3(tack.f64))
    entity = td.H1(data, order=order, **kind)
    data.fields["e"] = td.interpolate(data, _cubic, entity)
    data.fields["t"] = td.interpolate(data, _cubic, td.H1(data, order=order,
                                                          dofs=_table_of(data, entity), **kind))
    return data


def _points(cells, n=120):
    p = np.random.default_rng(4).uniform(0.02, 0.98, (n, 3))
    p[:, _DIMENSION[cells]:] = 0.0
    return p


@pytest.mark.parametrize("cells", list(_DIMENSION))
@pytest.mark.parametrize("order", [2, 3])
def test_a_table_space_holds_the_same_polynomials(cells, order):
    """Interpolated, probed, differentiated and projected, a table space's field is its
    twin's numbered by entity; the cells it is cut to number by entity again."""
    data = _twins(cells, order)
    space = data.fields["t"].space
    assert space.dofs is not None and space.size == data.fields["e"].space.size
    points = _points(cells)
    probed = td.probe(data, points)
    np.testing.assert_allclose(probed.fields["t"].values.numpy(),
                               probed.fields["e"].values.numpy(), rtol=0, atol=1e-13)
    np.testing.assert_allclose(td.gradients(data, "t").values.numpy(),
                               td.gradients(data, "e").values.numpy(), rtol=0, atol=1e-11)
    projected = td.project(data, "e", space)
    np.testing.assert_allclose(projected.values.numpy(), data.fields["t"].values.numpy(),
                               rtol=0, atol=1e-13)
    kept = td.extract_cells(data, np.arange(0, data.num_cells, 2))
    assert kept.fields["t"].space is td.H1(kept, **{k: v for k, v in space.params.items()
                                                    if k != "dofs"})
    np.testing.assert_allclose(kept.fields["t"].values.numpy(), kept.fields["e"].values.numpy(),
                               rtol=0, atol=1e-13)


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra", "wedges", "quads"])
def test_filters_see_the_twin(cells):
    """Contours (exact or not), clips and subdivisions of a table field, on a geometry
    curved in a table space, are its twin's."""
    data = _twins(cells, 2, curved=True)
    for name, cut in (("contour", lambda u: td.contour(data, u, 0.3)),
                      ("linear", lambda u: td.contour(data, u, 0.3, exact=False)),
                      ("clip", lambda u: td.clip(data, u, 0.3).partitions[1]),
                      ("subdivide", lambda u: td.subdivide(data, 2))):
        a, b = cut("e"), cut("t")
        assert a.num_cells == b.num_cells, name
        np.testing.assert_allclose(np.sort(b.positions(), 0), np.sort(a.positions(), 0),
                                   rtol=0, atol=1e-13, err_msg=name)


def test_a_bernstein_table():
    data = _twins("hexahedra", 3, basis="bernstein")
    probed = td.probe(data, _points("hexahedra"))
    np.testing.assert_allclose(probed.fields["t"].values.numpy(),
                               probed.fields["e"].values.numpy(), rtol=0, atol=1e-13)


def test_nodes_may_be_points():
    """VTK's own layout: every node of a high-order cell is a point, the cells using
    their corners. The table names the other nodes' points, which no cell uses, and
    the field's values are the points', as they come."""
    data = _mesh("quads")
    entity = td.H1(data, order=2)
    table = _table_of(data, entity)
    nodes_as_points = td.DataSet(
        td.UnstructuredTopology(data.topology.types, data.topology.offsets,
                                data.topology.connectivity, num_points=entity.size),
        np.concatenate([data.positions(), np.zeros((entity.size - data.num_points, 3))]),
        dtype=tack.f64)
    space = td.H1(nodes_as_points, order=2, dofs=table)
    assert space.size == nodes_as_points.num_points
    nodes_as_points.fields["u"] = td.interpolate(nodes_as_points, _cubic, space)
    data.fields["u"] = td.interpolate(data, _cubic, entity)
    points = _points("quads")
    np.testing.assert_allclose(td.probe(nodes_as_points, points).fields["u"].values.numpy(),
                               td.probe(data, points).fields["u"].values.numpy(), rtol=0,
                               atol=1e-13)


def test_tables_are_checked():
    data = _mesh("hexahedra")
    table = _table_of(data, td.H1(data, order=2))
    with pytest.raises(ValueError, match="rows of 27 nodes"):
        _ = td.H1(data, order=2, dofs=table[:, :20]).size
    swapped = table.copy()
    swapped[0, [0, 2]] = swapped[0, [2, 0]]
    with pytest.raises(ValueError, match="corners at its points"):
        _ = td.H1(data, order=2, dofs=swapped).size
    with pytest.raises(ValueError, match="order 2 and up"):
        td.H1(data, order=1, dofs=table)
    with pytest.raises(NotImplementedError, match="by its table"):
        td.H1(data, order=2, dofs=table)._layout()
