"""Every field view's nodes: ``NODES`` per cell, ``node_parametric(k)`` and
``dof_index(c, k)``, numbered as ``dof(c, k)``, on every shape, space, order and
basis. Checked against ``td.interpolate``, which places each space's values at its
nodes, and a kernel writing ``p + q`` at ``q``'s nodes against ``project`` and add,
on segments too, whose order-2 view has no quadratic nodes of the shape's."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes
from tack.dataset.topology import uniform_cells


@tack.kernel
def _at_nodes(cells, q, x, out):
    for c in cells:
        for k in range(q.NODES):
            out[q.dof_index(c, k)] = x.value(c, q.node_parametric(k))[0]


@tack.kernel
def _add_at_nodes(cells, p, q, out):
    for c in cells:
        for k in range(q.NODES):
            out[q.dof_index(c, k)] = p.value(c, q.node_parametric(k)) + q.dof(c, k)


@tack.func
def _first(y):
    return y[0]


@tack.func
def _bowl(y):
    return y[0] * y[0] + y[1] * y[2] - 0.5 * y[2] * y[0]


def _wedges(hexes):
    conn = []
    for h in hexes.topology.connectivity.numpy().reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    return td.DataSet(td.UnstructuredTopology(
        types, offsets, tack.array(np.array(conn, np.int32).reshape(-1)),
        num_points=hexes.num_points), hexes.geometry.values)


def _mesh(name):
    hexes = lambda: td.extract_cells(td.uniform_grid((3, 3, 3)), np.arange(8))
    return {"segments": lambda: td.uniform_grid((6,)),
            "quads": lambda: td.uniform_grid((4, 4)),
            "hexahedra": lambda: td.uniform_grid((3, 3, 3)),
            "unstructured hexahedra": hexes,
            "triangles": lambda: td.triangulate(td.uniform_grid((4, 4))),
            "tetrahedra": lambda: td.tetrahedralize(td.uniform_grid((3, 3, 3))),
            "wedges": lambda: _wedges(hexes())}[name]()


_MESHES = ["segments", "quads", "hexahedra", "unstructured hexahedra", "triangles",
           "tetrahedra", "wedges"]
_SPACES = {
    "Constant": lambda d: td.Constant(d),
    "H1(1)": lambda d: td.H1(d),
    "H1(2)": lambda d: td.H1(d, order=2),
    "H1(3)": lambda d: td.H1(d, order=3),
    "H1(2, bernstein)": lambda d: td.H1(d, order=2, basis="bernstein"),
    "H1(2, serendipity)": lambda d: td.H1(d, order=2, basis="serendipity"),
    "L2(0)": lambda d: td.L2(d, order=0),
    "L2(1)": lambda d: td.L2(d, order=1),
    "L2(2)": lambda d: td.L2(d, order=2),
    "L2(3, cells)": lambda d: td.L2(d, order=3, layout="cells"),
}
_TENSOR = ("segments", "quads", "hexahedra", "unstructured hexahedra")


def _cases():
    for mesh in _MESHES:
        for space in _SPACES:
            if space.endswith("serendipity)") and mesh == "segments":
                continue                                  # serendipity has no segments
            if space.endswith("cells)") and mesh not in _TENSOR:
                continue                                  # simplices' L2 is in blocks
            yield mesh, space


def _launch(kernel, data, fields, *args):
    for group in data.launch_groups("cells", fields):
        kernel(data.domain_view("cells", group), *(f.view(group) for f in fields), *args)


@pytest.mark.parametrize("mesh, space", list(_cases()))
def test_nodes_are_where_interpolate_puts_them(mesh, space):
    """``x`` at each ``node_parametric(k)``, written at ``dof_index(c, k)``, is
    ``interpolate``'s ``x``: every stored value written, each where its node is."""
    data = _mesh(mesh)
    space = _SPACES[space](data)
    q = td.Field(space, tack.zeros((space.size,), tack.f32))
    out = tack.full((space.size,), np.nan, tack.f32)
    _launch(_at_nodes, data, [q, data.geometry], out)
    want = td.interpolate(data, _first, space).values.numpy()
    np.testing.assert_allclose(out.numpy(), want, atol=1e-6)


@pytest.mark.parametrize("mesh, space", list(_cases()))
def test_a_lower_order_field_added_at_the_nodes_is_project_and_add(mesh, space):
    """``p + q`` at ``q``'s nodes, ``p`` of order 1 read through its own basis, is
    ``project(p) + q``: exact, since ``p`` is in the space (Bernstein included, its
    coefficients a multilinear ``p``'s values at the Greville points)."""
    data = _mesh(mesh)
    space = _SPACES[space](data)
    p = td.interpolate(data, _bowl, td.H1(data))
    q = td.interpolate(data, _bowl, space)
    out = tack.full((space.size,), np.nan, tack.f32)
    _launch(_add_at_nodes, data, [p, q], out)
    want = td.project(data, p, space).values.numpy() + q.values.numpy()
    np.testing.assert_allclose(out.numpy(), want, atol=2e-6)


def test_serendipity_visits_the_nodes_it_stores():
    """A serendipity view reads face centres and insides the full space would store,
    but its ``NODES`` are its corners and edges alone, so a loop over them writes
    within its values."""
    data = _mesh("hexahedra")
    group = data.launch_groups("cells")[0]

    def nodes(space):
        return td.Field(space, tack.zeros((space.size,), tack.f32)).view(group).NODES

    hexahedron = shapes.Hexahedron
    assert nodes(td.H1(data, order=2, basis="serendipity")) == (hexahedron.NUM_POINTS
                                                                + hexahedron.NUM_EDGES)
    assert nodes(td.H1(data, order=2)) == hexahedron.NUM_QUADRATIC
