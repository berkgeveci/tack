"""``project``: a field into another space of its topology. Exact where the target
holds the source (a continuous field into a higher order, anything of order up
to 1 into order-1 L2), averaged as VTK averages otherwise; the shortcuts
``to_points``, ``to_cells`` and ``discontinuous`` are projections."""

import numpy as np
import pytest
from test_dataset_api import _scalars
from test_shapes import EXPECTED

import tack
import tack.dataset as td


def _linear(x):
    return 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2]


def _hexes():
    return td.rectilinear_grid([0.0, 1.0, 2.5], [0.0, 0.5, 1.5], [0.0, 1.0])


MESHES = {"hexahedra": _hexes, "tetrahedra": lambda: td.tetrahedralize(_hexes())}


def _vectors(rows):
    rows = np.asarray(rows, np.float32)
    out = tack.empty((len(rows),), tack.vec(3, tack.f32))
    tack.copyto(out, rows)
    return out


@pytest.mark.parametrize("mesh", MESHES)
def test_a_continuous_field_rises_in_order_exactly(backend, mesh):
    data = MESHES[mesh]()
    data.fields["f"] = td.Field(td.H1(data), _scalars(_linear(data.positions())))
    order2 = td.H1(data, order=2)
    nodes = td.project(data, data.geometry, order2).values.numpy()
    assert nodes.shape[0] == order2.size > data.num_points
    up = td.project(data, "f", order2)
    assert up.space is order2
    np.testing.assert_allclose(up.values.numpy(), _linear(nodes), rtol=1e-5, atol=1e-5)
    # And back down: an order-2 field's values at the points.
    down = td.project(data, up, td.H1(data))
    np.testing.assert_allclose(down.values.numpy(), data.fields["f"].values.numpy(),
                               rtol=1e-6)
    # Through the DG layout and back, nothing changes.
    dg = td.project(data, "f", td.L2(data))
    np.testing.assert_allclose(td.project(data, dg, td.H1(data)).values.numpy(),
                               data.fields["f"].values.numpy(), rtol=1e-6)


def test_cell_values_average_onto_order_2_nodes(backend):
    # Two hexes, 1 and 3: the nodes on their shared face average to 2.
    data = td.rectilinear_grid([0.0, 1.0, 2.0], [0.0, 1.0], [0.0, 1.0])
    data.fields["c"] = td.Field(td.Constant(data), _scalars([1.0, 3.0]))
    space = td.H1(data, order=2)
    nodes = td.project(data, data.geometry, space).values.numpy()
    values = td.project(data, "c", space).values.numpy()
    want = np.where(np.isclose(nodes[:, 0], 1.0), 2.0, np.where(nodes[:, 0] < 1.0, 1.0, 3.0))
    np.testing.assert_allclose(values, want, rtol=1e-6)


def test_into_cells_and_dg(backend):
    data = _hexes()
    x = data.positions()
    data.fields["f"] = td.Field(td.H1(data), _scalars(_linear(x)))
    means = td.project(data, "f", td.Constant(data)).values.numpy()
    np.testing.assert_allclose(means, td.project(data, "f", td.Constant(data)).values.numpy())
    np.testing.assert_allclose(td.project(data, "f", td.L2(data, order=0)).values.numpy(),
                               means)
    # A cell value in order-1 L2: every corner holds it.
    data.fields["c"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells) + 1.0))
    dg = td.project(data, "c", td.L2(data)).values.numpy().reshape(data.num_cells, -1)
    np.testing.assert_array_equal(dg, np.repeat(np.arange(1.0, 5.0)[:, None], 8, axis=1))
    # An order per cell: order-1 cells take the corners, order-0 ones the mean.
    orders = np.array([1, 0, 0, 1], np.int32)
    mixed = td.L2(data, order=orders)
    got = td.project(data, "f", mixed).values.numpy()
    offsets = mixed.offsets.numpy()
    corners = td.project(data, "f", td.L2(data)).values.numpy().reshape(data.num_cells, 8)
    for c in range(data.num_cells):
        cell = got[offsets[c]:offsets[c + 1]]
        np.testing.assert_allclose(cell, corners[c] if orders[c] else [means[c]], rtol=1e-6)


def test_vectors_and_values(backend):
    data = _hexes()
    rows = np.arange(3 * data.num_cells, dtype=np.float32).reshape(-1, 3)
    data.fields["v"] = td.Field(td.Values(data, "cells"), _vectors(rows))
    points = td.project(data, "v", td.Values(data, "points"))
    assert points.space is td.Values(data, "points")
    scalar = td.Field(td.Constant(data), _scalars(rows[:, 1]))
    np.testing.assert_allclose(points.values.numpy()[:, 1],
                               td.project(data, scalar, td.H1(data)).values.numpy(), rtol=1e-6)
    # The same space: a copy, not the same storage.
    same = td.project(data, "v", td.Constant(data))
    assert same.values is not data.fields["v"].values
    np.testing.assert_array_equal(same.values.numpy(), rows)


def test_into_side_traces(backend):
    data = _hexes()
    data.fields["f"] = td.Field(td.H1(data), _scalars(_linear(data.positions())))
    np.testing.assert_array_equal(
        td.project(data, "f", td.SideTraces(data)).values.numpy(),
        td.project(data, "f", td.SideTraces(data)).values.numpy())


def test_polyhedra_project_between_points_and_cells(backend):
    data = td.as_polyhedra(_hexes())
    data.fields["f"] = td.Field(td.H1(data), _scalars(_linear(data.positions())))
    data.fields["c"] = td.Field(td.Constant(data), _scalars(np.arange(4.0)))
    np.testing.assert_allclose(td.project(data, "f", td.Constant(data)).values.numpy(),
                               td.project(fresh := _hexes_with_f(), "f", td.Constant(fresh)).values.numpy(), rtol=1e-6)
    np.testing.assert_allclose(td.project(data, "c", td.H1(data)).values.numpy(),
                               td.project(data, "c", td.H1(data)).values.numpy())
    with pytest.raises(NotImplementedError, match="reference element"):
        td.project(data, "f", td.L2(data))


def _hexes_with_f():
    data = _hexes()
    data.fields["f"] = td.Field(td.H1(data), _scalars(_linear(data.positions())))
    return data


def test_refusals(backend):
    data = _hexes()
    on_faces = td.Field(td.Values(data, "faces"), _scalars(np.zeros(data.topology.faces()
                                                                     .num_faces)))
    with pytest.raises(TypeError, match="from a space with a basis"):
        td.project(data, on_faces, td.Constant(data))
    data.fields["f"] = td.Field(td.H1(data), _scalars(np.zeros(data.num_points)))
    with pytest.raises(TypeError, match="into a space with a basis"):
        td.project(data, "f", td.Values(data, "faces"))
    with pytest.raises(TypeError, match="floating-point"):
        td.project(data, td.Field(td.H1(data), tack.arange(data.num_points)),
                   td.Constant(data))
    with pytest.raises(ValueError, match="one topology"):
        td.project(data, "f", td.Constant(_hexes()))
    traces = td.project(data, "f", td.SideTraces(data))
    with pytest.raises(TypeError, match="two values on each face"):
        td.project(data, traces, td.Constant(data))


def test_changing_h1_order_keeps_unused_points(backend):
    """A point no cell uses keeps its value through H1 orders: no cell gives it one,
    and the field's own value there is exact. It was zeroed."""
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    topology = td.UnstructuredTopology([12], [0, 8], [0, 1, 3, 2, 4, 5, 7, 6], num_points=9)
    data = td.DataSet(topology, np.vstack([corners, [2.0, 2.0, 2.0]]))
    field = td.Field(td.H1(data), _scalars(np.arange(9) + 1.0))
    up = td.project(data, field, td.H1(data, order=2))
    down = td.project(data, up, td.H1(data))
    np.testing.assert_allclose(up.values.numpy()[:9], np.arange(9) + 1.0)
    np.testing.assert_allclose(down.values.numpy(), np.arange(9) + 1.0)


@pytest.mark.parametrize("shape", [td.shapes.Tetra, td.shapes.Hexahedron,
                                   td.shapes.Voxel, td.shapes.Wedge])
@pytest.mark.parametrize("vector", [False, True])
def test_exact_quadratic_projection_at_independent_nodes(backend, shape, vector):
    # The shape fixture is a reference table of points/faces/edges, independent
    # of the device methods and of the geometry's own projection.
    reference = EXPECTED[shape]
    corners = np.asarray(reference["points"], np.float32)
    positions = np.vstack([corners, [[2., 3., 4.]]]).astype(np.float32)
    top = td.UnstructuredTopology([int(shape.ID)], [0, len(corners)],
                                   np.arange(len(corners)), num_points=len(positions))
    data = td.DataSet(top, positions)
    edges = sorted(tuple(sorted(pair)) for pair in reference["edges"])
    faces = sorted(tuple(sorted(face)) for face in reference["faces"] if len(face) == 4)
    nodes = [*positions, *(corners[list(e)].mean(0) for e in edges),
             *(corners[list(f)].mean(0) for f in faces)]
    if shape in (td.shapes.Hexahedron, td.shapes.Voxel):
        nodes.append(corners.mean(0))
    nodes = np.asarray(nodes)
    evaluate = (lambda x: np.c_[_linear(x), 2 * _linear(x) + 1, -_linear(x)]
                if vector else _linear(x))
    values = _vectors(evaluate(positions)) if vector else _scalars(evaluate(positions))
    field = td.Field(td.H1(data), values)
    out = td.project(data, field, td.H1(data, order=2))
    np.testing.assert_allclose(out.values.numpy(), evaluate(nodes),
                               rtol=1e-6, atol=1e-6)
    back = td.project(data, out, td.H1(data))
    np.testing.assert_allclose(back.values.numpy(), evaluate(positions),
                               rtol=1e-6, atol=1e-6)
