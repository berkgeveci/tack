"""``project``: a field into another space of its topology. Exact where the target
holds the source (a continuous field into a higher order, anything of order up
to 1 into order-1 L2), averaged otherwise: onto points as VTK averages, onto
cells by each cell's average (``average="nodes"``: VTK's mean of its nodes)."""

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


def _quads():
    return td.rectilinear_grid([0.0, 1.0, 2.5], [0.0, 0.5, 1.5], [0.25])


def _triangles():
    """``_quads``' cells, each cut into two triangles."""
    quads = _quads()
    rows = []
    for j in range(2):
        for i in range(2):
            a = i + 3 * j
            rows += [[a, a + 1, a + 4], [a, a + 4, a + 3]]
    topology = td.UnstructuredTopology(np.full(len(rows), 5, np.uint8),
                                       3 * np.arange(len(rows) + 1), np.concatenate(rows))
    return td.DataSet(topology, quads.positions())


MESHES = {"hexahedra": _hexes, "tetrahedra": lambda: td.tetrahedralize(_hexes()),
          "quads": _quads, "triangles": _triangles}


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
                                   td.shapes.Voxel, td.shapes.Wedge, td.shapes.Triangle,
                                   td.shapes.Quad, td.shapes.Pixel])
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
    if shape in (td.shapes.Hexahedron, td.shapes.Voxel, td.shapes.Quad, td.shapes.Pixel):
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


# ── Interpolating a function of position ────────────────────────────

@tack.func
def _quadratic_of(x):
    return x[0] * x[0] + x[1] * x[2] - 0.5 * x[2]


@tack.func
def _vector_of(x):
    return tack.vec3(x[0], 2.0 * x[1], x[0] * x[2])


def _quadratic(x):
    return x[:, 0] ** 2 + x[:, 1] * x[:, 2] - 0.5 * x[:, 2]


def _nodes(data, space):
    """Where ``space``'s values are, from the geometry projected there (stored, as
    interpolate does not)."""
    if isinstance(space, td.Values):
        return data.positions() if space.on == "points" else td.cell_centers(data).values.numpy()
    if isinstance(space, td.Constant) or (isinstance(space, td.L2) and space.order == 0):
        # The parametric center, which a projection (a mean of corners) is not
        # on a curved cell.
        return td.cell_centers(data).values.numpy()
    return td.project(data, data.geometry, space).values.numpy()


def _curved():
    data = _hexes()
    space = td.H1(data, order=2)
    nodes = td.project(data, data.geometry, space).values.numpy()
    bent = nodes + 0.1 * np.c_[nodes[:, 1] ** 2, nodes[:, 2] ** 2, nodes[:, 0] ** 2]
    data.geometry = td.Field(space, _vectors(bent))
    return data


@pytest.mark.parametrize("mesh", [*MESHES, "curved"])
@pytest.mark.parametrize("space", ["H1", "H1 order 2", "L2", "L2 order 0", "Constant",
                                   "points", "cells"])
def test_interpolate_evaluates_at_every_node(backend, mesh, space):
    """Each node's value is the function at the node's position, as the geometry
    puts it: on a curved geometry too, where nodes are not where straight cells
    would put them."""
    data = _curved() if mesh == "curved" else MESHES[mesh]()
    space = {"H1": td.H1(data), "H1 order 2": td.H1(data, order=2), "L2": td.L2(data),
             "L2 order 0": td.L2(data, order=0), "Constant": td.Constant(data),
             "points": td.Values(data, "points"), "cells": td.Values(data, "cells")}[space]
    u = td.interpolate(data, _quadratic_of, space)
    assert u.space is space and u.values.dtype == data.dtype
    np.testing.assert_allclose(u.values.numpy(), _quadratic(_nodes(data, space)), atol=1e-5)


def test_interpolate_gives_vectors_and_quadratics_exactly(backend):
    data = _hexes()
    space = td.H1(data, order=2)
    v = td.interpolate(data, _vector_of, space, dtype=tack.vec3())
    x = _nodes(data, space)
    np.testing.assert_allclose(v.values.numpy(), np.c_[x[:, 0], 2 * x[:, 1], x[:, 0] * x[:, 2]],
                               atol=1e-6)
    # The order-2 interpolant of a quadratic is the quadratic, inside cells too.
    u = td.interpolate(data, _quadratic_of, space)
    np.testing.assert_allclose(td.values_at_centers(data, u).values.numpy(),
                               _quadratic(td.cell_centers(data).values.numpy()), atol=1e-5)


def test_interpolate_writes_each_shared_node_once(backend):
    """Over an H1 geometry every node is visited once, from the space's node views,
    and follows a changed geometry; a discontinuous (L2) geometry, which gives a
    shared node no position of its own, has the first cell that has it write
    it, through its corners."""
    data = _hexes()
    space = td.H1(data, order=2)
    u = td.interpolate(data, _quadratic_of, space)
    np.testing.assert_allclose(u.values.numpy(), _quadratic(_nodes(data, space)), atol=1e-5)
    # Each cell moved its own way: point 1 (x = 1, on cells 0 and 1) is cell 0's.
    dg = td.project(data, data.geometry, td.L2(data))
    corners = dg.values.numpy().reshape(data.num_cells, 8, 3)
    corners += np.arange(data.num_cells)[:, None, None] * np.array([0.0, 0.0, 10.0])
    moved = td.DataSet(data.topology, td.Field(td.L2(data), _vectors(corners.reshape(-1, 3))))
    u = td.interpolate(moved, _quadratic_of, td.H1(moved))
    np.testing.assert_allclose(u.values.numpy()[1], _quadratic(data.positions()[[1]])[0],
                               atol=1e-5)
    _, writers = td.H1(moved)._writers
    assert int(writers.numpy().sum()) == td.H1(moved).size        # every node, once
    # A new geometry on the same topology: new positions.
    data.geometry = td.warp(data, (0.0, 0.0, 1.0), scale=2.0).geometry
    u = td.interpolate(data, _quadratic_of, space)
    np.testing.assert_allclose(u.values.numpy(), _quadratic(_nodes(data, space)), atol=1e-5)


def test_interpolate_unused_points_and_refusals(backend):
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    topology = td.UnstructuredTopology([12], [0, 8], [0, 1, 3, 2, 4, 5, 7, 6], num_points=9)
    data = td.DataSet(topology, np.vstack([corners, [2.0, 2.0, 2.0]]))
    for space in (td.H1(data), td.H1(data, order=2)):
        u = td.interpolate(data, _quadratic_of, space)
        assert u.values.numpy()[8] == pytest.approx(_quadratic(np.array([[2.0, 2.0, 2.0]]))[0])
    poly = td.as_polyhedra(_hexes())
    np.testing.assert_allclose(td.interpolate(poly, _quadratic_of, td.H1(poly)).values.numpy(),
                               _quadratic(poly.positions()), atol=1e-5)
    with pytest.raises(NotImplementedError, match="reference element"):
        td.interpolate(poly, _quadratic_of, td.Constant(poly))
    with pytest.raises(TypeError, match="points or cells, not on faces"):
        td.interpolate(data, _quadratic_of, td.Values(data, "faces"))
    with pytest.raises(TypeError, match="H1, L2, Constant or Values"):
        td.interpolate(data, _quadratic_of, td.SideTraces(data))
    with pytest.raises(ValueError, match="own topology"):
        td.interpolate(data, _quadratic_of, td.H1(_hexes()))
    with pytest.raises(TypeError, match="@tack.func"):
        td.interpolate(data, lambda x: x, td.H1(data))


# ── A cell's one value: its average, or its nodes' mean ─────────────

@tack.func
def _square_of_x(p):
    return p.x * p.x


@pytest.mark.parametrize("space", ["H1 2", "H1 3", "L2 2", "bernstein 3"])
def test_into_cells_is_each_cells_average(backend, space):
    """x² on the unit cube, in a space that holds it exactly: its average is 1/3,
    which its corners' mean (1/2) and its nodes' mean are not (but for Bernstein
    coefficients: each basis function has the same integral)."""
    data = td.uniform_grid((2, 2, 2))
    kind, order = space.split()
    space = (td.L2(data, order=int(order)) if kind == "L2"
             else td.H1(data, order=int(order), **({"basis": kind} if kind == "bernstein" else {})))
    f = td.interpolate(data, _square_of_x, space)
    for target in (td.Constant(data), td.L2(data, order=0), td.Values(data, "cells")):
        assert td.project(data, f, target).values.numpy() == pytest.approx([1 / 3], rel=1e-6)
    nodes = td.project(data, f, td.Constant(data), average="nodes").values.numpy()[0]
    assert nodes == pytest.approx(1 / 3, rel=1e-6) if kind == "bernstein" else nodes > 0.35


def _one_cell(vtk_type, corners):
    corners = np.asarray(corners, np.float64)
    topology = td.UnstructuredTopology(np.array([vtk_type], np.uint8), np.array([0, len(corners)]),
                                       np.arange(len(corners)))
    return td.DataSet(topology, corners)


DISTORTED = {
    "hexahedron": (12, [[0, 0, 0], [2, 0, 0], [2.5, 1.5, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1.5], [1, 1, 1], [0, 1, 1.2]]),
    "wedge": (13, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1.5, 0, 1.2], [0, 1.3, 0.8]]),
    "pyramid": (14, [[0, 0, 0], [1, 0, 0], [1.2, 1.1, 0], [0, 1, 0], [0.3, 0.4, 1]]),
    "quad": (9, [[0, 0, 0], [2, 0, 0], [2.5, 1.5, 0], [0, 1, 0]]),
    "tetra": (10, [[0, 0, 0], [2, 0, 0], [0.5, 1.5, 0], [0.2, 0.3, 1.1]]),
}


@pytest.mark.parametrize("shape", DISTORTED)
def test_a_distorted_cells_average_is_its_integral_over_its_measure(backend, shape):
    """Through the cell's own map: a pyramid's apex weighs a quarter, not a fifth,
    and a distorted hexahedron's corners do not weigh alike."""
    data = _one_cell(*DISTORTED[shape])
    x = data.positions()
    data.fields["f"] = td.Field(td.H1(data), _scalars(_linear(x)))
    data.fields["z"] = td.Field(td.H1(data), _scalars(x[:, 2]))
    whole = td.integrate(data)
    for name in "fz":
        average = td.project(data, name, td.Constant(data)).values.numpy()[0]
        assert average == pytest.approx(whole.values[name] / whole.measure, rel=1e-5)
        nodes = td.project(data, name, td.Constant(data), average="nodes").values.numpy()[0]
        assert nodes == pytest.approx(data.fields[name].values.numpy().mean(), rel=1e-6)
    if shape == "pyramid":                       # z: the apex's height times a quarter
        assert td.project(data, "z", td.Constant(data)).values.numpy()[0] == pytest.approx(0.25)


def test_polyhedra_average_over_their_pieces(backend):
    """A polyhedron's average is exact for a linear field: a pyramid's apex
    weighs a quarter; ``average="nodes"`` is its distinct points' mean."""
    data = td.as_polyhedra(_one_cell(*DISTORTED["pyramid"]))
    data.fields["z"] = td.Field(td.H1(data), _scalars(data.positions()[:, 2]))
    assert td.project(data, "z", td.Constant(data)).values.numpy()[0] == pytest.approx(0.25,
                                                                                       rel=1e-5)
    assert td.project(data, "z", td.Constant(data),
                      average="nodes").values.numpy()[0] == pytest.approx(0.2)


# VTK's higher-order cells: quadratic triangle, quad (8), tetrahedron, hexahedron
# (20), wedge (15), triquadratic hexahedron (27), Lagrange hexahedron of order 3.
_VTK_CELLS = [(22, 2), (23, 2), (24, 2), (25, 2), (26, 2), (29, 2), (72, 3)]


@pytest.mark.parametrize("vtk_type, order", _VTK_CELLS)
def test_vtks_higher_order_cells_average_as_vtk_with_nodes(f64_backend, vtk_type, order):
    """``average="nodes"`` is vtkPointDataToCellData on every point a cell has; the
    default is the average, whose integral over the mesh is the field's."""
    sources = pytest.importorskip("vtkmodules.vtkFiltersSources")
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
    from vtkmodules.vtkFiltersCore import vtkPointDataToCellData

    from tack.interop.vtk import vtk_to_dataset

    solid = vtk_type not in (22, 23)
    grid = sources.vtkCellTypeSource(cell_type=vtk_type, cell_order=order,
                                     blocks_dimensions=(2, 2, 2) if solid else (2, 2, 1))()
    p = np.asarray(grid.points.data, np.float64)
    p = p + 0.04 * np.c_[np.sin(p[:, 1]), np.sin(p[:, 2]), np.sin(p[:, 0])]   # bent
    grid.points = vtkPoints(data=p)
    grid.point_data["u"] = vtkDoubleArray(p[:, 0] ** 2 - p[:, 1] * p[:, 2] + 0.3 * p[:, 2])
    want = np.asarray(vtkPointDataToCellData()(grid).cell_data["u"])
    data = vtk_to_dataset(grid, dtype=tack.f64)
    np.testing.assert_allclose(
        td.project(data, "u", td.Constant(data), average="nodes").values.numpy(), want,
        rtol=1e-12)
    averages = td.project(data, "u", td.Constant(data)).values.numpy()
    measures = td.cell_geometry(data).measure.values.numpy()
    whole = td.integrate(data)
    assert np.dot(averages, measures) == pytest.approx(whole.values["u"], rel=1e-10)
    assert not np.allclose(averages, want, rtol=1e-6)


def test_average_is_for_one_value_a_cell(backend):
    data = _hexes_with_f()
    with pytest.raises(ValueError, match="average is one of"):
        td.project(data, "f", td.Constant(data), average="corners")
    with pytest.raises(ValueError, match="one value"):
        td.project(data, "f", td.H1(data, order=2), average="nodes")
