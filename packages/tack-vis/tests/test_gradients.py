"""Gradients at cells and at points, of scalar and vector fields, and the flow
quantities derived from them, against VTK's vtkGradientFilter and
vtkCellDerivatives."""

import numpy as np
import pytest
from test_dataset_api import _two_hexes_and_a_pyramid

import tack
import tack.dataset as td

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

KINDS = {"hexahedron": 12, "tetra": 10, "wedge": 13, "pyramid": 14,
         # 2D cells, on the surface the skew curves out of z = 0: in-plane gradients.
         "triangle": 5, "quad": 9}


def _grid(kind):
    """A small block of cells, a little skewed, with a nonlinear scalar ``s`` and
    vector ``v``: gradients vary inside each cell."""
    grid = vtk.vtkCellTypeSource(cell_type=KINDS[kind], blocks_dimensions=(3, 3, 2))()
    x = np.asarray(grid.points.data).astype(np.float64)
    x = x + 0.15 * np.c_[np.sin(2 * x[:, 1]), np.cos(x[:, 2]), np.sin(x[:, 0])]
    grid.points = vtk.vtkPoints(data=vtk.vtkFloatArray(x.astype(np.float32)))
    grid.point_data["s"] = (np.sin(x[:, 0]) * x[:, 1] + x[:, 2] ** 2).astype(np.float32)
    grid.point_data["v"] = np.c_[np.sin(x[:, 1]), x[:, 0] * x[:, 2],
                                 x[:, 1] ** 2 - x[:, 0]].astype(np.float32)
    return grid


def _vtk_gradients(grid, name):
    g = vtk.vtkGradientFilter(result_array_name="grad", compute_divergence=True,
                              compute_vorticity=True, compute_q_criterion=True)
    g.SetInputArrayToProcess(0, 0, 0, 0, name)
    point_data = g(grid).point_data
    # A missing name reads as a NoneArray, not None, so ask the keys.
    return {key: np.asarray(point_data[array])
            for key, array in (("gradient", "grad"), ("divergence", "Divergence"),
                               ("vorticity", "Vorticity"), ("q_criterion", "Q-criterion"))
            if array in point_data.keys()}


@needs_vtk
@pytest.mark.parametrize("kind", KINDS)
def test_point_gradients_are_vtks(backend, kind):
    grid = _grid(kind)
    data = vtk_to_dataset(grid)
    scalar = td.algorithms.gradients(data, data.fields["s"], at="points")
    assert scalar.space is td.Values(data, "points")
    np.testing.assert_allclose(scalar.values.numpy(),
                               _vtk_gradients(grid, "s")["gradient"], rtol=1e-4, atol=1e-4)
    vector = td.algorithms.gradients(data, data.fields["v"], at="points")
    reference = _vtk_gradients(grid, "v")
    np.testing.assert_allclose(vector.values.numpy(), reference["gradient"],
                               rtol=1e-4, atol=1e-4)
    derived = td.algorithms.flow_quantities(data, vector)
    for key in ("divergence", "vorticity", "q_criterion"):
        np.testing.assert_allclose(
            getattr(derived, key).values.numpy(),
            reference[key], rtol=1e-4, atol=1e-4, err_msg=key)


@needs_vtk
@pytest.mark.parametrize("kind", KINDS)
def test_cell_gradients_of_vectors_are_vtks(backend, kind):
    grid = _grid(kind)
    grid.GetPointData().SetActiveVectors("v")   # no pythonic setter
    derivatives = vtk.vtkCellDerivatives()
    derivatives.SetTensorModeToComputeGradient()   # the enum has no Python name
    expected = np.asarray(derivatives(grid).cell_data.tensors)
    data = vtk_to_dataset(grid)
    out = td.algorithms.gradients(data, data.fields["v"])
    assert out.space is td.Values(data, "cells")
    np.testing.assert_allclose(out.values.numpy(), expected, rtol=1e-4,
                               atol=1e-4)


def test_a_linear_field_has_its_gradient_everywhere(backend):
    data = _two_hexes_and_a_pyramid()
    x = data.positions()
    rows = np.c_[2 * x[:, 0] - x[:, 2], 3 * x[:, 1], x[:, 0] + x[:, 1] + x[:, 2]]
    v = tack.empty((data.num_points,), tack.vec(3, tack.f32))
    tack.copyto(v, rows.astype(np.float32))
    J = np.array([[2, 0, -1], [0, 3, 0], [1, 1, 1]], float)
    for at in ("cells", "points"):
        g = td.algorithms.gradients(data, td.Field(td.H1(data), v), at=at)
        np.testing.assert_allclose(g.values.numpy(),
                                   np.tile(J.reshape(-1), (len(g.values.numpy()), 1)), atol=1e-5)
        quantities = td.algorithms.flow_quantities(data, g)
        np.testing.assert_allclose(quantities.divergence.values.numpy(), 6.0, atol=1e-5)
        np.testing.assert_allclose(quantities.vorticity.values.numpy(),
                                   [[1, -2, 0]] * len(quantities.divergence.values.numpy()),
                                   atol=1e-5)
        np.testing.assert_allclose(quantities.q_criterion.values.numpy(),
                                   -0.5 * np.trace(J @ J), atol=1e-5)
    with pytest.raises(ValueError, match="'cells' or 'points'"):
        td.algorithms.gradients(data, td.Field(td.H1(data), v), at="faces")


def _thin_hexahedra(dtype):
    """Hexahedra 2500 times wider than thick, near (1, 1, 1), and a field of about
    1e5 varying slowly across them: positions and values stored in f32, so the
    f32 and f64 datasets hold the same numbers."""
    axis = np.linspace(1.0, 1.1, 6)
    grid = td.rectilinear_grid(axis, axis, 1.0 + np.array([0.0, 2e-5, 4e-5]), dtype=tack.f64)
    p = grid.positions()
    u = 1e5 + 30.0 * p[:, 0] + 20.0 * p[:, 1] + 5e4 * (p[:, 2] - 1.0)
    data = td.DataSet(grid.topology, p.astype(np.float32).astype(dtype.numpy_dtype), dtype=dtype)
    data.fields["u"] = td.Field(td.H1(data), tack.array(
        u.astype(np.float32).astype(dtype.numpy_dtype)))
    return data


def test_a_corner_where_faces_meet_flat_is_left_out_of_the_points_average(backend):
    """A hexahedron whose corners 1 and 5 lie on straight edges of its faces (two faces
    meeting flat there: AhmedML's mesh has such cells) has a singular Jacobian there,
    and its gradient grows without bound toward them. It is left out of those points'
    averages, which come from the cells around them that have one: exact for a linear
    field. With no other cell there, the points are NaN."""
    flat = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1)]
    below = [(1, -1, 0), (2, -1, 0), (2, 0, 0), (1, 0, 0), (1, -1, 1), (2, -1, 1), (2, 0, 1),
             (1, 0, 1)]
    x = np.array(flat + below[:2] + below[4:6], np.float32)
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 1, 10, 11, 6, 5]

    def dataset(n_cells):
        topology = td.UnstructuredTopology(tack.array([12] * n_cells, dtype=tack.u8),
                                           tack.array([0, 8, 16][:n_cells + 1], dtype=tack.i32),
                                           tack.array(ids[:8 * n_cells], dtype=tack.i32),
                                           num_points=8 if n_cells == 1 else 12)
        points = x[:topology.num_points]
        data = td.DataSet(topology, tack.array(points, tack.vec3(tack.f32)))
        data.fields["u"] = td.Field(td.H1(topology), tack.array(
            (points @ np.array([1.0, 2.0, 3.0])).astype(np.float32)))
        return data

    both = td.gradients(dataset(2), "u", at="points").values.numpy().reshape(-1, 3)
    np.testing.assert_allclose(both, np.tile([1.0, 2.0, 3.0], (12, 1)), rtol=1e-5)
    alone = td.gradients(dataset(1), "u", at="points").values.numpy().reshape(-1, 3)
    assert np.isnan(alone[[1, 5]]).all()
    np.testing.assert_allclose(np.delete(alone, [1, 5], axis=0), np.tile([1.0, 2.0, 3.0], (6, 1)),
                               rtol=1e-5)


@pytest.mark.parametrize("at", ["cells", "points"])
def test_thin_cells_keep_their_gradients_in_f32(at):
    """A cell's Jacobian and its field's parametric derivative are formed from its
    corners' and values' differences from the first's, exact in f32, not from
    sums of absolute positions and values: those lost a thin cell's short edge
    and a large value's small differences to rounding, and the ONERA M6's
    boundary-layer wedges' f32 pressure gradients were up to 93 times off
    (6.6M cells off by over 0.1%; now 31, as in f64). Here f32 gives f64's
    gradients from the same stored numbers, to f32's precision (cell gradients were
    1% off; point gradients, by another path here, were not)."""
    single = td.gradients(_thin_hexahedra(tack.f32), "u", at=at).values.numpy().astype(float)
    double = td.gradients(_thin_hexahedra(tack.f64), "u", at=at).values.numpy()
    error = np.linalg.norm(single - double, axis=1) / np.linalg.norm(double, axis=1)
    assert error.max() < 1e-6
