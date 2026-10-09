"""Gradients at cells and at points, of scalar and vector fields, and the flow
quantities derived from them, against VTK's vtkGradientFilter and
vtkCellDerivatives."""

import numpy as np
import pytest
from test_dataset_api import _two_hexes_and_a_pyramid

import tack
import tack.data as td

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

KINDS = {"hexahedron": 12, "tetra": 10, "wedge": 13, "pyramid": 14}


def _grid(kind):
    """A small block of cells, a little skewed, with a nonlinear scalar ``s`` and
    vector ``v``: gradients vary inside each cell."""
    source = vtk.vtkCellTypeSource()
    source.SetCellType(KINDS[kind])
    source.SetBlocksDimensions(3, 3, 2)
    source.Update()
    grid = source.GetOutput()
    x = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    x = x + 0.15 * np.c_[np.sin(2 * x[:, 1]), np.cos(x[:, 2]), np.sin(x[:, 0])]
    grid.GetPoints().SetData(numpy_to_vtk(x.astype(np.float32), deep=True))
    s = numpy_to_vtk((np.sin(x[:, 0]) * x[:, 1] + x[:, 2] ** 2).astype(np.float32), deep=True)
    s.SetName("s")
    v = numpy_to_vtk(np.c_[np.sin(x[:, 1]), x[:, 0] * x[:, 2], x[:, 1] ** 2 - x[:, 0]]
                     .astype(np.float32), deep=True)
    v.SetName("v")
    grid.GetPointData().AddArray(s)
    grid.GetPointData().AddArray(v)
    return grid


def _vtk_gradients(grid, name):
    g = vtk.vtkGradientFilter()
    g.SetInputData(grid)
    g.SetInputArrayToProcess(0, 0, 0, 0, name)
    g.SetResultArrayName("grad")
    g.ComputeDivergenceOn()
    g.ComputeVorticityOn()
    g.ComputeQCriterionOn()
    g.Update()
    point_data = g.GetOutput().GetPointData()
    return {key: vtk_to_numpy(point_data.GetArray(array))
            for key, array in (("gradient", "grad"), ("divergence", "Divergence"),
                               ("vorticity", "Vorticity"), ("q_criterion", "Q-criterion"))
            if point_data.GetArray(array) is not None}


@needs_vtk
@pytest.mark.parametrize("kind", KINDS)
def test_point_gradients_are_vtks(backend, kind):
    grid = _grid(kind)
    data = vtk_to_dataset(grid)
    scalar = td.algorithms.gradients(data, data.fields["s"], at="points")
    assert scalar.space is td.Values(data, "points")
    np.testing.assert_allclose(scalar.values.to_numpy(vectors=True),
                               _vtk_gradients(grid, "s")["gradient"], rtol=1e-4, atol=1e-4)
    vector = td.algorithms.gradients(data, data.fields["v"], at="points")
    reference = _vtk_gradients(grid, "v")
    np.testing.assert_allclose(vector.values.to_numpy(vectors=True), reference["gradient"],
                               rtol=1e-4, atol=1e-4)
    derived = td.algorithms.flow_quantities(vector)
    for key in ("divergence", "vorticity", "q_criterion"):
        np.testing.assert_allclose(
            derived[key].values.to_numpy(**({"vectors": True} if key == "vorticity" else {})),
            reference[key], rtol=1e-4, atol=1e-4, err_msg=key)


@needs_vtk
@pytest.mark.parametrize("kind", KINDS)
def test_cell_gradients_of_vectors_are_vtks(backend, kind):
    grid = _grid(kind)
    grid.GetPointData().SetActiveVectors("v")
    derivatives = vtk.vtkCellDerivatives()
    derivatives.SetInputData(grid)
    derivatives.SetTensorModeToComputeGradient()
    derivatives.Update()
    expected = vtk_to_numpy(derivatives.GetOutput().GetCellData().GetTensors())
    data = vtk_to_dataset(grid)
    out = td.algorithms.gradients(data, data.fields["v"])
    assert out.space is td.Values(data, "cells")
    np.testing.assert_allclose(out.values.to_numpy(vectors=True), expected, rtol=1e-4,
                               atol=1e-4)


def test_a_linear_field_has_its_gradient_everywhere(backend):
    data = _two_hexes_and_a_pyramid()
    x = data.positions()
    rows = np.c_[2 * x[:, 0] - x[:, 2], 3 * x[:, 1], x[:, 0] + x[:, 1] + x[:, 2]]
    v = tack.Vector.field(3, tack.f32, shape=(data.num_points,))
    v.from_numpy(rows.astype(np.float32))
    J = np.array([[2, 0, -1], [0, 3, 0], [1, 1, 1]], float)
    for at in ("cells", "points"):
        g = td.algorithms.gradients(data, td.Field(td.H1(data), v), at=at)
        np.testing.assert_allclose(g.values.to_numpy(vectors=True),
                                   np.tile(J.reshape(-1), (len(g.values.to_numpy(
                                       vectors=True)), 1)), atol=1e-5)
        quantities = td.algorithms.flow_quantities(g)
        np.testing.assert_allclose(quantities["divergence"].values.to_numpy(), 6.0, atol=1e-5)
        np.testing.assert_allclose(quantities["vorticity"].values.to_numpy(vectors=True),
                                   [[1, -2, 0]] * len(quantities["divergence"].values.to_numpy()),
                                   atol=1e-5)
        np.testing.assert_allclose(quantities["q_criterion"].values.to_numpy(),
                                   -0.5 * np.trace(J @ J), atol=1e-5)
    with pytest.raises(ValueError, match="'cells' or 'points'"):
        td.algorithms.gradients(data, td.Field(td.H1(data), v), at="faces")
