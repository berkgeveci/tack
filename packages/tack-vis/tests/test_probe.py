"""CellLocator and probe, against VTK's vtkProbeFilter.

vtkProbeFilter is the reference for which points lie in the mesh and, on
hexahedra, tetrahedra and pyramids, for the values there. On skewed wedges it
is not: vtkWedge::EvaluatePosition reports a distance of ~1e-31 for points
outside the wedge (r + s > 1), and the probe's cell search takes that for a
hit in the wrong neighbour, while the same EvaluatePosition puts the point
inside the cell Tack finds. There, and everywhere, a linear field -- which any
cell holding the point reproduces exactly -- checks the values instead.
"""

import numpy as np
import pytest
from test_dataset_api import _scalars, _two_hexes_and_a_pyramid

import tack.data as td

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

SHAPES = {"hexahedron": 12, "tetra": 10, "wedge": 13, "pyramid": 14, "voxel": 11}


def _grid(kind):
    source = vtk.vtkCellTypeSource()
    source.SetCellType(SHAPES[kind])
    source.SetBlocksDimensions(3, 3, 2)
    source.Update()
    grid = source.GetOutput()
    x = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    if kind != "voxel":                       # a voxel is axis-aligned by definition
        x = x + 0.12 * np.c_[np.sin(2 * x[:, 1]), np.cos(x[:, 2]), np.sin(x[:, 0])]
        grid.GetPoints().SetData(numpy_to_vtk(x.astype(np.float32), deep=True))
    x = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    for name, values in (("s", np.sin(x[:, 0]) * x[:, 1] + x[:, 2] ** 2),
                         ("lin", 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2])):
        array = numpy_to_vtk(values.astype(np.float32), deep=True)
        array.SetName(name)
        grid.GetPointData().AddArray(array)
    return grid


def _vtk_probe(grid, points, name="s"):
    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(*p)
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(vtk_points)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(cloud)
    probe.SetSourceData(grid)
    probe.Update()
    point_data = probe.GetOutput().GetPointData()
    return (vtk_to_numpy(point_data.GetArray("vtkValidPointMask")),
            vtk_to_numpy(point_data.GetArray(name)))


@needs_vtk
@pytest.mark.parametrize("kind", SHAPES)
def test_probe_is_vtks(backend, kind):
    grid = _grid(kind)
    data = vtk_to_dataset(grid)
    points = np.random.default_rng(3).uniform(-0.4, 3.4, (500, 3))
    out = td.probe(data, points)
    valid, expected = _vtk_probe(grid, points)
    np.testing.assert_array_equal(out.fields["valid"].values.to_numpy(), valid)
    inside = valid == 1
    assert 50 < inside.sum() < 500
    lin = 2 * points[:, 0] - points[:, 1] + 0.5 * points[:, 2]
    np.testing.assert_allclose(out.fields["lin"].values.to_numpy()[inside], lin[inside],
                               atol=2e-4)
    assert (out.fields["s"].values.to_numpy()[~inside] == 0).all()
    if kind != "wedge":
        np.testing.assert_allclose(out.fields["s"].values.to_numpy()[inside],
                                   expected[inside], atol=1e-3)


@needs_vtk
def test_probe_of_a_structured_grid_is_vtks(backend):
    source = vtk.vtkRTAnalyticSource()
    source.SetWholeExtent(-6, 6, -6, 6, -6, 6)
    source.Update()
    data = td.sources.wavelet(6)
    points = np.random.default_rng(5).uniform(-7, 7, (800, 3))
    out = td.probe(data, points)
    valid, expected = _vtk_probe(source.GetOutput(), points, "RTData")
    np.testing.assert_array_equal(out.fields["valid"].values.to_numpy(), valid)
    inside = valid == 1
    np.testing.assert_allclose(out.fields["RTData"].values.to_numpy()[inside],
                               expected[inside], rtol=2e-4)


@needs_vtk
def test_locator_finds_the_cell_and_its_coordinates(backend):
    grid = _grid("hexahedron")
    data = vtk_to_dataset(grid)
    points = np.random.default_rng(4).uniform(0.2, 2.8, (100, 3)) * [1, 1, 0.6]
    cells, pcs = td.CellLocator(data).find(points)
    cells, pcs = cells.to_numpy(), pcs.to_numpy(vectors=True)
    for q in range(0, 100, 9):
        cell = grid.GetCell(int(cells[q]))
        pc, weights = [0.0] * 3, [0.0] * 8
        inside = cell.EvaluatePosition(points[q], [0.0] * 3, vtk.reference(0), pc,
                                       vtk.reference(0.0), weights)
        assert inside == 1
        np.testing.assert_allclose(pcs[q], pc, atol=1e-4)


def test_probe_keeps_the_probed_dataset_and_evaluates_every_kind(backend):
    data = _two_hexes_and_a_pyramid()
    x = data.positions()
    data.fields["h"] = td.Field(td.H1(data), _scalars(3 * x[:, 0] + x[:, 2]))
    data.fields["c"] = td.Field(td.Constant(data), _scalars([1.0, 2.0, 3.0]))
    data.fields["dg"] = td.algorithms.discontinuous(data, data.fields["h"])
    where = td.sources.tangle((4, 3, 3))
    out = td.probe(data, where)
    assert out.topology is where.topology and "tangle" in out.fields
    p = where.positions()
    valid = out.fields["valid"].values.to_numpy() == 1
    np.testing.assert_allclose(out.fields["h"].values.to_numpy()[valid],
                               (3 * p[:, 0] + p[:, 2])[valid], atol=1e-5)
    np.testing.assert_allclose(out.fields["dg"].values.to_numpy()[valid],
                               (3 * p[:, 0] + p[:, 2])[valid], atol=1e-5)
    assert set(np.unique(out.fields["c"].values.to_numpy()[valid])) <= {1.0, 2.0, 3.0}
    assert out.fields["h"].space is td.H1(out)


def test_points_outside_every_cell(backend):
    data = _two_hexes_and_a_pyramid()
    out = td.probe(data, [[10.0, 10.0, 10.0], [-1.0, 0.5, 0.5]])
    np.testing.assert_array_equal(out.fields["valid"].values.to_numpy(), [0, 0])
    with pytest.raises(NotImplementedError):
        td.CellLocator(td.as_polyhedra(data))
