"""Statistics and entropy, the wavelet and tangle sources, and refinement --
tetrahedralize, triangulate, shrink, point cloud -- as Viskores defines them,
against VTK where VTK has the filter."""

import numpy as np
import pytest
from test_dataset_api import _scalars

import tack.data as td

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")


def _measure(grid, what):
    integrate = vtk.vtkIntegrateAttributes()
    integrate.SetInputData(grid)
    integrate.Update()
    return integrate.GetOutput().GetCellData().GetArray(what).GetValue(0)


def _blocks(cell_type, dims=(3, 2, 2)):
    source = vtk.vtkCellTypeSource()
    source.SetCellType(cell_type)
    source.SetBlocksDimensions(*dims)
    source.Update()
    return source.GetOutput()


# ── Statistics ──────────────────────────────────────────────────────

def test_statistics_are_viskores(backend):
    data = td.sources.wavelet(5)
    values = data.fields["RTData"].values.to_numpy().astype(np.float64)
    s = td.algorithms.statistics(data.fields["RTData"])
    n, mean = values.size, values.mean()
    d = values - mean
    m2, m3, m4 = (d ** 2).sum(), (d ** 3).sum(), (d ** 4).sum()
    expected = {"n": n, "min": values.min(), "max": values.max(), "sum": values.sum(),
                "mean": mean, "m2": m2, "sample_variance": m2 / (n - 1),
                "population_variance": m2 / n, "population_stddev": np.sqrt(m2 / n),
                "skewness": np.sqrt(n) * m3 / m2 ** 1.5, "kurtosis": n * m4 / m2 ** 2}
    for key, value in expected.items():
        assert s[key] == pytest.approx(value, rel=2e-4), key
    constant = td.algorithms.statistics(td.Field(data.fields["RTData"].space,
                                                 _scalars(np.full(n, 3.0))))
    assert constant["skewness"] == constant["kurtosis"] == 0.0


def test_entropy_is_viskores(backend):
    data = td.sources.wavelet(5)
    values = data.fields["RTData"].values.to_numpy()
    for bins in (10, 37):
        counts, _ = np.histogram(values, bins=bins)
        p = counts[counts > 0] / counts.sum()
        assert td.algorithms.entropy(data.fields["RTData"], bins) == pytest.approx(
            -(p * np.log2(p)).sum(), rel=1e-6)


# ── Sources ─────────────────────────────────────────────────────────

@needs_vtk
@pytest.mark.parametrize("extent", [10, (-3, 7, 0, 5, -2, 2)])
def test_wavelet_is_vtks(backend, extent):
    source = vtk.vtkRTAnalyticSource()
    e = [-extent, extent] * 3 if isinstance(extent, int) else list(extent)
    source.SetWholeExtent(*e)
    source.Update()
    reference = source.GetOutput()
    data = td.sources.wavelet(extent)
    assert data.num_points == reference.GetNumberOfPoints()
    np.testing.assert_allclose(data.positions(), np.array(
        [reference.GetPoint(i) for i in range(reference.GetNumberOfPoints())]))
    np.testing.assert_allclose(data.fields["RTData"].values.to_numpy(),
                               vtk_to_numpy(reference.GetPointData().GetArray("RTData")),
                               rtol=1e-4, atol=1e-3)


def test_tangle_is_viskores(backend):
    data = td.sources.tangle((9, 7, 5))
    x = data.positions().astype(np.float64)
    p = 3.0 * (-1.0 + 2.0 * x)
    expected = ((p ** 4 - 5 * p ** 2).sum(axis=1) + 11.8) * 0.2 + 0.5
    np.testing.assert_allclose(data.fields["tangle"].values.to_numpy(), expected, rtol=1e-5,
                               atol=1e-5)
    assert x.min() == 0.0 and x.max() == 1.0
    assert td.contour(data, "tangle", 0.5).num_cells > 0     # the handles


# ── Refinement ──────────────────────────────────────────────────────

@needs_vtk
@pytest.mark.parametrize("cell_type, per_cell", [(12, 5), (11, 5), (13, 3), (14, 2), (10, 1)])
def test_tetrahedralize_keeps_the_volume(backend, cell_type, per_cell):
    grid = _blocks(cell_type)
    data = vtk_to_dataset(grid)
    data.fields["id"] = td.transforms.cell_ids(data)
    out = td.tetrahedralize(data)
    assert out.num_cells == per_cell * data.num_cells and out.num_points == data.num_points
    assert {int(t) for t in out.topology.types.to_numpy()} == {10}
    assert _measure(dataset_to_vtk(out), "Volume") == pytest.approx(
        _measure(grid, "Volume"), rel=1e-5)
    # Every tetrahedron the right way out (Viskores' wedge table has one inside out).
    sizes = vtk.vtkCellSizeFilter()
    sizes.SetInputData(dataset_to_vtk(out))
    sizes.Update()
    assert (vtk_to_numpy(sizes.GetOutput().GetCellData().GetArray("Volume")) > 0).all()
    np.testing.assert_array_equal(np.asarray(td.arrays.to_host(out.fields["id"].values)),
                                  np.repeat(np.arange(data.num_cells), per_cell))


@needs_vtk
@pytest.mark.parametrize("cell_type, per_cell", [(9, 2), (8, 2), (5, 1)])
def test_triangulate_keeps_the_area(backend, cell_type, per_cell):
    grid = _blocks(cell_type, (3, 4, 0))
    data = vtk_to_dataset(grid)
    out = td.triangulate(data)
    assert out.num_cells == per_cell * data.num_cells
    assert _measure(dataset_to_vtk(out), "Area") == pytest.approx(_measure(grid, "Area"),
                                                                   rel=1e-5)


@needs_vtk
def test_triangulate_polygons_as_fans(backend):
    data = td.as_polygons(_plane())
    out = td.triangulate(data)
    loops, _ = data.topology.loops()
    sizes = np.diff(loops.to_numpy())
    assert out.num_cells == (sizes - 2).sum()
    assert _measure(dataset_to_vtk(out), "Area") == pytest.approx(
        td.algorithms.cell_geometry(data)[0].values.to_numpy().sum(), rel=1e-6)


def _plane():
    from test_polyhedra import _plane_mesh

    return _plane_mesh()


@needs_vtk
@pytest.mark.parametrize("cell_type", [12, 10, 13, 14])
def test_shrink_is_vtks(backend, cell_type):
    grid = _blocks(cell_type)
    shrinker = vtk.vtkShrinkFilter()
    shrinker.SetShrinkFactor(0.7)
    shrinker.SetInputData(grid)
    shrinker.Update()
    reference = shrinker.GetOutput()
    data = vtk_to_dataset(grid)
    x = data.positions()
    data.fields["x"] = td.Field(td.H1(data), _scalars(x[:, 0]))
    data.fields["id"] = td.transforms.cell_ids(data)
    out = td.shrink(data, 0.7)
    assert isinstance(out.geometry.space, td.L2) and out.topology is data.topology
    # VTK gives each cell its own points, in the cell's corner order: the L2 layout.
    corners = out.geometry.values.to_numpy(vectors=True)
    expected = np.concatenate([[reference.GetPoint(i) for i in
                                (reference.GetCell(c).GetPointId(k) for k in range(
                                    reference.GetCell(c).GetNumberOfPoints()))]
                               for c in range(reference.GetNumberOfCells())])
    np.testing.assert_allclose(corners, expected, rtol=1e-5, atol=1e-5)
    assert isinstance(out.fields["x"].space, td.L2)
    assert out.fields["id"].space is td.Values(data, "cells")


def test_point_cloud_keeps_every_point(backend):
    data = td.sources.tangle((4, 4, 4))
    out = td.point_cloud(data)
    assert out.num_points == out.num_cells == data.num_points
    np.testing.assert_array_equal(out.fields["tangle"].values.to_numpy(),
                                  data.fields["tangle"].values.to_numpy())
