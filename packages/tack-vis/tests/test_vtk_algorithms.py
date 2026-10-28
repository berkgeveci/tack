"""``tack.interop.vtk.algorithms``: Tack's filters as VTK algorithms, in VTK pipelines
beside VTK's own, on datasets and on partitioned datasets. Each gives what the
Tack filter gives called directly, and where VTK has the same filter, VTK's
counts."""

import numpy as np
import pytest

import tack
import tack.dataset as td

vtk = pytest.importorskip("vtkmodules.all")
from tack.interop.vtk import algorithms as ta
from tack.interop.vtk import vtk_to_dataset

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _wavelet(extent=(-8, 8, -8, 8, -8, 8)):
    return vtk.vtkRTAnalyticSource(whole_extent=extent)


def _halves():
    """Two pieces of one image, cut along x with a shared plane of points, so they
    hold the whole's values (VTK's wavelet source scales by the extent it is
    given, so two sources would not)."""
    whole = _wavelet()()
    pieces = vtk.vtkPartitionedDataSet()
    pieces.number_of_partitions = 2
    for i, voi in enumerate([(-8, 0, -8, 8, -8, 8), (0, 8, -8, 8, -8, 8)]):
        pieces.SetPartition(i, vtk.vtkExtractVOI(voi=voi)(whole))
    return whole, pieces


def _vtk_contour(data, value):
    f = vtk.vtkContourFilter(compute_scalars=False)
    f.SetValue(0, value)
    f.SetInputArrayToProcess(0, 0, 0, 0, "RTData")
    return f(data)


def test_contour_in_a_pipeline_gives_vtks_cells():
    out = (_wavelet() >> ta.Contour(field="RTData", value=150.0))()
    assert isinstance(out, vtk.vtkUnstructuredGrid)
    assert out.number_of_cells == _vtk_contour(_wavelet()(), 150.0).number_of_cells


def test_a_parameter_set_later_runs_the_filter_again():
    contour = ta.Contour(field="RTData", value=150.0)
    pipeline = _wavelet() >> contour
    first = pipeline().number_of_cells
    before = contour.GetMTime()
    contour.value = 200.0
    assert contour.GetMTime() > before
    assert pipeline().number_of_cells == _vtk_contour(_wavelet()(), 200.0).number_of_cells
    assert pipeline().number_of_cells != first


def test_an_error_is_vtks_and_kept():
    contour = ta.Contour(field="no such field", value=1.0)
    contour.SetInputDataObject(_wavelet()())
    assert not contour.Update()
    assert isinstance(contour.error, (KeyError, ValueError))
    contour.field = "RTData"
    assert contour.Update() and contour.error is None


def test_an_unknown_parameter_is_refused():
    with pytest.raises(TypeError, match="no parameter 'values'"):
        ta.Contour(field="RTData", values=[1.0])


@pytest.mark.parametrize("make, run", [
    (lambda: ta.SlicePlane(origin=(0.5, 0.0, 0.0), normal=(1.0, 0.2, 0.0)),
     lambda d: td.slice_plane(d, (0.5, 0.0, 0.0), (1.0, 0.2, 0.0))),
    (lambda: ta.Clip(by="RTData", value=150.0),
     lambda d: td.clip(d, "RTData", 150.0)),
    (lambda: ta.Threshold(field="RTData", lower=120.0, upper=200.0),
     lambda d: td.threshold(d, "RTData", 120.0, 200.0)),
    (lambda: ta.ExtractSurface(), lambda d: td.extract_surface(d)),
    (lambda: ta.Tetrahedralize(), lambda d: td.tetrahedralize(d)),
], ids=["slice_plane", "clip", "threshold", "extract_surface", "tetrahedralize"])
def test_each_filter_gives_what_the_tack_filter_gives(make, run):
    image = _wavelet()()
    out = make()(image)
    direct = run(vtk_to_dataset(image, dtype=None))
    assert out.number_of_cells == direct.num_cells
    assert out.number_of_points == direct.num_points
    np.testing.assert_array_equal(np.asarray(out.points.data), direct.positions())


def test_threshold_and_surface_match_vtks_counts():
    image = _wavelet()()
    ours = ta.Threshold(field="RTData", lower=120.0, upper=200.0)(image)
    theirs = vtk.vtkThreshold(lower_threshold=120.0, upper_threshold=200.0,
                              threshold_function=vtk.vtkThreshold.THRESHOLD_BETWEEN)
    theirs.SetInputArrayToProcess(0, 0, 0, 0, "RTData")
    theirs.all_scalars = True
    assert ours.number_of_cells == theirs(image).number_of_cells
    surface = ta.ExtractSurface()(ta.Tetrahedralize()(image))
    assert surface.number_of_cells == vtk.vtkGeometryFilter()(
        ta.Tetrahedralize()(image)).number_of_cells


def test_field_filters_add_arrays_to_a_copy_of_their_input():
    image = _wavelet()()
    out = ta.Gradients(field="RTData", at="points", name="g")(image)
    assert isinstance(out, vtk.vtkImageData)
    assert "g" in out.point_data.keys() and "g" not in image.point_data.keys()
    direct = td.gradients(vtk_to_dataset(image, dtype=None), "RTData", at="points")
    np.testing.assert_array_equal(np.asarray(out.point_data["g"]), direct.values.numpy())
    # The input's own arrays are the same arrays, not copies.
    assert (np.asarray(out.point_data["RTData"]).ctypes.data
            == np.asarray(image.point_data["RTData"]).ctypes.data)
    elevation = ta.Elevation(high_point=(0.0, 0.0, 8.0))(image)
    assert "elevation" in elevation.point_data.keys()
    quality = ta.CellQuality()(ta.Tetrahedralize()(image))
    assert len(quality.cell_data.keys()) >= 1


def test_streamlines_follow_a_rotation():
    image = vtk.vtkImageData(dimensions=(21, 21, 3), origin=(-1.0, -1.0, 0.0),
                             spacing=(0.1, 0.1, 0.1))
    x = np.asarray([image.GetPoint(i) for i in range(image.number_of_points)])
    image.point_data["v"] = vtk.vtkDoubleArray(np.c_[-x[:, 1], x[:, 0], np.zeros(len(x))])
    seeds = np.array([[0.5, 0.0, 0.1], [0.3, 0.0, 0.1]])
    out = ta.Streamlines(field="v", seeds=seeds, step_size=0.05, steps=20)(image)
    direct = td.streamlines(vtk_to_dataset(image, dtype=None), "v", seeds, 0.05, 20)
    np.testing.assert_array_equal(np.asarray(out.points.data), direct.positions())
    radii = np.linalg.norm(np.asarray(out.points.data)[:, :2], axis=1)
    assert np.allclose(radii[:21], 0.5, atol=1e-3)


def test_pieces_are_filtered_one_by_one():
    whole, pieces = _halves()
    out = ta.Contour(field="RTData", value=150.0)(pieces)
    assert isinstance(out, vtk.vtkPartitionedDataSet) and out.number_of_partitions == 2
    counts = [out.GetPartition(i).number_of_cells for i in range(2)]
    assert counts == [_vtk_contour(pieces.GetPartition(i), 150.0).number_of_cells
                      for i in range(2)]
    assert sum(counts) == _vtk_contour(whole, 150.0).number_of_cells


def test_filters_over_pieces_take_them_together():
    whole, pieces = _halves()
    merged = ta.MergePieces()(pieces)
    assert isinstance(merged, vtk.vtkImageData)
    assert merged.number_of_points == whole.number_of_points
    np.testing.assert_array_equal(np.asarray(merged.point_data["RTData"]),
                                  np.asarray(whole.point_data["RTData"]))
    tets = vtk.vtkPartitionedDataSet()
    tets.number_of_partitions = 2
    for i in range(2):
        tets.SetPartition(i, ta.Tetrahedralize()(pieces.GetPartition(i)))
    ghosted = ta.GhostCells()(tets)
    for i in range(2):
        piece = ghosted.GetPartition(i)
        assert "vtkGhostType" in piece.cell_data.keys()
        assert piece.number_of_cells > tets.GetPartition(i).number_of_cells
    ids = ta.GenerateGlobalIds()(tets)
    point_ids = np.concatenate([np.asarray(ids.GetPartition(i).point_data.GetGlobalIds())
                                for i in range(2)])
    assert len(np.unique(point_ids)) == whole.number_of_points


def test_a_reader_tack_and_vtk_filters_in_one_pipeline(tmp_path):
    path = str(tmp_path / "tets.vtu")
    writer = vtk.vtkXMLUnstructuredGridWriter(file_name=path)
    writer.SetInputData(ta.Tetrahedralize()(_wavelet()()))
    writer.Write()
    reader = vtk.vtkXMLUnstructuredGridReader(file_name=path)
    sizes = (reader >> ta.Contour(field="RTData", value=150.0) >> vtk.vtkCellSizeFilter())()
    area = float(np.asarray(sizes.cell_data["Area"]).sum())
    reference = vtk.vtkCellSizeFilter()(_vtk_contour(reader(), 150.0))
    assert area == pytest.approx(float(np.asarray(reference.cell_data["Area"]).sum()), rel=1e-5)
