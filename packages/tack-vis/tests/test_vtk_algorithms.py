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
    for voi in [(-8, 0, -8, 8, -8, 8), (0, 8, -8, 8, -8, 8)]:
        pieces.append(vtk.vtkExtractVOI(voi=voi)(whole))
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
    image = _wavelet()()
    contour = ta.Contour(field="no such field", value=150.0)
    assert contour(image).number_of_cells == 0
    assert isinstance(contour.error, (KeyError, ValueError))
    contour.field = "RTData"
    assert contour(image).number_of_cells > 0 and contour.error is None


def test_a_failed_run_keeps_its_error_without_its_frames():
    """The error is kept without its traceback: that held the run's frames, the
    algorithm and VTK's information objects in a cycle, which crashed Python's
    collector freeing them."""
    import gc

    source = _wavelet()
    contour = ta.Contour(field="no such field")
    contour.SetInputConnection(source.GetOutputPort())
    contour.Update()
    assert contour.error is not None and contour.error.__traceback__ is None
    del contour, source
    gc.collect()


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
    np.testing.assert_array_equal(out.points.data, direct.positions())


def test_threshold_and_surface_match_vtks_counts():
    image = _wavelet()()
    ours = ta.Threshold(field="RTData", lower=120.0, upper=200.0)(image)
    theirs = vtk.vtkThreshold(lower_threshold=120.0, upper_threshold=200.0,
                              threshold_function=vtk.vtkThreshold.THRESHOLD_BETWEEN,
                              all_scalars=True)
    theirs.SetInputArrayToProcess(0, 0, 0, 0, "RTData")
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
    np.testing.assert_array_equal(out.point_data["g"], direct.values.numpy())
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
    image.point_data["v"] = np.c_[-x[:, 1], x[:, 0], np.zeros(len(x))]
    seeds = np.array([[0.5, 0.0, 0.1], [0.3, 0.0, 0.1]])
    out = ta.Streamlines(field="v", seeds=seeds, step_size=0.05, steps=20)(image)
    direct = td.streamlines(vtk_to_dataset(image, dtype=None), "v", seeds, 0.05, 20)
    np.testing.assert_array_equal(out.points.data, direct.positions())
    radii = np.linalg.norm(out.points.data[:, :2], axis=1)
    assert np.allclose(radii[:21], 0.5, atol=1e-3)


def test_pieces_are_filtered_one_by_one():
    whole, pieces = _halves()
    out = ta.Contour(field="RTData", value=150.0)(pieces)
    assert isinstance(out, vtk.vtkPartitionedDataSet) and out.number_of_partitions == 2
    counts = [piece.number_of_cells for piece in out]
    assert counts == [_vtk_contour(piece, 150.0).number_of_cells for piece in pieces]
    assert sum(counts) == _vtk_contour(whole, 150.0).number_of_cells


def test_filters_over_pieces_take_them_together():
    whole, pieces = _halves()
    merged = ta.MergePieces()(pieces)
    assert isinstance(merged, vtk.vtkImageData)
    assert merged.number_of_points == whole.number_of_points
    np.testing.assert_array_equal(merged.point_data["RTData"], whole.point_data["RTData"])
    tets = ta.Tetrahedralize()(pieces)
    ghosted = ta.GhostCells()(tets)
    for piece, before in zip(ghosted, tets):
        assert "vtkGhostType" in piece.cell_data
        assert piece.number_of_cells > before.number_of_cells
    ids = ta.GenerateGlobalIds()(tets)
    # Global ids have no pythonic accessor in VTK 9.7.1.
    point_ids = np.concatenate([piece.point_data.GetGlobalIds() for piece in ids])
    assert len(np.unique(point_ids)) == whole.number_of_points


def test_a_reader_tack_and_vtk_filters_in_one_pipeline(tmp_path):
    path = tmp_path / "tets.vtu"
    vtk.vtkXMLUnstructuredGridWriter(file_name=str(path))(ta.Tetrahedralize()(_wavelet()()))
    assert path.exists()
    reader = vtk.vtkXMLUnstructuredGridReader(file_name=str(path))
    sizes = (reader >> ta.Contour(field="RTData", value=150.0) >> vtk.vtkCellSizeFilter())()
    area = float(sizes.cell_data["Area"].sum())
    reference = vtk.vtkCellSizeFilter()(_vtk_contour(reader(), 150.0))
    assert area == pytest.approx(float(reference.cell_data["Area"].sum()), rel=1e-5)


# ── One Tack dataset per VTK dataset ────────────────────────────────

def _rotation():
    grid = vtk.vtkImageData(dimensions=(12, 12, 3), spacing=(0.2, 0.2, 0.2),
                            origin=(-1.1, -1.1, 0.0))
    x = np.asarray(vtk.vtkImageDataToPointSet()(grid).points.data)
    v = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x))], axis=1)
    grid.point_data["v"] = v
    return grid


def _streamlines(grid, steps):
    f = ta.Streamlines(field="v", seeds=np.array([[0.5, 0.0, 0.2]]), step_size=0.01,
                       steps=steps)
    f.SetInputDataObject(grid)
    f.Update()
    return f.GetOutputDataObject(0)


def test_filters_reading_one_dataset_share_its_tack_dataset():
    """What one derives (here the locator) the next one reuses."""
    grid = _rotation()
    _streamlines(grid, 50)
    data = ta._as_dataset(grid)
    locator = data.topology._locator
    assert locator is not None
    _streamlines(grid, 80)
    assert ta._as_dataset(grid) is data
    assert data.topology._locator is locator


def test_a_filter_run_again_reuses_its_inputs_tack_dataset():
    contour = ta.Contour(field="RTData", value=150.0)
    source = _wavelet()
    contour.SetInputConnection(source.GetOutputPort())
    contour.Update()
    data = ta._as_dataset(contour.GetInputDataObject(0, 0))
    contour.value = 200.0
    contour.Update()
    assert ta._as_dataset(contour.GetInputDataObject(0, 0)) is data


def test_a_changed_dataset_is_converted_again():
    grid = _rotation()
    data = ta._as_dataset(grid)
    grid.point_data["w"] = np.ones(grid.number_of_points)
    changed = ta._as_dataset(grid)
    assert changed is not data and "w" in changed.fields


def test_an_entry_goes_with_its_vtk_dataset():
    import gc

    grid = _rotation()
    ta._as_dataset(grid)
    address = grid.GetAddressAsString("vtkObject")
    assert address in ta._datasets
    del grid
    gc.collect()
    assert address not in ta._datasets


def test_a_field_filters_output_keeps_its_inputs_topology():
    """Its output is its input with arrays added: the next filter keeps the topology
    and what was derived from it, and sees what a fresh conversion gives."""
    grid = _rotation()
    elevation = ta.Elevation(low_point=(0, 0, 0), high_point=(1, 0, 0))
    elevation.SetInputDataObject(grid)
    elevation.Update()
    out = elevation.GetOutputDataObject(0)
    kept = ta._as_dataset(out)
    assert kept.topology is ta._as_dataset(grid).topology
    fresh = vtk_to_dataset(out, dtype=None)
    assert set(kept.fields) == set(fresh.fields)
    for name in fresh.fields:
        np.testing.assert_array_equal(kept.fields[name].values.numpy(),
                                      fresh.fields[name].values.numpy())


def test_probe_gives_vtks_values():
    source = _wavelet()()
    points = vtk.vtkPointSource(number_of_points=60, radius=6.0)()
    probe = ta.Probe()
    probe.SetInputDataObject(points)
    probe.SetSourceData(source)
    probe.Update()
    out = probe.GetOutputDataObject(0)
    assert probe.error is None and isinstance(out, vtk.vtkUnstructuredGrid)
    reference = vtk.vtkProbeFilter()
    reference.SetInputData(points)
    reference.SetSourceData(source)
    reference.Update()
    np.testing.assert_allclose(out.point_data["RTData"],
                               reference.GetOutput().point_data["RTData"], rtol=1e-5)
    np.testing.assert_array_equal(out.point_data["valid"], 1)


def test_probe_in_a_pipeline_keeps_an_images_structure():
    source = _wavelet()
    image = vtk.vtkImageData(dimensions=(5, 5, 5), origin=(-4.0, -4.0, -4.0),
                             spacing=(2.0, 2.0, 2.0))
    probe = ta.Probe(fields=["RTData"])
    probe.SetInputDataObject(image)
    probe.SetSourceConnection(source.GetOutputPort())
    probe.Update()
    out = probe.GetOutputDataObject(0)
    assert isinstance(out, vtk.vtkImageData) and out.dimensions == (5, 5, 5)
    assert set(out.point_data.keys()) == {"RTData", "valid"}


# ── Trees: multiblocks and collections ──────────────────────────────

def _collection():
    """A vtkPartitionedDataSetCollection of the wavelet's two halves and the whole, named,
    with a data assembly naming them."""
    whole, halves = _halves()
    tree = vtk.vtkPartitionedDataSetCollection()
    tree.SetNumberOfPartitionedDataSets(2)
    tree.SetPartitionedDataSet(0, halves)
    one = vtk.vtkPartitionedDataSet()
    one.append(whole)
    tree.SetPartitionedDataSet(1, one)
    for i, name in enumerate(["halves", "whole"]):
        tree.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), name)
    assembly = vtk.vtkDataAssembly()
    assembly.SetRootNodeName("wavelet")
    for i, name in enumerate(["halves", "whole"]):
        assembly.AddDataSetIndex(assembly.AddNode(name), i)
    tree.SetDataAssembly(assembly)
    return tree


def _multiblock():
    """A multiblock of the whole wavelet, its halves as a vtkMultiPieceDataSet, and a
    nested multiblock holding the whole again, named."""
    whole, halves = _halves()
    pieces = vtk.vtkMultiPieceDataSet()
    pieces.SetNumberOfPieces(2)
    for i in range(2):
        pieces.SetPiece(i, halves.GetPartition(i))
    inner = vtk.vtkMultiBlockDataSet()
    inner.SetNumberOfBlocks(1)
    inner.SetBlock(0, whole)
    tree = vtk.vtkMultiBlockDataSet()
    tree.SetNumberOfBlocks(3)
    for i, block in enumerate([whole, pieces, inner]):
        tree.SetBlock(i, block)
        tree.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), f"block {i}")
    return tree


def test_a_collection_keeps_its_structure_names_and_assembly():
    out = ta.Contour(field="RTData", value=150.0)(_collection())
    assert isinstance(out, vtk.vtkPartitionedDataSetCollection)
    assert out.GetNumberOfPartitionedDataSets() == 2
    assert [out.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) for i in range(2)] == [
        "halves", "whole"]
    assert out.GetDataAssembly().GetRootNodeName() == "wavelet"
    whole = _vtk_contour(_wavelet()(), 150.0).number_of_cells
    assert out.GetPartitionedDataSet(1).GetPartition(0).number_of_cells == whole
    halves = out.GetPartitionedDataSet(0)
    assert sum(halves.GetPartition(i).number_of_cells for i in range(2)) == whole


def test_a_multiblock_keeps_its_structure_and_names():
    out = ta.Contour(field="RTData", value=150.0)(_multiblock())
    assert isinstance(out, vtk.vtkMultiBlockDataSet)
    assert [out.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) for i in range(3)] == [
        "block 0", "block 1", "block 2"]
    whole = _vtk_contour(_wavelet()(), 150.0).number_of_cells
    assert out.GetBlock(0).number_of_cells == whole
    pieces = out.GetBlock(1)
    assert isinstance(pieces, vtk.vtkMultiPieceDataSet)
    assert sum(pieces.GetPiece(i).number_of_cells for i in range(2)) == whole
    assert out.GetBlock(2).GetBlock(0).number_of_cells == whole


def _point_ids(pieces):
    return [np.asarray(pieces.GetPartition(i).GetPointData().GetGlobalIds())
            for i in range(pieces.GetNumberOfPartitions())]


@pytest.mark.parametrize("tree, pieces", [
    (_collection, lambda out: out.GetPartitionedDataSet(0)),
    (_multiblock, lambda out: out.GetBlock(1)),
])
def test_a_filter_over_pieces_takes_each_partitioned_block_whole(tree, pieces):
    """The halves are numbered together: the plane they share has one id per point, so
    the ids run to the whole's point count, once each (by VTK's executive alone, a
    multiblock's pieces would each be numbered from 0)."""
    out = ta.GenerateGlobalIds()(tree())
    ids = _point_ids(pieces(out))
    assert len(ids) == 2
    assert np.array_equal(np.unique(np.concatenate(ids)),
                          np.arange(_wavelet()().number_of_points))


def test_merging_in_a_collection_gives_one_partition_per_dataset():
    out = ta.MergePieces()(_collection())
    halves = out.GetPartitionedDataSet(0)
    assert halves.GetNumberOfPartitions() == 1
    assert halves.GetPartition(0).number_of_points == _wavelet()().number_of_points


def test_a_field_filter_on_a_tree_adds_its_arrays_to_every_block():
    out = ta.Gradients(field="RTData", at="points")(_collection())
    for i in range(2):
        node = out.GetPartitionedDataSet(i)
        for j in range(node.GetNumberOfPartitions()):
            assert "gradient" in node.GetPartition(j).point_data.keys()


def test_probe_into_a_tree_from_a_tree():
    source = _collection()
    where = _multiblock()
    probe = ta.Probe()
    probe.SetInputDataObject(0, where)
    probe.SetSourceData(source)
    probe.Update()
    out = probe.GetOutputDataObject(0)
    assert isinstance(out, vtk.vtkMultiBlockDataSet)
    values = np.asarray(out.GetBlock(0).point_data["RTData"])
    np.testing.assert_allclose(values, np.asarray(_wavelet()().point_data["RTData"]),
                               rtol=1e-5)


# ── Parameters from the Tack function ───────────────────────────────

def test_every_argument_of_the_tack_function_is_a_parameter():
    """Contour's are td.contour's, but for the data: the class's own ``field`` and
    ``value`` and generated ones such as ``merge_points``, with td.contour's defaults."""
    import inspect

    names = set(inspect.signature(td.contour).parameters) - {"data"}
    contour = ta.Contour()
    for name in names:
        assert hasattr(contour, name), name
    assert contour.merge_points is True and contour.dimension is None
    # A generated parameter is set and passed like any other.
    plain = (_wavelet() >> ta.Contour(field="RTData", value=150.0))()
    unmerged = (_wavelet() >> ta.Contour(field="RTData", value=150.0, merge_points=False))()
    assert unmerged.number_of_points > plain.number_of_points
    assert unmerged.number_of_cells == plain.number_of_cells


def test_what_the_wrapper_supplies_is_no_parameter_of_the_functions():
    """A locator and Probe's points are the wrapper's; ``world`` is a parameter of
    every algorithm (by default VTK's), not of the function's."""
    assert not hasattr(ta.Streamlines(), "locator")
    assert not hasattr(ta.Probe(), "where")
    assert ta.GhostCells().world is None and ta.Contour().world is None


def test_a_required_argument_unset_is_named():
    source = _wavelet()             # held: a temporary's port outlives it (VTK bugs)
    threshold = ta.Threshold(lower=0.0, upper=1.0)
    threshold.SetInputConnection(source.GetOutputPort())
    threshold.Update()
    assert "needs 'field'" in str(threshold.error)


# ── Fields: by association, and VTK's SetInputArrayToProcess ─────────

def _two_of_one_name():
    """The wavelet with ``p``, a point array (``RTData``) and a cell array (each cell's
    first point's value, plus a million) of one name."""
    grid = _wavelet()()
    for attributes, values in [
            (grid.point_data, np.asarray(grid.point_data["RTData"])),
            (grid.cell_data, np.full(grid.number_of_cells, 1e6, np.float32))]:
        # Copied into VTK's own memory: vtkFloatArray(values) would share the NumPy
        # array's without holding it.
        array = vtk.vtkFloatArray()
        array.SetNumberOfValues(values.size)
        np.asarray(array)[:] = values
        array.SetName("p")
        attributes.AddArray(array)
    return grid


@pytest.mark.parametrize("field, kept", [(("p", "points"), True), (("p", "cells"), False)])
def test_a_field_named_with_where_vtk_keeps_it(field, kept):
    out = ta.Threshold(field=field, lower=0.0, upper=200.0)(_two_of_one_name())
    assert (out.number_of_cells > 0) is kept


def test_set_input_array_to_process_chooses_the_field():
    threshold = ta.Threshold(lower=0.0, upper=200.0)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "p")
    assert threshold(_two_of_one_name()).number_of_cells == 0
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "p")
    assert threshold(_two_of_one_name()).number_of_cells > 0
    # A parameter set wins.
    threshold.field = ("p", "cells")
    assert threshold(_two_of_one_name()).number_of_cells == 0


def test_a_name_both_kinds_have_is_refused_unless_told_which():
    threshold = ta.Threshold(field="p", lower=0.0, upper=200.0)
    threshold.SetInputDataObject(0, _two_of_one_name())
    threshold.Update()
    assert threshold.error is not None


# ── VTK's implicit functions ────────────────────────────────────────

def _boolean(operation):
    b = vtk.vtkImplicitBoolean()
    b.SetOperationType(operation)
    b.AddFunction(vtk.vtkSphere(center=(0, 0, 0), radius=6.0))
    b.AddFunction(vtk.vtkPlane(origin=(1, 0, 0), normal=(1, 0, 0)))
    return b


@pytest.mark.parametrize("vtk_function, tack_function", [
    (lambda: vtk.vtkPlane(origin=(1, 0, 0), normal=(1, 2, 0)),
     lambda: td.Plane((1, 0, 0), (1, 2, 0))),
    (lambda: vtk.vtkSphere(center=(1, 0, 0), radius=5.0), lambda: td.Sphere((1, 0, 0), 5.0)),
    (lambda: vtk.vtkCylinder(center=(0, 1, 0), axis=(0, 0, 1), radius=4.0),
     lambda: td.Cylinder((0, 1, 0), (0, 0, 1), 4.0)),
    (lambda: _box(), lambda: td.Box((-3, -2, -1), (4, 5, 6))),
    (lambda: _planes(), lambda: td.Planes([(0, 0, 3), (0, 0, -3)], [(0, 0, 1), (0, 0, -1)])),
    (lambda: _boolean(0), lambda: td.Union(td.Sphere((0, 0, 0), 6.0),
                                           td.Plane((1, 0, 0), (1, 0, 0)))),
    (lambda: _boolean(1), lambda: td.Intersection(td.Sphere((0, 0, 0), 6.0),
                                                  td.Plane((1, 0, 0), (1, 0, 0)))),
    (lambda: _boolean(2), lambda: td.Difference(td.Sphere((0, 0, 0), 6.0),
                                                td.Plane((1, 0, 0), (1, 0, 0)))),
])
def test_vtks_implicit_functions_clip_as_tacks(vtk_function, tack_function):
    by_vtk = ta.Clip(by=vtk_function(), inside=True)(_wavelet()())
    by_tack = ta.Clip(by=tack_function(), inside=True)(_wavelet()())
    assert by_vtk.number_of_cells == by_tack.number_of_cells > 0
    np.testing.assert_array_equal(np.asarray(by_vtk.points.data),
                                  np.asarray(by_tack.points.data))


def _box():
    box = vtk.vtkBox()
    box.SetBounds(-3, 4, -2, 5, -1, 6)
    return box


def _planes():
    planes = vtk.vtkPlanes()
    planes.SetPoints(vtk.vtkPoints(data=np.array([[0, 0, 3], [0, 0, -3]], np.float64)))
    planes.SetNormals(vtk.vtkDoubleArray(np.array([[0, 0, 1], [0, 0, -1]], np.float64)))
    return planes


def test_an_implicit_function_tack_has_none_of_is_named():
    source = _wavelet()
    clip = ta.Clip(by=vtk.vtkSuperquadric())
    clip.SetInputConnection(source.GetOutputPort())
    clip.Update()
    assert "vtkSuperquadric" in str(clip.error)


# ── Time ────────────────────────────────────────────────────────────

def _time_source(tree):
    from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

    sddp = vtk.vtkStreamingDemandDrivenPipeline

    class Source(VTKPythonAlgorithmBase):
        """The wavelet at times 0, 1 and 2, as an image or a one-block multiblock."""

        def __init__(self):
            VTKPythonAlgorithmBase.__init__(
                self, nInputPorts=0, nOutputPorts=1,
                outputType="vtkMultiBlockDataSet" if tree else "vtkImageData")

        def RequestInformation(self, request, inInfo, outInfo):
            info = outInfo.GetInformationObject(0)
            info.Remove(sddp.TIME_STEPS())
            for step in (0.0, 1.0, 2.0):
                info.Append(sddp.TIME_STEPS(), step)
            info.Set(sddp.TIME_RANGE(), [0.0, 2.0], 2)
            if not tree:
                info.Set(sddp.WHOLE_EXTENT(), [-8, 8, -8, 8, -8, 8], 6)
            return 1

        def RequestData(self, request, inInfo, outInfo):
            info = outInfo.GetInformationObject(0)
            step = info.Get(sddp.UPDATE_TIME_STEP()) if info.Has(sddp.UPDATE_TIME_STEP()) else 0.0
            out = self.GetOutputData(outInfo, 0)
            if tree:
                out.SetNumberOfBlocks(1)
                out.SetBlock(0, _wavelet()())
            else:
                out.ShallowCopy(_wavelet()())
            out.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(), step)
            return 1

    return Source()


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("make", [lambda: ta.Contour(field="RTData", value=150.0),
                                  lambda: ta.Gradients(field="RTData")])
def test_time_passes_through(tree, make):
    f = make()
    source = _time_source(tree)
    f.SetInputConnection(source.GetOutputPort())
    f.UpdateInformation()
    assert f.GetOutputInformation(0).Get(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()) == (
        0.0, 1.0, 2.0)
    f.UpdateTimeStep(1.0)
    assert f.error is None
    out = f.GetOutputDataObject(0)
    assert out.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP()) == 1.0


# ── The world: MPI ranks ────────────────────────────────────────────

def test_without_an_mpi_controller_the_world_is_this_process():
    from tack.interop.vtk import world

    assert world().size == 1
    dummy = vtk.vtkDummyController()
    assert world(dummy).size == 1


def test_a_world_given_is_the_pieces_world():
    from tack.communication import World

    given = World()
    seen = []

    class Seeing(ta.GenerateGlobalIds):
        def over_pieces(self, pieces):
            seen.append(pieces.world)
            return super().over_pieces(pieces)

    _, halves = _halves()
    Seeing(world=given)(halves)
    assert seen == [given]


@pytest.mark.parametrize("source", ["comm", "vtk"])
def test_filters_over_pieces_span_mpi_ranks(mpiexec, source):
    """Three ranks, a slab of the wavelet each: global ids, ghost cells and the merge
    are the whole's, with the world given (an mpi4py communicator) or taken from
    VTK's controller (a VTK with MPI and vtkParallelMPI4Py; no wheel has them)."""
    import os

    if source == "vtk":
        pytest.importorskip("vtkmodules.vtkParallelMPI4Py",
                            reason="needs a VTK built with MPI and vtkParallelMPI4Py")
    run = mpiexec(3, os.path.join(os.path.dirname(__file__), "_vtk_algorithms_mpi.py"),
                  source)
    assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr
