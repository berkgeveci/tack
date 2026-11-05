# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``tack.interop.vtk.algorithms``: Tack's filters as VTK algorithms, in VTK pipelines
beside VTK's own, on datasets and on partitioned datasets. Each gives what the
Tack filter gives called directly, and where VTK has the same filter, VTK's
counts."""

from functools import lru_cache

import numpy as np
import pytest

import tack
import tack.dataset as td

vtk = pytest.importorskip("vtkmodules.all")
from tack.interop.vtk import algorithms as ta
from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

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
    """The entry is an attribute of the dataset's wrapper: it outlives the wrapper while
    the dataset lives (a fresh wrapper has it) and goes with the dataset."""
    import gc
    import weakref

    holder = vtk.vtkPartitionedDataSet()
    holder.append(_rotation())
    data = weakref.ref(ta._as_dataset(holder.GetPartition(0)))
    gc.collect()
    assert data() is not None and ta._as_dataset(holder.GetPartition(0)) is data()
    holder.SetPartition(0, None)
    gc.collect()
    assert data() is None


def test_an_entry_is_stale_after_modified_or_a_new_time():
    """``Modified()`` (here an array's) or another time step converts again; values
    written in place without ``Modified()`` reach the fields, which share VTK's memory,
    but the entry stays, as VTK's own pipeline would not run again either."""
    grid = _rotation()
    data = ta._as_dataset(grid)
    np.asarray(grid.GetPointData().GetArray("v"))[0] = 7.0
    assert ta._as_dataset(grid) is data
    assert data.fields["v"].values.numpy()[0, 0] == 7.0
    grid.GetPointData().GetArray("v").Modified()
    moved = ta._as_dataset(grid)
    assert moved is not data
    grid.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(), 3.0)
    assert ta._as_dataset(grid) is not moved


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


# ── Every filter, against the Tack filter called directly ───────────


@tack.func
def _x_plus_y(x):
    return x[0] + x[1]


@lru_cache
def _small():
    """A small wavelet image with vector fields: ``v`` (RTData in each component), ``w``,
    ``vg`` (``v``'s gradient at the points) and a cell field ``c``."""
    data = vtk_to_dataset(_wavelet((-3, 3, -3, 3, -3, 3))(), dtype=None)
    data.fields["v"] = td.composite(data, "RTData", "RTData", "RTData")
    data.fields["w"] = td.composite(data, "RTData", td.log_values(data, "RTData"), "RTData")
    data.fields["vg"] = td.gradients(data, "v", at="points")
    data.fields["c"] = td.values_at_centers(data, "RTData")
    return dataset_to_vtk(data)


@lru_cache
def _tets():
    return dataset_to_vtk(td.tetrahedralize(vtk_to_dataset(_small(), dtype=None)))


@lru_cache
def _surface():
    return dataset_to_vtk(td.extract_surface(vtk_to_dataset(_tets(), dtype=None)))


@lru_cache
def _edges():
    return dataset_to_vtk(td.feature_edges(vtk_to_dataset(_surface(), dtype=None)))


@lru_cache
def _polyhedra():
    return dataset_to_vtk(td.as_polyhedra(vtk_to_dataset(_tets(), dtype=None)))


@lru_cache
def _cloud():
    """200 points in the unit cube with ``h``, ``mass``, a vector ``g``, their SPH
    ``density`` and friends-of-friends ``point_region``."""
    points = np.random.default_rng(7).random((200, 3)).astype(np.float32)
    data = td.point_cloud(points)
    for name, value in [("h", 0.15), ("mass", 1.0)]:
        data.fields[name] = td.Field(td.H1(data.topology),
                                     tack.array(np.full(200, value, np.float32)))
    data.fields["g"] = td.composite(data, "h", "h", "mass")
    data = td.sph_density(data)
    data = td.connected_components(data, radius=0.08)
    return dataset_to_vtk(data)


@lru_cache
def _meters():
    """The tetrahedra with their positions in metres."""
    data = vtk_to_dataset(_tets(), dtype=None)
    return dataset_to_vtk(td.DataSet(data.topology, data.geometry, fields=data.fields,
                                     unit="m"))


@lru_cache
def _grid():
    return vtk.vtkImageData(dimensions=(4, 4, 4), origin=(-3, -3, -3), spacing=(2, 2, 2))


def _halves_of(vtk_data):
    pieces = vtk.vtkPartitionedDataSet()
    for piece in td.split(vtk_to_dataset(vtk_data, dtype=None), 2):
        pieces.append(dataset_to_vtk(piece))
    return pieces


@lru_cache
def _ghosted():
    pieces = vtk.vtkPartitionedDataSet()
    for piece in td.split(vtk_to_dataset(_small(), dtype=None), 2, ghost_layers=1):
        pieces.append(dataset_to_vtk(piece))
    return pieces


@lru_cache
def _cloud_pieces():
    data = vtk_to_dataset(_cloud(), dtype=None)
    ids = np.arange(data.num_cells)
    pieces = vtk.vtkPartitionedDataSet()
    for half in (ids[:100], ids[100:]):
        pieces.append(dataset_to_vtk(td.extract_cells(data, half)))
    return pieces


def _tack(vtk_data):
    """The Tack dataset (or pieces) a wrapper sees."""
    return vtk_to_dataset(vtk_data, dtype=None)


def _with(data, fields):
    """``data`` with ``fields`` added, as a field algorithm's output is."""
    return td.DataSet(data.topology, data.geometry, fields={**data.fields, **fields},
                      ghosts=data.ghosts, global_ids=data.global_ids, **data._frame())


def _on_faces(data, fields, on="faces"):
    with_fields = _with(data, fields)
    if on == "edges":
        return td.extract_edges(with_fields)
    return td.extract_surface(with_fields, faces=np.arange(data.topology.faces().num_faces))


def _sphere():
    return vtk.vtkSphere(center=(0, 0, 0), radius=2.0)


def _scale():
    transform = vtk.vtkTransform()
    transform.Scale(2.0, 1.0, 0.5)
    return transform


_EVERY = {
    # Cells kept whole
    "ExtractCells": (_small, lambda: ta.ExtractCells(cells=np.arange(0, 216, 7)),
                     lambda d: td.extract_cells(d, np.arange(0, 216, 7))),
    "ExtractGeometry": (_small, lambda: ta.ExtractGeometry(function=_sphere()),
                        lambda d: td.extract_geometry(d, td.Sphere((0, 0, 0), 2.0))),
    "ExtractExtent": (_small, lambda: ta.ExtractExtent(extent=((0, 4), (0, 4), (0, 6)), rate=2),
                      lambda d: td.extract_extent(d, ((0, 4), (0, 4), (0, 6)), rate=2)),
    "Mask": (_tets, lambda: ta.Mask(stride=3), lambda d: td.mask(d, 3)),
    "Mask of pieces": (lambda: _halves_of(_small()), lambda: ta.Mask(stride=3),
                       lambda d: td.PartitionedDataSet([td.mask(p, 3) for p in d])),
    # Cutting
    "Slice": (_small, lambda: ta.Slice(function=_sphere()),
              lambda d: td.slice(d, td.Sphere((0, 0, 0), 2.0))),
    # Surfaces and edges
    "BoundaryFaces": (_tets, ta.BoundaryFaces,
                      lambda d: td.extract_surface(d, faces=td.boundary_faces(d))),
    "ExtractEdges": (_small, ta.ExtractEdges, td.extract_edges),
    "FeatureEdges": (_surface, lambda: ta.FeatureEdges(angle=20.0),
                     lambda d: td.feature_edges(d, angle=20.0)),
    # Shapes for drawing
    "Tubes": (_edges, lambda: ta.Tubes(radius=0.1, sides=6), lambda d: td.tubes(d, 0.1, sides=6)),
    "Glyphs": (_cloud, lambda: ta.Glyphs(vectors="g", scale=0.1),
               lambda d: td.glyphs(d, "g", scale=0.1)),
    # Connectivity and winding
    "CheckWinding": (_polyhedra, ta.CheckWinding, lambda d: _with(d, {"inconsistent": td.Field(
        td.Constant(d.topology), td.check_winding(d).flags().astype(tack.u8))})),
    "MergePoints": (_tets, ta.MergePoints, td.merge_points),
    # Changing cell types
    "Triangulate": (_surface, ta.Triangulate, td.triangulate),
    "AsPolyhedra": (_tets, ta.AsPolyhedra, td.as_polyhedra),
    "AsPolygons": (_surface, ta.AsPolygons, td.as_polygons),
    "Shrink": (_tets, lambda: ta.Shrink(factor=0.8),
               lambda d: td.subdivide(td.shrink(d, 0.8), 1)),
    "Subdivide": (_small, lambda: ta.Subdivide(level=2), lambda d: td.subdivide(d, 2)),
    # Moving the geometry
    "Transform": (_small, lambda: ta.Transform(matrix=_scale()),
                  lambda d: td.transform(d, np.diag([2.0, 1.0, 0.5, 1.0]))),
    "Transform of tetrahedra": (_tets, lambda: ta.Transform(matrix=np.diag([2.0, 1.0, 0.5, 1.0])),
                                lambda d: td.transform(d, np.diag([2.0, 1.0, 0.5, 1.0]))),
    "Warp": (_small, lambda: ta.Warp(direction="v", scale=0.001),
             lambda d: td.warp(d, "v", scale=0.001)),
    "Cylindrical": (_small, ta.Cylindrical, td.cylindrical),
    "Spherical": (_tets, ta.Spherical, td.spherical),
    # Measures and quality
    "CellGeometry": (_small, ta.CellGeometry, lambda d: _with(d, td.cell_geometry(d)._asdict())),
    "FaceGeometry": (_small, ta.FaceGeometry,
                     lambda d: _on_faces(d, td.face_geometry(d)._asdict())),
    "CellCenters": (_tets, ta.CellCenters, lambda d: _with(d, {"center": td.cell_centers(d)})),
    "FaceCenters": (_tets, ta.FaceCenters, lambda d: _on_faces(d, {"center": td.face_centers(d)})),
    "EdgeLengths": (_small, ta.EdgeLengths,
                    lambda d: _on_faces(d, {"length": td.edge_lengths(d)}, "edges")),
    "ImplicitValues": (_small, lambda: ta.ImplicitValues(function=_sphere()), lambda d: _with(
        d, {"implicit": td.implicit_values(d, td.Sphere((0, 0, 0), 2.0))})),
    "CellIds": (_small, ta.CellIds, lambda d: _with(d, {"cell_id": td.cell_ids(d)})),
    "PointIds": (_small, ta.PointIds, lambda d: _with(d, {"point_id": td.point_ids(d)})),
    # Fields from fields
    "FlowQuantities": (_small, lambda: ta.FlowQuantities(gradient="vg"),
                       lambda d: _with(d, td.flow_quantities(d, "vg")._asdict())),
    "Project": (_small, lambda: ta.Project(field="RTData", to="cells", name="mean"), lambda d: _with(
        d, {"mean": td.project(d, "RTData", td.Constant(d.topology))})),
    "ValuesAtCenters": (_tets, lambda: ta.ValuesAtCenters(field="RTData", name="at"),
                        lambda d: _with(d, {"at": td.values_at_centers(d, "RTData")})),
    "Interpolate": (_small, lambda: ta.Interpolate(function=_x_plus_y), lambda d: _with(
        d, {"interpolated": td.interpolate(d, _x_plus_y, td.H1(d.topology))})),
    "Magnitude": (_small, lambda: ta.Magnitude(field="v"),
                  lambda d: _with(d, {"magnitude": td.magnitude(d, "v")})),
    "Dot": (_small, lambda: ta.Dot(a="v", b="w"), lambda d: _with(d, {"dot": td.dot(d, "v", "w")})),
    "Cross": (_small, lambda: ta.Cross(a="v", b="w"),
              lambda d: _with(d, {"cross": td.cross(d, "v", "w")})),
    "Composite": (_small, lambda: ta.Composite(fields=["RTData", ("RTData", "points"), "RTData"]),
                  lambda d: _with(d, {"composite": td.composite(d, "RTData", "RTData", "RTData")})),
    "LogValues": (_small, lambda: ta.LogValues(field="RTData", base=10.0),
                  lambda d: _with(d, {"log": td.log_values(d, "RTData", base=10.0)})),
    "Convert": (_meters, lambda: ta.Convert(unit="cm"), lambda d: td.convert(d, "cm")),
    # Finite volumes
    "Jump": (_small, lambda: ta.Jump(field="c"), lambda d: _on_faces(d, {"jump": td.jump(d, "c")})),
    "UpwindFlux": (_small, lambda: ta.UpwindFlux(field="c", velocity=(1.0, 0.5, 0.0)),
                   lambda d: _on_faces(d, {"flux": td.upwind_flux(d, "c", (1.0, 0.5, 0.0))})),
    # Probing, sampling and particles
    "Voxelize": (_tets, lambda: ta.Voxelize(grid=_grid(), samples=2),
                 lambda d: td.voxelize(d, _tack(_grid()), samples=2)),
    "Advect": (_small, lambda: ta.Advect(field="v", seeds=np.array([[0.5, 0.5, 0.5], [1, 0, 0]]),
                                         step_size=1e-4, steps=20),
               lambda d: td.advect(d, "v", np.array([[0.5, 0.5, 0.5], [1, 0, 0]]), 1e-4, 20)),
    # Point clouds
    "ExtractPoints": (_cloud, lambda: ta.ExtractPoints(function=vtk.vtkSphere(
        center=(0.5, 0.5, 0.5), radius=0.3)),
        lambda d: td.extract_points(d, td.Sphere((0.5, 0.5, 0.5), 0.3))),
    "ThresholdPoints": (_cloud, lambda: ta.ThresholdPoints(field="density", lower=0.0, upper=60.0),
                        lambda d: td.threshold_points(d, "density", 0.0, 60.0)),
    "MaskPoints": (_cloud, lambda: ta.MaskPoints(stride=3), lambda d: td.mask_points(d, 3)),
    "PointNormals": (_cloud, lambda: ta.PointNormals(k=8), lambda d: _with(
        d, {"normals": td.point_normals(d, k=8).fields["normals"]})),
    "Outliers": (_cloud, lambda: ta.Outliers(k=8), lambda d: td.outliers(d, k=8)),
    "VoxelDownsample": (_cloud, lambda: ta.VoxelDownsample(size=0.25),
                        lambda d: td.voxel_downsample(d, 0.25)),
    "Deposit": (_cloud, lambda: ta.Deposit(grid=vtk.vtkImageData(
        dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25))), lambda d: _deposited(d)),
    "SphDensity": (_cloud, lambda: ta.SphDensity(), lambda d: _with(
        d, {"density": td.sph_density(d).fields["density"]})),
    "HaloCatalog": (_cloud, ta.HaloCatalog, td.halo_catalog),
    # Pieces and ghosts
    "Split": (_small, lambda: ta.Split(pieces=2, ghost_layers=1),
              lambda d: td.split(d, 2, ghost_layers=1)),
    "RemoveGhosts": (_ghosted, ta.RemoveGhosts,
                     lambda d: td.PartitionedDataSet([td.remove_ghosts(p) for p in d])),
    "GhostPoints": (_cloud_pieces, lambda: ta.GhostPoints(radius=0.1),
                    lambda d: td.ghost_points(d, radius=0.1)),
}


def _deposited(data):
    grid = _tack(vtk.vtkImageData(dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25)))
    return td.DataSet(grid.topology, grid.geometry, fields={"deposit": td.deposit(data, grid)},
                      **grid._frame())


def _contents(vtk_data):
    """What a VTK dataset holds, as NumPy arrays by name: its class, structure, points,
    cells and arrays; for pieces, each piece's."""
    if isinstance(vtk_data, vtk.vtkPartitionedDataSet):
        return [_contents(vtk_data.GetPartition(i))
                for i in range(vtk_data.GetNumberOfPartitions())]
    out = {"class": vtk_data.GetClassName()}
    if isinstance(vtk_data, vtk.vtkImageData):
        out |= {"dims": vtk_data.dimensions, "origin": vtk_data.origin,
                "spacing": vtk_data.spacing}
    elif vtk_data.GetNumberOfPoints():
        out["points"] = np.asarray(vtk_data.points.data)
    if isinstance(vtk_data, vtk.vtkUnstructuredGrid):
        out |= {"types": np.asarray(vtk_data.cell_types),
                "offsets": np.asarray(vtk_data.cells.offsets_array),
                "connectivity": np.asarray(vtk_data.cells.connectivity_array)}
    for where, attributes in [("point", vtk_data.GetPointData()),
                              ("cell", vtk_data.GetCellData())]:
        for i in range(attributes.GetNumberOfArrays()):
            array = attributes.GetArray(i)
            if array is not None:
                out[f"{where} {array.GetName()}"] = np.asarray(array)
    return out


def _assert_same(got, expected):
    if isinstance(expected, list):
        assert isinstance(got, list) and len(got) == len(expected)
        for g, e in zip(got, expected):
            _assert_same(g, e)
        return
    assert sorted(got) == sorted(expected)
    for key, value in expected.items():
        if isinstance(value, str):
            assert got[key] == value, key
        else:
            np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(value), err_msg=key)


@pytest.mark.parametrize("name", sorted(_EVERY))
def test_each_wrapper_gives_its_tack_filters_output(name):
    data, make, direct = _EVERY[name]
    algorithm = make()
    out = algorithm(data())
    assert algorithm.error is None, algorithm.error
    _assert_same(_contents(out), _contents(dataset_to_vtk(direct(_tack(data())))))


def test_every_algorithm_has_a_wrapper():
    """Every algorithm of the reference page, but the in-place update_ghosts and the
    finite volume operators that take values on faces, which VTK keeps nowhere (and
    the first wrappers, named otherwise)."""
    import re
    from pathlib import Path

    page = (Path(__file__).parents[3] / "docs/vis-guide/reference/algorithms.md").read_text()
    names = set(re.findall(r"^\| \[`(\w+)`\]", page, re.M))
    later = {"update_ghosts", "net_flux", "divergence", "perot", "contour", "slice_plane", "clip", "threshold",
             "extract_surface", "tetrahedralize", "surface_normals", "connected_components",
             "streamlines", "probe", "gradients", "point_elevation", "cell_quality",
             "ghost_cells", "generate_global_ids", "merge_pieces"}
    camel = {n: "".join(part.capitalize() for part in n.split("_")) for n in names - later}
    missing = [n for n, c in camel.items() if not hasattr(ta, c)]
    assert not missing


# ── Active attributes ───────────────────────────────────────────────

def _active(data, where="points"):
    attributes = data.GetPointData() if where == "points" else data.GetCellData()
    return {role: (getattr(attributes, f"Get{role}")().GetName()
                   if getattr(attributes, f"Get{role}")() is not None else None)
            for role in ("Scalars", "Vectors", "Normals")}


def test_outputs_keep_the_inputs_active_attributes_as_vtks_filters_do():
    """Tack keeps no roles: a carried array came back plain, so a mapper colored by
    nothing (VTK's tests rendered white). The output's arrays of the names the
    input's active ones have are active again; the filter's own array where VTK's
    makes it so (an elevation's scalars, a surface's or tube's normals), over a
    carried one."""
    wavelet = _wavelet((-6, 6, -6, 6, -6, 6))()
    assert _active(ta.Elevation()(wavelet))["Scalars"] == "elevation"
    assert _active(ta.Contour(field="RTData", value=150.0)(wavelet))["Scalars"] == "RTData"
    assert _active(ta.Clip(by="RTData", value=150.0)(wavelet))["Scalars"] == "RTData"
    halves = ta.Threshold(field="RTData", lower=150.0, upper=300.0)(_halves()[1])
    assert all(_active(piece)["Scalars"] == "RTData" for piece in halves)
    probe = ta.Probe()
    probe.SetInputDataObject(0, vtk.vtkSphereSource()())
    probe.SetSourceData(wavelet)
    probe.Update()
    assert _active(probe.GetOutputDataObject(0))["Scalars"] == "RTData"   # the source's
    edges = ta.ExtractEdges()(vtk.vtkCubeSource()())         # carries the cube's Normals
    assert _active(ta.Tubes(radius=0.05, sides=8)(edges))["Normals"] == "normals"
    assert _active(ta.SurfaceNormals()(vtk.vtkSphereSource()()))["Normals"] == "normals"


# ── A component of a vector field ───────────────────────────────────

def test_contour_and_threshold_take_a_component_as_vtks_filters():
    """A vector field's component (a view of it, nothing copied), as VTK's
    ``ArrayComponent`` and ``vtkThreshold``'s selected component (its count of
    components meaning the magnitude): the cells VTK's give."""
    gradients = vtk.vtkGradientFilter()
    gradients.SetInputData(_wavelet((-6, 6, -6, 6, -6, 6))())
    gradients.SetInputArrayToProcess(0, 0, 0, 0, "RTData")
    gradients.Update()
    data = gradients.GetOutput()
    values = np.asarray(data.point_data["Gradients"])
    for c in (0, 1, 2):
        theirs = vtk.vtkContourFilter()
        theirs.SetInputData(data)
        theirs.SetInputArrayToProcess(0, 0, 0, 0, "Gradients")
        theirs.SetArrayComponent(c)
        theirs.SetValue(0, 2.0)
        theirs.Update()
        ours = ta.Contour(field="Gradients", component=c, value=2.0)(data)
        assert ours.number_of_cells == theirs.GetOutput().number_of_cells > 0
    for c in (0, 2, "magnitude"):
        v = np.linalg.norm(values, axis=1) if c == "magnitude" else values[:, c]
        lo, hi = np.percentile(v, [30, 80])
        theirs = vtk.vtkThreshold(lower_threshold=lo, upper_threshold=hi)
        theirs.SetInputData(data)
        theirs.SetInputArrayToProcess(0, 0, 0, 0, "Gradients")
        theirs.SetComponentModeToUseSelected()
        theirs.SetSelectedComponent(3 if c == "magnitude" else c)
        theirs.Update()
        ours = ta.Threshold(field="Gradients", component=c, lower=lo, upper=hi)(data)
        assert ours.number_of_cells == theirs.GetOutput().number_of_cells > 0
    first = ta.Contour(field="Gradients", value=2.0)(data)          # the first by default
    assert first.number_of_cells == ta.Contour(field="Gradients", component=0,
                                               value=2.0)(data).number_of_cells
    wrong = ta.Contour(field="RTData", component=1, value=150.0)
    wrong.SetInputDataObject(0, data)
    assert wrong.Update() == 0 and "scalar field" in str(wrong.error)


# ── Higher-order cells: fields at VTK's nodes ───────────────────────

def _quadratic_quads():
    """Two quad8 cells (VTK_QUADRATIC_QUAD) side by side, the shared edge's middle
    node pushed off the line: VTK's points are corners and edge middles, 13."""
    corners = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    middles = [(0.5, 0), (1, 0.5 + 0.0), (0.5, 1), (0, 0.5), (1.5, 0), (2, 0.5), (1.5, 1)]
    points = np.array([(x, y, 0.0) for x, y in corners + middles])
    points[7, 0] = 1.1                                  # the shared edge, curved
    grid = vtk.vtkUnstructuredGrid()
    grid.points = vtk.vtkPoints(data=points)
    grid.Allocate(2)
    grid.InsertNextCell(vtk.VTK_QUADRATIC_QUAD, 8, [0, 1, 4, 3, 6, 7, 8, 9])
    grid.InsertNextCell(vtk.VTK_QUADRATIC_QUAD, 8, [1, 2, 5, 4, 10, 11, 12, 7])
    return grid


def _quadratic_tetra():
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]        # VTK's tet10 order
    points = np.array(corners + [tuple((np.add(corners[a], corners[b])) / 2)
                                 for a, b in edges], float)
    points[4] += (0.0, -0.1, 0.0)                                    # one edge curved
    grid = vtk.vtkUnstructuredGrid()
    grid.points = vtk.vtkPoints(data=points)
    grid.Allocate(1)
    grid.InsertNextCell(vtk.VTK_QUADRATIC_TETRA, 10, list(range(10)))
    return grid


@pytest.mark.parametrize("make", [_quadratic_quads, _quadratic_tetra])
def test_a_field_filter_writes_every_node_of_higher_order_cells(make):
    """The new arrays have a value at each of VTK's points, the edge middles too, in
    VTK's order: the elevation of each node's own position, and its gradient
    constant. Tack numbers the nodes of its order-2 space otherwise, and keeps
    values on points at the corners only."""
    grid = make()
    elevation = ta.Elevation(low_point=(0, 0, 0), high_point=(2, 0, 0), name="e")(grid)
    assert elevation.GetNumberOfPoints() == grid.GetNumberOfPoints()
    x = np.asarray(grid.points.data)
    np.testing.assert_allclose(np.asarray(elevation.point_data["e"]),
                               np.clip(x[:, 0] / 2, 0, 1), atol=1e-6)
    gradient = ta.Gradients(field=("e", "points"), at="points", name="g")(elevation)
    values = np.asarray(gradient.point_data["g"])
    assert values.shape == (grid.GetNumberOfPoints(), 3)
    np.testing.assert_allclose(values, np.tile([0.5, 0, 0], (len(values), 1)), atol=1e-5)


def test_higher_order_nodes_on_another_device_and_integers_refused():
    """A copy of the dataset on another device finds the nodes through the dataset kept
    for the VTK one; an integer field on the points has no value at an edge's
    middle, and is refused."""
    from tack.runtime import dispatch
    from tack.runtime.cpu import CPUBackend

    class SecondCPU(CPUBackend):
        display_name = "second device"

        @property
        def memory_domain(self):
            return ("cpu", 1)

    grid = _quadratic_quads()
    other = dispatch.as_device(SecondCPU())
    moved = ta.Elevation(low_point=(0, 0, 0), high_point=(2, 0, 0), name="e", device=other)
    out = moved(grid)
    assert moved.error is None, moved.error
    np.testing.assert_allclose(np.asarray(out.point_data["e"]),
                               np.asarray(grid.points.data)[:, 0] / 2, atol=1e-6)
    ids = ta.PointIds()
    ids.SetInputDataObject(0, grid)
    assert ids.Update() == 0 and "integers" in str(ids.error)


def test_temporal_statistics_on_higher_order_cells():
    statistics = ta.RunningStatistics(fields=["e"])
    for t in (0.0, 1.0):
        step = ta.Elevation(low_point=(0, 0, 0), high_point=(2 + t, 0, 0), name="e")(
            _quadratic_quads())
        step.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(), t)
        statistics.SetInputDataObject(0, step)
        statistics.Update()
        assert statistics.error is None, statistics.error
    x = np.asarray(_quadratic_quads().points.data)[:, 0]
    np.testing.assert_allclose(
        np.asarray(statistics.GetOutputDataObject(0).point_data["e_mean"]),
        (x / 2 + x / 3) / 2, atol=1e-6)


def test_connected_components_keeps_its_inputs_type():
    """VTK's vtkConnectivityFilter keeps a polygonal input polygonal: the arrays go onto
    a copy of the input (dataset_to_vtk made it an unstructured grid, and the
    algorithm refused its own output)."""
    two = vtk.vtkAppendPolyData()
    two.AddInputData(vtk.vtkSphereSource()())
    two.AddInputData(vtk.vtkSphereSource(center=(3, 0, 0))())
    two.Update()
    out = ta.ConnectedComponents()(two.GetOutput())
    assert isinstance(out, vtk.vtkPolyData)
    assert sorted(np.unique(np.asarray(out.cell_data["region"]))) == [0, 1]
    assert "point_region" in out.point_data.keys()


# ── Seeds on a port ─────────────────────────────────────────────────

@pytest.mark.parametrize("make", [
    lambda: ta.Streamlines(field="v", step_size=1e-4, steps=20),
    lambda: ta.Advect(field="v", step_size=1e-4, steps=20)])
def test_seeds_come_from_port_1(make):
    seeds = np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [-1.0, 0.5, 0.0]])
    source = vtk.vtkPolyData()
    source.points = vtk.vtkPoints(data=seeds)
    from_port = make()
    from_port.SetInputDataObject(0, _small())
    from_port.SetSourceData(source)
    from_port.Update()
    assert from_port.error is None, from_port.error
    given = make()
    given.seeds = seeds
    _assert_same(_contents(from_port.GetOutputDataObject(0)), _contents(given(_small())))


def test_sph_interpolate_takes_its_particles_on_port_1():
    particles = _cloud()
    target = vtk.vtkPolyData()
    target.points = vtk.vtkPoints(data=np.array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]]))
    sph = ta.SphInterpolate(fields=["h"])
    sph.SetInputDataObject(0, target)
    sph.SetSourceData(particles)
    sph.Update()
    assert sph.error is None, sph.error
    expected = td.sph_interpolate(_tack(particles), _tack(target), fields=["h"])
    _assert_same(_contents(sph.GetOutputDataObject(0)), _contents(dataset_to_vtk(expected)))


# ── Filters that take the pieces together ───────────────────────────

def _whole_and_halves(with_velocity=False):
    whole, halves = _halves()
    if with_velocity:
        for data in [whole, *(halves.GetPartition(i) for i in range(2))]:
            values = np.asarray(data.point_data["RTData"])
            velocity = vtk.vtkFloatArray()       # VTK's own memory, not a NumPy temporary's
            velocity.SetNumberOfComponents(3)
            velocity.SetNumberOfTuples(values.size)
            np.asarray(velocity)[:] = np.stack([np.ones_like(values), 0.5 * np.ones_like(values),
                                                np.zeros_like(values)], axis=1)
            velocity.SetName("v")
            data.point_data.AddArray(velocity)
    return whole, halves


def _cells(pieces):
    return sum(pieces.GetPartition(i).number_of_cells
               for i in range(pieces.GetNumberOfPartitions()))


def test_the_surface_of_pieces_is_the_wholes():
    whole, halves = _whole_and_halves()
    assert _cells(ta.ExtractSurface()(halves)) == ta.ExtractSurface()(whole).number_of_cells


def test_connected_regions_join_across_pieces():
    whole, halves = _whole_and_halves()
    out = ta.ConnectedComponents()(halves)
    regions = np.concatenate([np.asarray(out.GetPartition(i).cell_data["region"])
                              for i in range(2)])
    assert set(np.unique(regions)) == {0}          # one region, not one per piece


@pytest.mark.parametrize("make", [ta.Streamlines, ta.Advect])
def test_particles_go_from_piece_to_piece(make):
    """A seed in the first half moving along +x crosses into the second half: traced
    over the whole, it ends where it ends on the whole mesh. On pieces with a layer
    of ghost cells, as GhostCells makes them, particles are handed over after whole
    steps; without one, a particle leaving a piece is pushed just past its boundary,
    a shortened step that counts against ``steps``."""
    whole, halves = _whole_and_halves(with_velocity=True)
    seeds = np.array([[-6.0, 0.0, 0.0], [-2.0, 1.0, 1.0]])
    options = {"field": "v", "seeds": seeds, "step_size": 0.25, "steps": 40}
    on_whole = make(**options)(whole)
    on_halves = (ta.GhostCells() >> make(**options))(halves)

    def ends(data):
        parts = ([data] if not isinstance(data, vtk.vtkPartitionedDataSet)
                 else [data.GetPartition(i) for i in range(data.GetNumberOfPartitions())])
        return np.concatenate([np.asarray(p.points.data) for p in parts
                               if p is not None and p.number_of_points])

    furthest_whole = ends(on_whole)[:, 0].max()
    furthest_halves = ends(on_halves)[:, 0].max()
    assert furthest_whole > 0.0                     # the paths do cross x = 0
    np.testing.assert_allclose(furthest_halves, furthest_whole, rtol=1e-6)


def test_a_dataset_the_collector_frees_takes_its_entry():
    """A dataset an algorithm read, freed by Python's collector in a cycle made after
    it was read (as a traceback's frames hold one): a DeleteEvent observer, referenced
    only by it, was cleared as garbage first and calling it crashed Python (VTK bugs
    19). The entry is the wrapper's attribute, no observer: it goes with the dataset."""
    import gc
    import weakref

    class Holder:
        pass

    entries = []
    for _ in range(3):
        data = _wavelet((-4, 4, -4, 4, -4, 4))()
        assert ta.Contour(field="RTData", value=150.0)(data).number_of_cells > 0
        entries.append(weakref.ref(ta._as_dataset(data)))
        holder = Holder()
        holder.cycle = holder
        holder.data = data
        del data, holder
        gc.collect()
    assert all(entry() is None for entry in entries)


# ── Reductions: Statistics, Entropy, Integrate ──────────────────────

def _row(table, row=0):
    return {table.GetColumnName(i): table.GetValue(row, i).ToString()
            for i in range(table.GetNumberOfColumns())}


def test_statistics_are_tacks_in_a_table():
    image = _wavelet()()
    table = ta.Statistics(fields=["RTData"])(image)
    assert isinstance(table, vtk.vtkTable) and table.GetNumberOfRows() == 1
    expected = td.statistics(vtk_to_dataset(image, dtype=None), "RTData")
    row = _row(table)
    assert (row["field"], row["where"], row["component"]) == ("RTData", "points", "-1")
    for name in ta.Statistics.NAMES:
        column = table.GetColumnByName(name)
        assert column.GetValue(0) == pytest.approx(getattr(expected, name), rel=1e-12), name


def test_a_vector_field_is_reduced_a_component_at_a_time():
    image = (_wavelet() >> ta.Gradients(field="RTData", at="points"))()
    table = ta.Statistics(fields=[("gradient", "points")])(image)
    assert table.GetNumberOfRows() == 3
    assert [table.GetColumnByName("component").GetValue(i) for i in range(3)] == [0, 1, 2]
    values = np.asarray(image.point_data["gradient"])
    np.testing.assert_allclose([table.GetColumnByName("mean").GetValue(i) for i in range(3)],
                               values.astype(np.float64).mean(axis=0), rtol=1e-6)


def test_every_field_by_default_and_by_set_input_array_to_process():
    image = (_wavelet() >> ta.Elevation())()
    table = ta.Statistics()(image)
    assert sorted(table.GetColumnByName("field").GetValue(i)
                  for i in range(table.GetNumberOfRows())) == ["RTData", "elevation"]
    chosen = ta.Statistics()
    chosen.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                  "elevation")
    assert _row(chosen(image))["field"] == "elevation"


def test_entropy_is_tacks():
    image = _wavelet()()
    table = ta.Entropy(fields=["RTData"], bins=16)(image)
    expected = td.entropy(vtk_to_dataset(image, dtype=None), "RTData", bins=16)
    assert table.GetColumnByName("entropy").GetValue(0) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("marks", [ta.GenerateGlobalIds, ta.GhostCells])
def test_a_reduction_of_pieces_and_trees_is_the_wholes(marks):
    """The halves share a plane of points: with its copies' owner marked (global ids or
    ghost cells), each is counted once, and the halves' statistics are the whole's."""
    whole, halves = _halves()
    expected = ta.Statistics(fields=["RTData"])(whole)
    tree = vtk.vtkPartitionedDataSetCollection()
    tree.SetNumberOfPartitionedDataSets(1)
    tree.SetPartitionedDataSet(0, marks()(halves))
    for table in [(marks() >> ta.Statistics(fields=["RTData"]))(halves),
                  ta.Statistics(fields=["RTData"])(tree)]:
        assert table.GetColumnByName("n").GetValue(0) == expected.GetColumnByName(
            "n").GetValue(0)
        # Over pieces, sums of f32 values and moments combine in another order.
        for name in ta.Statistics.NAMES:
            assert table.GetColumnByName(name).GetValue(0) == pytest.approx(
                expected.GetColumnByName(name).GetValue(0), rel=1e-6), name


def test_integrate_is_vtks_integrate_attributes():
    image = (_wavelet() >> ta.Gradients(field="RTData", name="g"))()
    ours = ta.Integrate()(image)
    theirs = vtk.vtkIntegrateAttributes()(image)
    assert isinstance(ours, vtk.vtkUnstructuredGrid) and ours.number_of_points == 1
    np.testing.assert_allclose(ours.points.data.GetTuple3(0), theirs.points.data.GetTuple3(0),
                               atol=1e-9)
    assert ours.cell_data["Volume"].GetValue(0) == pytest.approx(
        theirs.cell_data["Volume"].GetValue(0), rel=1e-12)
    assert ours.point_data["RTData"].GetValue(0) == pytest.approx(
        theirs.point_data["RTData"].GetValue(0), rel=1e-6)
    np.testing.assert_allclose(ours.cell_data["g"].GetTuple3(0),
                               theirs.cell_data["g"].GetTuple3(0), rtol=1e-6)
    _, halves = _halves()
    pieces = ta.Integrate()(halves)
    assert pieces.cell_data["Volume"].GetValue(0) == pytest.approx(
        theirs.cell_data["Volume"].GetValue(0), rel=1e-12)


def test_a_surfaces_measure_is_its_area():
    surface = (_wavelet() >> ta.ExtractSurface())()
    out = ta.Integrate()(surface)
    assert "Area" in out.cell_data.keys()
    assert out.cell_data["Area"].GetValue(0) == pytest.approx(6 * 16 * 16, rel=1e-12)


# ── AMR hierarchies ─────────────────────────────────────────────────

def _pulse():
    """VTK's AMR Gaussian pulse: a level-0 box and two finer boxes in it, the pulse as
    cell data."""
    from vtkmodules.vtkFiltersAMR import vtkAMRGaussianPulseSource

    return vtkAMRGaussianPulseSource(dimension=3)()


def _hierarchy(amr):
    return vtk_to_dataset(amr, dtype=None)


def _amr_bounds(amr):
    """``(lo, hi)``: a hierarchy's ``GetBounds()`` returns a pointer's name (9.7.1)."""
    bounds = [0.0] * 6
    amr.GetBounds(bounds)
    return np.asarray(bounds[::2]), np.asarray(bounds[1::2])


PULSE = ("Gaussian-Pulse", td.Constant)


def test_an_amr_hierarchy_is_reduced_once_per_region():
    """What a finer box covers counts once: the statistics, entropy and integral of the
    hierarchy are Tack's, the cells counted those no finer box covers."""
    amr = _pulse()
    hierarchy = _hierarchy(amr)
    stats = td.statistics(hierarchy, PULSE)
    table = ta.Statistics(fields=["Gaussian-Pulse"])(amr)
    for name in ta.Statistics.NAMES:
        assert table.GetColumnByName(name).GetValue(0) == pytest.approx(
            getattr(stats, name), rel=1e-12), name
    every_cell = sum(amr.GetDataSetAsCartesianGrid(level, i).number_of_cells
                     for level in range(amr.GetNumberOfLevels())
                     for i in range(amr.GetNumberOfBlocks(level)))
    assert stats.n < every_cell
    entropy = ta.Entropy(fields=["Gaussian-Pulse"], bins=16)(amr)
    assert entropy.GetColumnByName("entropy").GetValue(0) == pytest.approx(
        td.entropy(hierarchy, PULSE, bins=16), rel=1e-12)
    integral = ta.Integrate(fields=["Gaussian-Pulse"])(amr)
    lo, hi = _amr_bounds(amr)
    assert integral.cell_data["Volume"].GetValue(0) == pytest.approx(np.prod(hi - lo),
                                                                     rel=1e-12)
    expected = td.integrate(hierarchy._as_pieces(), fields=[PULSE]).values
    assert integral.cell_data["Gaussian-Pulse"].GetValue(0) == pytest.approx(
        next(iter(expected.values())), rel=1e-12)


def test_probe_and_voxelize_take_an_amr_hierarchy_at_its_finest_boxes():
    amr = _pulse()
    hierarchy = _hierarchy(amr)
    lo, hi = _amr_bounds(amr)
    image = vtk.vtkImageData(dimensions=(9, 9, 9), origin=tuple(lo),
                             spacing=tuple((hi - lo) / 8))
    probe = ta.Probe()
    probe.SetInputDataObject(0, image)
    probe.SetSourceData(amr)
    probe.Update()
    assert probe.error is None, probe.error
    expected = td.probe(hierarchy, vtk_to_dataset(image, dtype=None))
    np.testing.assert_array_equal(
        np.asarray(probe.GetOutputDataObject(0).point_data["Gaussian-Pulse"]),
        expected.fields["Gaussian-Pulse"].values.numpy())
    voxels = ta.Voxelize(grid=image, fields=["Gaussian-Pulse"], samples=2)(amr)
    assert isinstance(voxels, vtk.vtkImageData)
    direct = td.voxelize(hierarchy, vtk_to_dataset(image, dtype=None),
                         fields=[PULSE], samples=2)
    np.testing.assert_array_equal(np.asarray(voxels.cell_data["fraction"]),
                                  direct.fields["fraction"].values.numpy())


def test_ghost_cells_of_an_amr_hierarchy_feed_the_next_amr_filter():
    """The grown boxes go out as the input's hierarchy with larger blocks, and the next
    filter takes the grown hierarchy as it was made, in a pipeline or given what
    calling GhostCells returned (a shallow copy): VTK's blocks with ghost layers
    could not be converted again. Its ghosts count nowhere."""
    amr = _pulse()
    grown = ta.GhostCells(layers=1)(amr)
    assert isinstance(grown, vtk.vtkOverlappingAMR)
    expected = td.ghost_cells(_hierarchy(amr), 1)
    for (level, box), (lv, i, block) in zip(expected, ta._blocks(grown)):
        assert level == lv
        assert block.GetDimensions() == tuple(box.topology.point_dims)
        assert grown.GetAMRBox(lv, i).GetNumberOfCells() == amr.GetAMRBox(
            lv, i).GetNumberOfCells()
    assert ta._as_dataset(grown).layers == 1
    once = ta.Statistics(fields=["Gaussian-Pulse"])(amr)
    pipeline = ta.GhostCells(layers=1) >> ta.Statistics(fields=["Gaussian-Pulse"])
    for table in (pipeline(amr), ta.Statistics(fields=["Gaussian-Pulse"])(grown)):
        assert table.GetColumnByName("n").GetValue(0) == once.GetColumnByName(
            "n").GetValue(0)
        assert table.GetColumnByName("mean").GetValue(0) == pytest.approx(
            once.GetColumnByName("mean").GetValue(0), rel=1e-12)
    # The input's entry is its own: the output's metadata is not the input's.
    assert ta._as_dataset(amr).layers == 0


def test_average_down_puts_the_fine_values_in_the_coarse_cells_and_leaves_its_input():
    amr = _pulse()
    before = [np.asarray(block.cell_data["Gaussian-Pulse"]).copy()
              for _, _, block in ta._blocks(amr)]
    out = ta.AverageDown(field="Gaussian-Pulse")(amr)
    assert isinstance(out, vtk.vtkOverlappingAMR)
    for kept, (_, _, block) in zip(before, ta._blocks(amr)):
        np.testing.assert_array_equal(np.asarray(block.cell_data["Gaussian-Pulse"]), kept)
    expected = _hierarchy(amr)
    td.average_down(expected, PULSE)
    changed = False
    for (_, box), (_, _, block), kept in zip(expected, ta._blocks(out), before):
        values = np.asarray(block.cell_data["Gaussian-Pulse"])
        np.testing.assert_array_equal(values, box.fields[PULSE].values.numpy())
        changed |= not np.array_equal(values, kept)
    assert changed                              # the coarse box's refined cells


def test_a_field_filter_runs_box_by_box_and_keeps_the_hierarchy():
    """Each block gets the fields the filter computes on it alone; the output is the
    input's hierarchy, its blocks shallow copies, and the next filter takes it."""
    amr = _pulse()
    out = ta.Elevation()(amr)
    assert isinstance(out, vtk.vtkOverlappingAMR)
    assert out.GetNumberOfLevels() == amr.GetNumberOfLevels()
    for (level, index, block), (_, _, given) in zip(ta._blocks(out), ta._blocks(amr)):
        assert out.GetAMRBox(level, index).GetNumberOfCells() == given.number_of_cells
        alone = ta.Elevation()(given)
        np.testing.assert_array_equal(np.asarray(block.point_data["elevation"]),
                                      np.asarray(alone.point_data["elevation"]))
        assert "elevation" not in given.point_data.keys()
    # The input's Tack hierarchy with the field: the same topologies, nothing built again.
    box = next(box for _, box in ta._as_dataset(out))
    assert "elevation" in box.fields
    assert box.topology is next(box for _, box in ta._as_dataset(amr)).topology
    table = ta.Statistics(fields=["elevation"])(out)
    assert table.GetColumnByName("n").GetValue(0) > 0


def test_a_filter_that_makes_a_dataset_gives_a_partitioned_dataset_per_level():
    """Box by box, the cells finer boxes cover left out: each level's cells are those
    the Tack filter keeps of its boxes in the hierarchy, marks and all: the coarse
    level's, fewer than its block has. A contour after the points' values is one per
    level."""
    amr = _pulse()
    hierarchy = _hierarchy(amr)
    threshold = ta.Threshold(field=("Gaussian-Pulse", "cells"), lower=-1.0, upper=1.0)
    out = threshold(amr)
    assert isinstance(out, vtk.vtkPartitionedDataSetCollection)
    assert out.GetNumberOfPartitionedDataSets() == amr.GetNumberOfLevels()
    for level, pieces in enumerate(hierarchy.levels):
        assert out.GetMetaData(level).Get(vtk.vtkCompositeDataSet.NAME()) == f"level{level}"
        got = sum(block.number_of_cells for block in out.GetPartitionedDataSet(level))
        expected = sum(td.threshold(box, PULSE, -1.0, 1.0).topology.num_cells
                       for box in pieces)
        assert got == expected
    coarse = sum(block.number_of_cells for block in out.GetPartitionedDataSet(0))
    assert 0 < coarse < amr.GetDataSetAsCartesianGrid(0, 0).number_of_cells  # refined out
    peak = max(np.asarray(block.cell_data["Gaussian-Pulse"]).max()
               for _, _, block in ta._blocks(amr))
    surfaces = (ta.Project(field=("Gaussian-Pulse", "cells"), to="points", name="p")
                >> ta.Contour(field="p", value=peak / 4))(amr)
    assert isinstance(surfaces, vtk.vtkPartitionedDataSetCollection)
    assert sum(block.number_of_cells for level in range(amr.GetNumberOfLevels())
               for block in surfaces.GetPartitionedDataSet(level)) > 0


def test_a_hierarchy_is_probed_at_its_points_box_by_box():
    amr = _pulse()
    probe = ta.Probe()
    probe.SetInputDataObject(0, amr)
    probe.SetSourceData(_wavelet((-4, 4, -4, 4, -4, 4))())
    probe.Update()
    assert probe.error is None, probe.error
    out = probe.GetOutputDataObject(0)
    assert isinstance(out, vtk.vtkPartitionedDataSetCollection)
    assert all("RTData" in block.point_data.keys()
               for level in range(out.GetNumberOfPartitionedDataSets())
               for block in out.GetPartitionedDataSet(level))


def test_filters_whose_answer_box_by_box_would_mislead_refuse_a_hierarchy():
    amr = _pulse()
    ids = ta.GenerateGlobalIds()
    ids.SetInputDataObject(0, amr)
    assert ids.Update() == 0 and isinstance(ids.error, TypeError)
    assert "would mislead" in str(ids.error)
    lines = ta.Streamlines(field="v", seeds=np.zeros((1, 3)))
    lines.SetInputDataObject(0, amr)
    assert lines.Update() == 0 and isinstance(lines.error, TypeError)
    seeded = ta.Streamlines(field="v", step_size=1e-4, steps=2)
    seeded.SetInputDataObject(0, _small())
    seeded.SetSourceData(amr)
    assert seeded.Update() == 0 and "port 1" in str(seeded.error)


def test_an_amr_hierarchy_is_one_tack_hierarchy_until_a_block_changes():
    """In its shallow copies too; a block's array modified, one added, or a block
    replaced, it is converted again (its own modification time follows none)."""
    amr = _pulse()
    first = ta._as_dataset(amr)
    assert ta._as_dataset(amr) is first
    copy = amr.NewInstance()
    copy.ShallowCopy(amr)
    assert ta._as_dataset(copy) is first
    amr.GetDataSetAsCartesianGrid(1, 0).cell_data["Gaussian-Pulse"].Modified()
    second = ta._as_dataset(amr)
    assert second is not first
    amr.GetDataSetAsCartesianGrid(1, 0).cell_data["twice"] = 2 * np.asarray(
        amr.GetDataSetAsCartesianGrid(1, 0).cell_data["Gaussian-Pulse"])
    third = ta._as_dataset(amr)
    assert third is not second
    assert "twice" in [box for _, box in third][1].fields      # level 1's first box
    block = amr.GetDataSetAsCartesianGrid(0, 0).NewInstance()
    block.DeepCopy(amr.GetDataSetAsCartesianGrid(0, 0))
    amr.SetDataSet(0, 0, block)
    assert ta._as_dataset(amr) is not third


def test_an_amr_filter_runs_on_the_device_it_is_given():
    from tack.runtime import dispatch
    from tack.runtime.cpu import CPUBackend

    class SecondCPU(CPUBackend):
        """A CPU backend posing as another device's, as ``test_partitioned.py`` makes."""

        display_name = "second device"

        @property
        def memory_domain(self):
            return ("cpu", 1)

    amr = _pulse()
    other = dispatch.as_device(SecondCPU())
    table = ta.Statistics(fields=["Gaussian-Pulse"], device=other)(amr)
    once = ta.Statistics(fields=["Gaussian-Pulse"])(amr)
    assert all(box.device == other for _, box in ta._as_dataset(amr, other))
    assert table.GetColumnByName("mean").GetValue(0) == pytest.approx(
        once.GetColumnByName("mean").GetValue(0), rel=1e-12)


# ── Temporal filters: in situ and driving ───────────────────────────

TIMES = (0.0, 0.5, 1.0, 1.5, 2.0)


def _flow_step(t):
    """A step at ``t``: a grid with a velocity ``v`` turning with time and a scalar
    ``T`` growing with it."""
    grid = td.uniform_grid((11, 11, 3))
    grid.time = t
    x = grid.positions().astype(np.float64)
    v = np.zeros_like(x)
    v[:, 0], v[:, 1] = 1.0, 0.2 * t
    grid.fields["v"] = td.Field(td.H1(grid), tack.array(v.astype(np.float32), tack.vec3()))
    grid.fields["T"] = td.Field(td.H1(grid), tack.array((x[:, 0] * (1.0 + t) ** 2)
                                                      .astype(np.float32)))
    return grid


def _pipeline_keys():
    from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline

    return vtkStreamingDemandDrivenPipeline


def _steps_source(times=TIMES, no_prior_access=None):
    """A source of ``_flow_step``s at ``times`` that records each time it is asked
    for (``asked``); with ``no_prior_access``, it says so as an in situ adaptor does
    (``NO_PRIOR_TEMPORAL_ACCESS``, that value)."""
    from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

    s = _pipeline_keys()

    class Steps(VTKPythonAlgorithmBase):
        def __init__(self):
            VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1,
                                            outputType="vtkImageData")
            self.asked = []
            self.no_prior_access = no_prior_access

        def RequestInformation(self, request, inInfo, outInfo):
            info = outInfo.GetInformationObject(0)
            info.Set(s.TIME_STEPS(), list(times), len(times))
            info.Set(s.TIME_RANGE(), [times[0], times[-1]], 2)
            info.Set(s.WHOLE_EXTENT(), [0, 10, 0, 10, 0, 2], 6)
            if self.no_prior_access is not None:
                info.Set(s.NO_PRIOR_TEMPORAL_ACCESS(), self.no_prior_access)
            return 1

        def RequestData(self, request, inInfo, outInfo):
            info = outInfo.GetInformationObject(0)
            t = info.Get(s.UPDATE_TIME_STEP()) if info.Has(s.UPDATE_TIME_STEP()) else times[0]
            self.asked.append(t)
            out = self.GetOutputData(outInfo, 0)
            out.ShallowCopy(dataset_to_vtk(_flow_step(t)))
            out.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(), t)
            return 1

    return Steps()


def _in_situ(make, times=TIMES):
    """``make()``'s output after each step handed to it in turn, as in situ."""
    algorithm, outputs = make(), []
    for t in times:
        algorithm.SetInputDataObject(0, dataset_to_vtk(_flow_step(t)))
        algorithm.Update()
        assert algorithm.error is None, algorithm.error
        out = algorithm.GetOutputDataObject(0)
        copy = out.NewInstance()
        copy.DeepCopy(out)
        copy.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(),
                                  out.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP()))
        outputs.append(copy)
    return outputs


def _driven(make, source):
    algorithm = make()
    algorithm.SetInputConnection(0, source.GetOutputPort())
    return algorithm


def _time_of(data):
    info = data.GetInformation()
    key = vtk.vtkDataObject.DATA_TIME_STEP()
    return info.Get(key) if info.Has(key) else None


def _same_arrays(a, b, names):
    for name in names:
        np.testing.assert_array_equal(np.asarray(a.point_data[name]),
                                      np.asarray(b.point_data[name]), err_msg=name)


def test_running_statistics_driving_take_every_step_once_and_give_no_time():
    """Driving gives what the steps give in situ after the last, at no time, its output
    reporting no time steps; asked again (an update, another time requested) it runs
    nothing again."""
    make = lambda: ta.RunningStatistics(fields=["T"])
    source = _steps_source()
    stats = _driven(make, source)
    stats.Update()
    assert stats.error is None, stats.error
    out = stats.GetOutputDataObject(0)
    assert source.asked == list(TIMES)
    assert _time_of(out) is None
    s = _pipeline_keys()
    assert not stats.GetOutputInformation(0).Has(s.TIME_STEPS())
    assert out.GetFieldData().GetArray("count").GetValue(0) == len(TIMES)
    assert sorted(out.point_data.keys()) == ["T_max", "T_mean", "T_min", "T_variance"]
    last = _in_situ(make)[-1]
    assert _time_of(last) == TIMES[-1]
    _same_arrays(out, last, ["T_mean", "T_variance", "T_min", "T_max"])
    expected = np.mean([_flow_step(t).fields["T"].values.numpy() for t in TIMES], axis=0)
    np.testing.assert_allclose(np.asarray(out.point_data["T_mean"]), expected, rtol=1e-6)
    stats.Update()
    stats.UpdateTimeStep(1.0)
    assert source.asked == list(TIMES)
    stats.time_range = (0.5, 1.5)                   # a parameter: starts over
    stats.Update()
    assert source.asked == [*TIMES, 0.5, 1.0, 1.5]
    assert stats.GetOutputDataObject(0).GetFieldData().GetArray("count").GetValue(0) == 3


@pytest.mark.parametrize("make", [
    lambda: ta.Pathlines(field="v", seeds=[[1.0, 5.0, 1.0], [2.0, 2.0, 1.0]]),
    lambda: ta.Advection(field="v", seeds=[[1.0, 5.0, 1.0], [2.0, 2.0, 1.0]])])
def test_particles_driving_hold_every_step_and_in_situ_the_steps_so_far(make):
    """Driving: pathlines through every step; advection's particles at every step in
    one cloud, ``step`` beside ``time``. They are what the same steps give in situ."""
    source = _steps_source()
    driven = _driven(make, source)
    driven.Update()
    assert driven.error is None, driven.error
    out = driven.GetOutputDataObject(0)
    assert source.asked == list(TIMES) and _time_of(out) is None
    in_situ = _in_situ(make)
    if isinstance(driven, ta.Pathlines):
        last = in_situ[-1]
        np.testing.assert_array_equal(np.asarray(out.points.data), np.asarray(last.points.data))
        assert out.number_of_cells == 2
        return
    steps = np.asarray(out.point_data["step"])
    assert out.number_of_points == sum(o.number_of_points for o in in_situ)
    for k, step in enumerate(in_situ):
        at = steps == k
        np.testing.assert_array_equal(np.asarray(out.points.data)[at],
                                      np.asarray(step.points.data))
        np.testing.assert_array_equal(np.asarray(out.point_data["time"])[at],
                                      np.full(step.number_of_points, TIMES[k]))
        assert _time_of(step) == TIMES[k]


def test_streaklines_drive_to_the_time_requested_from_the_steps_taken():
    """At the time requested, stamped with it, as in situ after those steps; a later
    time takes only the steps after; an earlier one starts over; the same one
    takes none."""
    make = lambda: ta.Streaklines(field="v", seeds=[[1.0, 5.0, 1.0]], releases=2)
    source = _steps_source()
    streaks = _driven(make, source)
    streaks.UpdateTimeStep(1.0)
    assert streaks.error is None, streaks.error
    assert source.asked == [0.0, 0.5, 1.0]
    out = streaks.GetOutputDataObject(0)
    assert _time_of(out) == 1.0
    in_situ = _in_situ(make)
    np.testing.assert_array_equal(np.asarray(out.points.data),
                                  np.asarray(in_situ[2].points.data))
    streaks.UpdateTimeStep(2.0)
    assert source.asked == [0.0, 0.5, 1.0, 1.5, 2.0]
    np.testing.assert_array_equal(np.asarray(streaks.GetOutputDataObject(0).points.data),
                                  np.asarray(in_situ[4].points.data))
    streaks.UpdateTimeStep(0.5)
    assert source.asked[5:] == [0.0, 0.5]
    streaks.UpdateTimeStep(0.5)
    assert len(source.asked) == 7


def test_a_time_derivative_takes_the_step_before_once():
    source = _steps_source()
    rate = _driven(lambda: ta.TimeDerivative(fields=["T"]), source)
    rate.UpdateTimeStep(1.0)
    assert rate.error is None, rate.error
    assert source.asked == [0.5, 1.0]
    out = rate.GetOutputDataObject(0)
    assert _time_of(out) == 1.0 and "T" in out.point_data.keys()
    t = lambda time: _flow_step(time).fields["T"].values.numpy()
    np.testing.assert_allclose(np.asarray(out.point_data["T_dt"]), (t(1.0) - t(0.5)) / 0.5,
                               rtol=1e-5, atol=1e-5)
    rate.UpdateTimeStep(1.5)
    assert source.asked == [0.5, 1.0, 1.5]
    rate.UpdateTimeStep(0.0)                       # no step before: no rate
    assert source.asked[3:] == [0.0]
    assert "T_dt" not in rate.GetOutputDataObject(0).point_data.keys()
    in_situ = _in_situ(lambda: ta.TimeDerivative(fields=["T"]))
    assert "T_dt" not in in_situ[0].point_data.keys()
    rate.UpdateTimeStep(2.0)
    _same_arrays(rate.GetOutputDataObject(0), in_situ[4], ["T_dt"])


def test_in_situ_a_step_is_taken_once_and_going_back_or_a_parameter_starts_over():
    stats = ta.RunningStatistics(fields=["T"])
    count = lambda: stats.GetOutputDataObject(0).GetFieldData().GetArray("count").GetValue(0)
    for t in (0.0, 0.5, 0.5):                     # the same step again: not taken
        stats.SetInputDataObject(0, dataset_to_vtk(_flow_step(t)))
        stats.Update()
    assert stats.error is None and count() == 2
    assert _time_of(stats.GetOutputDataObject(0)) == 0.5
    stats.SetInputDataObject(0, dataset_to_vtk(_flow_step(0.0)))
    stats.Update()
    assert count() == 1                            # back in time: started over
    stats.SetInputDataObject(0, dataset_to_vtk(_flow_step(0.5)))
    stats.Update()
    assert count() == 2
    stats.fields = [("T", "points")]
    stats.SetInputDataObject(0, dataset_to_vtk(_flow_step(1.0)))
    stats.Update()
    assert count() == 1                            # a parameter changed


def test_an_in_situ_adaptor_keeps_a_filter_from_driving():
    """With VTK's NO_PRIOR_TEMPORAL_ACCESS on its input a filter takes the step it is
    given, as does mode='in situ' anywhere; RESET starts it over."""
    s = _pipeline_keys()
    source = _steps_source(no_prior_access=s.NO_PRIOR_TEMPORAL_ACCESS_CONTINUE)
    stats = _driven(lambda: ta.RunningStatistics(fields=["T"]), source)
    for t in (0.0, 0.5, 1.0):
        stats.UpdateTimeStep(t)
    assert stats.error is None, stats.error
    assert source.asked == [0.0, 0.5, 1.0]
    out = stats.GetOutputDataObject(0)
    assert out.GetFieldData().GetArray("count").GetValue(0) == 3 and _time_of(out) == 1.0
    source.no_prior_access = s.NO_PRIOR_TEMPORAL_ACCESS_RESET
    source.Modified()
    stats.UpdateTimeStep(1.5)
    assert stats.GetOutputDataObject(0).GetFieldData().GetArray("count").GetValue(0) == 1
    plain = _steps_source()
    forced = _driven(lambda: ta.RunningStatistics(fields=["T"], mode="in situ"), plain)
    forced.UpdateTimeStep(1.0)
    assert plain.asked == [1.0]


def test_seeds_on_port_1_and_trees_refused():
    seeds = vtk.vtkPolyData()
    seeds.points = vtk.vtkPoints(data=np.array([[1.0, 5.0, 1.0]]))
    source = _steps_source()
    paths = ta.Pathlines(field="v")
    paths.SetInputConnection(0, source.GetOutputPort())
    paths.SetSourceData(seeds)
    paths.Update()
    assert paths.error is None, paths.error
    assert paths.GetOutputDataObject(0).number_of_cells == 1
    tree = vtk.vtkMultiBlockDataSet()
    tree.SetNumberOfBlocks(1)
    tree.SetBlock(0, dataset_to_vtk(_flow_step(0.0)))
    stats = ta.RunningStatistics(fields=["T"])
    stats.SetInputDataObject(0, tree)
    assert stats.Update() == 0 and isinstance(stats.error, TypeError)
