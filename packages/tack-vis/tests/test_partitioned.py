"""Partitioned datasets: the container, ``split``, the reductions that combine
pieces, and VTK's partitioned datasets and collections.

``split``'s pieces are checked against the grid they came from, point by point
and cell by cell. ``statistics`` and ``entropy`` over pieces must equal the
whole's for cell data, which no two pieces share, and count each piece's own
copy of a shared point for point data. VTK's ``vtkRedistributeDataSetFilter``,
run in this process, makes pieces of an unstructured mesh the same way.
"""

import types

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import env_flag

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

try:
    from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet
    from vtkmodules.vtkFiltersCore import vtkPointDataToCellData
    from vtkmodules.vtkFiltersParallelDIY2 import vtkRedistributeDataSetFilter
    from vtkmodules.vtkImagingCore import vtkRTAnalyticSource

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(vtkCompositeDataSet=vtkCompositeDataSet)

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")


def _host(values):
    return np.asarray(tack.arrays.to_host(values))


def _with_fields(data):
    """``data`` with a point field, a cell field, values on points and cells, an L2
    field, a face field and a cell and a point set, all numbered by id."""
    n, c = data.num_points, data.num_cells
    data.fields["p"] = td.Field(td.H1(data), tack.array(np.arange(n, dtype=np.float32)))
    data.fields["c"] = td.Field(td.Constant(data), tack.array(np.arange(c, dtype=np.int32)))
    data.fields["vp"] = td.Field(td.Values(data, "points"),
                                 tack.array(10 * np.arange(n, dtype=np.float32)))
    data.fields["vc"] = td.Field(td.Values(data, "cells"),
                                 tack.array(10 * np.arange(c, dtype=np.float32)))
    corners = td.L2(data)
    data.fields["l2"] = td.Field(corners, tack.array(np.arange(corners.size, dtype=np.float32)))
    faces = data.topology.faces().num_faces
    data.fields["f"] = td.Field(td.Values(data, "faces"),
                                tack.array(np.arange(faces, dtype=np.float32)))
    data.sets["even"] = td.Set(data, "cells", np.arange(0, c, 2))
    data.sets["first"] = td.Set(data, "points", np.arange(0, n, 7))
    return data


def _blocks(dims, counts):
    """Each piece's first point and point counts, x fastest, as ``split`` cuts."""
    cells = [d - 1 for d in dims]
    bounds = [[k * n // m for k in range(m + 1)] for n, m in zip(cells, counts)]
    out = []
    for index in np.ndindex(*reversed(counts)):
        index = tuple(reversed(index))
        out.append((tuple(b[i] for b, i in zip(bounds, index)),
                    tuple(b[i + 1] - b[i] + 1 for b, i in zip(bounds, index))))
    return out


def _check_pieces(whole, pieces, counts):
    """Every piece is its block of ``whole``: points, positions, every carried field
    and both sets."""
    dims = whole.topology.point_dims
    cell_dims = tuple(d - 1 for d in dims)
    point_ids = np.arange(whole.num_points).reshape(dims[::-1])
    cell_ids = np.arange(whole.num_cells).reshape(cell_dims[::-1])
    positions = whole.positions()
    corners = len(_host(whole.fields["l2"].values)) // whole.num_cells
    blocks = _blocks(dims, counts)
    assert len(pieces) == len(blocks)
    for piece, (start, points) in zip(pieces, blocks):
        assert piece.topology.point_dims == points
        slab = tuple(slice(s, s + n) for s, n in zip(start, points))[::-1]
        pids = point_ids[slab].reshape(-1)
        cslab = tuple(slice(s, s + n - 1) for s, n in zip(start, points))[::-1]
        cids = cell_ids[cslab].reshape(-1)
        np.testing.assert_allclose(piece.positions(), positions[pids], atol=1e-5)
        np.testing.assert_array_equal(_host(piece.fields["p"].values), pids)
        np.testing.assert_array_equal(_host(piece.fields["vp"].values), 10 * pids)
        np.testing.assert_array_equal(_host(piece.fields["c"].values), cids)
        np.testing.assert_array_equal(_host(piece.fields["vc"].values), 10 * cids)
        assert isinstance(piece.fields["c"].space, td.Constant)
        assert isinstance(piece.fields["vp"].space, td.Values)
        l2 = (cids[:, None] * corners + np.arange(corners)).reshape(-1)
        np.testing.assert_array_equal(_host(piece.fields["l2"].values), l2)
        assert "f" not in piece.fields                  # faces are derived afresh
        np.testing.assert_array_equal(piece.sets["even"].numpy(),
                                      np.flatnonzero(cids % 2 == 0))
        np.testing.assert_array_equal(piece.sets["first"].numpy(),
                                      np.flatnonzero(pids % 7 == 0))
    assert sum(piece.num_cells for piece in pieces) == whole.num_cells


# ── The container ───────────────────────────────────────────────────

def test_a_partitioned_dataset_is_a_list_of_datasets():
    a, b = td.uniform_grid((3, 3)), td.uniform_grid((4, 2))
    pds = td.PartitionedDataSet([a, b])
    assert len(pds) == 2 and pds[0] is a and pds[1] is b and list(pds) == [a, b]
    assert pds.partitions == (a, b)
    assert "2 partitions, 7 cells" in repr(pds)
    assert len(td.PartitionedDataSet()) == 0
    with pytest.raises(TypeError, match="partition 1 is a str"):
        td.PartitionedDataSet([a, "b"])


# ── split ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("counts", [(2, 1, 1), (2, 3, 2), (1, 1, 3), (5, 4, 3)])
def test_split_cuts_a_uniform_grid_into_its_blocks(counts):
    whole = _with_fields(td.uniform_grid((6, 5, 4), origin=(1.0, -2.0, 0.5),
                                         spacing=(0.5, 1.0, 2.0)))
    pieces = td.split(whole, counts)
    assert all(isinstance(p.geometry.values, tack.arrays.UniformCoordinates) for p in pieces)
    _check_pieces(whole, pieces, counts)


def test_split_keeps_a_direction_matrix():
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    whole = _with_fields(td.uniform_grid((5, 4, 3), origin=(1.0, 2.0, 3.0),
                                         spacing=(1.0, 2.0, 0.5), direction=turn))
    _check_pieces(whole, td.split(whole, (2, 2, 1)), (2, 2, 1))


def test_split_cuts_rectilinear_axes():
    whole = _with_fields(td.rectilinear_grid([0.0, 0.1, 0.5, 1.5, 4.0],
                                             [0.0, 1.0, 3.0], [-1.0, 0.0, 2.0, 7.0]))
    pieces = td.split(whole, (2, 2, 3))
    assert all(isinstance(p.geometry.values, tack.arrays.CartesianProduct) for p in pieces)
    _check_pieces(whole, pieces, (2, 2, 3))


def test_split_takes_explicit_points():
    grid = td.uniform_grid((4, 5, 3))
    positions = grid.positions() + np.sin(grid.positions())         # curvilinear
    whole = _with_fields(td.DataSet(td.StructuredTopology((4, 5, 3)), positions))
    _check_pieces(whole, td.split(whole, (3, 2, 1)), (3, 2, 1))


@pytest.mark.parametrize("dims", [(7, 5), (9,), (1, 6, 4)])
def test_split_cuts_grids_of_fewer_axes(dims):
    whole = td.uniform_grid(dims, origin=(0.5, 1.5, 2.5), spacing=(0.25, 0.5, 2.0))
    n = whole.num_points
    whole.fields["p"] = td.Field(td.H1(whole), tack.array(np.arange(n, dtype=np.float32)))
    pieces = td.split(whole, 2)
    topo = whole.topology.point_dims
    point_ids = np.arange(n).reshape(topo[::-1])
    positions = whole.positions()
    counts = [1] * len(topo)
    counts[int(np.argmax(topo))] = 2
    for piece, (start, points) in zip(pieces, _blocks(topo, counts)):
        slab = tuple(slice(s, s + m) for s, m in zip(start, points))[::-1]
        pids = point_ids[slab].reshape(-1)
        np.testing.assert_array_equal(_host(piece.fields["p"].values), pids)
        np.testing.assert_allclose(piece.positions(), positions[pids], atol=1e-6)


def test_split_factors_a_count_over_the_axes():
    """Each prime factor, largest first, cuts the axis whose pieces have most cells."""
    grid = td.uniform_grid((13, 7, 4))                   # 12 x 6 x 3 cells
    assert [p.topology.point_dims for p in td.split(grid, 4)] == [(4, 7, 4)] * 4
    assert sorted({p.topology.point_dims for p in td.split(grid, 6)}) == [(5, 4, 4)]
    assert len(td.split(grid, 1)) == 1
    assert len(td.split(grid, 9)) == 9


def test_split_refuses_what_it_cannot_cut():
    grid = td.uniform_grid((3, 3, 3))                    # 2 x 2 x 2 cells
    with pytest.raises(ValueError, match="cannot cut"):
        td.split(grid, 16)
    with pytest.raises(ValueError, match="cannot cut"):
        td.split(grid, (3, 1, 1))
    with pytest.raises(ValueError, match="takes 3 counts"):
        td.split(grid, (2, 2))
    with pytest.raises(ValueError, match="at least 1 piece"):
        td.split(grid, 0)
    with pytest.raises(TypeError, match="structured grids"):
        td.split(td.tetrahedralize(grid), 2)


# ── Reductions over pieces ──────────────────────────────────────────

def test_statistics_and_entropy_of_cell_data_are_the_wholes():
    whole = td.wavelet(8)
    whole.fields["cells"] = td.project(whole, "RTData", td.Constant(whole))
    pieces = td.split(whole, (3, 2, 2))
    expected = td.statistics(whole, "cells")
    combined = td.statistics(pieces, "cells")
    assert combined.n == expected.n
    for name in ("min", "max"):
        assert getattr(combined, name) == getattr(expected, name)
    for name in ("sum", "mean", "m2", "m3", "m4", "sample_variance", "skewness",
                 "kurtosis"):
        assert getattr(combined, name) == pytest.approx(getattr(expected, name), rel=1e-5)
    assert td.entropy(pieces, "cells", bins=17) == pytest.approx(
        td.entropy(whole, "cells", bins=17), rel=1e-12)


def test_point_data_counts_each_pieces_copy_of_a_shared_point():
    whole = td.wavelet(5)
    pieces = td.split(whole, (2, 2, 1))
    values = np.concatenate([_host(p.fields["RTData"].values) for p in pieces]).astype(float)
    combined = td.statistics(pieces, "RTData")
    assert combined.n == len(values) > whole.num_points
    assert combined.mean == pytest.approx(values.mean(), rel=1e-6)
    assert combined.population_variance == pytest.approx(values.var(), rel=1e-5)
    assert combined.min == values.min() and combined.max == values.max()


def test_reductions_skip_pieces_without_values_and_name_what_is_wrong():
    a = td.uniform_grid((4, 3))
    a.fields["c"] = td.Field(td.Constant(a), tack.array(np.arange(6, dtype=np.float32)))
    empty = td.uniform_grid((0,))
    empty.fields["c"] = td.Field(td.Constant(empty), tack.empty((0,)))
    pds = td.PartitionedDataSet([empty, a, empty])
    assert td.statistics(pds, "c") == td.statistics(a, "c")
    assert td.entropy(pds, "c") == td.entropy(a, "c")
    with pytest.raises(ValueError, match="no values"):
        td.statistics(td.PartitionedDataSet([empty]), "c")
    with pytest.raises(ValueError, match="no values"):
        td.entropy(td.PartitionedDataSet([empty]), "c")
    with pytest.raises(KeyError, match="partition 1 has no field 'c'"):
        td.statistics(td.PartitionedDataSet([a, td.uniform_grid((3, 3))]), "c")
    with pytest.raises(TypeError, match="field's name"):
        td.statistics(pds, a.fields["c"])


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's, as ``test_dataset_devices.py`` makes."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


def test_pieces_on_two_devices_are_reduced_where_each_is():
    second = dispatch.as_device(_SecondCPU())
    whole = td.wavelet(6)
    whole.fields["cells"] = td.project(whole, "RTData", td.Constant(whole))
    first, other = td.split(whole, 2)
    pds = td.PartitionedDataSet([first, other.to_device(second)])
    assert pds[1].device == second
    expected = td.statistics(whole, "cells")
    assert td.statistics(pds, "cells").mean == pytest.approx(expected.mean, rel=1e-6)
    assert td.entropy(pds, "cells") == pytest.approx(td.entropy(whole, "cells"))


# ── VTK ─────────────────────────────────────────────────────────────

@needs_vtk
def test_split_pieces_go_to_vtk_and_back():
    pieces = td.split(td.wavelet(6), (2, 1, 2))
    out = dataset_to_vtk(pieces)
    assert out.GetClassName() == "vtkPartitionedDataSet"
    assert out.GetNumberOfPartitions() == 4
    assert out.GetPartition(1).GetClassName() == "vtkImageData"
    assert out.GetPartition(1).GetBounds() == (0.0, 6.0, -6.0, 6.0, -6.0, 0.0)
    back = vtk_to_dataset(out)
    assert isinstance(back, td.PartitionedDataSet) and len(back) == 4
    for a, b in zip(pieces, back):
        np.testing.assert_allclose(a.positions(), b.positions())
        np.testing.assert_array_equal(_host(a.fields["RTData"].values),
                                      _host(b.fields["RTData"].values))


@needs_vtk
def test_collections_are_dicts_of_partitioned_datasets():
    pieces = td.split(td.wavelet(3), 2)
    one = td.PartitionedDataSet([td.uniform_grid((3, 3, 3))])
    out = dataset_to_vtk({"fluid": pieces, "solid": one})
    assert out.GetClassName() == "vtkPartitionedDataSetCollection"
    name = vtk.vtkCompositeDataSet.NAME()
    assert [out.GetMetaData(i).Get(name) for i in range(2)] == ["fluid", "solid"]
    back = vtk_to_dataset(out)
    assert list(back) == ["fluid", "solid"]
    assert [len(back["fluid"]), len(back["solid"])] == [2, 1]
    # A name taken twice, and none at all.
    out.GetMetaData(1).Set(name, "fluid")
    out.SetNumberOfPartitionedDataSets(3)
    assert list(vtk_to_dataset(out)) == ["fluid", "fluid#1", "block2"]
    assert len(vtk_to_dataset(out)["block2"]) == 0
    with pytest.raises(TypeError, match="not a PartitionedDataSet"):
        dataset_to_vtk({"x": td.uniform_grid((2, 2))})


@needs_vtk
def test_vtks_redistributed_pieces_reduce_to_the_wholes_statistics():
    """``vtkRedistributeDataSetFilter`` assigns each cell to one piece, so cell
    statistics over its pieces are the whole's."""
    source = vtkRTAnalyticSource()
    source.SetWholeExtent(-6, 6, -6, 6, -6, 6)
    cells = vtkPointDataToCellData()
    cells.SetInputConnection(source.GetOutputPort())
    redistribute = vtkRedistributeDataSetFilter()
    redistribute.SetInputConnection(cells.GetOutputPort())
    redistribute.SetNumberOfPartitions(4)
    redistribute.PreservePartitionsInOutputOn()
    redistribute.Update()
    cells.Update()
    pieces = vtk_to_dataset(redistribute.GetOutputDataObject(0))
    whole = vtk_to_dataset(cells.GetOutputDataObject(0))
    assert len(pieces) == 4
    assert sum(p.num_cells for p in pieces) == whole.num_cells
    expected, combined = td.statistics(whole, "RTData"), td.statistics(pieces, "RTData")
    assert combined.n == expected.n
    assert combined.mean == pytest.approx(expected.mean, rel=1e-6)
    assert combined.m2 == pytest.approx(expected.m2, rel=1e-5)
    assert combined.kurtosis == pytest.approx(expected.kurtosis, rel=1e-5)
    assert td.entropy(pieces, "RTData") == pytest.approx(td.entropy(whole, "RTData"))
