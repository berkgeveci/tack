"""Partitioned datasets: the container, ghost marks, ``split``, the reductions
that combine pieces, and VTK's partitioned datasets and collections.

``split``'s pieces are checked against the grid they came from, point by point
and cell by cell, and every point they share must have one owner.
``statistics`` and ``entropy`` count what a piece owns, so over pieces they
must equal the whole's. VTK's ``vtkRedistributeDataSetFilter`` and
``vtkGhostCellsGenerator``, run in this process, make pieces of an
unstructured mesh, with ghost cells, the same way.
"""

import types

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import Ghosts
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import env_flag

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

try:
    from vtkmodules.vtkCommonCore import vtkFloatArray
    from vtkmodules.vtkCommonDataModel import (
        vtkCompositeDataSet,
        vtkImageData,
        vtkPartitionedDataSet,
        vtkPartitionedDataSetCollection,
    )
    from vtkmodules.vtkFiltersCore import vtkPointDataToCellData
    from vtkmodules.vtkImagingCore import vtkRTAnalyticSource

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(vtkCompositeDataSet=vtkCompositeDataSet)

# GetPartition, GetMetaData and the NAME key stay classic: 9.7.1 has no
# pythonic form for them (a vtkPartitionedDataSet iterates, but walks leaves).

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

# The DIY2 filters are an optional VTK module (``FiltersParallelDIY2``), which a
# build without MPI may leave out: headroom's VTK master build has none. A VTK
# without it still runs every other test here, so the three that redistribute
# and generate ghosts through VTK skip on their own.
try:
    from vtkmodules.vtkFiltersParallelDIY2 import (
        vtkGhostCellsGenerator,
        vtkRedistributeDataSetFilter,
    )
except ImportError:
    vtkGhostCellsGenerator = vtkRedistributeDataSetFilter = None

needs_diy2 = pytest.mark.skipif(vtkRedistributeDataSetFilter is None,
                                reason="needs VTK's FiltersParallelDIY2 module")


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


@pytest.mark.parametrize("counts", [(2, 1, 1), (3, 2, 2), (2, 2, 2)])
def test_split_gives_every_shared_point_one_owner(counts):
    whole = td.uniform_grid((6, 5, 4))
    n = whole.num_points
    whole.fields["p"] = td.Field(td.H1(whole), tack.array(np.arange(n, dtype=np.int32)))
    owned = []
    for piece in td.split(whole, counts):
        marks = piece.ghosts.points.numpy()
        assert set(np.unique(marks)) <= {0, Ghosts.DUPLICATEPOINT}
        assert piece.ghosts.cells is None
        owned.extend(_host(piece.fields["p"].values)[marks == 0])
    np.testing.assert_array_equal(np.sort(owned), np.arange(n))


def test_split_carries_the_inputs_ghost_marks():
    whole = td.uniform_grid((5, 4, 3))
    cells = np.arange(whole.num_cells) % 3 == 0
    points = np.arange(whole.num_points) % 5 == 0
    whole.ghosts = Ghosts(cells=np.where(cells, Ghosts.HIDDENCELL, 0),
                          points=np.where(points, Ghosts.HIDDENPOINT, 0))
    whole.fields["c"] = td.Field(td.Constant(whole), tack.array(np.arange(whole.num_cells,
                                                                          dtype=np.int32)))
    whole.fields["p"] = td.Field(td.H1(whole), tack.array(np.arange(whole.num_points,
                                                                     dtype=np.int32)))
    for piece in td.split(whole, (2, 1, 1)):
        cids, pids = _host(piece.fields["c"].values), _host(piece.fields["p"].values)
        np.testing.assert_array_equal(piece.ghosts.cells.numpy(),
                                      np.where(cells[cids], Ghosts.HIDDENCELL, 0))
        assert ((piece.ghosts.points.numpy() & Ghosts.HIDDENPOINT) != 0).tolist() == \
            points[pids].tolist()


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


def test_point_data_counts_each_shared_point_once():
    """A point two pieces share is owned by one and a ghost in the other."""
    whole = td.wavelet(5)
    pieces = td.split(whole, (2, 2, 1))
    expected, combined = td.statistics(whole, "RTData"), td.statistics(pieces, "RTData")
    assert combined.n == expected.n == whole.num_points
    assert combined.mean == pytest.approx(expected.mean, rel=1e-6)
    assert combined.m2 == pytest.approx(expected.m2, rel=1e-5)
    assert combined.min == expected.min and combined.max == expected.max
    assert td.entropy(pieces, "RTData", bins=13) == pytest.approx(
        td.entropy(whole, "RTData", bins=13), rel=1e-12)


def test_reductions_leave_out_ghosts():
    """Duplicates and hidden cells and points are left out, on one dataset too; an
    L2 field goes by its cells' marks."""
    data = td.uniform_grid((5, 4))                       # 12 cells, 20 points
    c, n = data.num_cells, data.num_points
    data.fields["c"] = td.Field(td.Constant(data), tack.array(np.arange(c, dtype=np.float32)))
    data.fields["p"] = td.Field(td.H1(data), tack.array(np.arange(n, dtype=np.float32)))
    corners = td.L2(data)
    data.fields["l2"] = td.Field(corners, tack.array(np.repeat(np.arange(c), 4).astype(np.float32)))
    cell_marks = np.zeros(c, np.uint8)
    cell_marks[[0, 5]] = Ghosts.DUPLICATECELL
    cell_marks[7] = Ghosts.HIDDENCELL
    cell_marks[9] = Ghosts.EXTERIORCELL                  # carried, not read
    point_marks = np.zeros(n, np.uint8)
    point_marks[[1, 2]] = Ghosts.DUPLICATEPOINT
    point_marks[19] = Ghosts.HIDDENPOINT
    data.ghosts = Ghosts(cells=cell_marks, points=point_marks)
    kept_cells = np.setdiff1d(np.arange(c), [0, 5, 7]).astype(float)
    kept_points = np.setdiff1d(np.arange(n), [1, 2, 19]).astype(float)
    s = td.statistics(data, "c")
    assert (s.n, s.sum, s.min, s.max) == (9, kept_cells.sum(), 1.0, 11.0)
    assert td.statistics(data, "p").sum == kept_points.sum()
    assert td.statistics(data, "l2").sum == 4 * kept_cells.sum()
    assert td.entropy(data, "c", bins=4) == pytest.approx(
        _entropy(kept_cells, bins=4, value_range=(1.0, 11.0)))
    data.fields["f"] = td.Field(td.Values(data, "faces"),
                                tack.array(np.zeros(data.topology.faces().num_faces,
                                                    np.float32)))
    with pytest.raises(NotImplementedError, match="ghost marks"):
        td.statistics(data, "f")


def _entropy(values, bins, value_range):
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    p = counts[counts > 0] / counts.sum()
    return -float((p * np.log2(p)).sum())


def test_ghost_marks_are_checked_when_they_are_put_in():
    data = td.uniform_grid((3, 3))
    with pytest.raises(ValueError, match="5 ghost marks for 4 cells"):
        data.ghosts = Ghosts(cells=np.zeros(5))
    with pytest.raises(ValueError, match="ghost marks for 9 points"):
        td.DataSet(data.topology, data.geometry.values, ghosts=Ghosts(points=np.zeros(3)))
    with pytest.raises(TypeError, match="u8 array"):
        Ghosts(cells=tack.zeros((4,), tack.i32))
    with pytest.raises(TypeError, match="Ghosts, not ndarray"):
        data.ghosts = np.zeros(4)
    data.ghosts = Ghosts(cells=[0, 1, 0, 32])
    assert data.ghosts.cells.numpy().tolist() == [0, 1, 0, 32] and data.ghosts.points is None
    data.ghosts = None


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
    assert pds[1].device == second and pds[1].ghosts.device == second
    np.testing.assert_array_equal(pds[1].ghosts.points.numpy(), other.ghosts.points.numpy())
    expected = td.statistics(whole, "cells")
    assert td.statistics(pds, "cells").mean == pytest.approx(expected.mean, rel=1e-6)
    assert td.entropy(pds, "cells") == pytest.approx(td.entropy(whole, "cells"))


# ── Filters and ghost marks ─────────────────────────────────────────

def _owned(marks, n, skipped):
    return n if marks is None else int(((marks.numpy() & skipped) == 0).sum())


def _owned_counts(data):
    """The cells and points ``data`` owns."""
    g = data.ghosts
    return (_owned(None if g is None else g.cells, data.num_cells, Ghosts.SKIPPED_CELLS),
            _owned(None if g is None else g.points, data.num_points, Ghosts.SKIPPED_POINTS))


_FILTERS = {
    "contour": lambda d: td.contour(d, "RTData", 160.0),
    "clip": lambda d: td.clip(d, "RTData", 160.0),
    "slice": lambda d: td.slice_plane(d, (0.3, 0.2, 0.1), (1.0, 2.0, 3.0)),
    "threshold": lambda d: td.threshold(d, "RTData", 150.0, 200.0),
    "extract_surface": td.extract_surface,
    "extract_edges": td.extract_edges,
    "tetrahedralize": td.tetrahedralize,
    "threshold_points": lambda d: td.threshold_points(d, "RTData", 150.0, 200.0),
    "clip, contour": lambda d: td.contour(td.clip(d, "RTData", 140.0), "RTData", 180.0),
    "tetrahedralize, surface": lambda d: td.extract_surface(td.tetrahedralize(d)),
    "as_polyhedra, contour": lambda d: td.contour(td.as_polyhedra(d), "RTData", 160.0),
    "as_polyhedra, surface": lambda d: td.extract_surface(td.as_polyhedra(d)),
}


def _check_shares(whole, pieces, filter):
    """What ``filter`` makes of the pieces, the owned part of each, is what it makes
    of the whole: as many cells and points, and the same statistics."""
    expected = filter(whole)
    outputs = [filter(piece) for piece in pieces]
    owned = np.sum([_owned_counts(out) for out in outputs], axis=0)
    assert tuple(int(v) for v in owned) == (expected.num_cells, expected.num_points)
    if "RTData" in expected.fields:
        a, b = td.statistics(expected, "RTData"), td.statistics(td.PartitionedDataSet(outputs),
                                                                "RTData")
        assert b.n == a.n
        assert b.mean == pytest.approx(a.mean, rel=1e-5)
        assert b.min == pytest.approx(a.min, rel=1e-6) and b.max == pytest.approx(a.max, rel=1e-6)
    return outputs


@pytest.mark.parametrize("name", list(_FILTERS))
def test_filters_give_each_piece_its_share_of_the_wholes_output(name):
    """Each piece marks what it makes from its ghost layers as another's: an output
    cell takes its source cell's owner, a point on an edge the edge's, a face
    or an edge shared by cells its smallest point's, a point inside a cell the
    cell's. So the owned output of the pieces is the whole's output, cell for
    cell and point for point, with one and with two ghost layers, through
    chains of filters and on polyhedra."""
    whole = td.wavelet(8)
    for layers in (1, 2):
        _check_shares(whole, td.split(whole, (2, 2, 2), ghost_layers=layers), _FILTERS[name])


def test_two_dimensional_pieces_give_their_shares():
    grid = td.uniform_grid((21, 17), spacing=(0.1, 0.1, 1.0))
    x = grid.positions()
    grid.fields["RTData"] = td.Field(td.H1(grid), tack.array(
        (np.sin(3 * x[:, 0]) + np.cos(4 * x[:, 1])).astype(np.float32)))
    pieces = td.split(grid, (3, 2), ghost_layers=1)
    for filter in (lambda d: td.contour(d, "RTData", 0.3), lambda d: td.clip(d, "RTData", 0.3),
                   td.extract_surface, td.extract_edges, td.triangulate,
                   lambda d: td.contour(td.as_polygons(d), "RTData", 0.3),
                   lambda d: td.extract_surface(td.as_polygons(d))):
        _check_shares(grid, pieces, filter)


def test_pieces_without_ghost_cells_refuse_what_their_neighbours_decide():
    """Without ghost cells a piece cannot tell what its neighbours make or keep from
    the cells, faces and edges they share: a point on a shared edge, a shared
    face, a shared point that only a neighbour's kept cells use. Such outputs
    record that they lost the marks, and reductions refuse them -- over pieces
    too. Points chosen by their own values, and outputs on the same cells and
    points, keep the marks."""
    whole = td.wavelet(4)
    pieces = td.split(whole, 2)                        # no ghost layers
    for filter in (_FILTERS["contour"], _FILTERS["threshold"], td.extract_surface):
        out = [filter(piece) for piece in pieces]
        with pytest.raises(NotImplementedError, match="no ghost cells"):
            td.statistics(td.PartitionedDataSet(out), "RTData")
        with pytest.raises(NotImplementedError, match="no ghost cells"):
            td.entropy(out[1], "RTData")
    points = [td.threshold_points(piece, "RTData", 0.0, 1e9) for piece in pieces]
    assert td.statistics(td.PartitionedDataSet(points), "RTData").n == whole.num_points


def test_filters_that_keep_the_cells_keep_the_marks():
    """warp, transform, shrink, as_polyhedra and a probe's target keep their input's
    cells and points, so they keep its marks, and reduce as it does."""
    whole = td.wavelet(4)
    piece = td.split(whole, 2)[1]
    marks = piece.ghosts.points.numpy()
    moved = td.transform(piece, np.eye(4))
    shrunk = td.shrink(piece, 0.9)
    poly = td.as_polyhedra(piece)
    probed = td.probe(whole, piece)
    for out in (moved, shrunk, poly, probed):
        assert out.ghosts is not None
        np.testing.assert_array_equal(out.ghosts.points.numpy(), marks)
    expected = td.statistics(piece, "RTData")
    for out in (moved, poly, probed):              # shrink's point data is per corner (L2)
        assert td.statistics(out, "RTData").n == expected.n


def test_a_lost_record_is_kept_until_marks_are_set():
    """Filtering an output that lost the marks keeps the record; so does moving it;
    setting marks clears it. Data without marks loses nothing."""
    whole = td.wavelet(4)
    cut = td.threshold(td.split(whole, 2)[1], "RTData", 0.0, 1e9)
    again = td.tetrahedralize(cut)
    with pytest.raises(NotImplementedError, match="no ghost cells"):
        td.statistics(again, "RTData")
    moved = again.to_device(dispatch.as_device(_SecondCPU()))
    with pytest.raises(NotImplementedError, match="no ghost cells"):
        td.statistics(moved, "RTData")
    again.ghosts = Ghosts(points=np.zeros(again.num_points, np.uint8))
    assert td.statistics(again, "RTData").n == again.num_points
    assert td.statistics(td.threshold(whole, "RTData", 0.0, 1e9), "RTData").n > 0


@pytest.mark.parametrize("counts", [(2, 1, 1), (3, 2, 2)])
@pytest.mark.parametrize("layers", [1, 2])
def test_split_adds_ghost_layers(counts, layers):
    """A piece holds its own block and ``layers`` of its neighbours' cells around it,
    as far as the grid goes, marked duplicates; its own cells, and its own
    points, are each in exactly one piece; and the fields are the whole's."""
    whole = td.uniform_grid((7, 6, 5))
    whole.fields["p"] = td.Field(td.H1(whole), tack.array(np.arange(whole.num_points,
                                                                     dtype=np.int32)))
    whole.fields["c"] = td.Field(td.Constant(whole), tack.array(np.arange(whole.num_cells,
                                                                          dtype=np.int32)))
    cells = [d - 1 for d in whole.topology.point_dims]
    owned_cells, owned_points = [], []
    pieces = td.split(whole, counts, ghost_layers=layers)
    plain = td.split(whole, counts)
    for piece, own in zip(pieces, plain):
        dims = piece.topology.point_dims
        assert all(a - b <= 2 * layers for a, b in zip(dims, own.topology.point_dims))
        assert all(d - 1 <= n for d, n in zip(dims, cells))
        c, p = piece.ghosts.cells.numpy(), piece.ghosts.points.numpy()
        assert set(np.unique(c)) <= {0, Ghosts.DUPLICATECELL}
        cids, pids = _host(piece.fields["c"].values), _host(piece.fields["p"].values)
        np.testing.assert_array_equal(np.sort(cids[c == 0]),
                                      np.sort(_host(own.fields["c"].values)))
        owned_cells.extend(cids[c == 0])
        owned_points.extend(pids[p == 0])
        np.testing.assert_allclose(piece.positions(), whole.positions()[pids], atol=1e-6)
    np.testing.assert_array_equal(np.sort(owned_cells), np.arange(whole.num_cells))
    np.testing.assert_array_equal(np.sort(owned_points), np.arange(whole.num_points))
    with pytest.raises(ValueError, match="count of layers"):
        td.split(whole, 2, ghost_layers=-1)


def test_remove_ghosts_keeps_what_a_piece_owns():
    """remove_ghosts takes out the duplicate and hidden cells: a structured piece
    stays a structured grid of its own block, the pieces' cells are the
    whole's, and the points kept keep their owners -- a point a piece owns
    stays though only ghost triangles used it -- so reductions are unchanged."""
    whole = td.wavelet(6)
    pieces = td.split(whole, (2, 2, 1), ghost_layers=1)
    kept = [td.remove_ghosts(piece) for piece in pieces]
    for k, own in zip(kept, td.split(whole, (2, 2, 1))):
        assert isinstance(k.topology, td.StructuredTopology)
        assert k.topology.point_dims == own.topology.point_dims
        np.testing.assert_allclose(k.positions(), own.positions(), atol=1e-6)
    assert td.statistics(td.PartitionedDataSet(kept), "RTData").n == whole.num_points
    assert td.remove_ghosts(kept[0]) is kept[0]                # nothing left to take out
    iso = td.contour(whole, "RTData", 160.0)
    surfaces = [td.remove_ghosts(td.contour(piece, "RTData", 160.0)) for piece in pieces]
    assert sum(s.num_cells for s in surfaces) == iso.num_cells
    assert td.statistics(td.PartitionedDataSet(surfaces), "RTData").n == iso.num_points
    for s in surfaces:
        assert not (s.ghosts.cells.numpy() & Ghosts.SKIPPED_CELLS).any()
    with pytest.raises(ValueError, match="lost"):
        td.remove_ghosts(td.contour(td.split(whole, 2)[1], "RTData", 160.0))
    assert td.remove_ghosts(whole) is whole                     # no marks


def test_hidden_cells_are_never_processed():
    """Filters leave hidden cells out, as if they were not there: their output is
    the output of the other cells, a field passed as an object included.
    Choosing faces or edges by id, or reusing a locator, on such data is
    refused rather than done with the hidden cells."""
    whole = td.wavelet(6)
    marks = np.zeros(whole.num_cells, np.uint8)
    marks[::3] = Ghosts.HIDDENCELL
    whole.ghosts = Ghosts(cells=marks)
    rest = td.extract_cells(td.wavelet(6), np.flatnonzero(marks == 0))
    for filter in (_FILTERS["contour"], _FILTERS["clip"], _FILTERS["threshold"],
                   td.extract_surface, td.extract_edges, td.tetrahedralize,
                   lambda d: td.contour(d, d.fields["RTData"], 160.0),
                   lambda d: td.extract_cells(d, np.arange(d.num_cells))):
        a, b = filter(whole), filter(rest)
        assert (a.num_cells, a.num_points) == (b.num_cells, b.num_points)
    seeds = np.array([[0.5, 0.5, 0.5]], np.float32)
    probed = td.probe(whole, seeds)
    assert probed.num_points == 1
    with pytest.raises(NotImplementedError, match="hidden or refined cells"):
        td.extract_surface(whole, np.arange(4))
    with pytest.raises(NotImplementedError, match="hidden or refined cells"):
        td.probe(whole, seeds, locator=td.CellLocator(whole))


@needs_vtk
@needs_diy2
def test_vtks_ghost_cells_give_each_piece_its_share():
    """The same over VTK's pieces of an unstructured mesh, with ghost cells from
    vtkGhostCellsGenerator."""
    image = vtkRTAnalyticSource(whole_extent=(-8, 8, -8, 8, -8, 8))()
    layered = (vtkRedistributeDataSetFilter(number_of_partitions=4,
                                            preserve_partitions_in_output=True)
               >> vtkGhostCellsGenerator(number_of_ghost_layers=1, build_if_required=False))
    pieces = vtk_to_dataset(layered(image))
    whole = vtk_to_dataset(image)
    assert all(isinstance(p.topology, td.UnstructuredTopology) for p in pieces)
    for name in ("contour", "clip", "threshold", "extract_surface", "extract_edges",
                 "clip, contour", "as_polyhedra, surface"):
        outputs = _check_shares(whole, pieces, _FILTERS[name])
        kept = [td.remove_ghosts(out) for out in outputs]
        assert sum(_owned_counts(k)[0] for k in kept) == sum(k.num_cells for k in kept)


# ── VTK ─────────────────────────────────────────────────────────────

@needs_vtk
def test_split_pieces_go_to_vtk_and_back():
    pieces = td.split(td.wavelet(6), (2, 1, 2))
    out = dataset_to_vtk(pieces)
    assert isinstance(out, vtkPartitionedDataSet)
    assert out.number_of_partitions == 4
    assert isinstance(out.GetPartition(1), vtkImageData)
    assert out.GetPartition(1).bounds == (0.0, 6.0, -6.0, 6.0, -6.0, 0.0)
    back = vtk_to_dataset(out)
    assert isinstance(back, td.PartitionedDataSet) and len(back) == 4
    for a, b in zip(pieces, back):
        np.testing.assert_allclose(a.positions(), b.positions())
        np.testing.assert_array_equal(_host(a.fields["RTData"].values),
                                      _host(b.fields["RTData"].values))


@needs_vtk
def test_ghost_marks_go_to_vtk_and_back():
    pieces = td.split(td.wavelet(4), 2)
    pieces[0].ghosts = Ghosts(cells=np.arange(pieces[0].num_cells) % 2,
                              points=pieces[0].ghosts.points)
    out = dataset_to_vtk(pieces)
    first = out.GetPartition(0)
    assert np.asarray(first.cell_data["vtkGhostType"]).dtype == np.uint8
    assert "vtkGhostType" in first.point_data.keys()
    back = vtk_to_dataset(out)
    for a, b in zip(pieces, back):
        assert "vtkGhostType" not in b.fields
        np.testing.assert_array_equal(a.ghosts.points.numpy(), b.ghosts.points.numpy())
    np.testing.assert_array_equal(back[0].ghosts.cells.numpy(), pieces[0].ghosts.cells.numpy())
    assert back[1].ghosts.cells is None


@needs_vtk
def test_collections_are_dicts_of_partitioned_datasets():
    pieces = td.split(td.wavelet(3), 2)
    one = td.PartitionedDataSet([td.uniform_grid((3, 3, 3))])
    out = dataset_to_vtk({"fluid": pieces, "solid": one})
    assert isinstance(out, vtkPartitionedDataSetCollection)
    name = vtk.vtkCompositeDataSet.NAME()
    assert [out.GetMetaData(i).Get(name) for i in range(2)] == ["fluid", "solid"]
    back = vtk_to_dataset(out)
    assert list(back) == ["fluid", "solid"]
    assert [len(back["fluid"]), len(back["solid"])] == [2, 1]
    # A name taken twice, and none at all.
    out.GetMetaData(1).Set(name, "fluid")
    out.number_of_partitioned_data_sets = 3
    assert list(vtk_to_dataset(out)) == ["fluid", "fluid#1", "block2"]
    assert len(vtk_to_dataset(out)["block2"]) == 0
    with pytest.raises(TypeError, match="not a PartitionedDataSet"):
        dataset_to_vtk({"x": td.uniform_grid((2, 2))})


@needs_vtk
@needs_diy2
def test_vtks_redistributed_pieces_reduce_to_the_wholes_statistics():
    """``vtkRedistributeDataSetFilter`` assigns each cell to one piece, so cell
    statistics over its pieces are the whole's."""
    image = (vtkRTAnalyticSource(whole_extent=(-6, 6, -6, 6, -6, 6))
             >> vtkPointDataToCellData())()
    redistribute = vtkRedistributeDataSetFilter(number_of_partitions=4,
                                                preserve_partitions_in_output=True)
    pieces = vtk_to_dataset(redistribute(image))
    whole = vtk_to_dataset(image)
    assert len(pieces) == 4
    assert sum(p.num_cells for p in pieces) == whole.num_cells
    expected, combined = td.statistics(whole, "RTData"), td.statistics(pieces, "RTData")
    assert combined.n == expected.n
    assert combined.mean == pytest.approx(expected.mean, rel=1e-6)
    assert combined.m2 == pytest.approx(expected.m2, rel=1e-5)
    assert combined.kurtosis == pytest.approx(expected.kurtosis, rel=1e-5)
    assert td.entropy(pieces, "RTData") == pytest.approx(td.entropy(whole, "RTData"))


@needs_vtk
@needs_diy2
def test_vtks_ghost_cells_give_the_wholes_statistics():
    """``vtkGhostCellsGenerator`` gives every cell and point of its pieces one owner,
    so point and cell statistics over pieces with ghost layers are the whole's."""
    image = vtkRTAnalyticSource(whole_extent=(-6, 6, -6, 6, -6, 6))()
    n = image.number_of_cells
    image.cell_data["C"] = vtkFloatArray((1.5 + np.sin(np.arange(n) * 0.37)).astype(np.float32))
    layered = (vtkRedistributeDataSetFilter(number_of_partitions=4,
                                            preserve_partitions_in_output=True)
               >> vtkGhostCellsGenerator(number_of_ghost_layers=1, build_if_required=False))
    pieces = vtk_to_dataset(layered(image))
    whole = vtk_to_dataset(image)
    assert all(p.ghosts is not None and p.ghosts.cells is not None for p in pieces)
    assert sum(p.num_cells for p in pieces) > whole.num_cells        # ghost layers
    for name, count in (("RTData", whole.num_points), ("C", whole.num_cells)):
        expected, combined = td.statistics(whole, name), td.statistics(pieces, name)
        assert combined.n == expected.n == count
        assert combined.mean == pytest.approx(expected.mean, rel=1e-6)
        assert combined.m2 == pytest.approx(expected.m2, rel=1e-5)
        assert td.entropy(pieces, name) == pytest.approx(td.entropy(whole, name))
