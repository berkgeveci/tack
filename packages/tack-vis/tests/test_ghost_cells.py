"""Ghost cells generated for unstructured pieces, and refreshing their values.

``ghost_cells`` is checked entity by entity, by position, against the ghost
layers ``split`` makes arithmetically, and against VTK's
``vtkGhostCellsGenerator`` on ``vtkRedistributeDataSetFilter``'s pieces: the
same cells, the same points, the same marks, the same values. Then filters
over the generated pieces give each piece its share of the whole's output,
``update_ghosts`` refreshes values a filter left wrong on the ghost rim, and
the same runs across MPI ranks (``_ghost_cells_mpi.py``) give what one
process gives.
"""

import os
import types

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack import communication as tc
from tack.dataset import Ghosts
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkImagingCore import vtkRTAnalyticSource

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace()

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

# The DIY2 filters are an optional VTK module (``FiltersParallelDIY2``), which a
# build without MPI may leave out (headroom's VTK master build); the tests
# comparing with them skip on their own, as in ``test_partitioned.py``.
try:
    from vtkmodules.vtkFiltersParallelDIY2 import (
        vtkGhostCellsGenerator,
        vtkRedistributeDataSetFilter,
    )
except ImportError:
    vtkGhostCellsGenerator = vtkRedistributeDataSetFilter = None

needs_diy2 = pytest.mark.skipif(vtkRedistributeDataSetFilter is None,
                                reason="needs VTK's FiltersParallelDIY2 module")

pytestmark = pytest.mark.quick


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's: its arrays are not the default's."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _with_fields(data):
    data.fields["c"] = td.project(data, "RTData", td.Constant(data))
    data.fields["ids"] = td.Field(td.Values(data, "cells"),
                                  tack.array(np.arange(data.num_cells, dtype=np.int32)))
    data.fields["v"] = td.Field(td.H1(data), tack.array(
        np.asarray(data.positions(), np.float32), tack.vec3()))
    return data


def _unstructured(pieces):
    """``split``'s pieces as unstructured meshes, as a reader or VTK would give them."""
    return td.PartitionedDataSet([td.extract_cells(p, np.arange(p.num_cells)) for p in pieces])


def _key(position):
    return tuple(np.round(np.asarray(position, float), 5))


def _on_cells(field):
    space = field.space
    return isinstance(space, td.Constant) or (isinstance(space, td.Values) and space.on == "cells")


def described(piece):
    """A piece by position, numbering aside: each cell (its sorted points), its
    marks and cell values; each point, its marks and point values."""
    if not isinstance(piece.topology, td.UnstructuredTopology):
        piece = td.extract_cells(piece, np.arange(piece.num_cells))   # keeps the marks
    positions = piece.positions()
    offsets = tack.arrays.to_host(piece.topology.offsets)
    connectivity = tack.arrays.to_host(piece.topology.connectivity)
    cell_marks = piece.ghosts.cells.numpy()
    point_marks = piece.ghosts.points.numpy()
    cell_fields = [n for n in sorted(piece.fields) if _on_cells(piece.fields[n])]
    point_fields = [n for n in sorted(piece.fields) if n not in cell_fields]
    values = {n: tack.arrays.to_host(piece.fields[n].values) for n in piece.fields}
    cells = {}
    for c in range(piece.num_cells):
        corners = tuple(sorted(_key(positions[p])
                               for p in connectivity[offsets[c]:offsets[c + 1]]))
        cells[corners] = (int(cell_marks[c]),
                          tuple(np.round(np.ravel(values[n][c]), 3).tolist() for n in cell_fields))
    points = {_key(positions[p]): (int(point_marks[p]),
                                   tuple(np.round(np.ravel(values[n][p]), 3).tolist()
                                         for n in point_fields))
              for p in range(piece.num_points)}
    return cells, points


def _same(ours, reference, compare_cell_values=True):
    a_cells, a_points = described(ours)
    b_cells, b_points = described(reference)
    assert set(a_cells) == set(b_cells)
    assert set(a_points) == set(b_points)
    for key in a_cells:
        assert a_cells[key][0] == b_cells[key][0], key
        if compare_cell_values:
            assert a_cells[key][1] == b_cells[key][1], key
    assert a_points == b_points


@pytest.mark.parametrize("pieces", [2, 4, (2, 2, 2), 6])
def test_ghost_cells_are_splits_ghost_layer(pieces):
    whole = _with_fields(td.wavelet(4))
    ours = td.ghost_cells(_unstructured(td.split(whole, pieces)))
    reference = td.split(whole, pieces, ghost_layers=1)
    assert len(ours) == len(reference)
    for a, b in zip(ours, reference):
        assert isinstance(a.topology, td.UnstructuredTopology)
        _same(a, b)


def test_ghost_cells_of_two_dimensional_pieces():
    grid = td.uniform_grid((9, 7), spacing=(0.5, 0.5, 1.0))
    grid.fields["RTData"] = td.Field(td.H1(grid), tack.array(
        grid.positions()[:, 0].astype(np.float32)))
    grid = _with_fields(grid)
    for a, b in zip(td.ghost_cells(_unstructured(td.split(grid, (3, 2)))),
                    td.split(grid, (3, 2), ghost_layers=1)):
        _same(a, b)


def test_pieces_on_two_devices():
    whole = _with_fields(td.wavelet(4))
    second = dispatch.as_device(_SecondCPU())
    plain = _unstructured(td.split(whole, 4))
    moved = td.PartitionedDataSet([p.to_device(second) if g % 2 else p
                                   for g, p in enumerate(plain)])
    ours = td.ghost_cells(moved)
    for g, (a, b) in enumerate(zip(ours, td.split(whole, 4, ghost_layers=1))):
        assert a.device == (second if g % 2 else tack.device("cpu"))
        _same(a.to_device("cpu"), b)


@pytest.mark.parametrize("name", ["contour", "clip", "threshold", "extract_surface",
                                  "tetrahedralize, surface"])
def test_filters_give_each_generated_piece_its_share(name):
    from test_partitioned import _FILTERS, _check_shares
    whole = td.wavelet(6)
    _check_shares(whole, td.ghost_cells(_unstructured(td.split(whole, (2, 2, 2)))),
                  _FILTERS[name])


def test_update_ghosts_refreshes_point_and_cell_values():
    whole = td.wavelet(6)
    whole.fields["c"] = td.project(whole, "RTData", td.Constant(whole))
    whole.fields["p"] = td.project(whole, "c", td.H1(whole))
    expected = td.gradients(whole, "p", at="points").values.numpy()
    where = {_key(x): i for i, x in enumerate(whole.positions())}
    pieces = td.ghost_cells(_unstructured(td.split(whole, 4)))

    def error():
        worst = 0.0
        for piece in pieces:
            got = td.gradients(piece, "p", at="points").values.numpy()
            owned = (piece.ghosts.points.numpy() & Ghosts.DUPLICATEPOINT) == 0
            ids = [where[_key(x)] for x in piece.positions()]
            worst = max(worst, np.abs(got[owned] - expected[ids][owned]).max())
        return worst

    for piece in pieces:
        piece.fields["p"] = td.project(piece, "c", td.H1(piece))
        piece.fields["k"] = td.Field(td.Constant(piece), tack.array(np.where(
            piece.ghosts.cells.numpy() & Ghosts.DUPLICATECELL, -1.0,
            piece.fields["c"].values.numpy()).astype(np.float32)))
    assert error() > 1e-2 * np.abs(expected).max()
    td.update_ghosts(pieces, "p", "k")
    assert error() <= 1e-5 * np.abs(expected).max()
    for piece in pieces:                     # ghost cells got their owners' values
        np.testing.assert_array_equal(piece.fields["k"].values.numpy(),
                                      piece.fields["c"].values.numpy())


def test_ghost_cells_refuse_what_they_cannot_do():
    whole = _with_fields(td.wavelet(3))
    plain = _unstructured(td.split(whole, 2))
    with pytest.raises(NotImplementedError, match="one layer"):
        td.ghost_cells(plain, layers=2)
    with pytest.raises(TypeError, match="unstructured pieces"):
        td.ghost_cells(td.split(whole, 2))
    with pytest.raises(ValueError, match="already has ghost cells"):
        td.ghost_cells(td.ghost_cells(plain))
    with pytest.raises(TypeError, match="PartitionedDataSet"):
        td.ghost_cells(list(plain))
    for piece in plain:
        piece.fields["dg"] = td.project(piece, "RTData", td.L2(piece, order=1))
    with pytest.raises(TypeError, match="point and cell fields"):
        td.ghost_cells(plain)
    out = td.ghost_cells(plain, fields=["RTData", "c"])
    assert sorted(out[0].fields) == ["RTData", "c"]
    with pytest.raises(ValueError, match="ghost_cells made"):
        td.update_ghosts(plain, "RTData")
    with pytest.raises(TypeError, match="point and cell fields"):
        for piece in out:
            piece.fields["dg"] = td.project(piece, "RTData", td.L2(piece, order=1))
        td.update_ghosts(out, "dg")


@needs_vtk
@needs_diy2
def test_ghost_cells_are_vtks():
    """On VTK's pieces of an unstructured mesh: the cells and points
    vtkGhostCellsGenerator adds, with its marks and values."""
    image = vtkRTAnalyticSource(whole_extent=(-6, 6, -6, 6, -6, 6))()
    redistribute = vtkRedistributeDataSetFilter(number_of_partitions=4,
                                                preserve_partitions_in_output=True)
    plain = redistribute(image)
    ours = td.ghost_cells(vtk_to_dataset(plain))
    reference = vtk_to_dataset(vtkGhostCellsGenerator(number_of_ghost_layers=1,
                                                      build_if_required=False)(plain))
    assert len(ours) == len(reference)
    for a, b in zip(ours, reference):
        _same(a, b)


def test_ghost_cells_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_ghost_cells_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def ghost_cells_across(world):
    """``ghost_cells`` over this rank's share of six pieces of a wavelet: each piece
    described by position, ``{gid: description}``."""
    whole = _with_fields(td.wavelet(4))
    every = list(_unstructured(td.split(whole, 6)))
    share = len(every) // world.size
    mine = td.PartitionedDataSet(every[world.rank * share:(world.rank + 1) * share], world)
    out = td.ghost_cells(mine)
    blocks = tc.Blocks([p.device for p in out], world)
    return {g: described(p) for g, p in zip(blocks.local, out)}
