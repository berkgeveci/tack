"""Global ids: the data model, what filters carry, VTK's arrays, and generating them.

``generate_global_ids`` is checked against VTK's ``vtkGenerateGlobalIds`` on
``vtkRedistributeDataSetFilter``'s pieces (ids and duplicate marks equal), and
on pieces with ghost cells for what makes a numbering right: every position
one id and every id one position, owned entities numbered from 0 without
gaps, copies taking their owner's id. Its three routes -- the exchanges
``ghost_cells`` kept, matching on ghosted pieces, and ids made before the
ghost cells and carried by ``ghost_cells`` -- agree, in one process and
across MPI ranks (``_global_ids_mpi.py``).
"""

import collections
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

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
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
        vtkGenerateGlobalIds,
        vtkGhostCellsGenerator,
        vtkRedistributeDataSetFilter,
    )
except ImportError:
    vtkGenerateGlobalIds = vtkGhostCellsGenerator = vtkRedistributeDataSetFilter = None

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


def _unstructured(pieces):
    return td.PartitionedDataSet([td.extract_cells(p, np.arange(p.num_cells)) for p in pieces])


def _point_marks(piece):
    marks = piece.ghosts
    return (np.zeros(piece.num_points, np.uint8) if marks is None or marks.points is None
            else marks.points.numpy())


def _cell_marks(piece):
    marks = piece.ghosts
    return (np.zeros(piece.num_cells, np.uint8) if marks is None or marks.cells is None
            else marks.cells.numpy())


def by_position(pieces):
    """Each piece's point ids by position, and its cell ids by corner positions."""
    out = []
    for piece in pieces:
        keys = [tuple(x) for x in np.round(piece.positions(), 5)]
        points = dict(zip(keys, piece.global_ids.points.numpy().tolist()))
        rows = td.extract_cells(piece, np.arange(piece.num_cells)) \
            if isinstance(piece.topology, td.StructuredTopology) else piece
        offsets = tack.arrays.to_host(rows.topology.offsets)
        connectivity = tack.arrays.to_host(rows.topology.connectivity)
        cells = {tuple(sorted(keys[p] for p in connectivity[offsets[c]:offsets[c + 1]])): int(i)
                 for c, i in enumerate(piece.global_ids.cells.numpy())}
        out.append((points, cells))
    return out


def _check_numbering(pieces):
    """Every position one id and every id one position, numbered from 0 without
    gaps; each owned once, its copies marked; the same for cells by corners."""
    described = by_position(pieces)
    for which, marks, bit in ((0, _point_marks, Ghosts.DUPLICATEPOINT),
                              (1, _cell_marks, Ghosts.DUPLICATECELL)):
        ids_at, owned = collections.defaultdict(set), []
        for piece, entry in zip(pieces, described):
            for key, i in entry[which].items():
                ids_at[key].add(i)
            ids = (piece.global_ids.points if which == 0 else piece.global_ids.cells).numpy()
            owned.extend(ids[(marks(piece) & bit) == 0].tolist())
        assert all(len(v) == 1 for v in ids_at.values())
        assert sorted(owned) == list(range(len(ids_at)))      # each owned exactly once


# -- the data model --------------------------------------------------------

def test_global_ids_are_checked_and_move_with_the_dataset():
    data = td.wavelet(2)
    data.global_ids = td.GlobalIds(cells=np.arange(data.num_cells) + 10,
                                   points=np.arange(data.num_points))
    assert data.global_ids.cells.dtype == tack.i64
    with pytest.raises(ValueError, match="global ids for"):
        data.global_ids = td.GlobalIds(points=np.arange(3))
    with pytest.raises(TypeError, match="one-dimensional i64"):
        td.GlobalIds(points=tack.zeros(3, tack.i32))
    with pytest.raises(TypeError, match="integers"):
        td.GlobalIds(points=np.zeros(3))
    with pytest.raises(TypeError, match="GlobalIds"):
        data.global_ids = np.arange(data.num_points)
    second = dispatch.as_device(_SecondCPU())
    moved = data.to_device(second)
    assert moved.global_ids.points._backend.memory_domain == ("cpu", 1)
    np.testing.assert_array_equal(moved.global_ids.cells.numpy(), data.global_ids.cells.numpy())
    with pytest.raises(ValueError, match="device"):
        td.GlobalIds(cells=tack.zeros(1, tack.i64), points=tack.zeros(1, tack.i64, device=second))


def test_filters_carry_the_ids_of_what_they_keep():
    data = td.wavelet(3)
    data.global_ids = td.GlobalIds(cells=np.arange(data.num_cells) + 1000,
                                   points=np.arange(data.num_points) + 5000)
    taken = td.extract_cells(data, [3, 9, 20])
    assert taken.global_ids.cells.numpy().tolist() == [1003, 1009, 1020]
    positions = {tuple(x): i for i, x in enumerate(data.positions())}
    expected = [positions[tuple(x)] + 5000 for x in taken.positions()]
    assert taken.global_ids.points.numpy().tolist() == expected
    kept = td.threshold(data, "RTData", 150.0, 250.0)
    assert set(kept.global_ids.cells.numpy().tolist()) <= set(range(1000, 1000 + data.num_cells))
    assert td.contour(data, "RTData", 150.0).global_ids is None       # made points
    surface = td.extract_surface(data)                                 # points kept, faces made
    assert surface.global_ids.cells is None and surface.global_ids.points is not None
    assert td.transform(data, np.eye(4)).global_ids is data.global_ids
    assert td.shrink(data, 0.5).global_ids is data.global_ids


# -- VTK -------------------------------------------------------------------

@needs_vtk
@needs_diy2
def test_vtks_global_ids_come_and_go_as_global_ids():
    image = vtkRTAnalyticSource(whole_extent=(-2, 2, -2, 2, -2, 2))()
    plain = vtkRedistributeDataSetFilter(number_of_partitions=2,
                                         preserve_partitions_in_output=True)(image)
    numbered = vtkGenerateGlobalIds()(plain)
    pieces = vtk_to_dataset(numbered)
    for i, piece in enumerate(pieces):
        source = numbered.GetPartition(i)                 # no pythonic form in 9.7.1
        assert "GlobalPointIds" not in piece.fields and "GlobalCellIds" not in piece.fields
        np.testing.assert_array_equal(piece.global_ids.points.numpy(),
                                      np.asarray(source.GetPointData().GetGlobalIds()))
        back = dataset_to_vtk(piece)
        np.testing.assert_array_equal(np.asarray(back.GetPointData().GetGlobalIds()),
                                      piece.global_ids.points.numpy())
        np.testing.assert_array_equal(np.asarray(back.GetCellData().GetGlobalIds()),
                                      piece.global_ids.cells.numpy())
        assert back.GetCellData().GetGlobalIds().GetName() == "GlobalCellIds"


@needs_vtk
@needs_diy2
def test_generated_ids_are_vtks():
    """On pieces without ghost cells: VTK's ids, and its duplicate marks."""
    image = vtkRTAnalyticSource(whole_extent=(-4, 4, -4, 4, -4, 4))()
    plain = vtkRedistributeDataSetFilter(number_of_partitions=4,
                                         preserve_partitions_in_output=True)(image)
    reference = vtk_to_dataset(vtkGenerateGlobalIds()(plain))
    ours = td.generate_global_ids(vtk_to_dataset(plain))
    for a, b in zip(ours, reference):
        np.testing.assert_array_equal(a.global_ids.points.numpy(), b.global_ids.points.numpy())
        np.testing.assert_array_equal(a.global_ids.cells.numpy(), b.global_ids.cells.numpy())
        np.testing.assert_array_equal(_point_marks(a), _point_marks(b))


@needs_vtk
@needs_diy2
def test_generated_ids_on_vtks_ghosted_pieces_keep_its_owners():
    """vtkGenerateGlobalIds gives a point to the lowest piece holding any copy of it,
    ghost copies too, and remarks the pieces; generate_global_ids keeps the
    owners the ghost marks say, so an owner still holds every cell around its
    points. Both are numberings; ours leaves the marks alone."""
    image = vtkRTAnalyticSource(whole_extent=(-4, 4, -4, 4, -4, 4))()
    layered = (vtkRedistributeDataSetFilter(number_of_partitions=4,
                                            preserve_partitions_in_output=True)
               >> vtkGhostCellsGenerator(number_of_ghost_layers=1, build_if_required=False))
    pieces = vtk_to_dataset(layered(image))
    ours = td.generate_global_ids(pieces)
    _check_numbering(ours)
    for a, b in zip(ours, pieces):
        np.testing.assert_array_equal(_point_marks(a), _point_marks(b))


# -- generating ------------------------------------------------------------

def test_generated_ids_number_plain_and_ghosted_pieces():
    whole = td.wavelet(4)
    plain = td.generate_global_ids(_unstructured(td.split(whole, 6)))
    _check_numbering(plain)
    for piece in plain[1:]:                         # copies on later pieces marked
        assert (_point_marks(piece) & Ghosts.DUPLICATEPOINT).any()
    _check_numbering(td.generate_global_ids(td.split(whole, 6, ghost_layers=1)))


def test_the_three_routes_agree():
    whole = td.wavelet(4)
    plain = _unstructured(td.split(whole, 6))
    kept = td.generate_global_ids(td.ghost_cells(plain))            # ghost_cells' exchanges
    matched = td.generate_global_ids(td.PartitionedDataSet(list(td.ghost_cells(plain))))
    carried = td.ghost_cells(td.generate_global_ids(plain))         # carried, matched by id
    structured = td.generate_global_ids(td.split(whole, 6, ghost_layers=1))
    expected = by_position(kept)
    assert by_position(matched) == expected
    assert by_position(carried) == expected
    assert by_position(structured) == expected
    td.update_ghosts(kept, *kept[0].fields)         # the kept exchanges still work


def test_pieces_on_two_devices_get_the_same_ids():
    whole = td.wavelet(4)
    second = dispatch.as_device(_SecondCPU())
    plain = _unstructured(td.split(whole, 4))
    moved = td.PartitionedDataSet([p.to_device(second) if g % 2 else p
                                   for g, p in enumerate(plain)])
    ours = td.generate_global_ids(td.ghost_cells(moved))
    assert all(p.global_ids.points._backend.memory_domain ==
               p.topology._device.memory_domain for p in ours)
    expected = by_position(td.generate_global_ids(td.ghost_cells(plain)))
    assert by_position([p.to_device("cpu") for p in ours]) == expected


def test_generate_global_ids_refuses():
    with pytest.raises(TypeError, match="PartitionedDataSet"):
        td.generate_global_ids([td.wavelet(2)])
    whole = td.wavelet(3)
    pieces = list(td.split(whole, 2, ghost_layers=1))
    marks = pieces[1].ghosts.points.numpy()
    marks[:] = Ghosts.DUPLICATEPOINT                # nobody owns piece 1's points
    pieces[1].ghosts = Ghosts(cells=pieces[1].ghosts.cells, points=marks)
    with pytest.raises(ValueError, match="no piece holds them unmarked"):
        td.generate_global_ids(td.PartitionedDataSet(pieces))


def test_generated_ids_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_global_ids_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def global_ids_across(world):
    """The three routes over this rank's share of six pieces: each piece's ids by
    position, ``{route: {gid: ids}}``."""
    whole = td.wavelet(4)
    every = list(_unstructured(td.split(whole, 6)))
    share = len(every) // world.size
    mine = every[world.rank * share:(world.rank + 1) * share]
    plain = td.PartitionedDataSet(mine, world)
    routes = {
        "kept": td.generate_global_ids(td.ghost_cells(plain)),
        "matched": td.generate_global_ids(td.PartitionedDataSet(list(td.ghost_cells(plain)),
                                                                world)),
        "carried": td.ghost_cells(td.generate_global_ids(plain)),
    }
    blocks = tc.Blocks([p.device for p in mine], world)
    return {route: dict(zip(blocks.local, by_position(pieces)))
            for route, pieces in routes.items()}
