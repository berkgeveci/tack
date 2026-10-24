"""``merge_pieces``: a partitioned dataset back as one dataset, each cell and point
once.

Structured pieces that tile a box come back as the box's grid -- ``split``'s
input exactly, coordinates and offset included -- with or without ghost
layers; other pieces as an unstructured dataset whose cells and points are
the whole's. Ghost cells go, hidden cells stay marked, global ids come along.
Across MPI ranks the result is on the root rank (``_merge_pieces_mpi.py``).
"""

import os
import types

import numpy as np
import pytest
from test_ghost_cells import (
    _curvilinear,
    _planar,
    _rectilinear,
    _unstructured,
    _with_fields,
    described,
)

import tack
import tack.dataset as td
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
try:
    from vtkmodules.vtkFiltersParallelDIY2 import vtkRedistributeDataSetFilter
except ImportError:
    vtkRedistributeDataSetFilter = None

needs_diy2 = pytest.mark.skipif(vtk is None or vtkRedistributeDataSetFilter is None,
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


def _marked(data):
    """``data`` with zero marks where it has none, for ``described``."""
    if data.ghosts is None:
        data.ghosts = Ghosts(cells=np.zeros(data.num_cells, np.uint8),
                             points=np.zeros(data.num_points, np.uint8))
    return data


def _is_whole(merged, whole):
    assert isinstance(merged.topology, td.StructuredTopology)
    assert merged.topology.point_dims == whole.topology.point_dims
    assert merged.topology.offset == whole.topology.offset
    assert type(merged.geometry.values) is type(whole.geometry.values)
    np.testing.assert_array_equal(merged.positions(), whole.positions())
    for name in whole.fields:
        np.testing.assert_array_equal(merged.fields[name].values.numpy(),
                                      whole.fields[name].values.numpy(), err_msg=name)
    assert merged.ghosts is None


@pytest.mark.parametrize("make", [lambda: td.wavelet(4), _rectilinear, _curvilinear, _planar],
                         ids=["uniform", "rectilinear", "explicit", "2d"])
@pytest.mark.parametrize("layers", [0, 1, 2])
def test_structured_pieces_merge_into_the_whole_grid(make, layers):
    whole = _with_fields(make())
    _is_whole(td.merge_pieces(td.split(whole, 3, ghost_layers=layers)), whole)
    _is_whole(td.merge_pieces(td.ghost_cells(td.split(whole, 3), layers=max(layers, 1))), whole)


@pytest.mark.parametrize("layers", [0, 1, 2])
def test_unstructured_pieces_merge_into_the_wholes_cells_and_points(layers):
    whole = _with_fields(td.wavelet(4))
    merged = td.merge_pieces(_unstructured(td.split(whole, (2, 2, 2), ghost_layers=layers)))
    assert isinstance(merged.topology, td.UnstructuredTopology)
    assert (merged.num_cells, merged.num_points) == (whole.num_cells, whole.num_points)
    reference = _marked(td.extract_cells(whole, np.arange(whole.num_cells)))
    assert described(_marked(merged)) == described(reference)


@needs_diy2
def test_vtks_pieces_merge_into_the_image():
    image = vtkRTAnalyticSource(whole_extent=(-6, 6, -6, 6, -6, 6))()
    pieces = vtk_to_dataset(vtkRedistributeDataSetFilter(
        number_of_partitions=4, preserve_partitions_in_output=True)(image))
    whole = vtk_to_dataset(image)
    merged = td.merge_pieces(pieces, fields=["RTData"])
    reference = _marked(td.extract_cells(whole, np.arange(whole.num_cells), fields=["RTData"]))
    assert described(_marked(merged)) == described(reference)


def test_hidden_cells_global_ids_and_fields():
    whole = _with_fields(td.wavelet(4))
    marks = np.zeros(whole.num_cells, np.uint8)
    marks[::9] = Ghosts.HIDDENCELL
    whole.ghosts = Ghosts(cells=marks)
    whole.global_ids = td.GlobalIds(cells=np.arange(whole.num_cells) + 100,
                                    points=np.arange(whole.num_points) + 7)
    merged = td.merge_pieces(td.split(whole, 4, ghost_layers=1), fields=["c"])
    assert sorted(merged.fields) == ["c"]
    np.testing.assert_array_equal(merged.ghosts.cells.numpy(), marks)
    np.testing.assert_array_equal(merged.global_ids.cells.numpy(), whole.global_ids.cells.numpy())
    np.testing.assert_array_equal(merged.global_ids.points.numpy(),
                                  whole.global_ids.points.numpy())
    # Unstructured pieces keep their hidden cells too (marked here, since
    # extract_cells, as every filter, leaves them out).
    plain = list(_unstructured(td.split(_with_fields(td.wavelet(4)), 4, ghost_layers=1)))
    hidden = 0
    for piece in plain:
        cells = piece.ghosts.cells.numpy().copy()
        owned = np.flatnonzero((cells & Ghosts.DUPLICATECELL) == 0)
        cells[owned[::5]] |= Ghosts.HIDDENCELL
        hidden += len(owned[::5])
        piece.ghosts = Ghosts(cells=cells, points=piece.ghosts.points)
    unstructured = td.merge_pieces(td.PartitionedDataSet(plain))
    assert int(((unstructured.ghosts.cells.numpy() & Ghosts.HIDDENCELL) != 0).sum()) == hidden
    # Structured pieces that do not tile a box merge as unstructured, hidden
    # cells kept.
    loose = [p for g, p in enumerate(td.split(whole, 4)) if g != 1]
    for p in loose:               # the dropped piece owned some points: unmark them
        p.ghosts = Ghosts(cells=p.ghosts.cells)
    merged_loose = td.merge_pieces(td.PartitionedDataSet(loose))
    assert isinstance(merged_loose.topology, td.UnstructuredTopology)
    assert int(((merged_loose.ghosts.cells.numpy() & Ghosts.HIDDENCELL) != 0).sum()) == \
        sum(int(((p.ghosts.cells.numpy() & Ghosts.HIDDENCELL) != 0).sum()) for p in loose)
    with pytest.raises(TypeError, match="PartitionedDataSet"):
        td.merge_pieces([whole])
    with pytest.raises(ValueError, match="root is a rank"):
        td.merge_pieces(td.split(whole, 2), root=1)


def test_pieces_on_two_devices_and_the_result_device():
    whole = _with_fields(td.wavelet(4))
    second = dispatch.as_device(_SecondCPU())
    pieces = td.PartitionedDataSet([p.to_device(second) if g % 2 else p for g, p in
                                    enumerate(td.split(whole, 4, ghost_layers=1))])
    merged = td.merge_pieces(pieces, device=second)
    assert merged.device == second
    _is_whole(merged.to_device("cpu"), whole)


def test_merge_pieces_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_merge_pieces_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def merged_across(world, structured):
    """This rank's share of six pieces merged onto rank 0: the dataset described,
    or ``None`` on the other ranks."""
    whole = _with_fields(td.wavelet(4))
    every = list(td.split(whole, 6, ghost_layers=1))
    if not structured:
        every = list(_unstructured(every))
    mine = np.array_split(np.arange(len(every)), world.size)[world.rank]
    merged = td.merge_pieces(td.PartitionedDataSet([every[i] for i in mine], world))
    if merged is None:
        return None
    return (type(merged.topology).__name__, described(_marked(merged)))
