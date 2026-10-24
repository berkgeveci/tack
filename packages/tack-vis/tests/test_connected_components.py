"""Connected components: cells connected through shared points, on one dataset and
across the pieces of a partitioned one.

On one dataset the regions are VTK's ``vtkConnectivityFilter``'s (all regions,
colored), numbered alike. Across pieces each copy of a cell gets one region,
the partition is the whole's, and with the whole's global ids on the pieces
the numbering is the whole's too, in one process and across MPI ranks
(``_connected_components_mpi.py``).
"""

import os
import types

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import Ghosts
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkFiltersCore import vtkConnectivityFilter

    from tack.interop.vtk import dataset_to_vtk
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace()

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

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


def lobes():
    """The tangle's low lobes: eight regions."""
    data = td.tangle((32, 32, 32))
    data.global_ids = td.GlobalIds(cells=np.arange(data.num_cells),
                                   points=np.arange(data.num_points))
    values = data.fields["tangle"].values.numpy()
    return data, float(values.min()), float(np.percentile(values, 8))


def _regions(data):
    return data.fields["region"].values.numpy(), data.fields["point_region"].values.numpy()


def test_one_dataset():
    data, lo, hi = lobes()
    blobs = td.threshold(data, "tangle", lo, hi)
    cells, points = _regions(td.connected_components(blobs))
    assert cells.max() + 1 == 8 and np.bincount(cells).min() > 150
    # Regions in order of their lowest cell; a point takes its cells' region.
    firsts = [int(np.flatnonzero(cells == r)[0]) for r in range(8)]
    assert firsts == sorted(firsts)
    offsets = tack.arrays.to_host(blobs.topology.offsets)
    conn = tack.arrays.to_host(blobs.topology.connectivity)
    for c in range(0, blobs.num_cells, 97):
        assert set(points[conn[offsets[c]:offsets[c + 1]]].tolist()) == {cells[c]}
    whole = td.connected_components(data)                          # structured, one region
    assert set(_regions(whole)[0].tolist()) == {0}


@needs_vtk
def test_one_dataset_is_vtks():
    data, lo, hi = lobes()
    blobs = td.threshold(data, "tangle", lo, hi)
    cells, _ = _regions(td.connected_components(blobs))
    out = vtkConnectivityFilter(extraction_mode=5, color_regions=True)(dataset_to_vtk(blobs))
    np.testing.assert_array_equal(cells, np.asarray(out.cell_data["RegionId"]))


def test_hidden_cells_and_unused_points():
    grid = td.uniform_grid((6, 3, 3))
    marks = np.zeros(grid.num_cells, np.uint8)
    marks[[2, 7, 12, 17]] = Ghosts.HIDDENCELL       # the x = 2 slab: two regions
    grid.ghosts = Ghosts(cells=marks)
    cells, points = _regions(td.connected_components(grid))
    assert set(cells[marks != 0].tolist()) == {-1}
    assert len(set(cells[marks == 0].tolist())) == 2
    planar = td.uniform_grid((5, 4))
    assert set(_regions(td.connected_components(planar))[0].tolist()) == {0}
    lonely = td.DataSet(td.UnstructuredTopology([10], [0, 4], [0, 1, 2, 3], num_points=5),
                        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [5, 5, 5]],
                                 np.float32))
    assert _regions(td.connected_components(lonely))[1].tolist() == [0, 0, 0, 0, -1]


def _across(pieces):
    """Each cell's region by its global id, refusing two regions for one cell."""
    out = {}
    for piece in td.connected_components(pieces):
        cells, _ = _regions(piece)
        for gid, r in zip(piece.global_ids.cells.numpy().tolist(), cells.tolist()):
            assert out.setdefault(gid, r) == r, f"cell {gid} in two regions"
    return out


@pytest.mark.parametrize("layers", [0, 1])
def test_pieces_number_regions_as_the_whole(layers):
    data, lo, hi = lobes()
    blobs = td.threshold(data, "tangle", lo, hi)
    whole = dict(zip(blobs.global_ids.cells.numpy().tolist(),
                     _regions(td.connected_components(blobs))[0].tolist()))
    pieces = td.PartitionedDataSet([td.threshold(p, "tangle", lo, hi)
                                    for p in td.split(data, (2, 2, 2), ghost_layers=layers)])
    assert _across(pieces) == whole


def test_pieces_without_global_ids_and_on_two_devices():
    """Matched by position, numbered by (piece, cell): the whole's partition."""
    data, lo, hi = lobes()
    blobs = td.threshold(data, "tangle", lo, hi)
    whole = dict(zip(blobs.global_ids.cells.numpy().tolist(),
                     _regions(td.connected_components(blobs))[0].tolist()))
    second = dispatch.as_device(_SecondCPU())
    pieces = []
    for g, p in enumerate(td.split(data, (2, 2, 2), ghost_layers=1)):
        piece = td.threshold(p, "tangle", lo, hi)
        cell_ids = piece.global_ids.cells
        piece.global_ids = None
        piece = piece.to_device(second) if g % 2 else piece
        pieces.append((piece, cell_ids.numpy()))
    out = td.connected_components(td.PartitionedDataSet([p for p, _ in pieces]))
    pairs = set()
    for (piece, ids), result in zip(pieces, out):
        assert result.device == piece.device
        pairs |= {(whole[i], r) for i, r in zip(ids.tolist(), _regions(result)[0].tolist())}
    assert len(pairs) == 8 == len({a for a, _ in pairs}) == len({b for _, b in pairs})


def test_connected_components_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_connected_components_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def components_across(world):
    """Each cell's region by global id over this rank's share of eight pieces."""
    data, lo, hi = lobes()
    every = [td.threshold(p, "tangle", lo, hi)
             for p in td.split(data, (2, 2, 2), ghost_layers=1)]
    mine = np.array_split(np.arange(len(every)), world.size)[world.rank]
    return _across(td.PartitionedDataSet([every[i] for i in mine], world))
