"""``test_vtk_algorithms``' check across MPI ranks: ``mpiexec -n 3 python
_vtk_algorithms_mpi.py comm|vtk``. Each rank holds one slab of VTK's wavelet, the
slabs sharing their planes of points, in a ``vtkPartitionedDataSet`` of one
partition; the filters over pieces run in a VTK pipeline on every rank. The world
is given as ``world=`` (an mpi4py communicator: ``comm``) or taken from VTK's global
controller (``vtk``, which needs a VTK built with MPI and vtkParallelMPI4Py).
Rank 0 prints ``ok``, or the failure."""

import sys

import numpy as np
import vtkmodules.vtkCommonDataModel as dm
from mpi4py import MPI
from vtkmodules.vtkImagingCore import vtkExtractVOI, vtkRTAnalyticSource

import tack
from tack.interop.vtk import algorithms as ta

EXTENT = (-8, 8, -8, 8, -8, 8)


def slab(rank, size):
    """This rank's slab of the wavelet along x, sharing its end planes with its
    neighbours', as one partition."""
    cuts = np.linspace(EXTENT[0], EXTENT[1], size + 1).round().astype(int)
    whole = vtkRTAnalyticSource(whole_extent=EXTENT)()
    voi = (int(cuts[rank]), int(cuts[rank + 1]), *EXTENT[2:])
    pieces = dm.vtkPartitionedDataSet()
    pieces.append(vtkExtractVOI(voi=voi)(whole))
    return pieces, whole.number_of_points, whole.number_of_cells


def main():
    mode = sys.argv[1]
    tack.set_default_device("cpu")
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    options = {}
    if mode == "vtk":
        from vtkmodules.vtkParallelCore import vtkMultiProcessController
        from vtkmodules.vtkParallelMPI import vtkMPIController

        controller = vtkMPIController()
        controller.Initialize()             # MPI started by mpi4py: initialized externally
        vtkMultiProcessController.SetGlobalController(controller)
        from tack.interop.vtk import world

        assert world().size == size and world() is world()
    else:
        options["world"] = comm
    pieces, points, cells = slab(rank, size)

    numbered = ta.GenerateGlobalIds(**options)(pieces)
    ids = np.asarray(numbered.GetPartition(0).GetPointData().GetGlobalIds())
    every = np.unique(np.concatenate(comm.allgather(ids)))
    assert np.array_equal(every, np.arange(points)), (every.size, points)

    ghosted = ta.GhostCells(**options)(pieces).GetPartition(0)
    marks = np.asarray(ghosted.cell_data["vtkGhostType"])
    owned = comm.allreduce(int(np.count_nonzero(marks == 0)))
    assert owned == cells, (owned, cells)
    assert size == 1 or np.count_nonzero(marks), "no ghost cells from the neighbours"

    merged = ta.MergePieces(**options)(pieces)
    if rank == 0:
        assert merged.number_of_points == points, (merged.number_of_points, points)
        print(f"ok: {size} ranks, world from {mode}", flush=True)


if __name__ == "__main__":
    main()
