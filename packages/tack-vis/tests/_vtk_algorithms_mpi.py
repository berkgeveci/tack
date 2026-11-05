# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``test_vtk_algorithms``' check across MPI ranks: ``mpiexec -n 3 python
_vtk_algorithms_mpi.py comm|vtk``. Each rank holds one slab of VTK's wavelet, the
slabs sharing their planes of points, in a ``vtkPartitionedDataSet`` of one
partition; the filters over pieces run in a VTK pipeline on every rank. Then each
holds blocks of VTK's AMR Gaussian pulse, the others' empty, as VTK spreads a
hierarchy; the AMR filters give the whole's. The world is given as ``world=`` (an
mpi4py communicator: ``comm``) or taken from VTK's global controller (``vtk``,
which needs a VTK built with MPI and vtkParallelMPI4Py). Rank 0 prints ``ok``, or
the failure."""

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

    # Reductions: every rank gets the whole's, each shared point counted once.
    stats = (ta.GenerateGlobalIds(**options) >> ta.Statistics(fields=["RTData"],
                                                              **options))(pieces)
    count = stats.GetColumnByName("n").GetValue(0)
    assert count == points, (count, points)
    volume = ta.Integrate(**options)(pieces).GetCellData().GetArray("Volume").GetValue(0)
    assert volume == 16.0 ** 3, volume

    merged = ta.MergePieces(**options)(pieces)
    if rank == 0:
        assert merged.number_of_points == points, (merged.number_of_points, points)
    hierarchy(comm, options)
    temporal(comm, options)
    if rank == 0:
        print(f"ok: {size} ranks, world from {mode}", flush=True)


def flow(t):
    """A step at ``t`` as VTK's: an image with a velocity ``v`` along x, turning."""
    import tack.dataset as td
    from tack.interop.vtk import dataset_to_vtk

    grid = td.uniform_grid((13, 7, 3))
    grid.time = t
    v = np.zeros((grid.num_points, 3), np.float32)
    v[:, 0], v[:, 1] = 1.0, 0.2 * t
    grid.fields["v"] = td.Field(td.H1(grid), tack.array(v, tack.vec3()))
    return dataset_to_vtk(grid)


def temporal(comm, options):
    """Pathlines in situ over slabs of each step, the seeds on rank 0: particles cross
    from rank to rank, and the paths are the whole's."""
    from tack.communication import World

    rank, size = comm.Get_rank(), comm.Get_size()
    seeds = np.array([[0.5, 2.0, 1.0], [1.5, 4.0, 1.0]])
    cuts = np.linspace(0, 12, size + 1).round().astype(int)
    spread = ta.Pathlines(field="v", seeds=seeds if rank == 0 else np.zeros((0, 3)),
                          **options)
    whole = ta.Pathlines(field="v", seeds=seeds, world=World())
    for t in (0.0, 1.0, 2.0, 3.0):
        step = flow(t)
        slab = dm.vtkPartitionedDataSet()
        slab.append(vtkExtractVOI(voi=(int(cuts[rank]), int(cuts[rank + 1]), 0, 6, 0, 2))(step))
        slab.GetInformation().Set(dm.vtkDataObject.DATA_TIME_STEP(), t)
        spread.SetInputDataObject(0, slab)
        spread.Update()
        assert spread.error is None, spread.error
        whole.SetInputDataObject(0, step)
        whole.Update()
    mine = spread.GetOutputDataObject(0)
    points = [np.asarray(block.points.data) for block in mine if block.number_of_points]
    every = comm.gather(np.concatenate(points) if points else np.zeros((0, 3)))
    if rank == 0:
        got = np.concatenate(every)
        expected = np.asarray(whole.GetOutputDataObject(0).points.data)
        assert got.shape == expected.shape, (got.shape, expected.shape)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def pulse():
    from vtkmodules.vtkFiltersAMR import vtkAMRGaussianPulseSource

    return vtkAMRGaussianPulseSource(dimension=3)()


def hierarchy(comm, options):
    """The pulse's blocks dealt to the ranks in turn: the statistics and integral of the
    spread hierarchy, its averages down and its ghost cells, and a filter box by box,
    are the whole's."""
    from tack.communication import World

    rank, size = comm.Get_rank(), comm.Get_size()

    alone = {"world": World()}
    whole, spread = pulse(), pulse()
    mine = []
    blocks = [(level, index) for level in range(spread.GetNumberOfLevels())
              for index in range(spread.GetNumberOfBlocks(level))]
    for k, (level, index) in enumerate(blocks):
        if k % size == rank:
            mine.append((level, index))
        else:
            # Through the level's partitioned dataset: SetDataSet ignores None.
            spread.GetPartitionedDataSet(level).SetPartition(index, None)
    for make in (lambda **o: ta.Statistics(fields=["Gaussian-Pulse"], **o),
                 lambda **o: ta.GhostCells(layers=1, **o) >> ta.Statistics(
                     fields=["Gaussian-Pulse"], **o)):
        expected, got = make(**alone)(whole), make(**options)(spread)
        for name in ("n", "sum", "min", "max"):
            a = got.GetColumnByName(name).GetValue(0)
            b = expected.GetColumnByName(name).GetValue(0)
            assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), (name, a, b)
    def cells(collection, level):
        return sum(block.number_of_cells for block in collection.GetPartitionedDataSet(level))

    threshold = {"field": ("Gaussian-Pulse", "cells"), "lower": 0.0, "upper": 1.0}
    kept = ta.Threshold(**threshold, **options)(spread)         # box by box
    whole_kept = ta.Threshold(**threshold, **alone)(whole)
    for level in range(whole.GetNumberOfLevels()):
        total = comm.allreduce(cells(kept, level))
        assert total == cells(whole_kept, level), (level, total)
    volume = ta.Integrate(**options)(spread).cell_data["Volume"].GetValue(0)
    assert abs(volume - ta.Integrate(**alone)(whole).cell_data["Volume"].GetValue(0)) < 1e-12
    averaged = ta.AverageDown(field="Gaussian-Pulse", **options)(spread)
    reference = ta.AverageDown(field="Gaussian-Pulse", **alone)(whole)
    for level, index in mine:
        got = np.asarray(averaged.GetDataSetAsCartesianGrid(level, index)
                         .cell_data["Gaussian-Pulse"])
        want = np.asarray(reference.GetDataSetAsCartesianGrid(level, index)
                          .cell_data["Gaussian-Pulse"])
        assert np.array_equal(got, want), (level, index)


if __name__ == "__main__":
    main()
