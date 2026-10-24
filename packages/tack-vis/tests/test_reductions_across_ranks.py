"""``statistics`` and ``entropy`` of a partitioned dataset across MPI ranks.

Each rank passes its share of the pieces, with a ``tack.communication.World``;
the moments and histograms of every piece are gathered and combined in the
pieces' global order, so the result has the same bits as in one process,
however the pieces are shared out -- a rank with none too
(``_reductions_mpi.py``).
"""

import os

import numpy as np
import pytest

import tack.dataset as td
from tack import communication as tc

pytestmark = pytest.mark.quick


def reductions(world, pieces):
    """Statistics of a point and a cell field and their entropy over this rank's
    share of ``pieces`` pieces of a wavelet with a ghost layer."""
    whole = td.wavelet(5)
    whole.fields["c"] = td.project(whole, "RTData", td.Constant(whole))
    every = list(td.split(whole, pieces, ghost_layers=1))
    mine = np.array_split(np.arange(pieces), world.size)[world.rank]
    data = td.PartitionedDataSet([every[i] for i in mine], world)
    return (tuple(td.statistics(data, "RTData")), tuple(td.statistics(data, "c")),
            td.entropy(data, "RTData", 16), td.entropy(data, "c", 7))


def test_one_process_gives_the_wholes():
    whole = td.wavelet(5)
    whole.fields["c"] = td.project(whole, "RTData", td.Constant(whole))
    points, cells, e_points, e_cells = reductions(tc.World(), 6)
    assert points[0] == whole.num_points and cells[0] == whole.num_cells
    assert points[4] == pytest.approx(td.statistics(whole, "RTData").mean, rel=1e-6)
    assert e_points == pytest.approx(td.entropy(whole, "RTData", 16), rel=1e-12)


@pytest.mark.parametrize("ranks, pieces", [(2, 6), (4, 6), (4, 3)])
def test_across_mpi_ranks(mpiexec, ranks, pieces):
    run = mpiexec(ranks, os.path.join(os.path.dirname(__file__), "_reductions_mpi.py"), pieces)
    assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr
