"""``test_amr``'s check under MPI: ``mpiexec -n 2 python _amr_mpi.py``. Each rank
holds a share of each level's boxes; rank 0 compares every box's marks with the
whole hierarchy's in one process, and prints ``ok`` or fails. (Which sibling owns a
shared point follows the boxes' global order, which depends on the ranks, so the
siblings' ``DUPLICATEPOINT`` is compared by count.)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_amr as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    gathered = comm.gather(case.across(tc.World(comm)), root=0)
    if comm.rank == 0:
        whole = case.hierarchy()
        expected = {}
        for level, boxes in enumerate(whole.levels):
            for k, box in enumerate(boxes):
                expected[(level, k)] = (box.ghosts.cells.numpy().tolist(),
                                        (box.ghosts.points.numpy()
                                         & (0xFF ^ case.Ghosts.DUPLICATEPOINT)).tolist())
        for level, boxes in enumerate(case.td.ghost_cells(whole).levels):
            for k, box in enumerate(boxes):
                expected[(level, k, "ghosts")] = (box.fields["c"].values.numpy().tolist(),
                                                  box.fields["p"].values.numpy().tolist())
        points = case.np.column_stack([case.np.linspace(0.1, 7.9, 40),
                                       case.np.linspace(7.9, 0.1, 40), case.np.zeros(40)])
        probed = case.td.probe(whole, points).fields["c"].values.numpy()
        for rank in range(comm.size):
            expected["probe", rank] = probed[rank::comm.size].tolist()
        got = {}
        for part, n in gathered:
            got.update(part)
            assert n == case.td.statistics(whole, "c").n, (n, comm.size)
        assert got == expected
        print(f"ok: {len(got)} boxes, {comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
