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
        got = {}
        for part, n in gathered:
            got.update(part)
            assert n == case.td.statistics(whole, "c").n, (n, comm.size)
        assert got == expected
        print(f"ok: {len(got)} boxes, {comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
