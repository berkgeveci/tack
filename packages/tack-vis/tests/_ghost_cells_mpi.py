"""``test_ghost_cells``' check across MPI ranks: ``mpiexec -n 2 python
_ghost_cells_mpi.py``. Every rank generates ghost cells for its share of the
pieces; rank 0 compares each piece with what one process makes of all of them,
and prints ``ok`` or fails."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_ghost_cells as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    for layers in (1, 2):
        mine = case.ghost_cells_across(tc.World(comm), layers)
        gathered = comm.gather(mine, root=0)
        if comm.rank == 0:
            alone = case.ghost_cells_across(tc.World(), layers)
            across = {g: d for part in gathered for g, d in part.items()}
            assert sorted(across) == sorted(alone)
            for g in alone:
                assert across[g] == alone[g], f"piece {g} differs, {layers} layers"
    if comm.rank == 0:
        print(f"ok: 6 pieces on {comm.size} ranks, 1 and 2 layers", flush=True)


if __name__ == "__main__":
    main()
