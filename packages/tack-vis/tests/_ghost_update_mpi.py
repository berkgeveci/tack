"""``test_ghost_update``'s check across MPI ranks:
``mpiexec -n 2 python _ghost_update_mpi.py 4`` (the number of pieces, shared out
evenly). Rank 0 prints ``ok`` with the errors, or the failure."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_ghost_update as case
from mpi4py import MPI

from tack import communication as tc


def main():
    world = tc.World(MPI.COMM_WORLD)
    before, after, scale = case.ghost_update_errors(world, int(sys.argv[1]))
    if world.rank == 0:
        case._check(before, after, scale)
        print(f"ok: before {before}, after {after}, largest gradient {scale}", flush=True)


if __name__ == "__main__":
    main()
