"""``test_global_ids``' check across MPI ranks: ``mpiexec -n 2 python
_global_ids_mpi.py``. Every rank numbers its share of the pieces by each route;
rank 0 compares every piece with what one process gives, and prints ``ok`` or
fails."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_global_ids as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    gathered = comm.gather(case.global_ids_across(tc.World(comm)), root=0)
    if comm.rank == 0:
        alone = case.global_ids_across(tc.World())
        for route, expected in alone.items():
            across = {g: d for part in gathered for g, d in part[route].items()}
            assert across == expected, f"route {route!r} differs"
            # The jittered pieces' ids are the plain pieces', by position.
            assert across == alone["kept"], f"route {route!r} differs from 'kept'"
        print(f"ok: {len(alone['kept'])} pieces on {comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
