"""``test_merge_pieces``' check under MPI: ``mpiexec -n 2 python
_merge_pieces_mpi.py``. Each rank passes its share of the pieces; rank 0 gets the
merged dataset and compares it with one process's, the others get ``None``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_merge_pieces as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    for structured in (True, False):
        mine = case.merged_across(tc.World(comm), structured)
        results = comm.gather(mine, root=0)
        if comm.rank == 0:
            assert results[0] is not None and all(r is None for r in results[1:])
            alone = case.merged_across(tc.World(), structured)
            assert results[0] == alone, f"structured={structured} differs"
    if comm.rank == 0:
        print(f"ok: merged on rank 0 of {comm.size}, structured and not", flush=True)


if __name__ == "__main__":
    main()
