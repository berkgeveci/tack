"""``test_reductions_across_ranks``' check under MPI: ``mpiexec -n 4 python
_reductions_mpi.py 6`` (the number of pieces). Every rank must get exactly what
one process gets; rank 0 prints ``ok`` or fails."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_reductions_across_ranks as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    pieces = int(sys.argv[1])
    got = comm.gather(case.reductions(tc.World(comm), pieces), root=0)
    if comm.rank == 0:
        alone = case.reductions(tc.World(), pieces)
        assert all(result == alone for result in got), (got, alone)
        print(f"ok: {pieces} pieces on {comm.size} ranks, the same bits", flush=True)


if __name__ == "__main__":
    main()
