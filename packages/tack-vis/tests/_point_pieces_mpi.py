"""``test_point_pieces``' check under MPI: ``mpiexec -n 2 python _point_pieces_mpi.py``.
Each rank runs the point filters over its share of the pieces; rank 0 compares
what every rank found with the whole's answers, and prints ``ok`` or fails."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import test_point_pieces as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    found = case.across(tc.World(comm))
    gathered = comm.gather(found, root=0)
    deposits = comm.gather(found["deposit"], root=0)
    if comm.rank == 0:
        whole = case.whole_answers()
        for name in ("labels", "outlier"):
            every = {}
            for part in gathered:
                every.update(part[name])
            assert every == whole[name], name
        assert [h for part in gathered for h in part["halos"]] == whole["halos"]
        assert sorted(k for part in gathered for k in part["kept"]) == whole["kept"]
        assert sum(part["copies"] for part in gathered) == whole["copies"]
        for deposit in deposits:            # every rank has the whole grid
            np.testing.assert_allclose(deposit, whole["deposit"], rtol=2e-6)
        print(f"ok: {len(whole['labels'])} points, {len(whole['halos'])} halos, "
              f"{comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
