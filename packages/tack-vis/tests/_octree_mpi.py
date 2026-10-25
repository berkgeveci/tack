"""``test_octrees``' check under MPI: ``mpiexec -n 2 python _octree_mpi.py``. Each
rank holds a share of the trees; every rank's statistics and entropy of the
octree must be the whole's, to rounding, and rank 0 prints ``ok``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import test_octrees as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    pieces, entropy, whole, whole_entropy = case.across(tc.World(comm))
    assert pieces.n == whole.n
    np.testing.assert_allclose([pieces.mean, pieces.weight, pieces.population_variance,
                                entropy],
                               [whole.mean, whole.weight, whole.population_variance,
                                whole_entropy], rtol=1e-6)
    done = comm.gather(True, root=0)
    if comm.rank == 0 and all(done):
        print(f"ok: {whole.n} leaves, {comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
