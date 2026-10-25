"""``test_temporal``'s check under MPI: ``mpiexec -n 2 python _temporal_mpi.py``. The
steps' pieces are shared out over the ranks, each rank advects its own seeds
through them (each point probed on the rank whose cells hold it), and its
particles must be where advection through the whole steps puts them; rank 0
prints ``ok``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import test_temporal as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    got, want = case.advect_across(tc.World(comm))
    np.testing.assert_allclose(got.world_positions(), want.world_positions(), atol=1e-5)
    np.testing.assert_array_equal(got.fields["status"].values.numpy(),
                                  want.fields["status"].values.numpy())
    done = comm.gather(True, root=0)
    if comm.rank == 0 and all(done):
        print(f"ok: {comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
