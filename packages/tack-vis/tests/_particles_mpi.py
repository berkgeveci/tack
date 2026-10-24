"""``test_particles_across_pieces``' check under MPI: ``mpiexec -n 2 python
_particles_mpi.py``. Each rank advects from its share of the seeds over its share
of the pieces; rank 0 compares with one process given all of both, and prints
``ok`` or fails."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_particles_across_pieces as case
from mpi4py import MPI

from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    gathered = comm.gather(case.particles_across(tc.World(comm)), root=0)
    if comm.rank == 0:
        alone = case.particles_across(tc.World())
        particles, lines = {}, {}
        for p, ln in gathered:
            assert not set(p) & set(particles), "a particle on two ranks"
            particles.update(p)
            for s, v in ln.items():
                lines.setdefault(s, []).extend(v)
        assert particles == alone[0], (particles, alone[0])
        assert {s: sorted(v) for s, v in lines.items()} == alone[1]
        print(f"ok: {len(particles)} particles on {comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
