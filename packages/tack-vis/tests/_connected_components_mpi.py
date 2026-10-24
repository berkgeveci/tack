"""``test_connected_components``' check under MPI: ``mpiexec -n 2 python
_connected_components_mpi.py``. Each rank labels its share of the pieces; rank 0
compares every cell's region with the whole's, and prints ``ok`` or fails."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_connected_components as case
from mpi4py import MPI

import tack.dataset as td
from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    gathered = comm.gather(case.components_across(tc.World(comm)), root=0)
    if comm.rank == 0:
        data, lo, hi = case.lobes()
        blobs = td.threshold(data, "tangle", lo, hi)
        whole = dict(zip(blobs.global_ids.cells.numpy().tolist(),
                         case._regions(td.connected_components(blobs))[0].tolist()))
        across = {}
        for part in gathered:
            for gid, r in part.items():
                assert across.setdefault(gid, r) == r, f"cell {gid} in two regions"
        assert across == whole
        print(f"ok: {len(across)} cells, {max(whole.values()) + 1} regions, "
              f"{comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
