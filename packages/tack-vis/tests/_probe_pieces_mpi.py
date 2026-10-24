"""``test_probe_pieces``' check under MPI: ``mpiexec -n 2 python
_probe_pieces_mpi.py``. Each rank probes its share of the points over its share
of the pieces; rank 0 compares with probing the whole grid, and prints ``ok``
or fails."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import test_probe_pieces as case
from mpi4py import MPI

import tack.dataset as td
from tack import communication as tc


def main():
    comm = MPI.COMM_WORLD
    gathered = comm.gather(case.probed_across(tc.World(comm)), root=0)
    if comm.rank == 0:
        whole = case.source()
        expected = td.probe(whole, case.points(whole))
        for name in ("RTData", "c", "v", "valid"):
            want = expected.fields[name].values.numpy()
            for rows, values in gathered:
                np.testing.assert_array_equal(np.asarray(values[name]), want[rows],
                                              err_msg=name)
        print(f"ok: {sum(len(r) for r, _ in gathered)} points on {comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
