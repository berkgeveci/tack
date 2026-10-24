"""Run ``_communication_cases`` under MPI: ``mpiexec -n 3 python _communication_mpi.py``.

Each case runs on a communicator of its first ranks, through
``tack.communication.World`` (``--gpu-aware`` passes arrays to MPI through
DLPack instead of staging them). Rank 0 prints ``ok <case>`` or
``FAILED <case>: <why>`` per case and exits with the number of failures.
``test_communication.py`` launches it.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _communication_cases as cases
from mpi4py import MPI

from tack import communication as tc


def main():
    gpu_aware = "--gpu-aware" in sys.argv
    staged_copies = []
    if "--not-direct" in sys.argv:
        # As HIP's pool memory: MPI may not take the arrays as they are, so the
        # exchange copies them into allocate_mpi_array's memory first.
        from tack.runtime.cpu import CPUBackend

        CPUBackend.mpi_direct = lambda self, buffer: getattr(buffer, "_for_mpi", False)
        plain = CPUBackend.allocate_mpi_array

        def counted(self, dtype, shape):
            buffer = plain(self, dtype, shape)
            buffer._for_mpi = True
            staged_copies.append(shape)
            return buffer
        CPUBackend.allocate_mpi_array = counted
    comm = MPI.COMM_WORLD
    failures = 0
    for name, (ranks, body, check) in cases.CASES.items():
        if ranks > comm.size:
            if comm.rank == 0:
                print(f"FAILED {name}: needs {ranks} ranks, have {comm.size}", flush=True)
                failures += 1
            continue
        sub = comm.Split(0 if comm.rank < ranks else MPI.UNDEFINED, comm.rank)
        if sub != MPI.COMM_NULL:
            try:
                outcome = (body(tc.World(sub, gpu_aware=gpu_aware)), None)
            except Exception:
                outcome = (None, traceback.format_exc())
            gathered = sub.gather(outcome, root=0)
            if sub.rank == 0:
                errors = [error for _, error in gathered if error]
                try:
                    if errors:
                        raise AssertionError(errors[0])
                    check([result for result, _ in gathered])
                    print(f"ok {name}", flush=True)
                except Exception:
                    print(f"FAILED {name}: {traceback.format_exc()}", flush=True)
                    failures += 1
            sub.Free()
        comm.Barrier()
    if "--not-direct" in sys.argv and not staged_copies:
        print(f"FAILED rank {comm.rank}: no array went through MPI memory", flush=True)
        failures += 1
    failures = comm.allreduce(failures) if "--not-direct" in sys.argv else failures
    failures = comm.bcast(failures, root=0)
    sys.exit(min(failures, 100))


if __name__ == "__main__":
    main()
