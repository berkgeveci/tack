# Run under MPI

This page shows how to run a computation written on blocks over many MPI
ranks: on a workstation with `mpiexec`, or on a cluster under Slurm with a GPU
per rank. It is for readers who have a computation on `tc.Blocks` that works
in one process (see the [section's first page](index.md)). After it you can
install mpi4py for Tack, give each rank its blocks and its GPU, choose how
messages travel between ranks, and keep compiled kernels for a large job.

Only the world changes. Everything else in the program stays as it is:

```python
from mpi4py import MPI

from tack import communication as tc

world = tc.World(MPI.COMM_WORLD)
blocks = tc.Blocks(["cpu"] * 2, world)             # two blocks on every rank
```

The ghost example of the section's first page takes `--mpi` to make its world
this way. Its six blocks are shared out over the ranks, and every run gives
the same line, bit for bit:

```console
$ mpiexec -n 3 python docs/examples/comm_ghosts.py --mpi --check
3 rank(s), 6 blocks: max 0.492189, sha256 4d622496db333f4d
```

## Install mpi4py

MPI is optional in Tack. The `mpi` extra installs mpi4py, which builds against
the MPI already on your system:

```bash
uv sync --extra mpi                                  # in a clone of Tack, with your other extras
pip install -e 'packages/tack-core[mpi]'             # or, in another environment
```

The build needs your MPI's compiler wrapper, `mpicc`, on the path. On a
cluster, load the MPI module first, so mpi4py links against the MPI the
cluster's network uses. On a workstation, Open MPI or MPICH from the system's
packages works. Tack's own continuous integration uses Open MPI from apt.

## Make a world from a communicator

`tc.World(comm)` spans the ranks of an mpi4py communicator. It is usually
`MPI.COMM_WORLD`, but any communicator works, including one made by
`comm.Split`. `world.rank` and `world.size` are this rank and the number of
ranks.

Tack communicates on a duplicate of `comm`, made when the world is made. Its
messages therefore never meet your application's messages, whatever tags you
use. Make one `World` for a run and pass it to every `Blocks`.

Tack imports no MPI module. You import mpi4py and make the communicator, which
lets you choose when MPI starts. That matters for GPUs, below.

## Share the blocks out over the ranks

Each rank makes only its own blocks. Global ids run over rank 0's blocks
first, then rank 1's, as [Blocks and worlds](blocks-and-worlds.md) explains.
To share a fixed number of blocks over any number of ranks, give each rank a
consecutive share:

```python
BLOCKS = 12
share = range(world.rank * BLOCKS // world.size, (world.rank + 1) * BLOCKS // world.size)
blocks = tc.Blocks(["cpu"] * len(share), world)
assert blocks.local == share                       # global id g is part g of the problem
```

Ranks may hold different numbers of blocks, and a rank may hold none.
Collectives combine in order of global id, so results do not depend on the
share-out. The rest of the program loops over `blocks.local` as before.

## Give each rank its own GPU

On a node with several GPUs, each rank usually drives one. Make that device
**before** importing `mpi4py.MPI`. A GPU-aware MPI uses the device that is
current when MPI starts, and importing `mpi4py.MPI` starts it. The rank's
number on its node comes from the launcher's environment, since MPI has not
started yet:

```python
import os

import tack

# Slurm's srun sets SLURM_LOCALID; Open MPI's mpiexec sets OMPI_COMM_WORLD_LOCAL_RANK.
local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))
device = tack.set_default_device(f"cuda:{local_rank}")

from mpi4py import MPI                             # MPI starts here, on that device

from tack import communication as tc

world = tc.World(MPI.COMM_WORLD, gpu_aware=True)
blocks = tc.Blocks([device], world)
```

Use `"hip:N"` or `"level_zero:N"` for AMD or Intel GPUs. A rank that moves to
another device after MPI has started still works, but more slowly. With Cray
MPICH on Perlmutter, such a rank's transfers within a node ran at about
7 GB/s instead of 22. Across nodes they can fail.

Under Slurm, let every rank see all of its node's GPUs and choose its own, as
above. Cray MPICH copies between GPUs on one node through CUDA IPC, which needs
the GPUs visible to both ranks. So bind with `--gpu-bind=none`, not
`--gpus-per-task=1`:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
export MPICH_GPU_SUPPORT_ENABLED=1                 # Cray MPICH: let MPI read device memory
srun --gpu-bind=none python run.py
```

## Choose how messages travel between ranks

Messages between blocks of one rank never touch MPI. Messages between ranks
travel one of two ways.

- **Staged**, the default. For each pair of ranks, the sender copies all the
  arrays of an exchange into one host buffer and sends it. The receiver
  copies the arrays to its blocks' devices. This works with any MPI.
- **GPU-aware**, with `tc.World(comm, gpu_aware=True)`. Tack hands MPI each
  array itself, through DLPack, and receives into device arrays. This needs an
  MPI built to read and write device memory, and saves the copies through
  host memory.

Some device memory cannot go to MPI as it is. CUDA's and HIP's arrays come
from stream-ordered memory pools, which Cray MPICH cannot export and
MVAPICH-Plus's UCX takes for host memory. For those, Tack copies each array
into device memory the MPI does accept, sends that, and receives into such
memory. This happens on its own. On Perlmutter's A100s with Cray MPICH, such
transfers ran at 22.2 GB/s a rank within a node and 5.9 GB/s across two nodes;
the measurement is recorded in
[vendor workarounds](../design/vendor-workarounds.md).

Choose GPU-aware transfers when your MPI supports them and messages are
large. Stay with staging when it does not, or when you are not sure: the
results are the same either way. `gpu_aware=True` without a communicator
raises `ValueError`, since one process has no ranks to send to.

## Run CPU ranks side by side

Tack's CPU device runs kernels on a pool of threads, one per CPU the process
may run on. When the launcher binds each rank to its own cores, the pools fit
the node. When it does not, every rank's pool uses every CPU, and the ranks
compete. Then set the number of threads per rank yourself:

```bash
TACK_CPU_THREADS=8 mpiexec -n 4 python run.py      # 4 ranks of 8 threads on 32 cores
```

## Keep compiled kernels for a large job

Each rank compiles the kernels it calls, the first time it calls them. Tack
keeps compiled kernels on disk, so a later run starts warm. With thousands of
ranks on a shared file system, looking up one file per kernel per rank is
slow. A *bundle* avoids it: one file that holds every compiled kernel, which
each rank reads once.

1. Warm the cache with a small run, on one rank or a few, with
   `TACK_CACHE=write` and `TACK_CACHE_DIR` set to a directory of your own on
   scratch.
2. Pack the directory into a bundle with
   `TACK_CACHE_DIR=... python -m tack.cache pack`.
3. Run the job with the same `TACK_CACHE_DIR`. Under MPI, each rank reads
   the bundle and writes nothing. A kernel the bundle lacks is compiled in
   memory.

[Keep compiled kernels between runs](../core/kernel-cache.md) covers the
settings, how Tack recognizes an MPI launch, and how to have rank 0 read the
bundle and broadcast it.

## Check your MPI setup

Tack's tests run every pattern of this section on three MPI ranks, staged
and GPU-aware, with CPU arrays:

```bash
uv run pytest packages/tack-core/tests/test_communication.py
```

Without mpi4py or `mpiexec`, the MPI tests are skipped. Set
`TACK_REQUIRE_MPI=1` to make them fail instead, so a missing MPI does not pass
unnoticed.

## Details

- Between ranks, an exchange first sends every rank a header with the names,
  element types and shapes of the arrays it will receive. The arrays follow as
  nonblocking sends and receives under one tag, on the world's duplicate
  communicator. Every request has completed when `exchange` returns.
- Every call on blocks is collective. A rank that raises before a call
  communicates leaves the others waiting in it. In a batch job, end the run
  with `comm.Abort()` when a call raises.
- `World(comm)` accepts any object with mpi4py's methods `Dup`, `Get_rank`,
  `Get_size`, `allgather`, `alltoall`, `Isend` and `Irecv`.

## Next

- See also [Filters over pieces](../visualization/over-pieces.md), in the
  layer above: Tack's filters over a dataset in pieces run over MPI ranks the
  same way.
- [Choose a device](../core/choose-a-device.md) lists the device specs a
  rank can use.
