# Blocks and worlds

Every call in `tack.communication` goes through two objects. A `Blocks` holds
this process's units of work and their devices. A `World` says which processes
there are and moves messages between them. Before writing a computation over
blocks, you decide how many blocks to make and on which devices. Each block is
addressed by its global id, and its arrays live on its device. One rule says
which calls every process must make.

<!-- tack-check: blocks -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")

world = tc.World()                         # this process alone
blocks = tc.Blocks(["cpu", "cpu", "cpu"], world)
print(world, blocks)
# World(rank=0, size=1) Blocks(3 blocks, 3 on rank 0)

# One array per block, made on that block's device.
cells = {g: tack.array(np.full(4, float(g)), device=blocks.device(g)) for g in blocks.local}

# A collective: every block's value, combined over every block.
total = blocks.all_reduce({g: cells[g].sum() for g in blocks.local})
assert list(blocks.local) == [0, 1, 2] and blocks.count == 3
assert blocks.rank_of(2) == 0 and total == 4 * (0 + 1 + 2)
```

## A block is a unit of work on one device

A **block** is the unit you cut a problem into: a part of a grid, a share of
particles, a piece of a dataset. It plays the part of a DIY block, or of one
piece of a VTK partitioned dataset. Each block lives on one device, and its
arrays are on that device. A **global id** names each block uniquely over
every process.

`Blocks(devices, world)` makes this process's blocks, one for each entry of
`devices`. An entry is a `tack.Device` or a device spec such as `"cpu"`,
`"cuda:1"` or `"hip"` (see [Choose a device](../core/choose-a-device.md)).
Several blocks may share one device. If `world` is left out, it is
`World()`, this process alone.

How many blocks to make is your choice. One block per device is the common
case. More blocks than devices is useful too: an algorithm developed with
four blocks on one CPU exercises every exchange it will make on four GPUs.

## Global ids number every process's blocks

Global ids run over every process's blocks, contiguous by process. Rank 0's
blocks come first, then rank 1's, and so on, as VTK numbers the pieces of a
partitioned dataset across ranks. Processes may hold different numbers of
blocks. With three ranks holding one, two and three blocks:

| rank | blocks it holds | `blocks.local` |
|---|---|---|
| 0 | 0 | `range(0, 1)` |
| 1 | 1, 2 | `range(1, 3)` |
| 2 | 3, 4, 5 | `range(3, 6)` |

Four attributes and methods describe the numbering:

| name | gives |
|---|---|
| `blocks.local` | the range of this process's global ids |
| `blocks.count` | the number of blocks over every process |
| `blocks.rank_of(g)` | the rank that holds block `g`, for any block |
| `blocks.device(g)` | the device of block `g`, for this process's blocks only |

`blocks.device(g)` raises `ValueError` for another process's block, since a
process knows only its own devices. A process with no blocks is allowed:
`Blocks([], world)` takes part in every call and holds an empty `local`.

## Arrays live on their block's device

Every array you give a call for block `g` must be on `blocks.device(g)`. Make
it there with `device=blocks.device(g)`, as the example does, or with
`array.to_device(...)`. A call refuses an array on another device with a
`ValueError` that names both.

Every array you receive for block `g` is on `blocks.device(g)` too. In one
process with blocks on two GPUs, a message from a block on `cuda:0` to a
block on `cuda:1` arrives as a copy on `cuda:1`. Where the destination's
device can already address the sender's memory, as two blocks on one CPU can,
the receiver gets the sent array itself. So a sender must not write to an
array after sending it.

One process can drive several GPUs this way:

```python
blocks = tc.Blocks(["cuda:0", "cuda:1"])          # two blocks, one per GPU
```

## A world is the set of processes

A **world** is the transport between processes. It plays the part of VTK's
`vtkMultiProcessController`.

- `tc.World()` is this process alone, rank 0 of 1. Its blocks may be on any
  of its devices, and messages between them are array copies.
- `tc.World(comm)` spans the ranks of `comm`, an mpi4py communicator.

```python
from mpi4py import MPI

world = tc.World(MPI.COMM_WORLD)
blocks = tc.Blocks(["cpu"] * 2, world)            # two blocks on every rank
```

`world.rank` and `world.size` are this process's rank and the number of
ranks. Tack imports no MPI module itself. You make the communicator, and Tack
communicates on a duplicate of it, so its messages never meet your
application's. Make one `World` for a run and pass it to every `Blocks`.
Anything with the methods Tack uses (`Dup`, `Get_rank`, `Get_size`,
`allgather`, `alltoall`, `Isend` and `Irecv`) works in place of an mpi4py
communicator.

How messages travel between ranks, staged through host memory or handed to a
GPU-aware MPI as device arrays, is on [Run under MPI](mpi.md).

## Every call is collective

Every call on blocks is **collective**: every process of the world makes it,
in the same order. That includes the constructors of `Blocks`, `Plan` and
`Routing`, which gather what every process has. It includes a process with
nothing to send, which still calls `exchange` with an empty dictionary. A
process that skips a call, or makes calls in another order, leaves the others
waiting.

The calls are:

| call | what it does | page |
|---|---|---|
| `Blocks(...)` | make this process's blocks | this page |
| `exchange`, `link_by_bounds`, `Plan(...)`, `Plan.run` | send messages, find neighbours, keep and rerun an exchange | [Exchange and plans](exchange-and-plans.md) |
| `Routing(...)`, `send`, `reply` | send points to the blocks that hold them | [Route points](routing.md) |
| `all_reduce`, `all_gather`, `exclusive_scan`, `sort_by_key` | combine and sort over every block | [Sort and reduce](sort-and-reduce.md) |

Kernels, by contrast, are not collective. Each process launches kernels on
its own blocks' arrays whenever it likes, as the ghost example in
[Exchange and plans](exchange-and-plans.md#refresh-ghost-rows-of-a-periodic-grid)
does between plan runs.

## Arguments are dictionaries keyed by global id

Every call takes and returns dictionaries keyed by global id, with entries for
this process's blocks only. Naming another process's block where one of this
process's is needed raises `ValueError`.

- Collectives (`all_reduce`, `all_gather`, `exclusive_scan`), `Routing` and
  `sort_by_key` need an entry for every one of this process's blocks.
- `exchange` and `Plan` need entries only for blocks that send or receive.
- Results have an entry for every one of this process's blocks, in order of
  global id.

So the usual way to write a step is a dictionary comprehension over
`blocks.local`:

```python
counts = {g: particles[g].size for g in blocks.local}
starts, total = blocks.exclusive_scan(counts)
```

## Details

- The numbering is fixed when `Blocks` is made. To change how many blocks a
  process holds, make a new `Blocks`, on every process.
- Some mistakes are found on one process only, before the call communicates.
  That process raises, and the others wait in the call. Under MPI, end the
  run when a call raises, for example with `comm.Abort()`.
- Every exception each call raises is listed in the
  [runtime contract](../contracts/runtime-api.md#communication-between-blocks).

## Next

- [Exchange data and refresh ghosts with plans](exchange-and-plans.md) sends
  arrays between blocks, the first thing most algorithms do.
- [Run under MPI](mpi.md) puts blocks on many ranks, one GPU each.
