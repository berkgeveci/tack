# Communicate between blocks

The algorithms that take a `PartitionedDataSet` -- ghost cells, global ids,
probes and particles across pieces, regions, sorts by voxel or label -- are
written on `tack.communication`, a small layer for moving Tack arrays between
*blocks*. This chapter is for writing one of your own: an algorithm whose
answer on a piece needs another piece's data.

The model is DIY's (blocks, links, exchanges, collectives), with Tack arrays
as the messages and plain dictionaries in place of callbacks. It runs the
same in one process, with blocks on any of its devices, and over MPI ranks:
an algorithm written on `Blocks` never asks which.

## Blocks and worlds

A **block** is a unit of work on one device, with a **global id**: in
`tack.dataset`, a piece. `Blocks(devices, world)` makes this process's blocks,
one per device given (several may share a device); global ids run over every
rank's blocks, contiguous by rank, as VTK numbers a partitioned dataset's
pieces. `blocks.local` is the range of this rank's, `blocks.count` the number
in all, `blocks.rank_of(g)` and `blocks.device(g)` where a block lives.

A **world** is the transport. `World()` is this process alone;
`World(comm)` spans the ranks of an mpi4py communicator (Tack imports no MPI
module itself). Every method takes and returns dictionaries keyed by global
id, holding an entry for each of this rank's blocks. Every call is
**collective**: each rank makes it, in the same order, even a rank with
nothing to send -- the `Blocks`, `Plan` and `Routing` constructors included,
which gather what every rank has.

**Collectives** combine small host values (numbers, NumPy arrays, anything
picklable for `all_gather`) over every block: `all_reduce` by `"sum"`,
`"min"` or `"max"`, `all_gather` into a list by global id, and
`exclusive_scan`, each block's offset in a global numbering. They combine in
order of global id, whichever rank holds which block, so a floating-point
sum has the same bits however the blocks are shared out.

**`exchange`** sends messages -- each a dict of names to Tack arrays, on the
sending block's device -- from any block to any block. It takes
`{source: {destination: message}}` and returns
`{destination: {source: message}}` for every block on this rank (an empty
dict where nothing came), each message's arrays on the destination block's
device. A sent array belongs to the receiver afterwards: where the receiver
can address its memory, it gets that array itself, so the sender must not
write to it again. `link_by_bounds` finds each block's neighbours by their
boxes, touching or within a margin.

<!-- tack-check: blocks -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")

world = tc.World()                                   # tc.World(MPI.COMM_WORLD) under mpiexec
blocks = tc.Blocks(["cpu", "cpu"], world)            # two blocks on this rank

# Block g holds g + 3 values: 10 g, 10 g + 1, ...
values = {g: tack.array(10.0 * g + np.arange(g + 3)) for g in blocks.local}

total = blocks.all_reduce({g: values[g].sum() for g in blocks.local})
largest = blocks.all_reduce({g: values[g].max() for g in blocks.local}, op="max")
sizes = blocks.all_gather({g: values[g].size for g in blocks.local})
starts, count = blocks.exclusive_scan({g: values[g].size for g in blocks.local})

every = [10.0 * g + np.arange(g + 3) for g in range(blocks.count)]
assert total == sum(v.sum() for v in every) and largest == every[-1].max()
assert sizes == [g + 3 for g in range(blocks.count)] and count == sum(sizes)
assert all(starts[g] == sum(sizes[:g]) for g in blocks.local)

# Each block sends its first value to the block before it, and its last to
# the block after it: neighbours from boxes, one exchange for every block.
boxes = {g: ((float(g),), (g + 1.0,)) for g in blocks.local}       # unit intervals
neighbours = blocks.link_by_bounds(boxes)                          # {g: [g - 1, g + 1]}
outgoing = {}
for g in blocks.local:
    first, last = values[g].numpy()[[0, -1]]
    outgoing[g] = {n: {"edge": tack.array([first if n < g else last])} for n in neighbours[g]}
incoming = blocks.exchange(outgoing)                 # {destination: {source: message}}

for g in blocks.local:
    for source, message in incoming[g].items():
        expected = every[source][-1] if source < g else every[source][0]
        assert message["edge"].numpy()[0] == expected
print(blocks, "total", total, "of", count, "values")
```

## Plans: ghosts refreshed

Most exchanges repeat: the same elements of the same blocks, every step. A
`Plan` keeps one: for each pair of blocks, the element indices the sender
reads and those the receiver writes, in matching order (checked across
ranks when the plan is made). `plan.run(values)` gathers, exchanges and
scatters in place; `{g: {name: array}}` moves several arrays in one
exchange. A block may send to itself, as a piece in a periodic box is its
own neighbour.

Here a line of cells is cut into blocks of `m` cells, each held with a
ghost cell at either end, and diffused by a kernel that reads a cell's two
neighbours. Before each step, the plan copies each block's edge cells into
its neighbours' ghosts; the blocks then give exactly the whole line's
values. To sample the result, `Routing` sends points to the block that
holds them and their answers back:

<!-- tack-check: plan -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")


@tack.kernel
def diffuse(u, out):
    for i in range(1, u.shape[0] - 1):
        out[i] = u[i] + 0.25 * (u[i - 1] - 2.0 * u[i] + u[i + 1])


world = tc.World()                                   # tc.World(MPI.COMM_WORLD) under mpiexec
blocks = tc.Blocks(["cpu"] * 2, world)
m = 50                                               # cells per block
n = m * blocks.count                                 # cells in all

# Block g owns cells [g m, (g + 1) m), held at 1 .. m with a ghost each side:
# 0 and m + 1, the neighbours' edge cells (or a fixed zero at the ends).
start = np.exp(-((np.arange(n) - n / 3) / 8.0) ** 2)
u = {g: tack.array(np.pad(start[g * m:(g + 1) * m], 1)) for g in blocks.local}
v = {g: tack.zeros(m + 2, tack.f64) for g in blocks.local}

# The ghost exchange: each block sends its edge cells, and the neighbour
# writes them into its ghost. Indices match in order, sends to receives.
sends = {g: {} for g in blocks.local}
receives = {g: {} for g in blocks.local}
for g in blocks.local:
    if g > 0:
        sends[g][g - 1], receives[g][g - 1] = [1], [0]
    if g < blocks.count - 1:
        sends[g][g + 1], receives[g][g + 1] = [m], [m + 1]
ghosts = tc.Plan(blocks, sends, receives)

for step in range(100):
    ghosts.run(u)                                    # ghosts from their owners
    for g in blocks.local:
        diffuse(u[g], v[g])
    u, v = v, u

# The same steps over the whole line, one array: the same bits.
whole, other = tack.array(np.pad(start, 1)), tack.zeros(n + 2, tack.f64)
for step in range(100):
    diffuse(whole, other)
    whole, other = other, whole
whole = whole.numpy()
for g in blocks.local:
    assert np.array_equal(u[g].numpy()[1:m + 1], whole[1 + g * m:1 + (g + 1) * m])

# Sample the result at points each block chooses: each point goes to the
# block whose cells hold it, which answers with its value there.
rng = np.random.default_rng(blocks.local.start)
points = {g: rng.uniform(0.0, n, (5, 1)) for g in blocks.local}
boxes = {g: ((g * m,), ((g + 1) * m,)) for g in blocks.local}


def holds(g, x):                                     # cells [g m, (g + 1) m)
    return (x[:, 0] >= g * m) & (x[:, 0] < (g + 1) * m)


routing = tc.Routing(blocks, points, boxes, holds)
arrived = routing.send()                             # {g: (points, values)}
answers = {}
for g in blocks.local:
    x, _ = arrived[g]
    cells = (np.floor(x[:, 0]) - g * m + 1).astype(np.int64)  # local index, after the ghost
    answers[g] = {"u": tack.array(u[g].numpy()[cells])}
sampled = routing.reply(answers, like={"u": tack.f64})          # in each block's order

for g in blocks.local:
    assert np.array_equal(sampled[g]["u"].numpy(), whole[1 + np.floor(points[g][:, 0]).astype(int)])
print("max", whole.max(), "sampled", sampled[blocks.local.start]["u"].numpy()[:3])
```

## Routing: points to their blocks

`Routing(blocks, positions, boxes, holds)` sends each point to the block
that holds it. Each block gives its points (an `(n, d)` host array), its box
(`None` if it holds nothing), and a function `holds(g, x)` that block `g`
is asked with the points inside its box: 0 where it does not hold a point,
or a rank, `True` being 1. The point goes to the block of the highest rank,
the lowest block on a tie -- so a mesh's cell that holds a point beats one
that holds it only within a tolerance, and a point on a face two pieces
share goes to the lower piece. `routing.destination[g]` is each point's
block, -1 where none holds it.

`send(values)` moves the points, and any values with them, and returns what
arrived at each block, in order of origin block and then of the points
there. `reply(answers, like)` sends an answer for each arrived point back,
in that order, and returns one per original point (zero where the point
went nowhere). Coordinates travel as the bits of f64 values, so the block
decides on exactly the point it was given. `probe` across pieces is this
pattern, and particle advection runs it in rounds, routing each particle
after every stretch it takes.

## Sorting over blocks

`sort_by_key(blocks, keys, values)` sorts every block's integer keys
together, carrying values along: block 0 ends with the smallest keys and
the last block with the largest, each ascending, and **every copy of a key
ends on one block**. That last property is what it is for: matching. Give
each point a key -- a hash of its exact position, a voxel, a halo label, a
global id -- sort, and the copies of each key meet on one block, which
matches or reduces them there; the values say where each came from. It is a
stable sample sort (DIY's): equal keys keep the order of their blocks, then
their order within a block. Heavily repeated keys can leave blocks uneven.

<!-- tack-check: sort -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")

world = tc.World()                                   # tc.World(MPI.COMM_WORLD) under mpiexec
blocks = tc.Blocks(["cpu"] * 3, world)

# Each block's keys, some on several blocks; the value says where each came
# from: (block << 40) | index, the identity Tack's dataset algorithms use.
keys, origin = {}, {}
for g in blocks.local:
    k = np.random.default_rng(g).integers(0, 40, 25 + 5 * g)
    keys[g] = tack.array(k, tack.i64)
    origin[g] = {"origin": tack.array((g << 40) | np.arange(k.size), tack.i64)}

ordered, carried = tc.sort_by_key(blocks, keys, origin)

# Block 0 has the smallest keys, the last block the largest, each ascending,
# and every copy of a key is on one block: count them where they landed.
mine = {g: ordered[g].numpy() for g in blocks.local}
every = blocks.all_gather(mine)
given = [np.random.default_rng(g).integers(0, 40, 25 + 5 * g) for g in range(blocks.count)]
assert np.array_equal(np.concatenate(every), np.sort(np.concatenate(given)))
held = [set(part.tolist()) for part in every]
assert all(not (held[a] & held[b]) for a in range(len(held)) for b in range(a))

# Each value still names its key's block and index.
for g in blocks.local:
    o = carried[g]["origin"].numpy()
    came = [given[s][i] for s, i in zip(o >> 40, o & ((1 << 40) - 1))]
    assert np.array_equal(came, mine[g])
    print("block", g, "holds", mine[g].size, "keys,", mine[g].min(), "to", mine[g].max())
```

Ghost cells match boundary points this way, `generate_global_ids` numbers
matched points, and `voxel_downsample` and `halo_catalog` bring each
voxel's or halo's points together.

## Blocks from a dataset's pieces

An algorithm over a `PartitionedDataSet` makes one block per piece, in the
pieces' order, in the dataset's world:

```python
blocks = tc.Blocks([piece.device for piece in pieces], pieces.world)
mine = dict(zip(blocks.local, pieces))           # global id -> this rank's piece
```

A piece's global id is then its index among every rank's pieces, the
numbering ghost marks, global ids and `statistics` use. Positions are
relative to each piece's origin, so compare boxes and points across pieces
only where the pieces share one, as Tack's algorithms check they do
(`partitioned.common_origin`).

## Over MPI

The examples above run under `mpiexec` unchanged but for the world:

```python
from mpi4py import MPI

world = tc.World(MPI.COMM_WORLD)
```

```bash
mpiexec -n 3 python plan.py
```

Each rank makes only its own blocks (here two each, so three ranks give six
blocks and a longer line); the checks pass on any number of ranks. Messages
between ranks are staged through host memory, or, with
`World(comm, gpu_aware=True)` and an MPI that reads device memory, handed
to MPI as the device arrays themselves. Make each rank's device
(`tack.device(f"cuda:{local_rank}")`) before importing `mpi4py.MPI`, which
starts MPI on the device current then. `World(comm)` communicates on a
duplicate of `comm`, so its messages never meet the application's; make one
per run.

## Run and check

```bash
uv run python docs/examples/validate.py --arch cpu
uv run pytest packages/tack-core/tests/test_communication.py
```

`validate.py` runs this chapter's three examples in one process. The test
suite runs every pattern on threads that stand in for ranks, and under
`mpiexec -n 3` staged and GPU-aware (`--extra mpi`); the reference for each
call is the [API page](../reference/api.md#communication-between-blocks), and
what it guarantees is in the
[Runtime API contract](../contracts/runtime-api.md#communication-between-blocks).
