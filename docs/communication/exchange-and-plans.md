# Exchange data and refresh ghosts with plans

This page shows how to move Tack arrays between blocks. It is for readers who
have blocks (see [Blocks and worlds](blocks-and-worlds.md)) and need each
block to see some of its neighbours' data. After it you can send any arrays
from any block to any block with `exchange`, find each block's neighbours from
their bounding boxes, and keep an exchange that repeats every step as a
`Plan`, which is how ghost cells are refreshed.

<!-- tack-check: exchange -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")

blocks = tc.Blocks(["cpu"] * 4)

# Block g covers [g, g + 1] on a line and holds g + 3 values: 10 g, 10 g + 1, ...
values = {g: tack.array(10.0 * g + np.arange(g + 3)) for g in blocks.local}
boxes = {g: ((float(g),), (g + 1.0,)) for g in blocks.local}
neighbours = blocks.link_by_bounds(boxes)
print(neighbours)
# {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}

# Each block sends its first value to the block before it, and its last value
# with a vector of its minimum, maximum and mean to the block after it.
outgoing = {}
for g in blocks.local:
    host = values[g].numpy()
    outgoing[g] = {}
    for n in neighbours[g]:
        if n < g:
            outgoing[g][n] = {"edge": tack.array(host[:1])}
        else:
            stats = [[host.min(), host.max(), host.mean()]]
            outgoing[g][n] = {"edge": tack.array(host[-1:]),
                              "stats": tack.array(stats, tack.vec3(tack.f64))}
incoming = blocks.exchange(outgoing)                 # {destination: {source: message}}

for source, message in incoming[2].items():
    print(source, {name: a.numpy().tolist() for name, a in message.items()})
# 1 {'edge': [13.0], 'stats': [[10.0, 13.0, 11.5]]}
# 3 {'edge': [30.0]}
assert incoming[2][3]["edge"].numpy()[0] == 30.0
assert incoming[2][1]["stats"].dtype == tack.vec3(tack.f64)
```

## Send messages with exchange

`blocks.exchange(outgoing)` delivers messages between any blocks, on any
process. Its argument and its result are nested dictionaries:

- `outgoing` is `{source: {destination: message}}`. The sources are this
  process's blocks. A destination may be any block, on any process, the
  source included.
- A **message** is a dictionary of names to Tack arrays, each on the source
  block's device. A message may be empty.
- The result is `{destination: {source: message}}`, with an entry for every
  one of this process's blocks, in order of global id. A block's messages are
  in order of source. A block that receives nothing gets an empty dictionary.

Neighbour exchanges, all-to-all exchanges and sends to a chosen block are all
this one call. When it returns, every message to this process's blocks has
arrived.

Arrays arrive as they were sent. Their element types and shapes are kept, so
vector and matrix arrays (`tack.vec3`, `tack.mat(2, 3)`) and
multidimensional arrays need no packing. Each received array is on the
destination block's device. Where that device can already address the sent
memory, the receiver gets the sent array itself. A sender must therefore not
write to an array after sending it. Make a copy first if it must keep
changing.

A structure-of-arrays array (`tack.soa`) cannot be sent as one array. Send
its components under names of their own.

## Find neighbours by their boxes

Most algorithms send to neighbours, not to every block.
`blocks.link_by_bounds(boxes, margin=0.0)` finds them from bounding boxes.
`boxes` is `{g: (lower, upper)}` for this process's blocks, two corners of any
one dimension. The result is `{g: [neighbours]}`: every block whose box
overlaps or touches block `g`'s box, grown by `margin` on every side.

With a margin of 0, blocks that share a face are neighbours, which is what a
decomposition into abutting parts needs. With a margin of `r`, every block
within distance `r` of a box is a neighbour, which is what a search radius
needs. Like every call on blocks, `link_by_bounds` is collective: it gathers
every process's boxes.

## Keep an exchange as a plan

Most exchanges repeat. A stencil code sends the same edge cells to the same
neighbours before every step, and only the values change. A `Plan` keeps such
an exchange: for each pair of blocks, which elements the sender reads and
where the receiver writes them. You build it once and run it every step.

```python
plan = tc.Plan(blocks, sends, receives)
plan.run(u)                                       # every step
```

`sends` is `{g: {destination: indices}}` and `receives` is
`{g: {source: indices}}`, for this process's blocks. The indices are element
indices into the block's array, as lists, NumPy arrays or integer Tack
arrays on the block's device. For a multidimensional array they are flat,
row-major indices. The rule that ties the two together is:

> The k-th element a sender reads for a destination goes to the k-th element
> that destination writes from that sender.

Every send must have a receive of the same length on the other block, and
the other way round. The plan checks this across every process when it is
made, and raises `ValueError` naming the two blocks if not.

`plan.run(values)` gathers the sent elements, exchanges them and writes them
into place:

- `values` is `{g: array}` for every block that sends or receives.
- `{g: {name: array}}` moves several arrays in one exchange. Every block must
  use the same names, and each name one element type on every block.
- Received elements are written in place. Every other element is left as it
  was.
- Elements move whole: for a `vec3` array, an index moves three components.

A block may send to itself. This is what a periodic domain needs, where a
block can be its own neighbour. What a block sends itself is read before
anything is written.

## Refresh ghost rows of a periodic grid

The example below cuts a 24 by 16 grid into strips of rows. Each block owns
its rows and stores one ghost row above and below them. The grid is periodic
in y, so the last block's neighbour above is block 0. With one block, the
block is its own neighbour above and below, and sends to itself.

<!-- tack-check: plan -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")

NX, NY = 16, 24                                      # the whole grid's columns and rows


@tack.kernel
def diffuse(u, out):
    # Rows 0 and u.shape[0] - 1 are ghosts; x is periodic within the block.
    for i, j in tack.ndindex((1, u.shape[0] - 1), u.shape[1]):
        left = u[i, (j + u.shape[1] - 1) % u.shape[1]]
        right = u[i, (j + 1) % u.shape[1]]
        out[i, j] = u[i, j] + 0.2 * (left + right + u[i - 1, j] + u[i + 1, j] - 4.0 * u[i, j])


def row(r):
    """The flat element indices of row r of a block's array."""
    return list(range(r * NX, (r + 1) * NX))


def ghost_plan(blocks, rows):
    """Each block's first and last owned rows into the ghost rows of the blocks
    below and above it, periodic in y."""
    sends = {g: {} for g in blocks.local}
    receives = {g: {} for g in blocks.local}
    for g in blocks.local:
        below, above = (g - 1) % blocks.count, (g + 1) % blocks.count
        # What g sends, in this order: its first owned row down, its last up.
        sends[g].setdefault(below, []).extend(row(1))
        sends[g].setdefault(above, []).extend(row(rows))
        # What g receives, in the order its sources send: the block above sends
        # its first row down, into g's top ghost row; the block below sends its
        # last row up, into g's bottom ghost row.
        receives[g].setdefault(above, []).extend(row(rows + 1))
        receives[g].setdefault(below, []).extend(row(0))
    return tc.Plan(blocks, sends, receives)


def simulate(count, steps=50):
    blocks = tc.Blocks(["cpu"] * count)
    rows = NY // count                               # rows each block owns
    y, x = np.mgrid[0:NY, 0:NX]
    start = np.exp(-((x - NX / 2) ** 2 + (y - NY / 3) ** 2) / 8.0)
    u, v, owner = {}, {}, {}
    for g in blocks.local:
        u[g] = tack.array(np.pad(start[g * rows:(g + 1) * rows], ((1, 1), (0, 0))))
        v[g] = tack.zeros((rows + 2, NX), tack.f64)
        owner[g] = tack.array(np.full((rows + 2, NX), g, np.int32))
    plan = ghost_plan(blocks, rows)
    plan.run({g: {"u": u[g], "owner": owner[g]} for g in blocks.local})   # two arrays
    for step in range(steps):
        for g in blocks.local:
            diffuse(u[g], v[g])
        u, v = v, u
        plan.run(u)                                  # one array
    grid = np.concatenate([u[g].numpy()[1:rows + 1] for g in blocks.local])
    return owner, grid


owner, three = simulate(3)
print(owner[0].numpy()[[0, 1, 8, 9], 0])             # ghost, owned, owned, ghost
# [2 0 0 1]
assert owner[0].numpy()[0, 0] == 2 and owner[0].numpy()[9, 0] == 1

# One block is its own neighbour above and below: it sends to itself.
owner, one = simulate(1)
assert np.array_equal(three, one)                    # three blocks or one: the same bits
```

The first run of the plan moves two arrays at once. The `owner` array shows
where each ghost row came from: block 0's bottom ghost row holds block 2's
last row, across the periodic boundary, and its top ghost row holds block 1's
first row. After 50 steps, three blocks and one block give the same grid, bit
for bit.

Notice how `ghost_plan` lists the receives. When there are only one or two
blocks, a block's neighbour below and above are the same block, and both rows
go in one list. The receiver must list the rows in the order the sender lists
them. That is why `g` lists what its neighbour above sends down before what
its neighbour below sends up: each neighbour sends its row down first.

## Details

- A plan holds its index arrays on each block's device, so running it
  launches only gather and scatter kernels and one exchange.
- Building a plan is collective, since it checks lengths across every
  process. Build it once, outside the time loop.
- `Plan.run` raises `TypeError` if a name's element type differs between the
  sender and the receiver.
- A plan does not know what the indices mean. Ghost cells, ghost points,
  shared faces and halo regions of any width are all lists of indices.
- Every exception is listed in the
  [runtime contract](../contracts/runtime-api.md#communication-between-blocks).

## Next

- [Route points to blocks](routing.md) sends values to whichever block holds
  a point, when you do not know the destination in advance.
- See also [Data in pieces](../data/pieces.md): Tack's ghost cells for a
  dataset in pieces are plans like these, made for you.
