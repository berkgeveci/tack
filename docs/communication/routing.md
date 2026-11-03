# Route points to blocks

This page shows how to send points to the blocks that hold them when you do
not know those blocks in advance. A probe asks for values at points anywhere
in a domain, and a particle that moves must go to the block whose part of the
domain it entered. `Routing` does both. It is for readers who know
[blocks](blocks-and-worlds.md) and [exchanges](exchange-and-plans.md). After
it you can route points by a test of your own, send values with them, return
answers to where the points came from, and move particles between blocks in
rounds.

<!-- tack-check: probe -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")

blocks = tc.Blocks(["cpu"] * 3)
TOLERANCE = 0.01

# Block g holds the strip g <= x < g + 1 of the rectangle [0, 3) x [0, 1).
# Its box is grown by the tolerance, so points just outside are asked too.
boxes = {g: ((g - TOLERANCE, 0.0), (g + 1 + TOLERANCE, 1.0)) for g in blocks.local}


def holds(g, points):
    """2 where block g holds a point, 1 where it is within the tolerance, else 0."""
    x = points[:, 0]
    inside = (x >= g) & (x < g + 1)
    near = (x >= g - TOLERANCE) & (x < g + 1 + TOLERANCE)
    return np.where(inside, 2, np.where(near, 1, 0))


# Each block asks about five points anywhere in the rectangle, and one outside it.
rng = np.random.default_rng(7)
positions = {g: np.vstack([rng.uniform((0, 0), (3, 1), (5, 2)), [[5.0, 0.5]]])
             for g in blocks.local}
routing = tc.Routing(blocks, positions, boxes, holds)
print(routing.destination[0])
# [ 1  2  0  0  2 -1]

# Each point goes to its block, which answers with 100 times its id plus x.
arrived = routing.send()                             # {g: (points, {name: array})}
answers = {}
for g in blocks.local:
    points, values = arrived[g]
    answers[g] = {"value": tack.array(100.0 * g + points[:, 0])}
back = routing.reply(answers, like={"value": tack.f64})

expected = 100.0 * np.floor(positions[0][:, 0]) + positions[0][:, 0]
expected[5] = 0.0                                    # the point no block holds
assert np.array_equal(routing.destination[0][:5], np.floor(positions[0][:5, 0]))
assert np.array_equal(back[0]["value"].numpy(), expected)
```

## How a point finds its block

`tc.Routing(blocks, positions, boxes, holds)` decides where every point goes,
in three steps:

1. Each point is sent to every block whose box contains it. A box is
   widened by a relative slack of one millionth of its largest side, so a
   point on a box's face is asked of the block.
2. Each block asked calls `holds(g, points)` on the points it received. The
   function returns one number per point: 0 where the block does not hold
   the point, and a positive *rank* where it does. `True` counts as 1.
3. The point goes to the block that gave the highest rank. On a tie, it goes
   to the block with the lowest global id. Where every rank is 0, the point
   goes nowhere.

`routing.destination` is `{g: array}`: for each of block `g`'s points, the
block it goes to, or -1 where no block holds it.

The arguments are, for each of this process's blocks:

| argument | per block | meaning |
|---|---|---|
| `positions` | an `(n, d)` NumPy array | the points this block asks about; `n` may be 0 |
| `boxes` | `(lower, upper)`, or `None` | the region the block may hold points in; `None` holds nothing |
| `holds` | one function for every block | `holds(g, points)` on an `(m, d)` float64 array, returning `m` ranks |

Ranks let a block say how well it holds a point. In the example, a block
gives 2 to a point inside its strip and 1 to a point within the tolerance
outside it. A point at x = 1.005 is near block 0 and inside block 1, so it
goes to block 1. A point just outside every strip, at x = 3.005, is still
taken by block 2, the only block near it. Rank 2 beats rank 1 whatever the
blocks' ids. Tack's cell locators rank the same way when probing a dataset in
pieces: a cell that holds a point beats one that is merely close to it.

The tie rule gives every point exactly one destination. If `holds` returned
only `True` and `False` on closed strips, a point at x = 1 would be held by
blocks 0 and 1, and it would go to block 0.

Positions travel as the bits of their float64 values. The block that decides
sees exactly the coordinates that were sent, on every device, so `holds`
never disagrees with itself across a round trip.

## Send values with the points

`routing.send(values)` moves the points to their destinations, with any
arrays of values that go with them. `values` is `{g: {name: array}}`: one
element per point of block `g`, on its device. It may be left out.

It returns `{g: (points, values)}` for each of this process's blocks:

- `points` is an `(m, d)` NumPy array of the points that arrived.
- `values` is `{name: array}`, the values that came with them, on the block's
  device.
- Points arrive in order of the block they came from, then in their order on
  that block.
- A block that receives no points gets an empty array of points and an empty
  dictionary of values.

## Send answers back

`routing.reply(answers, like)` returns one answer per point to the block it
came from. `answers` is `{g: {name: array}}`, one element for each point that
arrived at block `g`, in the order `send` gave them. `like` is
`{name: element type}`, which tells blocks that received no points what to
make.

The result is `{g: {name: array}}`, one element for each of block `g`'s
original points, in their original order. A point that went nowhere gets
zero. Call `send` before `reply`. A routing remembers what `send` delivered
and answers in that order.

## Move particles between blocks in rounds

A particle that moves leaves its block. To keep each particle on the block
that holds it, alternate two steps: advance every block's particles, then
route them to their new blocks with their state. Each round makes a new
`Routing` from the new positions.

<!-- tack-check: particles -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")

WIDTH = 3.0                                          # x is periodic on [0, 3)
POINT = tack.vec2(tack.f64)


@tack.kernel
def move(x, velocity, dt, width):
    for i in range(x.shape[0]):
        p = x[i] + velocity[i] * dt
        p[0] = p[0] - width * tack.floor(p[0] / width)  # wrap x into [0, width)
        x[i] = p


def holds(g, points):
    return (points[:, 0] >= g) & (points[:, 0] < g + 1)


blocks = tc.Blocks(["cpu"] * 3)
boxes = {g: ((float(g), 0.0), (g + 1.0, 1.0)) for g in blocks.local}

# Ten particles a block, each starting in its own block's strip.
rng = np.random.default_rng(1)
x, velocity, ids = {}, {}, {}
for g in blocks.local:
    x[g] = tack.array(rng.uniform((g, 0.0), (g + 1, 1.0), (10, 2)), POINT)
    velocity[g] = tack.array(rng.uniform((-1, 0), (1, 0), (10, 2)), POINT)
    ids[g] = tack.array(np.arange(10, dtype=np.int64) + 10 * g)

for step in range(20):
    for g in blocks.local:
        move(x[g], velocity[g], 0.1, WIDTH)
    # Each particle, with its state, to the block whose strip holds it now.
    routing = tc.Routing(blocks, {g: x[g].numpy() for g in blocks.local}, boxes, holds)
    arrived = routing.send({g: {"velocity": velocity[g], "id": ids[g]} for g in blocks.local})
    for g in blocks.local:
        points, values = arrived[g]
        x[g] = tack.array(points, POINT)
        velocity[g] = values.get("velocity", tack.empty(0, POINT))
        ids[g] = values.get("id", tack.empty(0, tack.i64))

counts = [ids[g].size for g in blocks.local]
print(counts)
# [15, 9, 6]
everyone = np.concatenate([ids[g].numpy() for g in blocks.local])
assert sorted(everyone.tolist()) == list(range(30))  # every particle, once
assert all(np.all(holds(g, x[g].numpy())) for g in blocks.local)
```

After 20 steps the 30 particles are spread unevenly over the blocks, and each
block holds exactly the particles inside its strip. The `values.get` calls
cover a block that ends a round with no particles.

The same pattern carries any state: a particle's age, its path so far, a flag
that says it left the domain. Points that go nowhere (destination -1) are
dropped by `send`, so check `routing.destination` first if you need to keep
them.

## Details

- `Routing(...)`, `send` and `reply` are collective. Every process makes them,
  in the same order, even with no points.
- A point is asked only of blocks whose boxes contain it. A box that is too
  small loses points before `holds` sees them. Grow each box by the tolerance
  your `holds` accepts, as the first example does.
- The points of a block may be sent to that block itself. Nothing special is
  needed.
- `holds` runs on the host with NumPy points. To test points on the device, as
  a cell locator does, copy them into a Tack array inside `holds` and launch a
  kernel there.
- Every exception is listed in the
  [runtime contract](../contracts/runtime-api.md#communication-between-blocks).

## Next

- [Sort and reduce over blocks](sort-and-reduce.md) brings together items
  that share a key, such as copies of one point, when no box says where they
  are.
- See also [Sample and trace](../visualization/sample-and-trace.md), in the
  layer above: probing and streamlines over a dataset in pieces are routing
  in rounds.
