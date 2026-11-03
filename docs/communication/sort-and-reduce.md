# Sort and reduce over blocks

This page shows how to combine values over every block and how to bring
matching items together when they are spread over many blocks. Sums, extrema
and counts over blocks are *collectives*. A global numbering is a scan. Finding
every copy of a point, a key or an id is a sort. The page is for readers who
know [blocks](blocks-and-worlds.md) and [exchanges](exchange-and-plans.md).
After it you can reduce, gather and number over blocks with results that do
not depend on how blocks are shared out, and match items across blocks with
`sort_by_key`.

<!-- tack-check: collectives -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")

blocks = tc.Blocks(["cpu"] * 4)

# Block g holds 1000 + 10 g values.
rng = np.random.default_rng(0)
values = {g: tack.array(rng.standard_normal(1000 + 10 * g)) for g in blocks.local}

total = blocks.all_reduce({g: values[g].sum() for g in blocks.local})
lowest = blocks.all_reduce({g: values[g].min() for g in blocks.local}, op="min")
sizes = blocks.all_gather({g: values[g].size for g in blocks.local})
starts, count = blocks.exclusive_scan({g: values[g].size for g in blocks.local})
print(sizes, starts, count)
# [1000, 1010, 1020, 1030] {0: 0, 1: 1000, 2: 2010, 3: 3030} 4060

# Element k of block g is element starts[g] + k of the whole, numbered once.
numbers = {g: tack.array(starts[g] + np.arange(values[g].size)) for g in blocks.local}
assert numbers[3].numpy()[-1] == count - 1

# The sum adds the blocks' sums in order of global id, so it is exactly this.
in_order = 0.0
for g in range(blocks.count):
    in_order += values[g].sum()
assert total == in_order

# NumPy arrays combine element by element.
ranges = blocks.all_reduce({g: np.array([values[g].min(), -values[g].max()])
                            for g in blocks.local}, op="min")
assert ranges[0] == lowest
```

## Combine values over every block

Three collectives combine small values held on the host, one value per block:

| call | gives, on every process |
|---|---|
| `all_reduce(values, op="sum")` | the values combined by `"sum"`, `"min"` or `"max"` |
| `all_gather(values)` | a list of every block's value, indexed by global id |
| `exclusive_scan(counts)` | `({g: offset}, total)`: each block's sum of the counts before it, and the sum of all |

`values` is `{g: value}` with an entry for every one of this process's
blocks. For `all_reduce`, a value is a number or a NumPy array, and arrays
combine element by element. For `all_gather`, a value may be any small Python
object that can be pickled, such as a tuple or a bounding box. For
`exclusive_scan`, a count is an integer of at least 0.

The values are host values, not Tack arrays. Reduce each block's array on its
device first, with `array.sum()` or a kernel (see
[Reductions, scans and sorts](../core/reductions-scans-sorts.md)), and give the
collective the result.

## Number items over every block

`exclusive_scan` is how a global numbering starts. Give it each block's count
of items. Each block's offset is where its share of the numbering begins, and
the total is the number of items in all. Block `g` then numbers its `k`-th item
`starts[g] + k`, as the example does. Each item gets one number, and the
numbers run from 0 to `total - 1` without gaps.

When items are duplicated across blocks, as shared points are, count only the
copies each block owns. The last section of this page shows how to decide who
owns what.

## Results do not depend on the arrangement

Collectives combine values in order of global id: block 0's value, then block
1's, and so on. The order is the same whichever process holds which block. A
floating-point sum over blocks therefore has the same bits on one process, on
three ranks, or on six. Tack's test suite checks this, with blocks shared out
over ranks in different ways.

The fixed order makes a sum reproducible, not more accurate. It is a plain
sum of the blocks' values from left to right, exactly what the example's
`in_order` loop computes.

## Match items across blocks with a sort

Some items have copies on several blocks: a point on the boundary between
two parts of a mesh, a particle near a seam, a global id held by a ghost and
by its owner. No box says where the copies are. `sort_by_key` brings them
together:

```python
ordered, carried = tc.sort_by_key(blocks, keys, values)
```

It sorts every block's keys together and cuts the global order into one range
per block. Block 0 ends with the smallest keys and the last block with the
largest, each block's in ascending order. The `values` move with their keys.
Most importantly, **every copy of a key ends on one block**. There, the copies
can be matched, counted or reduced without further communication.

The arguments are, for each of this process's blocks:

- `keys`: a one-dimensional Tack array of `i32`, `u32`, `i64` or `u64`, of
  the same type on every block.
- `values`: `{name: array}`, each array with one element per key. Each name
  has one element type on every block. Leave `values` out to sort only keys;
  `carried` is then `None`.

The sort is **stable**. Equal keys keep the order of their blocks, then their
order within a block. So the first copy of a key, after the sort, is the copy
on the lowest block. That gives a rule for ownership that every process
agrees on without asking: the lowest block that holds an item owns it.

A key is anything that is equal for copies and different otherwise: a grid
index, a hash of a point's exact coordinates, a cluster label, a global id. To
know where each copy came from, carry its *origin* as a value, for example
`(block << 40) | index`.

## Number shared points once

The example below has three blocks of a 12 by 5 lattice of points. Each
block holds five columns, so neighbouring blocks share a column. The sort
brings each point's copies together. Each block numbers the points it then
holds, a scan makes the numbers global, and an exchange sends each copy its
point's number.

<!-- tack-check: match -->
```python
import numpy as np

import tack
from tack import communication as tc

tack.set_default_device("cpu")

blocks = tc.Blocks(["cpu"] * 3)
NX, NY, M = 12, 5, 4                       # points per row, rows, columns a block steps

# Block g holds lattice columns 4 g to 4 g + 4, so neighbours share a column.
# A point's key is its lattice index; in a mesh it could be a hash of its
# exact position. Its origin is (block << 40) | index on that block.
keys, origin = {}, {}
for g in blocks.local:
    i, j = np.meshgrid(np.arange(M * g, min(M * g + M, NX - 1) + 1), np.arange(NY))
    keys[g] = tack.array((j * NX + i).ravel(), tack.i64)
    origin[g] = tack.array((g << 40) | np.arange(keys[g].size), tack.i64)

# Every copy of a key lands on one block, in order of block, then index.
ordered, carried = tc.sort_by_key(blocks, keys, {g: {"origin": origin[g]} for g in blocks.local})

# On each block, a run of equal keys is one point's copies. Number the runs,
# and make the numbers global with a scan of each block's count of runs.
new_run = {}
for g in blocks.local:
    k = ordered[g].numpy()
    new_run[g] = np.diff(k, prepend=k[:1] - 1) != 0  # True where a run starts
starts, points = blocks.exclusive_scan({g: int(new_run[g].sum()) for g in blocks.local})

# Send each copy its point's number, back to the block it came from.
outgoing = {}
for g in blocks.local:
    o = carried[g]["origin"].numpy()
    number = starts[g] + np.cumsum(new_run[g]) - 1
    outgoing[g] = {}
    for source in np.unique(o >> 40).tolist():
        mine = (o >> 40) == source
        outgoing[g][source] = {"index": tack.array(o[mine] & ((1 << 40) - 1)),
                               "number": tack.array(number[mine])}
incoming = blocks.exchange(outgoing)

ids = {}
for g in blocks.local:
    host = np.full(keys[g].size, -1, np.int64)
    for message in incoming[g].values():
        host[message["index"].numpy()] = message["number"].numpy()
    ids[g] = host

print(points, "points;", sum(keys[g].size for g in blocks.local), "copies")
# 60 points; 70 copies
for g in blocks.local:
    assert np.array_equal(ids[g], keys[g].numpy())   # every copy got its point's number
```

The 70 copies make 60 points, each numbered once. Here the numbers equal the
lattice indices, because the sort numbers points in key order and every
lattice index is present. With hashed positions as keys, the numbers would
be in hash order instead, which is just as good for an identity.

The bookkeeping on the host uses NumPy for clarity. On a GPU, keep it on the
device with Tack's own scans and kernels.

This is the pattern behind Tack's global ids and ghost cells for a dataset in
pieces. See also [Data in pieces](../data/pieces.md), in the layer above.

## Details

- `sort_by_key` is a sample sort, as DIY's is. Each block sorts its own keys
  on its device and sends 16 samples of them, weighted by its count. Every
  block then finds the same splitters, sends each range of its keys to its
  block, and sorts what arrives. The `samples` argument changes the number of
  samples.
- Blocks end with about equal numbers of keys. A key repeated many times
  cannot be split between blocks, so heavily repeated keys can leave the
  blocks uneven.
- When every block is in one process on one device, the sort joins the keys
  and sorts them once, with no messages. The result is the same.
- `all_gather` pickles its values under MPI. Keep them small: counts, boxes,
  a few numbers. Move arrays with `exchange`.
- The collectives raise `ValueError` if any of this process's blocks is
  missing from `values`. Every exception is listed in the
  [runtime contract](../contracts/runtime-api.md#communication-between-blocks).

## Next

- [Run under MPI](mpi.md) runs these calls over many ranks.
- [Reductions, scans and sorts](../core/reductions-scans-sorts.md) covers the
  sorts and scans within one array, which `sort_by_key` uses on each block.
