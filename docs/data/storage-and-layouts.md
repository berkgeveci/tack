# Storage and layouts

A field's space says what its values mean. Its array says how the values are
supplied and in what order. This page is about the array side. It is for
readers who attach a solver's arrays to a dataset, fill values themselves,
or read them back on the host. After reading it you can supply values
without storing them, keep a solver's separate component arrays as they
are, find any value of any space in its array, and choose how `L2` values
are laid out for a GPU. It assumes [Fields and where values live](fields.md)
and, for the higher-order spaces, [Higher-order data](higher-order.md).

## One mesh, three kinds of storage

The positions of an image are never stored: they are computed from its
origin and spacing. A field can be computed too, from a constant. And a
vector field can keep its components in three separate arrays, as many
solvers write them. Filters read all three as they read ordinary arrays.

<!-- tack-check: storage-first -->
```python
import numpy as np

import tack
import tack.dataset as td
from tack import arrays

tack.set_default_device("cpu")

data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0))      # two hexahedra
print(type(data.geometry.values).__name__)                       # UniformCoordinates

# A cell field that is 1 everywhere, stored as one number and a count.
data.fields["one"] = td.Field(td.Constant(data), arrays.ConstantArray(1.0, data.num_cells))

# A velocity kept as a solver's three component arrays: structure of arrays.
x = data.positions()
u, v, w = (tack.array(c.astype(np.float32)) for c in (x[:, 0], 2 * x[:, 1], -x[:, 2]))
data.fields["velocity"] = td.Field(td.H1(data), tack.soa(u, v, w))

probed = td.probe(data, np.array([[0.25, 0.5, 0.5]]))
assert np.allclose(probed.fields["velocity"].values.numpy(), [[0.25, 1.0, -0.5]])
assert np.allclose(probed.fields["one"].values.numpy(), 1.0)
print(data.fields["velocity"].values)                            # ..., layout='soa'
```

## Meaning and storage are separate

A field couples a space and an array: `td.Field(space, values)`. The space
says how many values there are and what each one means. The array supplies
them. Any Tack array of the right length works, and so does any implicit
array; `Field` checks that the count matches the space.

So do not infer a space from an array's class. A `CartesianProduct` is a way
to compute 3-vectors from three axes; `H1` says how values are attached to a
topology and interpolated. Both an image and a rectilinear grid have an `H1`
geometry of order 1. What differs is how each position is obtained:

| Geometry values | What is stored | How a position is obtained |
|---|---|---|
| `UniformCoordinates` | dimensions, origin, spacing, direction | from the image formula |
| `CartesianProduct` | three axis arrays | one coordinate from each axis |
| an array of `tack.vec3()` | one 3-vector per value | read |

The same holds for any field. Its values may change representation without
its meaning changing, and the reverse.

## Values computed, not stored: implicit arrays

An *implicit array* computes each element when it is read, from a formula or
from other arrays. Tack inlines that computation into the kernel that reads
it, so nothing is stored in between. The ones that matter most for data:

| Implicit array | Useful when |
|---|---|
| `UniformCoordinates`, `CartesianProduct` | positions follow an image or a rectilinear grid |
| `ConstantArray`, `CountingArray` | values are one constant, or count up |
| `Permutation`, `View`, `Strided` | values are another array's, selected or reordered |
| `ExtractComponent`, `Components` | scalar components and vector values need to be joined or split |
| `Transform`, `Function`, `Cast` | a device function or a conversion computes the values |

Topologies use them too. A point cloud's cell types are a `ConstantArray`
and its connectivity a `CountingArray` (see
[Topology and geometry](topology-and-geometry.md)). The whole list, and how
to write your own, is on [Arrays and types](../core/arrays-and-types.md).

An implicit array does its work every time it is read. Operands may
themselves be implicit, so long chains cost nothing to store but something
to read. If an expensive expression is read often, materialize it once.
`tack.arrays.materialize(values)` computes device storage, and
`tack.arrays.to_host(values)` a host snapshot. Neither changes the field's
own representation. An implicit array over a buffer that changes sees the
current values; a materialized copy is a snapshot until you compute it again.

## Vectors: interleaved or structure of arrays

A vector field holds one vector per value. By default its components are
interleaved: `tack.empty((space.size,), tack.vec3())` stores value 0's x, y
and z, then value 1's, and `numpy()` gives an array of shape
`(space.size, 3)`. Many solvers instead keep one array per component.
`tack.soa(u, v, w)` makes a vector array whose components are those arrays,
with no copy. Kernels and filters index it as they index an interleaved one.
[Arrays and types](../core/arrays-and-types.md) explains both layouts.

## Where each value is

When you fill an array yourself, or hand one over from a solver, you need to
know which value is which. The mesh's own numbering comes first. Points are
numbered from 0 to `num_points - 1`, and cells from 0 to `num_cells - 1`.
Edges and faces are derived when first needed (`topology.edges()`,
`topology.faces()`) and numbered the same way every time for a topology. Edge
`e`'s points are `edges().rows[e]`, lower id first. Face `f`'s points are
`faces().rows[f]`, in the order that goes around it outward from its first
cell. The arrays of the topology itself are on
[Topology and geometry](topology-and-geometry.md).

| Space | Number of values | The value of ... is at |
|---|---|---|
| `H1` order 1, `Values(data, "points")` | `num_points` | point `p`: `p` |
| `Constant`, `L2` order 0, `Values(data, "cells")` | `num_cells` | cell `c`: `c` |
| `Values(data, "faces")`, `Values(data, "edges")` | one per face or edge | face or edge `f`: `f` |
| `L2` order 1 | one per cell corner | cell `c`'s corner `j`: `offsets[c] + j`, in the cell's corner order |
| `L2` order 2 and up, or Gauss–Legendre | each cell's lattice nodes | cell `c`'s node `n`: `offsets[c] + n`, or `n * num_cells + c` (see below) |
| `H1` order 2 and up | points, then edges, faces, interiors | below |
| `SideTraces` | 8 per face | face `f`, side `s`, point `j` of its row: `(f * 2 + s) * 4 + j` |

`offsets` is `space.offsets`. For `L2` of order 1 it is the topology's own
offsets array, so each value lines up with an entry of the connectivity, and
a point shared by four cells has four values.

### A cell's nodes in order

Above order 1, and for `L2` on Gauss–Legendre nodes, a cell's nodes are
numbered on a *lattice* in its reference element.
`td.bases.lattice_ijk(shape, p)` lists them.

- **Segments, quads and hexahedra** (and pixels and voxels): node `(i, j, k)`,
  each from 0 to `p`, is number `i + (p + 1) * (j + (p + 1) * k)`, the first
  index fastest. `i` runs from corner 0 toward corner 1, `j` toward corner 3,
  and `k` toward corner 4.
- **Triangles and tetrahedra:** nodes `(i, j, k)` with `i + j + k <= p`, at
  `(i, j, k) / p`, the first index fastest. A triangle of order 2 numbers
  `(0,0), (1,0), (2,0), (0,1), (1,1), (0,2)`. `i` runs toward corner 1, `j`
  toward corner 2 and `k` toward corner 3.
- **Wedges:** the triangle's lattice in each of `p + 1` layers, the triangle
  fastest, from the bottom triangle (corners 0, 1, 2) to the top.

`L2` of order 1 is the exception. Its values are at the corners in the
cell's corner order, not on the lattice.

### `H1` above order 1: numbered by entity

A shared value is stored once, by the entity that holds it:

1. **Points:** `num_points` values, by point id, as at order 1.
2. **Edges:** `p - 1` values per edge, by edge id. Edge `e`'s node `k`, from
   1 to `p - 1` counted from its lower point id, is at
   `num_points + e * (p - 1) + (k - 1)`.
3. **Faces**, in a mesh of solids: by face id, `(p - 1)²` values on a quad
   face and `(p - 1)(p - 2) / 2` on a triangle face. Quad face `f`'s node
   `(u, v)`, each from 1 to `p - 1`, is its `(v - 1) * (p - 1) + (u - 1)`-th
   value, `u` running from `rows[f][0]` toward `rows[f][1]` and `v` toward
   `rows[f][3]`.
4. **Cell interiors:** by cell, each cell's interior lattice nodes in
   lattice order: `(p - 1)³` in a hexahedron, `(p - 1)²` in a quad of a 2D
   mesh, `(p - 1)(p - 2)(p - 3) / 6` in a tetrahedron.

Faces and cells of different kinds hold different counts, so each one's
block starts where the previous one ends. A 2D mesh has points, edges and
cell interiors. Serendipity keeps the points and edges alone. A space with
an element-to-DOF table (`dofs=`) numbers its values by your table instead
(see [Higher-order data](higher-order.md)).

This example checks those rules on two hexahedra. The function's value
encodes the position (`x + 10y + 100z`), so each value says where its node
is.

<!-- tack-check: storage-layouts -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")


@tack.func
def code(x):                       # each position's coordinates, readable at a glance
    return x[0] + 10.0 * x[1] + 100.0 * x[2]


# Two hexahedra side by side: 12 points at x = 0, 0.5, 1; y and z = 0, 1.
grid = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0))
data = td.DataSet(td.UnstructuredTopology(
    types=np.full(2, 12, np.uint8), offsets=np.array([0, 8, 16]),
    connectivity=np.array([0, 1, 4, 3, 6, 7, 10, 9, 1, 2, 5, 4, 7, 8, 11, 10]),
    num_points=12), grid.positions())
x = data.positions()


def values(space):
    return td.interpolate(data, code, space).values.numpy()


# H1 order 1: one value per point, by point id.
assert np.allclose(values(td.H1(data)), x @ [1, 10, 100])

# L2 order 1: each cell's corners in connectivity order, at the topology's offsets.
corners = data.topology.connectivity.numpy()
assert np.allclose(values(td.L2(data, order=1)), x[corners] @ [1, 10, 100])

# L2 order 2: 27 nodes a cell, the first index fastest: x, then y, then z.
l2 = values(td.L2(data, order=2))
assert l2.size == 54 and np.allclose(l2[:4], [0.0, 0.25, 0.5, 5.0])

# H1 order 2: points, then edges, quad faces and cell interiors.
h1 = values(td.H1(data, order=2))
edges, faces = data.topology.edges(), data.topology.faces()
assert h1.size == 12 + edges.num_edges + faces.num_faces + 2 == 45
low, high = edges.rows.numpy()[0]
assert np.isclose(h1[12], (x[low] + x[high]) @ [1, 10, 100] / 2)   # edge 0's midpoint
assert np.allclose(h1[-2:], [55.25, 55.75])                         # the cell centres
print("H1 order 2:", h1.size, "values; L2 order 2:", l2.size)
```

## `L2` layouts: blocks or cells

An `L2` field of order 2 or more, or on Gauss–Legendre nodes, has a choice
of layout. `layout="blocks"` keeps each cell's values together, as MFEM and
VTK's cell grids store them: value `n` of cell `c` is at `offsets[c] + n`.
`layout="cells"` stores value `n` of every cell together, at
`n * num_cells + c`, so that GPU threads working on neighbouring cells read
neighbouring memory.

Each is faster where the default picks it. On an RTX 4060 Ti at order 6,
cell-fastest reads 2.3 times and writes 3.6 times faster than blocks. On a
CPU, blocks read 1.5 to 1.8 times and write about twice as fast. So the
default, `layout="auto"`, is `"cells"` for order 3 and up on a GPU and
`"blocks"` everywhere else. Moving a dataset to another device may change an
`"auto"` layout, and `to_device` then reorders the values. Corner values
(orders 0 and 1), an order per cell, and `L2` on triangles, tetrahedra and
wedges are always blocks.

Every operation gives the same answer in either layout; only the order of
`field.values` differs. `space.blocks(values)` reads them per cell, an array
of shape `(cells, values a cell)`, whatever the layout, and `space.indices()`
says where each one is. The MFEM interop reads and writes both. Set the
environment variable `TACK_L2_LAYOUT=cells`, or `blocks`, to run everything
one way.

<!-- tack-check: storage-l2-layout -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")


@tack.func
def code(x):
    return x[0] + 10.0 * x[1] + 100.0 * x[2]


data = td.uniform_grid((3, 3, 3), spacing=(0.5, 0.5, 0.5))   # 8 hexahedra
blocks = td.L2(data, order=3, layout="blocks")
cells = td.L2(data, order=3, layout="cells")
by_blocks = td.interpolate(data, code, blocks).values.numpy()
by_cells = td.interpolate(data, code, cells).values.numpy()

n, c = 5, 3                                                  # node 5 of cell 3
assert by_blocks[blocks.offsets.numpy()[c] + n] == by_cells[n * data.num_cells + c]
assert np.array_equal(blocks.blocks(by_blocks), cells.blocks(by_cells))
print(td.L2(data, order=3).layout, "on the CPU;", blocks.blocks(by_blocks).shape)
```

## Reading values on the host

| Expression | What you get |
|---|---|
| `data.fields["t"].values` | the stored or implicit array |
| `tack.arrays.to_host(field.values)` | a host snapshot; vectors one row per value |
| `tack.arrays.materialize(field.values)` | device storage, computing implicit values |
| `data.positions()` | the points' positions on the host, relative to the origin |
| `tack.arrays.to_host(data.geometry.values)` | every geometry value, in its space's order |
| `space.offsets`, `space.indices()`, `space.blocks(values)` | where each cell's `L2` values are |

`positions()` gives the topology's points, even when the geometry has
higher-order nodes, and refuses an `L2` geometry. `space.indices()` and
`space.blocks()` give one row per cell, so they refuse cells holding
different numbers of values, such as a mesh of mixed shapes or an order per
cell. Read `space.offsets` there.

Use host reads for checks, diagnostics and output. Between filters and
kernels, keep values in fields. On a GPU a host read waits for the queued
work it depends on, so reading every step of a loop can cost more than the
loop.

## Details

Edge and face ids are stable for one topology, but not across topologies.
To match values between two meshes, match by position or by global id (see
[Data in pieces](pieces.md)), not by edge or face number. Kernels never
compute these storage indices themselves: a field view reads a cell's values
by node, whatever the space and layout (see
[Views](../visualization/views.md)).

## Where to go next

- [Connect a simulation's data](connect-simulation-data.md) hands a solver's
  arrays to Tack without copies.
- [Higher-order data](higher-order.md) explains the spaces whose layouts
  this page gives.
- [Arrays and types](../core/arrays-and-types.md) lists every implicit array
  and explains structure-of-arrays storage.
