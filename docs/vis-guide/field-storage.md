# Arrays, storage and layouts

A field's [space](03-fields-and-spaces.md) says what its values mean. Its
array says how they are supplied. Use this chapter when attaching solver
coefficients or writing code that needs their exact layout. Ordinary filters
read these representations through field views.

!!! tip "Choose the level of detail you need"
    For a first pass, read [computed versus stored values](#values-can-be-computed-instead-of-stored).
    The addressing tables are for advanced integration and custom kernels.

## How the values are stored

A field is one array of values; what each value means comes from the space,
and the space reads the topology to say which value belongs to which cell.
This section gives the exact layout of each: what to write when you fill an
array yourself, or hand one to Tack from a solver.

### The mesh: points, cells, edges and faces

Points are numbered `0` to `num_points - 1`. With first-order H1 geometry,
there is one 3-vector per point, in point order (`data.positions()` gives
them on the host). Higher-order or L2 geometry follows its own space's layout.
An image's or a rectilinear grid's positions are computed
from the point's index rather than stored.

An `UnstructuredTopology` is three arrays, VTK's layout:

| Array | Length | Holds |
|---|---|---|
| `types` | `num_cells` | each cell's VTK cell type (`12` a hexahedron, `10` a tetrahedron, ...) |
| `offsets` | `num_cells + 1` | where each cell's points start in `connectivity`; the last entry is its length |
| `connectivity` | `offsets[num_cells]` | the cells' point ids, one cell after another |

Cell `c`'s corners are `connectivity[offsets[c]:offsets[c + 1]]`, in its
shape's corner order, which is VTK's ([Cell Types and Bases](16-cells-and-bases.md)).
For a hexahedron that is the bottom face counterclockwise, then the top face
above it. Any of the three may be an implicit array: a mesh of one cell type
needs no stored `types` or `offsets`.

A `StructuredTopology` stores logical dimensions and derives incidence by arithmetic. On a grid of `nx × ny × nz` points,
point `(i, j, k)` is `i + nx * (j + ny * k)`, cell `(i, j, k)` is
`i + (nx - 1) * (j + (ny - 1) * k)`, and its corners are a hexahedron's
(a quad's in 2D): `(i, j, k)`, `(i + 1, j, k)`, `(i + 1, j + 1, k)`,
`(i, j + 1, k)`, then the same four at `k + 1`.

Edges and faces are derived when something needs them (`topology.edges()`,
`topology.faces()`) and numbered the same way every time. Edge `e`'s points
are `edges().rows[e]`, lower id first. Face `f`'s are `faces().rows[f]`, in
the order its side-0 cell (`faces().sides[f]`) goes around it outward.

### Where each value is

| Space | Values | Value of ... is at |
|---|---|---|
| `H1(data)` (order 1), `Values(data, "points")` | `num_points` | point `p`: `p` |
| `Constant(data)`, `L2(data, order=0)`, `Values(data, "cells")` | `num_cells` | cell `c`: `c` |
| `Values(data, "faces")`, `Values(data, "edges")` | one per face or edge | face or edge `f`: `f`, its id in `faces()` or `edges()` |
| `L2(data, order=1)` | one per cell corner | cell `c`'s corner `j`: `offsets[c] + j`, `j` in connectivity order |
| `L2(data, order=p)`, `p ≥ 2` (or `basis="gauss_legendre"`) | each cell's nodes | cell `c`'s node `n`: `offsets[c] + n` (`layout="blocks"`), or `n * num_cells + c` (`"cells"`) |
| `H1(data, order=p)`, `p ≥ 2` | points, then edges, faces, cell interiors | below |
| `SideTraces(data)` | 8 per face | face `f`, side `s`, point `j` of its row: `(f * 2 + s) * 4 + j` |

So an `H1` field of order 1 is a value array of `num_points` entries and the
topology's connectivity: cell `c`'s values are the values at its corner
points. An `L2` field of order 1 shares the topology's `offsets` (its
`space.offsets` *is* that array) and holds a value for each entry of the
connectivity, so a point shared by four cells has four values.

### A cell's nodes in order

Above order 1, and for `L2` on Gauss–Legendre nodes, a cell's nodes are
numbered on a *lattice* in its reference element (`bases.lattice_ijk(shape,
p)` lists them):

- **Segments, quads, hexahedra** (pixels, voxels): `(i, j, k)`, each from 0 to
  `p`, numbered `n = i + (p + 1) * (j + (p + 1) * k)`, the first index fastest.
  `i` runs from corner 0 toward corner 1, `j` toward corner 3, `k` toward
  corner 4. Node `i` along an axis sits at the basis' `i`-th 1D node
  (Gauss–Lobatto, equispaced or Gauss–Legendre); a Bernstein space keeps its
  `i`-th coefficient there instead.
- **Triangles and tetrahedra:** `(i, j, k)` with `i + j + k ≤ p`, at
  `(i, j, k) / p`, the first index fastest: a triangle of order 2 is
  `(0,0), (1,0), (2,0), (0,1), (1,1), (0,2)`. `i` runs toward corner 1, `j`
  toward corner 2, `k` toward corner 3.
- **Wedges:** the triangle's lattice in each of `p + 1` layers, the triangle
  fastest, the layers from the bottom triangle (corners 0, 1, 2) to the top.

`L2` of order 1 is the exception: its values are at the corners, in
connectivity order, not on the lattice. On a hexahedron, value 2 is corner 2,
`(1, 1, 0)`; lattice node 2 of order 1 would be `(0, 1, 0)`.

### `H1` above order 1: numbered by entity

Shared values are stored once, by the entity that holds them, so the cells
around an edge or face find the same value:

1. **Points:** `num_points` values, by point id, as at order 1.
2. **Edges:** `p - 1` values per edge, in edge-id order. Edge `e`'s node `k`
   (1 to `p - 1`, counted from its lower point id, `edges().rows[e][0]`) is at
   `num_points + e * (p - 1) + (k - 1)`.
3. **Faces** (meshes of solids): in face-id order, `(p - 1)²` values on a quad
   face and `(p - 1)(p - 2) / 2` on a triangle face (order 2: one on each quad
   face, none on triangles). Quad face `f`'s node `(u, v)`, each 1 to `p - 1`,
   is its `(v - 1) * (p - 1) + (u - 1)`-th value, `u` running from
   `faces().rows[f][0]` toward `rows[f][1]` and `v` toward `rows[f][3]`. A
   triangle face's nodes follow the triangle lattice from `rows[f][0]` toward
   `rows[f][1]` and `rows[f][2]`.
4. **Cell interiors:** in cell order, each cell's interior lattice nodes
   (every index from 1 to `p - 1` on a tensor shape), in lattice order:
   `(p - 1)³` on a hexahedron, `(p - 1)²` on a quad of a 2D mesh, `p - 1` on
   a segment, `(p - 1)(p - 2)(p - 3) / 6` on a tetrahedron, `(p - 1)²(p - 2) / 2`
   on a wedge.

Faces and cells of different kinds hold different counts, so each one's
block starts where the previous one's ends. A 2D mesh has points, edges and
cell interiors. `basis="serendipity"` keeps the points and the edges alone.
`H1(data, order=p, dofs=table)` numbers the values by your table instead: a
row per cell, each of its lattice nodes' value index.

<!-- tack-check: layouts -->
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

### Vectors

A vector field holds one vector per value, its components together:
`tack.empty((space.size,), tack.vec3())` stores value 0's `x, y, z`, then
value 1's, and `numpy()` gives `(space.size, 3)`. `layout="soa"` keeps each
component in an array of its own instead
([Arrays and Types](../users-guide/02-arrays-and-types.md)).

### Every value of a cell, at any order

A field's view numbers its cell's own values, whatever the space, order or
shape: `NODES` of them (a constant, so a loop over them unrolls), node `k` at
parametric coordinates `node_parametric(k)`, its value `dof(c, k)`, stored at
`dof_index(c, k)` in the field's array. At order 1 the nodes are the corners,
at order 0 one node at the cell's centre; an `H1` order-2 cell numbers them as
VTK's quadratic cells do -- corners, then edges, then faces and the centre --
and every other order on the lattice above. So one kernel reads every value
of a cell on any mesh, here each cell's largest:

<!-- tack-check: cell-nodes -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")


@tack.kernel
def largest_value(cells, u, out):
    for c in cells:                                  # every shape, space and order
        top = u.dof(c, 0)
        for k in range(1, u.NODES):                  # the cell's own values
            top = max(top, u.dof(c, k))
        out[cells.entity_id(c)] = top


@tack.func
def height(x):
    return x[0] + 2.0 * x[1] + 3.0 * x[2]


data = td.uniform_grid((4, 3, 3), spacing=(0.5, 0.5, 0.5))
centres = td.values_at_centers(data, td.interpolate(data, height, td.H1(data))).values.numpy()
for space in (td.H1(data, order=3), td.L2(data, order=2), td.Constant(data)):
    out = tack.empty((data.num_cells,), tack.f32)
    td.for_each(largest_value, data, "cells", td.interpolate(data, height, space), out)
    # A linear field is largest at a cell's far corner: its centre value plus
    # (1 + 2 + 3) / 4; a Constant cell's one node is its centre.
    expected = centres if isinstance(space, td.Constant) else centres + 1.5
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)
print("largest values:", out.numpy()[:3])
```

The same view gives values elsewhere in the cell: `corner_value(c, j)` at its
corner `j`, `value(c, pc)` at parametric coordinates `pc`. Reading another
field at these nodes, `p.value(c, q.node_parametric(k))`, computes a field of
one space from fields of others without projecting first:
[Work at a field's nodes](05-mesh-kernels.md#work-at-a-fields-nodes) adds an
order-1 field to an order-2 one that way. A kernel never computes storage
indices itself ([Write Mesh Algorithms](05-mesh-kernels.md)).

### Finding values on the host

`space.offsets` gives where each cell's `L2` values start; `space.indices()`
and `space.blocks(values)` give them as one row per cell, so they refuse cells
holding different numbers of values (a mixed mesh, an order per cell): read
`space.offsets` there. `td.interpolate` fills any space from a function of
position. Derived edge and face ids are stable for a topology, but not across
topologies: match values between meshes by position or global id, not by
edge or face number.

## Where a cell's values sit

An `L2` field of a tensor basis has a layout. `"blocks"` keeps each cell's
values together, as MFEM and VTK store them; `"cells"` stores value `n` of
every cell together, so that GPU threads working on neighbouring cells read
neighbouring memory. The default, `"auto"`, picks `"cells"` for order 3 and up
on a GPU and `"blocks"` everywhere else, because each is the faster one where
it is picked: at order 6 cell-fastest reads 2.3 times and writes 3.6 times
faster on an RTX 4060 Ti, and blocks read about 1.6 times and write about twice
as fast on a CPU.

```python
--8<-- "docs/examples/data_high_order.py:layout"
```

Every operation gives the same answer in either layout; only the order of
`field.values` differs. `space.blocks(values)` reads them per cell, `(cells,
values a cell)`, whatever the layout, and `space.indices()` says where each is.
Moving a dataset to another device may change an `"auto"` layout, and
`to_device` reorders the values. MFEM and the VTK interop take and give either.
Set `TACK_L2_LAYOUT=cells` (or `blocks`) to run everything one way.

## Values can be computed instead of stored

Uniform-grid coordinates need only dimensions, origin and spacing; allocating
three scalars for every point would be redundant. An implicit value array
provides a `get(k)` expression that Tack inlines into the consuming kernel.

### Geometry's space and its implicit array are different layers

Both an image and a rectilinear grid have an order-1 H1 geometry field. Its
space says how a cell's nodal positions define a coordinate mapping; its
implicit value array says how to obtain each nodal position. The two layers
can change independently:

| Geometry values | Information retained | How one nodal position is obtained |
|---|---|---|
| `UniformCoordinates` | Dimensions, origin, spacing and direction | Evaluate the uniform-grid coordinate formula |
| `CartesianProduct` | Three backend axis arrays | Gather one coordinate from each axis |
| Explicit `tack.Array` with dtype `tack.vec3(...)` | One stored 3-vector per geometry DOF | Read the vector |

Calling `materialize` on image coordinates changes their storage into an
explicit value array; it does not change H1 order, add cells, or turn
`StructuredTopology` into `UnstructuredTopology`. Calling `to_host` creates
a host snapshot without replacing the geometry's value representation.
See the [geometry formulas and topology combinations](02-meshes.md#image-rectilinear-and-curvilinear-data).

Do not infer a function space from an array class. A `CartesianProduct`
describes a way to compute 3-vectors; `H1` describes how coefficients are
associated and interpolated on a topology. `td.Field` couples these two
choices, with a check that the value count matches the space.

### Other implicit coefficient arrays

```python
--8<-- "docs/examples/data_fields.py:implicit"
```

Here `ConstantArray` stores the constant and count, not 12 separate values.
`materialize` computes storage on the default device. `to_host` evaluates and
downloads a snapshot. Neither changes the original field's representation.

| Implicit array | Useful when |
|---|---|
| `UniformCoordinates`, `CartesianProduct` | Coordinates follow a uniform or rectilinear grid |
| `ConstantArray`, `CountingArray` | Values are constant or a sequence |
| `Permutation`, `View`, `Strided` | Values are selected or reorganized without a new value buffer |
| `ExtractComponent`, `Components` | Scalar components and vector values need to be connected |
| `Transform`, `Function`, `Cast` | A device function or conversion can compute values on demand |

Operands can themselves be implicit arrays. This keeps intermediates out of
storage, but the expression still does work whenever it is read. Materialize
an expensive expression if reuse makes that worthwhile, then measure the whole
pipeline. An implicit wrapper around a mutable buffer observes its current
values; a materialized result is a previous computation until recomputed.

## Inspect without losing track of transfers

| Expression | What you get |
|---|---|
| `data.num_points`, `data.num_cells` | Counts for the mesh |
| `data.fields["temperature"]` | The associated `td.Field` |
| `data.fields["temperature"].space` | Its space, including the topology |
| `data.fields["temperature"].values` | The stored or implicit value array |
| `data.positions()` | A host array of corner positions for H1 geometry |
| `tack.arrays.to_host(field.values)` | A host snapshot; vectors have one row per value |
| `tack.arrays.materialize(field.values)` | Device storage; computes implicit values if necessary |

`positions()` returns the topological corners, including when H1 geometry has
higher-order nodes; it refuses L2 geometry. Use
`tack.arrays.to_host(data.geometry.values)` to inspect all of geometry's
coefficients in their space's ordering.

Use host snapshots for diagnostics, independent checks and file output. Keep
values in fields between filters and kernels. On GPU backends a host read waits
for the queued work it needs, so repeated inspection in a timestep loop can
dominate its cost.


## Meaning and storage are independent

`Constant(data)` defines a different constant function in each cell.
`ConstantArray(value, count)` supplies the same value at every array index;
it can back an H1, L2 or Constant field. `tack.constant` is instead a numerical
constant for compiled kernel code.

Continue with [Create and Convert Fields](create-fields.md),
[public views](17-views.md), or the [core array guide](../users-guide/02-arrays-and-types.md).
