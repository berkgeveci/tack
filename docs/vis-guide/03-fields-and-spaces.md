# Fields, spaces and implicit arrays

A field's space determines which values can disagree and how to evaluate them
inside a cell. Moving between spaces changes that interpretation. It can
introduce averaging even when the geometry and topology remain the same.
Start with [Function Spaces and Degrees of Freedom](00-function-spaces.md)
for the meaning of H1, H2, L2 and polynomial order. This chapter applies those
ideas to Tack's field and array representations.

For construction recipes, jump to [L2 fields](#construct-an-l2-field) or
[quadratic H1 fields](#construct-a-quadratic-h1-field). Both use the same
mesh and have runnable examples below.

## Choose where values live

| Space | Ownership | Use it for |
|---|---|---|
| `td.H1(data, order=1)` | One shared value per corner vertex | A continuous first-order nodal temperature or velocity |
| `td.H1(data, order=2)` | Shared corner, edge and supported face/interior nodes | A higher-order H1 approximation; not an H2 space |
| `td.H1(data, order=p, basis=...)` | From order 3, on segments, quads and hexahedra: `p - 1` per edge, `(p - 1)²` per face, `(p - 1)^d` per cell; on triangles, tetrahedra and wedges with `basis="equispaced"` or `"bernstein"` (Bernstein coefficients, from order 2) | Spectral-element and MFEM fields of any order, VTK's and gmsh's Lagrange simplices and wedges |
| `td.L2(data, order=p, basis=...)` | From order 2 (or on Gauss–Legendre nodes): each cell's own `(p + 1)^d` values (`C(p + d, d)` on a simplex, equispaced) | DG fields of any order |
| `td.Constant(data)` | One value per cell | Cell averages, material parameters |
| `td.L2(data)` | Each cell's own corner values | Discontinuous solution fields |
| `td.Values(data, "faces")` | One value per face, without an interpolation basis | Face areas, labels or fluxes |
| `td.Values(data, "points" / "cells" / "edges")` | One value per named entity kind | Entity attributes that need no basis |
| `td.SideTraces(data)` | Values per face, per side, per face corner | Two-sided evaluation, by projecting into it |

For `L2(order=0)` the value is constant per cell; `L2(order=1)` uses each
cell's own corners. `H1(order=2)` adds higher-order degrees of freedom on
supported shapes; [higher orders](#higher-orders) add more of them on the
tensor shapes. `Values` describes association without promising any
interpolation between entities.

`tack.empty((field.space.size,), tack.vec3())` allocates one
3-vector per DOF; `td.Field(field.space, values)` gives those values their
mesh meaning. The array has shape `(field.space.size,)` and downloads as
`(field.space.size, 3)`.

The field's coefficient count is `field.space.size`. It agrees with
`data.num_points` for H1 order 1, but not generally for other spaces. A
vector field has one vector per space DOF, so the scalar component storage
has that count times the vector width.

`Constant` means a separate constant function in each cell. `ConstantArray`
means an array with one repeated value, regardless of space. These two
independent choices can be combined: an H1 field with a `ConstantArray` is
a spatially constant H1 function, while a `Constant` field backed by an
ordinary array can assign different values to every cell.

Spaces belong to a topology and are interned there: repeated `td.H1(data)`
uses the same layout. A field with a matching length on another topology is
still a different association. Wrap values with the correct space when they
are created; do not repair an association by guessing from array length later.

### Fields that share a name

A dataset's fields are named, and two may share a name in different spaces:
a pressure at the points and a pressure per cell, as VTK keeps point data and
cell data apart. The space tells them apart. `data.fields["p", td.Constant]`
is the cell one; the space itself (`td.H1(data)`) works as well as its kind.
Where one field has a name, the name alone finds it. Where two do, the name
alone is refused, never guessed, so code that worked on one file cannot
quietly read the other field on the next:

<!-- tack-check: fields-by-space -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((3, 3, 3))
x = data.positions()
data.fields["p"] = td.Field(td.H1(data), tack.array(x[:, 0].astype(np.float32)))
data.fields["p", td.Constant] = td.Field(td.Constant(data), tack.zeros((data.num_cells,)))

assert isinstance(data.fields["p", td.H1].space, td.H1)
try:
    data.fields["p"]
except KeyError as refused:
    print(refused)                                 # 'p' names 2 fields, in H1(order=1) and ...
else:
    raise AssertionError("a name two fields share must be refused")

gradient = td.gradients(data, ("p", td.H1))        # algorithms take (name, space) too
kept = td.threshold(data, ("p", td.H1), 0.5, 2.0, fields="p")   # a name keeps both
print(list(kept.fields))                           # [('p', H1(order=1)), ('p', Constant())]
```

- **Putting a field in.** `data.fields["p"] = f` makes `f` the one field
  called `"p"`, as with a dictionary. `data.fields["p", td.Constant] = f`
  puts it beside fields of that name in other spaces, replacing the one in
  its own.
- **Keys.** Iterating `data.fields` gives each field's name, or `(name,
  space)` for a name fields share, so every key finds its field and
  `dict(data.fields)` keeps them all. `data.fields.named("p")` lists every
  field called `"p"`; `data.fields.entries()` gives each field with its name.
- **Filters.** `fields="p"` selects every field of that name, and
  `fields=[("p", td.Constant)]` the one. A filter that puts fields in one
  space, as `probe` puts them all at the target's points, would make the
  two `"p"`s two point fields of one name; it asks which instead (VTK's
  probe keeps the point one), so pass `fields=`.
- **VTK.** `vtk_to_dataset` reads a point array and a cell array of one name
  as two fields of that name, and `dataset_to_vtk` writes them back the same
  way. Two point fields of one name, which VTK cannot keep, are refused.

## How the values are stored

A field is one array of values; what each value means comes from the space,
and the space reads the topology to say which value belongs to which cell.
This section gives the exact layout of each: what to write when you fill an
array yourself, or hand one to Tack from a solver.

### The mesh: points, cells, edges and faces

Points are numbered `0` to `num_points - 1`. The geometry is an `H1` field of
order 1: one 3-vector per point, in point order (`data.positions()` gives
them on the host). An image's or a rectilinear grid's positions are computed
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

A `StructuredTopology` stores nothing. On a grid of `nx × ny × nz` points,
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

## Construct an L2 field

The [complete independent-spaces example](../examples/data_spaces.py) builds
L2 pressure and quadratic H1 temperature on the same two hexes. This section
constructs pressure directly; [the next section](#construct-a-quadratic-h1-field)
constructs temperature. The snippets assume NumPy, Tack and `tack.dataset`
imported as `np`, `tack` and `td`, with a default device set.

Use `uniform_grid` as a shortcut for the mesh setup here; [Build Meshes](02-meshes.md)
shows its explicit topology and geometry construction:

```python
--8<-- "docs/examples/data_spaces.py:mesh"
```

Each hex has eight corners. `td.L2(data, order=1)` assigns a separate value
to each of them: 16 coefficients, even though the topology has only 12 vertices.
The four corners on the shared face each have **two** pressure coefficients.
They occupy the same physical positions and may have different values. See
[the node-sharing diagram](00-function-spaces.md#count-dofs-separately-from-corners).

### Initialize each cell's coefficients

Construct pressure as `p = x` in cell 0 and `p = x + 2` in cell 1.
It varies inside both cells and has a jump of 2 at x = 0.5. An L2 field need
not be constant inside a cell.

The space supplies `offsets`: coefficient `j` of cell `c` is stored at
`offsets[c] + j`. For these two hexes, the offsets are `[0, 8, 16]`:

| Cell | Storage slots | Meaning |
|---|---|---|
| 0 | 0 through 7 | Its own corner values, in the cell's local corner order |
| 1 | 8 through 15 | Its own corner values, including its side of the shared face |

The initialization kernel walks each cell's corners and writes its block.
Every coefficient has one writer. Using offsets also works when cells have
different corner counts; multiplying the cell id by eight would not.

```python
--8<-- "docs/examples/data_spaces.py:l2-kernel"
```

Allocate the values and attach their space:

```python
--8<-- "docs/examples/data_spaces.py:l2-build"
```

`td.for_each` supplies the geometry-aware cell view, so `corner_position`
returns physical coordinates. The resulting `Field` contains the original
cell-local coefficients; no shared-point averaging is involved.

### Copy coefficients from an existing solver array

If you already have nodal DG coefficients in this order, upload them directly.
For this particular pair of hexes, this constructs the same pressure field:

```python
--8<-- "docs/examples/data_spaces.py:l2-upload"
```

The two eight-value rows describe the cells' local nodes, not eight globally
unique points each. Flattening these rows preserves both interface values.
For a mixed-cell mesh, use `pressure_space.offsets` and the appropriate local
corner order to assemble the blocks; a rectangular `(num_cells, 8)` array is
specific to the all-hex example. Coefficients in another basis need conversion
before this upload, as in [MFEM import](01-datasets.md#importing-mfem-preserves-the-separation).

### A constant inside each cell: L2 order 0

Choose order 0 for one independent coefficient per cell:

```python
--8<-- "docs/examples/data_spaces.py:l2-constant"
```

Here `cell_space.size` is two. Values 1 and 3 are constant inside their own
cells and differ across the interface. `td.Constant(data)` is another
representation of that piecewise-constant function meaning; it also supports
topologies without reference elements.

### Evaluate and check the discontinuity

```python
--8<-- "docs/examples/data_spaces.py:l2-check"
```

For the varying order-1 pressure, the cell centers hold `[0.25, 2.75]` and
both physical gradients are `[1, 0, 0]`. The one interior face has jump
magnitude 2; `jump`'s sign follows its side ordering. The runnable check
compares the uploaded coefficients with the kernel-initialized ones as well.
These are cell-interior gradients; the interface discontinuity is represented
separately by the jump.

## Construct a quadratic H1 field

On the same mesh, construct `T = x² + 2y + 3z` using `td.H1(data, order=2)`.
The geometry stays first-order H1 and still has 12 corner positions.
Temperature has 45 global nodal coefficients, including nodes away from corners.

![A quadratic hex has eight corner nodes, twelve edge midpoints, six face centers and one cell-interior node](../assets/quadratic-hex-nodes.svg)

A single quadratic hex uses three nodes per reference coordinate direction:
the tensor-product layout has `3 × 3 × 3 = 27` nodes. Two adjacent hexes share
nine on their common face: its four corners, four edge midpoints and face center. Thus
`27 + 27 - 9 = 45` global DOFs. Count the two-cell mesh by entity as follows:

| Location | Global temperature DOFs |
|---|---:|
| Topological corner vertices | 12 |
| Edge midpoints | 20 |
| Face centers | 11 |
| Cell interiors | 2 |
| **Total** | **45** |

These are field nodes on the original two cells. They do not subdivide the
mesh, change corner connectivity, or make its geometry quadratic.

### Sample at the field's nodes

First define the function to sample:

```python
--8<-- "docs/examples/data_spaces.py:sample"
```

`interpolate` evaluates it at every node of the temperature space, where
the geometry puts each one, and returns the field:

```python
--8<-- "docs/examples/data_spaces.py:h1-build"
```

The space builds local-to-global addressing for shared corners, edges and
faces. Its node numbering is not simply 27 consecutive entries per cell.
`interpolate` writes a shared node once, from the first cell that has it,
rather than from every adjacent cell, and stores no node positions: each is
evaluated through the geometry as it is needed. To see where the nodes are,
project the geometry into the space
(`td.project(data, data.geometry, temperature_space)`); that does not change
`data.geometry`.

Interpolating differs from projecting a first-order **solution** into a
quadratic space: that conversion retains the original first-order function.
To represent this quadratic, interpolate it as above, or supply the
original quadratic coefficients in Tack's nodal basis and ordering.

### Evaluate through the quadratic basis

```python
--8<-- "docs/examples/data_spaces.py:evaluate"
```

At the two cell centers, quadratic evaluation gives `[2.5625, 3.0625]` and
physical gradients `[0.5, 2, 3]` and `[1.5, 2, 3]`. Projecting this temperature
down to H1 order 1 gives center values `[2.625, 3.125]`. The latter keeps the
corner values but changes the function between them.

![Actual field evaluations: L2 pressure varies within each cell and has two traces at the interface; quadratic H1 temperature retains curvature that its order-1 projection loses](../assets/field-basis-profiles.svg)

Both profiles are evaluated through the fields' actual bases along
`y = z = 0.5`, including the two separate L2 traces at x = 0.5. The dotted
vertical pressure segment marks the jump, not an interpolated connection.
The geometry and topology are identical for every curve. Quadratic H1
enforces shared values; it does not generally enforce matching derivatives
or [H² conformity](00-function-spaces.md#h2-is-different-from-second-order-h1).

Run the complete program and reproduce all three diagrams:

```bash
uv run python docs/examples/data_spaces.py --check
uv run --with matplotlib python docs/examples/data_spaces.py --check --figures /tmp/tack-space-figures
```

The optional plotting step writes standalone SVG and PNG files. The numerical
checks run without Matplotlib.

## Higher orders

On segments, quads and hexahedra (pixels and voxels too) a space may have any
order. `basis=` names its polynomials: Lagrange polynomials on the products
of 1D nodes along each axis, `"gauss_lobatto"` (the default, MFEM's for H1 and
Nek5000's), `"equispaced"` (VTK's Lagrange cells, Exodus, gmsh) or, for `L2`
alone, `"gauss_legendre"` (MFEM's default for DG), which has no nodes at a
cell's ends to share. Up to order 2 the first two are the same nodes.

`basis="bernstein"` holds the Bernstein polynomials' coefficients instead of
values at nodes: control points, for a geometry. They bound the cell's field or
shape (it lies within their convex hull) and are what VTK's Bezier cells store.
A corner's coefficient is its value, and an edge's or a face's depends on that
entity alone, so they are shared as nodal values are. `interpolate` and
`project` fill a Bernstein space from values at the equispaced nodes, and every
filter sees the same polynomials as the equispaced Lagrange space of the order.
Bernstein spaces are on segments, quads, hexahedra, triangles, tetrahedra and
wedges. `dataset_to_vtk(..., higher_order=True)` writes a dataset with them as
VTK Bezier cells, and `vtk_to_dataset` reads Bezier cells back into them.

```python
--8<-- "docs/examples/data_high_order.py:bernstein"
```

`basis="serendipity"` (order 2) is the full order-2 space less its values at quad
faces' centres and cells' insides, which its functions determine from the
corners and edges: VTK's quad8, hex20 and wedge15 cells, and Exodus's and
Abaqus's quadratic elements. On triangles and tetrahedra it is the full space.
It holds every quadratic of position on straight cells, and `vtk_to_dataset`
reads VTK's quadratic cells into it (the quadratic triangle and tetrahedron into
the full order-2 space); `dataset_to_vtk(..., higher_order="quadratic")` writes
them back.

A space numbers its values by entity -- points, then each edge's, face's and
cell's -- so cells sharing an edge or face share its values. `dofs=` numbers
them by an element-to-DOF table instead, as MFEM and VTK's cell grids store
them: a row per cell giving each of its nodes' value index, in Tack's node order
for the shape, the corners at the cell's points. `H1(data, order=3,
dofs=table)` holds the same polynomials as `H1(data, order=3)`, numbered as the
table says, so data numbered that way is used as it comes; every filter reads it,
and a filter's output numbers its values by entity again.

```python
--8<-- "docs/examples/data_high_order.py:functions"
```

```python
--8<-- "docs/examples/data_high_order.py:spaces"
```

An order-4 `H1` space on 64 hexahedra holds `17³ = 4913` values: each edge,
face and cell interior has its own, and cells turned any way read a shared
edge or face's values alike. A polynomial of degree 4 in each coordinate is
represented exactly, so the cubic comes back anywhere it is probed. Positions
can be a field of such a space too: a curved geometry, whose cells are located
and evaluated as they are, not as their corners make them.

Filters that cut -- `contour`, `slice`, `slice_plane` -- cut a field above
order 1, or a curved geometry, on linear sub-cells of the cells the cut can
reach: those whose Bernstein coefficients (which bound the field over the
cell) straddle the value.

```python
--8<-- "docs/examples/data_high_order.py:contour"
```

By default a cell is cut into as many steps along each axis as the field's
or geometry's order, and `tolerance=` asks for the fewest steps whose
estimated error is within it. Each crossing of a sub-cell's edge is where the
field itself takes the value along that edge, found from the field's own
values, so every vertex lies on the true surface (to rounding) at any number
of steps; the steps decide only how finely the surface is faceted.
`exact=False` puts it where the straight line between the lattice values does,
and `subdivision=1, exact=False` is the cells' corners alone. A point where
sub-cells of neighbouring cells meet is computed once, so the surface has no
cracks, and every piece of a partitioned dataset holding it computes the same
bits, so pieces' surfaces carry ghost marks and merge into the whole's.
`td.subdivide` gives the linear sub-cells of every cell, for export
or for a filter that takes linear cells only:

```python
--8<-- "docs/examples/data_high_order.py:curved"
```

`clip` and `threshold` cut the same way, but keep what they need not cut: the
cells whose Bernstein coefficients lie wholly on the kept side stay as they are,
curved and of their order, and only the cells the bound cannot decide go to
linear sub-cells. The result is a `PartitionedDataSet` of the two parts, whole
cells first:

```python
--8<-- "docs/examples/data_high_order.py:clip"
```

The parts share the lattice points of the faces between them, where a linear
sub-face lies against a curved face. The whole cells' corners there are the
whole part's: the cut part's copies are marked `DUPLICATEPOINT`, so reductions
over the two (`owned_values`, `statistics`) count each once. A dataset of straight cells and order-1
fields clips and thresholds as before, into one `DataSet`.

MFEM fields of any order on quads, hexahedra, triangles, tetrahedra and wedges
come across through `tack.interop.mfem`, and VTK's Lagrange cells (curves,
quads, hexahedra, triangles, tetrahedra, wedges) through `vtk_to_dataset`
(`dataset_to_vtk(..., higher_order=True)` writes them). Pyramids stop at order 2
for now.

### Where a cell's values sit

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

## A material interface exposes the difference

The [two-cell example](01-datasets.md) has material 1 on the left and 3 on the
right. Converting it to shared point data gives value 2 on the common face.
Converting it to `L2` preserves both sides of the discontinuity:

![Cell values, averaged shared point values, and separate corner values](../assets/data-spaces.png)

The figure is generated from the example's values along the bottom front edge;
the dashed line is the shared face at x = 0.5. The middle ramp is an effect of
projection and interpolation, not an interface resolved by the original data.
H1 and L2 both use order 1 in this figure.

```python
--8<-- "docs/examples/data_fields.py:spaces"
```

The returned objects are fields. Assign one to `data.fields[name]` if later
filters should carry it. Computing `point_material` alone does not add it to
the dataset's named fields.

## Projection is an algorithmic choice

`project(data, field, target_space)` is the one operation: `td.H1(data)`,
`td.Constant(data)` and `td.L2(data)` are the common target spaces.

| Conversion | Meaning | What to check |
|---|---|---|
| H1 → Constant | Each cell's average: its integral over the cell divided by the cell's measure | `average="nodes"` for VTK's mean of the cell's nodes |
| Constant → H1 | Mean of incident cell values at each point | Material discontinuities become shared values |
| H1 → L2 of sufficient order | Evaluate the continuous field at each cell's local nodes | Values remain equal on shared interfaces |
| L2 → H1 | Evaluate from adjacent cells, then average | Jumps can be smoothed |
| Constant → L2(order=1) | Repeat a cell's value at its own corners | Discontinuities remain |

For the affine temperature `x + 2y + 3z` on the two regular cells, the
averages are `[2.75, 3.25]`, the center values.

A cell's average and the mean of its nodes are different numbers. A cell's
value from `project` is the average: the field integrated over the cell
through the cell's own map, divided by the cell's measure. So each value
times its cell's measure sums to `integrate`'s integral, for a field of any
order on curved cells too. VTK's vtkPointDataToCellData and Viskores'
CellAverage take the mean of the cell's points instead. With
`average="nodes"`, `project` does the same: the mean of the field's own
values in the cell, its `NODES`. The two agree for a linear field on a
tetrahedron or on the boxes of an image. They differ elsewhere:

- On a pyramid, the apex weighs a quarter of a linear field's average, not a
  fifth.
- A distorted hexahedron's corners do not weigh alike.
- At higher order, x² on the unit cube averages to 1/3. Its corners' mean is
  1/2, and its 27 order-2 nodes' mean is 0.42.

Polyhedral, polygonal and polyline topologies support point/cell projection:
a cell's average is exact for a field linear on the pieces it is cut into.
`L2`, higher-order projection and traces require the shape-based path. In
particular, being able to attach point values to a polyhedron does not imply
that it supplies shape-function evaluation for every algorithm.

## Read both sides of a face

```python
--8<-- "docs/examples/data_fields.py:traces"
```

Projecting into `SideTraces` evaluates each cell at its faces' corners, keeping the two sides
separate and aligned in the face's ordering. `jump` evaluates a difference
at each face center: side 1 minus side 0, or zero at a boundary.

The material has one nonzero jump of magnitude 2. The affine H1 temperature
has zero jump. The sign of a material jump depends on the face's side ordering;
its magnitude does not. A face normal points out of side 0. These conventions
matter when assembling a flux and are developed in [Mesh Kernels](05-mesh-kernels.md).

For a normal-oriented face quantity, use `Values(data, "faces", oriented=True)`.
A filter that reverses a carried face then negates the value. Area and labels
are ordinary, unoriented face values and should not change sign.

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

## Know the present higher-order limits

[Cell Types and Bases](16-cells-and-bases.md) lists which spaces, orders and
bases each shape takes. Filters that keep whole cells (`extract_cells`,
`split`, `remove_ghosts`) keep higher-order fields and a curved geometry at
their order, and `extract_surface` keeps a solid's fields and
geometry on its surface. The cutting filters (`contour`, `slice`, `clip`,
`threshold`) cut curved and high-order cells on linear sub-cells, their
crossings at the field's roots; `subdivide` gives those sub-cells. The
[reference](reference/dataset-api.md#spaces) records the supported layouts and
[filter propagation rules](reference/dataset-api.md#algorithms-and-filters).

## Run and reproduce the figure

```bash
uv run python docs/examples/data_fields.py --check
uv run --with matplotlib python docs/examples/data_fields.py --check --output spaces.png
```

[Download the complete program](../examples/data_fields.py).
Continue with [Filtering and Analysis](04-filters.md).
