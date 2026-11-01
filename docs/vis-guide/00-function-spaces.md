# Function spaces and higher-order geometry

Before choosing an array layout, decide what kind of function the data should
represent. A continuous temperature, a pressure that may jump between cells,
and a material label can live on the same mesh without using the same sample
locations, number of values or interpolation rule. A **space** makes that
choice explicit for each field.

This optional chapter develops the finite-element vocabulary introduced in
[Fields and Spaces](03-fields-and-spaces.md). You do
not need to assemble a finite-element system to use it: the same choices
determine what a contour, probe or gradient can mean.

## An array becomes a function through a basis

On a cell, a finite-element field is evaluated from coefficients and basis
functions. Schematically, in reference coordinates `xi`:

```text
u_h(xi) = sum_a U_a * phi_a(xi)
```

`U_a` are the stored coefficients, also called degrees of freedom (DOFs).
`phi_a` are the basis functions. The space supplies the basis and the map
from a cell's local coefficient numbers to the field's coefficient array.
The geometry maps `xi` to a physical position.

In Tack's nodal H1 and L2 representations, these coefficients are values at
the space's interpolation nodes. An interpolation node need not be a mesh
corner. Other finite-element families can use integral or flux DOFs instead;
the term DOF does not universally mean a point sample. MFEM explains this
distinction in its [discussion of grid functions and coefficients](https://mfem.org/pri-dual-vec/).

Separate three questions:

| Question | What answers it |
|---|---|
| Which cells meet, and which corner ids belong to each cell? | Topology |
| Where is a point of a reference cell in physical space? | Geometry field |
| Which coefficients determine a particular field in that cell? | That field's space and values |

## What L2, H1 and H2 mean

The names come from mathematical function spaces. They describe regularity,
not the degree of a polynomial:

| Mathematical space | Requirement | Consequence for common finite elements |
|---|---|---|
| L² | The function is square-integrable | Cellwise functions may have jumps |
| H¹ | The function and its first weak derivatives are square-integrable | Standard conforming nodal elements match field traces across shared interfaces; gradients may jump |
| H² | The function and weak derivatives through order two are square-integrable | A conforming piecewise-smooth construction must also control first-derivative matching |

A weak derivative extends differentiation beyond functions with a classical
derivative everywhere. A continuous piecewise-linear function can have a
different slope on each cell and still belong to H¹. A jump in the function
itself generally prevents that H¹ conformity. MFEM's
[weak-form introduction](https://mfem.org/fem_weak_form/) explains why this
distinction matters for finite-element derivatives.

Mathematically, H² is contained in H¹, which is contained in L². Thus “an L2
field” in finite-element software means the representation **permits** jumps;
it does not mean that every represented function must be discontinuous.
Likewise, the general mathematical definition of H¹ is not simply “a
classically continuous function everywhere”; matching nodal values is the
construction used by the finite elements discussed here.

### H2 is different from second-order H1

`td.H1(data, order=2)` is a quadratic **H1** space. It has more coefficients
per cell than order 1, but only H1 continuity is enforced between cells.
Its first derivatives can still jump there. It is not an H2-conforming space.
Tack currently has no `td.H2` family.

Use `order=2` when you mean a higher-degree approximation. Use the name H²
when you mean the stronger mathematical regularity. Differentiating a
polynomial twice inside one cell does not, by itself, establish H² conformity
across the mesh.

## Family, polynomial order and value width are separate choices

| Choice | Examples | What it controls |
|---|---|---|
| Family | H1, L2 | Sharing of DOFs and continuity between cells |
| Order | 0, 1, 2 | Richness of the local polynomial basis |
| Value width | Scalar, 3-vector | Number of components at each DOF |
| Storage | `tack.Array`, implicit array | How the coefficient values are supplied |

An H1 velocity with three components and an H1 scalar temperature can use
the same scalar nodal layout, with three coefficients per node for velocity.
A vector of H1 components does not automatically become an H(div) or H(curl)
finite-element field; those families have different continuity and mapping
rules. MFEM supports those families, but Tack's present dataset spaces do not
provide their finite-element bases.

### H1: share nodal values between adjacent cells

`td.H1(data)` defaults to order 1: one coefficient at each topological vertex.
Adjacent cells read the same coefficient on a shared vertex. Their compatible
shape functions give a matching field trace across the common interface.

On triangles and tetrahedra the order-1 basis is linear. On quadrilaterals
and hexahedra it is bilinear or trilinear: linear in each reference coordinate.
Consequently, “order 1” does not mean an affine function on every cell shape.

Order 2 adds nodes on edges and, where required by the shape, faces and cell
interiors. A shared edge's extra DOF is shared too. Adding those field DOFs
does not add corner ids to the topology or subdivide its cells.

### L2: keep each cell's coefficients separate

`td.L2(data, order=1)` gives every cell its own values at its corners.
Two neighbouring cells can therefore assign different values to the same
physical interface location. This is the representation used for linear
discontinuous Galerkin data in Tack.

With `order=0`, one coefficient supplies a constant value inside each cell.
L2 order 0 can still jump from one cell to the next. Uniform higher orders
are supported as listed in [Cell Types and Bases](16-cells-and-bases.md).
When order is a per-cell array, its currently supported entries are 0 and 1.

### Constant: one independent constant per cell

`td.Constant(data)` is Tack's direct representation of piecewise-constant
cell data, with the same local function meaning as L2 order 0. It is useful
for cell averages and material parameters and works on topologies that do
not have reference elements.

This is different from `tack.arrays.ConstantArray(value, size)`, which is a storage
representation returning the **same value at every index**. A field in
`Constant(data)` can have different values in every cell. Conversely, a
`ConstantArray` can supply all coefficients of an H1 field or an L2 field
without changing either field's space.

`tack.constant` is a third concept: a numerical constant declared for compiled
kernel code. It does not define a dataset field or a function space.

### Values: association without a finite-element basis

`td.Values(data, "faces")` describes one attribute per face, such as its area,
id or an already-integrated flux. It supplies no interpolation basis in the
cell interior. The analogous point, cell and edge associations work the same
way. A `Values` field on cells and a `Constant` field can have the same length
while supporting different operations: only the latter declares a function
that is constant inside each cell.

## Count DOFs separately from corners

Consider two line cells with three corner vertices, at x = 0, 0.5 and 1.
The topology's cells reference corner pairs `(0, 1)` and `(1, 2)`.

| Space | Scalar DOFs | Sharing |
|---|---:|---|
| H1 order 1 | 3 | The middle vertex is shared |
| H1 order 2 | 5 | Add one midpoint per cell; keep the shared vertex |
| L2 order 1 | 4 | Each cell has its own two endpoint values |
| L2 order 0 / Constant | 2 | One value per cell |

These line-cell spaces are supported; the table illustrates actual DOF counts.

For the supported example of **two hexahedra sharing one face**, the topology
has 12 corners. An order-2 hex has 27 nodes; the shared face contributes nine
shared nodes, giving `27 + 27 - 9 = 45` global H1 order-2 DOFs. L2 order 1
instead has `8 + 8 = 16` independent coefficients. Constant has two.
Geometry can stay in H1 order 1 throughout.

![Front faces of two adjacent hexes: H1 shares interface nodes, quadratic H1 adds nodes, and L2 keeps a separate coefficient on each side](../assets/field-dof-layouts.svg)

This is a front-face view (`z = 0`) of the **3D hex mesh**, not a 2D
quadratic-element example. The counts include nodes on the other z layers.
Green diamonds are single H1 DOFs shared by both cells. The L2 markers at
the interface are drawn slightly apart for visibility; their physical
positions coincide and the geometry remains connected.

Follow [Construct an L2 field](create-fields.md#construct-an-l2-field)
and [Construct a quadratic H1 field](create-fields.md#construct-a-quadratic-h1-field)
for allocation, initialization, node ordering and checked evaluations.

The [independent-spaces program](#each-field-has-its-own-space)
checks these counts and evaluates a genuinely quadratic temperature through
its own basis. `data.num_points` remains 12; use `field.space.size` to allocate
that field's coefficient array.

## What Tack currently implements

The [capability catalog](16-cells-and-bases.md#what-each-shape-supports) is the
authority for supported shape/space/basis combinations. Higher uniform orders
are available on segments, triangles, quads, tetrahedra, hexahedra and wedges,
subject to basis restrictions. Vertex and pyramid retain their listed
low-order support. H2, H(curl) and H(div) are not currently provided.

[Polyhedra, polygons and lines](18-polyhedra.md) use piecewise-linear
reconstructions rather than a single reference element.

## The MFEM model: mesh, space, grid function

In MFEM, a `Mesh` supplies elements and their incidence. A finite-element
collection specifies a family, order and basis; a `FiniteElementSpace` on the
mesh determines the local-to-global DOF maps and component layout. A
`GridFunction` couples that space with coefficients. This is why an MFEM field
is more than an array of values attached to the mesh's vertices. See
[MFEM's grid-function explanation](https://mfem.org/pri-dual-vec/).

Tack follows that separation for the supported parts of its data model:

| MFEM concept | Tack counterpart | Responsibility |
|---|---|---|
| Mesh element types and vertex connectivity | `data.topology` | Corner incidence and derived entities |
| Finite-element family, order and layout on the mesh | Each field's `space` | Basis, DOF ownership and local-to-global addressing |
| Grid function | `td.Field(space, values)` | Coefficients interpreted through that space |
| Mesh nodes describing geometry | `data.geometry`, itself a `td.Field` | Reference-to-physical coordinate mapping |

This is a conceptual correspondence, not a claim that Tack implements all
MFEM finite-element families or its parallel true-DOF and constraint machinery.
The present importer supports the H1/L2 families and orders described below.

### Corners are not all of a field's nodes

A linear hexahedron's topology names eight corners. A quadratic H1 field on
that same cell needs 27 nodal coefficients: corners, edge midpoints, face
centers and a cell-interior node. Its space obtains shared edge and face
incidence from the topology and builds the field's DOF map. The topology still
has the original eight corners for that cell; the field has not refined it.

Consequently, `data.num_points` counts topological vertices. It need not equal
`field.space.size`, and it need not be the number of geometry coefficients
on a curved or discontinuous mesh. Use the field's space when allocating its
values or interpreting their layout.

## Each field has its own space

One mesh can carry all of the following at once:

| Field on two adjacent hexahedra | Space | Logical values |
|---|---|---:|
| Geometry | H1 order 1, 3-vector positions | 12 vectors |
| Temperature | H1 order 2 | 45 scalars |
| Pressure | L2 order 1 | 16 scalars |
| Material parameter | Constant | 2 scalars |

The fields share **topology**, not necessarily a space, order, coefficient
count or storage representation. Compatible fields may deliberately share
one interned space object; “each field has its own space” means that the
choice belongs to the field, rather than being fixed once for the dataset.

For step-by-step recipes and node diagrams, see
[Construct an L2 field](create-fields.md#construct-an-l2-field) and
[Construct a quadratic H1 field](create-fields.md#construct-a-quadratic-h1-field).
The program below combines them on one dataset, using `uniform_grid` as a
shortcut for the mesh setup so we can focus on independent field spaces.
Temperature represents
`x² + 2y + 3z`; pressure is `x` in cell 0 and `x + 2` in cell 1, with a jump
at their shared face. Geometry remains an implicit uniform coordinate array
in first-order H1:

For the independent example, start a script with:

```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")
```

```python
--8<-- "docs/examples/data_spaces.py:sample"
```

```python
--8<-- "docs/examples/data_spaces.py:l2-kernel"
```

```python
--8<-- "docs/examples/data_spaces.py:spaces"
```

`interpolate` evaluates the temperature function at the **physical positions
of that space's interpolation nodes**, through the geometry; it does not
replace `data.geometry`. The original geometry, temperature and pressure
continue to use their separate spaces.

An evaluation combines two maps. The geometry gives physical position from
reference coordinates, while the solution field uses its own basis and DOFs
at those reference coordinates. A field view supplied to `for_each` preserves
that distinction even when geometry and solution orders differ.

```python
--8<-- "docs/examples/data_spaces.py:evaluate"
```

The quadratic field's center values are `[2.5625, 3.0625]`. Replacing it by an
order-1 projection gives `[2.625, 3.125]`: the topology and geometry are unchanged,
but the field approximation changed. The complete example also checks the
physical gradients against `[0.5, 2, 3]` and `[1.5, 2, 3]`.

```bash
uv run python docs/examples/data_spaces.py --arch cpu --check
```

[Download the independent-spaces program](../examples/data_spaces.py).

## Geometry is a field, too

The coordinate mapping has a family, order and values just as a solution does.
Straight cells can use order-1 H1 geometry with explicit corner positions or
implicit coordinates. A supported curved mesh uses H1 order-2 coordinate
coefficients; a solution on it may still use a different order.

With L2 geometry, neighbouring cells can assign separate positions to their
local corners while sharing corner ids in the topology. This can represent
gaps or detached cells without changing the incidence. Contour and slice
respect each cell's geometry; not every algorithm supports that representation.

The geometry's values can be a `UniformCoordinates`, a `CartesianProduct`,
or explicit position storage. Those are **value-array representations**, not
different finite-element spaces. Image and rectilinear geometry are explained
in [Build Meshes](02-meshes.md#image-rectilinear-and-curvilinear-data).

For origins and world positions, see
[Coordinates, Units and Time](12-units.md#relative-and-world-positions).

## Higher orders

The following independent example uses [data_high_order.py](../examples/data_high_order.py).
Set up its field before the basis comparisons:

```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")
```

```python
--8<-- "docs/examples/data_high_order.py:functions"
```

```python
--8<-- "docs/examples/data_high_order.py:spaces"
```

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





An order-4 `H1` space on 64 hexahedra holds `17³ = 4913` values: each edge,
face and cell interior has its own, and cells turned any way read a shared
edge or face's values alike. A polynomial of degree 4 in each coordinate is
represented exactly, so the cubic comes back anywhere it is probed. Positions
can be a field of such a space too: a curved geometry, whose cells are located
and evaluated as they are, not as their corners make them.


## Continue with an advanced task

[Create nodal or DG fields](create-fields.md), [attach a solver's layouts](field-storage.md),
[import MFEM/VTK data](connect-simulation-data.md), or
[cut curved and higher-order fields](04-filters.md#cut-higher-order-fields-and-curved-geometry).

??? info "Related detailed sections"
    These topics are developed in their own chapters.

    <span id="function-spaces-and-degrees-of-freedom" data-guide-forward="#function-spaces-and-higher-order-geometry"></span>
    [Function spaces and degrees of freedom](00-function-spaces.md#function-spaces-and-higher-order-geometry).
