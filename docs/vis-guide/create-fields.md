# Create and convert fields

Use this chapter to attach existing values, initialize coefficients, or choose
a new representation of a field. [Fields and Spaces](03-fields-and-spaces.md)
explains the choices; [storage layouts](field-storage.md) gives exact indices.

## Attach point and cell values

The [two-cell dataset](01-datasets.md#a-simpler-first-order-field) starts with
point temperature and cell material. Wrap a Tack value array in the space of
the dataset it belongs to, then name the returned field in `data.fields`.
Use `field.space.size` for allocations; equal array lengths alone do not
establish that two fields belong to the same topology.

Begin with [first-order point/cell conversion](#a-material-interface-exposes-the-difference)
if you have ordinary simulation arrays. The DG and quadratic recipes below
are complete advanced examples; they share the [independent-spaces program](../examples/data_spaces.py).

## Fields that share a name

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

## A material interface exposes the difference

Begin this recipe with the checked two-cell field setup below. It is
independent of the same-name example above.

```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")
```

```python
--8<-- "docs/examples/data_fields.py:sample"
```

```python
--8<-- "docs/examples/data_fields.py:build"
```

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
before this upload, as in [MFEM import](connect-simulation-data.md#import-mfem-fields).

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


## Run and reproduce the figure

```bash
uv run python docs/examples/data_fields.py --arch cpu --check
uv run python docs/examples/data_spaces.py --arch cpu --check
uv run python docs/examples/data_high_order.py --arch cpu --check
```

The optional `--output` argument on `data_fields.py` plots the material
interface. Numerical checks do not require Matplotlib. Continue with
[Filter and Analyze](04-filters.md) or [Sample and Trace Fields](06-pipelines.md).
