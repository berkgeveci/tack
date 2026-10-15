# Write mesh algorithms

Use `td.for_each` when an operation is naturally expressed per cell, face or
edge. Tack supplies a view of the mesh and compatible views of its fields.
Your kernel can use geometry and interpolation without manually switching on
cell type or maintaining a connectivity parser for every representation.

Read [Device Functions](../05-device-functions.md) and
[Templates](../06-templates.md) when the kernel syntax is new. The built-in
filters use the same dataset API as the examples here.

## Separate an algorithm from storage layout

This kernel computes each cell's mean corner temperature:

```python
--8<-- "docs/examples/data_mesh_kernels.py:cell"
```

`cells` is a domain view. `temperature` is a field view, so
`temperature.corner_value(c, j)` resolves through that field's space and storage.
It can read a stored value or evaluate an implicit array.

The dispatcher launches once per compatible group: usually each cell shape,
with further grouping for varying space order or size buckets. `NUM_POINTS`
is consequently a constant for this kernel specialization. It is eight for
a hex and five for a pyramid; the source stays the same.

This corner mean is a teaching algorithm. It is not a general quadrature
rule for integrating a field over an arbitrary cell.

## Use global ids for global output

The loop variable `c` belongs to its view. On structured cells it can represent
several indices rather than an ordinary integer. Pass it to view methods;
use `cells.entity_id(c)` when indexing output storage allocated for the entire
mesh. A group-local index is not an output cell id on a mixed mesh.

`td.for_each(kernel, data, "cells", *args)` supplies the domain view as the
first argument. Each `td.Field` in the remaining arguments becomes a matching
field view. A raw Tack field stays a raw buffer; the kernel is responsible for
indexing it correctly.

## Run over a selection and initialize unwritten output

On the [two-cell mesh](01-datasets.md), select just the right cell:

```python
--8<-- "docs/examples/data_mesh_kernels.py:subset"
```

The output has room for **all** cells because the kernel writes global ids.
`tack.zeros` initializes the cell that this selected launch will not visit.
The result is `[0, 3.25]`. Uninitialized allocation would be appropriate when
every element is written before anything reads it.

A cell or face set's name is a domain. For a point set, iterate its ids in an
ordinary kernel; point sets do not currently supply a dataset domain view.
Sets belong to their topology, so a set from an earlier filter input cannot
be substituted for a set on its renumbered output.

## Parametric and world coordinates have different meanings

`cells.parametric_center()` returns a location in the cell's reference element.
`temperature.value(c, pc)` evaluates the field there, and
`cells.position(c, pc)` maps it into physical space. A parametric coordinate
is not a world-space `[x, y, z]` query point.

Likewise, `temperature.parametric_gradient(c, pc)` differentiates with respect
to the reference coordinates. A physical gradient also needs the geometry's
Jacobian mapping. Use `td.gradients` for the built-in physical calculation;
use [probing](06-pipelines.md#sample-at-physical-positions) when query positions
are given in world coordinates.

## Compute one oriented flux per face

A face has side 0 and possibly side 1. Its normal points out of side 0;
`faces.num_sides(f) == 1` identifies a boundary face. Reading a neighbouring
cell must be guarded when that side is absent.

For the vector field `v(x,y,z) = (x,y,z)`, the following kernel computes its
integrated normal flux, `dot(v, normal) * area`, at each face's center:

```python
--8<-- "docs/examples/data_mesh_kernels.py:face"
```

Geometry, centers and output values are raw device fields here, indexed by
the face's global id. Each iteration owns one output, so it needs no atomic.
The normal and center fields use the same topology's face numbering.

```python
--8<-- "docs/examples/data_mesh_kernels.py:balance"
```

`td.divergence` is an **outward sum of the supplied face values**: a face adds
to its side-0 cell and subtracts from its side-1 cell. It does not divide by
cell volume. With integrated face flux as input, its result is integrated
net outflow. Divide by volume if you want an average divergence.

Each example cell has volume 0.5, net outflow 1.5, and therefore average
divergence 3. The sum of both cells' outflows is the boundary outflow 3;
the internal face cancels. This independent balance catches orientation and
association errors that a plausible image might hide.

For a scalar transported by a constant velocity, `td.upwind_flux` provides
the face calculation and returns an oriented field. Its boundary convention
uses the one existing cell's value; a solver needing prescribed inflow must
supply its own boundary treatment.

## Share face-based code with polyhedra

Shape-based cells have compile-time corner counts. Polyhedra instead expose
runtime `num_points(c)` and `num_faces(c)` counts. The following surface-area
kernel walks each cell's outward faces and works on both paths:

```python
--8<-- "docs/examples/datasets.py:faces"
```

The complete program compares its shape-based and polyhedral results. The
triangle fan is appropriate for the example's planar convex faces; general
nonplanar or nonconvex faces require a deliberate geometric interpretation.
For supported built-in measures, prefer `td.cell_geometry` and `td.face_geometry`.

## Keep parallel ownership explicit

The safe pattern is one writer per cell or face output. A face kernel that
adds to both neighbouring cells is a scatter; concurrent faces can target
the same cell and require atomics or a separate gather pass. A barrier cannot
order all mesh groups. Keep globally dependent stages in separate launches.

Allocate and derive topology before the repeated calculation. Time first-use
group construction and compilation separately from warm calls. Scalar model
parameters can change between calls without changing the source, but a new
cell shape or space layout can require another specialization.

## Run and check

```bash
uv run python docs/examples/data_mesh_kernels.py --arch cpu --check
uv run python docs/examples/datasets.py --arch cpu --check
```

[Download the cell/face kernel program](../../examples/data_mesh_kernels.py).
The [reference](../../reference/dataset-api.md#kernels) lists the full view methods.
Continue with [Sampling, Rendering and Simulation Pipelines](06-pipelines.md).
