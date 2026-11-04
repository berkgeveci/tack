# Write a mesh algorithm

When no built-in algorithm computes what you need per cell, per face or per
node of a mesh, write a [kernel](../core/kernels.md) and hand it to
`td.for_each`. It runs the kernel over the dataset with a [view](views.md) of
the mesh and of each field. The kernel never switches on the cell type and
never parses connectivity, so it runs on hexahedra, wedges, images and
polyhedra alike, on every device. Over the tutorial you will write six
kernels on one small mesh, and check each against an exact answer.

Read [Views](views.md) first for what a view is. The kernel syntax is Core's
([kernels](../core/kernels.md), [device functions](../core/device-functions.md));
the fields and spaces are the Data layer's ([fields](../data/fields.md)).
Tack's terms are in the [glossary](../reference/glossary.md).

## The mesh

A unit cube and, beside it, a wedge that shares the cube's face at x = 1. The
point field `T = x + 2y + 3z` is an `H1` field of order 1 (point data), linear,
so every answer below is known exactly:

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:data"
```

The cells are VTK's: cell types, offsets and connectivity, in VTK's corner
order. The complete program is
[vis3_mesh_algorithm.py](../examples/vis3_mesh_algorithm.py); every step
below is a section of it.

## Average a field over each cell's corners

The first kernel reads each cell's corner values through the field's view:

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:corner-mean"
```

Run it over the cells:

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:run-corner-mean"
```

Three things make this kernel work on any mesh:

- **`for_each` makes the views.** Its first argument after the dataset is the
  domain, here `"cells"`. The kernel's first parameter receives the domain
  view, and the `Field` `T` becomes the field view `u` for the same cells. The
  output array `means` passes through unchanged.
- **`NUM_POINTS` is a constant of the shape.** `for_each` launches the kernel
  once per shape: once for the hexahedron, where `NUM_POINTS` is 8, and once
  for the wedge, where it is 6. Each launch has its own compiled form.
- **`entity_id(c)` is where the output goes.** The loop variable `c` is the
  cell's place in its launch. `cells.entity_id(c)` is its id in the whole
  dataset, which is what an array covering every cell is indexed by.

A corner mean is a teaching algorithm, not a cell's average. The two agree
only for a linear field on a tetrahedron or on an image's box. For the true
average, `td.project(data, "T", td.Constant(data))` integrates through each
cell's map; see [Derive](derive.md).

## Write only the cells you need

A cell [set](../reference/glossary.md) is a domain too. Name one, and the
kernel visits only its cells:

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:subset"
```

The output still has room for every cell, because the kernel writes at global
ids. Cell 0 is never visited, so it keeps the zero `tack.zeros` gave it. With
`tack.empty`, its value would be whatever the memory held: use `empty` only
when every element is written before anything reads it.

Sets belong to their topology. A set from a filter's input cannot be used on
its output, whose cells are numbered afresh; the filter carries the set
across instead ([Write a filter](write-a-filter.md)).

## Evaluate a field inside a cell

A field view evaluates the field anywhere in a cell, given **parametric
coordinates**: a location in the cell's reference element. The domain view
maps the same coordinates into the world:

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:center"
```

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:run-center"
```

`cells.parametric_center()` is the centre of the reference element. At it,
`u.value(c, pc)` is the field's value and `cells.position(c, pc)` the world
position. `u.parametric_gradient(c, pc)` would give the derivative with
respect to the parametric coordinates. `td.world_gradient` turns that into the
gradient in the world through the geometry's Jacobian, as the built-in
`td.gradients` does. Both cells give (1, 2, 3), the gradient of `T`.

A parametric coordinate is not a world position. To sample a field at points
given in the world, locate them first with `td.probe`; see
[Sample and trace](sample-and-trace.md).

`td.world_gradient` is one of a few public [device functions](../core/device-functions.md)
for kernels like this one. `td.world_hessian` and `td.world_jet` give second
derivatives, and `td.hdiv_value` and `td.hcurl_value` map vector fields of
those spaces into the world.

## Walk each cell's faces

Every cell view can walk the cell's faces, each from the cell's side. This
kernel computes each cell's volume by the divergence theorem: the field
v = x has divergence 3, so its outward flux through the faces is three times
the volume.

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:volumes"
```

`cells.num_faces(c)` counts the cell's faces. `cells.side_num_points(c, k)`
and `cells.side_position(c, k, j)` walk face `k`'s points so that the face's
normal points out of this cell. The same face, read from the cell on its other
side, is walked the other way.

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:run-volumes"
```

Faces are derived once per topology, the first time something needs them.
Call `data.topology.faces()` before a kernel that walks faces, so the
derivation is not timed with the kernel.

The same kernel runs unchanged on polyhedra. `td.as_polyhedra` turns every
cell into a polyhedron of its faces. A polyhedron has no fixed shape, so its
view has no `NUM_POINTS`, but `num_faces`, `side_num_points` and
`side_position` work on every kind of cell. The volumes agree, 1 and 0.5.
The triangle fan is exact for planar faces. A face that is not planar needs a
deliberate choice of surface; for the built-in measures, use
`td.cell_geometry` and `td.face_geometry`.

## Loop over the faces themselves

`"faces"` is a domain too. A face has one or two sides: side 0, and side 1
unless it is on the boundary. Its normal points out of side 0. This kernel
takes the difference of a cell value across each face:

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:faces"
```

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:run-faces"
```

`faces.side_cell(f, s)` is the cell on side `s`. A boundary face has no
side 1, so the kernel reads it only when `faces.num_sides(f)` is 2. The mesh
has ten faces: six of the cube, five of the wedge, one of them shared. Only the
shared face has a difference, 3.5 - 3.

The corner means pass in as a plain array, indexed by cell id. A plain array is
not a field, so `for_each` passes it through as it is. Fluxes, divergence and
upwinding on faces have their own page,
[Finite-volume data and fluxes](finite-volume.md).

## Fill a field at its nodes

A field of a higher-order space has values at more places than the points. An
`H1` field of order 2 has values at the points, the edges' middles, the quad
faces' centres and the hexahedron's centre. Pass the space to `for_each` in
place of a domain, and the kernel visits each of these nodes once:

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:nodes"
```

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:run-nodes"
```

The kernel's view is a **node view**. `nodes.node_count(i)` is how many nodes
entity `i` holds, `nodes.node_dof(i, j)` is where node `j`'s value goes in the
space's array, and `nodes.node_position(i, j)` is where the geometry puts it.
A kernel over nodes takes arrays, not fields. This mesh's order-2 space has 36
nodes, and the kernel wrote each one once. The result is what
`td.interpolate(data, function, space)` gives for the same function; that
built-in is one such kernel.

## Work at a field's nodes

A field view numbers its own cell's nodes too: `q.NODES` of them, node `k`
at parametric coordinates `q.node_parametric(k)`, with value `q.dof(c, k)`
stored at `q.dof_index(c, k)`. That is enough to combine fields of different
spaces without projecting first. This kernel adds the order-1 field `T` to the
order-2 field `x * y`, into a new order-2 field:

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:field-nodes"
```

```python
--8<-- "docs/examples/vis3_mesh_algorithm.py:run-field-nodes"
```

`p.value(c, q.node_parametric(k))` reads `T` through its own basis at each of
`q`'s nodes. The same kernel runs on every shape and every space: 27 nodes on
the hexahedron, 18 on the wedge, an `L2` cell's own, a `Constant` cell's one at
its centre. Cells that share a node write the same value to the same place, so
the result does not depend on which cell writes last. It equals
`td.project(data, T, space)` plus `x * y`, since `T` lies in the order-2 space.

A loop over a field view's nodes visits a shared node once per cell holding
it, but has the cell's other fields at hand. A kernel over a space's nodes,
above, visits each stored value once.

## Keep each output to one writer

Every kernel above writes each output element from one iteration: a cell
writes its own value, a face its own. That needs no atomics, and the result is
the same on every device and every run.

A face kernel that adds to both of its cells is a scatter. Two faces of one
cell run at the same time and write the same element, so the sum needs atomics
or a separate gather pass. A gather is usually better: walk each cell's faces,
as the volume kernel does, and sum in the cell. Results then do not depend on
the order the faces ran in.

A barrier cannot order work across launches, and `for_each` makes one launch
per group. Keep stages that depend on each other's results in separate
`for_each` calls.

## Run it

```bash
uv run python docs/examples/vis3_mesh_algorithm.py --arch cpu --check
```

`--arch` takes `metal`, `cuda`, `hip` or `level_zero` too. `--check` holds every
result to its exact answer, or to the built-in algorithm that computes the same
thing.

## Details

- **Costs.** The first call of a kernel on a new shape compiles it, and the
  first face kernel derives the faces. Time warm calls separately from first
  ones. Changing values, positions or scalar arguments does not recompile; a new
  cell shape or space layout does.
- **Points.** There is no `"points"` domain. To work on each point, or on each
  value of a field alone, pass `field.values` to an ordinary kernel and loop
  over `range(len(values))`. A point set's ids are read the same way.
- **Bernstein fields.** In a Bernstein basis (`basis="bernstein"`), `dof(c, k)`
  is a coefficient, not the field's value at `node_parametric(k)`. A kernel
  that writes values there writes a Lagrange interpolant's values into a
  Bernstein field, which is right only where the two agree, as they do for
  multilinear fields.
- **Serendipity fields** store only their corners and edges. Their `NODES`
  are those stored nodes, which come first; the rest are computed when read.
- **The shape's nodes.** A view's `NUM_QUADRATIC` and `quadratic_node(k)` are
  the quadratic element's nodes, whatever the field. A loop over them is not a
  loop over a field's nodes: an order-1 field has fewer, an order-3 field more.

## Where to go next

- [Write a filter](write-a-filter.md) turns kernels like these into filters that
  return new datasets, with fields, sets, ghost marks and units carried along.
- [Finite-volume data and fluxes](finite-volume.md) builds face kernels into
  fluxes, divergence and upwinding.
- The [Dataset API reference](../vis-guide/reference/dataset-api.md#what-views-offer)
  lists every view method.
