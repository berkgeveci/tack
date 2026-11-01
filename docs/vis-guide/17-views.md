# Views: how a kernel sees a dataset

A kernel over a mesh asks questions in mesh terms: which points does this
cell have, where are they, what is the field's value here. A dataset answers
them from arrays, and the arrays differ from one mesh to the next. A **view**
is what stands between the two: a small object that holds a dataset's arrays
and answers those questions, compiled into the kernel. Every built-in filter
is written against views, and so are the kernels you write.

## A dataset is arrays; a view answers in mesh terms

Take the question "what is point `j` of cell `c`?":

- in an `UnstructuredTopology` it is `connectivity[offsets[c] + j]`, which a
  view reads from its shape's own copy
  ([What a group holds](#what-a-group-holds));
- in a `StructuredTopology` it is arithmetic on the cell's `(i, j, k)` index;
  nothing is stored;
- in a polyhedral topology it is read from a list of each cell's distinct
  points, derived from its faces the first time it is needed.

"Where is that point?" may read a stored array of positions, or compute it
from an image's origin and spacing. "What is the field's value there?"
depends on the field's space, its order and its
[layout](03-fields-and-spaces.md#how-the-values-are-stored).

A view hides all of that behind methods: `cells.point_id(c, j)`,
`cells.corner_position(c, j)`, `u.corner_value(c, j)`. It is an ordinary
Tack template, a [`@tack.device_class`](../users-guide/06-templates.md)
object holding the arrays it reads, so its methods are inlined into the
kernel and cost what the array reads they stand for cost.

## One kernel, every mesh

This kernel averages a field over each cell's corners. It does not know what
kind of mesh it runs on:

<!-- tack-check: views -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")


@tack.kernel
def corner_mean(cells, u, out):
    for c in cells:
        total = 0.0
        for j in range(cells.NUM_POINTS):
            total += u.corner_value(c, j)
        out[cells.entity_id(c)] = total / cells.NUM_POINTS


# A hexahedron, and above it a wedge on half of its top face.
points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                   [0, 0, 2], [1, 0, 2], [1, 1, 2]], np.float32)
topology = td.UnstructuredTopology(
    types=np.array([td.shapes.HEXAHEDRON, td.shapes.WEDGE], np.uint8),
    offsets=np.array([0, 8, 14]),
    connectivity=np.array([0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 8, 9, 10]),
    num_points=11)
data = td.DataSet(topology, points)
data.fields["T"] = td.Field(td.H1(data), tack.array(points[:, 2] * 10.0))

means = tack.zeros((data.num_cells,), tack.f32)
td.for_each(corner_mean, data, "cells", data.fields["T"], means)
assert np.allclose(means.numpy(), [5.0, 15.0])        # 8 corners, then 6

# The same kernel on an image: structured cells, positions computed, not stored.
image = td.uniform_grid((3, 3, 3))
image.fields["T"] = td.Field(td.H1(image), tack.array(image.positions()[:, 2]))
image_means = tack.zeros((image.num_cells,), tack.f32)
td.for_each(corner_mean, image, "cells", image.fields["T"], image_means)
assert np.allclose(image_means.numpy(), [0.5] * 4 + [1.5] * 4)
print(means.numpy(), image_means.numpy())
```

`cells` is a **domain view**: the cells, one shape at a time. `u` is a
**field view**: `T` read through its space for the same cells. On the mixed
mesh the kernel ran twice, once for the hexahedron, where `NUM_POINTS` is 8,
and once for the wedge, where it is 6. On the image it read no connectivity
and no stored positions, and its loop variable `c` was an `(i, j, k)` index
rather than an integer. The source is the same in every case.

## Three kinds of view

| View | Comes from | Answers |
|---|---|---|
| **Domain view** | `"cells"`, `"faces"`, `"edges"`, or a named set | the entities' points, positions, faces, sides, neighbours, shape constants |
| **Field view** | a `Field` passed to `for_each` | the field's value at a corner, at parametric coordinates, its parametric gradient, where a node is stored |
| **Node view** | a space passed to `for_each` | each of the space's nodes once, with where its value goes and where it is |

Domain and field views go together: the field view reads the entity the
domain view's loop is at, so `u.corner_value(c, j)` is the value at
`cells.point_id(c, j)`. Node views stand alone, and are how a field is
filled: each node, and so each stored value, is visited exactly once, whether
cells share it or not ([Visit a space's nodes](05-mesh-kernels.md#visit-a-spaces-nodes)).

## Where views come from: groups and launches

`td.for_each(kernel, data, domain, *args)` does three things:

1. **Splits the domain into groups** that one compiled kernel can serve:
   by cell shape, and further by anything that changes the code, such as an
   `L2` space whose order varies from cell to cell, or the size buckets of
   polyhedra.
2. **Makes the views for each group:** the domain view first, then each
   `Field` among `args` becomes its view for the same group. Anything else, a
   raw Tack array or a number, passes through unchanged.
3. **Launches the kernel once per group.**

The same steps are public, for an algorithm that needs to control them:

```python
T = data.fields["T"]
for group in data.launch_groups("cells", [T]):
    if group.count:
        cells = data.domain_view("cells", group)       # this shape's cells
        corner_mean(cells, T.view(group), means)       # T, for the same cells
```

`group.shape` is the cell shape and `group.count` how many cells it has.

## What a group holds

A group is more than a range of cell ids. The topology makes, once, the
arrays its views read, laid out so that every cell of the group has the
same shape and so the same counts:

- **Each cell's points.** An `UnstructuredTopology` stores VTK's flat form,
  every cell's point ids one after another at `offsets[c]`. Its groups
  gather them per shape, on the dataset's device, into rows of the shape's
  `NUM_POINTS` ids, with `ids`, each row's cell id. In a group's view the loop
  variable `c` is the row, `cells.point_id(c, j)` is `rows[c, j]`, and
  `cells.entity_id(c)` is `ids[c]`. When every cell has one shape and the
  offsets step evenly, the rows are the connectivity itself, reshaped; on a
  mixed mesh they are a second copy of it, kept with the topology. A
  structured grid's groups hold nothing: `c` is the cell's `(i, j, k)` and
  its points are arithmetic. A polyhedron's points and faces are read by id
  from the topology's own arrays, each group a bucket of cells of similar
  size.
- **Each cell's edges and faces.** `topology.edges()` and `topology.faces()`
  number the mesh's edges and faces once (by arithmetic on a structured grid,
  by sorting the cells' sides otherwise), and keep, for each cell and each of
  its local edges or faces, the global one with how the cell meets it. They
  are stored group by group, a fixed number a cell, so a view finds them at
  `index(c) * NUM_EDGES + e` from the group's start: `edge_id(c, e)` and `edge_sign(c, e)`, which
  way the cell runs along the edge; `face_id(c, f)`, `face_side(c, f)`, which
  side of the face the cell is on, and `face_orientation(c, f)`, where the
  cell's first point of the face is in the face's own row. They are made the
  first time a view needs them: a face kernel, or a field with values on
  edges and faces.

A field view adds only what its space owns, and reaches everything else
through these. A value of an `H1` order-2 field on a hexahedron is found in
one of four ways, by the kind of node ([where each value is](03-fields-and-spaces.md#h1-above-order-1-numbered-by-entity)):

| Node `k` | `u.dof_index(c, k)` |
|---|---|
| a corner | `cells.point_id(c, k)`: its point |
| on edge `e` | the number of points, plus `edge_id(c, e)` |
| on quad face `f` | `face_dofs[face_id(c, f)]`, the space's start for that face |
| the interior | `cell_dofs[entity_id(c)]`, the space's start for that cell |

`face_dofs` and `cell_dofs` are the space's: where each face's and each
cell's values start, since faces and cells of different shapes hold
different numbers of them. Every cell around an edge or face reads the same
entry, so they find the same value. Above order 2 an edge's or face's
several values are read in the edge's or the face's own frame, through
`edge_sign` and `face_orientation`, so that cells meeting it from different
directions agree on which is which.
[Work at a field's nodes](05-mesh-kernels.md#work-at-a-fields-nodes) loops
over them.

## What is compiled in, and what is not

A view's **class constants** are compiled into the kernel: `NUM_POINTS`,
`NUM_FACES`, `DIMENSION`, a field's `ORDER`. That is why
`range(cells.NUM_POINTS)` is a loop of known length, 8 for a hexahedron,
and why the kernel is compiled once per shape. A
view's **arrays** (connectivity, positions, values) are ordinary runtime
arguments: new data of the same kind runs the compiled kernel again without
recompiling.

The loop variable belongs to its view: an integer on unstructured cells, an
index vector on structured ones. Pass it to the view's methods; to index an
output array of the whole mesh, use `cells.entity_id(c)`, the entity's
global id. On a mixed mesh a group's first cell is not cell 0.

## Method names

A view's methods follow one rule per kind of answer, so a name says what
comes back:

| Ends in | Gives | Examples |
|---|---|---|
| `_id` | a global id | `entity_id(c)`, `point_id(c, j)`, `face_id(c, k)`, `side_point_id(c, k, j)` |
| `_position` | a position | `position(c, pc)`, `corner_position(c, j)`, `side_position(c, k, j)` |
| `_corner` | a shape's local corner number | `edge_corner(e, k)`, `face_corner(f, k)` |
| `num_` (prefix) | a count | `NUM_POINTS`, `num_faces(c)`, `side_num_points(c, k)` |

A field view adds `corner_value(c, j)`, `value(c, pc)` at parametric
coordinates and `parametric_gradient(c, pc)`. The
[Dataset API reference](reference/dataset-api.md#what-views-offer) lists
every method by kind of entity.

## How a view is put together

A view's class is generated for its group from parts, each a small class of
methods:

- the **entity kind**: cells, faces or edges;
- the **shape**: its constants, corner tables and shape functions;
- **addressing**: how a cell finds its points, whether by connectivity or by
  structured index;
- **storage**: how values are read, whether stored, implicit or by
  `(i, j, k)`;
- for a domain view, the **geometry**, and for a field view, the **space**
  (`H1`, `L2`, `Constant` and so on, of its order and basis).

The class's name lists its parts. On the mixed mesh above, the hexahedron's
domain view is `H1GeometryPointAddressExplicitStorageCornerBoxCurvedNewtonFaceWalk_CellsHexahedron`
and `T`'s view is `H1FieldPointAddressExplicitStorage_CellsHexahedron`. The
parts share one namespace, and a view refuses to build if two of them define
the same name differently, so every method has one meaning. The names of
these private classes are diagnostics, not an interface: what a kernel may
rely on is the methods.

Views are not nested inside a dataset object. Each field is its own argument,
viewed for the group being launched, which keeps a kernel's arguments
explicit and lets one kernel read any number of fields.

## Where to go next

[Write Mesh Algorithms](05-mesh-kernels.md) puts views to work: global ids,
selections, face fluxes, polyhedra. [Write a Filter](15-write-a-filter.md)
builds whole filters on them, and the
[Dataset API reference](reference/dataset-api.md#kernels) is the complete
list.
