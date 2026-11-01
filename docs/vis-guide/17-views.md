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
  ([What a group holds](design/view-composition.md#what-a-group-holds));
- in a `StructuredTopology` it is arithmetic on the cell's `(i, j, k)` index;
  nothing is stored;
- in a polyhedral topology it is read from a list of each cell's distinct
  points, derived from its faces the first time it is needed.

"Where is that point?" may read a stored array of positions, or compute it
from an image's origin and spacing. "What is the field's value there?"
depends on the field's space, its order and its
[layout](field-storage.md#how-the-values-are-stored).

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



## Implementation detail

For exact group storage, DOF addressing and generated class composition, see
[View Storage and Composition](design/view-composition.md). Public kernels
depend on the methods above, not those private class names.

## Where to go next

[Write Mesh Algorithms](05-mesh-kernels.md) puts views to work: global ids,
selections, face fluxes, polyhedra. [Write a Filter](15-write-a-filter.md)
builds whole filters on them, and the
[Dataset API reference](reference/dataset-api.md#kernels) is the complete
list.

??? info "Related detailed sections"
    These topics are developed in their own chapters.

    <span id="what-a-group-holds" data-guide-forward="../design/view-composition/#what-a-group-holds"></span>
    [What a group holds](design/view-composition.md#what-a-group-holds).

    <span id="how-a-view-is-put-together" data-guide-forward="../design/view-composition/#how-a-view-is-put-together"></span>
    [How a view is put together](design/view-composition.md#how-a-view-is-put-together).
