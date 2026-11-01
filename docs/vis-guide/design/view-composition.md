# View storage and composition

The [public view guide](../17-views.md) teaches the kernel interface. This
record describes its current implementation; generated class names and group
storage are diagnostics rather than stable public interfaces.

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
one of four ways, by the kind of node ([where each value is](../field-storage.md#h1-above-order-1-numbered-by-entity)):

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
[Work at a field's nodes](../05-mesh-kernels.md#work-at-a-fields-nodes) loops
over them.

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
