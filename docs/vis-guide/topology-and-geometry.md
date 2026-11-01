# Topology and geometry

The [first dataset](01-datasets.md) has two cells and twelve vertices. Its
topology says which vertices belong to each cell. Its geometry says where
they are. Changing one does not necessarily change the other.

## Topology defines corners, geometry defines positions

A cell's **corners** are topological vertices. A hexahedron has eight, in a
defined local order. The topology maps those corners to vertex ids. Moving
a corner changes positions without changing incidence; reconnecting the
same positions changes the cells without changing the coordinate values.

Geometry is a field mapping positions within each cell to physical space.
First-order geometry uses corner coefficients. Higher-order geometry can
add coefficients on edges, faces and interiors. Those extra field nodes
do not add corner vertices to the topology.

!!! tip "First reading"
    Start with straight cells and their corner positions. After
    [Fields and Spaces](03-fields-and-spaces.md), the optional
    [higher-order chapter](00-function-spaces.md#geometry-is-a-field-too)
    develops the coordinate mapping.

## Choose a representation

| Input structure | Topology | Geometry values |
|---|---|---|
| Uniform logical grid | `StructuredTopology` | `UniformCoordinates`: origin, spacing, direction |
| Variable spacing along coordinate axes | `StructuredTopology` | `CartesianProduct`: one array per axis |
| Logical grid with bent cells | `StructuredTopology` | Explicit corner positions |
| Arbitrary connectivity and cell shapes | `UnstructuredTopology` | Explicit positions or another supported coordinate array |
| Cells described by face rings | `PolyhedralTopology` | Vertex positions |
| Polygon rings or polyline chains | `PolygonalTopology` or `PolylineTopology` | Vertex positions |

`UniformCoordinates` is a value-array representation, not another topology
or function space. [Build Meshes](02-meshes.md) constructs these combinations.

## Image, rectilinear and curvilinear data

All three use structured incidence: logical neighboring indices determine
cells without storing a connectivity row for each one. Coordinate values differ.

### UniformCoordinates: image geometry as a formula

```text
position(i,j,k) = grid_origin + direction @ ([i,j,k] * spacing)
```

Constant-size coordinate metadata describes the whole grid. A direction
matrix can orient it; structured incidence does not imply axis alignment.

### CartesianProduct: rectilinear geometry from axis arrays

```text
position(i,j,k) = [x_axis[i], y_axis[j], z_axis[k]]
```

Spacing may vary along an axis. Keeping `nx + ny + nz` axis values describes
`nx * ny * nz` positions.

### Explicit positions: bend the grid without changing connectivity

A curvilinear grid stores corner vectors but keeps logical connectivity.
This is different from higher-order curved geometry, which also uses
non-corner coefficients. Geometry and solution orders can differ.

## Points, edges, faces and cells

A 3D cell's faces are surfaces; a 2D cell's faces are edges; polyline faces
are chain ends. A vertex cell is a point. Entity dimension depends on the mesh.

`topology.edges()` and `topology.faces()` derive shared entities from incidence
when needed and retain them for reuse. This first-use work may be substantial.
An internal face has two sides; a boundary face has one.

```mermaid
flowchart LR
    C0[Cell on side 0] --> F[Shared face]
    F --> C1[Cell on side 1]
    F -. normal points out of side 0 .-> C1
```

A normal flux is positive out of side 0 and negative out of side 1.
Area and pressure have no sign change under reversal.
[Fields and Spaces](03-fields-and-spaces.md#face-values-and-orientation)
explains the declaration; [the FV workflow](18-finite-volume.md) uses it.

Coarse/fine faces are currently not paired into one interface. Face-based
operations may treat a T-junction as boundary; see the
[nonconforming-mesh limitations](reference/interop.md#reading-and-writing-other-formats).

## Local numbers, entity ids and coefficient ids

Corner 0 of one cell need not be vertex 0 of the mesh. A local corner number
selects an incidence entry; its entity id indexes topology vertices. A field
can add another map from cell nodes to coefficients. Keep those numbers distinct.

Structured point dimensions count vertices. If all three dimensions are at
least two, cell count is `(nx - 1) * (ny - 1) * (nz - 1)`. Point ids run x
fastest. The [NumPy upload recipe](02-meshes.md#match-x-fast-storage-when-uploading-values)
and [layout tables](field-storage.md) give exact addresses. Derived face ids
are stable for a topology, not automatically shared with another solver.

## Beyond one ordinary mesh

[Point clouds](point-cloud-model.md) use implicit vertex cells.
[Collections](collections-and-partitions.md) hold pieces,
[AMR](10-amr.md) holds refined boxes, and [octrees](11-octrees.md) hold tree
nodes. These require their own representation rules.

Continue with [Fields and Spaces](03-fields-and-spaces.md), or
[Build Meshes](02-meshes.md) to construct input.
