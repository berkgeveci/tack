# Build meshes

Build a dataset from a topology and a geometry field. This chapter constructs
each representation from its inputs, using `td.DataSet(topology, geometry)`
throughout. The inputs might come from a simulation, a reader, a procedural
model, or arrays you wrote yourself. [Factory methods](#factory-methods-as-shortcuts)
then provide shortcuts for common combinations.

The snippets assume NumPy, Tack and `tack.dataset` imported as `np`, `tack`
and `td`, with a default device set. The [complete mesh-construction program](../examples/data_meshes.py)
and [mixed-cell tour](../examples/datasets.py) can be downloaded and run.

## Topology defines corners, geometry defines positions

In this chapter, a topological point is a **corner vertex**. Cell shapes specify
the local corner convention, and the topology maps those corners to global
vertex ids. A space can add field nodes on edges, faces and interiors without
altering this corner topology. The geometry then supplies a coordinate mapping
in its own space; it is not a mandatory XYZ array stored by the topology.

A hexahedral topology therefore does not say whether the mesh is a uniform
image, stretched along coordinate axes, curvilinear, or curved through a
higher-order geometry. Those descriptions depend on the coordinate field.
See [the dataset model](01-datasets.md#corners-are-not-all-of-a-fields-nodes)
for the distinction between corner ids and field DOFs.

## Choose a representation

| Input | Topology to construct | Geometry values |
|---|---|---|
| Point dimensions, origin and spacing | `StructuredTopology` | `UniformCoordinates` |
| Three coordinate axes | `StructuredTopology` | `CartesianProduct` |
| A logical grid with explicit corner positions | `StructuredTopology` | A `tack.Array` with vector elements |
| Cell types, offsets and corner ids | `UnstructuredTopology` | A `tack.Array` with vector elements |
| A list of face rings for each cell | `PolyhedralTopology` | A `tack.Array` with vector elements |
| A point ring for each polygon | `PolygonalTopology` | A `tack.Array` with vector elements |
| A point chain for each line | `PolylineTopology` | A `tack.Array` with vector elements |

In each first-order example below, wrap the coordinates in
`td.Field(td.H1(topology), coordinates)` to define geometry, then assemble the
dataset. All planar examples still use 3-vector coordinates.

These choices affect algorithm support. A hexahedron has a reference element
for evaluating shape functions and gradients. An arbitrary polyhedron has
faces and points, but no parametric reference element. It can be contoured,
thresholded or measured, while locating particles and computing basis gradients
currently require shape-based cells. The [filter chapter](04-filters.md) gives
a task-oriented support table.

## Image, rectilinear and curvilinear data

Tack uses **`StructuredTopology` for both image data and rectilinear grids**.
It stores the logical point dimensions and derives a cell's corners from
indices, without an explicit connectivity row for every cell. There is no
separate `ImageTopology` or `RectilinearTopology` class.

What differs is the geometry field's **values**:

| Dataset kind | Topology | Geometry space | Geometry values |
|---|---|---|---|
| Uniform image data | `StructuredTopology` | H1 order 1 | `UniformCoordinates`: origin, spacing, optional direction |
| Rectilinear grid | `StructuredTopology` | H1 order 1 | `CartesianProduct`: three coordinate axes |
| Curvilinear structured grid | `StructuredTopology` | H1 order 1 | Explicit 3-vector positions for its corner vertices |
| Linear unstructured mesh | `UnstructuredTopology` | H1 order 1 | Usually explicit corner positions |
| Supported curved finite-element mesh | Its original corner topology | H1 order 2 | Coordinate coefficients on the geometry's higher-order nodes |

### UniformCoordinates: image geometry as a formula

For logical indices `(i, j, k)`, `UniformCoordinates` evaluates:

```text
position(i,j,k) = origin + direction @ ([i,j,k] * spacing)
```

This represents VTK-style image geometry, including a direction matrix for
an oriented image. With `nx * ny * nz` corners it needs only constant-size
coordinate metadata, not three stored components per corner. The matching
structured topology obtains the cell's corner ids from its dimensions.

Build the topology, geometry field and dataset explicitly:

```python
--8<-- "docs/examples/data_meshes.py:uniform"
```

“Image data” here describes sampling geometry. Temperature, labels or RGB
values remain separate named fields, each with its own space and storage.
An image's geometry being implicit does not make all of its solution values
implicit as well.

### CartesianProduct: rectilinear geometry from axis arrays

For a rectilinear grid, `CartesianProduct` evaluates:

```text
position(i,j,k) = [x_axis[i], y_axis[j], z_axis[k]]
```

Spacing can vary along each axis, but each coordinate depends on only one
logical index. Storing `nx + ny + nz` axis values describes `nx * ny * nz`
corner positions. The full Cartesian product is evaluated on demand.
`CartesianProduct` stores the axes on the backend rather than allocating an
expanded XYZ array. Assemble it with a structured topology and geometry space:

```python
--8<-- "docs/examples/data_meshes.py:rectilinear"
```

### Explicit positions: bend the grid without changing connectivity

A curvilinear structured grid retains structured corner incidence but supplies
each corner's 3-vector position explicitly. The coordinate field is still
first-order H1; its physical cells need not be axis-aligned. This is different
from an order-2 geometry, whose mapping also uses non-corner DOFs.

Generate the corner positions directly, upload them as vector elements,
and associate them with the structured topology:

```python
--8<-- "docs/examples/data_meshes.py:curvilinear"
```

The three examples have the same point dimensions and corner numbering.
They differ in their geometry values. A topology can also be reused by
multiple datasets with different geometry fields. Algorithms that read
geometry through a domain view use the same interface for implicit and
explicit coordinates.

## Structured dimensions count points

For point dimensions `(nx, ny, nz)` with every dimension at least two, the
number of cells is `(nx - 1) * (ny - 1) * (nz - 1)`. Thus `(4, 3, 2)` gives
24 points and six cells. With uniform coordinates, spacing is the distance
between adjacent points; the domain length in x is `(nx - 1) * spacing_x`.

`StructuredTopology` takes one to three active point dimensions, each at
least two: `(9, 5)` describes a 2D grid and `(3,)` a line with two cells.
The factories described below also accept singleton and empty dimensions.

### Match x-fast storage when uploading values

Structured dataset points use the flat index
`p = i + nx * (j + ny * k)`: x runs fastest. This differs from the native
row-major layout of a scalar Tack array of shape `(nx, ny, nz)`, whose last
dimension runs fastest.

To upload a NumPy volume indexed `host[k, j, i]`, make it C-contiguous and
flatten it; its shape is `(nz, ny, nx)`. An array indexed `host[i, j, k]`
needs to be rearranged first. A small function of x alone is a useful layout
check before uploading a full simulation. The [dataset example](01-datasets.md)
avoids this conversion by evaluating coordinates in a kernel.

## Unstructured connectivity uses offsets

The existing tour's mixed mesh has a hexahedron and a pyramid sharing a face:

```python
--8<-- "docs/examples/datasets.py:unstructured"
```

`types=[12, 14]` uses VTK's hexahedron and pyramid ids. The offsets
`[0, 8, 13]` delimit each cell's row in the connectivity array. The first
eight entries are the hex's points; the next five are the pyramid's. Offsets
include the final end, so there is one more offset than cells.

Connectivity entries are **point ids**, not byte addresses or scalar component
offsets. Their order must match the cell shape's corner convention. Supplying
the right point set in the wrong order can invert a cell or change its faces.
Use `num_points=` when the geometry contains points no cell uses.

## Construct polyhedra from face rings

This cube has six outward-wound face rings. Using a cube makes its volume
and boundary count easy to check before trying a more general polyhedron:

```python
--8<-- "docs/examples/data_meshes.py:polyhedron"
```

`[faces]` is a list of cells, each a list of its faces, each face a point ring.
Shared copies of a face are matched by their point sets and stored once.
With `orient=True`, positions let the builder repair inconsistent winding and
orient connected components outward. It is still the caller's responsibility
to provide meaningful connectivity and valid geometry.

`td.as_polyhedra(data)` converts a 3D shape-based dataset to the same face-based
model and carries its fields. It is useful for testing one face-based kernel
on both representations; it does not add a reference element to polyhedra.

## Polygons and polylines are different objects

Construct one polygon from a ring of four points and explicit H1 geometry:

```python
--8<-- "docs/examples/data_meshes.py:polygon"
```

Construct a polyline from a chain of three points with its own geometry:

```python
--8<-- "docs/examples/data_meshes.py:polyline"
```

The offsets delimit point rings for polygons and point chains for polylines.
The square encloses area; the open chain has length and two ends. Closing a
polyline geometrically does not turn it into a polygon. Streamline output is
a `PolylineTopology`, with one chain per seed that moved.

On a 2D mesh, faces are edges. On a polyline topology, faces are chain ends.
A shape-based mesh of line cells does not yet derive endpoint faces, so do
not assume every line representation supports the same face operations.

## Factory methods as shortcuts

The explicit constructions above show the components. For the common uniform
and rectilinear combinations, the factories assemble those same components:

```python
--8<-- "docs/examples/data_meshes.py:grids"
```

| Convenience call | Components it assembles |
|---|---|
| `uniform_grid(dims, origin, spacing, direction=...)` | Structured topology, H1 order-1 geometry and `UniformCoordinates` |
| `rectilinear_grid(x, y, z)` | Structured topology, H1 order-1 geometry and `CartesianProduct` |
| `DataSet(topology, coordinate_values)` | The given topology and an H1 order-1 geometry field wrapping the coordinates |
| `DataSet(topology, host_positions)` | The given topology and H1 order-1 geometry, after uploading the positions |

`wavelet` and `tangle` additionally generate sample scalar fields on uniform
grids. They are useful for examples and checks, while simulation output,
file readers and other generators can supply any of the same components.

The grid factories handle singleton dimensions: `(9, 5, 1)` has two cell
directions, and `(1, 3)` has three points and two line cells. An all-singleton
grid such as `(1, 1, 1)` has one point and no cells; any zero extent is empty.
For these empty or point-only cases, the factories use an empty
`UnstructuredTopology` while retaining the coordinate representation.

The runnable example checks the manually constructed and factory-built grids
against each other: topology dimensions, geometry representation, positions,
and cell measures agree.

## Plan ids and derived connectivity

A topology chooses `i32` for ids when they fit. Pass `id_dtype=tack.i64` to a
topology constructor when needed; oversized meshes also promote automatically
where the API supports them. The same id type covers offsets, connectivity
and derived entity ids. Arrays from NumPy or VTK use the narrowest type that
holds their ids unless another width is requested.

`topology.faces()` and `topology.edges()` build and retain derived connectivity.
Call them before a kernel that needs those relations. First-use derivation can
be a substantial part of a filter's first call; distinguish it from steady
execution when measuring a repeated pipeline.

## Run and check

```bash
uv run python docs/examples/data_meshes.py --arch cpu --check
uv run python docs/examples/datasets.py --arch cpu --check
```

The first program checks each explicit construction, the factory equivalence,
cube volume, polygon area, path length and the cube's six boundary faces.
The second checks the mixed mesh and its fields.
[Download the mesh-construction program](../examples/data_meshes.py).

Next, choose how values live on these entities in [Fields and Spaces](03-fields-and-spaces.md).
