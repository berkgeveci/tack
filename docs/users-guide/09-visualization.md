# Datasets and visualization

`tack.dataset` holds meshes and the fields on them, and the algorithms that work
on both: contour, slice, clip, threshold, external faces, gradients,
projections between spaces, streamlines and the rest. Like VTK and Viskores,
it works on structured grids, unstructured meshes of the linear cells,
polyhedra, polygons and polylines. Unlike them, the algorithms are ordinary
Tack kernels, so they run on every backend, and the kernels you write
yourself see a mesh the way the built-in ones do.

```bash
uv sync --extra cpu      # in a clone; both packages are installed
```

This overview is a tour; the chapters below develop the data model and complete
workflows. The [Dataset API reference](../reference/dataset-api.md) has the
signatures and full support tables.

## Quick start

For this first example, use `uniform_grid`, a **shortcut factory** that
assembles a structured topology and uniform geometry for you. Add a scalar
field, then extract an isosurface. [Building a dataset](#building-a-dataset)
below shows how to construct these pieces explicitly for several mesh types.

```python
import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((17, 17, 17), origin=(-1, -1, -1), spacing=(0.125, 0.125, 0.125))
positions = data.positions()                   # corner positions as a NumPy array
values = tack.array((positions * positions).sum(axis=1), dtype=tack.f32)
data.fields["radius_squared"] = td.Field(td.H1(data), values)

surface = td.contour(data, "radius_squared", 0.5)
print(surface.num_cells, "triangles,", surface.num_points, "points")
print(td.statistics(data, "radius_squared")["mean"])
```

Save this as a `.py` file and run it after installing the packages above, or use
`uv run python` from a checkout installed as in [Getting Started](01-getting-started.md).
The scalar coefficients are sampled from `x*x + y*y + z*z`; the contour is
an approximation to a sphere. This example needs no VTK installation.

## Learn data and visualization

| Chapter | Start here when you want to… |
|---|---|
| [Function Space Fundamentals](data/00-function-spaces.md) | Understand H1, H2, L2, polynomial order and degrees of freedom |
| [Understand a Dataset](data/01-datasets.md) | Connect raw storage to topology, geometry and field meaning |
| [Build Meshes](data/02-meshes.md) | Construct grids, mixed cells, polyhedra, polygons or paths |
| [Fields and Spaces](data/03-fields-and-spaces.md) | Preserve a discontinuity, project a field or use implicit values |
| [Filter and Analyze](data/04-filters.md) | Choose between threshold, clip, contour and slice; carry fields and compute derivatives |
| [Write Mesh Algorithms](data/05-mesh-kernels.md) | Add cell and face kernels with correct ids, subsets and orientation |
| [Sample and Connect Simulations](data/06-pipelines.md) | Probe a solution, integrate paths or exchange datasets |

For a first reading, follow the fundamentals, dataset and mesh chapters,
then the field and filter chapters.
The [field-space comparison](data/03-fields-and-spaces.md#a-material-interface-exposes-the-difference)
and [filter comparison](data/04-filters.md#select-cells-cut-cells-or-extract-a-surface)
show why two plausible images can represent different numerical operations.

## Common conventions

Every algorithm takes the dataset first and returns something new: a
dataset, a field, or Python values. Nothing changes its input.

Fields passed to an algorithm belong to that dataset's topology. `dot`,
`cross` and `composite` require matching value types; an explicit
`tack.arrays.Cast(field.values, tack.f32)` converts values when needed. Integer
inputs to magnitude, logarithms and statistics compute in floating point.

## The model

A `DataSet` is four things:

| Part | What it is | Example |
|---|---|---|
| `topology` | the cells and how they connect | `UnstructuredTopology`, `StructuredTopology`, `PolyhedralTopology` |
| `geometry` | where the points are: a field of 3-vectors | positions per point, or the implicit coordinates of a grid |
| `fields` | named fields, each a *space* and its values | `Field(H1(data), values)` |
| `sets` | named selections of points, cells or faces | the boundary faces |

A *space* says which degrees of freedom define a field and how to evaluate
it. H1 order 1 shares corner values; H1 order 2 adds shared higher-order
nodes. `Constant` is a separate constant per cell; L2 can keep each cell's
own coefficients. `Values(data, on)` describes attributes without a basis.
Every field chooses its own space on the common topology, including geometry.
The [dataset chapter](data/01-datasets.md#each-field-has-its-own-space) demonstrates
12 corner positions, 45 quadratic temperature values and 16 discontinuous
pressure values on the same two cells.

A field's values are a `tack.Array` with a scalar or vector element dtype,
or an implicit array that computes values on demand. Uniform coordinates
need only dimensions, origin, spacing and direction metadata.

Geometry may also live in L2, giving each cell its own positions. Contour and
slice preserve the resulting gaps; clipping requires shared H1 geometry.
Changing an L2 geometry's values refreshes subsequent face-geometry results.

## Building a dataset

A dataset can be assembled from simulation arrays, a file reader, generated
coordinates, or another source. The construction follows the same pattern:

1. Create a topology describing cells and their corner incidence.
2. Create a geometry field: a space on that topology plus coordinate values.
3. Pass both to `td.DataSet`, then attach solution fields and named sets.

The following recipes build their topology and geometry explicitly. They
assume `import numpy as np`, `import tack`, `import tack.dataset as td`, and
an initialized backend. The complete [mesh-construction program](../examples/data_meshes.py)
and [unstructured tour](../examples/datasets.py) include that setup.

=== "Uniform image"

    Construct structured incidence and compute corner positions from origin,
    spacing and logical indices. The geometry is an H1 field whose values
    are implicit `UniformCoordinates`.

    ```python
    --8<-- "docs/examples/data_meshes.py:uniform"
    ```

=== "Rectilinear"

    Construct structured incidence and compute corner positions from three
    coordinate axes. The geometry uses `CartesianProduct` values in H1.

    ```python
    --8<-- "docs/examples/data_meshes.py:rectilinear"
    ```

=== "Curvilinear"

    Construct structured incidence and supply explicit positions at its
    corners. These positions bend the grid while retaining logical connectivity.

    ```python
    --8<-- "docs/examples/data_meshes.py:curvilinear"
    ```

=== "Unstructured"

    Supply the cell types, offsets and corner ids explicitly, along with an
    H1 coordinate field. This example has a hexahedron and a pyramid sharing
    a face. The cell type numbers follow VTK's convention; VTK is not required.

    ```python
    --8<-- "docs/examples/datasets.py:unstructured"
    ```

All four recipes use the same `DataSet(topology, geometry)` constructor.
The [mesh chapter](data/02-meshes.md#image-rectilinear-and-curvilinear-data)
explains their coordinate representations and also builds
[polyhedral](data/02-meshes.md#construct-polyhedra-from-face-rings),
[polygonal and polyline](data/02-meshes.md#polygons-and-polylines-are-different-objects)
datasets from their own connectivity and geometry.

The rest of this tour uses `mesh`, the hex-and-pyramid dataset from the
Unstructured tab. Attach height in H1 and a material constant per cell:

```python
--8<-- "docs/examples/datasets.py:fields"
```

### Factory methods are shortcuts

Once the pieces are familiar, `uniform_grid` and `rectilinear_grid` save the
repeated topology and geometry setup. These calls construct the same uniform
and rectilinear datasets as the explicit recipes above:

```python
--8<-- "docs/examples/data_meshes.py:grids"
```

`DataSet(topology, positions)` is another convenience: when given coordinate
values rather than a `Field`, it wraps them in H1 order-1 geometry.
`wavelet` and `tangle` are sample-data factories that also initialize known
scalar fields, useful for trying filters. They are examples of data sources;
a dataset's topology and geometry do not depend on where its values originated.

`vtk_to_dataset` and `tack.interop.mfem` import datasets from VTK and MFEM.
Neither library is required for the constructions above. In a checkout,
`uv sync --extra cpu --extra interop` installs the optional readers where
supported. See [dataset interoperability](data/06-pipelines.md#exchange-whole-datasets-with-vtk).

## Algorithms

Every algorithm is in `tack.dataset.algorithms`, re-exported from `tack.dataset`,
and they share one set of conventions:

- the dataset comes first;
- a field argument is a `Field` or the name of one of the dataset's fields;
- an algorithm that makes a dataset takes `fields=` (all by default, a name,
  a list, or `[]`) and carries those fields onto its output;
- one that computes a field returns a `Field`, or a named tuple of them;
- summaries (`statistics`, `entropy`) return Python values.

```python
--8<-- "docs/examples/datasets.py:pipeline"
```

Fields come along by how the output's entities come from the input's. Point
data is interpolated onto a contour's points, and cell data goes to the
pieces each cell was cut into. A kept cell keeps its values.

## Moving fields between spaces

`project(data, field, space)` gives a field in another space of the same
mesh. `to_points`, `to_cells` and `discontinuous` are shortcuts for it.

```python
--8<-- "docs/examples/datasets.py:spaces"
```

Projecting into a space that can hold the field exactly gives it back
exactly: a linear point field into `L2`, or into order-2 `H1`. Otherwise
`project` averages the way VTK does: a point takes the mean of the cells
around it, and a cell the mean of its corners.

## Sets

A `Set` selects points, cells or faces. Builders such as `boundary_faces`
return one; name it in `data.sets` to keep it. Filters carry sets the way
they carry fields:

```python
--8<-- "docs/examples/datasets.py:sets"
```

The name of a cell or face set is also a place to run a kernel, as the next
section shows.

## Writing your own algorithm

`for_each(kernel, data, domain, *args)` runs a kernel over `"cells"`,
`"faces"`, `"edges"` or a named set. A mesh may mix shapes, so it runs the
kernel once per shape. Each launch gets that shape's *view* of the cells,
and every `Field` argument becomes its view for the same cells. The kernel
is compiled for each shape, so a hexahedron's loop knows it has eight
corners.

```python
--8<-- "docs/examples/datasets.py:kernel"
```

A view's methods follow one naming rule per kind of answer, so a name says
what comes back:

| Ends in | Gives | Examples |
|---|---|---|
| `_id` | a global id | `entity_id(c)`, `point_id(c, j)`, `face_id(c, k)`, `side_point_id(c, k, j)` |
| `_position` | a position | `position(c, pc)`, `corner_position(c, j)`, `side_position(c, k, j)` |
| `_corner` | a shape's local corner number | `edge_corner(e, k)`, `face_corner(f, k)` |
| `num_` (prefix) | a count | `num_points`, `num_faces(c)`, `side_num_points(c, k)` |

A field view gives `value(c, pc)` at parametric coordinates,
`parametric_gradient(c, pc)` and `corner_value(c, j)`.

Face-based kernels walk each cell's faces outward, and the same code serves
every kind of cell. They need the faces, which are derived on first use, so
ask for them before the launch:

```python
--8<-- "docs/examples/datasets.py:faces"
```

## Polyhedra, polygons and lines

A polyhedral cell has no reference element: no shape functions and no
parametric coordinates. Algorithms that need one (gradients, probing,
advection) refuse a polyhedral mesh and say so. The ones that work from
points and faces run on both kinds: contour (by López's face walk), slice,
threshold, external faces, cell and face geometry, divergence, and moving
data between points and cells. A 2D mesh contours into lines, and
streamlines are polylines, one per seed:

```python
--8<-- "docs/examples/datasets.py:lines"
```

The [reference](../reference/dataset-api.md#algorithms-and-filters) has a
table of which algorithm runs on which topology.

## 64-bit ids

A topology keeps its ids in `i32` unless it is too large or asked for `i64`
(`id_dtype=tack.i64`). Kernels specialize on the id type, so a small mesh
pays nothing for large ones. VTK's 64-bit ids come in as the narrowest type
that holds them.

## Running the examples

The whole example is [`docs/examples/datasets.py`](../examples/datasets.py);
`--check` runs its assertions.

The new chapter examples can be checked together with the rest of the guide:

```bash
uv run python docs/examples/validate.py --arch cpu
```

They cover mesh counts and measures, continuous/discontinuous fields, analytic
gradients, flux balance, exact clipping volumes and flow integration. Their
optional comparison figures use Matplotlib, independently of VTK.

## VTK interop

`tack.interop.vtk` converts whole datasets:
- `vtk_to_dataset` reads a `vtkUnstructuredGrid`, `vtkRectilinearGrid` or
  `vtkImageData`, including polyhedra and polylines;
- `dataset_to_vtk` writes the matching VTK type back.

```python
from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

data = vtk_to_dataset(grid)               # point data -> H1, cell data -> Constant
grid = dataset_to_vtk(td.contour(data, "pressure", 0.5))
```

Arrays exchange through DLPack, sharing storage where the device allows:

```python
from tack.interop.vtk import array_to_vtk, vtk_to_array

values = vtk_to_array(vtk_array)            # a tack.Array sharing VTK's storage
vtk_array = array_to_vtk(values)
```

VTK's tuples and components become an array shape: a `(1000, 3)` scalar array
represents 1000 tuples of three components, and an array of `vec3` elements
exports as 1000 tuples of three as well. `vtk_to_array(array,
flatten=True)` and `array_to_vtk(values, n_components=3)` use the flat
interleaved layout instead. This requires a VTK build exposing
`vtkmodules.util.dlpack_support`. Not every VTK installation provides device
arrays. See [Sharing arrays with other libraries](16-interoperability.md) for
lifetime and read-only rules.

### Level Zero

Level Zero device pointers need a shared allocation context. With a compatible
VTK/Viskores Kokkos-SYCL build, initialize before allocating arrays:

```python
from tack.interop.vtk import init_level_zero

init_level_zero()     # instead of tack.set_default_device("level_zero")
```

The VTK DLPack support must provide `level_zero_handles()`. Fields created in a
separate Tack context cannot be exchanged through this path. For another library,
`tack.device("level_zero", external_context={"driver": ..., "device": ...,
"context": ...})` makes a device in its context; Tack never destroys an adopted context.
