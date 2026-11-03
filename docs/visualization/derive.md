# Derive quantities

Many questions about a simulation are about quantities it did not store: the
gradient of a pressure, the vorticity of a flow, the volume and shape of each
cell, the total mass in a region, a cell field seen at the points. This page
shows the Tack filters that compute them, for readers who know VTK's
`vtkGradientFilter`, `vtkCellSizeFilter`, `vtkMeshQuality`,
`vtkIntegrateAttributes` and `vtkCellDataToPointData`. After it you can derive
these quantities, read their layout and units, and choose between the ways of
moving a field between point and cell data.

The filters here return fields or numbers, not a new dataset. Put a field in
`data.fields` to keep it with the dataset, so that later filters carry it.

## Compute gradients and flow quantities

`gradients(data, field)` is VTK's `vtkGradientFilter`. By default it gives the
gradient at each cell's centre. With `at="points"` it gives each cell's
gradient at each point, averaged over the cells around the point, as VTK does.
`flow_quantities` derives the divergence, the vorticity and the Q-criterion
from a velocity's gradient.

<!-- tack-check: gradients -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((11, 11, 11), origin=(-1, -1, -1), spacing=(0.2, 0.2, 0.2), unit="m")
x = data.positions()
data.fields["T"] = td.Field(td.H1(data), tack.array(300 + 10 * x[:, 0]), unit="K")
rotation = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x), np.float32)], axis=1)
data.fields["velocity"] = td.Field(td.H1(data), tack.array(rotation, dtype=tack.vec3(tack.f32)),
                                   unit="m/s")

grad_T = td.gradients(data, "T")                         # one 3-vector per cell
at_points = td.gradients(data, "T", at="points")         # one per point
flow = td.flow_quantities(data, td.gradients(data, "velocity"))

print(grad_T.values.numpy()[0].round(3), grad_T.unit)              # [10. 0. 0.] K/m
print(flow.vorticity.values.numpy()[0].round(3), flow.vorticity.unit)  # [0. 0. 2.] 1/s
assert np.allclose(at_points.values.numpy(), [10, 0, 0], atol=1e-4)
assert np.allclose(flow.divergence.values.numpy(), 0, atol=1e-5)
assert np.allclose(flow.q_criterion.values.numpy(), 1, atol=1e-5)
```

The field may be point data (an [`H1` field](../reference/glossary.md#h1) of
order 1), cell data (a `Constant` field), or a field of any order, continuous
(`H1`) or discontinuous (`L2`). The gradient is computed through the field's
own interpolation in each cell, and the geometry's Jacobian turns it into a
gradient in space.

- **Layout.** A scalar's gradient is a 3-vector. A vector of `k` components
  has `3k` values, row by row: `d(u_i)/d(x_j)` is at `3 * i + j`. That is
  VTK's and Viskores' layout of the gradient tensor, and what
  `flow_quantities` reads.
- **Units.** The gradient's unit is the field's over the spatial unit:
  `K/m` above. A field or a dataset without a unit gives a gradient without
  one.
- **Higher order.** `at` may also be an `L2` space of the dataset:
  `td.gradients(data, "u", at=td.L2(data, order=p))`. That gives the gradient
  at every node of every cell, exactly, without the averaging that
  `at="points"` does. An order-`p` field keeps its detail this way.
- **2D and degenerate cells.** A 2D cell's gradient lies in its plane, as in
  VTK. Lines and vertices get zero, and a cell of no volume gets NaN.
- **Polyhedra.** A polyhedron has no reference element, so its gradient of
  point data is computed by Green-Gauss over its faces. That is exact for a
  linear field.

`flow_quantities` returns a named tuple of three fields, `divergence`,
`vorticity` and `q_criterion`, in the gradient's space: per cell from a cell
gradient, per point from a point gradient. See the reference for
[`gradients`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.gradients.gradients)
and [`flow_quantities`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.flow_quantities.flow_quantities).

### Other fields from fields

Smaller filters compute a field from fields pointwise. Each returns a field in
its input's space, with a derived unit.

| Tack | Gives | Viskores |
|---|---|---|
| `magnitude(data, field)` | each vector's length | `VectorMagnitude` |
| `dot(data, a, b)` | the dot product of two vector fields | `DotProduct` |
| `cross(data, a, b)` | the cross product of two 3-vector fields | `CrossProduct` |
| `composite(data, *fields)` | scalar fields as the components of a vector | `CompositeVectors` |
| `log_values(data, field)` | the logarithm, in any base | `LogValues` |
| `convert(data, unit, field)` | a field, or the geometry, in another unit | |

`dot`, `cross` and `composite` need fields of one value type. VTK's
`vtkArrayCalculator` has no counterpart: a formula of your own is a short
kernel, as [write a mesh algorithm](write-a-mesh-algorithm.md) shows.

## Measure cells

`cell_geometry(data)` gives each cell's measure (volume, area or length) and
its centroid, as VTK's `vtkCellSizeFilter` does, as a named tuple of two cell
fields. A 3D cell's volume comes from the divergence theorem over its faces,
so it works on every shape, polyhedra included. A curved cell is measured by
quadrature through its own map, so its volume is the curved cell's.

`cell_centers(data)` is each cell's parametric centre, as VTK's
`vtkCellCenters`. It is not the centroid on a cell that is not affine, such as
a skewed hexahedron. Use `cell_geometry(data).centroid` for the centroid.

<!-- tack-check: cell-geometry -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((11, 11, 11), origin=(-1, -1, -1), spacing=(0.2, 0.2, 0.2), unit="m")
cells = td.cell_geometry(data)
volume = cells.measure.values.numpy()
print(f"{volume.sum():.3f} {cells.measure.unit}")         # 8.000 m**3
assert np.isclose(volume.sum(), 8.0)
assert np.allclose(cells.centroid.values.numpy(), td.cell_centers(data).values.numpy())

faces = td.face_geometry(data)                             # each face's normal and area
assert np.isclose(faces.area.values.numpy().max(), 0.04)
```

On faces and edges, `face_geometry` gives each face's unit normal and area,
`face_centers` each face's centroid, and `edge_lengths` each edge's length.
Their fields live on the faces and edges that `data.topology.faces()` and
`edges()` number. See the reference for
[`cell_geometry`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.cell_geometry.cell_geometry)
and [`cell_centers`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.cell_centers.cell_centers).

## Measure cell quality

`cell_quality(data, measures)` says how well shaped each cell is: a dict of
cell fields, one per measure. These are Verdict's definitions, as VTK's
`vtkMeshQuality` uses them, and each measure is 1 on the ideal cell of its
shape: an equilateral triangle, a square, a cube, a regular tetrahedron.

| Measure | What it says |
|---|---|
| `scaled_jacobian` | the worst corner's edges' determinant, the edges scaled to unit length: 1 ideal, 0 flat, negative inverted |
| `jacobian_ratio` | the least corner determinant over the greatest: 1 for every simplex, below 1 as a cell tapers |
| `condition` | the worst corner's condition number: 1 ideal, growing as the cell distorts |
| `edge_ratio` | the longest edge over the shortest |
| `min_edge`, `max_edge` | the shortest and the longest edge, in the spatial unit |

<!-- tack-check: cell-quality -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

mesh = td.tetrahedralize(td.uniform_grid((4, 4, 4)))
quality = td.cell_quality(mesh, ["scaled_jacobian", "condition", "edge_ratio"])
sj = quality["scaled_jacobian"].values.numpy()
print(f"scaled Jacobian {sj.min():.3f} to {sj.max():.3f}")   # 0.707 to 1.000

mesh.fields["sj"] = quality["scaled_jacobian"]
worst = td.threshold(mesh, "sj", -1.0, 0.75)                 # the cells to look at
assert np.isclose(sj.max(), 1.0) and np.isclose(sj.min(), 1 / np.sqrt(2))
assert worst.num_cells == np.count_nonzero(sj < 0.75)
```

The measures extend to every shape Tack has. Wedges and pyramids get every
measure, and an order-2 cell is measured along its curved edges. Polyhedra and
polygons are measured by the tetrahedra and triangles they are fanned into.
Their Jacobian measures are then on the fan's scale, not the shape's: a cube
stored as a polyhedron scores 0.58, not 1. Read them as a validity check (a
positive scaled Jacobian means no fan tetrahedron is flat or inverted), and
compare polyhedra with each other. The reference has the
[formulas](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.cell_quality.cell_quality)
and where VTK 9.7.1 differs.

## Integrate fields

`integrate(data)` gives the mesh's measure (its total volume, area or length)
and the integral of each field over it, as VTK's `vtkIntegrateAttributes`
does. It returns an `Integrals` of `measure`, `measure_unit`, `values` and
`units`. A vector field's integral is a tuple.

<!-- tack-check: integrate -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((11, 11, 11), origin=(-1, -1, -1), spacing=(0.2, 0.2, 0.2), unit="m")
x = data.positions()
data.fields["T"] = td.Field(td.H1(data), tack.array(300 + 10 * x[:, 0]), unit="K")

totals = td.integrate(data, ["T"])
print(f"{totals.measure:.1f} {totals.measure_unit}, {totals.values['T']:.1f} {totals.units['T']}")
# 8.0 m**3, 2400.0 K*m**3
assert np.isclose(totals.measure, 8.0) and np.isclose(totals.values["T"], 2400.0, rtol=1e-6)
mean_T = totals.values["T"] / totals.measure            # the volume-weighted mean, 300 K
assert np.isclose(mean_T, 300.0, rtol=1e-6)
```

Each cell is integrated by Gauss quadrature through its own map, with enough
points for the field's order and the geometry's. A field of any order on a
curved mesh therefore integrates exactly, to rounding. Only the cells of the
mesh's dimension count, as in VTK: a volume's integrals leave out its surface
cells. Ghost copies are left out, and over a `PartitionedDataSet` the pieces
are summed over every rank.

An integral is not a sum of samples. `statistics` gives the mean of the stored
values, which on a graded mesh is not the volume-weighted mean that
`integrate` gives ([reduce](reduce.md) explains the difference). See the
[reference](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.integrate.integrate).

## Move fields between point and cell data

`project(data, field, space)` makes a new field of `field` in another space of
the same dataset. It covers VTK's `vtkCellDataToPointData` and
`vtkPointDataToCellData`, and more. The common targets:

| Target space | Each value | VTK |
|---|---|---|
| `td.H1(data)` (point data) | the mean of the source at the point, over the cells around it | `vtkCellDataToPointData` |
| `td.Constant(data)` (cell data) | the source's average over the cell: its integral over the cell's measure | |
| `td.Constant(data)` with `average="nodes"` | the mean of the source's values in the cell | `vtkPointDataToCellData` |
| `td.L2(data, order=1)` | the source's values at the cell's corners | |

<!-- tack-check: project -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((11, 11, 11), origin=(-1, -1, -1), spacing=(0.2, 0.2, 0.2))
x = data.positions()
data.fields["T"] = td.Field(td.H1(data), tack.array(300 + 10 * x[:, 0]))

per_cell = td.project(data, "T", td.Constant(data))          # each cell's average
per_point = td.project(data, per_cell, td.H1(data))          # back to the points
at_centres = td.values_at_centers(data, "T")                 # T at each cell's centre

print(per_cell.values.numpy()[:3])                           # [291. 293. 295.]
assert np.allclose(per_cell.values.numpy(), at_centres.values.numpy())
interior = (np.abs(x[:, 0]) < 0.9)                           # away from the ends
assert np.allclose(per_point.values.numpy()[interior], data.fields["T"].values.numpy()[interior])
```

The two averages into cell data differ in what they mean. The default,
`average="integral"`, is the field's true average over the cell, exact for
any order and any curved cell. `average="nodes"` is VTK's mean of the values
the cell stores. The two agree for a linear field on a tetrahedron or a box,
but not on a pyramid, a distorted cell or a field of higher order.

Into point data, a continuous field comes across exactly, and where cells
disagree (cell data, or a discontinuous field) each point takes their mean.
Values on faces or edges have no interpolation, so they cannot be projected.

`values_at_centers(data, field)` evaluates a field at each cell's parametric
centre through its interpolation. For a linear field that equals the cell
average, as in the example. For higher-order fields it is a point value, not
an average. See the reference for
[`project`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.project.project)
and [`values_at_centers`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.values_at_centers.values_at_centers).
The [fields page](../data/fields.md) explains spaces, and when each projection
is the right one.

## Details

**Precision.** Filters compute in the dataset's precision, which
`vtk_to_dataset` makes single by default. Jacobians and gradients are formed
from differences of corners and values, never from absolute positions, so
thin cells keep their short edges' accuracy in single precision. Where
double precision matters, read the data with `dtype=tack.f64` on a device
that has it.

**Singular corners.** At the points, a cell whose Jacobian is singular at a
corner (where two faces meet flat) has no gradient there. It is left out of
that point's average, which is NaN only if no cell around the point is left.
A pyramid's apex gets the gradient from just below it, as VTK's `vtkPyramid`
does.

**Cells without a reference element.** On polyhedra, polygons and polylines,
`project` moves only point data and cell data, one into the other.
`values_at_centers` and `cell_centers` use the *cell point*, the mean of the
cell's distinct points, which is where the cell's interpolation is fanned
from. The [polyhedra page](../data/polyhedra.md) explains why.

## Where to go next

- [Sample and trace fields](sample-and-trace.md): a derived field at probe
  points, or the paths a velocity carries particles along.
- [Reduce](reduce.md): statistics, histograms and connected regions of the
  fields you derived.
- [Finite-volume data and fluxes](finite-volume.md), for values on faces:
  jumps, upwind fluxes and divergence.
