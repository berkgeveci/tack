# Polyhedra, polygons and lines

A hexahedron is a map of a reference cube, and its fields are built on that
cube. [Fields and Spaces](03-fields-and-spaces.md) introduces that model;
[the optional theory chapter](00-function-spaces.md) develops its bases. A polyhedron has no reference
element. It is a list of faces, each a ring of points, with any number of
faces and points. A polygon is a ring of points, and a polyline a chain.
This page says what a field on these cells is, what a cell must be for
Tack to work in it, and what the algorithms compute there.

## A field is linear on the cell's pieces

Tack cuts each cell into simple pieces and makes a field linear on each:

- **A polyhedron** is cut into sub-tetrahedra. Each face is fanned into
  triangles from its *face point*, the mean of its points. Those triangles
  are fanned from the *cell point*, the mean of the cell's distinct points.
  Each sub-tetrahedron is (cell point, face point, *a*, *b*) for an edge
  *ab* of a face.
- **A polygon** is fanned into triangles from its cell point.
- **A polyline** is its segments.

The values are the points' values. The face point and the cell point take
the means of the values, as they take the means of the positions. The two
consequences that matter:

- **A linear field is exact.** The extra points' values are what a linear
  field has there, so the pieces reproduce it. Gradients, averages, probes
  and integrals of a linear field are exact on any cell Tack accepts.
- **The field is continuous.** A face is stored once, and both its cells fan
  it from the same face point, so the two cells agree on it, even where the
  face is not planar.

In the terms of [Function Spaces and Higher-order Geometry](00-function-spaces.md), this is
order-1 `H1` on a finer mesh whose extra nodes are not free: they are means
of the cell's points. That is why a polyhedron holds what it holds:

| Space | On polyhedra and polygons | Why |
|---|---|---|
| `H1` (order 1), values on points | yes | the fan's values at the points |
| `Constant`, values on cells | yes | one value a cell needs no basis |
| `L2`, `H1` above order 1, `SideTraces` | refused | each needs nodes inside the cell, on a reference element |

A refused space raises `NotImplementedError` naming the reference element.
Projecting between point and cell values works both ways
([Create and convert fields](create-fields.md#projection-is-an-algorithmic-choice)).

## What a cell must be

- **Its faces are wound outward.** Each face's points run counterclockwise
  as seen from outside the cell, as VTK specifies. A cell wound inward has a
  negative volume, and its normals and fluxes point the wrong way.
  `td.PolyhedralTopology.from_cell_faces(..., orient=True)` makes the winding
  consistent and turns each connected piece outward. Reading VTK data,
  `vtk_to_dataset` warns about inside-out polyhedra, and `orient=True`
  repairs them.
- **It is star-shaped from its cell point.** Every face must be visible from
  the mean of the cell's points, or some sub-tetrahedra turn inside out and
  overlap the others. Concave cells are fine as long as this holds. A
  C-shaped prism, whose points' mean lies in its notch, is not. The cell
  locator refuses such cells, and so do `probe` and advection, which use it.
- **Warped faces are their fans.** A face whose points do not lie in a plane
  is the fan of triangles from its face point. Volumes, normals and fluxes
  are those of that surface.

[Build Meshes](02-meshes.md#construct-polyhedra-from-face-rings) shows how to
construct polyhedra from face rings, and `td.as_polyhedra(data)` turns a 3D
mesh of shapes into the same face-based form.

## What the algorithms compute

| Algorithm | On a polyhedron | On a polygon |
|---|---|---|
| `cell_geometry` | volume by the divergence theorem over the fanned faces; centroid | area along its own normal; centroid |
| `gradients` | Green–Gauss over the fanned faces: the volume-weighted mean of the sub-tetrahedra's gradients | Green–Gauss over its edges, in its plane |
| `probe`, advection | locate the sub-tetrahedron holding the point, evaluate the field there | locate the fanned triangle |
| `project` into `Constant` | the field's integral over the sub-tetrahedra, over the volume | the same over the triangles |
| `values_at_centers`, `cell_centers` | the mean of the points' values, at the cell point | the same |
| `integrate` | exact on the sub-tetrahedra | exact on the triangles |
| `cell_quality` | measured on the fan's tetrahedra | measured on the fan's triangles |

A cell's average and its value at the cell point are different numbers. The
average is the field's integral over the cell divided by the volume: for a
linear field, its value at the centroid. The cell point is the mean of the
cell's points, which is not the centroid. `project(..., average="nodes")`
gives the points' mean, as VTK's `vtkPointDataToCellData` does.

Contour, slice and threshold work on polyhedra and polygons; clip works on
polyhedra and makes polyhedra, and polygons cannot be clipped yet.
[Filter and Analyze](04-filters.md#check-algorithm-support-before-building-a-pipeline)
has the full table.

## A concave cell, checked

A unit cube with its top pushed in: four triangles meet at a point at height
0.6. The cell is concave, but star-shaped from its points' mean. For the
linear field `1 + 2x - y + 3z`, the volume, gradient and average are exact,
and a point in the dent is outside the cell:

<!-- tack-check: polyhedra -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# A unit cube whose top is pushed in: four triangles down to a point at z = 0.6.
xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0.5, 0.5, 0.6]])
faces = [[0, 3, 2, 1], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
         [4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8]]
topology = td.PolyhedralTopology.from_cell_faces([faces], num_points=9)
data = td.DataSet(topology, xyz)
data.fields["f"] = td.Field(td.H1(data), tack.array(1 + 2 * xyz[:, 0] - xyz[:, 1] + 3 * xyz[:, 2]))

volume = td.cell_geometry(data).measure.values.numpy()[0]
gradient = td.gradients(data, "f").values.numpy()[0]
average = td.project(data, "f", td.Constant(data)).values.numpy()[0]
at_center = td.values_at_centers(data, "f").values.numpy()[0]
probed = td.probe(data, np.array([[0.5, 0.5, 0.55], [0.5, 0.5, 0.8]]))
print(f"volume {volume:.4f}, gradient {gradient}, average {average:.4f}, "
      f"at the points' mean {at_center:.4f}")

assert np.isclose(volume, 1 - 0.4 / 3, rtol=1e-6)                # the dent: a pyramid
assert np.allclose(gradient, [2, -1, 3], atol=1e-5)               # exact for a linear field
centroid_z = (0.5 - 0.9 * 0.4 / 3) / (1 - 0.4 / 3)
assert np.isclose(average, 1.5 + 3 * centroid_z, rtol=1e-6)       # f at the centroid
assert np.isclose(at_center, 1.5 + 3 * 4.6 / 9, rtol=1e-6)        # f at the points' mean
assert probed.fields["valid"].values.numpy().tolist() == [1, 0]   # the dent is outside
assert np.isclose(probed.fields["f"].values.numpy()[0], 1.5 + 3 * 0.55, rtol=1e-6)
```

## How this differs from VTK

VTK's `vtkPolyhedron` interpolates with *mean value coordinates*. They are
smooth inside the cell, exact for linear fields and defined on non-convex
cells. Tack's fan is linear on pieces instead, so inside a cell the two give
different values for anything but a linear field. At the points, and for
linear fields everywhere, they agree.

In VTK 9.7.1, several polyhedron operations are wrong, so VTK is not a
reference for these cells:

- `vtkCellSizeFilter` measures a concave polyhedron as its convex hull. The
  cell above measures 1.0, not 0.8667.
- `vtkProbeFilter` takes points outside a polyhedron as inside. The point in
  the dent above comes back valid, with 3.58.
- `vtkPolyhedron::Derivatives` is wrong in each direction whose parametric
  coordinate is 1, so `vtkGradientFilter`'s point gradients are wrong at the
  corners of a box.
- VTK accepts inside-out polyhedra without a word, and `vtkCellValidator`
  flags correctly oriented concave cells.

Tack's tests hold polyhedra to closed forms, to the same cells as shapes,
and to brute force instead.

## Cost

The same moved hexahedra as polyhedra, against themselves as hexahedra
(64³ cells, warm medians on an M1 Max, `benchmarks/polyhedra_compare.py`):

| Algorithm | CPU | Metal |
|---|---:|---:|
| `gradients` (cells) | 4.9× | 2.6× |
| `cell_centers`, `values_at_centers` | 0.9–1.1× | 1.3–1.4× |
| `probe` | 1.3× | 1.3× |
| `advect` | 1.4× | 1.7× |
| `streamlines` | 1.3× | 1.6× |

A hexahedron reads its eight corners by id. A polyhedron reaches each point
through its faces, and its gradient walks every face. Keep shaped cells as
shapes where you can; a VTK grid mixing shapes and polyhedra can be read as
two pieces, `vtk_to_dataset(grid, polyhedral="split")`, each on its fastest
path.

The reasoning behind these choices, and the measurements, are in the
[polyhedra design record](design/polyhedra.md).

??? info "Related detailed sections"
    These topics are developed in their own chapters.

    <span id="polyhedra-and-polygons" data-guide-forward="#polyhedra-polygons-and-lines"></span>
    [Polyhedra and polygons](18-polyhedra.md#polyhedra-polygons-and-lines).
