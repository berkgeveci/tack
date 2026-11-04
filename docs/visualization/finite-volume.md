# Finite-volume data and fluxes

A finite-volume code stores cell averages, cell-centred vectors, and normal
velocities or fluxes through faces. To check its balances, you keep each
quantity on the cells or faces it belongs to and compute with the faces'
geometry. Tack's face algorithms read the two sides of a face, take jumps,
upwind fluxes, net fluxes and divergences, and reconstruct cell vectors from
face-normal components. With them you can check a simulation's balances in
Tack, and write face kernels of your own in the same way.

The algorithms on this page work on every topology with faces: structured
grids, unstructured meshes and polyhedra. VTK has no counterparts for them.
In 2D a cell's faces are its edges, so a face's "area" is an edge's length and
a cell's measure is its area.

<!-- tack-check: fv-first -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# Two hexahedra in a unit cube, with a cell average of density in each.
data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0), unit="m")
data.fields["density"] = td.Field(td.Constant(data), tack.array([1.0, 2.0], tack.f32),
                                  unit="kg/m**3")

# Density carried by a uniform wind of 0.3 m/s along x.
mass_flux = td.upwind_flux(data, "density", ((0.3, 0.0, 0.0), "m/s"))  # kg/s through each face
outflow = td.net_flux(data, mass_flux)                                  # kg/s out of each cell
rate = td.divergence(data, mass_flux)                                   # outflow per volume

print(outflow.values.numpy(), outflow.unit, "|", rate.values.numpy(), rate.unit)
assert np.allclose(outflow.values.numpy(), [0.0, 0.3])
assert np.allclose(rate.values.numpy(), [0.0, 0.6])
```

```text
[0.  0.3] kg/s | [0.  0.6] kg/(m**3*s)
```

The first cell's inflow and outflow balance. The second cell sends out
0.6 kg/s through its boundary face and takes in 0.3 kg/s from the first.

## The algorithms

| Task | Algorithm | Gives |
|---|---|---|
| Face normals and areas | [`face_geometry`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.face_geometry.face_geometry) | fields on faces |
| Face centroids | [`face_centers`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.face_centers.face_centers) | a field on faces |
| Cell volumes and centroids | [`cell_geometry`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.cell_geometry.cell_geometry) | fields on cells |
| The boundary faces | [`boundary_faces`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.boundary_faces.boundary_faces) | a face set |
| Both sides of each face | [`project`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.project.project) into `SideTraces` | a field on faces |
| A field's difference across each face | [`jump`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.jump.jump) | a field on faces |
| A field's upwind flux through each face | [`upwind_flux`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.upwind_flux.upwind_flux) | an oriented field on faces |
| Each cell's outward sum of face fluxes | [`net_flux`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.net_flux.net_flux) | a field on cells |
| Each cell's finite-volume divergence | [`divergence`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.divergence.divergence) | a field on cells |
| Cell vectors from face-normal components | [`perot`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.perot.perot) | a field on cells |

## Keep quantities where the simulation put them

A finite-volume code stores control-volume averages and forms balances from
face fluxes. Tack keeps each quantity in the [space](../reference/glossary.md)
that says where it lives, without converting it.
[Fields](../data/fields.md) explains spaces. The ones finite-volume data uses
are:

| Simulation quantity | Space | Meaning |
|---|---|---|
| A cell average | `Constant(data)` | one constant in each cell (cell data) |
| A cell-centred value or vector | `Values(data, "cells")` | one value per cell, with no basis |
| A face pressure, area or label | `Values(data, "faces")` | one value per face |
| A normal component or an integrated flux | `Values(data, "faces", oriented=True)` | one value per face, positive along its normal |

`Constant` says a value holds across its cell. `Values` records where a value
belongs and nothing more. Neither turns an average into the point value of an
unknown original field.

A value on faces is oriented when it changes sign with the face's normal. A
filter that turns a carried face around, as `threshold` does when it keeps
only a face's second cell, negates oriented values and leaves plain ones
alone. A pressure, an area or a label is plain. The flag cannot check that an
imported value has the right sign.

## Faces and their two sides

Every face has side 0 and, inside the mesh, side 1. Its normal points out of
side 0. A face's flux is therefore positive out of its side-0 cell and
negative out of its side-1 cell. A boundary face has side 0 alone. This is
Tack's form of the owner and neighbour convention of codes such as OpenFOAM,
though Tack's face order need not be the simulation's.

The examples from here on use two hexahedra of 0.5 by 1 by 1 metres:

```python
--8<-- "docs/examples/vis2_finite_volume.py:mesh"
```

The geometry algorithms give what a balance needs:

```python
--8<-- "docs/examples/vis2_finite_volume.py:geometry"
```

`cell_geometry` gives volumes and volume centroids. Use these centroids in
volume calculations. `cell_centers` is a different point: the centre of the
reference cell, or a polyhedron's mean of points, which on a distorted cell
need not be its centroid.

`data.topology.faces()` numbers the faces Tack derives. Before attaching a
simulation's face arrays, map its face ids and orientations to these.
An array of the right length does not establish the correspondence. A
polyhedral topology built from explicit faces can keep the simulation's face
order, provided its sides agree.
[Build meshes and fields](../data/build-meshes-and-fields.md) shows how.

## Compute values on faces

A face kernel computes one value per face. `for_each(kernel, data, "faces",
...)` runs it over a view of the faces, and `faces.entity_id(f)` is the
face's index into arrays over all faces. Each face has one writer, so no
atomics are needed. This kernel computes a uniform velocity's normal
component, its volumetric flux, and the flux of a second flow, `u = (x, y,
z)`, whose divergence is 3:

```python
--8<-- "docs/examples/vis2_finite_volume.py:face-kernel"
```

```python
--8<-- "docs/examples/vis2_finite_volume.py:faces"
```

[Write a mesh algorithm](write-a-mesh-algorithm.md) explains views and
`for_each`.

### Normal velocity and flux are different inputs

For a unit normal `n_f`, an area `A_f`, a velocity `u_f` and a density
`rho_f`:

```text
un_f = dot(u_f, n_f)    normal velocity        m/s
Q_f  = A_f * un_f       volumetric flux        m³/s
M_f  = rho_f * Q_f      mass flux              kg/s
```

`perot` takes the normal velocity `un_f`. `net_flux` and `divergence` take an
integrated flux such as `Q_f` or `M_f`. None of them infers which an array
holds, or multiplies by the area for you.

## Read both sides of a face: traces and jumps

A field with a basis, such as an `H1` field (point data) or a discontinuous
`L2` field, has a value on each side of every face. Projecting it into
`SideTraces(data)` evaluates each cell at its faces' corners. The two sides
stay separate and line up point by point in the face's own order, as MFEM's
double-valued face vectors do. `jump(data, field)` gives the difference at
each face's centre, side 1 minus side 0, and zero on the boundary:

```python
--8<-- "docs/examples/vis2_finite_volume.py:traces"
```

The density has one nonzero jump, of magnitude 2, at the face between the
cells. The continuous `T` has none. The sign of a jump depends on which cell
is side 0, and its magnitude does not.

`SideTraces` holds eight values per face, two sides of up to four corners:
value `(face * 2 + side) * 4 + corner`. Triangles leave the fourth slot
unused, and boundary faces leave side 1 zero. Traces need 3D cells with a
reference element. `jump` evaluates each side's basis at the face's centre,
so a DG field's jumps are its own, of any order. Cell data is read from each
side's cell directly, on any topology, polyhedral included.

## Check a cell and boundary balance

With `s_cf` +1 when cell `c` is side 0 of face `f` and -1 when it is side 1:

```text
net_c = sum_f s_cf * Q_f
div_c = net_c / V_c
```

`net_flux` is the outward sum. `divergence` divides it by each cell's measure.
An interior face adds equal and opposite amounts to its two cells, so the
cells' outflows sum to the boundary's outflow:

```python
--8<-- "docs/examples/vis2_finite_volume.py:balance"
```

For `u = (x, y, z)` each cell of 0.5 m³ has a net outflow of 1.5 m³/s, a
divergence of 3/s. The cube's boundary outflow is 3 m³/s. The uniform
velocity's divergence is zero in both cells. Answers like these test the
orientation and the area and volume factors, not only a plausible picture.

`divergence` of face fluxes is not `flow_quantities`' divergence, which is
the trace of a field's gradient. Use the face-flux balance when the
simulation supplies fluxes.

## Upwind fluxes

`upwind_flux(data, field, velocity)` is the flux of a field carried by a
constant velocity, out of side 0 of each face: `(v · n) * area * u`. At an
interior face `u` is the field on the upwind side, the side the flow leaves,
evaluated at the face's centre as a DG advection scheme does. The result is
an oriented, integrated face flux, ready for `net_flux` and `divergence`, as
the first example shows. `velocity` is three numbers, or a quantity such as
`((0.3, 0.0, 0.0), "m/s")`, and the flux's unit follows.

A boundary face uses its one cell's value whichever way the flow goes, even
for inflow. Prescribe your own boundary flux when that rule does not
represent the simulation. `upwind_flux` is a helper for checks and
analysis, not a finite-volume solver.

## Reconstruct a cell vector with Perot's method

C-grid models such as MPAS and ICON store each face's normal velocity. To show
it at cells, `perot(data, flux)` reconstructs one vector per cell by Perot's
method:

```text
u_c = sum_f s_cf * A_f * un_f * (x_f - x_c) / V_c
```

Here `x_f` is the face's area centroid and `x_c` the cell's volume centroid.

```python
--8<-- "docs/examples/vis2_finite_volume.py:perot"
```

Although the argument is named `flux`, pass the normal velocity in m/s, not a
volumetric flux in m³/s: the method multiplies by the area itself. The
result is a `Values(data, "cells")` field of 3-vectors in the input's unit.
It reproduces a uniform vector exactly on cells with planar faces. It is
first order and does not recover an arbitrary field.
[Perot's paper](https://doi.org/10.1006/jcph.2000.6424) gives the method.

Probing and advection read cell values as constant over each cell. That is an
interpretation for analysis. Projecting the vector to `H1` averages the cells
around each point, which smooths jumps and is not a conservative remap.

## Run and check

```bash
uv run python docs/examples/vis2_finite_volume.py --arch cpu
```

The program runs every check on the two hexahedra and again after
`as_polyhedra` makes them polyhedra: volumes and centroids, the uniform
velocity's reconstruction, the two divergences, the boundary balance, the
upwind mass outflow, and the traces and jumps.

## Details

- On pieces, refresh the ghost values before anything that reads a
  neighbour's cell, and count each owned cell once when reducing.
  [Filters over pieces](over-pieces.md) shows both.
- A coarse-fine face of an octree's leaves, a T-junction, counts as boundary
  for now, since such faces are not yet paired.
- A cell of zero volume gives NaN from `divergence` and `perot`. A cell of a
  lower dimension than the mesh has no faces, and its net flux is zero.
- Limiters, boundary conditions and conservative remapping belong to the
  numerical method, not to these helpers.

## Where to go next

- [Write a mesh algorithm](write-a-mesh-algorithm.md) explains face and cell
  kernels.
- [Fields](../data/fields.md) explains spaces, for choosing an interpolation
  or a higher-order representation.
- [Derive](derive.md) covers gradients and flow quantities from fields with a
  basis.
