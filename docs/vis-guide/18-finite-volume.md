# Finite-volume data and fluxes

Start here when a simulation supplies cell averages, cell-centered vectors or
fluxes through faces. Tack can keep these associations and work directly over
cells and faces. Polynomial bases and nodal interpolation become relevant
when you choose a reconstruction for sampling or visualization.

Cell-centered finite volume stores control-volume averages and forms balances
from face fluxes. Some staggered schemes store normal velocity components on
faces; other arrangements, including vertex-centered FV, also exist. This
chapter follows the cell/face workflow; [NIST's FV introduction](https://pages.nist.gov/fipy/en/stable/numerical/discret.html)
explains the different control-volume arrangements.

## Represent quantities in their original association

| Simulation quantity | Tack space | Interpretation |
|---|---|---|
| Cell average, represented as piecewise constant | `Constant(data)` | One independent constant in each cell |
| Cell-centered attribute or reconstructed vector | `Values(data, "cells")` | One value per cell, without an interpolation basis |
| Face pressure, area or label | `Values(data, "faces")` | One attribute per face |
| Signed normal component or integrated flux | `Values(data, "faces", oriented=True)` | Positive along the normal out of side 0 |

An array of cell averages does not specify a higher-order variation inside
each cell. `Constant` supplies a piecewise-constant interpretation; `Values`
records the association without adding that interpretation. Neither choice
converts an average into the exact point value of an unknown original field.
Some sampling algorithms explicitly read cell `Values` as piecewise constant.

The mesh still has point coordinates and cell connectivity. Its geometry is
separate from the solution's cell and face data. Structured, unstructured and
polyhedral topologies use the same field associations and face-based kernels.
In 2D a cell's faces are its edges: face "area" is edge length and cell
measure is area.

The [complete program](../examples/data_finite_volume.py) uses two hexahedra
and repeats every check after converting them to a polyhedral topology. It
needs NumPy and a Tack backend, with no VTK dependency. Begin with:

```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")
```

```python
--8<-- "docs/examples/data_finite_volume.py:mesh"
```

Attach density averages of 1 and 2 kg/m³ and obtain the geometric measures:

```python
data = grid
```

```python
--8<-- "docs/examples/data_finite_volume.py:cells"
```

`cells.measure` holds volumes; `cells.centroid` holds volume centroids. Use
these centroids for volume-based calculations. `td.cell_centers` instead
maps the reference-cell center, or uses a polyhedron's point mean; it need
not be the volume centroid on a distorted cell.

## One shared face, with a defined orientation

A face has side 0 and, for an interior face, side 1. Its normal points out of
side 0. The face's scalar flux is therefore positive out of the side-0 cell
and negative out of the side-1 cell. A boundary face has only side 0.
This is Tack's equivalent of an owner/neighbour convention; it need not use
the same ordering as the simulation's faces.

Use `oriented=True` for normal components and signed fluxes. If a filter
reverses a carried face, it also negates these values. A pressure, area or
label does not change sign with the normal and should be unoriented.
The flag supports propagation; it cannot check that imported values have the
correct sign or physical units.

Map imported face ids and orientation to Tack's faces before attaching
solver face arrays. Derived face numbering is local to the topology; array
length alone does not establish correspondence. Building a polyhedral
topology from explicit faces can preserve the input face ordering, provided
its incidence and side conventions agree. See [Build Meshes](02-meshes.md)
and the [face view methods](reference/dataset-api.md#kernels).

## Normal velocity and integrated flux are different inputs

For a unit face normal `n_f`, face area `A_f` and velocity `u_f`:

```text
un_f = dot(u_f, n_f)       normal velocity:       m/s
Q_f  = A_f * un_f          volumetric face flux:  m³/s
M_f  = rho_f * Q_f         mass face flux:        kg/s
```

`perot` takes `un_f`. `net_flux` and `divergence` take an integrated flux
such as `Q_f` or `M_f` when computing a physical balance. They do not infer
which quantity an array contains or automatically multiply it by face area.

This face kernel constructs `un_f` and `Q_f` for a uniform velocity, plus a
second volumetric flux for the test flow `u(x,y,z) = (x,y,z)`:

```python
--8<-- "docs/examples/data_finite_volume.py:face-kernel"
```

The loop variable belongs to the face view; `faces.entity_id(f)` indexes
the arrays allocated for all faces. Each face has one writer, so this
calculation needs no atomics. Allocate and attach the three quantities:

```python
--8<-- "docs/examples/data_finite_volume.py:faces"
```

## Check the cell and boundary balance

With `s_cf` equal to +1 for side 0 and −1 for side 1:

```text
net_c = sum_f s_cf * Q_f
div_c = net_c / V_c
```

`net_flux` is an outward sum of the supplied values. `divergence` divides
that sum by cell measure. An internal face contributes equal and opposite
amounts to its two cells, so the sum of all cell outflows equals the boundary
outflow. The complete program checks this independently on the boundary ids.

```python
--8<-- "docs/examples/data_finite_volume.py:balance"
```

For `u = (x,y,z)`, the divergence is 3/s. Each example cell has volume
0.5 m³ and net outflow 1.5 m³/s; the cube's boundary outflow is 3 m³/s. For
the uniform velocity, both cell divergences are zero. These analytical
answers test orientation and area/volume factors, not just a plausible image.
`flow_quantities` computes a different divergence from a field's gradient;
use the face-flux balance when that is what the simulation supplies.

## Reconstruct a cell vector with Perot

`td.perot` reconstructs one three-component vector per cell from scalar
face-normal components:

```text
u_c = sum_f s_cf * A_f * un_f * (x_f - x_c) / V_c
```

Here `x_f` is the face area centroid and `x_c` the cell volume centroid.
Tack supplies the geometry and orientation. Its first-order reconstruction
reproduces uniform vectors on nondegenerate 3D cells with planar faces; it
does not exactly recover an arbitrary original vector field. Nonplanar
faces use Tack's geometric conventions. A cell of zero volume gives NaN.
See [Perot's paper](https://doi.org/10.1006/jcph.2000.6424) and
[USACE's related 2D reconstruction](https://www.hec.usace.army.mil/confluence/hecras/beta/technical-reference/numerical-methods/shallow-water-equations-solver/momentum-equation/cell-velocities).

```python
--8<-- "docs/examples/data_finite_volume.py:perot"
```

Although the API argument is named `flux`, pass normal velocity in m/s,
not volumetric flux in m³/s: Perot already multiplies by area. For mass
flux, obtaining a normal velocity also requires the appropriate density.
Keep the original face field; the reconstructed vector is another quantity.

The result uses `Values(data, "cells")` and retains the input component's
unit. Algorithms such as probing and advection can read these cell values
piecewise constantly. That is an interpretation for analysis, not a promise
that resampling the vector will recover every original face flux. Projecting
to `H1(data)` averages incident cell values at shared points, which can smooth
jumps and does not establish conservative remapping.

## Transport and present limits

For a scalar carried by a constant prescribed velocity, `upwind_flux` already
returns an integrated, oriented face flux:

```python
--8<-- "docs/examples/data_finite_volume.py:upwind"
```

At an interior face it takes the upwind cell value. At a boundary it uses the
one available cell value even for inflow; prescribe your own boundary flux
when that rule does not represent the simulation. This helper is not a full
FV solver or a general variable-velocity reconstruction scheme.

For partitioned data, perform the required ghost refresh before
neighbor-dependent work and count owned entities once when reducing the
whole mesh. See [Data in Pieces](07-partitioned.md). Coarse/fine T-junction
faces are currently treated as boundary by face-based algorithms; their
pairing is planned. The [cell/basis guide](reference/interop.md#reading-and-writing-other-formats)
describes this limit. Additional choices about limiting, boundary conditions
and conservative remapping belong to the numerical method.

## Run and check

```bash
uv run python docs/examples/data_finite_volume.py --arch cpu --check
uv run python docs/examples/data_finite_volume.py --arch metal --check
```

Use a backend installed on the machine. The script checks both shaped and
polyhedral meshes, centroid/volume values, uniform-flow reconstruction,
zero/linear-flow divergence, boundary cancellation and upwind mass outflow.
Device readback occurs at its explicit reporting/checking checkpoints.

Continue with [Write Mesh Algorithms](05-mesh-kernels.md) for custom face and
cell kernels, or [Fields and Spaces](03-fields-and-spaces.md) when choosing
an interpolation or higher-order representation.
