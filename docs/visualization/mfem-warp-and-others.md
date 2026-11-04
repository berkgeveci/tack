# MFEM, Warp and others

Besides VTK, Tack exchanges data with two libraries directly. **MFEM**, the
finite-element library, gives meshes and solutions of any order, curved and
discontinuous; Tack reads them into its spaces without resampling them onto
linear cells. **NVIDIA Warp** runs Python kernels on NVIDIA GPUs; Tack shares
arrays with it, and datasets with its simulation-data library, warp-simdata.
Data in either one goes through Tack's algorithms as it is, and the last
section says where to go for other libraries. Tack's terms are in the
[glossary](../reference/glossary.md).

## Bring an MFEM solution into Tack

MFEM comes through PyMFEM, which Tack's `interop` extra installs
(`uv sync --extra interop`, or `pip install mfem`). `mfem_to_dataset` takes an
MFEM mesh and named grid functions on it:

<!-- tack-check: mfem-import -->
```python
import mfem.ser as mfem
import numpy as np

import tack
import tack.dataset as td
from tack.interop.mfem import mfem_grid_function, mfem_to_dataset

tack.set_default_device("cpu")

mesh = mfem.Mesh.MakeCartesian2D(4, 4, mfem.Element.QUADRILATERAL, True, 1.0, 1.0)


class Cubic(mfem.PyCoefficient):
    def EvalValue(self, x):
        return x[0] ** 2 * x[1]


continuous = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(3, mesh.Dimension()))
u = mfem.GridFunction(continuous)            # H1 of order 3
u.ProjectCoefficient(Cubic())
discontinuous = mfem.FiniteElementSpace(mesh, mfem.L2_FECollection(2, mesh.Dimension()))
w = mfem.GridFunction(discontinuous)         # L2 (DG) of order 2
w.ProjectCoefficient(Cubic())

data = mfem_to_dataset(mesh, {"u": u, "w": w})
print(data.fields["u"].space, data.fields["w"].space)
# H1(order=3, basis='gauss_lobatto') L2(order=2, basis='gauss_legendre')

# The integral of x^2 y over the unit square is 1/6, from the order-3 field itself.
assert abs(td.integrate(data, "u").values["u"] - 1 / 6) < 1e-6

# And back to MFEM, the same values in MFEM's numbering.
back = mfem_grid_function(data, data.fields["u"], mesh)
assert np.allclose(back.GetDataArray(), u.GetDataArray(), atol=1e-6)
```

The mesh's elements become the topology, in VTK's corner order. Its element
attributes become an integer `Constant` field, `"attribute"`. Each grid
function becomes a field in the space its collection matches:

| MFEM collection | Tack field |
|---|---|
| `H1` of order 1 | `H1` of order 1 (point data), one value per vertex |
| `L2` of order 0 | `Constant` (cell data) |
| `L2` of order 1 | `L2` of order 1, each element's values at its corners |
| `H1` of order 2 and up, `L2` of order 2 and up | Tack's space of that order and basis |
| a vector grid function | a field of vectors, in either of MFEM's orderings |

For orders 2 and up, Tack has MFEM's common bases: Gauss-Lobatto (MFEM's
default for `H1`), Gauss-Legendre (its default for `L2`), closed uniform, and
the positive (Bernstein) bases. On those, values are copied and permuted into
Tack's numbering. Any other basis is evaluated at Tack's nodes, which is exact,
since both spaces hold the same polynomials. Triangles, tetrahedra and wedges
use Tack's equispaced nodes. [Higher order](../data/higher-order.md) explains
the spaces.

`mfem_grid_function(data, field, mesh)` gives a field back as an MFEM
`GridFunction`, its values permuted into MFEM's numbering. Keep the returned
function's `fes` and `fec` attributes alive as long as it is used.

## Kinds of MFEM mesh

- **Curved meshes.** A mesh with nodes of order 2 or more gets a geometry of
  that order, an `H1` field of positions. Filters see the curved cells: cuts
  are made on sub-cells, with each crossing on the field itself.
- **Periodic meshes** keep their nodes per element, so the geometry is an `L2`
  field. Cells on the two sides of the seam share points in the topology but
  sit apart in space.
- **Nonconforming meshes** from adaptive refinement read with their hanging
  points as points of the finer cells. Their fields come with every value MFEM
  keeps per element, so values agree across each T-junction. Contours, slices,
  clips, probes, gradients and subdivision work as on a conforming mesh.
- **MFEM's numbering.** `mfem_to_dataset(mesh, fields, dofs="table")` keeps an
  `H1` field of order 2 and up, and a curved geometry, numbered as MFEM numbers
  them: its values as they are, in a space numbered by MFEM's element-to-DOF
  table. By default they are renumbered by entity, as Tack's own spaces are.

MFEM fields above order 1 on pyramids are refused: Tack has no such space on
pyramids yet.

One limit matters for face-based algorithms. Tack does not yet know that a
coarse cell's edge or face and the finer ones covering it are the same
interface. `faces()` gives each of them one side, as if it were on the
boundary. So `boundary_faces`, `extract_surface`, `jump`, `upwind_flux`,
`net_flux` and other face-based algorithms treat T-junctions as boundary. On
MFEM's ex6 star after refinement, 1280 of the 1450 one-sided edges are
T-junctions.

The [MFEM gallery](../vis-guide/mfem-gallery.md) runs MFEM's examples through
Tack: curved tetrahedra and hexahedra of order 3, elasticity in `H1` and DG,
adaptive refinement, a curved surface, and DG advection on a periodic mesh.
Its scripts are in `docs/gallery/mfem_examples.py`.

## Share arrays and datasets with Warp

Warp support is a prototype. Tack has a Warp device, `tack.device("warp")`,
which runs Tack's kernels through Warp on Warp's default device: its first CUDA
device, else its CPU. On it, a Tack array's memory is a `wp.array`, and Tack
launches on the Warp stream a caller's `wp.launch` uses. So one program can run
its own Warp kernels and Tack's filters on the same arrays. Install it with
`uv sync --extra warp`, which adds warp-simdata; see
[Backends](../reference/backends.md) for what the Warp device lacks.

`tack.interop.warp.from_warp` and `to_warp` share arrays both ways, mapping
Warp's vector and matrix types to Tack's (`wp.vec3` and
`tack.vec(3, tack.f32)`). Here Warp makes a field, Tack contours it, and Warp
moves the contour's points:

```python
import warp as wp

import tack
import tack.dataset as td
from tack.interop.warp import from_warp, to_warp


@wp.kernel
def sphere(field: wp.array(dtype=wp.float32), n: int):
    t = wp.tid()
    i, j, k = t % n, (t // n) % n, t // (n * n)
    p = wp.vec3(float(i), float(j), float(k)) - wp.vec3(float(n - 1) * 0.5)
    field[t] = wp.length(p)


@wp.kernel
def lift(points: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    out[t] = points[t] + wp.vec3(0.0, 0.0, 100.0)


device = tack.device("warp")
n = 24
f = wp.empty(n ** 3, dtype=wp.float32)
wp.launch(sphere, n ** 3, inputs=[f, n])
with device:
    data = td.uniform_grid((n, n, n))
    data.fields["r"] = td.Field(td.H1(data), from_warp(f))   # Warp's memory, no copy
    iso = td.contour(data, "r", 8.0)
points = to_warp(iso.geometry.values)                         # a wp.array of wp.vec3
out = wp.empty_like(points)
wp.launch(lift, points.shape[0], inputs=[points, out])
```

Each side keeps the other's memory alive for as long as it is used. An array
on another Tack device goes to Warp through DLPack, which waits for its queued
launches. An implicit array has no memory, so `to_warp` stores its values
first. Warp arrays have no read-only flag, so a read-only Tack array given to
Warp can be written there.

`to_simdata` and `from_simdata` exchange whole datasets with warp-simdata.
Image data, structured grids, polydata and unstructured grids (polyhedra
included) map to Tack's topologies. NODE fields become `H1` fields and ELEMENT
fields `Constant` ones, and collections become `PartitionedDataSet`s. Arrays
that both store alike are shared. Several of warp-simdata's other models come
across as they are, sharing their arrays: point clouds, surface meshes,
OpenFOAM meshes, CGNS sections, EnSight parts, corner-point grids and FLASH
data. The rest go through warp-simdata's own VTK conversion.
`to_simdata` leaves behind what warp-simdata has no place for: ghost marks,
global ids, sets, units and time. It writes positions as single-precision world
positions, so far coordinates lose the precision Tack's origin kept.

## Other libraries

- **PyTorch, CuPy, NumPy and anything with DLPack** share arrays with Tack
  directly; see [Share arrays](../core/share-arrays.md). Wrap the shared array
  in a `td.Field` to give it a meaning in a dataset.
- **A simulation's own arrays**, on the host or on its device, become a dataset
  through [Connect simulation data](../data/connect-simulation-data.md).
- **ParaView** reads what VTK writes; see [VTK](vtk.md) for writing files and
  for Tack's filters in VTK pipelines.

## Details

- MFEM lists every element's corners in VTK's order, the prism included. MFEM's
  own VTK writer swaps a prism's corners, which turns such a prism inside out
  against VTK's convention; `mfem_to_dataset` reads the mesh directly and is
  not affected.
- The full signatures are in the
  [Data Model reference](../vis-guide/reference/data-model.md#interoperability).

## Where to go next

- The [MFEM gallery](../vis-guide/mfem-gallery.md): MFEM's examples, drawn
  through Tack.
- [Higher order](../data/higher-order.md): the spaces MFEM's fields arrive in.
