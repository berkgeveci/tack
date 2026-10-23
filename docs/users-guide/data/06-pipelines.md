# Sample and connect simulations

A useful visualization pipeline connects a simulation's fields to a scientific
question and an output. Keep the data's space and lifetime explicit at each
stage: a streamline needs an interpolated velocity, and a file writer needs
host-accessible data in its format.

```mermaid
flowchart LR
    A[Simulation or reader] --> B[Dataset and named fields]
    B --> C[Derived fields and selections]
    C --> D[Contour, slice or boundary]
    D --> G
    C --> F[Probe or integrate trajectories]
    C --> G[Export a dataset]
```

## Sample at physical positions

`probe` evaluates a source dataset's fields at positions in world coordinates.
The example asks for two points in the mesh and one outside it:

```python
--8<-- "docs/examples/data_analysis.py:probe"
```

The returned dataset contains the query positions as vertex cells, sampled
fields as point data, and `valid` as a point attribute. Invalid samples are
zero-filled. Always consult `valid`: zero may also be a valid physical result.
The example checks that valid scalar samples equal the query x coordinates.

Passing a target dataset instead of a point array preserves that target's
topology and geometry, samples at its point positions, and adds the source
fields. This is a way to resample onto another grid. A name collision is
resolved by the sampled field replacing the target's field of that name.

Reuse a `CellLocator` when geometry is fixed and only the sampled fields change.
Rebuild it after moving geometry; its spatial bins describe the mesh when it
was constructed. Where a query lies in more than one cell, the smallest cell
id wins (a query on a face, edge or vertex that cells share is in all of them);
a cell the query is only within the tolerance of counts only where no cell
holds it. At a discontinuity that choice can determine which side is
sampled; it is not an average across the interface.

Probing takes shape-based cells, polyhedra (located in sub-tetrahedra), polygons
(in the triangles fanning each from its points' mean) and polylines (on their
segments). A `Values` field on faces is not a
continuous field that can be sampled arbitrarily through the volume.

## Integrate a steady velocity field

```python
--8<-- "docs/examples/data_analysis.py:flow"
```

`advect` returns the particles where they stop, with `steps`, `time` and
`status`. `streamlines` returns the paths as a polyline per seed that moved,
with `seed` on cells and `time` on points. It also keeps a point for a seed
that did not move, without inventing a line.

The default integrator is fixed-step RK4; Euler is also available. The step
is in the integration parameter for `dx/dt = velocity`, so its spatial length
depends on speed. This is not an adaptive error-controlled integration.

The example's `[-y, x, 0]` velocity rotates a point at unit angular speed.
Twenty steps of 0.05 reach time 1. The program checks the final positions
against sine and cosine, and checks path lengths against the radii times
elapsed time, allowing for polyline chord approximation.

Particles stop at the step limit, zero velocity, or domain bounds; an outside
seed does not move. Inspect status and time before treating a short path as
evidence of low speed. These functions sample one steady field. A pathline
through a changing simulation requires a time-dependent integration strategy;
repeating a steady streamline calculation does not by itself provide one.

## Keep a simulation's state in the pipeline

A common pattern is to retain topology and geometry, attach each working field
once, update its values with simulation kernels, and run the analysis stages
at selected timesteps. Intermediate datasets hold results of those stages;
they are not a graph that recomputes automatically when an input changes.

| What changed | What needs attention |
|---|---|
| Values in an existing simulation field | Re-run dependent filters, samples and summaries |
| A field was replaced by a new buffer | Update the named `td.Field` attachment |
| Point positions changed | Rebuild locators |
| Connectivity changed | Construct a corresponding topology, spaces and associations |

Prefer kernels that compute through `data.geometry.values` to downloading all
positions just to form a derived field on the host. Keep readback at checkpoints
or output boundaries. A named field holding simulation storage can share its
allocation: follow the same ownership and synchronization rules as any other
shared buffer.

## Exchange whole datasets with VTK

With `tack-vis` and VTK installed, `vtk_to_dataset` reads VTK image, rectilinear
or unstructured data. Point fields become H1, cell fields Constant; geometry
and connectivity are converted into the dataset model. `dataset_to_vtk` creates
the matching VTK dataset for visualization or writing to a file.

For a contour, the VTK result is an unstructured triangle mesh. The following
fragment assumes `surface` was produced by `td.contour`:

```python
from tack.interop.vtk import dataset_to_vtk
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

vtkXMLUnstructuredGridWriter(file_name="surface.vtu")(dataset_to_vtk(surface))
```

Tack uses VTK's pythonic API (VTK 9.7.1 or later) in its interop and
examples: keyword constructors, properties, a writer or filter called on
its input. A contour's cells come across as VTK's own implicit arrays: its
cell types a `vtkConstantArray`, its offsets a `vtkAffineArray`.

These whole-dataset conversions copy arrays and provide a straightforward
host boundary. Zero-copy array exchange is a separate route using
`array_to_vtk`, `vtk_to_array` and a DLPack-enabled VTK build. A standard VTK
wheel sufficient for file I/O may not expose that route. Level Zero sharing
also needs a shared allocation context. See [Sharing Arrays](../16-interoperability.md).

MFEM meshes and grid functions enter through `tack.interop.mfem`; supported
spaces include H1 orders 1 and 2, and L2 orders 0 and 1, with supported curved
geometry at order 2. Check the [interop reference](../../reference/dataset-api.md#interoperability)
for signatures and the current mapping limits.

## Measure the boundaries you actually use

| Stage | Host interaction to account for |
|---|---|
| Construct from NumPy or a reader | Input upload and topology setup |
| Derive faces, edges or a locator | First-use work and retained auxiliary storage |
| Extract variable-size geometry | Count readbacks and output allocation |
| Return a scalar summary | Device wait and result readback |
| Probe or integrate | Locator setup, query upload; streamlines also read path lengths for output allocation |
| Export to VTK or a file | Array copies and writer time |

Queued GPU launches preserve Tack's program order, but host reads synchronize.
Call `tack.sync()` before stopping a timer when the next operation is not a
readback. Separate first-use compilation from repeated work, and include the
output stage when reporting application time. See [Debugging and Timing](../15-debugging-and-timing.md).

## Complete programs to run next

```bash
uv run python docs/examples/data_analysis.py --arch cpu --check
uv run python docs/examples/datasets.py --arch cpu --check
```

The [analysis program](../../examples/data_analysis.py) covers probing and flow,
and the [dataset tour](../../examples/datasets.py) connects mixed meshes,
spaces, sets and streamlines.
