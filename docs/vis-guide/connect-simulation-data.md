# Connect simulation data

Bring simulation arrays or reader output into Tack, run analysis, and hand
results on. The [model](01-datasets.md) explains the objects; this chapter
owns the copy, unit, lifetime and update decisions at those boundaries.
The common model and NumPy examples do not require VTK or MFEM.

## Attach solver arrays

An existing buffer needs its topology, [coefficient layout](field-storage.md)
and unit before it is a field. Upload host values with `tack.array`, or use
the supported [array sharing interfaces](../users-guide/16-interoperability.md)
for compatible memory. Wrap that value array in `td.Field(space, values)`
and attach it by name. A nodal sample, a cell average and a normal component
cannot be identified from an array length alone.

Sharing retains the original allocation. Keep the producer and its storage
alive, coordinate external writes with Tack work, and follow each adapter's
read-only and synchronization rules. Materialization or dtype/layout conversion
can require a new allocation. Retaining a pipeline output does not promise
a snapshot of every shared buffer.

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

With `tack-vis` and VTK installed, `vtk_to_dataset` reads VTK image, rectilinear,
structured, unstructured or polygonal (`vtkPolyData`) data. Point fields become H1, cell fields Constant; geometry
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

Whole-dataset conversion copies by default. `copy=False` requests supported
host-memory sharing, including CPU and supported Metal unified-memory paths;
this route does not require VTK's DLPack module. Unsupported representations,
dtype conversions or devices can require copies. Check each adapter's
[interop reference](reference/dataset-api.md#interoperability) for the conditions.

Array-level `array_to_vtk` and `vtk_to_array` also support the DLPack route,
which needs a VTK build exposing `vtkmodules.util.dlpack_support`. Level Zero
device sharing needs a common allocation context. See
[Sharing Arrays](../users-guide/16-interoperability.md) for lifetime and device rules.



## Array-level exchange and Level Zero

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
arrays. See [Sharing arrays with other libraries](../users-guide/16-interoperability.md) for
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


## Import MFEM fields

`tack.interop.mfem.mfem_to_dataset(mesh, fields=...)` preserves the separation
between corner topology, geometry coefficients and each grid function's space.
Supported H1/L2 families and orders are listed in the [interop catalog](reference/interop.md).
Coefficient arrays in a different nodal basis may be resampled; equal lengths
do not make their order interchangeable. `mfem_grid_function` writes a field
back. The [MFEM gallery](mfem-gallery.md) supplies complete worked examples.

## Adaptive data

### AMR

`vtk_to_dataset` takes a `vtkOverlappingAMR`, and `dataset_to_vtk` gives one
back for images and rectilinear grids; a hierarchy with curvilinear boxes goes
to VTK as a `vtkPartitionedDataSetCollection` of a partitioned dataset per
level. `amr_gaussian_pulse` makes VTK's Gaussian pulse hierarchy, for tests.

### Octrees

- `OctreeDataSet.from_cells(roots, level, position, fields)` takes each cell's
  level and a position in it, as a RAMSES reader or yt gives them (whole octs),
  over a grid of root cells (an image or a rectilinear grid); it adds the
  ancestors no cell gives, their values NaN.
- `vtk_to_dataset` takes a `vtkHyperTreeGrid` (branching factor 2) and
  `dataset_to_vtk` gives one back, a mask as `HIDDENCELL`.


## Convert units before combining data

This complete example converts positions and a temperature field, then checks
derived units and parameter conversion. See the [unit reference](reference/units.md)
for the full operation catalog.

<!-- tack-check: units -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# A raster at State Plane coordinates, in US survey feet, with a temperature.
image = td.uniform_grid((5, 4, 3), spacing=(10.0, 10.0, 10.0), unit="ftUS")
image.origin = (1_400_000.0, 600_000.0, 0.0)
celsius = np.linspace(0.0, 30.0, image.num_points).astype(np.float32)
image.fields["T"] = td.Field(td.H1(image), tack.array(celsius), unit="degC")
print(image.unit, image.fields["T"].unit)

# Filters that keep a quantity keep its unit, and the spatial unit with it.
cut = td.clip(image, "T", 15.0)
print(cut.unit, cut.fields["T"].unit)
assert cut.unit == "ftUS" and cut.fields["T"].unit == "°C"

# convert scales on the device; the origin converts in double precision.
metres = td.convert(image, "m")
kelvin = td.convert(metres, "K", field="T")
assert isinstance(metres.geometry.values, tack.arrays.UniformCoordinates)  # still an image
assert np.allclose(metres.world_positions(), image.world_positions() * 1200 / 3937)
assert np.allclose(kelvin.fields["T"].values.numpy(), celsius + 273.15)

# A field converts on its own too, into a field in the same space.
fahrenheit = td.convert(image.fields["T"], "degF")
assert np.allclose(fahrenheit.values.numpy(), celsius * 9 / 5 + 32, atol=1e-4)

# Derived quantities derive their units; a difference of degrees Celsius is in
# kelvin, so a temperature gradient is in kelvin per foot.
assert td.gradients(image, "T").unit == "K/ftUS"
stats = td.statistics(image, "T")
print(stats.unit_of("mean"), stats.unit_of("population_variance"))
assert stats.unit_of("population_variance") == "K**2"

# Parameters take quantities: a threshold in degrees Celsius on a field in
# kelvin, a radius in metres on positions in feet.
warm = td.threshold(kelvin, "T", "20 degC", "30 degC")
assert warm.num_cells == td.threshold(kelvin, "T", 293.15, 303.15).num_cells
cloud = td.point_cloud(image.world_positions(), unit="ftUS")
coarse = td.voxel_downsample(cloud, "10 m")             # 32.8 US survey feet
assert coarse.num_points == td.voxel_downsample(cloud, 10 * 3937 / 1200).num_points

# Positions in two different units do not meet.
try:
    td.probe(image, td.point_cloud(np.array([[1_400_010.0, 600_010.0, 5.0]]), unit="m"))
except ValueError as error:
    print("refused:", error)
```

## Converting

`convert(field, unit)` gives a field in `unit`: a new field in the same space,
its values scaled (and offset, from degrees Celsius to kelvin, say) on the
device. `convert(data, unit, field=name)` does the same to a dataset's field
and gives the dataset with it, everything else kept, and `convert(data, unit)`
gives the dataset with its geometry in `unit`. The geometry's conversion
scales the origin on the host in double precision, so survey coordinates keep
their digits; an image stays an image and a rectilinear grid keeps its axes.
Positions are the dataset's to convert, not the geometry field's alone, which
would leave the origin in the old unit. The unit converted from must be
known. Integer fields convert only by relabelling (`"1000*m"` to `"km"`).

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
output stage when reporting application time. See [Debugging and Timing](../users-guide/15-debugging-and-timing.md).
