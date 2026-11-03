# Visualization: filters you know

This part of the documentation is about Tack's visualization and analysis
algorithms: contour, slice, clip, threshold, gradients, probe, streamlines,
statistics and the other filters a VTK or ParaView user already knows. They
are written in Python on Tack's kernels and run on the CPU or the GPU, wherever
the dataset lives. This page is for readers who know VTK. After it you can run
a first pipeline (read with VTK, filter with Tack, write or draw with VTK),
find the Tack function for a VTK filter, and know the conventions every Tack
filter follows.

## Read with VTK, filter with Tack, write with VTK

The example below first makes a file with VTK's wavelet source, so that it has
something to read. The pipeline itself is the second half: VTK reads the file,
Tack computes the point gradients and an isosurface, and VTK writes the
result.

<!-- tack-check: first-pipeline -->
```python
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource
from vtkmodules.vtkIOXML import (vtkXMLImageDataReader, vtkXMLImageDataWriter,
                                 vtkXMLUnstructuredGridWriter)

import tack
import tack.dataset as td
from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

tack.set_default_device("cpu")

# A file to read: VTK's wavelet, 41 x 41 x 41 points.
writer = vtkXMLImageDataWriter(file_name="wavelet.vti")
vtkRTAnalyticSource(whole_extent=(-20, 20, -20, 20, -20, 20)) >> writer
writer.Write()

# The pipeline.
image = vtkXMLImageDataReader(file_name="wavelet.vti")()
data = vtk_to_dataset(image)
data.fields["gradient"] = td.gradients(data, "RTData", at="points")
surface = td.contour(data, "RTData", 157.0)

grid = dataset_to_vtk(surface)
out = vtkXMLUnstructuredGridWriter(file_name="surface.vtu")
grid >> out
assert out.Write() == 1
print(surface.num_cells, "triangles carrying", list(grid.point_data.keys()))
# 25880 triangles carrying ['RTData', 'gradient']

# VTK's contour of the same image has the same triangles.
vtk_contour = vtkContourFilter()
vtk_contour.SetValue(0, 157.0)
assert (image >> vtk_contour)().number_of_cells == surface.num_cells
```

Open `surface.vtu` in ParaView, or hand `grid` to a VTK mapper. Tack has no
renderer of its own: VTK draws what Tack computes.

The pipeline has three stages:

```mermaid
flowchart LR
    A["VTK reader"] -->|vtk_to_dataset| B["Tack dataset"]
    B --> C["Tack filters<br/>(on the dataset's device)"]
    C -->|dataset_to_vtk| D["VTK writer<br/>or renderer"]
```

- `vtk_to_dataset` turns a VTK dataset into a Tack
  [dataset](../reference/glossary.md#dataset): a topology (how points join into
  cells), a geometry (the positions) and [fields](../reference/glossary.md#field).
  VTK's point data become [`H1`](../reference/glossary.md#h1) fields and its
  cell data [`Constant`](../reference/glossary.md#constant) fields.
  An `H1` field of order 1 is exactly point data: one value per point, linear
  in each cell. A `Constant` field is cell data: one value per cell. With
  `copy=False` the Tack dataset shares VTK's arrays instead of copying them.
- Each filter takes the dataset and returns something new. `gradients`
  returns a field, which the example puts in `data.fields` so that `contour`
  carries it onto the surface. `contour` returns a new dataset.
- `dataset_to_vtk` turns the result back into a VTK dataset, here a
  `vtkUnstructuredGrid` of triangles. `H1` fields become point data and
  `Constant` fields cell data.

The [VTK page](vtk.md) covers the conversions in full: sharing memory,
partitioned and AMR datasets, higher-order cells, and Tack's filters as VTK
algorithms that sit in a VTK pipeline (`tack.interop.vtk.algorithms`).

## Find the Tack filter for a VTK filter

Every filter is a function in `tack.dataset`, imported here as `td`. The table
lists the VTK filters this part of the documentation covers, by task, with the
page that shows each one. The
[algorithms reference](../vis-guide/reference/algorithms.md) has every
algorithm's full signature and its Viskores counterpart.

| VTK | Tack | Page |
|---|---|---|
| `vtkContourFilter`, `vtkFlyingEdges3D` | `contour` | [Extract](extract.md) |
| `vtkCutter` | `slice`, `slice_plane` | [Extract](extract.md) |
| `vtkTableBasedClipDataSet`, `vtkClipDataSet` | `clip` | [Extract](extract.md) |
| `vtkThreshold` | `threshold` | [Extract](extract.md) |
| `vtkExtractCells`, `vtkExtractGeometry` | `extract_cells`, `extract_geometry` | [Extract](extract.md) |
| `vtkExtractGrid`, `vtkExtractVOI` | `extract_extent` | [Extract](extract.md) |
| `vtkDataSetSurfaceFilter`, `vtkGeometryFilter` | `extract_surface` | [Extract](extract.md) |
| `vtkExtractEdges`, `vtkFeatureEdges` | `extract_edges`, `feature_edges` | [Extract](extract.md) |
| `vtkPolyDataNormals` | `surface_normals` | [Extract](extract.md) |
| `vtkGradientFilter` | `gradients`, `flow_quantities` | [Derive](derive.md) |
| `vtkCellSizeFilter`, `vtkCellCenters` | `cell_geometry`, `cell_centers` | [Derive](derive.md) |
| `vtkMeshQuality` | `cell_quality` | [Derive](derive.md) |
| `vtkIntegrateAttributes` | `integrate` | [Derive](derive.md) |
| `vtkCellDataToPointData`, `vtkPointDataToCellData` | `project` | [Derive](derive.md) |
| `vtkProbeFilter`, `vtkResampleWithDataSet`, `vtkResampleToImage` | `probe` | [Sample and trace](sample-and-trace.md) |
| `vtkStreamTracer` | `streamlines`, `advect` | [Sample and trace](sample-and-trace.md) |
| `vtkDescriptiveStatistics` | `statistics`, `entropy` | [Reduce](reduce.md) |
| `vtkConnectivityFilter`, `vtkEuclideanClusterExtraction` | `connected_components` | [Reduce](reduce.md) |

Other tasks have pages of their own:
[point clouds and particles](points-and-particles.md) (VTK's
vtkFiltersPoints), [AMR and octrees](adaptive-data.md),
[finite-volume fluxes](finite-volume.md), [filters over pieces](over-pieces.md)
(`vtkGhostCellsGenerator`, `vtkGenerateGlobalIds`, `vtkMergeBlocks`), and
[steps over time](over-time.md) (`vtkParticlePathFilter`,
`vtkStreaklineFilter`, temporal statistics).

## What every filter has in common

Tack's filters follow one set of conventions, so what you learn on one applies
to all of them.

- **The dataset comes first.** A field argument is a `td.Field` or the name of
  one of the dataset's fields: `td.contour(data, "RTData", 157.0)`.
- **Nothing changes its input.** A filter that makes geometry returns a new
  dataset. One that computes a field returns a `Field`, and one that computes
  several returns a named tuple of them. A summary returns Python numbers. To
  keep a computed field with the dataset, put it in `data.fields`.
- **`fields=` chooses what the output carries.** By default a filter that
  makes a dataset carries every field it can. Pass a name, a list of names, or
  `[]` for the geometry alone.
- **Filters run on the dataset's device.** A dataset on a GPU is filtered on
  that GPU, and its result stays there. `data.to_device("cuda")` moves a
  dataset, as the [devices page](../data/devices.md) explains.
- **Parameters may carry units.** A length, a time or a field value (a radius,
  a step size, a threshold's range) also accepts a quantity such as `"10 cm"`,
  which is converted to the dataset's or the field's unit. A plain number is
  used as it is. Fields keep their units through filters
  ([units](../data/units-coordinates-time.md)).
- **Hidden cells are never processed.** Cells that the dataset's
  [ghost marks](../reference/glossary.md#ghost-marks) mark hidden are left out of every
  output. Summaries such as `statistics` also leave out the ghost copies that
  another piece owns.

## How Tack's filters differ from VTK's

The names and the results are VTK's, and Tack's tests hold its filters to
VTK's output. Four things work differently.

**Fields have a space, not only an association.** Besides `H1` (point data)
and `Constant` (cell data), a field may be `L2`: values that belong to each
cell alone, so the field may jump between cells, as a discontinuous Galerkin
solution does. `H1` and `L2` fields may be of any order, polynomials inside
each cell. The filters use the field's own interpolation, so a contour of an
order-4 field lies on that field's level set, not on a linear approximation of
it. The [fields page](../data/fields.md) explains spaces, and the
[higher-order page](../data/higher-order.md) the polynomial ones.

**Results do not depend on thread scheduling.** The filters are built from
scans, sorts and segmented reductions, mostly without atomic operations. Their
output on a device does not change with how its threads happened to run, so
two runs give the same bits.

**Pieces are first-class.** A filter given one piece of a
[`PartitionedDataSet`](../reference/glossary.md#piece) computes that piece's share of
the whole dataset's output. The summaries take all the pieces at once, over
MPI ranks or not. [Filters over pieces](over-pieces.md) explains this.

**Your filters see the mesh as the built-in ones do.** The views a filter uses
(cells, faces, a field's values at a cell's corners) are public, so a filter
you write runs as fast as a built-in one. See
[write a filter](write-a-filter.md).

## How fast, against VTK

Tack's filters were measured against VTK's on data from production
simulations, read with VTK's readers. Tack ran on the CPU and VTK on 8
threads, both on an Apple M1 Max. Every Tack output was checked against VTK's
before its time counted. The numbers say how many times faster Tack is: VTK's
time over Tack's.

<!-- The skeleton asks for the AhmedML car's filters here (point gradients 0.84 s
against VTK's 159). Those numbers are not in the repository yet; these are
performance.md's "Real data, against VTK". -->

| dataset | filter | VTK / Tack |
|---|---|---|
| ONERA M6 wing, 46.9M tetrahedra and wedges | isosurface | 3.2× |
| | slice | 1.0× |
| | gradients | 14.6× |
| | low-density cells (threshold) | 1.3× |
| | outer surface of a fresh mesh | 0.95× |
| aneurysm blood flow, 8.7M tetrahedra | isosurface, slice, gradients | 2.7×, 2.0×, 17× |
| LANL wildfire, 18.7M curvilinear cells | plume, slice, gradients, fuel cells, surface | 4.3×, 1.9×, 33×, 4.0×, 5.9× |

On meshes of 20M cells and a 28-core CPU, Tack is ahead of VTK on every filter
measured. On an RTX 6000 Ada GPU it is 11 to 2,269 times faster than VTK on
the CPU. The [performance page](../performance.md) has every measurement, the
machines and the versions.

## Where to go next

- [Extract parts of a dataset](extract.md): contours, slices, clips,
  thresholds and surfaces, the filters most pipelines start with.
- [The data model](../data/index.md), if `H1`, `L2` and `Constant` are new:
  what a dataset holds and how VTK's datasets map onto it.
- [VTK](vtk.md), for the conversions in full and Tack's filters inside a VTK
  pipeline.
