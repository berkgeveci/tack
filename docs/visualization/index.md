# Visualization: filters you know

Contour, slice, clip, threshold, gradients, probe, streamlines, statistics:
Tack has the filters a VTK or ParaView user already knows, and they give
VTK's results. They are written in Python on Tack's kernels, and they run on
the CPU or a GPU, wherever the dataset lives. This page runs two of them,
shows where the others are, and lists the conventions they all follow.

## Make a dataset and filter it

The example makes the image of VTK's wavelet source, computes the gradient of
its field at the points, and extracts an isosurface. It needs nothing but
Tack.

<!-- tack-check: first-pipeline -->
```python
import tack
import tack.dataset as td

tack.set_default_device("cpu")           # or "metal", "cuda", "hip", "level_zero"

data = td.wavelet(20)                    # an image of 41 x 41 x 41 points, field "RTData"
data.fields["gradient"] = td.gradients(data, "RTData", at="points")
surface = td.contour(data, "RTData", 157.0)

print(surface.num_cells, "triangles carrying", list(surface.fields))
# 25880 triangles carrying ['RTData', 'gradient']
assert surface.num_cells == 25880        # VTK's contour of the same image has as many
```

`td.wavelet` makes the same image as VTK's `vtkRTAnalyticSource`, with the
point field `RTData`. A Tack [dataset](../reference/glossary.md#dataset)
holds what a VTK dataset holds: a topology (how points join into cells), a
geometry (the positions) and [fields](../reference/glossary.md#field).
VTK's point data is an [`H1`](../reference/glossary.md#h1) field of order 1,
with one value per point, linear in each cell. VTK's cell data is a
[`Constant`](../reference/glossary.md#constant) field, with one value per
cell.

Each filter takes the dataset and returns something new. `gradients` returns
a field, and the example puts it in `data.fields` so that `contour` carries it
onto the surface. `contour` returns a new dataset of triangles. If datasets,
fields and filters are new to you, read
[the data model's first page](../data/index.md) before going on.

## Read and draw with VTK

Tack does not read most file formats, and it does not render. VTK does both,
so a pipeline on your own data reads with VTK, filters with Tack, and writes
or draws with VTK:

```mermaid
flowchart LR
    A["VTK reader"] -->|vtk_to_dataset| B["Tack dataset"]
    B --> C["Tack filters<br/>(on the dataset's device)"]
    C -->|dataset_to_vtk| D["VTK writer<br/>or renderer"]
```

`vtk_to_dataset` turns a VTK dataset into a Tack dataset. Its point data
becomes `H1` fields and its cell data `Constant` fields. `dataset_to_vtk`
turns a result back. The contour above would become a
`vtkUnstructuredGrid` of triangles, which a writer saves or a mapper draws.
With `copy=False` the two share arrays instead of copying them. The [VTK page](vtk.md#convert-a-dataset-both-ways)
shows the pipeline in code. It also covers partitioned and AMR datasets,
higher-order cells, and Tack's filters as VTK algorithms in a VTK pipeline.

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
VTK's output. Four things work differently:

- **Fields have a space, not only an association.** A field may also jump
  between cells, as a discontinuous Galerkin solution does, and be of any
  polynomial order; filters use the field's own interpolation
  ([fields](../data/fields.md), [higher order](../data/higher-order.md)).
- **Results do not depend on thread scheduling.** Two runs on a device give
  the same bits.
- **Pieces are first-class.** A filter given one piece computes that piece's
  share of the whole dataset's output ([filters over pieces](over-pieces.md)).
- **Your filters see the mesh as the built-in ones do**, through public views,
  so they can be as fast ([write a filter](write-a-filter.md)).

## How fast, against VTK

On production data read with VTK's readers, on an Apple M1 Max's CPU, Tack's
filters are mostly faster than VTK's on 8 threads: point gradients 14 to 33
times, isosurfaces 2.7 to 4.3 times. Not every filter is ahead: a slice of the
ONERA M6 wing takes as long, and its outer surface slightly longer. On a GPU
the filters are 11 to 2,269 times faster than VTK on the CPU. The
[performance page](../performance.md) has every measurement, the machines and
the versions.

## Where to go next

- [Extract parts of a dataset](extract.md): contours, slices, clips,
  thresholds and surfaces, the filters most pipelines start with.
- [The data model](../data/index.md), if `H1`, `L2` and `Constant` are new:
  what a dataset holds and how VTK's datasets map onto it.
- [VTK](vtk.md), for the conversions in full and Tack's filters inside a VTK
  pipeline.
