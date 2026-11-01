# Introduction

Tack connects simulation and instrument data to analysis on CPUs and GPUs.
Use the quick start to run a filter, then choose a reading route below.
The [background](index.md) explains the visualization, VTK and Viskores context.

```bash
uv sync --extra cpu
```

## Quick start

For this first example, use `uniform_grid`, a **shortcut factory** that
assembles a structured topology and uniform geometry for you. Add a scalar
field, then extract an isosurface. [Build Meshes](02-meshes.md) shows how to construct these pieces explicitly for several mesh types.

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
print(td.statistics(data, "radius_squared").mean)
```

Save this as a `.py` file and run it after installing the packages above, or use
`uv run python` from a checkout installed as in [Getting Started](../users-guide/01-getting-started.md).
The scalar coefficients are sampled from `x*x + y*y + z*z`; the contour is
an approximation to a sphere. This example needs no VTK installation.

## Learn data and visualization

### A first reading

Read [Understand a Dataset](01-datasets.md),
[Topology and Geometry](topology-and-geometry.md),
[Fields and Spaces](03-fields-and-spaces.md), then
[Coordinates, Units and Time](12-units.md). These share one small example;
advanced FE theory and layout formulas are optional.

### Find your representation

| Your data | Model chapter | Next task |
|---|---|---|
| Uniform, rectilinear or unstructured mesh | [Topology and Geometry](topology-and-geometry.md) | [Build Meshes](02-meshes.md) |
| Polyhedra, polygons or paths | [Their piecewise model](18-polyhedra.md) | [Build Meshes](02-meshes.md#construct-polyhedra-from-face-rings) |
| Particles or survey points | [Point Clouds](point-cloud-model.md) | [Process Point Clouds](08-point-clouds.md) |
| Domain in pieces | [Collections and Partitions](collections-and-partitions.md) | [Data in Pieces](07-partitioned.md) |
| Refined boxes | [AMR Hierarchies](10-amr.md) | [Analyze Adaptive Data](adaptive-data.md) |
| Tree-based refinement | [Octrees](11-octrees.md) | [Analyze Adaptive Data](adaptive-data.md#construct-and-summarize-an-octree) |

### Advanced routes

For FE/DG input, read [Function Spaces and Higher-order Geometry](00-function-spaces.md),
the [capability catalog](16-cells-and-bases.md), and
[Create and Convert Fields](create-fields.md). For solver memory, use
[Arrays, Storage and Layouts](field-storage.md) and
[Connect Simulation Data](connect-simulation-data.md). Kernel authors can
go through [Views](17-views.md), [Mesh Algorithms](05-mesh-kernels.md) and
[Write a Filter](15-write-a-filter.md), with
[view composition](design/view-composition.md) available for deeper inspection.

## Choose a task

| Goal | Chapter |
|---|---|
| Construct or attach data | [Build Meshes](02-meshes.md), [Create Fields](create-fields.md), [Connect Simulation Data](connect-simulation-data.md) |
| Extract geometry and derive quantities | [Filter and Analyze](04-filters.md) |
| Probe, voxelize or integrate steady paths | [Sample and Trace Fields](06-pipelines.md) |
| Balance face fluxes and reconstruct velocity | [Finite-volume Data and Fluxes](18-finite-volume.md) |
| Clean, grid or summarize particles | [Process Point Clouds](08-point-clouds.md) |
| Process refined representations | [Analyze Adaptive Data](adaptive-data.md) |
| Follow changing fields | [Stream Steps](13-streaming.md) |
| Write kernels and complete filters | [Write Mesh Algorithms](05-mesh-kernels.md), [Write a Filter](15-write-a-filter.md) |

## Common conventions

Every algorithm takes the dataset first and returns something new: a
dataset, a field, or Python values. Nothing changes its input.

Fields passed to an algorithm belong to that dataset's topology. `dot`,
`cross` and `composite` require matching value types; an explicit
`tack.arrays.Cast(field.values, tack.f32)` converts values when needed. Integer
inputs to magnitude, logarithms and statistics compute in floating point.

## Running the examples

```bash
uv run python docs/examples/validate.py --arch cpu
```

The [worked examples](examples.md) have numerical checks; the
[gallery](gallery.md) and [MFEM gallery](mfem-gallery.md) show their results.
The complete [mixed-dataset tour](../examples/datasets.py) remains available
as a runnable program. Reference readers can go directly to
[Dataset API](reference/dataset-api.md), [Algorithms](reference/algorithms.md),
[Data Model](reference/data-model.md), [format mappings](reference/interop.md)
or [units](reference/units.md).

??? info "Related detailed sections"
    These topics are developed in their own chapters.

    <span id="the-model" data-guide-forward="../01-datasets/#four-parts-of-one-dataset"></span>
    [The model](01-datasets.md#four-parts-of-one-dataset).

    <span id="sets" data-guide-forward="../01-datasets/#a-simpler-first-order-field"></span>
    [Sets](01-datasets.md#a-simpler-first-order-field).

    <span id="building-a-dataset" data-guide-forward="../02-meshes/#image-rectilinear-and-curvilinear-data"></span>
    [Building a dataset](02-meshes.md#image-rectilinear-and-curvilinear-data).

    <span id="factory-methods-are-shortcuts" data-guide-forward="../02-meshes/#factory-methods-as-shortcuts"></span>
    [Factory methods are shortcuts](02-meshes.md#factory-methods-as-shortcuts).

    <span id="64-bit-ids" data-guide-forward="../02-meshes/#plan-ids-and-derived-connectivity"></span>
    [64-bit ids](02-meshes.md#plan-ids-and-derived-connectivity).

    <span id="algorithms" data-guide-forward="../04-filters/#select-cells-cut-cells-or-extract-a-surface"></span>
    [Algorithms](04-filters.md#select-cells-cut-cells-or-extract-a-surface).

    <span id="moving-fields-between-spaces" data-guide-forward="../create-fields/#projection-is-an-algorithmic-choice"></span>
    [Moving fields between spaces](create-fields.md#projection-is-an-algorithmic-choice).

    <span id="writing-your-own-algorithm" data-guide-forward="../05-mesh-kernels/#separate-an-algorithm-from-storage-layout"></span>
    [Writing your own algorithm](05-mesh-kernels.md#separate-an-algorithm-from-storage-layout).

    <span id="polyhedra-polygons-and-lines" data-guide-forward="../18-polyhedra/#a-field-is-linear-on-the-cells-pieces"></span>
    [Polyhedra, polygons and lines](18-polyhedra.md#a-field-is-linear-on-the-cells-pieces).

    <span id="vtk-interop" data-guide-forward="../connect-simulation-data/#exchange-whole-datasets-with-vtk"></span>
    [VTK interop](connect-simulation-data.md#exchange-whole-datasets-with-vtk).

    <span id="level-zero" data-guide-forward="../connect-simulation-data/#level-zero"></span>
    [Level Zero](connect-simulation-data.md#level-zero).
