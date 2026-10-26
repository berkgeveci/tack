# Data Model

The classes and constructors of `tack.dataset`, and the interop with VTK,
MFEM and Warp, with their parameters, methods and properties. The
[Dataset API](dataset-api.md) reference explains how they fit together; the
algorithms that work on them are in [Algorithms](algorithms.md).

## Contents

- [Datasets and fields](#datasets-and-fields): `DataSet`, `Field`, the
  constructors and sources
- [Data in pieces and hierarchies](#data-in-pieces-and-hierarchies):
  `PartitionedDataSet`, `AMRDataSet`, `OctreeDataSet`
- [Topologies](#topologies): structured, unstructured, polyhedral, polygonal,
  polyline; cell shapes
- [Spaces](#spaces): where a field's values live
- [Sets, ghost marks and global ids](#sets-ghost-marks-and-global-ids)
- [Implicit functions](#implicit-functions)
- [Locating points](#locating-points): `CellLocator`, `PointLocator`
- [Steps in time](#steps-in-time): temporal filters and particle paths
- [Kernels over meshes](#kernels-over-meshes): `for_each`
- [Interoperability](#interoperability): VTK, MFEM, Warp

## Datasets and fields

::: tack.dataset.DataSet

::: tack.dataset.Field

### Making datasets

::: tack.dataset.uniform_grid

::: tack.dataset.rectilinear_grid

::: tack.dataset.point_cloud

### Sources

::: tack.dataset.sources
    options: {members: false}

::: tack.dataset.wavelet

::: tack.dataset.tangle

::: tack.dataset.amr_gaussian_pulse

## Data in pieces and hierarchies

::: tack.dataset.PartitionedDataSet

::: tack.dataset.AMRDataSet

::: tack.dataset.OctreeDataSet

## Topologies

::: tack.dataset.StructuredTopology

::: tack.dataset.UnstructuredTopology

::: tack.dataset.polyhedra
    options: {members: false}

::: tack.dataset.PolyhedralTopology

::: tack.dataset.PolygonalTopology

::: tack.dataset.PolylineTopology

::: tack.dataset.SizeBuckets

::: tack.dataset.orient

### Cell shapes

::: tack.dataset.shapes
    options: {members: false}

## Spaces

::: tack.dataset.spaces
    options: {members: false}

::: tack.dataset.Space

::: tack.dataset.H1

::: tack.dataset.L2

::: tack.dataset.Constant

::: tack.dataset.Values

::: tack.dataset.SideTraces

## Sets, ghost marks and global ids

::: tack.dataset.Set

::: tack.dataset.Ghosts

::: tack.dataset.GlobalIds

## Implicit functions

::: tack.dataset.implicit
    options: {members: false}

::: tack.dataset.Plane

::: tack.dataset.Sphere

::: tack.dataset.Cylinder

::: tack.dataset.Box

::: tack.dataset.Planes

::: tack.dataset.Union

::: tack.dataset.Intersection

::: tack.dataset.Difference

## Locating points

::: tack.dataset.CellLocator

::: tack.dataset.Locations

::: tack.dataset.PointLocator

## Steps in time

::: tack.dataset.RunningStatistics

::: tack.dataset.TimeDerivative

::: tack.dataset.Advection

::: tack.dataset.Pathlines

::: tack.dataset.Streaklines

::: tack.dataset.extend_paths

## Kernels over meshes

::: tack.dataset.for_each

## Interoperability

::: tack.interop.vtk

::: tack.interop.mfem

::: tack.interop.warp
