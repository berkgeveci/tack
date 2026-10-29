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
- [Filters of your own](#filters-of-your-own): `carry` and its maps, ghost
  marks, hidden cells, owned values
- [Units](#units): parameters as quantities, units of what a filter computes
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

::: tack.dataset.bases
    options: {members: [FAMILIES, BERNSTEIN, nodes, weights, bernstein, dof_tables, tensor_shape]}

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

## Filters of your own

What the built-in filters are written with. A filter that makes a new
dataset states how its output's points, cells, faces and edges come from
the input's, and `carry` takes the fields, sets, ghost marks, global ids,
origin, unit and time along by those maps. A filter that builds its
`DataSet` some other way passes its input's marks on with `keep_marks` (the
same cells and points) or records their loss with `drop_marks`, so that
reductions over pieces refuse what they would count twice. Filters that
make cells start from `visible(data)`, which leaves out hidden cells, and a
reduction over a piece takes its `owned_values`. A selection made by a
kernel becomes a set with `Set.from_flags`, and `extract_cells` keeps it.

::: tack.dataset.carry
    options: {members: false}

::: tack.dataset.carry.carry
    options: {show_root_full_path: false}

::: tack.dataset.Same

::: tack.dataset.Take

::: tack.dataset.Pieces

::: tack.dataset.Interpolate

::: tack.dataset.keep_marks

::: tack.dataset.drop_marks

::: tack.dataset.visible

::: tack.dataset.owned_values

## Units

A filter's parameters take quantities (`"300 K"`, `(1, "cm")`) where the
built-ins do: `units.magnitude(value, field.unit, "the bounds")` gives a
number in a field's unit, `units.magnitude(value, data.unit, "the radius")`
one in the spatial unit, and a plain number passes through. A field a filter
computes gets the unit of what it is: `units.delta` for a difference (a
difference of degrees Celsius is in kelvin), `units.derived` for a product
of powers, `units.named` for the SI name. Unknown in, unknown out: `None`,
never a guess.

::: tack.dataset.units

## Interoperability

::: tack.interop.vtk

::: tack.interop.vtk.algorithms

::: tack.interop.mfem

::: tack.interop.warp
