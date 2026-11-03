# Glossary

This page defines every term the documentation uses, once. It is for readers
who meet a word they do not know, and for readers who know VTK, ParaView or
Viskores and want the word they already use. Each entry gives a definition,
the counterparts in VTK, ParaView and Viskores, and the page that explains the
term in full. The terms are grouped by Tack's four layers and listed
alphabetically within each.

Two words mean something different in ParaView. A *block* here is a unit of
distributed work, not one block of a multiblock dataset; that is a *piece*. A
*view* here is how a kernel sees a dataset, not a render view.

A dash means there is no direct counterpart.

## Core

### array

A `tack.Array`: memory on one device, with a shape and an element type. The
elements are scalars, small vectors or small matrices.

*VTK:* `vtkDataArray` · *ParaView:* an array · *Viskores:* `ArrayHandle`

Explained in [arrays and types](../core/arrays-and-types.md).

### atomic

An operation (`tack.atomic_add`, `atomic_min`, `atomic_max`) that updates one
array element so that threads writing it at the same time lose no update. It
returns the element's old value, which lets a thread claim a slot.

*VTK:* `std::atomic` in `vtkSMPTools` code · *ParaView:* — · *Viskores:* `AtomicArray`

Explained in [workgroups, atomics, textures and random numbers](../core/workgroups-atomics-textures.md).

### backend

The code generator and runtime for one kind of device. Tack has five: the CPU,
Metal, CUDA, HIP and Level Zero, and NVIDIA Warp as a prototype.

*VTK:* a `vtkSMPTools` backend · *ParaView:* — · *Viskores:* a device adapter

Explained in [choose a device](../core/choose-a-device.md). The
[backends reference](backends.md) lists what each supports.

### device

A backend made a value, such as `tack.device("cuda:0")`. It is where arrays
live and kernels run. Every array records its device, and a kernel runs where
its arrays are.

*VTK:* — · *ParaView:* — · *Viskores:* a device adapter tag and device id

Explained in [choose a device](../core/choose-a-device.md).

### device class

A class decorated with `@tack.device_class`, whose objects a kernel takes as
arguments. Its methods are compiled into the kernel, and its arrays and numbers
become the kernel's arguments.

*VTK:* — · *ParaView:* — · *Viskores:* an execution object (`ExecutionObjectBase`)

Explained in [device classes](../core/device-classes.md).

### device function

A function decorated with `@tack.func`, which kernels call. It is inlined into
each kernel that calls it.

*VTK:* — · *ParaView:* — · *Viskores:* a `VISKORES_EXEC` function · *CUDA:* a `__device__` function

Explained in [device functions](../core/device-functions.md).

### implicit array

An array whose values are computed from its index or from other arrays, not
stored: a constant, a count, uniform coordinates, a permutation. The module
`tack.arrays` holds them, and kernels index them like any array.

*VTK:* `vtkImplicitArray` (`vtkConstantArray`, `vtkAffineArray`) · *ParaView:* — ·
*Viskores:* fancy array handles (`ArrayHandleConstant`, `ArrayHandleCounting`,
`ArrayHandleUniformPointCoordinates`)

Explained in [arrays and types](../core/arrays-and-types.md).

### kernel

A Python function decorated with `@tack.kernel`, with one parallel loop. Tack
compiles it for a device when it is first called, and each iteration of the
loop runs as its own thread.

*VTK:* — · *ParaView:* — · *Viskores:* a worklet · *CUDA:* a kernel

Explained in [kernels](../core/kernels.md).

### launch

One call of a kernel over its loop's range. On a GPU a launch is queued and the
call returns before it finishes. Reading an array or calling `tack.sync()`
waits for it.

*VTK:* — · *ParaView:* — · *Viskores:* invoking a worklet · *CUDA:* a kernel launch

Explained in [how a Tack program runs](../core/how-programs-run.md).

### variant

One compiled form of a kernel, for one set of argument types and, where the
kernel's code depends on them, array shapes. A call with a new combination
compiles a new variant. Later calls reuse it.

*VTK:* — · *ParaView:* — · *Viskores:* a template instantiation, made when the C++ is built

Explained in [how a Tack program runs](../core/how-programs-run.md).

### workgroup

The threads of a GPU launch that share fast on-chip memory and wait for each
other at barriers. The CPU has no workgroups.

*VTK:* — · *ParaView:* — · *Viskores:* — · *CUDA:* a thread block

Explained in [workgroups, atomics, textures and random numbers](../core/workgroups-atomics-textures.md).

## Communication

### block

One unit of distributed work, on one rank and one device, such as one piece of
a dataset. Blocks are numbered over all ranks by a global id, which is not a
dataset's [global ids](#global-ids).

*VTK:* — · *ParaView:* the part of the data one server process holds (not a
multiblock's block) · *Viskores:* a DIY block · *DIY:* a block

Explained in [blocks and worlds](../communication/blocks-and-worlds.md).

### plan

A `Plan`: the sends and receives between blocks, worked out once and run every
step. Refreshing ghost values after each time step is the usual use.

*VTK:* — · *ParaView:* — · *Viskores:* —

Explained in [exchange and plans](../communication/exchange-and-plans.md).

### rank

One MPI process. A rank holds one or more blocks.

*VTK:* a process of a `vtkMultiProcessController` · *ParaView:* one `pvserver`
process · *Viskores:* an MPI rank

Explained in [blocks and worlds](../communication/blocks-and-worlds.md).

### routing

Sending points to the blocks that hold them, with values that travel with the
points, and answering them in their order. `Routing` does it, for example to
probe a dataset in pieces.

*VTK:* — · *ParaView:* — · *Viskores:* —

Explained in [route points to blocks](../communication/routing.md).

### world

The processes that work together: `World()` for one process, or
`World(comm)` for the ranks of an MPI communicator. The same code runs in
either.

*VTK:* `vtkMultiProcessController` · *ParaView:* the server's processes ·
*Viskores:* the communicator `EnvironmentTracker` holds

Explained in [blocks and worlds](../communication/blocks-and-worlds.md).

## Data

### AMR dataset

An `AMRDataSet`: levels of structured boxes, each level finer than the one
before it, with finer boxes covering parts of coarser ones. It is block-structured
adaptive mesh refinement.

*VTK:* `vtkOverlappingAMR` · *ParaView:* an AMR dataset · *Viskores:* —

Explained in [AMR hierarchies and octrees](../data/amr-and-octrees.md).

### cell

One element of a mesh, such as a tetrahedron, a hexahedron or a polyhedron.
Its corners are points.

*VTK:* a cell · *ParaView:* a cell · *Viskores:* a cell (its shape a `CellShapeTag`)

Explained in [cells and meshes](../data/cells-and-meshes.md).

### cell data

A field with one value per cell, constant inside each cell. In Tack it is a
[`Constant`](#constant) field.

*VTK:* cell data · *ParaView:* cell data · *Viskores:* a field with cell association

Explained in [fields and where values live](../data/fields.md).

### `Constant`

The space of fields with one value per cell, constant inside the cell. A
`Constant` field is cell data.

*VTK:* cell data · *ParaView:* cell data · *Viskores:* cell association

Explained in [fields and where values live](../data/fields.md).

### dataset

A `DataSet`: a topology, a geometry and fields. An image, an unstructured
mesh and a point cloud are all datasets.

*VTK:* `vtkDataSet` · *ParaView:* the output of a source or filter ·
*Viskores:* `viskores::cont::DataSet`

Explained in [the data layer's first page](../data/index.md).

### field

A `Field`: values with a meaning. It holds an array of values, the space that
says how they define a function, and a physical unit.

*VTK:* a data array with its association (point or cell) · *ParaView:* an
array · *Viskores:* `viskores::cont::Field`

Explained in [fields and where values live](../data/fields.md).

### geometry

The field that gives a dataset's positions. On an image it is computed from an
origin and a spacing; on an unstructured mesh it is stored.

*VTK:* `vtkPoints`, or an image's origin and spacing · *ParaView:* the points ·
*Viskores:* a `CoordinateSystem`

Explained in [topology and geometry](../data/topology-and-geometry.md).

### ghost cells

Cells copied from neighbouring pieces into a piece, so that filters on the
piece see across its boundary. Ghost marks say they belong to another piece.

*VTK:* ghost cells (`vtkGhostCellsGenerator`) · *ParaView:* the Ghost Cells
filter · *Viskores:* ghost cells

Explained in [data in pieces](../data/pieces.md).

### ghost marks

Bits for each point and each cell that say a copy is owned by another piece, or
that a cell or point is hidden. Tack uses VTK's bits, so marks pass both ways
unchanged.

*VTK:* `vtkGhostType` · *ParaView:* `vtkGhostType` · *Viskores:* the ghost cell field

Explained in [data in pieces](../data/pieces.md).

### global ids

Numbers for points and cells that are unique over all the pieces of a dataset.
Copies of one point in two pieces have the same global id.

*VTK:* `GlobalIds` (`vtkGenerateGlobalIds`) · *ParaView:* the Global Point And
Cell Ids filter · *Viskores:* —

Explained in [data in pieces](../data/pieces.md).

### `H1`

The space of continuous fields, whose values at shared nodes belong to every
cell around them. It has any order. At order 1 an `H1` field is point data.

*VTK:* point data; the point data of Lagrange cells · *ParaView:* point data ·
*Viskores:* point association · *MFEM:* `H1_FECollection`

Explained in [fields and where values live](../data/fields.md) and
[higher-order and finite-element data](../data/higher-order.md).

### higher order

Fields or geometry that are polynomials of order 2 or more inside each cell. A
geometry of higher order makes curved cells.

*VTK:* Lagrange and Bézier cells; cell grids · *ParaView:* the same ·
*Viskores:* —

Explained in [higher-order and finite-element data](../data/higher-order.md).

### `L2`

The space of discontinuous fields. Each cell holds its own values, of any
order, so values may jump across a face. Discontinuous Galerkin (DG) solutions
are `L2` fields.

*VTK:* a discontinuous cell-grid attribute · *ParaView:* the same ·
*Viskores:* — · *MFEM:* `L2_FECollection`

Explained in [fields and where values live](../data/fields.md) and
[higher-order and finite-element data](../data/higher-order.md).

### octree

An `OctreeDataSet`: adaptive refinement by trees, in which each refined cell
splits into eight children (four in 2D), down to leaves of different sizes.

*VTK:* `vtkHyperTreeGrid` · *ParaView:* a hyper tree grid · *Viskores:* —

Explained in [AMR hierarchies and octrees](../data/amr-and-octrees.md).

### origin

The point a dataset's positions are stored relative to. It is three
double-precision numbers, so positions far from zero stay accurate when they
are stored in single precision.

*VTK:* — · *ParaView:* — · *Viskores:* —

Explained in [coordinates, units and time](../data/units-coordinates-time.md).

### piece

One part of a dataset split for distribution over devices or ranks. A
`PartitionedDataSet` holds a rank's pieces.

*VTK:* a partition of a `vtkPartitionedDataSet` · *ParaView:* a partition ·
*Viskores:* a partition of a `PartitionedDataSet`

Explained in [data in pieces](../data/pieces.md).

### point

A node of a mesh, the corner of one or more cells. Point clouds have points and
no cells.

*VTK:* a point · *ParaView:* a point · *Viskores:* a point

Explained in [cells and meshes](../data/cells-and-meshes.md).

### point data

A field with one value per point, interpolated linearly inside each cell. In
Tack it is an [`H1`](#h1) field of order 1.

*VTK:* point data · *ParaView:* point data · *Viskores:* a field with point association

Explained in [fields and where values live](../data/fields.md).

### set

A `Set`: a selection of points, cells, faces or edges, made from their ids or
from flags. A dataset can keep named sets.

*VTK:* a `vtkSelection`, or a list of ids · *ParaView:* a selection ·
*Viskores:* —

Explained in [build meshes and fields](../data/build-meshes-and-fields.md).

### space

Where a field's values live and how they define a function inside each cell.
`H1`, `L2` and `Constant` are spaces.

*VTK:* a field's association (point or cell); the basis of a cell-grid
attribute · *ParaView:* the association · *Viskores:* a field's association

Explained in [fields and where values live](../data/fields.md).

### topology

How points join into cells, without positions. A structured topology is a grid
of cells numbered by their index; an unstructured one stores each cell's
points.

*VTK:* the cells of a `vtkUnstructuredGrid`, or an image's extent ·
*ParaView:* — · *Viskores:* a `CellSet`

Explained in [topology and geometry](../data/topology-and-geometry.md).

### unit

A field's physical unit, such as `"m/s"`. A dataset's spatial unit is its
geometry's. Filters carry units to their outputs and derive the units of new
quantities.

*VTK:* the `UNITS_LABEL` of an array's information · *ParaView:* — ·
*Viskores:* —

Explained in [coordinates, units and time](../data/units-coordinates-time.md).

## Visualization

### algorithm

A function that takes a dataset first and returns something new: a dataset, a
field or values. It never changes its input. An algorithm that returns a
dataset is a [filter](#filter).

*VTK:* a filter (`vtkAlgorithm`) · *ParaView:* a filter · *Viskores:* a filter

Explained in [the visualization layer's first page](../visualization/index.md).
The [algorithms reference](../vis-guide/reference/algorithms.md) lists them all.

### carry

Moving fields, ghost marks and global ids from a filter's input to its output.
`td.carry` does it from a description of where each output point and cell
comes from.

*VTK:* `vtkDataSetAttributes::CopyData` and `InterpolateAllocate` ·
*ParaView:* — · *Viskores:* a filter's mapping of fields onto its output

Explained in [write a filter](../visualization/write-a-filter.md).

### filter

An algorithm that returns a new dataset, such as `contour` or `clip`.

*VTK:* a filter · *ParaView:* a filter · *Viskores:* `viskores::filter::Filter`

Explained in [the visualization layer's first page](../visualization/index.md).

### in situ

Running analysis inside a simulation while it runs, on its memory and its
devices, instead of on files it writes.

*VTK:* in situ through Catalyst · *ParaView:* Catalyst · *Viskores:* in situ through Ascent

Explained in [connect a simulation's data](../data/connect-simulation-data.md).

### locator

A search structure that finds the cell holding a point (`CellLocator`) or the
points near a point (`PointLocator`). A topology keeps the locator it builds,
so later searches reuse it.

*VTK:* `vtkCellLocator`, `vtkStaticCellLocator`, `vtkStaticPointLocator` ·
*ParaView:* — · *Viskores:* `CellLocatorTwoLevel`, `PointLocatorSparseGrid`

Explained in [sample and trace](../visualization/sample-and-trace.md).

### pathline

The path of one particle through a velocity field that changes in time.
`Pathlines` follows particles over a dataset's steps.

*VTK:* `vtkParticlePathFilter` · *ParaView:* the Particle Path filter ·
*Viskores:* the `Pathline` filter

Explained in [steps over time](../visualization/over-time.md).

### streakline

The line made by all the particles released from one point over time, as dye
released into a flow. `Streaklines` makes them.

*VTK:* `vtkStreaklineFilter` · *ParaView:* the Streak Line filter ·
*Viskores:* —

Explained in [steps over time](../visualization/over-time.md).

### streamline

A path through a velocity field that does not change in time. `td.streamlines`
traces them from seed points.

*VTK:* `vtkStreamTracer` · *ParaView:* the Stream Tracer filter ·
*Viskores:* the `Streamline` filter

Explained in [sample and trace](../visualization/sample-and-trace.md).

### view

How a kernel sees a dataset's cells, faces, points or fields. A view is a
device class that holds a dataset's arrays and answers in mesh terms: a cell's
points, their positions, a field's value inside a cell. It is not a render
view.

*VTK:* a `vtkCell`, and `vtkDataSet` methods such as `GetCellPoints` ·
*ParaView:* — · *Viskores:* the arguments of a `WorkletVisitCellsWithPoints`

Explained in [views](../visualization/views.md).
