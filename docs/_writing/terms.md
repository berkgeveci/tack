# Tack documentation: terms

One word for one thing. The first column is the word pages use; "VTK" and
"Viskores" give the counterparts a visualization expert knows. This list seeds
the site's glossary (`reference/glossary.md`).

## Core

| term | means | VTK / Viskores / other |
|---|---|---|
| kernel | a Python function decorated with `@tack.kernel`, compiled for a device and launched over a range | a CUDA kernel; a Viskores worklet |
| device function | `@tack.func`, inlined into kernels | a `__device__` function |
| device class | `@tack.device_class`, an object kernels can take, its methods inlined | a Viskores ExecObject |
| device | a backend made a value: `tack.device("cuda:0")` | a Viskores device adapter |
| backend | the code generator and runtime for one kind of device: CPU, Metal, CUDA, HIP, Level Zero | |
| array | `tack.Array`, memory on one device with a shape and element type | `vtkDataArray`; `viskores::cont::ArrayHandle` |
| implicit array | an array computed from others or from a formula, not stored (`tack.arrays`) | `vtkImplicitArray`; fancy array handles |
| launch | one call of a kernel over its range | a dispatch |
| variant | one compiled form of a kernel, for one set of argument types and shapes | |

## Communication

| term | means | counterpart |
|---|---|---|
| block | one unit of distributed work, on one rank and one device | a DIY block |
| world | the set of processes: one process, or an MPI communicator | `vtkMultiProcessController` |
| rank | one MPI process | |
| plan | precomputed sends and receives, run each step (ghost refresh) | |
| routing | sending points to the blocks that hold them | |

## Data

| term | means | counterpart |
|---|---|---|
| dataset | `DataSet`: a topology, a geometry and fields | `vtkDataSet` |
| topology | how points join into cells; no positions | the cells of a `vtkUnstructuredGrid`, an image's extent |
| geometry | the field giving positions | `vtkPoints`, an image's origin and spacing |
| point | a mesh node | point |
| cell | an element of the mesh | cell |
| field | values with a meaning: a space, an array, a unit | a data array plus its association |
| space | where a field's values live and how they define a function | point/cell association; a cell attribute's basis |
| point data | a field with one value per point, linear in each cell; Tack: an `H1` field of order 1 | point data |
| cell data | a field with one value per cell; Tack: a `Constant` field | cell data |
| `H1` | continuous fields, shared between cells, of any order | Lagrange point data; MFEM's H1 |
| `L2` | discontinuous fields, per cell, of any order | DG; MFEM's L2 |
| higher order | fields or geometry above order 1, polynomials inside cells | Lagrange and Bézier cells; cell grids |
| set | a selection of points, cells, faces or edges | a selection; a set of ids |
| pieces | the parts of a dataset split over blocks: `PartitionedDataSet` | `vtkPartitionedDataSet` |
| ghost marks | per point and cell bits saying a copy is owned elsewhere or hidden | `vtkGhostType` |
| ghost cells | cells copied from a neighbouring piece so filters see across | ghost cells |
| global ids | numbers for points and cells unique over all pieces | `GlobalIds` |
| AMR dataset | levels of refined boxes | `vtkOverlappingAMR` |
| octree | tree-based AMR | `vtkHyperTreeGrid` |
| origin | the point positions are stored relative to | |
| unit | a field's physical unit | `UNITS_LABEL` |

## Visualization

| term | means | counterpart |
|---|---|---|
| algorithm | a function that takes datasets and returns a dataset or values; the user-facing word is *filter* when it returns a dataset | filter |
| filter | an algorithm that returns a new dataset | filter |
| view | how a kernel sees a dataset's cells or fields | a Viskores ExecObject / cell set view |
| carry | moving fields, marks and ids from an input to a filter's output | `vtkDataSetAttributes::CopyData`, `InterpolateAllocate` |
| locator | a search structure for finding cells or points | `vtkCellLocator`, `vtkStaticPointLocator` |
| streamline | a path through a steady velocity field | `vtkStreamTracer` |
| pathline | a particle's path through a field that changes in time | `vtkParticlePathFilter` |
| streakline | the particles released from one point over time | `vtkStreaklineFilter` |
