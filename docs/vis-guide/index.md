# Visualization Guide

This guide is about `tack.dataset` (the `tack-vis` package), and this page is
its background: what scientific visualization does with data, the two toolkits
`tack.dataset` grew from -- VTK and Viskores -- and what it does differently.
If you know VTK, skim the first sections and read [What Tack keeps, and what
it changes](#what-tack-keeps-and-what-it-changes).

## What visualization does

A simulation or an instrument produces numbers attached to places: the
temperature at the nodes of a mesh, the velocity in each cell, the
positions and masses of particles, the returns of a LiDAR scan. Scientific
visualization turns those numbers into something a person can understand
and check, and most of that work happens *before* anything is drawn:

- **Extract** the part that matters: the surface where a value crosses a
  threshold (a *contour* or isosurface), a cut through the volume (a
  *slice*), the cells inside a region (*clip*, *threshold*), the outer
  surface of a solid.
- **Derive** new quantities: gradients, vorticity, cell volumes and quality,
  the values of one field where another field's cells are.
- **Sample and follow**: values at probe points or along a line, paths of
  particles carried by a flow (*streamlines*, *pathlines*).
- **Reduce**: statistics, histograms, integrals, connected regions, halos
  in a cosmology run.

Each of these is an *algorithm* (VTK calls them *filters*) that takes a
dataset and gives a new one, so they chain into a *pipeline*: read or
receive data, filter it, then render, save or hand it on. A pipeline that
runs inside the simulation, on its memory and its devices, while it runs,
is *in situ*, which is where the data is largest and copying it is most
expensive.

Everything rests on a **data model**: how a dataset says where its points
are, how they join into cells, and what its values mean. A value at a mesh
node is interpolated across the cells around it; a value per cell is
constant in that cell; a finite-element solution of order 3 is a polynomial
inside each cell. An algorithm that gets that meaning wrong draws a
plausible picture of the wrong thing, so the data model comes first in this
guide.

## VTK: the data model and the filters

[VTK](https://vtk.org), the Visualization Toolkit, started in the 1990s and
is developed by Kitware with a large community; ParaView, VisIt and many
domain applications are built on it. Its data model is the common language
of scientific visualization:

- **Dataset types** by how much structure they have: image data (a uniform
  grid, described by an origin and spacing), rectilinear grids (one array
  of coordinates per axis), structured (curvilinear) grids, unstructured
  grids of cells of any type, and polygonal data.
- **Cell types** with a fixed corner order and *parametric coordinates*:
  vertices, lines, triangles, quads, tetrahedra, hexahedra, wedges,
  pyramids, polygons, polyhedra, and their quadratic and arbitrary-order
  (Lagrange and Bézier) forms.
- **Attributes** attached to points or to cells, and, more recently, cell
  grids, whose fields carry their own basis functions.
- Datasets in **pieces** with ghost cells, AMR hierarchies of refined boxes,
  and hyper tree grids (octrees).
- Hundreds of **filters**, and a demand-driven pipeline that runs them.

## Viskores: the same ideas, data-parallel

[Viskores](https://github.com/Viskores/viskores) (until 2025 VTK-m) re-thought
VTK's algorithms for many-core processors: GPUs and multi-core CPUs, where
an algorithm has to be written as many independent pieces of work rather
than a loop over cells. Its building blocks are:

- **Worklets**: small functions, one per cell or per point, that the
  library launches across the data on a device.
- **Data-parallel primitives** -- scans, sorts, reductions by key -- in place
  of the serial bookkeeping a single-threaded filter does as it goes (a
  contour, for instance, counts its output per cell, scans the counts, then
  writes each cell's triangles where the scan says).
- **Cell sets and coordinate systems** in which a uniform grid's points are
  computed rather than stored, and **implicit arrays** (counting, constant,
  Cartesian products) that save memory a GPU does not have.
- **Device adapters** that run one C++ source, through templates, on CUDA,
  Kokkos (for HIP and SYCL), OpenMP, TBB or a serial CPU.

## What Tack keeps, and what it changes

`tack.dataset` (the `tack-vis` package) is VTK's data model and filters,
written the way Viskores writes them, in Python, on Tack's kernels.

**From VTK** it keeps the vocabulary, so data and results move between the
two without translation:

- the dataset kinds: `uniform_grid` is VTK's image data, `rectilinear_grid`
  its rectilinear grid, a `StructuredTopology` with explicit positions its
  structured grid, `UnstructuredTopology` its unstructured grid;
- the cell shapes, their corner orders and parametric coordinates;
- the filters' names and meaning -- `contour`, `slice`, `clip`,
  `threshold`, `extract_surface`, `probe`, `streamlines` -- with tests that
  hold them to VTK's output;
- ghost marks in VTK's `vtkGhostType` bits, global ids, partitioned
  datasets, AMR (`vtkOverlappingAMR`) and octrees (`vtkHyperTreeGrid`);
- interop both ways, sharing memory rather than copying where it can
  ([VTK interop](connect-simulation-data.md#exchange-whole-datasets-with-vtk)), so VTK, ParaView and their
  renderers draw what Tack computes. Tack has no renderer of its own.

**From Viskores** it keeps the execution model:

- the algorithms are data-parallel, run on the dataset's device and are
  built from Tack's scans, sorts and segmented reductions, mostly without
  atomics, so their results don't depend on how threads were scheduled;
- structured grids and their coordinates stay implicit, and `tack.arrays`
  has implicit arrays as Viskores does (constant, counting, Cartesian
  products, uniform coordinates, permutations, functions of the index);
- cell location by a two-level grid of bins, as Viskores'
  `CellLocatorTwoLevel`, and its halo finder's pruning;
- the [Algorithms reference](reference/algorithms.md) names each
  algorithm's VTK and Viskores counterparts.

**What it changes:**

- **Algorithms are Python.** A filter is a few Tack kernels, compiled at
  run time for the CPU, Metal, CUDA, HIP or Level Zero. There is no C++
  template layer, and nothing to rebuild for a new device.
- **Your kernels see a mesh as the built-in ones do.** The views a filter
  uses -- cells, faces, a field's values at a cell's corners -- are public,
  so a filter of your own is as fast and as correct as a built-in
  ([Write Mesh Algorithms](05-mesh-kernels.md), [Write a Filter](15-write-a-filter.md)).
- **Fields have a space, not just an association.** Instead of "point data"
  and "cell data", a field is values in a *function space*: `H1` (shared at
  nodes, continuous), `L2` (each cell's own, discontinuous), `Constant`,
  of any order and basis. This is the finite-element view of MFEM and of
  VTK's cell grids, and it is what lets one model hold a linear mesh, a
  discontinuous Galerkin solution and a curved order-4 geometry
  ([Function Spaces and Higher-order Geometry](00-function-spaces.md)).
- **Positions keep their precision.** A dataset's points are an origin plus
  values, so coordinates far from zero stay accurate in `f32` on a GPU.
- **Units and time are part of the data**: fields carry physical units
  through filters, and a dataset carries its time step.
- **Pieces communicate through Tack.** Data in pieces, on several devices
  or MPI ranks, is handled by `tack.communication`, the same layer any
  Tack program can use ([Communicate Between Blocks](../users-guide/18-communication.md)).

## How this guide is organized

1. **The Data Model** starts with [one small dataset](01-datasets.md), then
   separates [topology and geometry](topology-and-geometry.md),
   [field meaning](03-fields-and-spaces.md), and [coordinates, units and time](12-units.md).
   Representation chapters cover polyhedra, point clouds, pieces, AMR and octrees.
2. **Work with Data** gives complete tasks: build and connect data, filter,
   sample, trace, balance face fluxes, process particles and stream steps.
3. **Further Detail and Reference** serve advanced readers: function-space
   mathematics, arbitrary-order geometry, exact layouts, interop mappings,
   units, public APIs and implementation design.
4. **Data in Pieces** develops ghost ownership and distributed algorithms.
   **Examples and galleries** provide runnable programs and pictures.

The [Introduction](introduction.md) offers a quick start and routes for both
new readers and experienced solver or kernel authors.
