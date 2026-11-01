# Data model overview

Tack represents scientific data from meshes, refined grids and particles.
The representation describes where values live; their numerical meaning
describes how to interpret and reconstruct them. The same mesh can carry
fields from different numerical methods.

This page is a map of the model. Continue with [Understand a Dataset](01-datasets.md)
for a small working example, then the chapters that develop each part.

## What kinds of data can Tack represent?

| Representation | What it describes | Learn more |
|---|---|---|
| Uniform and rectilinear grids | A logical lattice, with constant spacing or one coordinate array per axis; connectivity and coordinates can stay implicit | [Topology and Geometry](topology-and-geometry.md) |
| Curvilinear structured grids | A logical lattice whose corner positions bend the physical cells | [Structured geometry](topology-and-geometry.md#image-rectilinear-and-curvilinear-data) |
| Unstructured meshes | Explicit connectivity, including mixtures of triangles, quads, tetrahedra, hexahedra, wedges and other supported shapes | [Mesh representations](topology-and-geometry.md#choose-a-representation) |
| Polyhedra, polygons and paths | General cells described by face rings, point rings or point chains | [Their representation](18-polyhedra.md) |
| Point clouds and particles | Positions with per-point quantities, without a surface or volume mesh connecting them | [Point Clouds and Particles](point-cloud-model.md) |
| Block-structured adaptive mesh refinement (AMR) | Levels of structured boxes, with finer boxes superseding covered coarse regions | [AMR Hierarchies](10-amr.md) |
| Octrees | A root grid and trees of refined cells; leaves describe the finest available regions | [Octrees](11-octrees.md) |

Geometry and solution detail are independent. Supported shaped meshes can
have higher-order curved geometry and fields of a different order. A
curvilinear grid's bent corner layout and a higher-order coordinate mapping
are different choices.

Data can also be held in [pieces](collections-and-partitions.md), placed on
devices or distributed over processes using MPI (Message Passing Interface).
Partitioning describes ownership and placement; it is another axis of the model, rather than a numerical method.

## How numerical methods fit

The method names describe how a simulation approximates physical fields and
its governing equations:

- **FE — finite element methods:** represent a field inside each mesh cell
  (an *element*) using basis functions and coefficients. For example, nodal
  temperatures determine a polynomial temperature within an element.
- **DG — discontinuous Galerkin:** a family of finite element methods in which
  each cell has independent coefficients. Values may jump across a shared
  face; the numerical method couples the cells through their interfaces.
- **FV — finite volume methods:** express conservation as a balance over
  control volumes. They commonly store cell averages and use fluxes through
  faces to describe what enters and leaves each volume.
- **SPH — smoothed particle hydrodynamics:** a particle method that estimates
  fields using weighted sums over nearby particles. A smoothing kernel and
  its smoothing length determine how each particle contributes.
- **FD — finite difference methods:** approximate derivatives using combinations
  of values at neighboring sample locations, often on a structured grid.
  The chosen neighbors and weights form a *stencil*.

Here support means representing a method's quantities and providing analysis
and visualization operations on them. Each operation still has its own
requirements on shape, geometry and interfaces.

| Representation | [FE / DG](00-function-spaces.md) | [FV](18-finite-volume.md) | [SPH](08-point-clouds.md#sph-particles) | FD |
|---|---|---|---|---|
| Structured grids | H1/L2 fields | Cell and face fields | Sampling target | Grid samples today; native support planned |
| Unstructured shaped meshes | H1/L2 fields | Cell and face fields | Sampling target | No native FD operators |
| Polyhedra and polygons | Linear-on-pieces point fields | Cell and face fields | Sampling target | No native FD operators |
| Block AMR | Fields on each box | Cell data and per-box face fields; interface limits | Individual boxes as targets | Grid samples on each box |
| Octrees | No native FE coefficient space | Leaf cell data; interface limits | Leaf mesh as a target | No native FD operators |
| Particles | Point attributes, without an FE mesh | No control-volume mesh | Particle fields and kernel sums | No native FD operators |

**FE and DG data in Tack** retain a basis and coefficients: shared nodal values
for H1, or independent cell-local values for L2. Higher orders and curved
geometry are supported on the shapes listed in [Cell Types and Bases](16-cells-and-bases.md).
Polyhedra and polygons instead use a piecewise-linear reconstruction; they
do not supply the general higher-order FE spaces.

**FV data in Tack** includes cell averages, normal components and integrated face
fluxes. Keeping their locations, orientation and meaning supports balances
and reconstruction. The [finite-volume workflow](18-finite-volume.md) shows
these operations on shaped and polyhedral cells.

**SPH data in Tack** lives on particles, with quantities such as mass, density and
smoothing length. Tack computes density and kernel-weighted interpolation.
“Sampling target” in the table means evaluating particle quantities at a
mesh's points; it does not turn that mesh into a particle discretization.
Use individual AMR boxes or the octree's leaf-mesh adapter as targets.

**FD output in Tack** can already be retained as samples on a structured lattice and
visualized with a chosen reconstruction such as trilinear interpolation.
Explicit FD sampling/staggering support and stencil operators are planned.
Differentiating the visualization reconstruction need not reproduce the
solver's difference operator.

For refined data, local representation and cross-interface operations need
separate consideration. AMR boxes hold ordinary datasets, while octrees
currently hold cell values on their nodes. Coarse/fine faces are not yet
paired for general face-based operations, and AMR ghost generation does not
yet carry arbitrary higher-order fields. The representation chapters explain
the current limits.

## These choices can be combined

A structured mesh can carry a continuous temperature, a cell material and
face-normal velocity together. Particle quantities can be sampled onto that
mesh. Refinement or partitioning changes how the domain is organized without
making all its fields use one numerical interpretation.

Start with [Understand a Dataset](01-datasets.md) to see topology, geometry,
fields and sets in one example. Then read [Topology and Geometry](topology-and-geometry.md),
[Fields and Spaces](03-fields-and-spaces.md), and
[Coordinates, Units and Time](12-units.md), or follow the representation and
method links above to your own data.
