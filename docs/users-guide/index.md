# Tack User's Guide

Learn how to compute on scientific datasets, build visualization pipelines,
and write the kernels that extend them. Tack's mesh algorithms and your own
kernels run on CPU, Metal, CUDA, HIP and Level Zero, within
[each backend's capabilities](08-backends.md#capabilities).

`tack-core` provides the language, arrays and algorithms; `tack-vis` adds
datasets and scientific visualization.
The guide describes the current checkout. See [Getting Started](01-getting-started.md)
for installation and the relationship to released APIs.

## Start with kernels

1. **Run one kernel.** [Getting Started](01-getting-started.md) takes you from
   installation to a checked result on the CPU. Then choose your GPU backend.
2. **Understand a program.** [From Python to a Tack program](12-execution-model.md)
   explains which code runs in Python, which code is compiled, and how to order
   calculation stages.
3. **Build a simulation.** [Heat diffusion](tutorials/heat.md) combines arrays,
   a stencil, buffer swapping and an independent NumPy check in one complete script.

You can follow this path before reading the language chapters in detail. Keep
the [quick reference](quick-reference.md) nearby as you write your own program.

## Find the task you have

| I want to… | Start with |
|---|---|
| Turn a NumPy expression into a kernel | [One operation per element](12-execution-model.md#start-with-one-operation-per-element) |
| Allocate arrays and exchange data with NumPy | [Arrays and Types](02-arrays-and-types.md) |
| Store positions, colours or local matrices | [Vectors and Matrices](14-vectors-and-matrices.md) |
| Update a grid over many timesteps | [Heat diffusion](tutorials/heat.md) and [stable stencil inputs](13-memory-and-parallelism.md#neighbour-reads-need-a-stable-input) |
| Combine contributions from many particles | [Gather and scatter](13-memory-and-parallelism.md#gather-and-scatter), then [MPM](tutorials/mpm.md) |
| Filter items into a variable-size output | [Count, scan, write](tutorials/contours.md) |
| Organize several kernels around shared state | [Templates](06-templates.md) and [trail-following agents](tutorials/physarum.md) |
| Understand a discontinuous solution or a face flux | [Fields and Spaces](../vis-guide/03-fields-and-spaces.md) |
| Sample a mesh at measurement locations | [Probing](../vis-guide/06-pipelines.md#sample-at-physical-positions) |
| Share memory with another library | [Sharing arrays](16-interoperability.md) |
| Split work over several GPUs or MPI ranks | [Communicate between blocks](18-communication.md) |
| Investigate wrong results or slow calls | [Debugging and Timing](15-debugging-and-timing.md) |

If you already write GPU kernels, read [Kernels](03-kernels.md),
[Memory and Parallel Correctness](13-memory-and-parallelism.md), and
[Backend capabilities](08-backends.md#capabilities). The
[N-body tutorial](tutorials/nbody.md) shows workgroup cooperation with a portable CPU path.

## Language and program structure

| Chapter | What it explains |
|---|---|
| [Getting Started](01-getting-started.md) | Install and run the first kernel |
| [From Python to a Tack program](12-execution-model.md) | Host/device boundary, stages and compilation |
| [Arrays and Types](02-arrays-and-types.md) | Storage, NumPy exchange, scalar types and precision |
| [Vectors and matrices](14-vectors-and-matrices.md) | Component values, masks, products and arrays |
| [Kernels](03-kernels.md) | Parallel ranges, constants, returns and specialization |
| [Control Flow and Math](04-control-flow.md) | Conditions, sequential loops, casts and numerical helpers |
| [Device Functions](05-device-functions.md) | Reusable compiled helpers |
| [Templates](06-templates.md) | Device-class objects and generic kernels |
| [Memory and parallel correctness](13-memory-and-parallelism.md) | Layout, buffers, aliases, gathering and scattering |

## Kernel algorithms

| Chapter | What it explains |
|---|---|
| [Reductions and Scans](11-reductions-and-scans.md) | Statistics, output offsets, sorting and block reductions |
| [Advanced Features](07-advanced.md) | Private arrays, random state, atomics, shared memory and textures |

## Datasets and visualization

Meshes, point clouds, fields and the filters over them (`tack.dataset`, the
`tack-vis` package) have their own guide: the
[Visualization Guide](../vis-guide/index.md). You can use its algorithms
before writing a kernel; read the kernel chapters here when you want to
initialize data on the device, write a custom operation, or connect a
simulation's working arrays to a dataset.

## Run and diagnose

| Chapter | What it explains |
|---|---|
| [Backends](08-backends.md) | Setup, capabilities, queued launches and device differences |
| [Sharing arrays](16-interoperability.md) | NumPy copies, external pointers, DLPack and VTK |
| [Communicate between blocks](18-communication.md) | Work split over devices and MPI ranks: collectives, exchanges, ghost plans, routing and sorting |
| [Debugging and timing](15-debugging-and-timing.md) | Independent checks, bounds checks, numerical tolerance and warm measurements |
| [The Kernel Cache](17-kernel-cache.md) | Compiled kernels kept on disk, settings and MPI jobs |

## Worked examples

The [tutorial gallery](tutorials/index.md) includes six complete programs:
heat diffusion, coloured dye, trail-following agents, marching squares, MPM
and tiled N-body forces. Each page explains its data and
kernels, includes the complete script, shows output generated from that script,
and credits its origin. All sources are included in this documentation.

Use the [Contracts](../contracts/index.md) when you need precise guarantees,
and the [Design and Implementation](../design/index.md) pages when you need to
understand compilation or runtime internals.
