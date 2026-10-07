# Tack User's Guide

Learn how to turn a numerical calculation into a Tack application: allocate
fields, write a kernel, organize its stages, and check the result. Tack compiles
one supported kernel source for CPU, Metal, CUDA, HIP and Level Zero, within
[each backend's capabilities](08-backends.md#capabilities).

`tack-core` provides the language, fields and algorithms; `tack-vis` adds
scientific visualization; `tack-rendering` adds surface and volume rendering.
The guide describes the current checkout. See [Getting Started](01-getting-started.md)
for installation and the relationship to released APIs.

## A first reading path

Read [Getting Started](01-getting-started.md),
[From Python to a Tack program](12-execution-model.md), and
[Fields and Types](02-fields-and-types.md). Then run
[Heat diffusion](tutorials/heat.md): it is a complete program with a small
independent numerical check and a figure showing its output.

If you already write GPU kernels, focus on [Kernels](03-kernels.md),
[Memory and parallel correctness](13-memory-and-parallelism.md), and
[Backend capabilities](08-backends.md#capabilities).

## Language and program structure

| Chapter | What it explains |
|---|---|
| [Getting Started](01-getting-started.md) | Install and run the first kernel |
| [From Python to a Tack program](12-execution-model.md) | Host/device boundary, stages and compilation |
| [Fields and Types](02-fields-and-types.md) | Storage, NumPy exchange, scalar types and precision |
| [Vectors and matrices](14-vectors-and-matrices.md) | Component values, masks, products and fields |
| [Kernels](03-kernels.md) | Parallel ranges, constants, returns and specialization |
| [Control Flow and Math](04-control-flow.md) | Conditions, sequential loops, casts and numerical helpers |
| [Device Functions](05-device-functions.md) | Reusable compiled helpers |
| [Templates](06-templates.md) | Data-oriented objects and generic kernels |
| [Memory and parallel correctness](13-memory-and-parallelism.md) | Layout, buffers, aliases, gathering and scattering |

## Algorithms and applications

| Chapter | What it explains |
|---|---|
| [Advanced Features](07-advanced.md) | Private arrays, random state, atomics, shared memory and textures |
| [Reductions and Scans](11-reductions-and-scans.md) | Statistics, output offsets, sorting and block reductions |
| [Backends](08-backends.md) | Setup, capabilities and device differences |
| [Sharing arrays](16-interoperability.md) | NumPy copies, external pointers, DLPack and VTK |
| [Visualization](09-visualization.md) | Uniform grids, surfaces, normals and cell-to-point averaging |
| [Rendering](10-rendering.md) | Mesh scenes, cameras, materials, volumes and output |
| [Debugging and timing](15-debugging-and-timing.md) | Independent checks, numerical tolerance and warm measurements |

## Worked examples

The [tutorial gallery](tutorials/index.md) includes seven complete programs:
heat diffusion, coloured dye, trail-following agents, marching squares, MPM,
tiled N-body forces and isosurface rendering. Each page explains its data and
kernels, includes the complete script, shows output generated from that script,
and credits its origin. All sources are included in this documentation.

Use the [Contracts](../contracts/index.md) when you need precise guarantees,
and the [Design and Implementation](../design/index.md) pages when you need to
understand compilation or runtime internals.
