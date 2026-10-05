# Tack User's Guide

Tack (Portable GPU Compute) is a Python-first GPU compute framework. You write
kernels as decorated Python functions and Tack compiles them at runtime to run
on CPUs and GPUs across multiple backends.

Tack is split into three packages:
- **tack-core** — the compute framework (chapters 1-9)
- **tack-vis** — scientific visualization algorithms (chapter 10)
- **tack-rendering** — GPU path tracing renderer (chapter 11)

## Table of Contents

1. [Getting Started](01-getting-started.md) — Installation, first kernel, choosing a backend
2. [Fields and Types](02-fields-and-types.md) — Data containers, scalar types, numpy interop
3. [Kernels](03-kernels.md) — Writing kernels, parallel loops, scalar arguments
4. [Control Flow and Math](04-control-flow.md) — Loops, conditionals, math builtins
5. [Device Functions](05-device-functions.md) — `@tack.func` for reusable device-side code
6. [Templates](06-templates.md) — `@tack.data_oriented` classes for zero-cost abstraction
7. [Advanced Features](07-advanced.md) — Atomics, shared memory, local arrays, textures, vectors
8. [Reductions and Scans](11-reductions-and-scans.md) — Field reductions, statistics, prefix scans, block reductions
9. [Backends](08-backends.md) — CPU, Metal, CUDA, HIP, Level Zero
10. [Visualization](09-visualization.md) — Flying edges, normals, cell to point, VTK interop (tack-vis)
11. [Rendering](10-rendering.md) — Path tracing, BVH, camera, scene (tack-rendering)
