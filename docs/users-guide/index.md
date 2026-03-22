# PGC User's Guide

PGC (Portable GPU Compute) is a Python-first GPU compute framework. You write
kernels as decorated Python functions and PGC compiles them at runtime to run
on CPUs and GPUs across multiple backends.

PGC is split into three packages:
- **pgc-core** — the compute framework (chapters 1-8)
- **pgc-vis** — scientific visualization algorithms (chapter 9)
- **pgc-rendering** — GPU path tracing renderer (chapter 10)

## Table of Contents

1. [Getting Started](01-getting-started.md) — Installation, first kernel, choosing a backend
2. [Fields and Types](02-fields-and-types.md) — Data containers, scalar types, numpy interop
3. [Kernels](03-kernels.md) — Writing kernels, parallel loops, scalar arguments
4. [Control Flow and Math](04-control-flow.md) — Loops, conditionals, math builtins
5. [Device Functions](05-device-functions.md) — `@pgc.func` for reusable device-side code
6. [Templates](06-templates.md) — `@pgc.data_oriented` classes for zero-cost abstraction
7. [Advanced Features](07-advanced.md) — Atomics, shared memory, local arrays, textures, vectors
8. [Backends](08-backends.md) — CPU, Metal, CUDA, HIP, Level Zero
9. [Visualization](09-visualization.md) — Flying edges, normals, scan, VTK interop (pgc-vis)
10. [Rendering](10-rendering.md) — Path tracing, BVH, camera, scene (pgc-rendering)
