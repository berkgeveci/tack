# PGC Examples

Examples are split across packages. Each package has its own `examples/` directory.
Cross-package examples (requiring both pgc-rendering and pgc-vis) live here at the root.

## Core Examples (`packages/pgc-core/examples/`)

| # | Example | Concepts |
|---|---------|----------|
| 01 | Hello PGC | `pgc.init`, `pgc.field`, `@pgc.kernel`, `from_numpy`/`to_numpy` |
| 02 | Math Builtins | `sqrt`, `sin`, `cos`, `exp`, `log`, `abs`, `min`, `max`, ... |
| 03 | Scalar Arguments | Passing Python int/float values directly to kernels |
| 04 | Control Flow | `if`/`else`, `while`, `break`, ternary expressions, nested loops |
| 05 | ndrange | 2D parallel iteration with `pgc.ndrange`, stencil patterns |
| 06 | Device Functions | `@pgc.func` helpers inlined at compile time |
| 07 | Atomics | `atomic_add`, `atomic_min`, `atomic_max`, histogram |
| 08 | Reductions | `field.sum()`, `field.min()`, `field.max()` |
| 09 | Shared Memory | `pgc.shared`, `pgc.shared_like`, `pgc.thread_id`, `pgc.barrier` |
| 10 | Vectors | `pgc.Vector`, `.dot()`, `.cross()`, `.normalized()` |
| 11 | Templates | `@pgc.data_oriented` classes with `pgc.template()` |
| 12-20 | Applications | Mandelbrot, N-body, Jacobi, matmul, Game of Life, heat/wave, image processing |
| 21 | Array Abstraction | VTK-style array abstraction with compile-time dispatch |

## Visualization Examples (`packages/pgc-vis/examples/`)

| # | Example | Description |
|---|---------|-------------|
| 22 | Contour | Marching squares contour with GPU prefix sum |
| 25 | Point to Cell | Point-to-cell averaging with structured vs explicit connectivity |
| 26 | Marching Cubes | Cell-based marching cubes isosurface on 3D grids |
| 27 | Flying Edges | True FlyingEdges with edge ownership, merged unique points |
| 28 | Flying Edges Trimmed | Trimmed version of flying edges |
| 31 | Multiblock FE | Multi-block flying edges |
| 39 | VTK Interop | VTK ↔ PGC zero-copy array exchange |

## Rendering Examples (`packages/pgc-rendering/examples/`)

| # | Example | Description |
|---|---------|-------------|
| 33 | Pathtrace Primitives | Path tracing with sphere/box primitives |
| 34 | Pathtrace Scale | Large-scale path tracing |
| 35 | Pathtrace Closeup | Close-up path tracing |
| 36 | Pathtrace Denoise | Path tracing with OIDN denoising |
| 38 | Point Colors | Per-point color rendering |

## Cross-Package Examples (this directory)

| # | Example | Description | Requires |
|---|---------|-------------|----------|
| 32 | Pathtrace | Flying edges → path trace render | pgc-rendering + pgc-vis |
| 37 | In-Situ Pipeline | Simulation → flying edges → render | pgc-rendering + pgc-vis |

## Running

All examples accept `--arch cpu|metal|cuda|hip` to select the backend:

```bash
uv run python packages/pgc-core/examples/01_hello_pgc.py               # default: CPU
uv run python packages/pgc-core/examples/12_mandelbrot.py --arch metal  # Apple Silicon
uv run python packages/pgc-vis/examples/27_flying_edges.py --arch cuda  # NVIDIA GPU
uv run python examples/32_pathtrace.py --arch metal                     # cross-package
```
