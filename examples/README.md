# PGC Examples

Progressive examples covering all PGC features, from basic to advanced.

## Getting Started (Features)

| # | Example | Concepts |
|---|---------|----------|
| 01 | [Hello PGC](01_hello_pgc.py) | `pgc.init`, `pgc.field`, `@pgc.kernel`, `from_numpy`/`to_numpy` |
| 02 | [Math Builtins](02_math_builtins.py) | `sqrt`, `sin`, `cos`, `exp`, `log`, `abs`, `min`, `max`, ... |
| 03 | [Scalar Arguments](03_scalar_args.py) | Passing Python int/float values directly to kernels |
| 04 | [Control Flow](04_control_flow.py) | `if`/`else`, `while`, `break`, ternary expressions, nested loops |
| 05 | [ndrange](05_ndrange.py) | 2D parallel iteration with `pgc.ndrange`, stencil patterns |
| 06 | [Device Functions](06_device_functions.py) | `@pgc.func` helpers inlined at compile time |
| 07 | [Atomics](07_atomics.py) | `atomic_add`, `atomic_min`, `atomic_max`, histogram |
| 08 | [Reductions](08_reductions.py) | `field.sum()`, `field.min()`, `field.max()` |
| 09 | [Shared Memory](09_shared_memory.py) | `pgc.shared`, `pgc.thread_id`, `pgc.barrier` |
| 10 | [Vectors](10_vectors.py) | `pgc.Vector`, `.dot()`, `.cross()`, `.normalized()` |
| 11 | [Templates](11_templates.py) | `@pgc.data_oriented` classes with `pgc.template()` |

## Applications

| # | Example | Description |
|---|---------|-------------|
| 12 | [Mandelbrot](12_mandelbrot.py) | Fractal rendering with smooth coloring |
| 13 | [N-body](13_nbody.py) | Gravitational simulation (O(N²) all-pairs) |
| 14 | [Jacobi Solver](14_jacobi_solver.py) | 2D Laplace equation with iterative stencil |
| 15 | [Matrix Multiply](15_matmul.py) | Dense matrix multiplication with benchmarking |
| 16 | [Game of Life](16_game_of_life.py) | Conway's cellular automaton on a 2D grid |
| 17 | [Heat Equation](17_heat_equation.py) | 2D diffusion with explicit Euler + 5-point stencil |
| 18 | [Wave Equation](18_wave_equation.py) | 1D wave propagation with leapfrog integration |
| 19 | [Image Processing](19_image_processing.py) | Brightness/contrast, Sobel edges, separable blur |
| 20 | [Multi-Backend](20_multi_backend.py) | Same kernel on CPU/Metal/CUDA/HIP with benchmarks |

## Data Abstractions & Algorithms

| # | Example | Description |
|---|---------|-------------|
| 21 | [Array Abstraction](21_array_abstraction.py) | VTK-style array abstraction with compile-time dispatch |
| 22 | [Contour](22_contour.py) | Marching squares contour with GPU prefix sum |
| 23 | [Tuple Arrays](23_tuple_arrays.py) | Multi-component AOS vs SOA layouts with generic kernels |
| 24 | [Point Coordinates](24_point_coordinates.py) | Product vs AOS vs SOA coordinate performance benchmark |
| 25 | [Point to Cell](25_point_to_cell.py) | Point-to-cell averaging with structured vs explicit connectivity |
| 26 | [Marching Cubes](26_marching_cubes.py) | Cell-based marching cubes isosurface on 3D grids |
| 27 | [Flying Edges](27_flying_edges.py) | True FlyingEdges with edge ownership, merged unique points |

## Running

All examples accept `--arch cpu|metal|cuda|hip` to select the backend:

```bash
uv run python examples/01_hello_pgc.py               # default: CPU
uv run python examples/12_mandelbrot.py --arch metal  # Apple Silicon GPU
uv run python examples/13_nbody.py --arch cuda        # NVIDIA GPU
uv run python examples/15_matmul.py --arch hip        # AMD GPU (ROCm)

# Run validation suite (auto-detects all available backends)
uv run python examples/validate_all.py
```

Examples 12, 14, 17, 18, 19 save PNG files if matplotlib is installed.
