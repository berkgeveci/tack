# PGC — Portable GPU Compute

A Python-first GPU compute framework. Write compute kernels as decorated Python functions, run them on CPU, Metal (Apple Silicon), CUDA (NVIDIA), or HIP (AMD) — same code, any backend.

```python
import pgc
import numpy as np

pgc.init(arch=pgc.metal)  # or pgc.cpu, pgc.cuda, pgc.hip

n = 1_000_000
x = pgc.field(dtype=pgc.f32, shape=(n,))
y = pgc.field(dtype=pgc.f32, shape=(n,))
out = pgc.field(dtype=pgc.f32, shape=(n,))

x.from_numpy(np.random.randn(n).astype(np.float32))
y.from_numpy(np.random.randn(n).astype(np.float32))

@pgc.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]

vector_add(x, y, out)
result = out.to_numpy()
```

## Installation

PGC is split into three packages: **pgc-core** (compute framework), **pgc-rendering** (path tracer), and **pgc-vis** (visualization algorithms).

```bash
# Install everything from source with uv
git clone <repo-url>
cd pgc
uv sync

# Or install individual packages
pip install pgc-core              # core only
pip install pgc-core[cpu]         # core + CPU backend (LLVM JIT)
pip install pgc-core[metal]       # core + Metal backend (macOS)
pip install pgc-core[cuda]        # core + CUDA backend
pip install pgc-rendering         # path tracer (pulls in pgc-core)
pip install pgc-vis               # visualization (pulls in pgc-core)
```

## Backends

| Backend | Platform | Init | Dependencies |
|---------|----------|------|-------------|
| CPU | Any | `pgc.init(arch=pgc.cpu)` | `llvmlite`, `numpy` |
| Metal | macOS (Apple Silicon) | `pgc.init(arch=pgc.metal)` | `pyobjc-framework-Metal` |
| CUDA | Linux/Windows (NVIDIA) | `pgc.init(arch=pgc.cuda)` | `cuda-python>=13.2`, CUDA toolkit |
| HIP | Linux (AMD) | `pgc.init(arch=pgc.hip)` | `hip-python`, ROCm toolkit |

## Compilation Pipeline

```
@pgc.kernel Python function
    → Python AST
    → PGC IR (intermediate representation)
    → IR passes (resolve, type inference, LICM, copy propagation, CSE)
    → Backend codegen:
        CPU:   LLVM IR → llvmlite JIT → native code
        Metal: MSL source → Metal compile → compute pipeline
        CUDA:  CUDA C source → NVRTC → PTX → cuLaunchKernel
        HIP:   HIP C source → hipRTC → code object → hipLaunchKernel
```

Kernels are compiled on first call and cached by type signature. Subsequent calls with the same types skip compilation.

## Data Types

| Type | Description |
|------|-------------|
| `pgc.f32` | 32-bit float |
| `pgc.f64` | 64-bit float (CPU/CUDA only — Metal does not support double) |
| `pgc.i32` | 32-bit signed integer |
| `pgc.i64` | 64-bit signed integer |
| `pgc.u32` | 32-bit unsigned integer |
| `pgc.u64` | 64-bit unsigned integer |

## Fields

Fields are n-dimensional arrays that live on the backend device.

```python
x = pgc.field(dtype=pgc.f32, shape=(1024,))         # 1D
img = pgc.field(dtype=pgc.f32, shape=(512, 512))     # 2D
vol = pgc.field(dtype=pgc.f32, shape=(64, 64, 64))   # 3D

# Vector fields (3-component per element)
pixels = pgc.Vector.field(3, dtype=pgc.f32, shape=(width, height))

# Data transfer
x.from_numpy(np_array)    # host → device
result = x.to_numpy()     # device → host
x.fill(0.0)               # fill with scalar

# Reductions (GPU-accelerated on Metal)
total = x.sum()
lo = x.min()
hi = x.max()
```

## Kernel Language

### Loops

```python
@pgc.kernel
def kern(x, out):
    # Top-level for-range is parallelized across GPU threads
    for i in range(x.shape[0]):
        # Nested for-range runs sequentially per thread
        for j in range(10):
            out[i] += x[i] * float(j)

    # Step support
    for i in range(0, 100, 2):     # i = 0, 2, 4, ..., 98
        out[i // 2] = float(i)

    # Multi-dimensional parallel iteration
    for i, j in pgc.ndrange(width, height):
        img[i, j] = compute(i, j)

    # While loops with break/continue
    for i in range(n):
        val = x[i]
        while val > 1.0:
            val = val / 2.0
        out[i] = val
```

### Scalar Arguments

Kernels accept Python scalars directly alongside fields:

```python
@pgc.kernel
def saxpy(x, y, out, alpha, n):
    for i in range(n):
        out[i] = alpha * x[i] + y[i]

saxpy(x, y, out, 2.5, 1000)  # alpha=2.5, n=1000 passed as scalars
```

### Math Builtins

`sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `exp2`, `log`, `log2`, `log10`, `floor`, `ceil`, `abs`, `min`, `max`, `pow`

### Atomic Operations

```python
@pgc.kernel
def histogram(data, bins):
    for i in range(data.shape[0]):
        bin_idx = int(data[i] * 10.0)
        pgc.atomic_add(bins, bin_idx, 1.0)

@pgc.kernel
def find_minmax(x, min_out, max_out):
    for i in range(x.shape[0]):
        pgc.atomic_min(min_out, 0, x[i])
        pgc.atomic_max(max_out, 0, x[i])
```

### Shared Memory & Synchronization

```python
@pgc.kernel
def reduce_kernel(x, out):
    smem = pgc.shared(pgc.f32, 256)
    for i in range(x.shape[0]):
        tid = pgc.thread_id()          # thread index within workgroup
        smem[tid] = x[i]
        pgc.barrier()                  # synchronize workgroup threads
        # ... use smem for cooperative computation ...
```

### Debug Printing

```python
@pgc.kernel
def debug_kern(x, out):
    for i in range(x.shape[0]):
        if x[i] < 0.0:
            print("negative at", i, "value:", x[i])  # printf on CPU/CUDA/HIP
        out[i] = abs(x[i])
```

### Device Functions

```python
@pgc.func
def lerp(a, b, t):
    return a + t * (b - a)

@pgc.kernel
def interpolate(x, y, out, t):
    for i in range(x.shape[0]):
        out[i] = lerp(x[i], y[i], t)  # inlined at compile time
```

### Vector Operations

```python
@pgc.kernel
def normalize_field(positions, normals):
    for i in range(positions.shape[0]):
        v = pgc.Vector([positions[i][0], positions[i][1], positions[i][2]])
        n = v.normalized()
        normals[i] = n

# Vector methods: .dot(w), .cross(w), .normalized(), .norm(), .norm_sqr()
```

### Template Classes

```python
@pgc.data_oriented
class Grid:
    def __init__(self, data, dx):
        self.data = data   # field → becomes kernel parameter
        self.dx = dx       # scalar → compile-time constant

    @pgc.func
    def sample(self, i):
        return self.data[i] * self.dx

@pgc.kernel
def process(grid: pgc.template(), out):
    for i in range(out.shape[0]):
        out[i] = grid.sample(i)

process(Grid(data_field, 0.1), output)
```

## Examples

40+ progressive examples split across packages. All accept `--arch cpu|metal|cuda|hip`.

```bash
uv run python packages/pgc-core/examples/01_hello_pgc.py              # simplest kernel
uv run python packages/pgc-core/examples/12_mandelbrot.py --arch metal # fractal on GPU
uv run python packages/pgc-vis/examples/27_flying_edges.py --arch cuda # isosurface
uv run python examples/32_pathtrace.py --arch metal                    # path tracing
```

| # | Package | Example | Concepts |
|---|---------|---------|----------|
| 01-11 | core | Getting Started | Kernels, fields, math, control flow, vectors, templates |
| 12-20 | core | Applications | Mandelbrot, N-body, Jacobi, matmul, Game of Life, heat/wave |
| 22-28 | vis | Visualization | Contour, flying edges, marching cubes, point-to-cell |
| 32-38 | rendering | Path Tracing | GPU BVH, path tracing, denoising |
| 32, 37 | cross | Pipelines | Flying edges → path trace, in-situ simulation |

## Testing

```bash
uv run pytest                                                # all tests
uv run pytest packages/pgc-core/tests/test_cpu_jit.py        # CPU backend
uv run pytest packages/pgc-core/tests/test_metal.py          # Metal backend
uv run pytest packages/pgc-core/tests/test_hip.py            # HIP backend
uv run pytest packages/pgc-vis/tests/test_algorithms.py      # vis algorithms
```

## ROCm / HIP Setup

On a system with ROCm installed:

```bash
# Install hip-python and PGC
pip install hip-python
pip install pgc-core

# Run HIP test suite
uv run pytest packages/pgc-core/tests/test_hip.py -v
```

## Project Structure

PGC is a monorepo with three packages sharing the `pgc` namespace:

```
packages/
  pgc-core/                # Core compute framework
    src/pgc/
      __init__.py           # Public API + namespace merging
      lang/                 # AST transform, IR, type inference, optimization
      codegen/              # LLVM, MSL, CUDA, HIP, OpenCL code generators
      runtime/              # Backend dispatch and device management
      algorithms/           # General primitives (scan, copy)
    tests/                  # Core tests (CPU, Metal, CUDA, HIP, codegen)
    examples/               # Core examples (01-21)

  pgc-rendering/            # Path tracing renderer
    src/pgc/rendering/      # Camera, BVH, scene, pathtrace kernels
    examples/               # Rendering examples (33-36, 38)

  pgc-vis/                  # Scientific visualization
    src/pgc/
      algorithms/           # Flying edges, normals, cell-to-point
      data/                 # Data abstractions
      interop/              # VTK interop
    tests/                  # Vis algorithm tests
    examples/               # Vis examples (22, 25-28, 39)

examples/                   # Cross-package examples (32, 37)
docs/                       # Shared documentation
```
