# Tack — GPU compute framework

A Python-first GPU compute framework. Write compute kernels as decorated Python functions, run them on CPU, Metal (Apple Silicon), CUDA (NVIDIA), or HIP (AMD) — same code, any backend.

```python
import tack
import numpy as np

tack.init(arch=tack.metal)  # or tack.cpu, tack.cuda, tack.hip

n = 1_000_000
x = tack.field(dtype=tack.f32, shape=(n,))
y = tack.field(dtype=tack.f32, shape=(n,))
out = tack.field(dtype=tack.f32, shape=(n,))

x.from_numpy(np.random.randn(n).astype(np.float32))
y.from_numpy(np.random.randn(n).astype(np.float32))

@tack.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]

vector_add(x, y, out)
result = out.to_numpy()
```

## Installation

Tack is split into three packages: **tack-core** (compute framework), **tack-rendering** (path tracer), and **tack-vis** (visualization algorithms).

```bash
# Install everything from source with uv
git clone <repo-url>
cd tack
uv sync

# Or install individual packages
pip install tack-core              # core only
pip install tack-core[cpu]         # core + CPU backend (LLVM JIT)
pip install tack-core[metal]       # core + Metal backend (macOS)
pip install tack-core[cuda]        # core + CUDA backend
pip install tack-rendering         # path tracer (pulls in tack-core)
pip install tack-vis               # visualization (pulls in tack-core)
```

## Backends

| Backend | Platform | Init | Dependencies |
|---------|----------|------|-------------|
| CPU | Any | `tack.init(arch=tack.cpu)` | `llvmlite`, `numpy` |
| Metal | macOS (Apple Silicon) | `tack.init(arch=tack.metal)` | `pyobjc-framework-Metal` |
| CUDA | Linux/Windows (NVIDIA) | `tack.init(arch=tack.cuda)` | `cuda-python>=13.2`, CUDA toolkit |
| HIP | Linux (AMD) | `tack.init(arch=tack.hip)` | `hip-python`, ROCm toolkit |

## Compilation Pipeline

```
@tack.kernel Python function
    → Python AST
    → Tack IR (intermediate representation)
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
| `tack.i8` / `tack.u8` | 8-bit signed / unsigned integer |
| `tack.i16` / `tack.u16` | 16-bit signed / unsigned integer |
| `tack.i32` / `tack.u32` | 32-bit signed / unsigned integer |
| `tack.i64` / `tack.u64` | 64-bit signed / unsigned integer |
| `tack.f32` | 32-bit float |
| `tack.f64` | 64-bit float (CPU/CUDA only — Metal does not support double) |

## Fields

Fields are n-dimensional arrays that live on the backend device.

```python
x = tack.field(dtype=tack.f32, shape=(1024,))         # 1D
img = tack.field(dtype=tack.f32, shape=(512, 512))     # 2D
vol = tack.field(dtype=tack.f32, shape=(64, 64, 64))   # 3D

# Vector fields (3-component per element)
pixels = tack.Vector.field(3, dtype=tack.f32, shape=(width, height))

# Data transfer
x.from_numpy(np_array)    # host → device
result = x.to_numpy()     # device → host
x.fill(0.0)               # fill with scalar

# Reductions (GPU-accelerated on Metal)
total = x.sum()
lo = x.min()
hi = x.max()

# Convenience constructors
z = tack.zeros(dtype=tack.f32, shape=(1024,))
o = tack.ones(dtype=tack.i32, shape=(256,))
idx = tack.arange(100)

# Copy, convert, reshape, concatenate
backup = x.copy()
x_f64 = x.astype(tack.f64)
flat = img.reshape((512 * 512,))
combined = tack.concat([part_a, part_b])
```

## Kernel Language

### Loops

```python
@tack.kernel
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
    for i, j in tack.ndrange(width, height):
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
@tack.kernel
def saxpy(x, y, out, alpha, n):
    for i in range(n):
        out[i] = alpha * x[i] + y[i]

saxpy(x, y, out, 2.5, 1000)  # alpha=2.5, n=1000 passed as scalars
```

### Math Builtins

`sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `exp2`, `log`, `log2`, `log10`, `floor`, `ceil`, `abs`, `min`, `max`, `pow`

### Atomic Operations

```python
@tack.kernel
def histogram(data, bins):
    for i in range(data.shape[0]):
        bin_idx = int(data[i] * 10.0)
        tack.atomic_add(bins, bin_idx, 1.0)

@tack.kernel
def find_minmax(x, min_out, max_out):
    for i in range(x.shape[0]):
        tack.atomic_min(min_out, 0, x[i])
        tack.atomic_max(max_out, 0, x[i])
```

### Shared Memory & Synchronization

```python
@tack.kernel
def reduce_kernel(x, out):
    smem = tack.shared(tack.f32, 256)
    for i in range(x.shape[0]):
        tid = tack.thread_id()          # thread index within workgroup
        smem[tid] = x[i]
        tack.barrier()                  # synchronize workgroup threads
        # ... use smem for cooperative computation ...
```

### Debug Printing

```python
@tack.kernel
def debug_kern(x, out):
    for i in range(x.shape[0]):
        if x[i] < 0.0:
            print("negative at", i, "value:", x[i])  # printf on CPU/CUDA/HIP
        out[i] = abs(x[i])
```

### Device Functions

```python
@tack.func
def lerp(a, b, t):
    return a + t * (b - a)

@tack.kernel
def interpolate(x, y, out, t):
    for i in range(x.shape[0]):
        out[i] = lerp(x[i], y[i], t)  # inlined at compile time
```

### Vector Operations

```python
@tack.kernel
def normalize_field(positions, normals):
    for i in range(positions.shape[0]):
        v = tack.Vector([positions[i][0], positions[i][1], positions[i][2]])
        n = v.normalized()
        normals[i] = n

# Vector methods: .dot(w), .cross(w), .normalized(), .norm(), .norm_sqr()
```

### Template Classes

```python
@tack.data_oriented
class Grid:
    def __init__(self, data, dx):
        self.data = data   # field → becomes kernel parameter
        self.dx = dx       # scalar → compile-time constant

    @tack.func
    def sample(self, i):
        return self.data[i] * self.dx

@tack.kernel
def process(grid: tack.template(), out):
    for i in range(out.shape[0]):
        out[i] = grid.sample(i)

process(Grid(data_field, 0.1), output)
```

## Examples

40+ progressive examples split across packages. All accept `--arch cpu|metal|cuda|hip`.

```bash
uv run python packages/tack-core/examples/01_hello_tack.py              # simplest kernel
uv run python packages/tack-core/examples/12_mandelbrot.py --arch metal # fractal on GPU
uv run python packages/tack-vis/examples/27_flying_edges.py --arch cuda # isosurface
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
uv run pytest packages/tack-core/tests/test_cpu_jit.py        # CPU backend
uv run pytest packages/tack-core/tests/test_metal.py          # Metal backend
uv run pytest packages/tack-core/tests/test_hip.py            # HIP backend
uv run pytest packages/tack-vis/tests/test_algorithms.py      # vis algorithms
```

## ROCm / HIP Setup

On a system with ROCm installed:

```bash
# Install hip-python and Tack
pip install hip-python
pip install tack-core

# Run HIP test suite
uv run pytest packages/tack-core/tests/test_hip.py -v
```

## Project Structure

Tack is a monorepo with three packages sharing the `tack` namespace:

```
packages/
  tack-core/                # Core compute framework
    src/tack/
      __init__.py           # Public API + namespace merging
      lang/                 # AST transform, IR, type inference, optimization
      codegen/              # LLVM, MSL, CUDA, HIP, OpenCL code generators
      runtime/              # Backend dispatch and device management
      algorithms/           # General primitives (scan, copy)
    tests/                  # Core tests (CPU, Metal, CUDA, HIP, codegen)
    examples/               # Core examples (01-21)

  tack-rendering/            # Path tracing renderer
    src/tack/rendering/      # Camera, BVH, scene, pathtrace kernels
    examples/               # Rendering examples (33-36, 38)

  tack-vis/                  # Scientific visualization
    src/tack/
      algorithms/           # Flying edges, normals, cell-to-point
      data/                 # Data abstractions
      interop/              # VTK interop
    tests/                  # Vis algorithm tests
    examples/               # Vis examples (22, 25-28, 39)

examples/                   # Cross-package examples (32, 37)
docs/                       # Shared documentation
```
