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

```bash
# CPU only (requires llvmlite)
pip install -e .

# Metal (macOS, Apple Silicon)
pip install -e ".[metal]"

# CUDA (Linux/Windows, NVIDIA GPU)
pip install -e ".[cuda]"

# HIP (Linux, AMD GPU with ROCm)
pip install hip-python
pip install -e .
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

20 progressive examples covering all features, from hello-world to real applications. All accept `--arch cpu|metal|cuda|hip`.

```bash
uv run python examples/01_hello_pgc.py              # simplest kernel
uv run python examples/12_mandelbrot.py --arch metal # fractal on GPU
uv run python examples/13_nbody.py --arch hip        # N-body on AMD
uv run python examples/20_multi_backend.py           # compare all backends
```

| # | Example | Concepts |
|---|---------|----------|
| 01 | Hello PGC | `pgc.init`, `pgc.field`, `@pgc.kernel`, `from_numpy`/`to_numpy` |
| 02 | Math Builtins | `sqrt`, `sin`, `cos`, `exp`, `log`, `abs`, `min`, `max`, ... |
| 03 | Scalar Arguments | Passing Python int/float directly to kernels |
| 04 | Control Flow | `if`/`else`, `while`, `break`, ternary, nested loops |
| 05 | ndrange | 2D parallel iteration, stencil patterns |
| 06 | Device Functions | `@pgc.func` helpers inlined at compile time |
| 07 | Atomics | `atomic_add`, `atomic_min`, `atomic_max`, histogram |
| 08 | Reductions | `field.sum()`, `field.min()`, `field.max()` |
| 09 | Shared Memory | `pgc.shared`, `pgc.thread_id`, `pgc.barrier` |
| 10 | Vectors | `pgc.Vector`, `.dot()`, `.cross()`, `.normalized()` |
| 11 | Templates | `@pgc.data_oriented` classes with `pgc.template()` |
| 12 | Mandelbrot | Fractal rendering with smooth coloring |
| 13 | N-body | Gravitational simulation (O(N²) all-pairs) |
| 14 | Jacobi Solver | 2D Laplace equation with iterative stencil |
| 15 | Matrix Multiply | Dense matrix multiplication with benchmarking |
| 16 | Game of Life | Conway's cellular automaton on a 2D grid |
| 17 | Heat Equation | 2D diffusion with explicit Euler + 5-point stencil |
| 18 | Wave Equation | 1D wave propagation with leapfrog integration |
| 19 | Image Processing | Brightness/contrast, Sobel edges, separable blur |
| 20 | Multi-Backend | Same kernel on all available backends with benchmarks |

## Testing

```bash
uv run pytest                           # all tests
uv run pytest tests/test_cpu_jit.py     # CPU backend
uv run pytest tests/test_metal.py       # Metal backend
uv run pytest tests/test_hip.py         # HIP backend (requires ROCm)
uv run pytest tests/test_cuda.py        # CUDA backend (requires CUDA)
uv run pytest tests/test_llvm_gen.py    # LLVM codegen unit tests
uv run pytest tests/test_hip_gen.py     # HIP codegen unit tests
uv run pytest tests/test_ast_transform.py  # AST transform tests
```

## ROCm / HIP Setup

On a system with ROCm installed:

```bash
# Install hip-python
pip install hip-python

# Install PGC
pip install -e .

# Test
python -c "
import pgc, numpy as np
pgc.init(arch=pgc.hip)
x = pgc.field(dtype=pgc.f32, shape=(1024,))
out = pgc.field(dtype=pgc.f32, shape=(1024,))
x.from_numpy(np.arange(1024, dtype=np.float32))

@pgc.kernel
def double(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] * 2.0

double(x, out)
print('Result:', out.to_numpy()[:5])  # [0, 2, 4, 6, 8]
"

# Run HIP test suite
pytest tests/test_hip.py -v
```

## Project Structure

```
src/pgc/
  __init__.py              # Public API: field, kernel, func, types, atomics, shared, etc.
  lang/
    ir.py                  # IR node definitions (20+ node types)
    ast_transform.py       # Python AST → PGC IR
    type_inference.py      # Runtime type annotation from actual arguments
    ir_resolve.py          # Resolve dimension sizes to constants
    ir_optimize.py         # LICM, copy propagation, CSE
    field.py               # Field, DeviceBuffer, Vector, NumpyBuffer
    kernel.py              # @pgc.kernel decorator
    func.py                # @pgc.func decorator (device functions)
    data_oriented.py       # @pgc.data_oriented decorator (templates)
    template_rewrite.py    # Template argument expansion
    types.py               # ScalarType: f32, f64, i32, i64, u32, u64
  codegen/
    llvm_gen.py            # PGC IR → LLVM IR (CPU backend)
    msl_gen.py             # PGC IR → Metal Shading Language
    cuda_gen.py            # PGC IR → CUDA C source
    hip_gen.py             # PGC IR → HIP C source (extends CUDACodeGen)
    spirv_gen.py           # PGC IR → SPIR-V binary (legacy, used for Vulkan path)
  runtime/
    dispatch.py            # Backend selection (pgc.init)
    cpu.py                 # CPU backend: llvmlite JIT, ThreadPoolExecutor
    metal.py               # Metal backend: pyobjc, compute pipelines, GPU reductions
    cuda_backend.py        # CUDA backend: cuda-python, NVRTC, cuLaunchKernel
    hip_backend.py         # HIP backend: hip-python, hipRTC, hipLaunchKernel
tests/
  test_cpu_jit.py          # CPU end-to-end tests
  test_metal.py            # Metal end-to-end tests
  test_cuda.py             # CUDA end-to-end tests
  test_hip.py              # HIP end-to-end tests
  test_ast_transform.py    # AST transform unit tests
  test_llvm_gen.py         # LLVM codegen tests
  test_hip_gen.py          # HIP codegen tests
  test_type_inference.py   # Type inference tests
  test_new_features.py     # Scalar args, ndrange, while loops, etc.
  test_templates.py        # @pgc.data_oriented template tests
  test_spirv_gen.py        # SPIR-V codegen tests
  test_vector_add.py       # Basic vector add integration test
```
