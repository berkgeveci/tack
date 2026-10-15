# Tack — GPU compute framework

[![CI](https://github.com/berkgeveci/tack/actions/workflows/ci.yml/badge.svg)](https://github.com/berkgeveci/tack/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

A Python-first GPU compute framework. Write compute kernels as decorated Python functions, run them on CPU, Metal (Apple Silicon), CUDA (NVIDIA), HIP (AMD) or Level Zero (Intel) — same code, any backend.

```python
import tack
import numpy as np

tack.set_default_device("metal")  # or "cpu", "cuda", "hip", "level_zero"

n = 1_000_000
x = tack.empty((n,))
y = tack.empty((n,))
out = tack.empty((n,))

tack.copyto(x, np.random.randn(n).astype(np.float32))
tack.copyto(y, np.random.randn(n).astype(np.float32))

@tack.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]

vector_add(x, y, out)
result = out.numpy()
```

## Installation

Tack is split into two packages: **tack-core** (compute framework) and **tack-vis** (datasets and visualization: `tack.dataset`).

```bash
# Install everything from source with uv
git clone https://github.com/berkgeveci/tack.git
cd tack
uv sync --extra cpu    # add --extra metal, cuda, hip or level_zero for a GPU,
                       # and --extra interop for VTK and MFEM

# Or with pip, from the same clone (tack-core first: tack-vis needs it)
pip install -e "packages/tack-core[cpu]"         # core + CPU backend (LLVM JIT)
pip install -e "packages/tack-core[metal]"       # ...or + Metal (macOS)
pip install -e "packages/tack-core[cuda]"        # ...or + CUDA
pip install -e "packages/tack-core[hip]"         # ...or + HIP (see note below)
pip install -e "packages/tack-core[level_zero]"  # ...or + Level Zero (Intel)
pip install -e packages/tack-vis                 # datasets and visualization
```

Tack is not on PyPI yet; install it from source as above.

## Backends

| Backend | Platform | Device | Dependencies |
|---------|----------|------|-------------|
| CPU | Any | `tack.device("cpu")` | `llvmlite`, `numpy` |
| Metal | macOS (Apple Silicon) | `tack.device("metal")` | `pyobjc-framework-Metal` |
| CUDA | Linux/Windows (NVIDIA) | `tack.device("cuda")` | `cuda-python>=13.2`, CUDA toolkit |
| HIP | Linux (AMD) | `tack.device("hip")` | `hip-python`, ROCm toolkit |
| Level Zero | Linux (Intel) | `tack.device("level_zero")` | `libze_loader.so`, `libocloc.so` |

`tack.set_default_device("cuda")` makes a device the default, where arrays
are allocated; `with tack.device("cuda"):` does so for a block, and
`device=` on any constructor picks one for an array. Every array records its
device (`a.device`, `a.to_device(...)`), and a kernel runs where its arrays
are. With nothing set, the default is the CPU.

`hip-python` is on PyPI, so the `[hip]` extra is enough. It ships
manylinux x86_64 wheels only, so the dependency carries a platform marker and
installs nothing elsewhere rather than failing to resolve.

Its version tracks the ROCm release it binds to — 7.1.x against ROCm 7.1, 7.2.x
against 7.2 — so if yours does not match, ask for the one that does:

```bash
pip install "hip-python~=7.2.0"
```

Level Zero needs no Python package — just the system Level Zero runtime and Intel's
offline compiler (`intel-opencl-icd`).

## Compilation Pipeline

```
@tack.kernel Python function
    → Python AST
    → Tack IR (intermediate representation)
    → IR passes (resolve, type inference, conservative copy propagation)
    → Backend codegen:
        CPU:   LLVM IR → llvmlite JIT → native code
        Metal: MSL source → Metal compile → compute pipeline
        CUDA:  CUDA C source → NVRTC → PTX → cuLaunchKernel
        HIP:   HIP C source → hipRTC → code object → hipModuleLaunchKernel
        Level Zero: OpenCL C source → libocloc → SPIR-V → zeCommandListAppendLaunchKernel
```

Kernels are compiled on first call and cached by type signature. Subsequent calls with the same types skip compilation.
Compiled kernels are also kept on disk, so a later process starts warm; see
[The Kernel Cache](docs/users-guide/17-kernel-cache.md).

## Data Types

| Type | Description |
|------|-------------|
| `tack.i8` / `tack.u8` | 8-bit signed / unsigned integer |
| `tack.i16` / `tack.u16` | 16-bit signed / unsigned integer |
| `tack.i32` / `tack.u32` | 32-bit signed / unsigned integer |
| `tack.i64` / `tack.u64` | 64-bit signed / unsigned integer |
| `tack.f32` | 32-bit float |
| `tack.f64` | 64-bit float (not on Metal; on Level Zero, where the device has it) |

## Arrays

`tack.Array` is typed storage on a backend. Array shape counts elements;
the dtype describes a scalar, vector or matrix element. A dataset
`tack.dataset.Field` combines a function space with coefficient values.

```python
# empty leaves contents unspecified until a kernel or host write sets them;
# shape first, then dtype (f32 by default; NumPy dtypes work too)
x = tack.empty((1024,))                  # 1D
img = tack.empty((512, 512))             # 2D
vol = tack.empty((64, 64, 64))           # 3D
ids = tack.empty((1024,), tack.i32)

# Vector and matrix elements use the same array constructor
pixels = tack.empty((width, height), tack.vec3())
F = tack.empty((n,), tack.mat2())

# Data transfer
tack.copyto(x, np_array)                 # host → device, into an existing array
result = x.numpy()                       # device → host (a copy; np.asarray(x) too)
x.fill(0.0)                              # fill with a scalar
tack.copyto(pixels, rgb)                 # rgb has shape (width, height, 3)
rgb = pixels.numpy()                     # (width, height, 3)
x[7], pixels[3, 4]                       # read one element from the host (not written this way)

# Reductions (on the device for f32 arrays on GPU backends)
total = x.sum()
lo = x.min()
hi = x.max()

# Convenience constructors
copied = tack.array(np_array)            # independent storage, copies values and dtype
scratch = tack.empty_like(pixels)        # same shape and element type, no copy
z = tack.zeros((1024,))
o = tack.ones((256,), tack.i32)
c = tack.full((256,), 2.5)               # f32 from a Python float
idx = tack.arange(100)                   # i32

# Copy, convert, reshape, concatenate
backup = x.copy()
x_f64 = x.astype(tack.f64)
flat = img.reshape(-1)
combined = tack.concat([part_a, part_b])
```

## Kernel Language

### Loops

```python
# The top-level for-range is the kernel's one parallel loop.
@tack.kernel
def weighted(x, out):
    for i in range(x.shape[0]):
        # Nested for-range runs sequentially per thread
        for j in range(10):
            out[i] += x[i] * float(j)

# Step support
@tack.kernel
def evens(out):
    for i in range(0, 100, 2):     # i = 0, 2, 4, ..., 98
        out[i // 2] = float(i)

# Multi-dimensional parallel iteration
@tack.kernel
def fill(img):
    for i, j in tack.ndindex(img.shape[0], img.shape[1]):
        img[i, j] = float(i * j)

# A (start, end) pair instead of a size: the interior of a grid
@tack.kernel
def smooth(u, out, n, m):
    for i, j in tack.ndindex((1, n - 1), (1, m - 1)):
        out[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

# While loops with break/continue
@tack.kernel
def halve(x, out):
    for i in range(x.shape[0]):
        val = x[i]
        while val > 1.0:
            val = val / 2.0
        out[i] = val
```

### Scalar Arguments

Kernels accept Python scalars directly alongside arrays:

```python
@tack.kernel
def saxpy(x, y, out, alpha, n):
    for i in range(n):
        out[i] = alpha * x[i] + y[i]

saxpy(x, y, out, 2.5, 1000)  # alpha=2.5, n=1000 passed as scalars
```

A kernel whose last statement is `return expr` (with a return annotation,
`-> int` or `-> tack.f32`) gives the caller that value, computed after
the launch: `live = count_live(cells, count, n)`.

### Constants

A kernel does not see Python variables around it; reading one raises
`NameError`. Declare a value with `tack.constant` and kernels and device
functions can use the name. It is still an ordinary number in host code.

```python
DT = tack.constant(0.01)
GRID = tack.constant(128)
MULTIPLIER = tack.constant(747796405, tack.u32)   # typed: u32 arithmetic that wraps

@tack.kernel
def step(x, v):
    for i in range(GRID):
        x[i] += v[i] * DT
```

`math.pi`, `math.e` and `math.tau` can be written directly. A tuple
makes a vector constant (`SUN = tack.constant((0.5, 0.5, 0.0))`), a tuple
of tuples a matrix. `tack.math` adds `fract`, `mix`, `clamp`, `saturate`,
`smoothstep`, `step`, `sign`, `length`, `distance` and `normalize` as device functions
that work on scalars and vectors alike. `tack.random`
draws uniform, normal and unit-vector values from an explicit `u32`
state (`state = random.seed(i, frame)`, `u, state = random.uniform(state)`),
reproducible on every backend and in NumPy.

### Math Builtins

`sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `exp2`, `log`, `log2`, `log10`, `floor`, `ceil`, `abs`, `min`, `max`, `pow`

`min` and `max` take two or more values. All of them apply to each
component of a vector or matrix.

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

# The old value: how a thread claims a unique slot
@tack.kernel
def compact(values, out, count, n):
    for i in range(n):
        if values[i] > 0.0:
            out[tack.atomic_add(count, 0, 1)] = values[i]

# One index per dimension, and a whole vector into a vector array
@tack.kernel
def scatter(pos, vel, mass, grid_m, grid_v, inv_dx, n):
    for p in range(n):
        cell = int(pos[p] * inv_dx)                    # a vector of cell indices
        tack.atomic_add(grid_m, cell, mass[p])
        tack.atomic_add(grid_v, (cell[0], cell[1]), mass[p] * vel[p])
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

To find an index out of bounds, run in debug mode: `tack.set_debug(True)`
or `TACK_DEBUG=1` checks every array index and raises `IndexError` naming
the access. See [Debugging and timing](docs/users-guide/15-debugging-and-timing.md).

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

A device function can return several values, vectors and matrices among
them: `distance, normal, color = closest_hit(origin, direction)`.

### Vectors and Matrices

Vectors and small matrices (up to 4x4) are values in kernels. They are
scalarized at compile time, so they cost what the scalar code would. In a
kernel the element-type names build them: `tack.vec3(x, y, z)`,
`tack.vec3(s)` (every component), `tack.mat2([a, b], [c, d])`,
`tack.mat2(a, b, c, d)` (row-major), `tack.mat(row, ...)` for other
shapes and `tack.identity(n)`; a list of scalars `[x, y]` is a vector too.

```python
@tack.kernel
def step(pos, vel, F, C, grid, out, dt, n):
    for p in range(n):
        v = vel[p]                              # a whole vector from a vector array
        v = min(max(v, -10.0), 10.0)            # builtins apply to each component
        v.y -= 9.8 * dt                         # components: v.x, v[1], pos[p].z, pos[p][0]
        cell = int(floor(pos[p] / 0.25))        # a vector of integers
        out[p] = grid[cell] * v.norm()          # a vector indexes an array, one dimension each
        pos[p] += v * dt                        # update an array element in place
        vel[p], out[p] = v, v.dot(v)            # tuple assignment to array elements

        F[p] = (tack.identity(2) + dt * C[p]) @ F[p]     # '@' multiplies
        J = F[p].determinant()
        stress = (F[p] - F[p].inverse().transpose()) * J        # '*' is entry by entry
        C[p] = v.outer_product(stress @ v)
```

Comparing vectors gives a mask of `0`/`1` per component: `any(p > 1.0)`,
`all(-1.0 <= p <= 1.0)`, and `tack.select(mask, a, b)` picks per component.

Vector methods: `.dot(w)`, `.cross(w)`, `.norm()`, `.norm_sqr()`,
`.normalized()`, `.sum()`, `.min()`, `.max()`, `.outer_product(w)`.
Matrix methods: `.transpose()`, `.trace()`, and for 2x2 and 3x3
`.determinant()` and `.inverse()`; entries are `A[i, j]`. See the
[User's Guide](docs/users-guide/07-advanced.md) for the rules.

### Template Classes

```python
@tack.device_class
class Grid:
    scale = 2.0            # class-level scalar → compile-time constant

    def __init__(self, data, dx):
        self.data = data   # array → becomes kernel parameter
        self.dx = dx       # instance scalar → runtime parameter, no recompilation

    @tack.func
    def sample(self, i):
        return self.data[i] * self.dx

@tack.kernel
def process(grid, out):
    for i in range(out.shape[0]):
        out[i] = grid.sample(i)

process(Grid(data, 0.1), output)
```

A device class can derive from another and inherits its
`@tack.func` methods and class constants. An object can also hold a
`@tack.func` as an attribute (`self.smoothing = cubic`) and call it.
A kernel can be a method too: `@tack.kernel def step(self, dt)` is called
as `grid.step(0.1)` and reaches the object's arrays through `self`.

## Datasets and Visualization

`tack.dataset` (in tack-vis) holds meshes and the fields on them -- structured
grids, unstructured meshes of the linear cells, polyhedra, polygons and
polylines -- and the algorithms that work on them: contour, slice, clip,
threshold, external faces, gradients, projections between spaces,
streamlines and more. They are Tack kernels, so they run on every backend,
and your own kernels see a mesh the same way through `for_each`.

```python
import tack
import tack.dataset as td

tack.set_default_device("cpu")
data = td.wavelet(extent=10)                    # VTK's wavelet: 21^3 points
surface = td.contour(data, "RTData", 157.0)     # an isosurface, as triangles
print(surface.num_cells, "triangles")
```

See [Datasets and visualization](docs/users-guide/09-visualization.md) in the
User's Guide and the [Dataset API reference](docs/reference/dataset-api.md).

## Examples

32 progressive examples split across packages. All accept `--arch cpu|metal|cuda|hip|level_zero`.

```bash
uv run python packages/tack-core/examples/01_hello_tack.py              # simplest kernel
uv run python packages/tack-core/examples/12_mandelbrot.py --arch metal # fractal on GPU
uv run python packages/tack-vis/examples/42_dataset_api_tour.py --arch cuda # tack.dataset; needs --extra interop
```

| Location | Count | Covers |
|----------|------:|--------|
| `packages/tack-core/examples/` | 28 | Kernels, arrays, math, control flow, vectors, templates, atomics, shared memory; then Mandelbrot, N-body, Jacobi, matmul, Game of Life, heat/wave, textures, DLPack |
| `packages/tack-vis/examples/` | 4 | A tour of `tack.dataset`, an MFEM solution through it, multi-block contour, VTK interop |

Numbers are unique within a package but repeat across packages — `tack-vis/31_multiblock_contour.py`
and `tack-core/31_dawn_interop.py` both exist.

## Testing

```bash
uv sync --extra cpu --extra dev --extra interop               # pytest, pytest-xdist, VTK, MFEM
uv run pytest packages/tack-core/tests/test_cpu_jit.py        # CPU backend
uv run pytest packages/tack-core/tests/test_metal.py          # Metal backend
uv run pytest packages/tack-core/tests/test_hip.py            # HIP backend
uv run pytest packages/tack-vis/tests                         # datasets and vis algorithms
uv run pytest -m quick -n 4                                   # a module per area, on 4 workers
uv run pytest -n auto                                         # all tests, across processes
```

Run the files a change touches first, then `-m quick`; the whole suite is for a
validation round. Without `--extra interop` the comparisons against VTK and MFEM
skip.

## ROCm / HIP Setup

On a system with ROCm installed:

```bash
# Install Tack with the HIP extra, which brings hip-python
uv sync --extra cpu --extra hip

# Run HIP test suite
uv run pytest packages/tack-core/tests/test_hip.py -v
```

## Project Structure

Tack is a monorepo with two packages sharing the `tack` namespace:

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
    examples/               # Core examples

  tack-vis/                  # Datasets and visualization
    src/tack/
      dataset/              # tack.dataset: topologies, spaces, fields, sets, views
        algorithms/         # Every algorithm, one module each
      interop/              # VTK and MFEM interop
    tests/                  # Dataset and algorithm tests
    examples/               # Vis examples

archive/                    # Retired packages kept for reference (tack-rendering)
benchmarks/                 # Performance benchmarks
docs/                       # Users guide, developers guide, performance notes
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `uv sync --extra cpu --extra dev`,
then `uv run pytest -m quick -n 4`. Tests for a backend you don't have skip themselves, so a green
run on one machine does not mean every backend was exercised — the CI log prints
which backends it found.

## License

BSD 3-Clause. See [LICENSE](LICENSE).

Copyright (c) 2026, Kitware, Inc.
