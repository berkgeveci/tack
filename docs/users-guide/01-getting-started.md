# Getting Started

## Installation

Tack is a pure Python package with JIT compilation at runtime — no build step.
It is split into three packages:

- **tack-core** — the compute framework (kernels, fields, types, backends)
- **tack-rendering** — path tracing renderer
- **tack-vis** — scientific visualization algorithms (flying edges, VTK interop)

```bash
# Install everything from source with uv
git clone https://github.com/berkgeveci/tack.git
cd tack
uv sync --extra cpu

# Or install individual packages
pip install tack-core          # core only
pip install 'tack-core[cpu]'     # core + CPU backend (LLVM JIT)
pip install tack-rendering     # rendering (pulls in tack-core)
pip install tack-vis           # visualization (pulls in tack-core)
```

Python 3.11 or newer is required. A bare `tack-core` install has the numerical
API but no CPU JIT dependency; choose the backend extra you intend to run.
`tack-vis` and `tack-rendering` also need a working backend. See
[Backends](08-backends.md) for system runtimes required by GPU extras.

The guide describes the current checkout, including additions recorded under
[Unreleased in the changelog](https://github.com/berkgeveci/tack/blob/main/CHANGELOG.md).
If an installed release lacks an API used here, run the source checkout or check
its release notes. The tutorial scripts below are included in this checkout.

## Your First Kernel

```python
import numpy as np
import tack

tack.init(arch=tack.cpu)

# Create fields from numpy arrays
n = 1024
x = tack.field_like(np.arange(n, dtype=np.float32))
y = tack.field_like(np.ones(n, dtype=np.float32) * 2.0)
out = tack.field(dtype=tack.f32, shape=(n,))

# Define a kernel
@tack.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]

# Run it
vector_add(x, y, out)

# Read results back to numpy
result = out.to_numpy()
print(result[:5])  # [2. 3. 4. 5. 6.]
```

Save this example as `first_kernel.py` and run `python first_kernel.py` in your
installed environment, or `uv run python first_kernel.py` from the checkout.
The result assertion you can add is:

```python
np.testing.assert_array_equal(result, np.arange(n, dtype=np.float32) + 2.0)
```

A kernel is a Python function decorated with `@tack.kernel`. The outermost
`for i in range(...)` becomes a parallel loop — each iteration runs as a
separate thread on the GPU (or is split across CPU cores).

## Choosing a Backend

```python
tack.init(arch=tack.cpu)         # CPU via LLVM JIT
tack.init(arch=tack.metal)       # Apple GPU
tack.init(arch=tack.cuda)        # NVIDIA GPU
tack.init(arch=tack.hip)         # AMD GPU (ROCm)
tack.init(arch=tack.level_zero)  # Intel GPU
```

All examples accept `--arch` on the command line:

```bash
uv run python packages/tack-core/examples/01_hello_tack.py --arch metal
```

This kernel uses the portable numerical subset and runs on all five backends.
Backend capabilities still matter: Metal has no `f64`, and CPU has no shared
workgroup memory. Initialize before allocating fields; a later backend switch
does not migrate them.

## How It Works

When you call a kernel for the first time, Tack:

1. Reads the Python source of the decorated function
2. Transforms the AST into Tack's internal IR
3. Resolves shapes and types, then propagates safe copies
4. Generates backend-specific code (LLVM IR, MSL, CUDA C, etc.)
5. Compiles and dispatches

LLVM and vendor compilers optimize the generated code. Subsequent calls
with the same specialization reuse the compiled kernel; dtypes, vector
widths, baked-in shapes, and template structure can require another variant.

## A useful next step

Read [From Python to a Tack program](12-execution-model.md) to understand the
host/kernel boundary, then work through [Heat diffusion](tutorials/heat.md).
The [tutorial gallery](tutorials/index.md) offers seven complete programs with
figures, downloadable source, and checks of their results.

The numerical tutorials need NumPy and the selected Tack backend. To save their
optional plots, also install VTK:

```bash
pip install vtk
# From a uv checkout, without changing project dependencies:
uv run --with vtk python docs/examples/heat.py --output heat.png
```
