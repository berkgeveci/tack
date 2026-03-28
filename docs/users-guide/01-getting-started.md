# Getting Started

## Installation

Tack is a pure Python package with JIT compilation at runtime — no build step.
It is split into three packages:

- **tack-core** — the compute framework (kernels, fields, types, backends)
- **tack-rendering** — path tracing renderer
- **tack-vis** — scientific visualization algorithms (flying edges, VTK interop)

```bash
# Install everything from source with uv
git clone <repo-url>
cd tack
uv sync

# Or install individual packages
pip install tack-core          # core only
pip install tack-core[cpu]     # core + CPU backend (LLVM JIT)
pip install tack-rendering     # rendering (pulls in tack-core)
pip install tack-vis           # visualization (pulls in tack-core)
```

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

The same kernel code runs on all backends — Tack handles the compilation
pipeline for each target.

## How It Works

When you call a kernel for the first time, Tack:

1. Reads the Python source of the decorated function
2. Transforms the AST into Tack's internal IR
3. Runs optimization passes (LICM, CSE, copy propagation)
4. Generates backend-specific code (LLVM IR, MSL, CUDA C, etc.)
5. Compiles and dispatches

Subsequent calls with the same argument types reuse the compiled kernel.
