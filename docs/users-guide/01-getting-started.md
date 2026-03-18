# Getting Started

## Installation

PGC is a pure Python package with JIT compilation at runtime — no build step.

```bash
# Base install (numpy only, no backend)
pip install pgc

# With CPU backend (LLVM JIT)
pip install 'pgc[cpu]'

# Or install from source with uv
git clone <repo-url>
cd pgc
uv sync
```

## Your First Kernel

```python
import numpy as np
import pgc

pgc.init(arch=pgc.cpu)

# Create fields from numpy arrays
n = 1024
x = pgc.field_like(np.arange(n, dtype=np.float32))
y = pgc.field_like(np.ones(n, dtype=np.float32) * 2.0)
out = pgc.field(dtype=pgc.f32, shape=(n,))

# Define a kernel
@pgc.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]

# Run it
vector_add(x, y, out)

# Read results back to numpy
result = out.to_numpy()
print(result[:5])  # [2. 3. 4. 5. 6.]
```

A kernel is a Python function decorated with `@pgc.kernel`. The outermost
`for i in range(...)` becomes a parallel loop — each iteration runs as a
separate thread on the GPU (or is split across CPU cores).

## Choosing a Backend

```python
pgc.init(arch=pgc.cpu)         # CPU via LLVM JIT
pgc.init(arch=pgc.metal)       # Apple GPU
pgc.init(arch=pgc.cuda)        # NVIDIA GPU
pgc.init(arch=pgc.hip)         # AMD GPU (ROCm)
pgc.init(arch=pgc.level_zero)  # Intel GPU
```

All examples accept `--arch` on the command line:

```bash
uv run python examples/01_hello_pgc.py --arch metal
```

The same kernel code runs on all backends — PGC handles the compilation
pipeline for each target.

## How It Works

When you call a kernel for the first time, PGC:

1. Reads the Python source of the decorated function
2. Transforms the AST into PGC's internal IR
3. Runs optimization passes (LICM, CSE, copy propagation)
4. Generates backend-specific code (LLVM IR, MSL, CUDA C, etc.)
5. Compiles and dispatches

Subsequent calls with the same argument types reuse the compiled kernel.
