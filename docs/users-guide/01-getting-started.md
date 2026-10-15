# Getting Started

## Installation

Use Python 3.11 or newer. Tack compiles kernels at runtime; installing the
Python packages needs no Tack build step. Start with the CPU backend so the
first example needs no GPU runtime.

### Follow this guide from source

The guide and its tutorial scripts describe the current checkout, including
[unreleased additions](https://github.com/berkgeveci/tack/blob/main/CHANGELOG.md).
To use the same API and run the commands shown throughout the guide, install
the checkout with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/berkgeveci/tack.git
cd tack
uv sync --extra cpu
```

Run Python with `uv run python` from this directory. Both Tack packages are
installed: `tack-core` for compute and `tack-vis` for datasets and scientific
visualization.

### Install into another environment

Tack is not on PyPI yet. For a separate application, create an environment
and install the packages from the clone, `tack-core` first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '/path/to/tack/packages/tack-core[cpu]'
# Optional: datasets and visualization
python -m pip install -e /path/to/tack/packages/tack-vis
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Run scripts
with `python` in the activated environment. Download tutorial scripts from their
pages instead of using checkout-relative paths such as `docs/examples/heat.py`.

An installed release may lack APIs added in the checkout. If an example uses
an unavailable API, check the release notes or use the source installation above.
A bare `tack-core` install does not include the CPU JIT dependency; the `[cpu]`
extra does. The visualization and rendering packages also need a working backend.

## Your First Kernel

Save the following complete program as `first_kernel.py`. Tack needs the
decorated function's source, so a `.py` file is the starting point for this guide.

```python
import numpy as np
import tack

tack.set_default_device("cpu")

# Create arrays from numpy arrays
n = 1024
x = tack.array(np.arange(n, dtype=np.float32))
y = tack.array(np.ones(n, dtype=np.float32) * 2.0)
out = tack.empty((n,))

# Define a kernel
@tack.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]

# Run it
vector_add(x, y, out)

# Read results back to numpy
result = out.numpy()
np.testing.assert_array_equal(result, np.arange(n, dtype=np.float32) + 2.0)
print(result[:5])  # [2. 3. 4. 5. 6.]
```

Run it from the checkout:

```bash
uv run python first_kernel.py
```

Or use `python first_kernel.py` in your installed environment. You should see:

```text
[2. 3. 4. 5. 6.]
```

A kernel is a Python function decorated with `@tack.kernel`. The outermost
`for i in range(...)` becomes a parallel loop — each iteration runs as a
separate thread on the GPU (or is split across CPU cores).

## Choosing a Backend

After the CPU example works, install your backend's extra and replace the
single `tack.set_default_device` line. For a checkout, keep `--extra cpu` if you want both:

| Hardware | Checkout installation | Default device |
|---|---|---|
| CPU | `uv sync --extra cpu` | `tack.set_default_device("cpu")` |
| Apple silicon GPU | `uv sync --extra cpu --extra metal` | `tack.set_default_device("metal")` |
| NVIDIA GPU | `uv sync --extra cpu --extra cuda` | `tack.set_default_device("cuda")` |
| AMD GPU | `uv sync --extra cpu --extra hip` | `tack.set_default_device("hip")` |
| Intel GPU | `uv sync --extra cpu --extra level_zero` | `tack.set_default_device("level_zero")` |

For released packages, use the matching extra, such as `tack-core[cuda]`.
GPU extras also require compatible system runtimes; see [Backends](08-backends.md).

The supplied tutorial scripts accept `--arch` on the command line; the small
`first_kernel.py` above selects its backend in code:

```bash
uv run python packages/tack-core/examples/01_hello_tack.py --arch metal
```

This kernel uses the portable numerical subset and runs on all five backends.
Backend capabilities still matter: Metal has no `f64`, and CPU has no shared
workgroup memory. Initialize before allocating arrays; a later backend switch
does not migrate them.

## How It Works

The [execution-model chapter](12-execution-model.md) develops these ideas through
examples. You can continue there now; the compilation outline below is optional
on a first reading.

When you call a kernel for the first time, Tack:

1. Reads the Python source of the decorated function
2. Transforms the AST into Tack's internal IR
3. Resolves shapes and types, then propagates safe copies
4. Generates backend-specific code (LLVM IR, MSL, CUDA C, etc.)
5. Compiles and dispatches

LLVM and vendor compilers optimize the generated code. Subsequent calls
with the same specialization reuse the compiled kernel; dtypes, vector
widths, baked-in shapes, and template structure can require another variant.
Compiled kernels are also kept on disk, so the next process that calls the
same kernel loads it instead of compiling it; see
[The Kernel Cache](17-kernel-cache.md).

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
# Or keep VTK in the checkout's environment (uv sync would otherwise remove it):
uv sync --extra cpu --extra interop
```
