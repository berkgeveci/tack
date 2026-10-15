# Tack

A Python framework for scientific data analysis, visualization and GPU compute.
Build pipelines over structured grids, mixed meshes and polyhedra: associate
fields, derive quantities, extract surfaces, sample solutions and render results.
The dataset algorithms are Tack kernels, and you can write your own as decorated
Python functions. **The same supported kernel source runs on five backends**, within
each backend's [capabilities](contracts/backend-capabilities.md). Metal has
no `f64`, for example, and the CPU has no workgroup primitives.

Start with [Datasets and Visualization](users-guide/09-visualization.md) for a
mesh-to-image workflow, or [Getting Started](users-guide/01-getting-started.md)
to write a first kernel. A small compute example is:

```python
import tack, numpy as np

tack.init(arch=tack.cpu)          # or metal, cuda, hip, level_zero

@tack.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):   # the outermost loop maps to thread parallelism
        out[i] = x[i] + y[i]

n = 1_000_000
x = tack.empty((n,))
y = tack.empty((n,))
out = tack.empty((n,))
tack.copyto(x, np.arange(n, dtype=np.float32))
tack.copyto(y, np.ones(n, dtype=np.float32))

vector_add(x, y, out)
print(out.numpy()[:4])         # [1. 2. 3. 4.]
```

Only the `tack.init()` line changes between backends:

=== "CPU"

    ```python
    tack.init(arch=tack.cpu)
    ```
    LLVM IR through llvmlite, dispatched with ctypes. Threads are used only
    when a measurement says they pay — see [Backends](users-guide/08-backends.md).

=== "Metal"

    ```python
    tack.init(arch=tack.metal)
    ```
    MSL compiled by the Metal API. Buffers are shared memory on Apple silicon,
    so host and device see the same allocation.

=== "CUDA"

    ```python
    tack.init(arch=tack.cuda)
    ```
    CUDA C through NVRTC to PTX, launched with `cuLaunchKernel`.

=== "HIP"

    ```python
    tack.init(arch=tack.hip)
    ```
    HIP C through hipRTC. The code generator is the CUDA one with a different
    `#include` — device syntax is shared.

=== "Level Zero"

    ```python
    tack.init(arch=tack.level_zero)
    ```
    OpenCL C compiled to SPIR-V by `libocloc`, launched through Level Zero.

## Install

```bash
pip install 'tack-core[cpu]'      # CPU only
pip install 'tack-core[metal]'    # + Apple silicon
pip install 'tack-core[cuda]'     # + NVIDIA
```

See [Backends](reference/backends.md) for what each extra pulls in and which
scalar types each backend supports.

## Where to go next

<div class="grid cards" markdown>

-   **[Datasets and Visualization](users-guide/09-visualization.md)**

    Meshes and field spaces, scientific filters, custom cell and face kernels,
    probing and streamlines, then simulation and rendering pipelines. Start
    with complete programs whose results are checked independently.

-   **[User's Guide](users-guide/index.md)**

    Writing kernels, arrays and types, control flow, device functions,
    templates, atomics and shared memory, reductions and scans, alongside
    the data and visualization chapters.

-   **[Developer's Guide](developers-guide/index.md)**

    How a decorated function becomes machine code: the AST transform, the IR
    and its passes, the five code generators, and the runtime that caches and
    dispatches them.

-   **[Design and Implementation](design/index.md)**

    Why Tack is built the way it is: the principles, the compilation
    pipeline, specialization and caching, numerical semantics, parallel
    execution, memory and aliasing, and interoperability.

-   **[Contracts](contracts/index.md)**

    What Tack promises and what callers must ensure: the kernel language
    contract, backend capabilities, the runtime API and its errors, and how
    conformance is tested and validated.

-   **[Backends](reference/backends.md)**

    The capability table — codegen, runtime compiler, dispatch, buffer type
    and supported dtypes for each of the five.

-   **[API](reference/api.md)**

    The public surface, generated from the source.

</div>

## What Tack is not

It is not a drop-in numpy replacement and not an autograd framework. Kernels
are written in a restricted subset of Python that is read as source and
compiled — the [User's Guide](users-guide/03-kernels.md) covers what that
subset includes.
