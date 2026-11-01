# Tack

Tack is a Python framework for data-parallel computing on CPUs and GPUs. You
write a kernel as an ordinary Python function with one parallel loop, and Tack
compiles it when you call it, for the CPU (through LLVM), Apple Metal, NVIDIA
CUDA, AMD HIP or Intel Level Zero. Arrays live on a device, kernels run where
their arrays are, and **the same kernel source runs on every backend**, within
each backend's [capabilities](contracts/backend-capabilities.md): Metal has no
`f64`, for example, and the CPU has no workgroup primitives.

Tack comes in two packages, and the second is built on the first:

- **Tack core** (`tack-core`, the `tack` module): the kernel language, arrays and
  devices, parallel algorithms and communication between devices and MPI ranks.
  It is general purpose: simulations, data processing, analysis, anything you
  would write as loops over arrays.
- **Tack vis** (`tack-vis`, `tack.dataset` and `tack.interop`): datasets and
  scientific visualization -- meshes, point clouds, fields and the filters over
  them -- written as Tack kernels, with VTK, MFEM and Warp interop.

```python
import tack, numpy as np

tack.set_default_device("cpu")          # or metal, cuda, hip, level_zero

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

Only the `tack.set_default_device()` line changes between backends:

=== "CPU"

    ```python
    tack.set_default_device("cpu")
    ```
    LLVM IR through llvmlite, dispatched with ctypes. Threads are used only
    when a measurement says they pay — see [Backends](users-guide/08-backends.md).

=== "Metal"

    ```python
    tack.set_default_device("metal")
    ```
    MSL compiled by the Metal API. Buffers are shared memory on Apple silicon,
    so host and device see the same allocation.

=== "CUDA"

    ```python
    tack.set_default_device("cuda")
    ```
    CUDA C through NVRTC to PTX, launched with `cuLaunchKernel`.

=== "HIP"

    ```python
    tack.set_default_device("hip")
    ```
    HIP C through hipRTC. The code generator is the CUDA one with a different
    `#include` — device syntax is shared.

=== "Level Zero"

    ```python
    tack.set_default_device("level_zero")
    ```
    OpenCL C compiled to SPIR-V by `libocloc`, launched through Level Zero.

## Tack core

A kernel call goes from Python source to native code in one step: Tack reads
the function's source, lowers it to its own intermediate representation,
specializes it for the argument types, generates the backend's code and
compiles it. The compiled kernel is kept in memory and on disk, so later calls,
and later processes, skip the compilation. Around that compiler, the core has
these parts:

| Part | What it gives you | Read |
|---|---|---|
| **The kernel language** | `@tack.kernel` with one parallel loop; `@tack.func` device functions; control flow, math, and integer and floating-point arithmetic with defined semantics on every backend; vectors and matrices; atomics; workgroups (shared memory, barriers, block reductions) on GPUs | [Kernels](users-guide/03-kernels.md), [Control Flow and Math](users-guide/04-control-flow.md), [Device Functions](users-guide/05-device-functions.md), [Vectors and Matrices](users-guide/14-vectors-and-matrices.md), [Advanced Features](users-guide/07-advanced.md) |
| **Templates** | `@tack.device_class` objects: data and methods that a kernel takes as an argument and compiles in, composed of other templates, so one kernel serves many data structures | [Templates](users-guide/06-templates.md) |
| **Arrays** | typed arrays of scalars, vectors or matrices on a device, structure-of-arrays layout, implicit arrays computed from their index (`tack.arrays`), exchange with NumPy | [Arrays and Types](users-guide/02-arrays-and-types.md), [Memory and Parallel Correctness](users-guide/13-memory-and-parallelism.md) |
| **Devices and backends** | devices as values, a default device, several GPUs, launches queued on GPUs and waited for when you read | [Backends](users-guide/08-backends.md) |
| **Parallel algorithms** | `tack.algorithms`: reductions, scans, sorts, segmented reductions and statistics, the building blocks of data-parallel programs | [Reductions and Scans](users-guide/11-reductions-and-scans.md) |
| **Communication** | `tack.communication`: work split into blocks over devices and MPI ranks, with exchanges, ghost plans, routing and distributed sorting | [Communicate Between Blocks](users-guide/18-communication.md) |
| **Interoperability** | sharing memory without copies through DLPack and external pointers, with PyTorch, CuPy, Warp and VTK | [Sharing Arrays](users-guide/16-interoperability.md) |
| **Tools** | the generated code (`tack.inspect`), bounds checks in debug mode, timing, the kernel cache | [Debugging and Timing](users-guide/15-debugging-and-timing.md), [The Kernel Cache](users-guide/17-kernel-cache.md) |

The [User's Guide](users-guide/index.md) teaches these in order: start with
[Getting Started](users-guide/01-getting-started.md) and
[From Python to a Tack program](users-guide/12-execution-model.md), then the
language chapters. Its [worked examples](users-guide/tutorials/index.md) are
complete programs with independent checks: heat diffusion, dye advection,
trail-following agents, marching squares, a particle-to-grid simulation and
tiled N-body forces.

## Tack vis: datasets and visualization

Tack vis brings scientific visualization to Tack: the data model and filters
of [VTK](https://vtk.org), written data-parallel as
[Viskores](https://github.com/Viskores/viskores) writes them, with fields in
function spaces as finite-element codes such as MFEM define them. Its
algorithms -- contour, slice, clip, threshold, gradients, probes,
streamlines, point-cloud cleaning, halo finding and many more -- are Tack
kernels, so they run on every backend, on one dataset or on one cut into
pieces over devices and MPI ranks. Tack draws nothing itself: results go to
VTK and ParaView, sharing memory where they can.

```python
import tack
import tack.dataset as td

data = td.uniform_grid((17, 17, 17), origin=(-1, -1, -1), spacing=(0.125, 0.125, 0.125))
positions = data.positions()
values = tack.array((positions * positions).sum(axis=1), dtype=tack.f32)
data.fields["radius_squared"] = td.Field(td.H1(data), values)

surface = td.contour(data, "radius_squared", 0.5)     # a sphere of triangles
```

Using its filters needs no kernel programming. Writing your own needs the
core's kernels and templates, because a filter of your own sees a mesh
through the same views the built-ins use. The
[Visualization Guide](vis-guide/index.md) goes in this order:

1. **[The guide's opening page](vis-guide/index.md):** what visualization
   does with data, and where Tack vis comes from; then the
   [Introduction](vis-guide/introduction.md), a tour with a runnable example.
2. **The data model:** function spaces, datasets, fields, cell types and
   bases, physical units.
3. **Working with data:** building meshes, point clouds, AMR and octrees,
   the filters, [writing mesh algorithms](vis-guide/05-mesh-kernels.md) and
   [filters of your own](vis-guide/15-write-a-filter.md), sampling and
   connecting simulations, streaming time steps.
4. **Data in pieces:** partitioned datasets with ghost cells, and
   algorithms across pieces, devices and ranks.
5. **[Examples](vis-guide/examples.md)** and the
   [Gallery](vis-guide/gallery.md), then the reference.

## Install

Tack is not on PyPI yet; install it from a clone of the repository:

```bash
git clone https://github.com/berkgeveci/tack.git && cd tack
pip install -e 'packages/tack-core[cpu]'      # Tack core: CPU only
pip install -e 'packages/tack-core[metal]'    # ...or Apple silicon
pip install -e 'packages/tack-core[cuda]'     # ...or NVIDIA
pip install -e packages/tack-vis              # Tack vis, on top of the core
```

See [Backends](reference/backends.md) for what each extra pulls in and which
scalar types each backend supports.

## The documentation

<div class="grid cards" markdown>

-   **[User's Guide](users-guide/index.md)**

    Tack core: writing kernels, arrays and types, control flow, device
    functions, templates, atomics and shared memory, reductions and scans,
    backends, communication and interoperability.

-   **[Visualization Guide](vis-guide/index.md)**

    Tack vis: the data model, meshes, point clouds and field spaces,
    scientific filters, custom cell and face kernels, probing and
    streamlines, data in pieces, and simulation pipelines.

-   **[Developer's Guide](developers-guide/index.md)**

    How a decorated function becomes machine code: the AST transform, the IR
    and its passes, the code generators, and the runtime that caches and
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

    The capability table: codegen, runtime compiler, dispatch, buffer type
    and supported dtypes for each backend.

-   **[API](reference/api.md)**

    The public surface, generated from the source.

</div>

## What Tack is not

It is not a drop-in NumPy replacement and not an autograd framework. Kernels
are written in a restricted subset of Python that is read as source and
compiled — the [User's Guide](users-guide/03-kernels.md) covers what that
subset includes. And it is not a renderer: Tack vis computes what is drawn,
and VTK or ParaView draws it.
