# Backends

Tack compiles the same kernel source to different GPU APIs. Each backend has
its own compilation pipeline and memory model.

## Overview

| Backend | Platform | GPU | Compilation | Dependencies |
|---------|----------|-----|-------------|-------------|
| CPU | All | None (uses CPU cores) | LLVM JIT via llvmlite | `llvmlite` |
| Metal | macOS | Apple Silicon | MSL source → Metal API | `pyobjc-framework-Metal` |
| CUDA | Linux/Windows | NVIDIA | CUDA C → NVRTC → PTX | `cuda-python>=13.2`, NVRTC 13 |
| HIP | Linux | AMD (ROCm) | HIP C → hipRTC | `hip-python` |
| Level Zero | Linux | Intel | OpenCL C → libocloc → SPIR-V | Level Zero runtime |

Every backend keeps what it compiles on disk, so a later process does not
compile the same kernel again. See [The Kernel Cache](17-kernel-cache.md).

## Devices

A device is where arrays live and kernels run. `tack.device("cuda")`
returns one: the spec is `"cpu"`, `"metal"`, `"cuda"`, `"hip"` or
`"level_zero"`, and the numbered types also take `"cuda:1"`: the second
CUDA GPU. Asked for twice, it is the same device, with the kernels it has
compiled.

- `tack.set_default_device("cuda")` makes it the default device: where
  arrays are allocated, and where kernels with no array arguments run.
  With nothing set, the default is the CPU. `tack.get_default_device()`
  says which it is.
- `with tack.device("cuda"):` makes it the default inside the block, on
  that thread only. Blocks nest.
- Every array constructor takes `device=`, and every array records its
  device in `a.device`. `a.to_device(d)` returns the array itself if its
  memory is usable on `d`, else a copy there. `empty_like` and the other
  `*_like` constructors allocate on the prototype's device.
- A kernel runs where its arrays are: on the default device when that can
  address them, otherwise on the arrays' own device. Arrays on two devices
  in one call raise `ValueError`; move one with `to_device` first. Copies,
  reductions and `tack.algorithms` functions run on their input's device.
- Switching the default doesn't invalidate anything. Arrays made before
  the switch keep their device and keep working.
- `tack.sync()` waits for every device; `d.synchronize()` waits for one.

On a machine with several GPUs, `tack.device_count("cuda")` says how many
there are (also for `"hip"` and `"level_zero"`), and `tack.device("cuda:1")`
is the second. A bare `"cuda"` or `"hip"` is the GPU the thread already
uses, as a framework that embeds Tack may have chosen, else the first.
Each device compiles its own kernels, and a kernel's arrays must all be on
one: `b = a.to_device(tack.device("cuda:1"))` copies through the host. To
split one problem over several devices, or MPI ranks, see
[Communicate Between Blocks](18-communication.md).

Options make a separate device: `tack.device("cpu", num_threads=1)` is a
second CPU device with one thread. It shares memory with the default CPU
device, so it runs kernels on CPU arrays made before it.

<!-- tack-check: devices -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def scale(x, out, factor):
    for i in range(x.shape[0]):
        out[i] = x[i] * factor


x = tack.arange(8, dtype=tack.f32)
out = tack.empty_like(x)            # on x's device
scale(x, out, 2.0)                  # runs where x and out are
assert x.device == tack.get_default_device() and str(x.device) == "cpu"

serial = tack.device("cpu", num_threads=1)
with serial:                        # the default device inside the block
    y = tack.ones(8)
    scale(x, out, 3.0)              # serial can address CPU arrays: runs there
assert y.device == serial
assert x.to_device(serial) is x     # same memory, so no copy
np.testing.assert_allclose(out.numpy(), np.arange(8) * 3.0)
```

On a machine with a GPU the same calls move data to it:

```python
gpu = tack.device("cuda")
grid = tack.zeros(1024, device=gpu) # allocated on the GPU
b = x.to_device(gpu)                # a copy of the CPU array
c = tack.empty_like(b)              # on b's device
scale(b, c, 2.0)                    # runs on the GPU, whatever the default
```

## Capabilities

Every backend runs the same kernel language, but within its own
capabilities. A kernel that uses something a backend lacks is rejected with
an error naming the kernel and the missing capability. It is never run
with a different meaning.

| | CPU | Metal | CUDA | HIP | Level Zero |
|---|---|---|---|---|---|
| `f64` arrays | yes | no | yes | yes | depends on the device |
| Shared memory, barriers, `thread_id`, block reductions | no | yes | yes | yes | yes |
| Atomic array types | all ten | `i32`, `u32`, `f32` | 32- and 64-bit ints, `f32`, `f64` | 32- and 64-bit ints, `f32`, `f64` | `i32`, `u32`, `f32` |
| `sum`/`min`/`max` on the device | no (NumPy) | `f32` | `f32` | `f32` | `f32` |
| 3D textures | software | hardware | hardware | hardware where the device has image support | hardware where the device has samplers |
| Iterations per launch | no limit | 2^32 | device limit (max grid × 256) | device limit, at most 2^32 − 256 | device limit (max group count × group size) |

A launch past a backend's limit raises `ValueError` naming the kernel, the
count and the limit; split the work across several calls.

The [backend capability contract](../contracts/backend-capabilities.md) has
the full matrix, what each capability rejects and when, and the subset that
is portable to all five. In code, ask the backend rather than checking its
name:

```python
from tack.runtime.dispatch import get_backend
be = get_backend()
be.supports_f64, be.supports_workgroups, be.supported_atomic_dtypes
```

## When a call returns

On the CPU a kernel call returns when the kernel has finished. On Metal,
CUDA, HIP and Level Zero it returns once the launch is queued; the device
runs queued launches in order, and fills and host writes take their place
in that order. The host waits where it reads an array (`numpy()`,
indexing, reductions, DLPack export) or calls `tack.sync()`. Most code
never needs `tack.sync()`: it is for timing kernels (see
[Debugging and timing](15-debugging-and-timing.md#measure-first-call-and-warm-time-separately)),
and for memory another library reads without Tack. An error in a queued
launch is raised at that wait rather than at the call. A launch that uses
memory shared with another library (exported through DLPack, or wrapped
with `tack.array_from_ptr`) completes before the call returns. In an
`asyncio` program, `await tack.sync_async()` and
`await array.numpy_async()` do the waiting in a worker thread, so the
event loop keeps running (see
[Waiting from `asyncio`](12-execution-model.md#waiting-from-asyncio)).

## CPU

The CPU backend uses LLVM JIT (via llvmlite) to compile kernels to native
machine code. Work is split across physical CPU cores using a persistent
thread pool. The pool is itself JIT-compiled from LLVM IR, so a fan-out is
one call that never enters Python; `TACK_CPU_POOL=python` uses a
`ThreadPoolExecutor` instead, which is also the fallback on Windows or
where the pool cannot be started.

Threading is not free — a fan-out costs a few microseconds when the
workers are warm and about a hundred after they have been idle, and the
work between launches stays on the calling thread — so the backend only
uses it when the serial run would take meaningfully longer than that. It measures both sides: the fan-out cost of
the machine it is on, and the per-element cost of each compiled kernel.
That matters because the break-even point depends on how much work the
kernel does per element, and moves by a factor of a thousand between a
memory-bound expression and a compute-heavy one.

Set `TACK_CPU_THREADS`, or use `tack.device("cpu", num_threads=n)`, to
override the thread count. A thread count of 1 runs everything on the
calling thread, which is useful when profiling or
when Tack is embedded in a host that manages its own threads.

The CPU has no workgroup execution model, and Tack doesn't emulate one.
Kernels that use `tack.shared`, `tack.shared_like`, `tack.barrier`,
`tack.thread_id`, `tack.block_sum`, `tack.block_min` or `tack.block_max`
are **rejected on CPU** before anything is compiled or run, even when the
primitive sits in a branch that never executes. A call raises
`RuntimeError`, and `tack.inspect` raises `NotImplementedError`. For
per-iteration scratch storage, use
[`tack.local_array` or `tack.local_array_like`](07-advanced.md#local-arrays),
which work on every backend.

```bash
uv sync --extra cpu      # or, in another environment: pip install -e 'packages/tack-core[cpu]'
```

```python
tack.set_default_device("cpu")
```

## Metal

On Apple silicon the GPU buffer and Tack's internal NumPy view share the
same allocation. There is no separate device-memory transfer, but the public
`tack.array`, `tack.copyto` and `numpy()` calls still copy values. Use the
[interoperability APIs](16-interoperability.md) when you need shared storage.

```bash
uv sync --extra metal    # or: pip install -e 'packages/tack-core[metal]'
```

```python
tack.set_default_device("metal")
```

Metal has a 31-buffer binding limit per kernel. Tack automatically packs
scalar parameters into constant buffers to stay within this limit.

Hardware 3D texture sampling is supported via `texture3d<float>.sample()`.

## CUDA

NVIDIA GPUs via the CUDA driver API. Arrays use device memory (`cuMemAlloc`)
with explicit host-device copies.

```bash
pip install 'cuda-python>=13.2'
```

```python
tack.set_default_device("cuda")
```

Tack works in the device's **primary context**, the one the CUDA runtime
and the libraries on it (PyTorch, CuPy, Warp) use. So Tack and those
libraries can take turns on one thread in any order, and arrays pass between
them through DLPack without a context switch. If a context is already
current when Tack first makes the device (an embedding application's own),
Tack adopts that one instead. Tack retains the primary context and, when it
is done with it, releases it rather than destroying it, since other
libraries may hold it too.

Tack makes the context current on the thread that first asks for the device
(`tack.device("cuda")`, `set_default_device` or `device=`). To call kernels
from another thread, that thread needs it current too. A thread on which
PyTorch, CuPy or the CUDA runtime already used the device has it; otherwise
make it current first:

```python
from cuda.bindings import driver

_, cu_device = driver.cuDeviceGet(0)
_, primary = driver.cuDevicePrimaryCtxRetain(cu_device)   # in the other thread
driver.cuCtxSetCurrent(primary)
```

Calls of one kernel from several threads are safe on every backend: each
call runs with its own arguments. On GPU backends calls of the same
compiled kernel take turns.

## HIP

AMD GPUs via ROCm. The codegen extends CUDA — HIP device code uses the same
syntax (`blockIdx`, `threadIdx`, `__global__`).

```bash
uv sync --extra hip        # hip-python is on PyPI
```

```python
tack.set_default_device("hip")
```

**ROCm 7.0.2 and 7.1.1 miscompile some integer code.** Their device
compiler, AMD clang 20 inside hipRTC, can evaluate a 64-bit signed comparison wrongly
when it sits inside a long integer expression. The kernel then returns a
wrong value with no error. Tack's own tests found one case: a generated
expression in which `-253 < -1` evaluated false at `-O1` and above. The
same source is correct on CUDA, as host C++, and under ROCm's clang 23.
Whether a kernel is affected depends on the surrounding expression, so
Tack can't rewrite around it. Use a ROCm release newer than 7.1 if you
can. Go by the ROCm release, not `hiprtcVersion()`, which reports 9.0 on
7.0.2. On 7.0 and 7.1, check integer-heavy kernels against the CPU backend.

## Level Zero

Intel GPUs via the Level Zero API. OpenCL C source is compiled to SPIR-V
in-process using `libocloc`, then loaded via `zeModuleCreate`.

**`f64` depends on the device.** Some Intel GPUs implement double precision
and some don't. The backend asks the device when it starts and adds
`f64` to its supported types only when the device reports it. Level Zero array atomics remain limited to `i32`, `u32` and `f32`, even
on a device with `f64` storage. Check `tack.device("level_zero")._backend.supports_f64` rather
than assuming.

**VTK device interop needs a shared context.** A Level Zero pointer is
meaningful only inside the context it was allocated from, so Tack and VTK
must use the same one. Start Tack with
`tack.interop.vtk.init_level_zero()` instead of
`tack.set_default_device("level_zero")`; it returns the Level Zero device,
in VTK's context, that it made the default. Arrays on another Level Zero
device can't be shared with VTK, and can't be mixed with this device's in
one kernel call.

Hardware 3D texture sampling (`image3d_t` with `read_imagef`) is used on
devices with texture units (Xe-HPG/Xe-LPG). Xe-HPC (Ponte Vecchio) falls
back to software trilinear since it has no sampler hardware.

```python
tack.set_default_device("level_zero")
```

## Error Messages

If a backend is unavailable, `tack.device()` gives a clear error:

```
RuntimeError: Cannot start the 'hip' device: missing dependency.
  No module named 'hip'
  Requires an AMD GPU with ROCm, plus hip-python; in a clone:
    pip install -e 'packages/tack-core[hip]'   (or: uv sync --extra hip)
  ...
```

Kernel compilation errors show the kernel name, backend, and the relevant
error lines without dumping the full generated source. To see the generated
code on any backend, use `tack.inspect(kernel, *args, mode="source")`. On
Metal, `TACK_DUMP_MSL=1` also writes each compiled kernel's MSL to `/tmp`.
Other backends ignore it.

## Type Checking

Tack validates array dtypes at dispatch time before compilation. If an array
uses a dtype not supported by the target backend, you get a clear error:

```
TypeError: Kernel 'my_kernel': parameter 'data' has dtype tack.f64, which is
not supported on Metal. Supported dtypes: tack.f32, tack.i16, tack.i32,
tack.i64, tack.i8, tack.u16, tack.u32, tack.u64, tack.u8
```

`tack.inspect` applies the same check.

Supported dtypes per backend:

| Backend | Supported dtypes |
|---------|-----------------|
| CPU | i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 |
| Metal | i8, u8, i16, u16, i32, u32, i64, u64, f32 (no f64) |
| CUDA | i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 |
| HIP | i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 |
| Level Zero | i8, u8, i16, u16, i32, u32, i64, u64, f32, plus f64 on devices that report it |
