# Backends

Tack compiles the same kernel source to different GPU APIs. Each backend has
its own compilation pipeline and memory model.

## Overview

| Backend | Platform | GPU | Compilation | Dependencies |
|---------|----------|-----|-------------|-------------|
| CPU | All | None (uses CPU cores) | LLVM JIT via llvmlite | `llvmlite` |
| Metal | macOS | Apple Silicon | MSL source → Metal API | `pyobjc-framework-Metal` |
| CUDA | Linux/Windows | NVIDIA | CUDA C → NVRTC → PTX | `cuda-python>=13.2` |
| HIP | Linux | AMD (ROCm) | HIP C → hipRTC | `hip-python` |
| Level Zero | Linux | Intel | OpenCL C → libocloc → SPIR-V | Level Zero runtime |

## Capabilities

Every backend runs the same kernel language, but within its own
capabilities. A kernel that uses something a backend lacks is rejected with
an error naming the kernel and the missing capability. It is never run
with a different meaning.

| | CPU | Metal | CUDA | HIP | Level Zero |
|---|---|---|---|---|---|
| `f64` fields | yes | no | yes | yes | depends on the device |
| Shared memory, barriers, `thread_id`, block reductions | no | yes | yes | yes | yes |
| Atomic field types | all ten | `i32`, `u32`, `f32` | 32- and 64-bit ints, `f32`, `f64` | 32- and 64-bit ints, `f32`, `f64` | `i32`, `u32`, `f32` |
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

## CPU

The CPU backend uses LLVM JIT (via llvmlite) to compile kernels to native
machine code. Work is split across physical CPU cores using a persistent
thread pool.

Threading is not free — a fan-out costs on the order of 200 µs of Python
thread wakeup — so the backend only uses it when the serial run would take
meaningfully longer than that. It measures both sides: the fan-out cost of
the machine it is on, and the per-element cost of each compiled kernel.
That matters because the break-even point depends on how much work the
kernel does per element, and moves by a factor of a thousand between a
memory-bound expression and a compute-heavy one.

Set `TACK_CPU_THREADS`, or pass `tack.init(arch=tack.cpu, num_threads=n)`,
to override the thread count. A thread count of 1 runs everything on the
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
pip install 'tack-core[cpu]'
```

```python
tack.init(arch=tack.cpu)
```

## Metal

Apple Silicon unified memory means fields are zero-copy — the numpy view
and GPU buffer share the same physical memory. No host-device transfers.

```bash
pip install pyobjc-framework-Metal
```

```python
tack.init(arch=tack.metal)
```

Metal has a 31-buffer binding limit per kernel. Tack automatically packs
scalar parameters into constant buffers to stay within this limit.

Hardware 3D texture sampling is supported via `texture3d<float>.sample()`.

## CUDA

NVIDIA GPUs via the CUDA driver API. Fields use device memory (`cuMemAlloc`)
with explicit host-device copies.

```bash
pip install 'cuda-python>=13.2'
```

```python
tack.init(arch=tack.cuda)
```

Tack's CUDA context is current only on the thread that called
`tack.init()`. To call kernels from another thread, make the context
current there first:

```python
from cuda.bindings import driver
from tack.runtime.dispatch import get_backend

driver.cuCtxSetCurrent(get_backend()._context)  # in the other thread
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
tack.init(arch=tack.hip)
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
and some don't. The backend asks the device during `tack.init()` and adds
`f64` to its supported types only when the device reports it. Level Zero field atomics remain limited to `i32`, `u32` and `f32`, even
on a device with `f64` storage. Check `get_backend().supports_f64` rather than
assuming.

**VTK device interop needs a shared context.** A Level Zero pointer is
meaningful only inside the context it was allocated from, so Tack and VTK
must use the same one. Start Tack with
`tack.interop.vtk.init_level_zero()` instead of
`tack.init(arch=tack.level_zero)`. Fields allocated before that call can't
be shared with VTK.

Hardware 3D texture sampling (`image3d_t` with `read_imagef`) is used on
devices with texture units (Xe-HPG/Xe-LPG). Xe-HPC (Ponte Vecchio) falls
back to software trilinear since it has no sampler hardware.

```python
tack.init(arch=tack.level_zero)
```

## Error Messages

If a backend is unavailable, `tack.init()` gives a clear error:

```
RuntimeError: Cannot initialize 'hip' backend: missing dependency.
  No module named 'hip'
  Requires an AMD GPU with ROCm, plus hip-python:
    pip install 'tack-core[hip]'
  ...
```

Kernel compilation errors show the kernel name, backend, and the relevant
error lines without dumping the full generated source. To see the generated
code on any backend, use `tack.inspect(kernel, *args, mode="source")`. On
Metal, `TACK_DUMP_MSL=1` also writes each compiled kernel's MSL to `/tmp`.
Other backends ignore it.

## Type Checking

Tack validates field dtypes at dispatch time before compilation. If a field
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
