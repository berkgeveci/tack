# Backend capabilities

Tack compiles one kernel source for five targets, but the targets don't
all offer the same things. Metal has no `f64`, the CPU has no workgroups,
and atomics are narrower on some GPUs than ordinary arrays. This page is
the contract for those differences. It covers what a backend declares, what
each declaration means, what each one rejects and when, and the subset of
the language that runs everywhere.

All statements describe release candidate `23d6e1d`, updated for the
queued launches, disk cache, debug mode and CPU pool merged through
`b94af23`, and for the device model of `bba0846`. The declarations live
in `packages/tack-core/src/tack/runtime/backend.py` (class `Backend`) and in
the five subclasses in `runtime/cpu.py`, `runtime/metal.py`,
`runtime/cuda_backend.py`, `runtime/hip_backend.py` and
`runtime/level_zero_backend.py`. For the kernel semantics that these
capabilities gate, see the
[kernel language contract](../reference/language-contract.md). For the
implementation story, see
[Backend Implementations](../design/backend-implementations.md).

## The `Backend` contract

**Required:** every backend subclasses `Backend` and *declares* its
capabilities as attributes. Callers read those attributes and don't probe
for methods with `hasattr` or compare backend class names. Anything that
can be derived is derived, so two declarations can't disagree.

### Required methods

| Method | Contract |
|---|---|
| `allocate_array(dtype, shape, exportable=False)` | Allocates backend storage and returns a `DeviceBuffer`. `exportable=True` asks for memory that `Array.export_memory()` can share without a copy. Only CUDA acts on the flag, and the other backends ignore it |
| `wrap_ptr(ptr, dtype, shape)` | Wraps existing memory as a `DeviceBuffer` without copying or taking ownership. What counts as `ptr` is backend-specific. See [Pointer interop](runtime-api.md#pointer-interop) |
| `execute(kernel, args, kwargs)` | Compiles or reuses a variant and runs it. All five backends delegate to `resolve_variant()` in `runtime/kernel_utils.py`, which is where every capability check below runs |

The base class raises `NotImplementedError` for all three.

### Optional methods with defaults

| Method | Default | Overridden by |
|---|---|---|
| `synchronize()` | Does nothing, which is correct for a backend whose launches complete before `execute()` returns. Otherwise it waits for every launch, fill and copy queued so far, and a queued launch's error is raised here. `tack.sync()` calls it on every backend, `Device.synchronize()` on one | Metal, CUDA, HIP and Level Zero, which queue launches (see [Launches](#launches)) |
| `fill_array(array, value)` | Returns `False`, and `Array.fill` fills the buffer in place | CPU, which runs a fill of 256 KiB or more of its own buffers as a kernel through its dispatch, so the threads first-touch the pages |
| `cache_target()` | Returns `None`: compiled variants are not stored on disk, only lowered IR. Otherwise a string naming what compiled code depends on besides the kernel (the device, the compiler and driver versions, the compile options) | all five |
| `variant_artifact(payload)`, `load_variant(ir_func, artifact, effective_args)` | Raise `NotImplementedError`. Called only when `cache_target()` is not `None`: the picklable part of a compiled variant, and the payload rebuilt from it in another process. `effective_args` lets a GPU backend remake its scalar pack buffers | all five |
| `memory_space(ptr)` | Returns `"cpu"`, which is correct for backends whose allocations are host-addressable | CUDA, HIP and Level Zero, which ask their driver. CPU overrides it with the same answer |
| `reduce_array(array, op)` | Raises `NotImplementedError`. Called only when `supports_device_reductions` is true | Metal, CUDA, HIP and Level Zero |
| `texture_in_hardware(shape_3d)` | Returns `False`: a `Texture3D` keeps a private `f32` array and the generated code interpolates it in software | Metal and CUDA (always `True`), HIP and Level Zero (per device and extent) |
| `create_texture_image(shape_3d)` | Raises `NotImplementedError`. Called only when `texture_in_hardware()` is true. Returns an image object with `upload(array)`, owned by one `Texture3D` | Metal, CUDA, HIP and Level Zero |
| `_store_texture_shapes(ir_func, args)` | Records each `Texture3D` argument's extents on its parameter | HIP and Level Zero, which clear the parameter's `_is_texture` flag when `texture_in_hardware()` is false |

### Attributes

| Attribute | Kind | Meaning |
|---|---|---|
| `name` | declared | Device type, as a device spec writes it (`tack.device("cuda")`) and `Device.type` reports it: `cpu`, `metal`, `cuda`, `hip` or `level_zero` |
| `ordinal` | declared | Which of the type's devices the backend drives (`Device.index`). Always 0 for now |
| `display_name` | declared | How messages spell the backend, if different from `name` |
| `label` | derived | `display_name or name`. Use it in messages |
| `supported_dtypes` | declared | A `frozenset` of the scalar types accepted as array argument dtypes. Dispatch checks every array argument against this set |
| `supports_f64` | derived | `f64 in supported_dtypes` |
| `supports_device_reductions` | declared | True when `Array.sum()`, `min()` and `max()` can run on the device. When false they run in NumPy on the host. It selects a path and never rejects anything |
| `supports_workgroups` | declared | Native workgroup execution for `shared`, `shared_like`, `barrier`, `thread_id` and `block_sum`/`block_min`/`block_max`. It does **not** promise uniform participation, atomic support or safe partial workgroups |
| `supported_atomic_dtypes` | derived | Array dtypes that support `atomic_add`, `atomic_min` and `atomic_max`. This is the backend's entry in `ATOMIC_DTYPES` (`lang/atomic_support.py`) intersected with `supported_dtypes` |
| `device_options` | declared | Keyword options that `tack.device()` forwards to the constructor. `tack.device()` rejects any other option |
| `memory_domain` | derived | What the backend's arrays are valid in, `(name, ordinal)`. Level Zero adds its context handle. A backend runs kernels on another backend's arrays when the domains are equal (two CPU devices with different thread counts); arrays from two domains in one launch are refused |
| `device_memory_spaces` | declared | `memory_space()` results that `array_from_ptr()` accepts. When the set is non-empty, every pointer must be an address (anything `int()` accepts) and is checked. An empty set means that the backend doesn't distinguish memory spaces and nothing is checked |
| `dlpack_refusal_note` | declared | Text appended to the error when a DLPack import is refused, for backends whose reason the device type doesn't convey |

Because `supported_atomic_dtypes` is an intersection, a device without
`f64` also loses `f64` atomics. That matters on Level Zero, where
`supported_dtypes` depends on the device.

## Capability matrix

Read from the backend classes at `23d6e1d`. "All ten" means `i8`, `u8`,
`i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32` and `f64`.

| | CPU | Metal | CUDA | HIP | Level Zero |
|---|---|---|---|---|---|
| `name` / `label` | `cpu` / CPU | `metal` / Metal | `cuda` / CUDA | `hip` / HIP | `level_zero` / Level Zero |
| `supported_dtypes` | All ten | All except `f64` | All ten | All ten | All except `f64`, plus `f64` when the device reports it |
| `supports_f64` | yes | no | yes | yes | per device |
| `supports_device_reductions` | no (NumPy) | yes | yes | yes | yes |
| `supports_workgroups` | no | yes | yes | yes | yes |
| `supported_atomic_dtypes` | All ten | `i32`, `u32`, `f32` | `i32`, `u32`, `i64`, `u64`, `f32`, `f64` | `i32`, `u32`, `i64`, `u64`, `f32`, `f64` | `i32`, `u32`, `f32` |
| `device_options` | `num_threads` | none | none | none | `external_context` |
| `device_memory_spaces` | none (unchecked) | none (unchecked) | `cuda`, `cuda_pinned`, `cuda_managed` | `hip`, `hip_pinned`, `hip_managed` | `level_zero` |
| `dlpack_refusal_note` | none | set | none | none | none |
| 3D textures | software trilinear over a private copy | hardware | hardware | hardware if the device has image support and the extents fit, else software | hardware if the device has samplers and the extents fit, else software |
| Largest launch (iterations) | no limit | 2^32 | max `gridDim.x` × 256 | max `gridDim.x` × 256, capped at 2^32 − 256 | `maxGroupCountX` × the workgroup size |
| `Array.export_memory()` | no | yes (`mtl_buffer`) | yes (OS handle) | no | no |
| DLPack export device | `kDLCPU` | `kDLCPU` (host view of unified memory) | `kDLCUDA` | `kDLROCM` | `kDLOneAPI` |
| DLPack zero-copy import | `kDLCPU`, `kDLCUDAHost`, `kDLROCMHost` | none in practice (see [DLPack](runtime-api.md#dlpack)) | `kDLCUDA`, `kDLCUDAManaged` | `kDLROCM` | `kDLOneAPI` |
| A kernel call returns | after the range has run | once queued | once queued | once queued | once queued (with a driver older than API 1.6, after it completes) |
| Disk-cache artifact | object code | MSL source | PTX | code object | native binary, with SPIR-V to fall back on |
| Device chosen | host | `MTLCreateSystemDefaultDevice()` | the current CUDA context if one exists, else device 0 | `hipGetDevice()` | the first device of the first driver, or the one in `external_context` |
| Python requirements | `llvmlite` (`[cpu]` extra) | `pyobjc-framework-Metal` (`[metal]`) | `cuda-python>=13.2.0` (`[cuda]`) | `hip-python>=7.0`, Linux x86_64 only (`[hip]`) | none. The `[level_zero]` extra is empty |
| System requirements | none | macOS with a Metal GPU | NVIDIA driver and NVRTC (13 or newer; older warns at the first compile) | ROCm with hipRTC | `libze_loader.so` and `libocloc.so` (Intel compute runtime) |

### Level Zero and `f64`

**Current behavior:** `LevelZeroBackend._query_device()` runs during
construction and calls `zeDeviceGetModuleProperties`. It then *shadows* the
class attribute with an instance `frozenset`: all ten types when the
reported `fp64flags` is nonzero, else the nine non-`f64` types. `supports_f64` and
`supported_atomic_dtypes` follow automatically. To find out what the
current device supports, ask:

```python
import tack
from tack.runtime.dispatch import get_backend

tack.set_default_device("level_zero")
be = get_backend()
print(be.supports_f64, sorted(t.name for t in be.supported_atomic_dtypes))
```

### Textures

On every backend, a texture samples the data its array held when
`tack.texture3d()` created it, or when `tex.update()` last copied it (see
[Memory and aliasing](../reference/language-contract.md#memory-and-aliasing)
in the language contract). Writes to the array in between, whether by kernels,
`tack.copyto`, or external and DLPack writers, don't reach the texture.
The copy is storage the `Texture3D` owns (`tex._storage`): a hardware image
from `create_texture_image()` where `texture_in_hardware()` is true, else a
private `f32` array. Dispatch binds that storage through `bind_textures()`
(`runtime/kernel_utils.py`), never the source array. No compiled variant
caches images, and an image is released with its texture. Two textures over
one array are separate copies.

`tack.texture3d()` raises `ValueError` for an array that isn't `f32`,
because every hardware path builds a single-channel 32-bit float image
(`R32Float` on Metal, `CU_AD_FORMAT_FLOAT` on CUDA and HIP,
`ZE_IMAGE_FORMAT_TYPE_FLOAT` on Level Zero). It also rejects an `interp`
other than `"linear"`, since sampling is trilinear on every backend, and a
shape whose `W*H*D` differs from the array's size.

The variant key records which texture path a call uses, because the path
changes the generated code. HIP and Level Zero answer
`texture_in_hardware()` per device and extent, and their
`_store_texture_shapes()` asks it again before the key is built, so a
texture always holds the storage its compiled kernel binds:

- **HIP** uses hardware sampling when `hipDeviceAttributeImageSupport` is
  nonzero and every extent is within the smallest reported maximum 3D
  texture extent. If any reported maximum is non-positive, the extents are
  treated as unbounded. CDNA parts such as MI300 report no image support,
  so they sample in software.
- **Level Zero** uses hardware sampling when the device reports
  `maxSamplers > 0` and every extent is at most `maxImageDims3D`. Xe-HPC
  (Ponte Vecchio) reports no samplers.

Metal and CUDA always answer true. CPU uses the base class's false.

## Launches

**Current behavior:** the CPU's kernel call returns once the range has run.
Metal, CUDA, HIP and Level Zero queue the launch and return: Metal into an
open command buffer, CUDA on one stream per context, HIP on one stream per
device, Level Zero on one asynchronous immediate command list with a
barrier after each command (a driver without `zeCommandListHostSynchronize`
gets a synchronous list). On every backend, launches, fills and host
writes take effect in the order they were made, host reads wait for the
launches before them, an array dropped while a queued launch uses it
outlives that launch, and a launch that binds memory handed outside Tack
(a wrapped pointer, a DLPack export) completes before it returns. CUDA,
HIP and Level Zero also complete a launch that samples a hardware texture.
A queued launch's error is raised at the next wait. The user-facing rules
are in [Runtime API](runtime-api.md#host-operations); the implementation
is in [Backend Implementations](../design/backend-implementations.md#launches-are-queued).

## What each capability gates

All dispatch-time checks run inside `resolve_variant()`. "Cold" means the
first call with a new variant key, before optimization, code generation and
compilation. "Every dispatch" also covers cached calls, and those checks run
before launch.

| Capability | Checked by | When | Rejection |
|---|---|---|---|
| `device_options` | `tack.device()` in `runtime/dispatch.py` | When a device is made with options, after the backend module imports and before construction | `ValueError`: "The '`metal`' device does not accept the option(s) `num_threads`. Accepted: none." The default device is unchanged |
| `supported_dtypes` | `check_dispatch_types()` in `lang/type_inference.py` | Cold dispatch and inspection. A new dtype always produces a new key | `TypeError` naming the kernel once: "Kernel '`k`': parameter '`data`' has dtype `tack.f64`, which is not supported on Metal. Supported dtypes: ..." |
| `supports_workgroups` | `check_workgroup_support()` in `lang/workgroup_support.py` | Every dispatch, before the key lookup. The feature scan is memoized on the IR template. Also runs during inspection and direct LLVM generation | `NotImplementedError` naming the kernel, backend and primitives, with the hint "use `local_array` or `local_array_like`". Dispatch wraps it in `RuntimeError` ("Kernel '`k`' failed on CPUBackend: ..."). Inspection raises it directly |
| Workgroup participation | `check_workgroup_participation()` in `lang/workgroup_participation.py` | Cold dispatch and inspection, on backends with `supports_workgroups` | `ValueError`: "Kernel '`k`': `barrier` at `<IR path>` requires uniform workgroup participation: ..." |
| Full 256-lane groups | `check_workgroup_launch()` | Cold dispatch, every cached dispatch, and inspection, for kernels with barriers or block reductions | `ValueError`: "...workgroup collectives require complete 256-lane groups; iteration count `N` would create a partial workgroup." |
| Device group size | `CompiledMetalKernel` and `CompiledL0Kernel` | At compilation, for collective kernels, when the Metal pipeline or Level Zero device allows fewer than 256 lanes | `ValueError`: "...workgroup collectives require 256 lanes; this pipeline/device selects `n`." |
| `supported_atomic_dtypes` | `check_atomic_support()` in `lang/atomic_support.py` | Cold dispatch and inspection in every mode, including structurally unreachable atomics and empty grids | `TypeError`: "Kernel '`k`': `metal` atomic_add does not support target dtype `tack.i64`". The same check rejects targets that aren't global array parameters |
| Atomic alignment | `check_atomic_alignment()` | Every dispatch and inspection | `ValueError`: "Kernel '`k`': atomic target parameter `i` requires `n`-byte alignment" |
| Read-only arrays | `check_writable_arrays()` in `runtime/kernel_utils.py`, over the indices the variant recorded in `KernelVariant.written_arrays` | Every dispatch | `ValueError`: "Kernel '`k`': parameter '`out`' may be written, but its array is read-only. ..." |
| Launch size | `check_launch_size()` in `runtime/kernel_utils.py`, against the limit in the [capability matrix](#capability-matrix) | Every GPU dispatch, and every native `f32` reduction on CUDA, HIP and Level Zero | `ValueError`: "Kernel '`k`': `N` iterations exceed the `M` that one CUDA launch can index; split the work across several launches." A reduction is named "Array sum()" and so on. Metal reduces arrays of 2^32 or more elements through NumPy instead |
| `device_memory_spaces` | `array_from_ptr()` in `lang/array.py` | When wrapping, for any `ptr`, on backends with a non-empty set | `TypeError` if `ptr` isn't an address. `ValueError`: "Pointer is in '`cpu`' memory but the device is '`CUDA`'. array_from_ptr() requires a device pointer. ..." |
| DLPack device type | `dlpack_to_array()` in `lang/dlpack.py` | On import, when `device=` names a device that cannot wrap the tensor (without it, the tensor's own device is used) | `RuntimeError` naming the device and DLPack device type, followed by `dlpack_refusal_note` if set. `ValueError` for an unknown device type and for `kDLMetal` |
| `supports_device_reductions` | `Array._reduce()` | Never rejects | Selects the device or NumPy path. See [Reductions](runtime-api.md#reductions) |
| Texture hardware | `texture_in_hardware()`, at `texture3d()` and in `_store_texture_shapes()` (HIP, Level Zero) | At texture creation, and before the key is built | Never rejects. Falls back to software sampling |

A `TypeError` raised during dispatch reaches the caller prefixed with
`Kernel '<name>': ` unless its message already names the kernel, and with
the original exception chained. `ValueError` passes through unchanged. See
[Exceptions](runtime-api.md#exceptions).

For the full text of the workgroup and atomic rules, see
[Workgroups and synchronization](../reference/language-contract.md#workgroups-and-synchronization)
and [Atomic array updates](../reference/language-contract.md#atomic-array-updates).

!!! note "What inspection does not check"

    `tack.inspect()` runs the dtype, workgroup, participation,
    workgroup-launch, atomic and alignment checks above, and it makes the
    texture decision through the backend's `_store_texture_shapes()`, so
    the source it shows is the source dispatch would compile. It doesn't
    check read-only arrays or the backend's launch-size limit, which
    belong to a launch rather than to the generated code.

### Device reductions

`supports_device_reductions` does not mean that every reduction runs on the
device. **Current behavior:** on every GPU backend, `reduce_array()` reduces
only `f32` arrays natively, and other dtypes use the shared NumPy fallback.
Level Zero also falls back when the device's maximum group size is below
256. Empty arrays return without a device launch. CUDA, HIP and Level Zero
pass the element count as a 64-bit kernel argument and reject an array
larger than one launch can index. Metal's reduction kernel holds the count
and thread position in 32 bits, so Metal reduces an array of 2^32 or more
elements with NumPy over the shared buffer. The
[reduction contract](../reference/language-contract.md#array-and-parallel-reductions)
defines results, precision and ordering for both paths.

## Backend notes

**CPU.** There is no workgroup model, and the CPU doesn't emulate one.
`tack.device("cpu", num_threads=n)` makes a CPU device with that worker count;
it shares memory with the other CPU devices.
Kernels that use shared memory, barriers, `thread_id` or block reductions
are rejected even when the primitive is unreachable. For private scratch
storage, use `local_array` or `local_array_like`. Parallelism comes from a
thread pool JIT'd from LLVM IR, or a `ThreadPoolExecutor` with
`TACK_CPU_POOL=python`, on Windows, or where the pool cannot be made. See
[CPU Threading Policy](../design/cpu-threading.md). The CPU
backend also compiles a separate `noalias` variant for calls whose written
arrays provably don't overlap other arrays. See
[Memory and Aliasing](../design/memory-and-aliasing.md).

**Metal.** There is no `f64`, because Apple GPUs don't implement double
precision. Arrays are shared buffers in unified memory, so `memory_space()`
answers `"cpu"` for them and `array_from_ptr()` wraps an `MTLBuffer`
object, not an address. Collectives need a pipeline whose
`maxTotalThreadsPerThreadgroup` is at least 256.

**CUDA.** If a CUDA context is already current, for example one that a
simulation framework created, the backend adopts it and never destroys it.
Otherwise it creates one on device 0. `exportable=True` allocates through
the CUDA virtual memory management API, so `export_memory()` can hand out
an OS handle without a copy.

**HIP.** The code generator is the CUDA generator with a different include
and texture handle type. hipRTC compiles with `-ffp-contract=on`, so
contraction stays inside one expression. Texture hardware is queried at
initialization (see [Textures](#textures)).

**Level Zero.** `f64` depends on the device (see
[Level Zero and `f64`](#level-zero-and-f64)). `external_context` takes a
mapping with non-null `driver`, `device` and `context` handles, so that
Tack allocates in a context that another library owns. A missing or null
handle raises `ValueError`. For VTK, call
`tack.interop.vtk.init_level_zero()` instead of
`tack.set_default_device("level_zero")`. See
[Interoperability](../design/interoperability.md).

## Portable subset

A program stays within every backend's capabilities if it keeps to the
following rules. Everything else in the
[kernel language contract](../reference/language-contract.md) still
applies.

| Area | Portable on all five backends |
|---|---|
| Array and scalar dtypes | `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`. Don't use `f64`, which Metal and some Level Zero devices lack. A Python `float` scalar becomes `f64` only when an array argument is `f64` |
| Atomics | `atomic_add`, `atomic_min` and `atomic_max` on `i32`, `u32` or `f32` global arrays, as statements |
| Launch size | At most 2^32 − 256 iterations per launch, HIP's cap, unless the CUDA or Level Zero device reports a lower limit. Split larger work across launches |
| Workgroups | None. If you don't need the CPU, collectives are portable across the four GPU backends in complete 256-lane groups with proven-uniform participation |
| Scratch storage | `local_array` and `local_array_like` |
| Textures | `f32` sources, refreshed with `tex.update()` after the array changes |
| Reductions | `Array.sum()`, `min()`, `max()` and `mean()` on any dtype. Only `f32` is reduced natively on the device |
| Interop | Backend-specific by nature. DLPack device types and pointer kinds differ per backend |

To keep code portable, branch on declared capabilities rather than on
backend names, as the test fixtures in `conftest.py` do:

```python
from tack.runtime.dispatch import get_backend

be = get_backend()
dtype = tack.f64 if be.supports_f64 else tack.f32
use_block_sum = be.supports_workgroups
use_atomic_i64 = tack.i64 in be.supported_atomic_dtypes
```

`test_backend_contract.py` checks that the declarations match behavior. For
example, it checks that `f64` dispatch agrees with `supports_f64`, that
reductions agree with `supports_device_reductions`, and that the memory
spaces are self-consistent. See [Conformance](conformance.md).
