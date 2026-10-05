# Backend capabilities

Tack compiles one kernel source for five targets, but the targets don't
all offer the same things. Metal has no `f64`, the CPU has no workgroups,
and atomics are narrower on some GPUs than ordinary fields. This page is
the contract for those differences. It covers what a backend declares, what
each declaration means, what each one rejects and when, and the subset of
the language that runs everywhere.

All statements describe release candidate `745e01f`. The declarations live
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
| `allocate_field(dtype, shape, exportable=False)` | Allocates backend storage and returns a `DeviceBuffer`. `exportable=True` asks for memory that `Field.export_memory()` can share without a copy. Only CUDA acts on the flag, and the other backends ignore it |
| `wrap_ptr(ptr, dtype, shape)` | Wraps existing memory as a `DeviceBuffer` without copying or taking ownership. What counts as `ptr` is backend-specific. See [Pointer interop](runtime-api.md#pointer-interop) |
| `execute(kernel, args, kwargs)` | Compiles or reuses a variant and runs it. All five backends delegate to `resolve_variant()` in `runtime/kernel_utils.py`, which is where every capability check below runs |

The base class raises `NotImplementedError` for all three.

### Optional methods with defaults

| Method | Default | Overridden by |
|---|---|---|
| `memory_space(ptr)` | Returns `"cpu"`, which is correct for backends whose allocations are host-addressable | CUDA, HIP and Level Zero, which ask their driver. CPU overrides it with the same answer |
| `reduce_field(field, op)` | Raises `NotImplementedError`. Called only when `supports_device_reductions` is true | Metal, CUDA, HIP and Level Zero |

### Attributes

| Attribute | Kind | Meaning |
|---|---|---|
| `name` | declared | Arch identifier, matching `tack.init(arch=...)`: `cpu`, `metal`, `cuda`, `hip` or `level_zero` |
| `display_name` | declared | How messages spell the backend, if different from `name` |
| `label` | derived | `display_name or name`. Use it in messages |
| `supported_dtypes` | declared | Scalar types accepted as field and texture argument dtypes. Dispatch checks every field argument against this set |
| `supports_f64` | derived | `f64 in supported_dtypes` |
| `supports_device_reductions` | declared | True when `Field.sum()`, `min()` and `max()` can run on the device. When false they run in NumPy on the host. It selects a path and never rejects anything |
| `supports_workgroups` | declared | Native workgroup execution for `shared`, `shared_like`, `barrier`, `thread_id` and `block_sum`/`block_min`/`block_max`. It does **not** promise uniform participation, atomic support or safe partial workgroups |
| `supported_atomic_dtypes` | derived | Field dtypes that support `atomic_add`, `atomic_min` and `atomic_max`. This is the backend's entry in `ATOMIC_DTYPES` (`lang/atomic_support.py`) intersected with `supported_dtypes` |
| `init_options` | declared | Keyword options that `tack.init()` forwards to the constructor. `tack.init()` rejects any other option |
| `device_memory_spaces` | declared | `memory_space()` results that `field_from_ptr()` accepts for an integer pointer. An empty set means that the backend doesn't distinguish memory spaces and nothing is checked |
| `dlpack_refusal_note` | declared | Text appended to the error when a DLPack import is refused, for backends whose reason the device type doesn't convey |

Because `supported_atomic_dtypes` is an intersection, a device without
`f64` also loses `f64` atomics. That matters on Level Zero, where
`supported_dtypes` depends on the device.

## Capability matrix

Read from the backend classes at `745e01f`. "All ten" means `i8`, `u8`,
`i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32` and `f64`.

| | CPU | Metal | CUDA | HIP | Level Zero |
|---|---|---|---|---|---|
| `name` / `label` | `cpu` / CPU | `metal` / Metal | `cuda` / CUDA | `hip` / HIP | `level_zero` / Level Zero |
| `supported_dtypes` | All ten | All except `f64` | All ten | All ten | All except `f64`, plus `f64` when the device reports it |
| `supports_f64` | yes | no | yes | yes | per device |
| `supports_device_reductions` | no (NumPy) | yes | yes | yes | yes |
| `supports_workgroups` | no | yes | yes | yes | yes |
| `supported_atomic_dtypes` | All ten | `i32`, `u32`, `f32` | `i32`, `u32`, `i64`, `u64`, `f32`, `f64` | `i32`, `u32`, `i64`, `u64`, `f32`, `f64` | `i32`, `u32`, `f32` |
| `init_options` | none | none | none | none | `external_context` |
| `device_memory_spaces` | none (unchecked) | none (unchecked) | `cuda`, `cuda_pinned`, `cuda_managed` | `hip`, `hip_pinned`, `hip_managed` | `level_zero` |
| `dlpack_refusal_note` | none | set | none | none | none |
| 3D textures | software trilinear | hardware | hardware | hardware if the device has image support and the extents fit, else software | hardware if the device has samplers and the extents fit, else software |
| `Field.export_memory()` | no | yes (`mtl_buffer`) | yes (OS handle) | no | no |
| DLPack export device | `kDLCPU` | `kDLCPU` (host view of unified memory) | `kDLCUDA` | `kDLROCM` | `kDLOneAPI` |
| DLPack zero-copy import | `kDLCPU`, `kDLCUDAHost`, `kDLROCMHost` | none in practice (see [DLPack](runtime-api.md#dlpack)) | `kDLCUDA`, `kDLCUDAManaged` | `kDLROCM` | `kDLOneAPI` |
| Device chosen | host | `MTLCreateSystemDefaultDevice()` | the current CUDA context if one exists, else device 0 | `hipGetDevice()` | the first device of the first driver, or the one in `external_context` |
| Python requirements | `llvmlite` (`[cpu]` extra) | `pyobjc-framework-Metal` (`[metal]`) | `cuda-python>=13.2.0` (`[cuda]`) | `hip-python>=7.0`, Linux x86_64 only (`[hip]`) | none. The `[level_zero]` extra is empty |
| System requirements | none | macOS with a Metal GPU | NVIDIA driver and NVRTC | ROCm with hipRTC | `libze_loader.so` and `libocloc.so` (Intel compute runtime) |

### Level Zero and `f64`

**Current behavior:** `LevelZeroBackend._query_device()` runs during
construction and calls `zeDeviceGetModuleProperties`. It then *shadows* the
class attribute with an instance set that contains the nine non-`f64` types,
plus `f64` when the reported `fp64flags` is nonzero. `supports_f64` and
`supported_atomic_dtypes` follow automatically. To find out what the
current device supports, ask:

```python
import tack
from tack.runtime.dispatch import get_backend

tack.init(arch=tack.level_zero)
be = get_backend()
print(be.supports_f64, sorted(t.name for t in be.supported_atomic_dtypes))
```

### Textures

The variant key records which texture path a call uses, because the path
changes the generated code. HIP and Level Zero make that decision in their
`_store_texture_shapes()` before the key is built:

- **HIP** uses hardware sampling when `hipDeviceAttributeImageSupport` is
  nonzero and every extent is within the smallest reported maximum 3D
  texture extent. If any reported maximum is non-positive, the extents are
  treated as unbounded. CDNA parts such as MI300 report no image support,
  so they sample in software.
- **Level Zero** uses hardware sampling when the device reports
  `maxSamplers > 0` and every extent is at most `maxImageDims3D`. Xe-HPC
  (Ponte Vecchio) reports no samplers.

**Current behavior and caller constraint:** `tack.texture3d()` accepts `f32`
and `f64` fields. However, the hardware paths copy the field into a 32-bit
float texture (`R32Float` on Metal, `CU_AD_FORMAT_FLOAT` on CUDA and HIP,
`ZE_IMAGE_FORMAT_TYPE_FLOAT` on Level Zero). Pass `f32` fields to textures.
Hardware paths also cache the texture per compiled variant, keyed by buffer
address and extents, so later writes to the source field aren't seen by
those paths. Software paths read the field directly. Don't write to a
texture's source field between dispatches that sample it. If you must,
sample from a new field. The `interp` argument of `texture3d()` is stored
but isn't read by any code generator. Sampling is trilinear on every
backend.

## What each capability gates

All dispatch-time checks run inside `resolve_variant()`. "Cold" means the
first call with a new variant key, before optimization, code generation and
compilation. "Every dispatch" also covers cached calls, and those checks run
before launch.

| Capability | Checked by | When | Rejection |
|---|---|---|---|
| `init_options` | `tack.init()` in `runtime/dispatch.py` | At initialization, after the backend module imports and before construction | `ValueError`: "The '`cpu`' backend does not accept the option(s) `num_threads`. Accepted: none." The previous backend stays active |
| `supported_dtypes` | `check_dispatch_types()` in `lang/type_inference.py` | Cold dispatch. A new dtype always produces a new key | `TypeError`, re-raised by `Kernel.__call__` with a `Kernel '<name>': ` prefix: "...parameter '`data`' has dtype `tack.f64`, which is not supported on Metal. Supported dtypes: ..." |
| `supports_workgroups` | `check_workgroup_support()` in `lang/workgroup_support.py` | Every dispatch, before the key lookup. The feature scan is memoized on the IR template. Also runs during inspection and direct LLVM generation | `NotImplementedError` naming the kernel, backend and primitives, with the hint "use `local_array` or `local_array_like`". Dispatch wraps it in `RuntimeError` ("Kernel '`k`' failed on CPUBackend: ..."). Inspection raises it directly |
| Workgroup participation | `check_workgroup_participation()` in `lang/workgroup_participation.py` | Cold dispatch and inspection, on backends with `supports_workgroups` | `ValueError`: "Kernel '`k`': `barrier` at `<IR path>` requires uniform workgroup participation: ..." |
| Full 256-lane groups | `check_workgroup_launch()` | Cold dispatch, every cached dispatch, and inspection, for kernels with barriers or block reductions | `ValueError`: "...workgroup collectives require complete 256-lane groups; iteration count `N` would create a partial workgroup." |
| Device group size | `CompiledMetalKernel` and `CompiledL0Kernel` | At compilation, for collective kernels, when the Metal pipeline or Level Zero device allows fewer than 256 lanes | `ValueError`: "...workgroup collectives require 256 lanes; this pipeline/device selects `n`." |
| `supported_atomic_dtypes` | `check_atomic_support()` in `lang/atomic_support.py` | Cold dispatch and inspection in every mode, including structurally unreachable atomics and empty grids | `TypeError`: "Kernel '`k`': `metal` atomic_add does not support target dtype `tack.i64`". The same check rejects targets that aren't global field parameters |
| Atomic alignment | `check_atomic_alignment()` | Every dispatch and inspection | `ValueError`: "Kernel '`k`': atomic target parameter `i` requires `n`-byte alignment" |
| `device_memory_spaces` | `field_from_ptr()` in `lang/field.py` | When wrapping, for `ptr` of Python type `int` only | `ValueError`: "Pointer is in '`cpu`' memory but the active backend is '`CUDA`'. field_from_ptr() requires a device pointer. ..." |
| DLPack device type | `dlpack_to_field()` in `lang/dlpack.py` | On import | `RuntimeError` naming the backend and DLPack device type, followed by `dlpack_refusal_note` if set. `ValueError` for an unknown device type |
| `supports_device_reductions` | `Field._reduce()` | Never rejects | Selects the device or NumPy path. See [Reductions](runtime-api.md#reductions) |
| Texture hardware | `_store_texture_shapes()` (HIP, Level Zero) | Before the key is built | Never rejects. Falls back to software sampling |

A `TypeError` raised during dispatch reaches the caller with an extra
`Kernel '<name>': ` prefix, which `Kernel.__call__` adds, and with the
original exception chained. `ValueError` passes through unchanged. See
[Exceptions](runtime-api.md#exceptions).

For the full text of the workgroup and atomic rules, see
[Workgroups and synchronization](../reference/language-contract.md#workgroups-and-synchronization)
and [Atomic field updates](../reference/language-contract.md#atomic-field-updates).

!!! note "What inspection does not check"

    `tack.inspect()` runs the workgroup, participation, launch-count,
    atomic and alignment checks above. It doesn't run
    `check_dispatch_types()`, and it doesn't make the HIP or Level Zero
    texture fallback decision. On Metal, the MSL generator rejects `f64`
    parameters itself with a `TypeError`. On HIP or Level Zero, inspection
    can return hardware-texture or `f64` source that dispatch on the same
    device would generate differently or reject.

### Device reductions

`supports_device_reductions` does not mean that every reduction runs on the
device. **Current behavior:** on every GPU backend, `reduce_field()` reduces
only `f32` fields natively, and other dtypes use the shared NumPy fallback.
Level Zero also falls back when the device's maximum group size is below
256. Empty fields return without a device launch. The
[reduction contract](../reference/language-contract.md#field-and-parallel-reductions)
defines results, precision and ordering for both paths.

## Backend notes

**CPU.** There is no workgroup model, and the CPU doesn't emulate one.
Kernels that use shared memory, barriers, `thread_id` or block reductions
are rejected even when the primitive is unreachable. For private scratch
storage, use `local_array` or `local_array_like`. Parallelism comes from a
thread pool. See [CPU Threading Policy](../design/cpu-threading.md). The CPU
backend also compiles a separate `noalias` variant for calls whose written
fields provably don't overlap other fields. See
[Memory and Aliasing](../design/memory-and-aliasing.md).

**Metal.** There is no `f64`, because Apple GPUs don't implement double
precision. Fields are shared buffers in unified memory, so `memory_space()`
answers `"cpu"` for them and `field_from_ptr()` wraps an `MTLBuffer`
object, not an address. Collectives need a pipeline whose
`maxTotalThreadsPerThreadgroup` is at least 256.

**CUDA.** If a CUDA context is already current, for example one that a
simulation framework created, the backend adopts it and never destroys it.
Otherwise it creates one on device 0. `exportable=True` allocates through
the CUDA virtual memory management API, so `export_memory()` can hand out
an OS handle without a copy.

**HIP.** The code generator is the CUDA generator with a different include
and texture handle type. Texture hardware is queried at initialization (see
[Textures](#textures)).

**Level Zero.** `f64` depends on the device (see
[Level Zero and `f64`](#level-zero-and-f64)). `external_context` takes a
mapping with non-null `driver`, `device` and `context` handles, so that
Tack allocates in a context that another library owns. A missing or null
handle raises `ValueError`. For VTK, call
`tack.interop.vtk.init_level_zero()` instead of `tack.init()`. See
[Interoperability](../design/interoperability.md).

## Portable subset

A program stays within every backend's capabilities if it keeps to the
following rules. Everything else in the
[kernel language contract](../reference/language-contract.md) still
applies.

| Area | Portable on all five backends |
|---|---|
| Field and scalar dtypes | `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`. Don't use `f64`, which Metal and some Level Zero devices lack. A Python `float` scalar becomes `f64` only when a field argument is `f64` |
| Atomics | `atomic_add`, `atomic_min` and `atomic_max` on `i32`, `u32` or `f32` global fields, as statements |
| Workgroups | None. If you don't need the CPU, collectives are portable across the four GPU backends in complete 256-lane groups with proven-uniform participation |
| Scratch storage | `local_array` and `local_array_like` |
| Textures | `f32` sources, treated as immutable while sampled |
| Reductions | `Field.sum()`, `min()`, `max()` and `mean()` on any dtype. Only `f32` is reduced natively on the device |
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
