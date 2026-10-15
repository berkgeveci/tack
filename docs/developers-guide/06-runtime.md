# Runtime and Dispatch

## The Backend Contract

Every backend subclasses `Backend` (`runtime/backend.py`), which declares
what a backend must implement and what callers may ask it.

Required: `allocate_field()`, `wrap_ptr()`, `execute()`.

Optional, with defaults that suit a backend lacking the feature:

| Method | Default | Overridden by |
|---|---|---|
| `fill_field(field, value)` | `False`: `Field.fill` fills the buffer in place | CPU, which runs a fill of 256 KiB or more as a kernel through its dispatch, so the threads first-touch the pages |
| `synchronize()` | Nothing to wait for | Metal, CUDA, HIP and Level Zero, which queue launches (`tack.sync()` calls it) |
| `memory_space(ptr)` | `"cpu"` | Backends with device memory |
| `cache_target()`, `variant_artifact()`, `load_variant()` | `None`: compiled variants are not stored on disk | Every backend; see [The disk cache](#the-disk-cache) |
| `reduce_field()`, `texture_in_hardware()`, `create_texture_image()` | Host reductions, software sampling | Backends with device reductions or texture units |

Declared capabilities — read these instead of probing with `hasattr`:

| Attribute | Meaning |
|---|---|
| `name` | Arch identifier, matching `tack.init(arch=...)` |
| `display_name` / `label` | How to write the backend in a message |
| `supported_dtypes` | Scalar types accepted as field dtypes |
| `supports_f64` | Derived from `supported_dtypes` — never declared separately |
| `supports_device_reductions` | Whether `reduce_field()` runs on device; otherwise `Field.sum()` and friends fall back to numpy |
| `supports_workgroups` | Native workgroup execution for shared memory, barriers, local thread IDs and block reductions; false on CPU |
| `device_memory_spaces` | Memory spaces a pointer must be in for `array_from_ptr()`; empty means no check |

Anything derivable is derived. `supports_f64` used to be declared
independently, existed on Level Zero alone, and so
`getattr(backend, 'supports_f64', True)` reported `True` for Metal — which
has no f64 at all. Deriving it from `supported_dtypes` means there is
nothing to keep in step.

Level Zero sets `supported_dtypes` in `__init__` rather than on the class,
because f64 depends on what the device reports. That shadows the class
attribute, and `supports_f64` follows automatically.

## Backend Lifecycle

Each backend follows the same lifecycle:

```
tack.init(arch=tack.metal)
    → dispatch.py creates MetalBackend()
    → Backend discovers device, creates command queue / context

kernel(x, y, out, alpha, n)
    → Kernel.__call__
    → backend.execute(kernel, args, kwargs)
    → Template expansion + IR pipeline + codegen + dispatch
```

## Backend.execute() Flow

Every backend uses `resolve_variant()` in `runtime/kernel_utils.py` to
find or build a specialization. A call first tries the dispatch fast path
(below); otherwise:

1. Expand template arguments and detect vector and texture fields.
2. Obtain the pristine IR template for this specialization (lowered, or read
   from the disk cache) and check required
   workgroup primitives against the backend's `supports_workgroups`. CPU
   memoizes feature discovery on the immutable template so warm dispatch
   does not walk the body again. GPU targets bypass this rejection check.
   In debug mode the template is the bounds-checked lowering, and the two
   hidden arguments (`debug_bounds.hidden_args`) are appended to the call's.
3. Infer argument dtypes and categories on a private parameter probe, record
   texture extents, and derive resolved shape dependencies.
4. Look up the variant in the backend's weakly keyed per-kernel cache, and
   then in the disk cache (`_stored_variant`).
5. On a miss of both, clone the template (`clone_ir` in `lang/ir_traversal.py`, not
   `copy.deepcopy`), resolve dimensions, infer/check types, localize outer
   scalars the loop body assigns (`_localize_outer_scalars`), check atomics
   and workgroup participation, and run conservative copy propagation. The
   backend build callback then packs scalars where needed, annotates types,
   generates code, and compiles. The `KernelVariant` records the atomic
   targets and the field arguments the kernel may write. `_store_variant`
   then writes it to disk.
6. On every call, check atomic alignment and refuse read-only fields bound
   to written parameters (`check_writable_fields`), and record the call's
   signature for the fast path.
7. Resolve the launch range from the variant's IR for this dispatch; GPU
   backends refuse a range past their launch limit (`check_launch_size`).
   Replace each `Texture3D` with the storage it owns (`bind_textures`),
   bind arguments (updating any scalar pack buffers), and launch. GPU
   backends do the last two under a lock: the variant's `dispatch_lock` on
   CUDA and HIP, the backend's `_batch_lock` on Metal and `_launch_lock`
   on Level Zero. The CPU pool's run is a transaction under `CPool._txn`.

### The dispatch fast path

`resolve_variant` first reads the call's *signature* off its arguments
(`_call_signature`): each field's dtype, shape and element layout, each
scalar's kind and integer width, and for a template its class token, its
class constants' current values and what the key and the expansion read of
each attribute (nested templates by path). Equal signatures imply an equal
variant key, so a signature seen before (`_fast_resolve`) expands the
templates through a stored plan, decides disjointness over precomputed
field slots, runs the per-call checks of step 6 (`_check_call`) and returns
the variant the entry holds, without the template scope, IR lookup or key.
`drop_variants` drops the entries with the variants. Debug mode and keyword
arguments never use or record entries. A warm CPU dispatch with eight field
arguments costs about 34 µs of host time on a 2 GHz Xeon. Anything that
changes the variant key must change the signature too, or a call will run
the wrong variant; `test_dispatch_fast_path.py` covers what the signature
must separate (integer widths, the float context, class constants, template
structure) and what a hit must still check.

The compiled key includes dtypes, field/scalar/texture categories, vector
widths, texture extents, template structure/constants, and baked-in dimension
sizes. Template structure includes actual class identity and runtime scalar
attribute names. Changing scalar values alone does not recompile; changing
an attribute layout or a vector width does.

Passes run only on a cache miss and never mutate the pristine template.
Parameter probing is private to each dispatch so concurrent calls cannot
observe another call's types. Fields may overlap in storage; generated field
parameters therefore carry no unconditional `noalias` or `restrict` promise.

The CPU backend adds one more key element: whether this call's fields are
disjoint. `fields_disjoint()` compares the byte ranges of the field arguments
on every dispatch and passes when no field the kernel writes overlaps another
field. Qualifying calls use a variant compiled with `noalias` on its field
pointers; the rest use the variant without it. Without the promise LLVM must
reload after every store, which matters for kernels that accumulate through a
field in an inner loop. Measured on 2026-10-03, single-threaded: on a 2012
Xeon E5-2650 the store-accumulator kernel at 2^20 elements ran in 2465 µs
without the promise and 736 µs with it, and a stencil in 776 µs and 604 µs;
the check added about 3–4 µs per 16-element dispatch there and about 1–1.6 µs
on an Apple M1 Max, where the accumulator showed no comparable gain. See
[Specialization and Caching](../design/specialization-and-caching.md) and
[Memory and Aliasing](../design/memory-and-aliasing.md).

Template classes appear in keys as a token rather than the class object.
When a `@tack.device_class` class is collected, a finalizer drops the IR and
the compiled variants specialized on it from every backend's cache.

Public inspection checks the selected backend's workgroup capability before
preparing a variant. Private `_prepare_ir` may omit a backend for cross-target
codegen tools/tests. Direct LLVM generation always checks the supplied IR
afresh because callers may mutate it.

GPU variant construction checks conservative workgroup participation after
scalar localization, before optimization/packing. `KernelVariant` caches
the full-group requirement; every collective dispatch checks its logical
count, including cache hits. Direct GPU generators check mutable IR afresh,
using `_is_scalar_pack` metadata for immutable runtime scalar inputs. Metal
pipeline capacity and Level Zero X/total limits must admit 256 lanes.
These checks do not establish atomic type/scope support or race freedom.

## The Disk Cache

`runtime/disk_cache.py` (public as `tack.cache`) keeps two levels across
processes. `Kernel._lower` asks it for the lowered IR (`.ir` entries),
keyed by content: the kernel's source and closure, the specialization,
template content (MRO names, `@tack.func` sources, constants), Tack's
fingerprint and the Python version. While lowering runs, every name
resolved through `CallBindings` is recorded with a fingerprint of what it
was bound to, and an entry is used only if each still resolves the same
way, so an edited device function or constant misses. Anything lowering
reads from Python must go through `CallBindings` or into the key.

`_resolve_variant` asks it for a compiled variant (`.variant` entries) on an
in-memory miss, keyed by the `.ir` entry, the probed parameter types, vector
widths, baked-in shapes, disjointness, the backend name and
`Backend.cache_target()`. An entry holds the IR after the passes and
whatever `variant_artifact()` returned; `load_variant()` rebuilds the
payload from it, so a hit skips the passes, code generation and as much of
compilation as the backend can (the CPU stores object code, Metal its MSL,
CUDA PTX, HIP code objects, Level Zero a native binary with its SPIR-V).
The target must name everything the artifact depends on besides the
kernel and its argument types: the device, the compiler's and driver's
versions, and the compile options, which the backends keep in module
constants (`_NVRTC_OPTIONS`, `_HIPRTC_OPTIONS`, `_OCLOC_OPTIONS`) that
compiler and target both read. A GPU payload's scalar pack buffers are
device memory of the process, so `load_variant` makes them anew from the
call's arguments. Debug mode is part of the lowered key, so its kernels are
stored apart. The test suite runs with `TACK_CACHE=off` unless the
environment says otherwise; `test_disk_cache.py` runs its cases in fresh
processes. See [The Kernel Cache](../users-guide/17-kernel-cache.md) for the
settings and MPI bundles.

## Launch Completion

Only the CPU backend finishes a launch before `execute` returns. Metal,
CUDA, HIP and Level Zero queue launches in program order (an open Metal
command buffer, one stream per CUDA context or HIP device, one asynchronous
immediate command list with a barrier after each command on Level Zero)
and return; CUDA, HIP and Level Zero queue fills and host-to-device copies
in the same order. The host waits where it reads field memory (`numpy()`,
`read_range`, reductions; on Metal every host access to a buffer) or calls
`tack.sync()`, which is `Backend.synchronize()`. A field dropped
while a queued launch still uses it is freed after that launch
(stream-ordered allocation, Level Zero's deferred free, Metal's
per-launch references). Memory handed outside Tack (`wrap_ptr`, DLPack
export, `share()`) makes every launch that uses it complete before it
returns, and so does a CUDA, HIP or Level Zero launch with a texture. A
launch's error surfaces at the next wait. `test_cuda_async.py`,
`test_hip_async.py` and `test_level_zero_async.py` check ordering, host
writes and the waits.

## Loop Range Resolution

`_get_loop_range()` extracts the parallel for-loop bound from the IR and
resolves it against actual arguments. It supports:

- `IRConstant(N)` — literal bound
- `IRName("n")` — scalar parameter
- `IRDimSize("x", 0)` — `x.shape[0]` or `len(x)`
- `IRBinOp` over any of the above — e.g. `range(n - 1)`

This must run before scalar packing since packing removes scalar params.

The grid bound is the one `IRDimSize` that `resolve_ir` deliberately leaves
unresolved. Everywhere else it folds to a literal, because the value has to
appear in the generated code; the grid bound does not — codegen reads the
`__loop_end__` parameter — so folding it would make the compiled kernel
depend on the array's length and force a recompile for every new size.
`ir_shape_deps()` excludes it from the variant key for the same reason.

## Field Allocation

Each backend implements `allocate_field(dtype, shape)` returning a
backend-specific buffer:

| Backend | Buffer type | Memory model |
|---------|------------|--------------|
| CPU | `NumpyBuffer` | numpy array (host memory) |
| Metal | `MetalBuffer` | Metal shared buffer (unified CPU+GPU) |
| CUDA | `CUDABuffer` | Stream-ordered device pointer (`cuMemAllocAsync`) |
| HIP | `HIPBuffer` | Stream-ordered device pointer (`hipMallocAsync`; `hipMalloc` without memory pools) |
| Level Zero | `L0Buffer` | Device pointer (`zeMemAllocDevice`) |

`Field` (`tack.Array`) wraps a buffer with dtype and shape metadata.
`tack.copyto()` and `numpy()` handle host-device transfers through the
buffer's `from_numpy()` / `to_numpy()` (on Metal this is a memcpy
within unified memory; on CUDA/HIP/L0 it involves explicit copies).

A new field's contents are unspecified on every backend; nothing fills it.
`tack.zeros`, `ones` and `full` call `Field.fill`, which first offers the
fill to `Backend.fill_field`. The CPU backend takes a fill of 256 KiB or
more and runs `tack.algorithms.copy._fill_kernel` through its dispatch, so
the threads that will use the pages touch them first; GPU buffers fill on
the device. Metal's new buffers are already zero (`starts_zeroed`), so
`tack.zeros` skips the fill there. Code that reads a field before writing
it must make it with `tack.zeros`.

## Kernel Dispatch

### CPU

The LLVM-JIT'd function is called via ctypes. The backend compares measured
serial work against measured thread fan-out cost, splitting worthwhile
ranges across a persistent pool of pthreads that is itself JIT'd from LLVM
IR (`cpu_pool.py`): one ctypes call posts the launch to the workers it
uses (each waits on its own mailbox; the rest sleep on), each worker
calls the compiled function with a `(start, end)` sub-range through a
per-kernel wrapper, and none of them touches Python. Workers and the
caller claim chunks from a counter rather than taking one equal share
each, so a slow worker takes fewer. Idle workers block on a futex (Linux)
or `os_sync_wait_on_address` (macOS 14.4+) and poll elsewhere. Arguments
reach the wrapper as field addresses packed into 8-byte slots with one
`struct.pack_into`. `TACK_CPU_POOL=python` uses a `ThreadPoolExecutor`
instead, which is also the fallback on Windows and wherever the pool
cannot be built; tests that simulate workers through the Python path's
seams set it.

Periodic serial rechecks sample different positions in the range. Long
worker spans establish a floor under the serial estimate, preventing cheap
image slices from making an expensive frame look cheap. That floor applies
to the measured workload: runtime inputs can change without recompilation.
When every worker of a complete dispatch later finishes below the trusted
span duration, the backend retires the old floor and schedules a serial
recheck on the next call. Partial head/tail dispatches and a short median
with any long worker cannot retire it. Worker spans below the rate clock's
resolution still establish that the complete dispatch was short.

### Metal

Encodes a compute command: the field pointers go into one argument buffer
(declared resident with `useResource`), and `dispatchThreads` covers the
grid. Textures use a separate binding namespace (`setTexture_atIndex_`) and
bind the `MetalTextureImage` the `Texture3D` owns. Each launch owns a fresh
argument buffer and its packed scalars, encodes into the open command
buffer, and returns; the command buffer is committed every 64 launches.

### CUDA / HIP

Launches via `cuLaunchKernel` / `hipModuleLaunchKernel` with a pointer array
of arguments, on the context's (CUDA) or device's (HIP) stream, and returns
once queued. Grid size = `ceil(loop_end / 256)`, block size = 256. A
texture argument passes the texture object of the `CUDATextureImage` or
`HIPTextureImage` its `Texture3D` owns.

### Level Zero

Sets kernel arguments via `zeKernelSetArgumentValue`. Dispatches via
`zeCommandListAppendLaunchKernel` on the backend's asynchronous immediate
command list, followed by a barrier, and returns once appended; the
arguments are captured at the append, so the next call may set its own.
Textures use the `L0TextureImage` (`zeImageCreate`, bound as `image3d_t`)
their `Texture3D` owns on devices with sampler hardware.

## Scalar Packing at Dispatch

Pack field buffers are allocated once during compilation and cached.
On subsequent calls, `_update_pack_fields()` just writes the new scalar
values into the existing device buffers via `from_numpy()`. This avoids
per-dispatch allocation overhead. With queued launches this stays correct
because the copy is queued too: on CUDA, HIP and Level Zero the next
call's update is queued after the launch that reads the old values. Metal
launches carry their own copy of the scalars (`_Uniforms`) instead.

## Device Pointer Interop

`tack.array_from_ptr()` wraps an existing device pointer as a Field without
allocation or copy. Each backend implements `wrap_ptr(ptr, dtype, shape)`:

| Backend | `ptr` type | Implementation |
|---------|-----------|----------------|
| CPU | numpy array or int address | `np.frombuffer` view into existing memory |
| Metal | `MTLBuffer` object | Creates numpy view via `contents().as_buffer()` |
| CUDA | device address (int, NumPy integer or `CUdeviceptr`) | Stores pointer, skips `cuMemAlloc` |
| HIP | device address (anything `int()` accepts) | Stores it as an `int` (hip-python reads a NumPy scalar as a buffer), skips `hipMalloc` |
| Level Zero | device address (anything `int()` accepts) | Stores `c_void_p`, skips `zeMemAllocDevice` |

On CUDA, HIP and Level Zero, `array_from_ptr()` first checks the pointer
with `as_address()` (`TypeError` if it is not an address) and
`memory_space()` (`ValueError` if it is not device memory).

### Ownership

Wrapped buffers set `_owned = False`. Buffer destructors (`__del__`) check
this flag to skip freeing external memory:

```python
def __del__(self):
    if hasattr(self, '_device_ptr') and getattr(self, '_owned', True):
        # in stream order, after the launches queued so far
        driver.cuMemFreeAsync(self._device_ptr, token.queue())  # skipped for wrapped ptrs
```

### Read-Only Protection

`Field._writable` defaults to `True` for allocated fields and `False` for
`array_from_ptr()`; a read-only DLPack import is also not writable. The
`_check_writable()` method guards `tack.copyto()` and `fill()`. For kernels,
the variant records which field arguments the kernel may store to
(`KernelVariant.written_fields`, from `written_field_params`, all of them
when a store can't be traced), and `check_writable_fields()` raises
`ValueError` on every dispatch that binds a read-only field to one of
them.

## Error Handling

`Kernel.__call__` wraps backend errors:
- `TypeError` → includes kernel name, once (a message that already starts
  with it is kept as is)
- Compilation failure → extracts error lines, suppresses full source dump
- `RuntimeError` → includes kernel name and backend class name

In debug mode, `Kernel.__call__` (and the result epilogue) calls
`debug_bounds.check()` after the launch, which waits for it and raises
`IndexError` naming the first out-of-bounds access it recorded. On a
backend that queues launches, any other launch's error surfaces at the
next wait rather than at its call.

`tack.init()` wraps backend initialization errors with the architecture
name, platform, and install instructions.
