# Runtime API

This page is the contract for Tack's host-side surface: choosing
devices, creating and moving arrays, reductions, sharing memory with other
libraries, inspecting generated code, the decorators, and the rules for
calling a kernel. Each entry gives the signature, what is guaranteed, what
the caller must ensure, and which exception a violation raises. What a
kernel *computes* is defined by the
[kernel language contract](../reference/language-contract.md). What each
backend supports is defined in [Backend capabilities](backend-capabilities.md).

All statements describe release candidate `23d6e1d`, updated for the
queued launches, disk cache, debug mode and CPU pool merged through
`b94af23`, and for the device model of `bba0846`. They use the labels
defined in [Contracts](index.md#how-to-read-a-contract).

## Devices

Defined in `packages/tack-core/src/tack/runtime/dispatch.py` and
re-exported as `tack.device`, `tack.Device`, `tack.set_default_device`,
`tack.get_default_device`, `tack.set_debug`, `tack.sync` and
`tack.sync_async`.

### `tack.device(spec="cpu", **options)`

Returns the `tack.Device` that `spec` names. A `Device` passed as `spec` is
returned as it is.

| | |
|---|---|
| `spec` | `"cpu"`, `"metal"`, `"cuda"`, `"hip"` or `"level_zero"`. The numbered types (`cuda`, `hip`, `level_zero`) also take `"type:N"`, such as `"cuda:0"`; `"cuda"` means `"cuda:0"`. `cpu` and `metal` are written bare |
| `**options` | Forwarded to the backend constructor. Each option must be in that backend's `device_options`. CPU declares `num_threads`, which takes precedence over `TACK_CPU_THREADS`, and Level Zero declares `external_context` (the driver, device and context handles of a context owned by another library). The other backends declare none |

**Current behavior:**

- Without options, each `(type, index)` is made on first use and is the
  same `Device` on every later call. Its compiled kernels and its arrays
  are shared by everything that asks for it.
- With options, every call makes a new device, which is not equal to the
  one without options or to another made with the same options.
- Only device 0 of each type is reachable.
- A CUDA context is current on the thread that made the device (the first
  call that names it). See [Calling a kernel](#calling-a-kernel).
- A CUDA context already current on that thread, such as one an embedding
  framework created, is adopted and never destroyed. It must be current
  before the first call that names `"cuda"`: the device is made once, so a
  context made current later is not adopted.

**Errors:**

| Condition | Exception |
|---|---|
| `spec` neither a string nor a `Device` | `TypeError`: "a device is a Device or a string such as 'cpu' or 'cuda:0', not ..." |
| Options with a `Device` | `TypeError`: "options make a new device: pass a device spec, not a Device" |
| Unknown type | `ValueError`: "Unknown device type: '`vulkan`'. Available: cpu, cuda, hip, level_zero, metal" |
| A number on `cpu` or `metal`, or a non-numeric suffix | `ValueError`: "'`cpu:0`' is not a device: write '`cpu`'" (with "or '`cuda:N`'" for numbered types) |
| A device other than 0 | `ValueError`: "device 1 of '`cuda`': Tack reaches device 0 only, for now" |
| Missing Python dependency (an `ImportError` from the backend module or constructor) | `RuntimeError`: "Cannot start the '`hip`' device: missing dependency." followed by the original error and install instructions. The original exception is chained |
| Option not in `device_options` | `ValueError`: "The '`metal`' device does not accept the option(s) `num_threads`. Accepted: none." Raised after the backend module is imported, so a missing dependency is reported first |
| Device or runtime failure (a `RuntimeError` from the constructor) | `RuntimeError`: "Cannot start the '`cuda`' device on Linux." followed by the original message and install instructions. Chained |
| Any other constructor failure, such as `ValueError` for an incomplete `external_context` or a non-integer `TACK_CPU_THREADS` | Propagates unwrapped |

### `tack.Device`

| Member | Meaning |
|---|---|
| `d.type` | `"cpu"`, `"metal"`, `"cuda"`, `"hip"` or `"level_zero"` (the backend's `name`) |
| `d.index` | Which of the type's devices (the backend's `ordinal`); always 0 for now |
| `d.synchronize()` | Waits for every kernel launched on this device so far, and raises a queued launch's error |
| `str(d)` | The spec: `"cuda:0"` for numbered types, `"cpu"` and `"metal"` bare. `repr(d)` is `Device('cuda:0')` |
| `d == e` | True when both are the same backend; hashable on the same terms |
| `with d:` | Makes `d` the default device inside the block, on this thread only. Blocks nest; leaving one restores the previous default |

### `tack.set_default_device(spec="cpu", **options)`

Makes `tack.device(spec, **options)` the default device and returns it. The
default device is where arrays are allocated when no `device=` is given,
and where kernels with no arrays run. It is process-wide; a `with` block
overrides it on its own thread.

**Current behavior:**

- With nothing set, the default is the CPU, chosen at the first
  allocation, launch or `get_default_device()` call.
- Without options, `set_default_device` builds nothing new: the device is
  the one `tack.device(spec)` always returns. Arrays made before a switch
  keep their device and keep working, and kernels given them run there
  (below). Compiled kernels survive a switch and back.
- **`TACK_NO_REINIT`:** if a default is already set (including the CPU
  default picked by an earlier allocation or launch) and this variable is
  switched on, `set_default_device` returns the current default. It ignores
  `spec` and `options`, and it doesn't validate either. The variable is
  read by `dispatch.env_flag`: unset, empty, `0`, `false`, `no` and `off`
  (in any case, ignoring surrounding spaces) mean off, and any other value
  means on. This exists for embedding, for example inside an ANARI device
  that shares the process.

Errors are those of `tack.device`. If the device cannot be made, the
previous default stays.

### `tack.get_default_device()`

Returns the default device: the innermost `with` block's on this thread,
else the one `set_default_device` chose, else the CPU.
`tack.runtime.dispatch.get_backend()` returns the same device's `Backend`.

### Where arrays live and kernels run

- Every array records its device: `a.device` is a `tack.Device`. Every
  constructor takes `device=` (a `Device` or a spec): `empty`, `zeros`,
  `ones`, `full`, `arange`, `array`, `asarray`, `array_from_ptr`,
  `from_dlpack` and `memory_space`, which default to the default device,
  and `empty_like`, `zeros_like`, `ones_like` and `full_like`, which
  default to a Tack prototype's device.
- `a.to_device(device)` returns `a` itself when its memory is usable on
  `device`, else a copy made through the host. `tack.array(a, device=...)`
  and `tack.asarray(a, device=...)` move a Tack array the same way.
- Copies, `astype`, `reshape`, reductions, `fill`, `concat` (the first
  array's device) and textures stay on their array's device.
- **Launch placement:** a kernel runs on the default device when that
  device can address all its arrays, or when it has none; otherwise on
  the arrays' own device. A device can address an array when both are in
  one memory domain (`Backend.memory_domain`): two CPU devices with
  different thread counts share one, so a default
  `tack.device("cpu", num_threads=1)` runs kernels on CPU arrays made
  before it. A `@tack.device_class` object's arrays choose the device when
  no plain array does, and must agree with it. `tack.inspect` makes the
  same choice.
- Arrays in two memory domains raise `ValueError`: "Kernel '`k`' was given
  arrays on two devices (cpu and cuda:0); a launch runs on one. Copy them
  to one device first (a.to_device(...))." The check runs on the dispatch
  fast path too.
- `tack.algorithms` functions allocate their temporaries and launch on
  their first array argument's device. They, `tack.copyto` from a Tack
  array and `tack.concat` refuse arrays in two memory domains under their
  own name: "exclusive_scan() was given arrays on two devices (cpu and
  cuda:0); ...", not the name of an internal kernel.
- An array holds its device, so a device's backend (and a CUDA context it
  created) lives at least as long as any of its arrays.

### `tack.set_debug(enabled=True)`

Turns debug mode on or off for every kernel launched from then on.
`TACK_DEBUG` (a boolean flag, read like `TACK_NO_REINIT`) turns it on when
Tack is imported. `tack.runtime.dispatch.debug_enabled()` reports the
current setting.

**Debug mode** checks every index of every array and array access, one
dimension at a time: an array's leading dimension against its length, the
others against theirs, a flat index against the element count (a component
count for a scalar atomic on a vector array), and a local or shared array
index against its size. Constant tables and textures clamp and are not
checked.

- An access with an index outside its extent is not performed: a load gives
  0 of its type, an atomic's value is 0, and a store or atomic does nothing.
  The rest of the launch runs.
- After the launch, `Kernel.__call__` raises `IndexError` for the first
  violation it recorded: the kernel (and the device function an access was
  inlined from), the file, line and column, the access as written, the
  index, its dimension when there are several, the extent, and how many
  more accesses were skipped. Which violation is first is unspecified when
  several threads go out of bounds.
- Kernels lower and compile separately in debug mode, and the extents are
  run-time arguments: debug mode compiles no variant per array length.
  Each launch is checked when it returns, which on the GPU backends means
  waiting for it, since their launches are otherwise queued. Launches from
  different threads report their own violations.
- `tack.inspect` shows the code a debug launch runs.

Implemented in `lang/ir_bounds.py` (the checks and guards),
`runtime/debug_bounds.py` (the hidden arguments and the error) and the
code generators (`codegen/bounds.py`, `llvm_gen.py`).

## Arrays

`tack.Array` is defined in `packages/tack-core/src/tack/lang/array.py`.
Its `dtype` describes one scalar, vector or matrix element; `shape` is the
array's dimensions, excluding component dimensions. `size` counts elements.
`scalar_dtype`, `scalar_size` and `storage_shape` describe the scalar storage.
The buffer belongs to the array's device (`a.device`; see
[Where arrays live and kernels run](#where-arrays-live-and-kernels-run)).
`tack.dataset.Field` is a separate class coupling a space with coefficient values.

### Creation

| Function | Contract |
|---|---|
| `tack.empty(shape, dtype=f32, *, device=None, exportable=False)` | Allocates on `device`, by default the default device. An `int` shape becomes a 1-tuple. `exportable=True` requests memory that can be exported without a copy (CUDA only) |
| `tack.zeros(shape, dtype=f32, *, device=None)`, `tack.ones(...)`, `tack.full(shape, fill_value, dtype=None, *, device=None)` | `empty()` followed by `fill()`. `zeros` skips the fill where the buffer starts zeroed (`starts_zeroed`: Metal, which clears new buffers). `full` without a dtype makes f32 from a Python float and i32 from an int (i64 if the value needs it); a sequence `fill_value` needs an element dtype |
| `tack.arange([start,] stop[, step], dtype=None, *, device=None)` | Built in NumPy and copied in. Without a dtype, integer arguments give i32 (i64 if a value needs it) and a float argument gives f32 |
| `tack.array(obj, dtype=None, *, device=None)` | Copies values into independent storage. Without dtype, uses all input dimensions; a NumPy array keeps its dtype, Python floats give f32 and Python ints i32 (i64 if a value needs it). An element dtype consumes matching trailing component dimensions; a mismatch raises `ValueError`. Existing Tack arrays copy on their device, or through the host to another `device` |
| `tack.asarray(obj, dtype=None, *, device=None)` | `obj` itself when it is a Tack array, `dtype` is `None` or equals its dtype, and `device` is `None` or can address it; otherwise a converted or moved array |
| `tack.empty_like(prototype, dtype=None, *, device=None)` | Allocates independent, writable storage on `device`, by default a Tack prototype's device (else the default device), without reading or copying values. Accepts a NumPy array or Tack array; takes its dtype unless overridden. A scalar override of a vector/matrix prototype keeps the element dimensions. NumPy arrays produce scalar arrays with the array's shape. Contents are unspecified |
| `tack.zeros_like`, `tack.ones_like`, `tack.full_like(prototype, fill_value, dtype=None)` | `empty_like()` followed by `fill()`, as for `zeros`, `ones` and `full` |
| `tack.concat(arrays)` | A new 1-D array holding all elements in order, copied on the first array's device |
| `tack.texture3d(array, shape=None, interp="linear")` | A `Texture3D` holding a copy of the array taken at creation, for `sample(u, v, w)` in kernels. `tex.update()` copies the array again. See [Textures](backend-capabilities.md#textures) |

**Current behavior:**

- A new array's contents are unspecified until a kernel, `tack.copyto` or
  `fill` writes them (the
  [language contract](../reference/language-contract.md#memory-and-aliasing));
  use `tack.zeros` for zeros. The CPU leaves a new array untouched, so that
  the kernel that writes it touches its pages first; the GPU backends
  currently zero new buffers, which is not a promise.
- On the CPU, a fill of 256 KiB or more runs as a kernel through the
  dispatch, in parallel when the threading policy fans it out, and stores
  and refuses exactly what `ndarray.fill` would.
- Allocation doesn't check the dtype against `supported_dtypes`. An `f64`
  array can be allocated on Metal, and the
  [dispatch check](backend-capabilities.md#what-each-capability-gates)
  rejects it when a kernel uses it.
- There is no `tack.fields()` function.
- Vector/matrix arrays expose their array shape and element dtype on
  both host and device. `numpy()` appends component dimensions.

**Required caller constraints:**

- `dtype` is a Tack scalar or element type, or anything `np.dtype()`
  accepts that maps to a Tack scalar type (`np.float32`, `"int64"`;
  `float` means f64, as in NumPy). Other values raise `TypeError`.
- Shapes contain integer, nonnegative extents. Noninteger extents raise
  `TypeError`; negative extents raise `ValueError`.

**Errors:**

| Condition | Exception |
|---|---|
| A dtype, given or inferred, that has no Tack type, such as `float16` | `TypeError`: "Unsupported numpy dtype: float16" |
| Boolean host values without a dtype | `TypeError`: "Tack has no boolean arrays; ..." |
| `empty_like` (or another `*_like`) with a prototype other than a NumPy array or Tack array | `TypeError`: "expected a Tack array or NumPy array as the prototype" |
| `concat([])` | `ValueError` |
| `concat` with mixed dtypes | `TypeError` |
| `texture3d` on an array that isn't `f32` | `ValueError`: "texture3d requires an f32 array, got f64; convert it first with array.astype(tack.f32)" |
| `texture3d` without a 3-D shape | `ValueError`: "texture3d requires a 3D shape" |
| `texture3d` with a shape whose `W*H*D` differs from the array's size | `ValueError` naming both element counts |
| `texture3d` with `interp` other than `"linear"` | `ValueError`: "texture3d supports interp='linear' only, ..." |

### Host operations

| Operation | Guarantee | Errors |
|---|---|---|
| `tack.copyto(a, src)` | With host `src` (a NumPy array, sequence or scalar): converts it to `a`'s scalar dtype with `np.asarray`, an unchecked cast, and copies it in. Its shape must be `a.numpy()`'s shape or broadcast to it, as in `np.copyto`; a flat array is not accepted for a vector or matrix array. Works on reshaped views. With a Tack array `src` of the same shape and scalar count on `a`'s device: a device copy (`src.to_device(a.device)` brings one from another device) | `RuntimeError` if `a` is read-only. `ValueError` "copyto() into an array whose values have shape (8,): cannot broadcast shape (4,)", for a Tack `src` of another shape, or "copyto() was given arrays on two devices ..." for one on another device. `TypeError` if `a` is not a Tack array |
| `a.numpy()`, `np.asarray(a)` | Returns a **new** NumPy array with the scalar component dtype and component axes appended to `a.shape`. Never a view, including on Metal; `np.asarray(a, copy=False)` raises `ValueError` | — |
| `a.fill(value)` | Sets every element to a scalar, or, for a vector or matrix array, to one element's components | `RuntimeError` if read-only. `TypeError` for a sequence on a scalar array; `ValueError` for a sequence of the wrong element shape |
| `a[i]`, `a[i, j]`, `a[()]` | Reads one element: a Python number, or a NumPy array for a vector or matrix element. `a[()]` reads a 0-d array, as in NumPy. There is no host assignment: `a[i] = x` and iteration raise `TypeError` | `TypeError` for anything but one integer per dimension, including `a[None]`, whose message names `a[()]`; `IndexError` out of range |
| `a.reshape(*shape)` | A metadata-only view that shares the buffer, the element type, the writability and any DLPack hold. Takes a tuple or separate integers; one extent may be `-1` | `ValueError` if the element count differs or more than one extent is `-1` |
| `a.copy()` / `a.astype(dtype)` | A new array on `a`'s device, filled by a device copy kernel. `astype` converts with the kernel language's conversion rules; a scalar dtype keeps a vector or matrix element's dimensions | Kernel errors, if any. `TypeError` if `astype` would change element dimensions |
| `a.shape`, `a.ndim`, `a.size`, `len(a)`, `a.nbytes` | `shape`, `ndim` and `size` count elements, not components. `len` is `shape[0]`. `nbytes` is the storage in bytes | `TypeError` from `len` of a 0-d array |
| `a.export_memory()` | Returns an `ExportedMemory` handle for cross-API sharing. On CUDA, an array allocated without `exportable=True` is copied **once** into exportable memory, so the export is a snapshot | `RuntimeError` on CPU, HIP and Level Zero |

**Current behavior:** on Metal, CUDA, HIP and Level Zero a kernel call
returns once the launch is queued, not once it has run. Host operations
keep this invisible: `numpy`, `a[i]` and reductions wait for the
launches before them; `copyto` and `fill` take effect after them (on
CUDA, HIP and Level Zero they are queued in order; on Metal they wait
first). A queued launch's error, such as a device fault, is raised at the
next wait rather than by its own call. An array dropped while a queued
launch uses it is freed after that launch.

**Required caller constraint:** an array is synchronized at kernel-call
and host-operation boundaries only. Host reads and writes between calls
are safe. Host access during a call from another thread is not. Memory
that another library reads or writes without a Tack call in between must
be ordered with `tack.sync()`, except where Tack does it: a DLPack export
and a wrapped pointer make every launch that uses them complete before it
returns (see [DLPack](#dlpack)).

### `tack.sync()`

Defined in `runtime/dispatch.py` (`sync`). Waits until every kernel
launched so far, on every device, has finished, by calling each backend's
`Backend.synchronize()`; a no-op on the CPU, whose launches complete before
returning. `d.synchronize()` waits for one device. Most code never needs it, since reading an array waits already.
It is for timing kernels, and for memory shared with another library that
reads it without Tack. It raises the error of a queued launch that
failed.

`tack.sync_async()` (`runtime/aio.py`) is the `asyncio` form for the
default device: awaited, it waits in one of Tack's worker threads for the launches made before the
call, not for later ones, and raises a queued launch's error at the
`await`; `await a.numpy_async()` does the same for a read. A backend
waits off the loop through `Backend.async_marker()`; without one (the
CPU), the wait is inline.

### Reductions

`f.sum()`, `f.min()`, `f.max()` and `f.mean()` reduce every element and
return a Python `float`. Integer arrays included.

| | `sum` | `min` / `max` | `mean` |
|---|---|---|---|
| Empty array | `0.0` | `ValueError`: "min reduction requires a nonempty array" | `nan` |
| Path | On the device when the array's device has `supports_device_reductions` and the array is `f32`. Otherwise NumPy on the host | same | `sum() / size` in host binary64 |

Accumulation precision, NaN and signed-zero rules, permitted reordering and
the error budget are **Required** items of the
[reduction contract](../reference/language-contract.md#array-and-parallel-reductions).
A reduction runs on the array's device, whatever the default.

## Pointer interop

### `tack.array_from_ptr(ptr, shape, dtype, *, device=None, writable=False)`

Wraps existing memory as an array without copying. `shape` counts
elements; `dtype` may be an element type such as `tack.vec3()`. Tack never frees it.

| Backend | What `ptr` must be |
|---|---|
| CPU | An integer address or a NumPy array |
| Metal | An `MTLBuffer` object. Any other value raises `TypeError` ("the Metal backend wraps an MTLBuffer object, not int ...") |
| CUDA, HIP | A device address: an `int`, or anything `int()` accepts, such as a NumPy integer or a `CUdeviceptr` |
| Level Zero | A USM device or shared address in the same forms, allocated in Tack's context |

**Guarantee:** on a backend that declares `device_memory_spaces` (CUDA,
HIP and Level Zero), every pointer is converted with `as_address`
(`runtime/kernel_utils.py`) and classified with `memory_space()`. A value
that isn't an address raises `TypeError` ("array_from_ptr() on the CUDA
backend takes a device address ..."), and a pointer outside those spaces
raises `ValueError` ("Pointer is in 'cpu' memory but the device is
'CUDA'. ..."). On Level Zero the driver answers for pointers from any
context on the device, so the check can't detect a pointer from a
different context.

**Guarantee:** an array created with `writable=False`, the default, is
read-only on the host and in kernels. `fill()` and `tack.copyto()` raise
`RuntimeError`. A dispatch that binds it to a parameter the kernel may
store to, or use as an atomic target, raises `ValueError` before the
launch: "Kernel '`k`': parameter '`out`' may be written, but its array is
read-only. ...". A parameter whose stores can't be traced counts as
written.

**Required caller constraints:**

- Keep the memory alive and unmoved for the array's lifetime.
- Provide at least `prod(shape) * itemsize` contiguous bytes.

### `tack.memory_space(ptr, *, device=None)`

Delegates to `device`'s `memory_space()` (by default the default device's). CPU and Metal always
answer `"cpu"`. CUDA answers `"cuda"`, `"cuda_pinned"`, `"cuda_managed"` or
`"cpu"`. HIP answers `"hip"`, `"hip_pinned"`, `"hip_managed"` or `"cpu"`.
Level Zero answers `"level_zero"` or `"cpu"`. A value that can't be a
64-bit address answers `"cpu"` without querying the driver. On Level Zero,
a failed driver query raises `RuntimeError` instead of being reported as
host memory.

### DLPack

Defined in `packages/tack-core/src/tack/lang/dlpack.py`.

**Export:** `Array.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)`
and `Array.__dlpack_device__()`.

- Zero-copy. The capsule keeps the array alive until the consumer calls
  the deleter.
- A consumer passing `max_version >= (1, 0)` receives a versioned capsule
  that carries the read-only flag for non-writable arrays. The legacy
  capsule can't express read-only.
- The device type is `kDLCPU` on CPU and Metal, `kDLCUDA` on CUDA,
  `kDLROCM` on HIP, and `kDLOneAPI` on Level Zero. The device id is always
  `0`.
- Strides are C-contiguous. All ten Tack dtypes are exported.
- `copy=True` raises `BufferError`. `stream` and `dl_device` are ignored.
- On Metal, CUDA, HIP and Level Zero the export waits for the launches
  queued so far, and from then on every launch that uses the array
  completes before it returns (`DeviceBuffer.share()`), so the consumer
  may read the memory at once and after later launches.

**Import:** `tack.from_dlpack(source, *, device=None, copy=False)`. The array
belongs to `device`, by default the default device.

- `source` is a capsule or an object with `__dlpack__`. A versioned
  capsule is requested first.
- With `copy=False`, the array shares the tensor's memory and holds the
  capsule until the array and every view of it are collected. Then the
  producer's deleter runs exactly once.
- A read-only versioned tensor produces a non-writable array, which
  host writes and kernel stores refuse as for `array_from_ptr`.
- As for `array_from_ptr`, a launch that uses the imported array completes
  before it returns.
- With `copy=True`, the source is read through `np.from_dlpack` and copied
  into a new array with `tack.array`. That works only for sources that
  NumPy can read on the host.

| Condition | Exception |
|---|---|
| Capsule already consumed | `ValueError` |
| Null data pointer | `ValueError` |
| Non-C-contiguous strides | `ValueError`: "only C-contiguous tensors can be wrapped without copying; ..." |
| DLPack dtype without a Tack type, or `lanes != 1` | `TypeError` |
| Unknown device type | `ValueError` |
| `kDLMetal` tensor | `ValueError`: "Metal DLPack tensors (kDLMetal) cannot be imported: ..." |
| Device type that `device` can't wrap | `RuntimeError` naming the backend and the device type, followed by `dlpack_refusal_note` |
| Wrapped pointer outside `device_memory_spaces` | `ValueError` from `array_from_ptr` |

**Current behavior:** Metal has no zero-copy import. Host tensors
(`kDLCPU`) are refused with an explanation, because wrapping a host pointer
as an `MTLBuffer` without a copy needs page alignment. `kDLMetal` tensors
are refused up front on every backend, because their data is an opaque
`MTLBuffer` handle plus a byte offset, which nothing in Tack can wrap yet.
On Metal, use `copy=True` with a host tensor.

### VTK

`tack.interop.vtk` (package `tack-vis`) provides `vtk_to_array(vtk_array, flatten=False)`,
`array_to_vtk(array, n_components=None, name=None)` and `init_level_zero()`.
Both directions are zero-copy through DLPack and need VTK with
`vtkmodules.util.dlpack_support`. Without it they raise `RuntimeError`.
`array_to_vtk` validates the layout before importing VTK and raises
`ValueError` for a contradictory `n_components` or for arrays with more
than two dimensions.

**Required for Level Zero:** start Tack with
`tack.interop.vtk.init_level_zero()` instead of
`tack.set_default_device("level_zero")`. It makes a Level Zero device in
the context that VTK's device uses the default, and returns it, so that
Tack allocates there. Exporting an array of a Level Zero device in another
context, or importing while the default Level Zero device is in one,
raises `RuntimeError`. The
check is necessary because the driver accepts pointers from any context.
See [Interoperability](../design/interoperability.md).

## Inspection

### `tack.inspect(kernel, *args, mode="source")`

Defined in `packages/tack-core/src/tack/lang/inspect_kernel.py`. Runs the
compilation pipeline for these arguments on the device a call would run on
(see [launch placement](#where-arrays-live-and-kernels-run)) and returns
a string. It doesn't execute the kernel, doesn't compile a GPU variant, and
doesn't populate the backend's variant cache.

| `mode` | Returns |
|---|---|
| `"ir"` | Tack IR after resolution, inference, scalar localization, optimization and type annotation (`ir.dump`), before GPU scalar packing |
| `"source"` | The backend's source. On CPU, LLVM IR for the variant these arguments would select, including the disjoint-array specialization. On Metal, CUDA, HIP and Level Zero, MSL, CUDA C, HIP C or OpenCL C after scalar packing |
| `"optimized"` | CPU only: the LLVM module after the backend's O3 pipeline. On any other backend it raises `ValueError` ("inspect mode 'optimized' is CPU only: ..."), after the kernel's own checks have run, because the GPU backends' optimizers run inside vendor compilers whose output is never shown |

**Guarantees:** inspection enforces the same dtype, workgroup-support,
atomic, alignment, participation and workgroup launch-count checks as
dispatch, and it raises the frontend's errors. It records texture extents through the
backend's `_store_texture_shapes()`, so on HIP and Level Zero devices
without texture hardware the source it shows uses software sampling, as
dispatch would. See
[the note on inspection](backend-capabilities.md#what-each-capability-gates)
for the dispatch checks it skips.

**Errors:** `TypeError` "Expected a @tack.kernel, got function" when the
first argument isn't a `Kernel`. `ValueError` "Unknown inspect mode:
'`ptx`'. Use 'ir', 'source', or 'optimized'." Unlike dispatch, inspection
doesn't wrap errors. For example, a workgroup primitive on CPU raises
`NotImplementedError` directly, and an unsupported array dtype raises the
dtype check's `TypeError`.

## Decorators

### `@tack.kernel`

Returns a `Kernel` (`lang/kernel.py`). Decoration reads the function's
source with `read_source` (`lang/func.py`, a wrapper around
`inspect.getsource`) and parses it. **Current behavior:**
nothing is validated or compiled at decoration. Source validation, name
resolution and lowering happen at the first dispatch or inspection, and
their errors are raised there.

| Condition | Exception, and when |
|---|---|
| Source isn't readable (`exec()`, a bare REPL, `python -c` before Python 3.13) | `RuntimeError` at decoration, which explains where kernels must be defined |
| Unsupported construct, including calls to functions that aren't statically bound `@tack.func`s or intrinsics | `UnsupportedSyntaxError` at first dispatch or inspection: "Kernel '`k`': unsupported Assert at line 4, column 9". Lines refer to the dedented captured source |
| Free name that is neither a parameter nor assigned | `NameError` at first dispatch or inspection, naming the kernel or inlined device function and its position |

### `@tack.func`

Returns a `Func` (`lang/func.py`) that is inlined at each call site in a
kernel. Device-function binding, return restructuring and recursion
rejection are **Required** items of the language contract. See
[Source and supported constructs](../reference/language-contract.md#source-and-supported-constructs).

| Condition | Exception |
|---|---|
| Decorating something that isn't a `def`, such as a lambda | `TypeError` at decoration |
| Source isn't readable | `RuntimeError` at decoration, the same explanation as for kernels, naming the device function. The `OSError` is chained |
| Calling the function from Python | `RuntimeError`: "@tack.func '`f`' cannot be called from Python. ..." |

### `@tack.device_class`

`@tack.device_class` (`lang/device_class.py`) marks a class so that its
instances can be passed as kernel arguments, and it collects the class's
`@tack.func` methods. When an instance is passed:

| Attribute | Becomes | Recompiles when changed? |
|---|---|---|
| Class-level `int` or `float` defined on the class itself | A compile-time constant | Yes. The value, with its type and float bit pattern, is part of the key |
| Instance attribute that shadows such a class attribute | A compile-time constant with the instance's value | Yes |
| Instance `int` or `float` | A runtime scalar parameter | Only when its inferred type changes. The attribute *names* are in the key, and the values are not |
| Instance `Array` | An extra array parameter | Only when dtype, shape or vector width changes |
| Names starting with `_`, and attributes of other types | Ignored | — |

Calling a method that isn't a `@tack.func`, or referencing a `self.`
attribute that is none of the above, raises `ValueError` at first
dispatch. Specializations are keyed on the class object itself, so two
classes with the same name don't share variants. When a class is garbage
collected, its variants are dropped.

Instances are recognized by argument type (the class's `_device_class`
flag), so a parameter that takes one needs no annotation.

## Calling a kernel

`Kernel.__call__(*args)` dispatches on the device its arrays choose (see
[launch placement](#where-arrays-live-and-kernels-run)) and returns
`None`, or, for a kernel that ends in an annotated `return`, that value,
read from a hidden one-element array (which waits for the launch). On the
CPU the call returns after the kernel completes; on the GPU backends once
it is queued (see [Host operations](#host-operations)).

**Arguments**, matched positionally against the parameters:

| Argument | Treated as |
|---|---|
| `Array` | An array parameter. Its dtype must be in `supported_dtypes` |
| `Texture3D` | A sampled texture parameter. The kernel samples the texture's own copy, not its array |
| `int`, `bool` or NumPy integer | A scalar: `i32` if it fits, else `i64`, else `u64`. Outside the `u64` range raises `TypeError` |
| `float` or NumPy floating | A scalar: `f64` if any array argument is `f64`, otherwise `f32`. Textures are always `f32`. A NumPy `float64` with only `f32` arrays becomes `f32` |
| `@tack.device_class` instance | Expanded as described above |
| Anything else | `TypeError`: "Unsupported argument type for parameter '`a`': <class 'str'>" |

**Current behavior:**

- Changing a scalar's *value* never recompiles. Changing its inferred
  *type*, an array's dtype or vector width, a baked shape dimension, or a
  texture extent compiles a new variant. See
  [Specialization and compilation identity](../reference/language-contract.md#specialization-and-compilation-identity).
- One kernel may be dispatched from several threads concurrently. Each
  thread's dispatch uses its own arguments. On CUDA and HIP, dispatches of
  the same compiled variant are serialized by a per-variant lock,
  `KernelVariant.dispatch_lock`, held from packing the scalars until the
  launch is queued, so different variants still dispatch concurrently.
  Metal launches own their argument buffer and scalars and are encoded
  one at a time under the backend's batch lock. Level Zero serializes all
  launches, reductions and copies on one backend lock (`_launch_lock`).
  CPU binds arguments per call and takes no lock, but fan-outs through its
  JIT'd pool run one at a time. `test_concurrent_dispatch.py` covers
  this.
- **Required caller constraint:** on CUDA, Tack's context is current only
  on the thread that made the device (the first `tack.device("cuda")`,
  `set_default_device("cuda")` or allocation on it). Another thread must make it
  current, for example with `cuCtxSetCurrent(backend._context)`, before
  dispatching. Otherwise the launch fails with
  `CUDA_ERROR_INVALID_CONTEXT`.

**Rejected:**

- **Keyword arguments.** `resolve_variant` raises
  `NotImplementedError("Keyword arguments not supported in kernels")`,
  which reaches the caller as `RuntimeError`: "Kernel '`k`' failed on
  CPUBackend: Keyword arguments not supported in kernels".
- **A wrong argument count.** `TypeError`: "Kernel '`k`' expects 2
  arguments, got 1". Arguments are counted after template expansion.
- **A read-only array where the kernel may store.** `ValueError` before
  the launch. See [Pointer interop](#pointer-interop).

## Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `TACK_CPU_THREADS` | `CPUBackend.__init__` | CPU worker count. Read each time a CPU backend is constructed, unless `tack.device("cpu", num_threads=...)` gives one. Values below 1 become 1. `1` runs everything on the calling thread. A non-integer raises `ValueError` from `tack.device` |
| `TACK_CPU_POLICY` | `CPUBackend.__init__` | Threading policy: `v1` or `v2`. Unset or empty means `v2`, the default. Any other value raises `ValueError` from `tack.device`, listing the accepted values |
| `TACK_CPU_MARGIN` | `CPUBackend.__init__` | Float override of the threading decision's safety margin |
| `TACK_CPU_POOL` | `CPUBackend.__init__` | `python` (or `0`, `off`, `no`, `false`) fans out through a `ThreadPoolExecutor` instead of the JIT'd pool, which is the default wherever it can be made (not on Windows) |
| `TACK_CPU_POOL_SPIN` | The JIT'd pool, when it is made | How many pause iterations a pool worker spins before sleeping (default 20000). For experiments |
| `TACK_CPU_MALLOC` | `CPUBackend.__init__` | **Linux.** A boolean flag read like `TACK_NO_REINIT`. When on, the first CPU backend sets glibc's `malloc` thresholds so freed blocks keep their pages (`cpu._tune_malloc`), process-wide. Off by default |
| `TACK_DEBUG` | `import tack` (`runtime/dispatch.py`) | A boolean flag read like `TACK_NO_REINIT`: turns on debug mode's bounds checks. `tack.set_debug` overrides it (above) |
| `TACK_NO_REINIT` | `tack.set_default_device` | A boolean flag (`dispatch.env_flag`): when on, `set_default_device` leaves an existing default alone. `0`, `false`, `no`, `off` and empty mean off. See above |
| `TACK_CACHE` | The disk cache, at the first kernel compiled | `write` (default), `read`, `bundle` (default under MPI) or `off`; also `0`, `false`, `no`, `none` for off. Any other value raises `ValueError`. See [The Kernel Cache](../users-guide/17-kernel-cache.md) |
| `TACK_CACHE_SIZE` | The disk cache, at the first kernel compiled | The most the entries of every Tack version may take: bytes, a size such as `500M` or `2G` (binary units), or `unlimited`; default `1G`. Past it, a process that stores entries deletes the least recently used, at most once an hour. Any other value raises `ValueError` |
| `TACK_CACHE_DIR` | The disk cache, at the first kernel compiled | The cache directory. Default: the platform's per-user cache directory. It must belong to the user and be writable by nobody else, or it is not used (with a `RuntimeWarning`) |
| `TACK_DUMP_MSL` | Metal kernel compilation | **Metal only.** A boolean flag read like `TACK_NO_REINIT`. When on, writes each compiled kernel's MSL to `/tmp/tack_<entry>.msl` and prints the path. A variant loaded from the disk cache is not generated, so not dumped. Other backends ignore it. Use `tack.inspect` instead |
| `TACK_REQUIRE_CLANG` | Test suite (`tests/compiler_tools.py`) | Any value other than empty or `0` turns missing or unusable Clang tooling into a test failure instead of a skip |
| `TACK_CLANG`, `TACK_CLANGXX` | Test suite | An explicit Clang driver for OpenCL/C and C++ host checks. An explicit choice that fails its preflight is a failure, and there is no fallback to `PATH` |
| `TACK_EXAMPLES_ARCH` | Test suite (`tests/test_examples.py`) | Backend for the slow example sweep. The default is `cpu` |

The CPU threading variables are tuning controls, not part of the language
contract. They change scheduling and never change results of race-free
programs. See [CPU Threading Policy](../design/cpu-threading.md).

### `tack.cache`

| Function | Effect |
|---|---|
| `configure(directory=None, mode=None, max_size=None)` | Sets the directory, mode and size limit for this process over `TACK_CACHE_DIR`, `TACK_CACHE` and `TACK_CACHE_SIZE`; `None` leaves one to the environment. Takes effect at the next kernel compiled |
| `settings()` | `(directory, mode, MPI rank or None)` as they apply now |
| `size_limit()` | The size limit in bytes as it applies now, or `None` for no limit |
| `load_bundle(path=None, comm=None)` | Reads a bundle (by default the directory's, for this Tack version) into the process. With an mpi4py communicator it is collective: rank 0 reads and broadcasts. Returns the number of entries |
| `pack(directory=None)` | Writes this Tack version's entries into one bundle file; returns `(path, count)` |
| `prune(directory=None, max_size=None)` | Deletes the least recently used entries of every Tack version until they take at most 90% of `max_size` (default: the size limit), and abandoned temporary files; keeps bundles. Returns `(files removed, bytes left)` |
| `clear(directory=None, all_versions=False)` | Deletes this Tack version's entries and bundle, or every version's (only the names the cache makes); returns the number of files removed |
| `describe(directory=None)` | The settings, entry count, size and bundle, as a dict |
| `stats()`, `reset_stats()` | Hit, miss and write counters for this process |

`python -m tack.cache info | pack | prune [--max-size SIZE] | clear [--all]` runs the same from a shell.

## Exceptions

Tack defines one exception class of its own, `UnsupportedSyntaxError`
(`lang/source_validation.py`). It subclasses `NotImplementedError`, which
in Python subclasses `RuntimeError`.

`Kernel.__call__` translates exceptions as follows:

- It re-raises `UnsupportedSyntaxError` unchanged.
- It re-raises a `TypeError` as a `TypeError` prefixed with
  `Kernel '<name>': `, unless the message already starts with that name,
  so the kernel is named once.
- It re-raises any other `RuntimeError`, including `NotImplementedError`,
  as `RuntimeError("Kernel '<name>' failed on <BackendClass>: ...")`,
  dropping an inner `Kernel '<name>': ` prefix. For compiler failures, it
  uses a short "failed to compile" form that lists the error lines.
- It passes everything else through unchanged, notably `ValueError` and
  `NameError`.

Translated exceptions keep the original as `__cause__`.

| Exception | Raised for | Where |
|---|---|---|
| `UnsupportedSyntaxError` | Syntax outside the kernel language, in a kernel or a device function, including unreachable statements; a kernel without exactly one parallel loop directly in its body; a store, atomic, barrier, block reduction or `print` outside the parallel loop, including through an inlined device function or template method | First dispatch or inspection (frontend) |
| `NameError` | A name that is neither a parameter nor assigned. Kernels don't capture Python globals | First dispatch or inspection (frontend) |
| `TypeError` | A device spec that is neither a string nor a `Device`, options with a `Device`, argument count or type, an integer outside 64 bits, an array dtype unsupported by the backend, an unsupported atomic dtype or target, the MSL generator receiving `f64`, `inspect` of a non-kernel, `@tack.func` on a non-function, a dtype argument that is neither a Tack type nor a NumPy dtype with a Tack type, `array`, `empty_like` or DLPack dtypes, an invalid `*_like` prototype, `concat` dtypes, `copyto` into a non-array, host element assignment or iteration, a non-`MTLBuffer` on Metal, an `array_from_ptr` pointer that isn't an address on CUDA, HIP or Level Zero | Dispatch (cold), inspection, host calls |
| `ValueError` | Unknown device type, device number or option, arrays on two devices in one launch, an unknown `TACK_CPU_POLICY`, incomplete `external_context`, workgroup participation, launch counts or device group size, a launch or reduction larger than one grid can index, atomic alignment, a read-only array the kernel may store to, `array_from_ptr` memory space, shape mismatches (including `copyto` shapes that do not broadcast), `reshape`, `texture3d` arguments, empty `min`/`max`, empty `concat`, unknown inspect mode or `"optimized"` off CPU, template method or attribute errors, malformed or `kDLMetal` DLPack tensors | Device creation, dispatch (cold or every call), inspection, host calls |
| `NotImplementedError` | Workgroup primitives on CPU, keyword arguments, and the few lowering checks that source validation doesn't already report as `UnsupportedSyntaxError` | Raised directly by inspection. Wrapped in `RuntimeError` at dispatch |
| `RuntimeError` | Device start-up failures, compilation and launch failures, unreadable kernel or device-function source, host writes to read-only arrays, calling device-only functions (`tack.shared`, `tack.atomic_add`, `@tack.func`s) from Python, DLPack device refusals, `export_memory` on unsupported backends, `tack.interop.vtk` requirements, and IR verification failures (`IRVerificationError`, a `RuntimeError` subclass, raised when a pass leaves IR that breaks a structural invariant) | Everywhere |
| `BufferError` | `__dlpack__(copy=True)` | Export |
| `IndexError` | In debug mode, a launch that indexed an array or array out of bounds; host indexing `f[i]` outside the array | After the launch, from `Kernel.__call__`; host calls |

**Current behavior:** every dispatch rejection listed on this page and in
[Backend capabilities](backend-capabilities.md) is raised before the kernel
launches, so a rejected call leaves array storage unchanged.
`test_workgroup_contract.py` tests this property for workgroup rejections,
`test_atomic_contract.py` for alignment, `test_launch_limits.py` for
oversized launches, and `test_outside_parallel_loop.py` for effects outside
the parallel loop.
