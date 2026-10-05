# Runtime API

This page is the contract for Tack's host-side surface: selecting a
backend, creating and moving fields, reductions, sharing memory with other
libraries, inspecting generated code, the decorators, and the rules for
calling a kernel. Each entry gives the signature, what is guaranteed, what
the caller must ensure, and which exception a violation raises. What a
kernel *computes* is defined by the
[kernel language contract](../reference/language-contract.md). What each
backend supports is defined in [Backend capabilities](backend-capabilities.md).

All statements describe release candidate `745e01f`. They use the labels
defined in [Contracts](index.md#how-to-read-a-contract).

## Backend selection

### `tack.init(arch="cpu", **options)`

Defined in `packages/tack-core/src/tack/runtime/dispatch.py` (`init`) and
re-exported as `tack.init`.

| | |
|---|---|
| `arch` | One of the strings `"cpu"`, `"metal"`, `"cuda"`, `"hip"` and `"level_zero"`. The constants `tack.cpu`, `tack.metal`, `tack.cuda`, `tack.hip` and `tack.level_zero` are those strings |
| `**options` | Forwarded to the backend constructor. Each option must be in that backend's `init_options`. Only Level Zero declares one, `external_context` |

**Current behavior:**

- `init` constructs a new backend and makes it the active one. It does this
  on every call, including when the same `arch` is already active.
- The new backend is built before the old one is released. If construction
  fails, the previous backend stays active.
- Fields belong to the backend that was active when they were allocated.
  **Required caller constraint:** re-allocate fields after calling `init`.
  When the new backend is a different kind, passing an old field to a
  kernel raises `RuntimeError`, which names the backends involved and tells
  you to re-allocate. `Kernel.__call__` diagnoses this from the
  `AttributeError` that the mismatched buffer produces. Re-initializing the
  same kind of backend isn't diagnosed.
- **Required caller constraint:** don't call `init` while another thread is
  dispatching.

**Errors:**

| Condition | Exception |
|---|---|
| Unknown `arch` | `ValueError`: "Unknown architecture: '`vulkan`'. Available: cpu, cuda, hip, level_zero, metal" |
| Missing Python dependency (an `ImportError` from the backend module or constructor) | `RuntimeError`: "Cannot initialize '`hip`' backend: missing dependency." followed by the original error and install instructions. The original exception is chained |
| Option not in `init_options` | `ValueError`: "The '`cpu`' backend does not accept the option(s) `num_threads`. Accepted: none." Raised after the backend module is imported, so a missing dependency is reported first |
| Device or runtime failure (a `RuntimeError` from the constructor) | `RuntimeError`: "Cannot initialize '`cuda`' backend on Linux." followed by the original message and install instructions. Chained |
| Any other constructor failure, such as `ValueError` for an incomplete `external_context` or a non-integer `TACK_CPU_THREADS` | Propagates unwrapped |

**`TACK_NO_REINIT`:** if a backend is already active and this variable is
set to *any non-empty value*, `init` returns immediately and does nothing.
It ignores `arch` and `options`, and it doesn't validate either. `"0"`
counts as set. This exists for embedding, for example inside an ANARI
device that shares the process. Unset it for testing, because it
suppresses backend switching.

### `get_backend()`

`tack.runtime.dispatch.get_backend()` returns the active `Backend`. If none
is active, it calls `init("cpu")` first. **Current behavior:** creating a
field or calling a kernel before `tack.init()` therefore silently selects
the CPU backend, and those fields become stale after a later
`tack.init(arch=...)`.

## Fields

`Field` is defined in `packages/tack-core/src/tack/lang/field.py`. A field
has a `dtype` (a Tack `ScalarType`), a `shape` (a tuple), and a buffer
owned by the backend that allocated it.

### Creation

| Function | Contract |
|---|---|
| `tack.field(dtype=f32, shape=(), exportable=False)` | Allocates on the active backend. An `int` shape becomes a 1-tuple. `exportable=True` requests memory that can be exported without a copy (CUDA only) |
| `tack.zeros(dtype, shape)`, `tack.ones(dtype, shape)`, `tack.full(dtype, shape, value)` | `field()` followed by `fill()` |
| `tack.arange(n, dtype=i32)` | `[0, n)` built in NumPy and copied in |
| `tack.field_like(arr, dtype=None)` | Allocates `arr.shape`. The dtype is inferred from `arr.dtype` unless given. Then copies `arr` in |
| `tack.concat(fields)` | A new 1-D field holding all elements in order, copied on the device |
| `tack.Vector.field(n, dtype=f32, shape=())` | A flat scalar field of `prod(shape) * n` elements that kernels index as `n`-component vectors |
| `tack.texture3d(field, shape=None, interp="linear")` | A `Texture3D` view of a field for `sample(u, v, w)` in kernels. See [Textures](backend-capabilities.md#textures) |

**Current behavior:**

- New fields are zero-filled on every backend.
- Allocation doesn't check the dtype against `supported_dtypes`. An `f64`
  field can be allocated on Metal, and the
  [dispatch check](backend-capabilities.md#what-each-capability-gates)
  rejects it when a kernel uses it.
- There is no `tack.fields()` function.
- A vector field's host-side `shape`, `size` and `len()` are those of the
  flat storage. Inside kernels, `v.shape[0]` and `len(v)` use the logical
  element count.

**Required caller constraints:**

- `dtype` is a Tack type such as `tack.f32`, not a NumPy dtype. This isn't
  validated. A NumPy type raises `AttributeError`.
- Shapes are non-negative. On CPU a negative extent raises NumPy's
  `ValueError`.

**Errors:**

| Condition | Exception |
|---|---|
| `field_like` with a dtype that has no Tack type, such as `float16` | `TypeError`: "Unsupported numpy dtype: float16" |
| `concat([])` | `ValueError` |
| `concat` with mixed dtypes | `TypeError` |
| `texture3d` on a field that isn't `f32` or `f64` | `ValueError`: "texture3d requires f32 or f64 dtype" |
| `texture3d` without a 3-D shape | `ValueError`: "texture3d requires a 3D shape" |

### Host operations

| Operation | Guarantee | Errors |
|---|---|---|
| `f.from_numpy(arr)` | Copies `arr` into the field. If `arr.dtype` differs from the field's dtype, the array is converted first with NumPy `astype`, which is an unchecked cast | `RuntimeError` if the field is read-only. `ValueError` "Shape mismatch: field is (8,), got (4,)" unless `arr.shape == f.shape` exactly. `AttributeError` if `arr` isn't an ndarray |
| `f.to_numpy()` | Returns a **new** array with shape `f.shape` and the field's dtype. Never a view, including on Metal | — |
| `f.fill(value)` | Sets every element | `RuntimeError` if read-only |
| `f.reshape(shape)` | A metadata-only view that shares the buffer, the writability and any DLPack hold. Accepts an `int` | `ValueError` if the element count differs |
| `f.copy()` / `f.astype(dtype)` | A new field on the **active** backend, filled by a device copy kernel. `astype` converts with the kernel language's conversion rules | Kernel errors, if any |
| `f.size`, `len(f)` | `size` is the product of the extents. `len` is `shape[0]`, or `0` for `shape == ()` | — |
| `f.export_memory()` | Returns an `ExportedMemory` handle for cross-API sharing. On CUDA, a field allocated without `exportable=True` is copied **once** into exportable memory, so the export is a snapshot | `RuntimeError` on CPU, HIP and Level Zero |

**Current behavior (known defect):** `from_numpy` on a reshaped view
raises a NumPy `ValueError` ("could not broadcast") on CPU and Metal,
because the buffer keeps its allocated shape. Write through the original
field instead.

**Required caller constraint:** a field is synchronized at kernel-call
boundaries only. Dispatch is synchronous on every backend, so a kernel has
completed when its call returns. Host reads and writes between calls are
safe. Host access during a call from another thread is not.

### Reductions

`f.sum()`, `f.min()`, `f.max()` and `f.mean()` reduce every element and
return a Python `float`. Integer fields included.

| | `sum` | `min` / `max` | `mean` |
|---|---|---|---|
| Empty field | `0.0` | `ValueError`: "min reduction requires a nonempty field" | `nan` |
| Path | On the device when the **active** backend has `supports_device_reductions` and the field is `f32`. Otherwise NumPy on the host | same | `sum() / size` in host binary64 |

Accumulation precision, NaN and signed-zero rules, permitted reordering and
the error budget are **Required** items of the
[reduction contract](../reference/language-contract.md#field-and-parallel-reductions).
**Required caller constraint:** reduce a field only while its backend is
active.

## Pointer interop

### `tack.field_from_ptr(ptr, dtype, shape, writable=False)`

Wraps existing memory as a field without copying. Tack never frees it.

| Backend | What `ptr` must be |
|---|---|
| CPU | An integer address or a NumPy array |
| Metal | An `MTLBuffer` object. Any other value raises `TypeError` ("the Metal backend wraps an MTLBuffer object, not int ...") |
| CUDA, HIP | A device pointer as an integer |
| Level Zero | A USM device or shared pointer as an integer, allocated in Tack's context |

**Guarantee:** when `ptr` is a Python `int` and the backend declares
`device_memory_spaces`, the pointer is classified with `memory_space()`. A
pointer outside those spaces raises `ValueError` ("Pointer is in 'cpu'
memory but the active backend is 'CUDA'. ..."). **Current behavior:** other
pointer types, such as NumPy integers and `CUdeviceptr`, skip this check.
On Level Zero the driver answers for pointers from any context on the
device, so the check can't detect a pointer from a different context.

**Required caller constraints:**

- Keep the memory alive and unmoved for the field's lifetime.
- Provide at least `prod(shape) * itemsize` contiguous bytes.
- `writable=False` is enforced **only on the host**: `fill()` and
  `from_numpy()` raise `RuntimeError`. Kernels can still store to a
  read-only field. Don't pass one as a kernel output.

### `tack.memory_space(ptr)`

Delegates to the active backend's `memory_space()`. CPU and Metal always
answer `"cpu"`. CUDA answers `"cuda"`, `"cuda_pinned"`, `"cuda_managed"` or
`"cpu"`. HIP answers `"hip"`, `"hip_pinned"`, `"hip_managed"` or `"cpu"`.
Level Zero answers `"level_zero"` or `"cpu"`. A value that can't be a
64-bit address answers `"cpu"` without querying the driver. On Level Zero,
a failed driver query raises `RuntimeError` instead of being reported as
host memory.

### DLPack

Defined in `packages/tack-core/src/tack/lang/dlpack.py`.

**Export:** `Field.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)`
and `Field.__dlpack_device__()`.

- Zero-copy. The capsule keeps the field alive until the consumer calls
  the deleter.
- A consumer passing `max_version >= (1, 0)` receives a versioned capsule
  that carries the read-only flag for non-writable fields. The legacy
  capsule can't express read-only.
- The device type is `kDLCPU` on CPU and Metal, `kDLCUDA` on CUDA,
  `kDLROCM` on HIP, and `kDLOneAPI` on Level Zero. The device id is always
  `0`.
- Strides are C-contiguous. All ten Tack dtypes are exported.
- `copy=True` raises `BufferError`. `stream` and `dl_device` are ignored.

**Import:** `tack.from_dlpack(source, copy=False)`.

- `source` is a capsule or an object with `__dlpack__`. A versioned
  capsule is requested first.
- With `copy=False`, the field shares the tensor's memory and holds the
  capsule until the field and every view of it are collected. Then the
  producer's deleter runs exactly once.
- A read-only versioned tensor produces a non-writable field.
- With `copy=True`, the source is read through `np.from_dlpack` and copied
  into a new field with `field_like`. That works only for sources that
  NumPy can read on the host.

| Condition | Exception |
|---|---|
| Capsule already consumed | `ValueError` |
| Null data pointer | `ValueError` |
| Non-C-contiguous strides | `ValueError`: "only C-contiguous tensors can be wrapped without copying; ..." |
| DLPack dtype without a Tack type, or `lanes != 1` | `TypeError` |
| Unknown device type | `ValueError` |
| Device type that the active backend can't wrap | `RuntimeError` naming the backend and the device type, followed by `dlpack_refusal_note` |
| Wrapped pointer outside `device_memory_spaces` | `ValueError` from `field_from_ptr` |

**Current behavior:** Metal has no working zero-copy import. Host tensors
(`kDLCPU`) are refused with an explanation, because wrapping a host pointer
as an `MTLBuffer` without a copy needs page alignment. `kDLMetal` is
accepted by the device table, but the pointer that reaches
`MetalBackend.wrap_ptr` is an integer, so the import raises `TypeError`. On
Metal, use `copy=True`.

### VTK

`tack.interop.vtk` (package `tack-vis`) provides `vtk_to_field(vtk_array, flatten=False)`,
`field_to_vtk(field, n_components=None, name=None)` and `init_level_zero()`.
Both directions are zero-copy through DLPack and need VTK with
`vtkmodules.util.dlpack_support`. Without it they raise `RuntimeError`.
`field_to_vtk` validates the layout before importing VTK and raises
`ValueError` for a contradictory `n_components` or for fields with more
than two dimensions.

**Required for Level Zero:** start Tack with
`tack.interop.vtk.init_level_zero()` instead of `tack.init()`, so that Tack
allocates in the context that VTK's device uses. Exchanging arrays while
the Level Zero backend runs in another context raises `RuntimeError`. The
check is necessary because the driver accepts pointers from any context.
See [Interoperability](../design/interoperability.md).

## Inspection

### `tack.inspect(kernel, *args, mode="source")`

Defined in `packages/tack-core/src/tack/lang/inspect_kernel.py`. Runs the
compilation pipeline for these arguments on the active backend and returns
a string. It doesn't execute the kernel, doesn't compile a GPU variant, and
doesn't populate the backend's variant cache.

| `mode` | Returns |
|---|---|
| `"ir"` | Tack IR after resolution, inference, scalar localization, optimization and type annotation (`ir.dump`), before GPU scalar packing |
| `"source"` | The backend's source. On CPU, LLVM IR for the variant these arguments would select, including the disjoint-field specialization. On Metal, CUDA, HIP and Level Zero, MSL, CUDA C, HIP C or OpenCL C after scalar packing |
| `"optimized"` | On CPU, the LLVM module after the backend's O3 pipeline. **Current behavior:** on GPU backends, the same text as `"source"`. Vendor compilers' output is never shown |

**Guarantees:** inspection enforces the same workgroup-support, atomic,
alignment, participation and launch-count checks as dispatch, and it raises
the frontend's errors. See
[the note on inspection](backend-capabilities.md#what-each-capability-gates)
for the dispatch checks it skips.

**Errors:** `TypeError` "Expected a @tack.kernel, got function" when the
first argument isn't a `Kernel`. `ValueError` "Unknown inspect mode:
'`ptx`'. Use 'ir', 'source', or 'optimized'." Unlike dispatch, inspection
doesn't wrap errors. For example, a workgroup primitive on CPU raises
`NotImplementedError` directly.

## Decorators

### `@tack.kernel`

Returns a `Kernel` (`lang/kernel.py`). Decoration reads the function's
source with `inspect.getsource` and parses it. **Current behavior:**
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
| Source isn't readable | The `OSError` from `inspect.getsource`, unwrapped, at decoration |
| Calling the function from Python | `RuntimeError`: "@tack.func '`f`' cannot be called from Python. ..." |

### `@tack.data_oriented` and `tack.template()`

`@tack.data_oriented` (`lang/data_oriented.py`) marks a class so that its
instances can be passed as kernel arguments, and it collects the class's
`@tack.func` methods. When an instance is passed:

| Attribute | Becomes | Recompiles when changed? |
|---|---|---|
| Class-level `int` or `float` defined on the class itself | A compile-time constant | Yes. The value, with its type and float bit pattern, is part of the key |
| Instance attribute that shadows such a class attribute | A compile-time constant with the instance's value | Yes |
| Instance `int` or `float` | A runtime scalar parameter | Only when its inferred type changes. The attribute *names* are in the key, and the values are not |
| Instance `Field` | An extra field parameter | Only when dtype, shape or vector width changes |
| Names starting with `_`, and attributes of other types | Ignored | — |

Calling a method that isn't a `@tack.func`, or referencing a `self.`
attribute that is none of the above, raises `ValueError` at first
dispatch. Specializations are keyed on the class object itself, so two
classes with the same name don't share variants. When a class is garbage
collected, its variants are dropped.

`tack.template()` returns a marker for use as a parameter annotation. It is
documentation only. Instances are recognized by the class's
`_data_oriented` flag, not by the annotation.

## Calling a kernel

`Kernel.__call__(*args)` dispatches on the active backend, returns `None`,
and returns only after the kernel completes.

**Arguments**, matched positionally against the parameters:

| Argument | Treated as |
|---|---|
| `Field` | A field parameter. Its dtype must be in `supported_dtypes` |
| `Texture3D` | A sampled texture parameter |
| `int`, `bool` or NumPy integer | A scalar: `i32` if it fits, else `i64`, else `u64`. Outside the `u64` range raises `TypeError` |
| `float` or NumPy floating | A scalar: `f64` if any field or texture argument is `f64`, otherwise `f32`. A NumPy `float64` with only `f32` fields becomes `f32` |
| `@tack.data_oriented` instance | Expanded as described above |
| Anything else | `TypeError`: "Unsupported argument type for parameter '`a`': <class 'str'>" |

**Current behavior:**

- Changing a scalar's *value* never recompiles. Changing its inferred
  *type*, a field's dtype or vector width, a baked shape dimension, or a
  texture extent compiles a new variant. See
  [Specialization and compilation identity](../reference/language-contract.md#specialization-and-compilation-identity).
- One kernel may be dispatched from several threads concurrently.
  `test_concurrent_dispatch.py` covers this.

**Rejected:**

- **Keyword arguments.** `resolve_variant` raises
  `NotImplementedError("Keyword arguments not supported in kernels")`,
  which reaches the caller as `RuntimeError`: "Kernel '`k`' failed on
  CPUBackend: Keyword arguments not supported in kernels".
- **A wrong argument count.** `TypeError`: "Kernel '`k`': Kernel '`k`'
  expects 2 arguments, got 1". Arguments are counted after template
  expansion.

## Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `TACK_CPU_THREADS` | `CPUBackend.__init__` | CPU worker count. Read each time a CPU backend is constructed. Values below 1 become 1. `1` runs everything on the calling thread. A non-integer raises `ValueError` from `tack.init` |
| `TACK_CPU_POLICY` | `CPUBackend.__init__` | Threading policy. `v2` is the default. `v1` restores the earlier policy. Any value other than `v2` currently selects the v1 code paths |
| `TACK_CPU_MARGIN` | `CPUBackend.__init__` | Float override of the threading decision's safety margin |
| `TACK_NO_REINIT` | `tack.init` | Any non-empty value makes `init` a no-op while a backend is active. See above |
| `TACK_DUMP_MSL` | Metal kernel compilation | **Metal only.** Writes each compiled kernel's MSL to `/tmp/tack_<entry>.msl` and prints the path. Other backends ignore it. Use `tack.inspect` instead |
| `TACK_REQUIRE_CLANG` | Test suite (`tests/compiler_tools.py`) | Any value other than empty or `0` turns missing or unusable Clang tooling into a test failure instead of a skip |
| `TACK_CLANG`, `TACK_CLANGXX` | Test suite | An explicit Clang driver for OpenCL/C and C++ host checks. An explicit choice that fails its preflight is a failure, and there is no fallback to `PATH` |
| `TACK_EXAMPLES_ARCH` | Test suite (`tests/test_examples.py`) | Backend for the slow example sweep. The default is `cpu` |

The CPU threading variables are tuning controls, not part of the language
contract. They change scheduling and never change results of race-free
programs. See [CPU Threading Policy](../design/cpu-threading.md). Two vis
examples (`40_fe_isoline.py`, `41_fe_isosurface.py`) read a default
`--arch` from `Tack_ARCH`, spelled with that capitalization.

## Exceptions

Tack defines one exception class of its own, `UnsupportedSyntaxError`
(`lang/source_validation.py`). It subclasses `NotImplementedError`, which
in Python subclasses `RuntimeError`.

`Kernel.__call__` translates exceptions as follows:

- It re-raises `UnsupportedSyntaxError` unchanged.
- It re-raises a `TypeError` as a `TypeError` prefixed with
  `Kernel '<name>': `.
- It re-raises any other `RuntimeError`, including `NotImplementedError`,
  as `RuntimeError("Kernel '<name>' failed on <BackendClass>: ...")`. For
  compiler failures, it uses a short "failed to compile" form that lists
  the error lines.
- It diagnoses an `AttributeError` caused by stale fields as a
  `RuntimeError`.
- It passes everything else through unchanged, notably `ValueError` and
  `NameError`.

Translated exceptions keep the original as `__cause__`.

| Exception | Raised for | Where |
|---|---|---|
| `UnsupportedSyntaxError` | Syntax outside the kernel language, in a kernel or a device function, including unreachable statements | First dispatch or inspection (frontend) |
| `NameError` | A name that is neither a parameter nor assigned. Kernels don't capture Python globals | First dispatch or inspection (frontend) |
| `TypeError` | Argument count or type, an integer outside 64 bits, a field dtype unsupported by the backend, an unsupported atomic dtype or target, the MSL generator receiving `f64`, `inspect` of a non-kernel, `@tack.func` on a non-function, `field_like` or DLPack dtypes, `concat` dtypes, a non-`MTLBuffer` on Metal | Dispatch (cold), inspection, host calls |
| `ValueError` | Unknown arch or option, incomplete `external_context`, workgroup participation, launch counts or device group size, atomic alignment, `field_from_ptr` memory space, shape mismatches, `reshape`, `texture3d` arguments, empty `min`/`max`, empty `concat`, unknown inspect mode, template method or attribute errors, malformed DLPack tensors | Initialization, dispatch (cold or every call), inspection, host calls |
| `NotImplementedError` | Workgroup primitives on CPU, keyword arguments, and the few lowering checks that source validation doesn't already report as `UnsupportedSyntaxError` | Raised directly by inspection. Wrapped in `RuntimeError` at dispatch |
| `RuntimeError` | Initialization failures, compilation and launch failures, stale fields after a backend switch, unreadable kernel source, host writes to read-only fields, calling device-only functions (`tack.shared`, `tack.atomic_add`, `@tack.func`s) from Python, DLPack device refusals, `export_memory` on unsupported backends, `tack.interop.vtk` requirements, and IR verification failures (`IRVerificationError`, a `RuntimeError` subclass, for example a kernel without exactly one top-level parallel loop) | Everywhere |
| `BufferError` | `__dlpack__(copy=True)` | Export |
| `AttributeError` | Passing a NumPy dtype where a Tack type is expected, or a non-array to `from_numpy` or `field_like` | Host calls (unvalidated caller constraint) |

**Current behavior:** every dispatch rejection listed on this page and in
[Backend capabilities](backend-capabilities.md) is raised before the kernel
launches, so a rejected call leaves field storage unchanged.
`test_workgroup_contract.py` tests this property for workgroup rejections,
and `test_atomic_contract.py` tests it for alignment.
