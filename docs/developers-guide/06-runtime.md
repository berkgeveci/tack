# Runtime and Dispatch

## The Backend Contract

Every backend subclasses `Backend` (`runtime/backend.py`), which declares
what a backend must implement and what callers may ask it.

Required: `allocate_field()`, `wrap_ptr()`, `execute()`.

Declared capabilities — read these instead of probing with `hasattr`:

| Attribute | Meaning |
|---|---|
| `name` | Arch identifier, matching `tack.init(arch=...)` |
| `display_name` / `label` | How to write the backend in a message |
| `supported_dtypes` | Scalar types accepted as field dtypes |
| `supports_f64` | Derived from `supported_dtypes` — never declared separately |
| `supports_device_reductions` | Whether `reduce_field()` runs on device; otherwise `Field.sum()` and friends fall back to numpy |
| `device_memory_spaces` | Memory spaces a pointer must be in for `field_from_ptr()`; empty means no check |

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
find or build a specialization:

1. Expand template arguments and detect vector and texture fields.
2. Obtain the pristine IR template for this specialization.
3. Infer argument dtypes and categories on a private parameter probe, record
   texture extents, and derive resolved shape dependencies.
4. Look up the variant in the backend's weakly keyed per-kernel cache.
5. On a miss, deep-copy the template, resolve dimensions, infer/check types,
   and run conservative copy propagation. The backend build callback then
   packs scalars where needed, annotates types, generates code, and compiles.
6. Resolve the launch range from the variant's IR for this dispatch, bind
   arguments (updating any scalar pack buffers), and execute.

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
on every dispatch (about a microsecond) and passes when no field the kernel
writes overlaps another field. Qualifying calls use a variant compiled with
`noalias` on its field pointers; the rest use the variant without it. Without
the promise LLVM must reload after every store, which costs 2-3x on x86 for
kernels that accumulate through a field in an inner loop.

Template classes appear in keys as a token rather than the class object.
When a `@tack.data_oriented` class is collected, a finalizer drops the IR and
the compiled variants specialized on it from every backend's cache.

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
| CUDA | `CUDABuffer` | Device pointer (`cuMemAlloc`) |
| HIP | `HIPBuffer` | Device pointer (`hipMalloc`) |
| Level Zero | `L0Buffer` | Device pointer (`zeMemAllocDevice`) |

`Field` wraps a buffer with dtype and shape metadata. `from_numpy()` /
`to_numpy()` handle host-device transfers (on Metal this is a memcpy
within unified memory; on CUDA/HIP/L0 it involves explicit copies).

## Kernel Dispatch

### CPU

The LLVM-JIT'd function is called via ctypes. The backend compares measured
serial work against measured thread fan-out cost, splitting worthwhile
ranges across a persistent `ThreadPoolExecutor`. Each thread calls the
compiled function with a `(start, end)` sub-range.

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

Encodes a compute command: `setBuffer` for each field, `dispatchThreads`
for the grid size. Textures use a separate binding namespace
(`setTexture_atIndex_`). Scalar pack buffers are regular Metal buffers.

### CUDA / HIP

Launches via `cuLaunchKernel` / `hipLaunchKernel` with a pointer array
of arguments. Grid size = `ceil(loop_end / 256)`, block size = 256.

### Level Zero

Sets kernel arguments via `zeKernelSetArgumentValue`. Dispatches via
`zeCommandListAppendLaunchKernel` on an immediate command list.
Textures use `zeImageCreate` + `image3d_t` on devices with sampler
hardware.

## Scalar Packing at Dispatch

Pack field buffers are allocated once during compilation and cached.
On subsequent calls, `_update_pack_fields()` just writes the new scalar
values into the existing device buffers via `from_numpy()`. This avoids
per-dispatch allocation overhead.

## Device Pointer Interop

`tack.field_from_ptr()` wraps an existing device pointer as a Field without
allocation or copy. Each backend implements `wrap_ptr(ptr, dtype, shape)`:

| Backend | `ptr` type | Implementation |
|---------|-----------|----------------|
| CPU | numpy array or int address | `np.frombuffer` view into existing memory |
| Metal | `MTLBuffer` object | Creates numpy view via `contents().as_buffer()` |
| CUDA | `CUdeviceptr` (int) | Stores pointer, skips `cuMemAlloc` |
| HIP | device pointer (int) | Stores pointer, skips `hipMalloc` |
| Level Zero | device pointer (int) | Stores `c_void_p`, skips `zeMemAllocDevice` |

### Ownership

Wrapped buffers set `_owned = False`. Buffer destructors (`__del__`) check
this flag to skip freeing external memory:

```python
def __del__(self):
    if hasattr(self, '_device_ptr') and getattr(self, '_owned', True):
        driver.cuMemFree(self._device_ptr)  # skipped for wrapped ptrs
```

### Read-Only Protection

`Field._writable` defaults to `True` for allocated fields and `False` for
`field_from_ptr()`. The `_check_writable()` method guards `from_numpy()`
and `fill()`. Kernel-level write protection is not enforced — the user is
responsible for not writing to read-only external memory.

## Error Handling

`Kernel.__call__` wraps backend errors:
- `TypeError` → includes kernel name
- Compilation failure → extracts error lines, suppresses full source dump
- `RuntimeError` → includes kernel name and backend class name

`tack.init()` wraps backend initialization errors with the architecture
name, platform, and install instructions.
