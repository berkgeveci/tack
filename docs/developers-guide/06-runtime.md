# Runtime and Dispatch

## Backend Lifecycle

Each backend follows the same lifecycle:

```
pgc.init(arch=pgc.metal)
    → dispatch.py creates MetalBackend()
    → Backend discovers device, creates command queue / context

kernel(x, y, out, alpha, n)
    → Kernel.__call__
    → backend.execute(kernel, args, kwargs)
    → Template expansion + IR pipeline + codegen + dispatch
```

## Backend.execute() Flow

All 5 backends follow the same flow in their `execute()` method:

```python
def execute(self, kernel, args, kwargs):
    # 1. Detect and expand template arguments
    template_args = _detect_template_args(kernel, args)
    effective_args = _expand_template_args(args, template_args)

    # 2. Detect vector and texture fields
    vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)
    texture_fields = _detect_texture_fields(kernel, args, template_args)

    # 3. Get IR (cached by kernel + specialization key)
    ir_module = kernel.get_ir(vector_fields, template_args, texture_fields)
    ir_func = ir_module.functions[0]

    # 4. Resolve, type inference, type checking, optimization
    resolve_ir(ir_func, name_to_field)
    infer_param_types(ir_func, effective_args)
    check_dispatch_types(ir_func, effective_args, supported_dtypes, backend_name)
    optimize_ir(ir_func)

    # 5. Extract loop range BEFORE packing
    loop_end = _get_loop_range(ir_func, kernel_args)

    # 6. Cache check — compile on miss
    if cache_key not in self._cache:
        ir_func_copy = copy.deepcopy(ir_func)
        pack_scalars(ir_func_copy, effective_args)
        annotate_types(ir_func_copy)
        compiled = self._compile_kernel(ir_func_copy)
        pack_fields = _create_pack_fields(pack_info, effective_args, self)
        self._cache[cache_key] = (compiled, pack_info, pack_fields)

    # 7. Build dispatch args (update cached pack fields)
    _update_pack_fields(pack_fields, pack_info, effective_args)
    kernel_args = kept_field_args + pack_fields

    # 8. Dispatch
    compiled(kernel_args, loop_end)
```

## Loop Range Resolution

`_get_loop_range()` extracts the parallel for-loop bound from the IR and
resolves it against actual arguments. It supports:

- `IRConstant(N)` — literal bound
- `IRName("n")` — scalar parameter
- `IRFieldLoad(IRAttribute(IRName("x"), "shape"), IRConstant(0))` — `x.shape[0]`
- `IRAttribute(IRName("x"), "__len__")` — `len(x)`

This must run before scalar packing since packing removes scalar params.

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

The LLVM-JIT'd function is called via ctypes. For loop ranges > 1024
elements, work is split across physical CPU cores using a persistent
`ThreadPoolExecutor`. Each thread calls the compiled function with
a `(start, end)` sub-range.

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

`pgc.field_from_ptr()` wraps an existing device pointer as a Field without
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

`pgc.init()` wraps backend initialization errors with the architecture
name, platform, and install instructions.
