# Codegen

Each backend has a code generator that walks the PGC IR and emits
target-specific code. All codegens follow the same pattern: iterate over
IR nodes, emit text or LLVM IR.

## Backend Summary

| Backend | Codegen | Output | Key class |
|---------|---------|--------|-----------|
| CPU | `llvm_gen.py` | LLVM IR | `LLVMCodeGen` (uses llvmlite builder) |
| Metal | `msl_gen.py` | MSL source | `MSLCodeGen` |
| CUDA | `cuda_gen.py` | CUDA C source | `CUDACodeGen` |
| HIP | `hip_gen.py` | HIP C source | `HIPCodeGen(CUDACodeGen)` |
| Level Zero | `opencl_gen.py` | OpenCL C source | `OpenCLCodeGen(CUDACodeGen)` |

## Inheritance

HIP and OpenCL extend CUDA codegen because their device code syntax is
nearly identical:

- **HIP**: Same as CUDA (`blockIdx`, `threadIdx`, `__global__`), just adds
  `#include <hip/hip_runtime.h>`.
- **OpenCL**: Different qualifiers (`__kernel`, `get_global_id(0)`,
  `__local`, `barrier()`), overloaded math (no `f` suffix). Also maps
  `long long` → `long` for integer types.

Both override only the differences (kernel signature, type names, a few
expressions) and inherit everything else.

## C-Like Codegens (CUDA, MSL, OpenCL)

These share common patterns:

### Variable Declaration and Type Inference

When emitting `IRAssign`, the codegen needs a C type for the variable
declaration. It uses `_resolved_type` from the IR type annotation pass,
mapped to a C type string via the backend's type map (`_C_TYPE_MAP`,
`_MSL_TYPE_MAP`, `_OCL_C_TYPE_MAP`).

For expression nodes used in other contexts (index casts, printf format
selection, floor division), the codegen reads `node.dtype` directly.
The `_infer_c_type(node)` method checks `node.dtype` first and falls
back to `_local_vars` only for field pointer references (which have
`dtype=None`).

The `_resolved_type_to_c()` method maps `ScalarType` → C type string,
and is overridden by OpenCL to use `long` instead of `long long`.

### If-Branch Variable Hoisting

C requires variable declarations before use at the scope level. When an
`IRAssign` appears inside an `if` branch, the variable might not be
visible in the `else` branch or after the `if`. The codegen "hoists"
these declarations before the `if`:

```c
float temp;          // hoisted
if (cond) {
    temp = 3.14;
} else {
    temp = 2.71;
}
// temp is visible here
```

The hoisting uses `_find_assign_type` to determine the type, which
prefers `_resolved_type` from the annotation pass.

### Sequential For Loop Variables

Loop variables are always re-declared in the `for` header to handle
reuse of the same variable name in sibling loops (C block scoping):

```c
for (long k = 0; k < 8; k++) { ... }
for (long k = 0; k < 8; k++) { ... }  // re-declared, not "k = 0"
```

### Integer Types

GPU backends use 64-bit integers for loop variables and index arithmetic
to support grids with more than 2^31 elements:
- CUDA/HIP: `long long`
- MSL: `long`
- OpenCL: `long`

## LLVM Codegen (`llvm_gen.py`, 1,008 lines)

Uses llvmlite's `IRBuilder` to construct LLVM IR. Fields become pointer
arguments; scalars become value arguments. All allocas are placed in the
entry block for SSA dominance.

Key differences from C-like codegens:
- Types are LLVM types (`FloatType()`, `IntType(64)`, etc.)
- No variable declaration needed — LLVM uses SSA
- Uses `alloca` for mutable local variables
- `IRSharedAlloc` and `IRLocalAlloc` both map to stack allocas (no shared
  memory on CPU)

## MSL Codegen (`msl_gen.py`, 689 lines)

Generates Metal Shading Language for Apple GPUs. Notable features:

- Buffer bindings: `device float* name [[buffer(N)]]`
- Textures use a separate binding namespace: `texture3d<float> [[texture(N)]]`
- Threadgroup detection: scans IR for `IRSharedAlloc`/`IRBarrier` to decide
  between `dispatch_threads` (simple) and threadgroup-based dispatch
- Hardware texture sampling: `tex.sample(__samp__, float3(u, v, w))`
  with coordinate transform for PGC's texel-center convention
- Float atomic min/max via CAS loop on `atomic_uint`

## CUDA Codegen (`cuda_gen.py`, 643 lines)

Generates `extern "C" __global__` kernel functions. Thread index:
```c
long long __idx__ = blockIdx.x * blockDim.x + threadIdx.x;
if (__idx__ >= __n__) return;
```

Float atomic min/max use CAS-based helper functions emitted on demand.

## OpenCL Codegen (`opencl_gen.py`, 338 lines)

Extends CUDA with OpenCL syntax differences. Also handles:

- Hardware texture sampling: `read_imagef(image, sampler, coords)` when
  the device supports it (checked at runtime via `maxSamplers`)
- Software trilinear fallback: generates an inline helper function with
  the texture dimensions baked in as constants
