# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_cpu_jit.py    # run one test file
uv run pytest tests/test_hip.py        # run HIP backend tests (requires ROCm)
uv run pytest -k "test_saxpy"          # run tests matching a pattern
uv run python examples/validate_all.py # validation suite (CPU + GPU)
uv run python examples/01_hello_pgc.py --arch hip   # run example on specific backend
uv run python bench_cpu_vs_hip.py                    # CPU vs HIP/ROCm benchmark
```

All examples (01-20) accept `--arch cpu|metal|cuda|hip` to select the backend.

No build step — pure Python with JIT compilation at runtime.

## Architecture

PGC is a Python-first GPU compute framework inspired by Taichi. Kernels are decorated Python functions that compile at call time through this pipeline:

```
@pgc.kernel Python function
    → AST transform (ast_transform.py)
    → PGC IR (ir.py)
    → IR passes: resolve (ir_resolve.py) → type inference → optimize (ir_optimize.py)
    → Backend-specific codegen + dispatch
```

### Four codegen backends from PGC IR

| Backend | Codegen | Runtime compilation | Dispatch |
|---------|---------|-------------------|----------|
| CPU | `llvm_gen.py` → LLVM IR | llvmlite JIT | ctypes function call |
| Metal | `msl_gen.py` → MSL source | Metal API (pyobjc) | compute pipeline |
| CUDA | `cuda_gen.py` → CUDA C source | NVRTC → PTX | cuLaunchKernel |
| HIP | `hip_gen.py` → HIP C source (extends CUDA) | hipRTC → code object | hipLaunchKernel |

### Key abstraction: Field with DeviceBuffer

`Field` (in `lang/field.py`) holds a `DeviceBuffer` — the backend-specific storage:
- **CPU**: `NumpyBuffer` wraps a numpy array. CPU backend accesses via ctypes pointers.
- **Metal**: `MetalBuffer` — Metal shared buffer (zero-copy unified memory on Apple Silicon). The numpy array view points directly into Metal buffer memory.
- **CUDA**: `CUDABuffer` holds a device pointer (`cuMemAlloc`). Explicit host↔device copies on `from_numpy`/`to_numpy`.
- **HIP**: `HIPBuffer` holds a device pointer (`hipMalloc`). Explicit host↔device copies.

`pgc.field()` calls `backend.allocate_field()` to create the appropriate buffer type.

### Kernel execution flow

1. `Kernel.__call__` → `backend.execute(kernel, args)`
2. Detect template arguments (`@pgc.data_oriented` classes) and expand them
3. Detect vector fields and set up scalarization metadata
4. AST transform → IR, with dimension size resolution (`ir_resolve.py`)
5. Type inference (`infer_param_types`) — annotates params from actual args, sets `_is_field` flag
6. IR optimization: LICM, copy propagation, CSE (`ir_optimize.py`)
7. Codegen produces backend-specific code (cached by kernel name + type signature + template key)
8. Dispatch: CPU splits range across threads; GPU launches grid of threads

### IR structure (lang/ir.py)

The IR is a simple tree of nodes:

**Loops**: `IRParallelFor` (outermost, maps to thread parallelism), `IRSequentialFor` (inner loops, supports `step`), `IRWhile`, `IRBreak`, `IRContinue`

**Control flow**: `IRIf`, `IRIfExp` (ternary)

**Field access**: `IRFieldLoad`, `IRFieldStore`, `IRAtomicOp` (atomic_add/min/max)

**GPU primitives**: `IRSharedAlloc` (threadgroup memory), `IRBarrier` (sync), `IRThreadId` (local thread index)

**Expressions**: `IRBinOp`, `IRUnaryOp`, `IRCompare`, `IRBoolOp`, `IRCall` (math builtins), `IRCast`, `IRConstant`, `IRName`, `IRAttribute`, `IRDimSize`

**Other**: `IRAssign`, `IRReturn`, `IRPrint` (kernel debugging)

### IR passes

- **ir_resolve.py**: Replaces `IRDimSize` nodes with concrete constants from field shapes, and resolves `IRAtomicOp` sub-expressions
- **ir_optimize.py**: Three passes — Loop-Invariant Code Motion (LICM), copy propagation, Common Subexpression Elimination (CSE)
- **type_inference.py**: Annotates IR params with types from actual arguments. Fields get `_is_field=True`, scalars get `_is_field=False`. Float scalars map to `f32` (not `f64`) for GPU compatibility.

### Scalar kernel arguments

Kernels accept both fields and Python scalars (int, float) directly. The `_is_field` attribute on IR params controls codegen: fields become pointers, scalars become values. All four backends handle this distinction in their codegen and dispatch paths.

### Vector scalarization

`pgc.Vector.field(n, dtype, shape)` creates a flat scalar field of size `prod(shape) * n`. In kernels, `field[i]` expands to n component loads/stores. Vector operations (add, dot, cross, normalize) are scalarized at the IR level.

### @pgc.func inlining

Functions decorated with `@pgc.func` are inlined at the AST level into kernels. Supports return values, multi-return (tuple), nested inlining, and vector propagation. Variables are renamed with unique suffixes to avoid collisions.

### @pgc.data_oriented templates

Classes decorated with `@pgc.data_oriented` can be passed as template arguments. Scalar attributes become compile-time constants, field attributes become kernel parameters, and `@pgc.func` methods are inlined with `self` resolved.

### CPU threading threshold

CPU backend uses `ThreadPoolExecutor` only when loop range > 1024 elements. Below that, Python thread dispatch overhead outweighs the parallelism benefit.

## Kernel language features

- **Loops**: `for i in range(n)`, `for i in range(start, end)`, `for i in range(start, end, step)`, `for i, j in pgc.ndrange(w, h)`, `while`, `break`, `continue`
- **Math**: `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `exp2`, `log`, `log2`, `log10`, `floor`, `ceil`, `abs`, `min`, `max`, `pow`
- **Types**: `int()`, `float()` casts
- **Atomics**: `pgc.atomic_add(field, idx, val)`, `pgc.atomic_min(...)`, `pgc.atomic_max(...)`
- **GPU primitives**: `pgc.shared(dtype, size)`, `pgc.barrier()`, `pgc.thread_id()`
- **Debug**: `print("label:", value)` — emits printf on CPU/CUDA/HIP, no-op on Metal
- **Fields**: `field[i]`, `field[i, j]`, `field[None]`, `field.shape[0]`, `len(field)`
- **Reductions**: `field.sum()`, `field.min()`, `field.max()` — GPU-native on Metal, numpy on CPU

## Platform-specific dependencies

- **macOS (Metal)**: `pyobjc-framework-Metal`
- **Linux/Windows (CUDA)**: `cuda-python>=13.2`, NVIDIA driver + CUDA toolkit
- **Linux (HIP/ROCm)**: `hip-python`, ROCm toolkit
- **CPU-only**: `llvmlite`, `numpy`

## HIP backend notes

The HIP codegen (`hip_gen.py`) extends `CUDACodeGen` — HIP device code uses the same syntax as CUDA (`blockIdx`, `threadIdx`, `__global__`, `__shared__`, `__syncthreads`). The only difference is `#include <hip/hip_runtime.h>`. The runtime (`hip_backend.py`) uses `hip-python` bindings for hipRTC compilation and dispatch.

To install hip-python (packages are on **Test PyPI**, not regular PyPI):

```bash
uv pip install --prerelease=allow --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match \
  "hip-python~=7.1.0"
```

Then: `pgc.init(arch=pgc.hip)`.

**Known issue**: `hiprtcDestroyProgram` segfaults in hip-python 7.1 bindings. The backend skips this call (minor leak, mitigated by kernel caching).

## Do not mention Claude in git commits
