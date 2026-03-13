# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_cpu_jit.py    # run one test file
uv run pytest -k "test_saxpy"          # run tests matching a pattern
uv run python examples/validate_all.py # validation suite (CPU + GPU)
uv run python benchmarks/microbenchmarks.py --bench saxpy  # run specific benchmark
```

No build step — pure Python with JIT compilation at runtime.

## Architecture

PGC is a Python-first GPU compute framework. Kernels are decorated Python functions that compile at call time through this pipeline:

```
@pgc.kernel Python function
    → AST transform (ast_transform.py)
    → PGC IR (ir.py)
    → Backend-specific codegen + dispatch
```

### Three codegen paths from PGC IR

| Backend | Codegen | Runtime compilation | Dispatch |
|---------|---------|-------------------|----------|
| CPU | `llvm_gen.py` → LLVM IR | llvmlite JIT | ctypes function call |
| Metal | `spirv_gen.py` → SPIR-V binary → MSL (spirv-cross) | Metal API (pyobjc) | compute pipeline |
| CUDA | `cuda_gen.py` → CUDA C source | NVRTC → PTX | cuLaunchKernel |

### Key abstraction: Field with DeviceBuffer

`Field` (in `lang/field.py`) holds a `DeviceBuffer` — the backend-specific storage:
- **CPU**: `NumpyBuffer` wraps a numpy array. CPU backend accesses via ctypes pointers.
- **Metal**: Metal shared buffer (zero-copy unified memory on Apple Silicon). The numpy array view points directly into Metal buffer memory.
- **CUDA**: `CUDABuffer` holds a device pointer (`cuMemAlloc`). Explicit host↔device copies on `from_numpy`/`to_numpy`.

`pgc.field()` calls `backend.allocate_field()` to create the appropriate buffer type.

### Kernel execution flow

1. `Kernel.__call__` → `backend.execute(kernel, args)`
2. Backend runs type inference (`infer_param_types`) to annotate IR params from actual Field dtypes
3. Codegen produces backend-specific code (cached by kernel name + type signature)
4. Backend resolves parallel for-loop range from IR + actual field shapes
5. Dispatch: CPU splits range across threads; GPU launches grid of threads

### IR structure (lang/ir.py)

The IR is a simple tree of nodes. Key patterns:
- `IRParallelFor` — the outermost for-loop, maps to thread parallelism
- `IRSequentialFor` — inner loops, emitted as regular loops on all backends
- `IRFieldLoad`/`IRFieldStore` — array access (field[index])
- `IRAttribute` for `field.shape[0]` — resolved at dispatch time, not in codegen

### SPIR-V codegen specifics (spirv_gen.py)

The SPIR-V emitter writes raw binary (no text assembly). Notable implementation details:
- `_id_types` dict tracks integer vs float type of every SPIR-V result ID — critical for choosing correct opcodes (OpIMul vs OpFMul, etc.)
- `_func_vars` list collects OpVariable instructions for hoisting to the entry block (SPIR-V requirement)
- Structured control flow: OpLoopMerge must immediately precede the branch instruction

### CPU threading threshold

CPU backend uses `ThreadPoolExecutor` only when loop range > 4,000,000 elements. Below that, Python thread dispatch overhead outweighs the parallelism benefit.

## Platform-specific dependencies

- **macOS (Metal)**: `pyobjc-framework-Metal`, `spirv-cross` (brew), `spirv-val` (brew)
- **Linux (CUDA)**: `cuda-python>=13.2`, NVIDIA driver + CUDA toolkit
- **CPU-only**: `llvmlite`, `numpy`

## Do not mention Claude in git commits
