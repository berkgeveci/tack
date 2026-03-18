# Architecture Overview

## Compilation Pipeline

When a `@pgc.kernel` is called, PGC runs this pipeline:

```
@pgc.kernel Python function
    → Template rewrite (if @pgc.data_oriented args)     [template_rewrite.py]
    → AST transform → PGC IR                            [ast_transform.py]
    → IR resolve (dimension sizes, texture shapes)       [ir_resolve.py]
    → Type inference (from actual arguments)              [type_inference.py]
    → IR optimize (LICM, copy prop, CSE)                 [ir_optimize.py]
    → IR type annotate (resolved types for codegen)      [ir_type_annotate.py]
    → Scalar packing (GPU backends only)                 [ir_pack_scalars.py]
    → Backend-specific codegen + dispatch
```

Each step mutates or annotates the IR in place. The compiled kernel is
cached by name + type signature, so subsequent calls skip the pipeline.

## Module Layout

```
src/pgc/
    __init__.py              # Public API: field, kernel, func, init, types
    lang/                    # Frontend: AST → IR
        kernel.py            # @pgc.kernel decorator, Kernel class
        func.py              # @pgc.func decorator, Func class, registry
        data_oriented.py     # @pgc.data_oriented decorator
        ast_transform.py     # Python AST → PGC IR (1156 lines, the largest)
        template_rewrite.py  # AST pre-pass for template parameters
        ir.py                # IR node definitions (25+ node types)
        ir_resolve.py        # Replace IRDimSize with constants
        ir_optimize.py       # LICM, copy propagation, CSE
        ir_type_annotate.py  # Annotate IRAssign with resolved types
        ir_pack_scalars.py   # Group scalar params into field buffers
        type_inference.py    # Annotate IR params from actual args
        types.py             # ScalarType: f32, i32, i64, etc.
        field.py             # Field, Texture3D, Vector
    codegen/                 # IR → target code
        llvm_gen.py          # → LLVM IR (for CPU backend)
        msl_gen.py           # → Metal Shading Language
        cuda_gen.py          # → CUDA C
        hip_gen.py           # → HIP C (extends cuda_gen)
        opencl_gen.py        # → OpenCL C (extends cuda_gen)
    runtime/                 # Dispatch and device management
        dispatch.py          # pgc.init(), backend selection
        cpu.py               # CPU backend (llvmlite JIT, thread pool)
        metal.py             # Metal backend (pyobjc)
        cuda_backend.py      # CUDA backend (cuda-python)
        hip_backend.py       # HIP backend (hip-python)
        level_zero_backend.py # Level Zero backend (ctypes)
```

## Code Size

The entire framework is ~12,000 lines of Python:

| Directory | Lines | Role |
|-----------|-------|------|
| `lang/` | ~3,300 | Frontend: AST transform, IR, passes |
| `codegen/` | ~4,500 | 6 code generators |
| `runtime/` | ~4,400 | 6 backend runtimes |

No C/C++ code. No build step. Everything is pure Python with JIT
compilation at runtime.

## Key Design Decisions

**Pure Python throughout.** Unlike Taichi (which has a C++ core with pybind11
bindings), PGC implements the entire pipeline in Python. This makes the
codebase easy to read, modify, and debug. The performance cost is negligible
— JIT compilation is a one-time cost per kernel, and the generated GPU code
is equally fast.

**Separate IR from codegen.** The IR is a simple tree of Python objects
(not strings, not LLVM objects). Each codegen backend walks the same IR and
emits its own output format. This makes adding a new backend straightforward.

**AST-level function inlining.** `@pgc.func` calls are inlined by
copy-and-rename at the AST level, before IR transformation. This avoids
needing function call support in the IR or any backend.

**Template expansion at AST level.** `@pgc.data_oriented` objects are
resolved by rewriting the kernel AST before IR transformation. Scalar
attributes become constants, field attributes become parameters, and method
calls become inlined function bodies. The IR and codegens never see
templates.

**Non-owning pointer interop.** `pgc.field_from_ptr()` wraps external
device memory without allocation or copy. Each backend implements
`wrap_ptr()` with an `_owned = False` flag to prevent deallocation.
Fields are read-only by default for safety, with an explicit `writable`
opt-in. This enables interop with in-situ frameworks (Catalyst), GPU
libraries (pycuda, cupy), and cross-library buffer sharing.
