# Architecture Overview

## Compilation Pipeline

When a `@tack.kernel` is called, every backend's `execute()` goes through
`resolve_variant()` in `runtime/kernel_utils.py`, which runs this pipeline:

```
@tack.kernel Python function
  Once per IR template (vector widths, texture extents, template structure):
    → Template rewrite (if @tack.data_oriented args)     [template_rewrite.py]
    → Source validation (reject unsupported syntax)      [source_validation.py]
    → AST transform → Tack IR, @tack.func inlining       [ast_transform.py, call_bindings.py]
    → Verify 'lowered', cache the template on the Kernel [ir_verify.py, kernel.py]
  Once per compiled variant (cache miss), on a clone_ir copy of the template:
    → IR resolve (dimension sizes, texture shapes)       [ir_resolve.py]
    → Type inference (from actual arguments)             [type_inference.py]
    → Dispatch type check (backend dtypes)               [type_inference.py]
    → Scalar localization (params and outer locals)      [kernel_utils.py]
    → Atomic / workgroup checks                          [atomic_support.py, workgroup_*.py]
    → IR optimize (conservative copy propagation)        [ir_optimize.py]
    → Scalar packing (GPU backends only, on a copy)      [ir_pack_scalars.py]
    → IR type annotate (resolved types for codegen)      [ir_type_annotate.py]
    → Backend-specific codegen + native compile
  Every call:
    → Derive the variant key, look it up
    → Alignment, read-only and launch-size checks        [kernel_utils.py]
    → Bind texture storage and arguments, launch
```

`verify_ir()` checks the IR after each of these stages. The IR template
cached on the `Kernel` is never mutated; each new variant clones it and the
passes mutate the clone. Compiled variants are cached per backend and per
kernel object, under a key built from argument dtypes and field/scalar/texture
categories, vector widths, texture extents, template structure and constants,
the dimension sizes baked into the code, and (on the CPU) whether the field
arguments overlap. A repeat call with the same key runs no passes and no
verification. See [Compilation Pipeline](../design/compilation-pipeline.md)
and [Specialization and Caching](../design/specialization-and-caching.md)
for the full account.

## Package Structure

Tack is split into three packages in a monorepo:

```
packages/
    tack-core/                    # Core compute framework
        src/tack/
            __init__.py          # Public API + pkgutil.extend_path
            lang/                # Frontend: AST → IR
                kernel.py        # @tack.kernel decorator, Kernel class, IR template cache
                func.py          # @tack.func decorator, Func class
                data_oriented.py # @tack.data_oriented decorator
                source_validation.py # Reject unsupported syntax before lowering
                call_bindings.py # Static resolution of device-function calls
                ast_transform.py # Python AST → Tack IR
                template_rewrite.py  # AST pre-pass for template parameters
                ir.py            # IR node definitions (30+ node types)
                ir_traversal.py  # walk_ir, transform_ir, clone_ir
                ir_names.py      # Binding names, fresh_name
                ir_verify.py     # verify_ir: invariants at each pass boundary
                ir_resolve.py    # Replace IRDimSize with constants
                ir_optimize.py   # conservative copy propagation
                ir_type_annotate.py  # Annotate expressions with dtype
                ir_pack_scalars.py   # Group scalar params into field buffers
                type_inference.py    # Annotate IR params from actual args
                atomic_support.py    # Atomic target and dtype checks
                workgroup_support.py, workgroup_participation.py  # Workgroup checks
                inspect_kernel.py    # tack.inspect()
                types.py         # ScalarType: f32, i32, i64, etc.
                field.py         # Field, Texture3D, Vector
                dlpack.py        # DLPack import/export
            codegen/             # IR → target code
                identifiers.py   # Collision-free target identifiers
                llvm_gen.py      # → LLVM IR (for CPU backend)
                msl_gen.py       # → Metal Shading Language
                cuda_gen.py      # → CUDA C
                hip_gen.py       # → HIP C (extends cuda_gen)
                opencl_gen.py    # → OpenCL C (extends cuda_gen)
            runtime/             # Dispatch and device management
                backend.py       # Backend base class: declared capabilities
                kernel_utils.py  # resolve_variant(), variant keys, kernel caches
                dispatch.py      # tack.init(), backend selection
                cpu.py           # CPU backend (llvmlite JIT, thread pool)
                metal.py         # Metal backend (pyobjc)
                cuda_backend.py  # CUDA backend (cuda-python)
                hip_backend.py   # HIP backend (hip-python)
                level_zero_backend.py # Level Zero backend (ctypes)
            algorithms/          # General-purpose parallel primitives
                scan.py          # Parallel prefix sum (exclusive/inclusive)
                copy.py          # copy, fill_value
                stats.py         # Statistics and histograms

    tack-rendering/               # Path tracing renderer
        src/tack/
            rendering/
                camera.py        # PerspectiveCamera, OrthographicCamera
                canvas.py        # Canvas (framebuffer)
                scene.py         # Scene, Actor, PointLight
                bvh.py           # GPU BVH construction
                pathtrace.py     # Path tracing kernel

    tack-vis/                     # Scientific visualization algorithms
        src/tack/
            algorithms/          # Vis-specific algorithms
                flying_edges.py  # FlyingEdges isosurface
                compute_normals.py
                cell_to_point.py
                amr_blanking.py
            data/                # Data abstractions (placeholder)
                array_handle.py
                cell_set.py
            interop/             # External framework interop
                vtk.py           # VTK ↔ Tack zero-copy exchange
```

All three packages share the `tack` namespace via `pkgutil.extend_path`.
`tack-rendering` and `tack-vis` depend on `tack-core` but not on each other.

## Code Size

`tack-core` is about 16,700 lines of Python, including docstrings and
comments (`wc -l` over `packages/tack-core/src` at this release);
`tack-rendering` and `tack-vis` add about 3,800 and 1,900:

| Directory | Lines | Role |
|-----------|-------|------|
| `lang/` | ~6,000 | Frontend: validation, AST transform, IR, passes, verification |
| `codegen/` | ~3,900 | 5 code generators and shared helpers |
| `runtime/` | ~6,300 | 5 backend runtimes, variant resolution, CPU threading |

No C/C++ code. No build step. Everything is pure Python with JIT
compilation at runtime.

## Key Design Decisions

The decisions below are the structural ones. The principles behind them,
and the correctness rules the compiler follows, are in
[Design Principles](../design/principles.md).

**Pure Python throughout.** Unlike Taichi (which has a C++ core with pybind11
bindings), Tack implements the entire pipeline in Python. This makes the
codebase easy to read, modify, and debug. Kernels run as native code
generated by LLVM or the vendor compiler, not through Python; the Python
cost is in compilation, paid once per compiled variant, and in per-call
dispatch overhead.

**Separate IR from codegen.** The IR is a simple tree of Python objects
(not strings, not LLVM objects). Each codegen backend walks the same IR and
emits its own output format. This makes adding a new backend straightforward.

**AST-level function inlining.** `@tack.func` calls are inlined by
copy-and-rename at the AST level: during IR transformation, each call site
lowers a renamed copy of the callee's validated AST. This avoids needing
function call support in the IR or any backend.

**Template expansion at AST level.** `@tack.data_oriented` objects are
resolved by rewriting the kernel AST before IR transformation. Class-level
scalar attributes become constants, instance scalar attributes and field
attributes become parameters, and method calls become inlined function
bodies. The IR and codegens never see
templates.

**Non-owning pointer interop.** `tack.field_from_ptr()` wraps external
device memory without allocation or copy. Each backend implements
`wrap_ptr()` so that the wrapped memory is never freed by Tack (the
CUDA, HIP and Level Zero buffers carry an `_owned = False` flag).
Such fields are read-only by default, with an explicit `writable`
opt-in. The flag is checked by host-side `Field` methods and, for
kernels, at dispatch: a read-only field bound to a parameter the kernel
may store to is refused before the launch. This enables interop with in-situ frameworks
(Catalyst), GPU libraries (pycuda, cupy), and cross-library buffer
sharing. See [Interoperability](../design/interoperability.md).
