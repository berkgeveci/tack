# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Monorepo Structure

Tack is split into 3 packages under `packages/`:

| Package | Path | Contents |
|---------|------|----------|
| `tack-core` | `packages/tack-core/` | Kernels, fields, types, IR, codegen, backends, scan/copy |
| `tack-rendering` | `packages/tack-rendering/` | Path tracer (BVH, camera, scene, ColorTable) |
| `tack-vis` | `packages/tack-vis/` | Visualization algorithms (flying edges, normals, VTK interop) |

All share the `tack` namespace via `pkgutil.extend_path`.

**One catch worth knowing.** `extend_path` merges the *contents* of same-named directories, but only one `__init__.py` runs — the first found on the path. Both tack-core and tack-vis have a `tack/algorithms/` directory, so tack-core's `__init__.py` wins and is where the vis worklets are re-exported (guarded, so tack-core still works alone). Adding a second `algorithms/__init__.py` makes it dead code; `packages/tack-vis/tests/test_namespace.py` guards against that.

## Build & Test Commands

```bash
uv sync --extra cpu --extra dev                             # install everything the tests need
uv run pytest                                                # run all tests
uv run pytest packages/tack-core/tests/test_cpu_jit.py        # run one test file
uv run pytest packages/tack-core/tests/test_hip.py            # run HIP backend tests
uv run pytest -k "test_saxpy"                                # run tests matching a pattern
uv run python packages/tack-core/examples/validate_all.py     # validation suite
uv run python packages/tack-core/examples/01_hello_tack.py --arch hip  # example on backend
```

Examples accept `--arch` to select the backend. `09_shared_memory.py`
requires an explicit GPU choice (`metal|cuda|hip|level_zero`); CPU has no
workgroup execution model.

On this Mac, sandboxed processes cannot discover the Apple M1 Max GPU:
`MTLCreateSystemDefaultDevice()` returns `None` even with the bindings installed.
Run Metal hardware validation with GPU access outside that sandbox and explicitly
initialize `tack.metal` first. A sandboxed CPU-only pytest collection does not
validate Metal. Keep `TACK_NO_REINIT` unset for multi-backend runs.

No build step — pure Python with JIT compilation at runtime.

## Architecture

Tack is a Python-first GPU compute framework inspired by Taichi. Kernels are decorated Python functions that compile at call time through this pipeline:

```
@tack.kernel Python function
    → Source validation (source_validation.py)
    → AST transform (ast_transform.py)
    → Tack IR (ir.py)
    → IR passes: resolve (ir_resolve.py) → type inference → optimize (ir_optimize.py)
    → Backend-specific codegen + dispatch
```

### Codegen backends from Tack IR

| Backend | Codegen | Runtime compilation | Dispatch |
|---------|---------|-------------------|----------|
| CPU | `llvm_gen.py` → LLVM IR | llvmlite JIT | ctypes function call |
| Metal | `msl_gen.py` → MSL source | Metal API (pyobjc) | compute pipeline |
| CUDA | `cuda_gen.py` → CUDA C source | NVRTC → PTX | cuLaunchKernel |
| HIP | `hip_gen.py` → HIP C source (extends CUDA) | hipRTC → code object | hipModuleLaunchKernel |
| Level Zero | `opencl_gen.py` → OpenCL C source (extends CUDA) | libocloc → SPIR-V | zeCommandListAppendLaunchKernel |

### Key abstraction: Field with DeviceBuffer

`Field` (in `lang/field.py`) holds a `DeviceBuffer` — the backend-specific storage:
- **CPU**: `NumpyBuffer` wraps a numpy array. CPU backend accesses via ctypes pointers.
- **Metal**: `MetalBuffer` — Metal shared buffer (zero-copy unified memory on Apple Silicon). The numpy array view points directly into Metal buffer memory.
- **CUDA**: `CUDABuffer` holds a device pointer (`cuMemAlloc`). Explicit host↔device copies on `from_numpy`/`to_numpy`.
- **HIP**: `HIPBuffer` holds a device pointer (`hipMalloc`). Explicit host↔device copies.
- **Level Zero**: `L0Buffer` holds a device pointer (`zeMemAllocDevice`). Explicit host↔device copies via immediate command list.

`tack.field()` calls `backend.allocate_field()` to create the appropriate buffer type. `Field.__getitem__` reads one element on the host through `DeviceBuffer.read_range(start, count)`: CPU and Metal slice their views, CUDA copies just the range with one `cuMemcpyDtoH` and Level Zero with one `zeCommandListAppendMemoryCopy` on the immediate list, and HIP falls back to a whole-buffer `to_numpy()` (correct, slow; narrow it where it can be tested). `__setitem__` and `__iter__` raise.

### Backend contract

All five backends subclass `Backend` (`runtime/backend.py`), which declares the required methods (`allocate_field`, `wrap_ptr`, `execute`) and the capability attributes callers read instead of probing with `hasattr`: `name`, `display_name`/`label`, `supported_dtypes`, `supports_f64`, `supports_device_reductions`, `supports_workgroups`, `device_memory_spaces`.

`supports_workgroups` is false on CPU and true on GPU backends. CPU rejects
shared allocations, barriers, thread IDs and block reductions before variant
construction; public inspection and direct LLVM generation also reject them.
`lang/workgroup_support.py` discovers nested/inlined requirements, memoized
only on immutable frontend templates. Direct LLVM generation scans mutable
IR afresh. GPU targets bypass this rejection check. Local arrays, ordinary
atomics and host field reductions remain supported on CPU. This flag does
not itself prove participation or atomic type/scope support. GPU collectives
also require full 256-lane groups and conservative uniform participation
(`workgroup_participation.py`). Check variants before optimization/packing;
cached launches recheck counts without structural analysis. Inspection and
direct GPU generators validate participation too. Scalar-pack params carry
`_is_scalar_pack` to preserve uniformity. Metal pipeline and Level Zero
X/total device limits cannot silently shrink collective groups. MSL uses
structural feature discovery, including thread IDs in conditions. Explicit
user barriers fence shared and global field memory within the workgroup.
See `test_workgroup_contract.py`, `test_workgroup_participation.py` and the
language contract.

`Backend.supported_atomic_dtypes` declares add/min/max targets: CPU all
shipped scalar widths; CUDA/HIP i32/u32/i64/u64/f32/f64; Metal/Level Zero
only i32/u32/f32. `lang/atomic_support.py` checks global field targets,
including uniquely traced inlined pointer copies, before optimization and
in all inspection modes. Direct generators scan mutable IR afresh. Reject
private/shared arrays, textures, unsupported widths and ambiguous targets.
Atomic fields require natural alignment; `DeviceBuffer.address` exposes
current storage for the cold/cached checks. Variants retain target parameter
indices/alignments, not addresses; warm dispatch does not scan IR. CPU
floating extrema use real CAS loops; unsigned extrema use unsigned RMW.
CUDA/HIP wide operations use 64-bit CAS. OpenCL user atomics and field
reductions use explicit relaxed device scope, rather than legacy atomics'
workgroup-only guarantee. Values convert once to target precision and are
captured outside CAS retries. Atomics are relaxed, spanning CPU workers or
GPU workgroups on one device; they are not publication fences or host/device
system-scope operations. Floating extrema support finite values and numeric
zeros, with unspecified result sign on a zero tie; NaNs/infinities remain
outside their domain.
See `test_atomic_contract.py` and the language contract.

Rasterized points and wireframes use three completed passes per actor:
atomic depth reduction, atomic lowest-ID selection at the completed depth,
and one color writer per pixel. Never combine atomic depth updates with
ordinary depth reads and competing RGB stores in one geometry dispatch.
`rendering/rasterize.py` uses cached i32 winner scratch; equal-depth
primitives select the lowest index, while equal-depth actors overwrite in
scene order. See `test_raster_winners.py` for contention across workgroups,
forced CPU workers, ties, repeated frames and compositing.

Anything derivable is derived — `supports_f64` comes from `supported_dtypes`, so the two cannot disagree. Level Zero sets `supported_dtypes` in `__init__` because f64 depends on the device.

### Kernel execution flow

All five backends share one entry point: `resolve_variant()` in `runtime/kernel_utils.py`. It turns a call into a compiled variant, running the IR pass pipeline **only when that variant is new**.

Every dispatch:
1. `Kernel.__call__` → `backend.execute(kernel, args)` → `resolve_variant(...)`
2. Detect template arguments (`@tack.data_oriented` classes) and expand them
3. Detect vector fields and set up scalarization metadata
4. `kernel.get_ir(...)` returns the **pristine IR template** for this specialization; check required workgroup support against the backend
5. Type inference (`infer_param_types`) — annotates params from actual args, sets `_is_field`
6. Build the variant key and look it up (see below)
7. Check the call's storage against what the variant recorded: atomic target alignment, and no read-only field (`field_from_ptr`'s default, a read-only DLPack import) bound where the kernel may store (`written_fields`, from `written_field_params`)
8. Resolve the loop range from the variant's IR; dispatch (CPU decides serial vs threads, GPU launches a grid)

Only on a cache miss:
1. Deep-copy the template with `clone_ir()` — the passes below mutate IR in place and must not touch the template
2. Dimension size resolution (`ir_resolve.py`)
3. Dispatch-time type checking (`check_dispatch_types`) — validates field dtypes against the backend
4. IR optimization: conservative copy propagation (`ir_optimize.py`)
5. Backend `build` callback: scalar packing (GPU), type annotation (`ir_type_annotate.py`), codegen, compile

### Variant cache key

Keyed per `Kernel` (weakly, so compiled code is released with the kernel), then by:
argument type signature + field/scalar/texture categories + vector widths + texture extents + template structure/constants + **`shape_signature`** + (CPU only) whether the call's fields are disjoint. Template keys preserve actual class identity through a `_ClassToken` rather than the class object, so a `@tack.data_oriented` class that is garbage-collected takes its IR and compiled variants with it on every backend (a class defined per call still compiles per call, but no longer accumulates). Template keys also preserve typed constants, field metadata, and runtime scalar attribute names; runtime scalar values do not specialize.

That last one matters for correctness, not speed. `ir_resolve` substitutes dimension sizes as literals — `a[i, j]` linearizes to `i * dim1 + j` with `dim1` baked in — so the row stride is part of the compiled code's identity. `shape_signature()` reports exactly the dimensions a kernel bakes in (memoized per IR; empty for 1-D kernels, so varying a flat length does **not** re-specialize).

Because the passes mutate IR in place, the template from `get_ir()` must be treated as immutable — a pass that consumed its `IRDimSize` nodes would leave nothing for a later shape to resolve.

Fields may share storage, including distinct views and imported pointers. Preserve program order within each race-free iteration; field parameters must not carry unconditional `noalias`/`restrict` promises.

The CPU backend makes that promise *conditionally*. On every dispatch `fields_disjoint()` (`runtime/kernel_utils.py`) compares the byte ranges of the field arguments: the call is disjoint when no field the kernel stores to (`written_field_params`, memoized per IR) overlaps any other field. Read-only fields may overlap each other. The answer is the last element of the variant key, so a kernel holds at most two CPU variants per specialization: one compiled with `noalias` on its field pointers, used only for calls that passed the check, and one without. The check is based on storage ranges, never object identity, and costs about 1 µs per dispatch. `cpu._SPECIALIZE_DISJOINT = False` turns it off. GPU backends do not specialize: removing `__restrict__` cost nothing measurable on CUDA.

Metal field pointers are members of one argument buffer, rather than separate
device-buffer kernel arguments (which implicitly promise disjoint storage in
MSL). Members use the packed parameter positions as `[[id(N)]]` indices;
textures retain their separate binding namespace. Each cached dispatch refreshes
the buffer references and declares indirect-resource residency with
`useResource`. The encoder and argument buffer are reused after synchronous
completion. This fixes the four overlap cases confirmed at `922b642` and
`e265e7f`, without alias-based specialization or disabling vendor optimization.

A Metal kernel with a sequential loop that stores to a field (or runs an
atomic) has its body in `__tack_body__`, a `noinline` function the entry
calls (`_stores_inside_sequential_loop` in `msl_gen.py`); workgroup arrays
are declared in the entry and passed as pointers. Compiled inside the
entry function, such a loop gave wrong results: Apple's compiler read a
field element once before the loop when its address did not depend on the
thread and the loop stored to another same-typed field
(`total[0] += x[k]; counter[0] += 1`). The front end's LLVM IR is correct
and the fault is in the GPU back end at pipeline creation; alias metadata
is not the cause. Do not merge the two functions or drop `noinline`;
`restrict`, opaque pointers and compile options do not avoid it. Loop-free kernels keep one function (the call costs 7-15% on the
smallest). See `test_field_updates_in_loops.py` and
`docs/design/memory-and-aliasing.md`.

Dispatching one variant from several Python threads: CPU binds arguments
per call and shares no launch state. GPU variants do — the scalar pack
buffers, and Metal's argument buffer — so CUDA, HIP and Metal hold
`KernelVariant.dispatch_lock` from the pack update through the synchronous
launch. Level Zero holds one backend `_launch_lock` (reentrant) over
launches, reductions and copies, because its command lists are
backend-wide. Unlocked, half of the CUDA dispatches from four threads
computed with another thread's scalars (`test_concurrent_dispatch.py`).
CUDA's context comes from `cuCtxCreate` and is current only on the thread
that called `tack.init`; another thread must make it current first, or
its dispatch fails with `CUDA_ERROR_INVALID_CONTEXT`.

### Field dimensions

`field.shape[k]` and `len(field)` both lower to `IRDimSize` in `ast_transform.py`, which `ir_resolve.py` folds to a literal wherever it appears — loop bounds, conditions, arithmetic, indices. The dimension index must be a literal (`x.shape[d]` with a runtime `d` raises).

**One exception, and it matters:** the outermost parallel loop's bound is left unresolved. It never reaches generated code — codegen reads the `__loop_end__` parameter — so `_resolve_range_expr` evaluates it per dispatch instead, and `ir_shape_deps` excludes it from the variant key. Without that, `for i in range(x.shape[0])` — the most common line in any kernel — would compile a new variant for every array length.

Everywhere else the dimension *is* compiled in, so it specializes:

```python
for i in range(x.shape[0]):        # one variant for all lengths
    out[i] = x[x.shape[0] - 1 - i] # ...but this bakes the length in → one variant per length
```

To avoid that, pass the length as a scalar argument (`def reverse(x, out, n)`) — scalars are runtime parameters and don't specialize.

### Textures

`tack.texture3d(field, shape)` copies the field into storage the `Texture3D`
owns, on every backend, and `tex.update()` copies it again; writes to the
field in between do not reach the texture. `Backend.texture_in_hardware()`
picks the storage: a hardware image from `create_texture_image()` (CUDA,
Metal, HIP with image support, Level Zero with samplers, within the device's
3D limit) or a private f32 field sampled in software (CPU and the fallbacks).
Dispatch binds that storage through `bind_textures()`; the grid bound still
reads the source field. No compiled kernel caches texture objects, and none
is keyed by device address: images free their device objects when the
texture is collected. Only f32 fields of exactly W*H*D elements and
`interp='linear'` are accepted. Volume rendering calls `update()` per
`render_volume()`, since the path tracer reads `scalar_field` directly. See
`test_texture_snapshot.py`.

### Kernel code inspection

Generated GPU variables and all backend kernel entry names use the shared,
injective encoding in `codegen/identifiers.py`. The GPU generator renames
only a structural IR copy; canonical IR names and dispatch metadata remain
original. Runtime lookup must call `kernel_entry_name` on the original
function name exactly once. Lowering/templates/vector components/scalar
localization/packing allocate generated bindings through `fresh_name`
(`lang/ir_names.py`) so they cannot merge with Python source names.

`tack.inspect(kernel, *args, mode=...)` runs the compilation pipeline and returns the generated code as a string without executing. Modes: `"ir"` (Tack IR), `"source"` (backend code: LLVM IR / MSL / CUDA C / HIP C / OpenCL C), `"optimized"` (post-LLVM-O3 IR; CPU only, other backends raise `ValueError` after the kernel's own checks). It makes dispatch's decisions too: `check_dispatch_types` and the backend's `_store_texture_shapes` (software sampling on HIP/Level Zero devices without texture hardware). Implementation in `lang/inspect_kernel.py`.

### IR structure (lang/ir.py)

The IR is a simple tree of nodes:

**Loops**: `IRParallelFor` (outermost, maps to thread parallelism), `IRSequentialFor` (inner loops, supports `step`), `IRWhile`, `IRBreak`, `IRContinue`

**Control flow**: `IRIf`, `IRIfExp` (ternary)

**Field access**: `IRFieldLoad`, `IRFieldStore`, `IRAtomicOp` (atomic_add/min/max)

**GPU primitives**: `IRSharedAlloc` (threadgroup memory), `IRBarrier` (sync), `IRThreadId` (local thread index)

**Expressions**: `IRBinOp`, `IRUnaryOp`, `IRCompare`, `IRBoolOp`, `IRCall` (math builtins), `IRCast`, `IRConstant`, `IRName`, `IRAttribute`, `IRDimSize`

**Other**: `IRAssign`, `IRReturn`, `IRPrint` (kernel debugging)

### IR passes

- **ir_resolve.py**: Replaces `IRDimSize` nodes with concrete constants from field shapes, resolves `IRAtomicOp` sub-expressions, and resolves `shared_like` dtypes from fields
- **ir_optimize.py**: Conservative copy propagation for inlined arguments. Assignment counts are computed once per kernel and reused in nested blocks: the copy target must have one binding and its source must have no assignments, loop bindings, or allocations anywhere in the kernel. Each statement is tested after earlier copies are applied, so a chain resolves to its root in one walk; nested device functions pass a field down as such a chain, and an unresolved link is a local holding a field. This leaves some block-local copies to LLVM/vendor optimization and avoids repeated subtree counting during cold compilation. Custom LICM and CSE remain disabled because they lack memory/control-flow safety analysis.
- **type_inference.py**: Annotates IR params with types from actual arguments. Fields get `_is_field=True`, scalars get `_is_field=False`. Float scalars auto-promote to `f64` when any field arg uses `f64`; otherwise default to `f32`. Int scalars exceeding i32 range auto-promote to `i64`. `check_dispatch_types()` validates field dtypes against backend capabilities.
- **ir_type_annotate.py**: Sets `dtype` (a `ScalarType`) on every expression IR node. Codegens read `node.dtype` directly instead of reimplementing type inference heuristics.
- **ir_traversal.py**: Explicit structural child schema, preorder `walk_ir`, and postorder `transform_ir`. Resolution, scalar packing, copy substitution, and shape-dependency queries share it. Metadata is not traversed; unregistered node kinds fail loudly. `clone_ir` specializes deep copying for registered plain IR nodes and list/dict containers, copying every attribute (including metadata) with one identity memo. It preserves shared references, cycles and ScalarType identity; other metadata retains Python's deepcopy protocol. Variant preparation, scalar localization, GPU packing and inspection use it; cache hits do not clone IR. Keep all verifier boundaries. Compare copying costs with `uv run --no-sync python benchmarks/ir_clone.py --output /tmp/ir-clone.json`; this is a CPU component benchmark, not an end-to-end latency measurement.
- **ir_verify.py**: Checks lowered templates before caching and variant IR after resolve, infer, scalar localization, optimize, GPU packing, and annotation. Errors report kernel, stage, node kind, and tree path. Checks cover structure, bindings, loop targets, unresolved dimensions/allocation types/texture extents, parameter categories, and scalar annotations. Host-evaluated grid ends retain dimension queries; field pointers need no scalar dtype. Cache hits do not run the verifier. This is not definite-assignment, bounds, race, or barrier-uniformity analysis. Direct pass/codegen calls on IR fragments must request verification themselves.

### Scalar kernel arguments

Kernels accept both fields and Python scalars (int, float) directly. The `_is_field` attribute on IR params controls codegen: fields become pointers, scalars become values. All four backends handle this distinction in their codegen and dispatch paths.

### Vector scalarization

`tack.Vector.field(n, dtype, shape)` creates a flat scalar field of size `prod(shape) * n`. In kernels, `field[i]` expands to n component loads/stores. Vector operations (add, dot, cross, normalize) are scalarized at the IR level.

Comparisons with a vector operand lower per component to a mask
(`_vector_compare`; chains evaluate every link), `and`/`or` over masks
per component without short-circuiting, `any`/`all` to an `IRBoolOp` over
the components, and `tack.select` to one `IRIfExp` per component.
`_scalar_condition` rejects a mask as an `if`/`while` condition with the
remedy. See `test_vector_comparisons.py`.
A list of scalars in kernel source is a vector (`visit_List`), the same
value as `tack.Vector([...])`; the validator rejects an empty or nested
list, a list target and a loop over a list.
In `ast_transform.py` a vector value is a `VectorValue`, a list of
component expressions with a `shape`; a source tuple (indices, multiple
results) is a `TupleValue`. Neither enters the IR: `_check_single_values`
scans each lowered statement once and raises `UnsupportedSyntaxError` at
the statement's position for one left in a scalar position, instead of
the verifier's "expected expr node". Arithmetic, math builtins, casts
and conditional expressions map over components (`_componentwise`
repeats scalars and rejects mixed widths). An assignment lowers to one
assignment per component, so `_settle_components` first evaluates any
component that could observe an earlier one's write: a read of an earlier
target component for a variable, any field load for a field element
(fields may alias). Components are `v[c]`/`v.x`, on variables and on field
elements (`vf[i][c] = x` stores one component); a runtime index selects
through a chain and stores through per-component guards. A vector index
supplies one dimension per component, in subscripts and in atomics, whose
index may be a tuple with one entry per dimension; a vector value given
to an atomic on a vector field lowers to one atomic per component, while
a scalar value keeps the index a flat component index. Tuple assignment
accepts subscript and component targets through temporaries, stored at
their own position among the targets (left to right, after the whole
right side). `_linearize_index` tags each `IRDimSize` with the number of
indices given, and `ir_resolve` raises `TypeError` when that differs from
the field's dimension count; a single index is a flat index and is not
checked. A device
function returning several values yields a `TupleValue` whose vector
elements are `VectorValue`s; only tuple unpacking consumes that. `v.min()`/`v.max()`/
`v.sum()` are intercepted before the same-named builtins, which need
arguments. See the vector tests in `test_new_features.py`.

A matrix (`tack.Matrix`, `Matrix.field(n, m, ...)`) is a `VectorValue`
with a two-dimensional shape and row-major components, so everything
componentwise applies unchanged. A matrix field is a vector field of
`n * m` components with `_matrix_shape`; the runtime reports it to
lowering as the tuple `(n, m)` in the `vector_fields` dict, which also
keeps a 2x2 matrix field and a 4-vector field in different IR cache
entries. `_matrix_vars`/`_matrix_fields` hold shapes beside the component
counts; `_bind_vector` and `_variable_value` keep them in step. `@` is
`_matmul` (operands captured first, since every entry is used several
times); `transpose`/`trace`/`determinant`/`inverse` are in
`_emit_matrix_method`; `m[i, j]` goes through `_matrix_selector`. Limit
4x4 (`MAX_MATRIX_EXTENT`). See `test_matrix.py`.

### @tack.func inlining

Functions decorated with `@tack.func` are inlined at the AST level into kernels. Supports return values, multi-return (tuple), nested inlining, and vector propagation. Variables are renamed with unique suffixes to avoid collisions.

`tack.constant(value, dtype=None)` (`lang/constant.py`) returns an `int` or
`float` subclass. `visit_Name`/`visit_Attribute` resolve a name the
function does not bind through the same `CallBindings`; one bound to a
constant lowers to an `IRConstant`, under an `IRCast` when typed, so it is
baked into the cached IR like a literal and needs no variant-key entry.
Nothing else is captured: other names still raise `NameError`.
`math.pi`/`e`/`tau` lower the same way. A tuple (or tuple of tuples)
makes an `ArrayConstant`, lowered by `_named_constant` to a `VectorValue`
of per-component constants; `visit_Subscript` selects its components.
`tack.math` (`src/tack/math.py`) is plain `@tack.func` definitions with
GLSL semantics. See `test_constant.py`, `test_math_helpers.py`.

`tack.random` (`src/tack/random.py`) is PCG RXS-M-XS over an LCG in u32
with explicit state; draws return `(value, next_state)` and have `np_`
mirrors. Keep `uniform` as the top 24 bits times 2^-24 so it stays
bit-exact everywhere.

Device calls resolve by object identity from the defining callable's globals
and closure bindings (`call_bindings.py`), including aliases and module-qualified
calls. Arguments use caller bindings; nested bodies use callee bindings.
Validation and lowering share the resolver. Runtime callable parameters,
ordinary Python functions, object-property lookup, and recursion are rejected.
Numeric captures remain unsupported. Template rewrite returns a local method
map with marked synthetic calls and retains each method's original Python
callable; no global function registry or register/cleanup sequence remains.
Keep the IR cache construction lock. Cached IR retains its inlined bodies;
rebinding a callable requires recreating the kernel. AST-only transforms may
pass an explicit `bindings` dictionary. See `test_func_bindings.py`.

Original kernels and device functions are validated before lowering; unsupported syntax raises `UnsupportedSyntaxError` with the function and captured-source line/column. Device source is checked before return restructuring, including unreachable statements. `and`, `or`, and conditional expressions short-circuit, including inlined effects. Scalar operands/arguments evaluate left to right; augmented stores evaluate their index once, and sequential `range` bounds/steps are captured at entry. Comparisons and logical expressions yield i32 `0`/`1`. Outer parallel `break`, keyword arguments, and assertions are rejected; a kernel `return` is allowed only as the last statement, with a return annotation: `Kernel._split_result` strips it and builds a one-thread epilogue `Kernel` (statements outside the loop plus a store of the expression into a hidden one-element field, parameter `__tack_result__`) that `__call__` runs after the launch and reads with `result[0]`; barriers are statement-only, while an atomic may be an expression whose value is the element's old value (lowered to a temporary among the pre-statements, so it runs in source order; every generator's atomic yields the old value, the CAS loops through helpers). Reading a name the kernel never binds (typically a module-level Python value) raises `NameError` naming the kernel or inlined device function and the source position; binding a second name to a local or shared array (`view = tmp`) raises `UnsupportedSyntaxError`. `Kernel.__call__` lets `UnsupportedSyntaxError` through unwrapped. See `docs/reference/language-contract.md` and `test_source_validation.py` / `test_differential.py`.

The parallel loop must be a `for` directly in the kernel body, exactly once. Statements outside it run any number of times per launch (per GPU thread; per CPU chunk, probe and timing sample), so they may only bind locals, load fields and declare arrays. `ast_transform` rejects stores, atomics, barriers, block reductions and `print` there by scanning each statement's lowered IR, which catches effects from inlined functions and template methods at their own source line; `ir_verify` enforces the same invariant on later IR. `_localize_outer_scalars` gives each iteration its own copy of any outer local (or scalar parameter) the loop reassigns. See `test_outside_parallel_loop.py`.

### @tack.data_oriented templates

Classes decorated with `@tack.data_oriented` can be passed as template arguments. A kernel defined in the class body binds the instance it is called on (`Kernel.__get__` returns a bound method), so `model.step(dt)` passes `model` as the first argument; template detection is by argument type, so nothing else changes. `template_func_methods` unwraps `@staticmethod` over `@tack.func`. Methods and class constants are collected along the MRO when a kernel is lowered (`template_func_methods`, `classify_template_attrs`), so subclasses inherit and override them. An instance attribute holding a `@tack.func` (`template_func_attrs`) is called as that plain function, with no synthetic parameters, and its identity is part of the template cache key. Class-level scalar attributes become compile-time constants (part of cache key), instance scalar attributes become runtime kernel parameters (no recompilation on change), field attributes become kernel buffer parameters, and `@tack.func` methods are inlined with `self` resolved. Methods can call sibling methods on `self`.

### Integer floor division and remainder

Integer `//` rounds down and `%` is zero or follows the divisor's sign,
matching Python within the defined fixed-width domain. Both operands are
converted to the annotated promoted type before the operation; unsigned
types use unsigned division. LLVM applies an integer correction to signed
truncating division/remainder. The CUDA/HIP, Metal, and OpenCL generators
share typed helpers in `codegen/integer_division.py`, evaluating operands
once. Do not replace these with floating-point division or rely on C's
implicit signed/unsigned promotions. Evaluated divisors must be nonzero;
signed minimum with divisor -1 is excluded for both operators. See
`test_integer_division.py` and the language contract.

Integer `+`, `-`, `*`, unary negation, bitwise operations, and left shifts
wrap at their annotated width; signed right shifts are arithmetic and
unsigned right shifts logical. Counts must lie in `[0, width)`. Integer
casts/stores wrap modulo the destination width, with source signedness
controlling extension. Promotion preserves both complete operand ranges;
any signed/u64 pair requires an explicit cast. Comparisons and integer
`abs`/`min`/`max` retain exact integer semantics. `codegen/integer_ops.py`
shares unsigned-carrier helpers across GPU generators; Metal uses `as_type`
for signed bits and separate noinline i64/u64-add helpers for accumulator
updates inside runtime-bound loops to avoid an M1 Max compiler crash.
OpenCL emits signed `neg` below 32 bits and `abs` below 64 as noinline
helpers: IGC 2.7.11 widened the wrapped minimum as its magnitude.
LLVM tracks signedness on every
annotated integer expression, including loads of locals and scalar arguments.
Literals/scalars choose i32/i64/u64 by magnitude and reject unrepresentable
integers. Float-to-int inputs must be finite with a representable truncated
value. See `test_integer_semantics.py` and the language contract.

Integer `/` converts operands independently to f32 and returns f32, rather
than truncating; explicit floating casts request f64 on capable backends.
The output field does not widen the intermediate precision. The evaluated
integer divisor must be nonzero. With a floating operand, division uses
the promoted floating precision. Integer `**` and two-argument `pow`
preserve the base type, with an independently typed nonnegative exponent,
and compute exact power modulo the base width. Signed/u64 pairs are valid
for `/` and power. Negative literal integer exponents are rejected; dynamic
negative ones are outside the caller domain. `0 ** 0` is 1. Cast the base
to float for negative powers. GPU generators share exponentiation-by-squaring
helpers in `integer_ops.py`; LLVM emits an internal helper with wrapping
products and a logical count shift. With a floating operand, both power
arguments use the promoted floating precision. See `test_division_and_power.py`.

Floating `//` and `%` convert to f64 if either operand is f64, otherwise
f32, and return that floating type. Shared GPU helpers in
`codegen/float_division.py` and typed LLVM helpers use a truncating
remainder with divisor-sign correction and reconstruct/snap the quotient
instead of directly flooring rounded division. Zero remainder follows the
divisor's sign; zero quotient follows true division's sign. NaNs and
infinite dividends return NaNs; finite/infinite pairs follow Python-style
sign correction. Evaluated divisors must be nonzero, and denormal support
remains outside this portable increment. CUDA omits `--use_fast_math`
and explicitly selects non-flushing, precise division/sqrt and permitted
multiply/add contraction. Metal disables `fastMathEnabled` for every
kernel, including runtime reduction sources on both backends.
CPU floating negation uses LLVM `fneg`; floating `!=` and truth tests use
unordered inequality so NaN guards and signed-zero expressions agree.
See `test_float_division.py` and the language contract.

The floating-point baseline requires annotated precision, exceptional
classes/signs and expression grouping. Adjacent multiply/add contraction
is permitted; general reassociation is not. CPU emits no fast-math flags;
HIP/OpenCL retain default settings without unsafe options. Floating math
calls convert arguments to the result precision first; integer arguments
default to f32, irrespective of the destination field. CPU libm calls use
the matching f32/f64 symbols and return that precision before nested
arithmetic. Scalar floating min/max prefer a number to NaN and permit
either zero sign on ties. Denormals, NaN payloads/signaling, exception
flags/traps, cross-backend bitwise agreement and global math error bounds
are outside the portable baseline. CPU callers must retain the default
round-to-nearest, untrapped environment. `test_float_semantics.py` checks
classes/signs, grouping, permitted contraction and bounded-domain math.
Reduction order/accuracy guarantees remain the next stage-five task.

Float literals are weak, as Python scalars are under NumPy's NEP 50.
`ir_type_annotate.py` marks literal expressions (literals under unary
+/-, arithmetic, math builtins and conditional arms); one containing a
float literal is weak and f32 until it meets a non-weak floating operand,
an explicit floating cast, or a floating store/atomic/local target. It
then takes that type: each literal converts once from its Python value
and its operations run at that precision, so `x_f64 * 0.1` and
`tack.f64(0.1)` are exact. Literal-only subtrees are not folded. Alone,
or beside integers only, literals stay f32; `i * 0.1` is a non-weak f32.
A local assigned only literals is f32 (`tack.f64(0.1)` is the remedy).
LLVM emits a `double` constant and CUDA/HIP/OpenCL an unsuffixed `repr`
when the constant's dtype is f64; f32 code is unchanged. See
`test_float_literals.py` and the language contract.

### 64-bit loop indices on GPU

GPU backends use 64-bit integers for loop variables and index arithmetic (`long` on Metal, `long long` on CUDA/HIP) to support grids with more than 2^31 elements. CUDA/HIP widen `blockIdx.x` before multiplying by `blockDim.x`: the built-ins are 32-bit unsigned, and the product wrapped silently at 2^32 threads. OpenCL builds the index from `get_group_id(0)` the same way, since Intel's `get_global_id(0)` wraps at 2^32 despite its `size_t` type; native reductions use the same index. The CPU backend already used i64 via LLVM. Metal's `thread_position_in_grid` attribute is limited to `uint`, so max single dispatch is 2^32 threads. The `int()` cast in kernel code remains 32-bit (user semantics).

Each GPU backend's `execute` and native `reduce_field` call `check_launch_size` (`runtime/kernel_utils.py`), which raises `ValueError` for more iterations than one grid can index, instead of a driver error or a wrap. `_max_launch` is queried at init: CUDA max `gridDim.x` × 256; HIP the same, capped at 2^32 − 256 (AMD's dispatch packet counts work-items in 32 bits); Level Zero `maxGroupCountX` × workgroup size (the group count is a ctypes uint32 that would truncate). Metal uses `_MAX_LAUNCH = 2**32`, and reduces fields of 2^32+ elements through NumPy because its reduction kernel's count and thread position are 32-bit. CUDA/HIP reduction kernels take the count as a `long long` argument. See `test_launch_limits.py`.

### Algorithms (tack.algorithms)

`exclusive_scan` and `inclusive_scan` implement Blelloch-style parallel prefix sums. They use a `_read_last` kernel to return the total sum without copying the entire buffer to numpy. The Blelloch scan uses O(log n) kernel launches, so for small arrays (< ~1M elements) a numpy CPU roundtrip may be faster due to kernel launch overhead.

`argsort`/`sort_by_key` (`algorithms/sort.py`) are a stable LSD radix sort over 8-bit digits for i32/u32/i64/u64 keys: keys map to u64 with the sign bit flipped, one thread per 256-element chunk builds a private histogram, the exclusive scan assigns slots, and a second kernel scatters each chunk in order. The pass count follows the key spread (min subtracted on the fly), found with u32 atomics on the low/high words. `unique`/`reduce_by_key` flag run starts, scan them, and reduce one thread per run serially, so results are reproducible. Portable: no workgroup primitives. Empty fields (`shape=(0,)`) are valid outputs and allocate on every backend.

### ColorTable (tack.rendering)

`ColorTable` maps per-vertex scalar fields to RGB colors via a sampled lookup table. Presets: `viridis`, `cool_to_warm`, `inferno`, `plasma`, `grayscale`, `rainbow`. The `Actor` class accepts `scalars` (tack.field or numpy) + `color_table` (ColorTable) to enable scalar field coloring. During `Scene._prepare()`, scalars are mapped to per-vertex colors on GPU via linear interpolation into the lookup table. The pathtrace kernel's existing per-vertex color interpolation handles the rest — no kernel changes needed.

### Multiple lights (tack.rendering)

Multiple `PointLight` instances can be added to a `Scene`. Each light contributes independently with its own shadow ray. Light data is packed into a flat field (7 floats per light: x, y, z, intensity, r, g, b) and passed to the pathtrace kernel, which loops over all lights in the direct-illumination section. Light color (previously ignored) is now applied to each light's contribution. The `render()` function's `light_position` kwarg still works as a single-light override for backward compatibility.

### Volume rendering (tack.rendering)

`Volume` represents a uniform-grid scalar field for ray-casting volume rendering. `TransferFunction` maps scalars to RGBA (reuses ColorTable preset colors + user-defined opacity function). `render_volume(canvas, volume, camera)` ray marches through the volume with front-to-back compositing, trilinear interpolation via `tack.texture3d`, and opacity-corrected compositing.

### Unified render dispatcher (tack.rendering)

`render()` in `render.py` is the unified entry point. When a scene has both surfaces and volumes, the path tracer integrates volume ray marching directly — for each primary ray, after BVH traversal finds the closest surface hit, the volume is marched from the ray origin to the hit distance with front-to-back compositing. Volume opacity attenuates the throughput so surfaces are correctly visible through semi-transparent volumes. Volume-only scenes use the standalone ray caster. `render_volume()` remains available for direct single-volume rendering without a Scene.

### Annotations (tack.rendering)

`annotate(image, annotations)` draws overlays on `(H, W, 4) uint8` numpy arrays (from `Canvas.to_numpy()`). Three annotation types: `ColorBar` (gradient strip + tick labels from ColorTable/TransferFunction), `AxisIndicator` (projected XYZ axes from camera orientation), `TextOverlay` (arbitrary text). Lines and rectangles use pure numpy; text requires Pillow (soft dependency, gracefully skipped if absent). Camera stores `_right`, `_up`, `_forward` basis vectors for the axis indicator.

### Orthographic camera (tack.rendering)

`OrthographicCamera` generates parallel rays — all rays have the same direction, but origins vary per pixel. Uses the same scalar attribute pattern as `PerspectiveCamera` with added `odx_x/y/z` and `ody_x/y/z` origin-per-pixel deltas. `PerspectiveCamera` sets these to zero for backward compatibility. Works with both path tracer and volume renderer. The `view_height` parameter controls the world-space height of the view rectangle.

### Material system (tack.rendering)

`Material` class with three types: `MATTE` (Lambertian diffuse, default), `SPECULAR` (perfect mirror reflection), `TRANSPARENT` (glass with Snell's law refraction + Schlick Fresnel). Set per-actor via `Actor(..., material=Material(Material.SPECULAR))`. Per-triangle material IDs (`mat_ids` i32 field) index into a flat material table (`mat_table` f32 field, stride 4: type, ior, reserved, reserved). Direct lighting is only computed for matte surfaces. Specular bounces reflect perfectly; transparent surfaces refract or reflect based on Fresnel probability using Halton random numbers.

### Wireframe and point rendering (tack.rendering)

`Actor` accepts `render_mode="solid"` (default, path traced), `"wireframe"` (triangle edges), or `"points"` (vertex discs). Wireframe and point actors are dispatched to GPU rasterization kernels in `rasterize.py` instead of the path tracer. Uses an MVP projection matrix passed as a flat f32 field, Bresenham line rasterization for wireframe, disc rasterization for points, and `tack.atomic_min` depth testing. Supports perspective and orthographic cameras, per-actor colors, scalar coloring, and configurable point size.

In a scene that also has solid actors or volumes, `render()` path traces a solid-only sub-scene (cached on the parent so the BVH survives across frames) and then calls `render_raster(..., composite=True)`: wireframe and point actors are drawn into scratch buffers and laid over the existing image instead of clearing it. With solids, the path tracer's primary-ray distances are converted to the rasterizer's clip-space depth (rebuilding each pixel's jittered first-sample ray) so surfaces hide rasterized geometry behind them; `_SURFACE_DEPTH_BIAS` pushes surfaces back 0.5% so a wireframe lying on its own surface is drawn. A ray-cast volume has no depth, so rasterized actors are drawn over it. `canvas.depth` keeps the path tracer's ray distances in a mixed scene.

### CPU threading decision

The CPU backend fans a loop range out to its `ThreadPoolExecutor` only when the serial run would cost meaningfully more than the fan-out. Both sides are measured, not assumed:

- Each `CompiledKernel` carries `ns_per_elem`, a smoothed estimate of its serial cost, updated on every serial dispatch. From it the backend precomputes `parallel_min_elems`, so the dispatch hot path is one integer compare.
- The backend measures its own fan-out cost by dispatching *empty* ranges through the real path — no loop iterations, so the probe has no side effects. Under the default policy (v2; `TACK_CPU_POLICY=v1` opts out) that is a curve over idle gaps, interpolated on the actual idleness at dispatch, refined from real fan-outs, and re-measured directly at most every 100 ms. Only its hot end is measured at the first fan-out; each idle knot starts as a pessimistic prior (`_FAN_OUT_IDLE_PRIOR`, 4× hot, applied as a step) and is replaced, and moved, by the first fan-out measured at a pause the program actually took — measured at once rather than on the 100 ms clock. Nothing sleeps: calibrating by sleeping through each gap cost 180 ms inside whichever dispatch first looked worth threading (P8). v1 measures it once.
- The first time a kernel is seen at a range large enough to matter, a small slice is timed serially and the rest is decided on that sample. The slice comes from inside the range (the same golden-ratio walk as the rechecks), not its prefix.
- Once a kernel threads, occasional rechecks (dispatches 1, 2, 4, … 1024, then every 1024) re-time a serial slice. The slice walks the range in golden-ratio steps rather than always sampling the prefix, because an image kernel's first rows are background and a prefix sample read a volume render 10–70× too cheap. A range whose serial run costs no more than a fan-out is re-timed whole; sufficiently long worker spans establish a floor under the serial estimate. Worker spans and rates are summarized by `_worker_median`, the lower middle value for an even count: the upper one of two workers is the slower worker, so one descheduled thread set the floor on two-thread machines. If every worker of a complete dispatch later finishes below that duration, the old floor is retired and the next call rechecks serial cost. Cheap partial slices or a short median alone cannot retire it.
- A serial sample is *clean* only when a serial run laid the range out before it (`CompiledKernel.scattered` is false). One taken on untouched pages, or straight after a fan-out while other cores own the lines, reads a bandwidth-bound kernel 3–5× dear, and believing it was a fixed point: it kept the kernel fanning out, which kept the next sample dear (P7 in the audit). Under v2, a scattered sample may lower the estimate but not raise it, wherever a serial run of the range costs at most `_CONFIRM_BAND` margin-weighted fan-outs; and until a kernel has one clean sample, a fan-out inside that budget runs serially instead, at most twice per kernel. The first clean sample is believed outright when it is lower, since everything before it was an upper bound; smoothing it in left the estimate stuck near 3× in two harness runs of three. Outside the budget a rise is real evidence (an image kernel's estimate starts low) and is believed as before. A recheck slice reading 8× under the estimate is believed outright only if a second slice from later in the range agrees (`_second_opinion`); otherwise one background-row slice could send an image kernel's whole frame serial.
- The parallel rate has the same one-way door from the other side (P9): only fan-outs measure `r_p`, so a high one holds a kernel serial and nothing corrects it. When a range would fan out on the serial rate alone but `r_p` keeps it serial, v2 fans out anyway on the recheck back-off schedule (`_parallel_recheck_due`). The serial floor takes the lesser of the qualifying sample and the smoothed `r_p`: worker rates for identical fan-outs spread 8× run to run, and a floor from one high draw raised `r_s` to `r_p`, where the formula credits threading with no gain. `scattered` is cleared by `_run_whole_serial` around `_run_serial`, not inside it, because tests replace `_run_serial` with a four-argument lambda.

A fixed element count cannot work here: the crossover moves ~1000× with arithmetic intensity (~4M elements for `out[i] = x[i]*2+1`, ~130K for a `sqrt`/`sin` expression, ~4K for a 20-iteration inner loop). The previous constant of 1024 sat below all of them, making mid-size dispatches of cheap kernels 3–10× slower than running them serially.

`TACK_CPU_THREADS` overrides the thread count, as does `tack.init(arch="cpu", num_threads=N)`; `1` keeps everything on the calling thread. `TACK_CPU_POLICY` accepts `v1` or `v2` (unset or empty means v2); anything else raises. Boolean variables such as `TACK_NO_REINIT` go through `dispatch.env_flag`: `0`, `false`, `no`, `off` and empty mean off.

## Kernel language features

- **Loops**: `for i in range(n)`, `for i in range(start, end)`, `for i in range(start, end, step)`, `for i, j in tack.ndrange(w, h)` (each argument a size or a `(start, end)` pair, whose extent is clamped at zero; the host evaluates `max`/`min` in the launch size), `while`, `break`, `continue`
- **Masks**: comparisons of vectors give a vector of `0`/`1` per component; `any(mask)`, `all(mask)`, `tack.select(mask, a, b)`
- **Math**: `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `exp2`, `log`, `log2`, `log10`, `floor`, `ceil`, `abs`, `min`, `max` (two or more values), `pow`; all apply to each component of a vector
- **Types**: `int()`, `float()` casts, plus explicit `tack.i8()`, `tack.u8()`, `tack.i16()`, `tack.u16()`, `tack.i32()`, `tack.u32()`, `tack.i64()`, `tack.u64()`, `tack.f32()`, `tack.f64()`
- **Atomics**: `tack.atomic_add(field, idx, val)`, `tack.atomic_min(...)`, `tack.atomic_max(...)`
- **GPU primitives**: `tack.shared(dtype, size)`, `tack.shared_like(field, size)`, `tack.barrier()`, `tack.thread_id()`
- **Debug**: `print("label:", value)` — emits printf on CPU/CUDA/HIP, no-op on Metal
- **Fields**: `field[i]`, `field[i, j]`, `field[None]`, `field.shape[k]`, `len(field)` — usable anywhere in a kernel (loop bounds, conditions, arithmetic, indices), not just as the outer loop bound. The dimension index must be a literal. See "Field dimensions" below for what specializes.
- **Reductions**: `field.sum()`, `field.min()`, `field.max()`, `field.mean()` return Python floats. Eligible f32 fields reduce on GPU; CPU and other dtypes use shared NumPy semantics. f32/f64 sums retain their precision; signed/unsigned integer sums use wrapping i64/u64 accumulators before float conversion. Floating extrema propagate NaNs and use negative-zero min / positive-zero max ties. Empty sum is +0, empty mean NaN, empty extrema raise. Floating addition order may vary; see the absolute error budget in `docs/reference/language-contract.md`. Runtime kernels and GPU block extrema share `codegen/reductions.py`; block arguments/results are f32, requiring explicit casts for other inputs. CPU rejects cooperative kernels; GPU participation and atomic target/order/scope domains are defined in the language contract. See `test_reduction_semantics.py`.

## Platform-specific dependencies

- **macOS (Metal)**: `pyobjc-framework-Metal`
- **Linux/Windows (CUDA)**: `cuda-python>=13.2`, NVIDIA driver + CUDA toolkit
- **Linux (HIP/ROCm)**: `hip-python`, ROCm toolkit
- **Linux (Level Zero/Intel)**: `libze_loader.so`, `libocloc.so` (Intel compute runtime)
- **CPU-only**: `llvmlite`, `numpy`

## HIP backend notes

The HIP codegen (`hip_gen.py`) extends `CUDACodeGen` — HIP device code uses the same syntax as CUDA (`blockIdx`, `threadIdx`, `__global__`, `__shared__`, `__syncthreads`). The differences are `#include <hip/hip_runtime.h>` and the texture handle type, which HIP spells `hipTextureObject_t` (`_TEXTURE_OBJECT_TYPE`, overridden from CUDA's). The runtime (`hip_backend.py`) uses `hip-python` bindings for hipRTC compilation and dispatch.

**Textures need a device that has them.** CDNA parts — gfx940/941/942, i.e. MI300 — have no texture/image hardware, and hipRTC refuses `tex3D` outright ("The image/texture API not supported on the device"). The backend asks `hipDeviceAttributeImageSupport` at init and falls back to software trilinear sampling where the answer is no, the same way the Level Zero backend handles Xe-HPC. gfx90a (MI200) says yes but refuses a linearly filtered 3D texture (`hipErrorNotSupported` from `hipCreateTextureObject`; point sampling works), so init also makes one 2×2×2 texture and falls back when that is refused. The decision is `texture_in_hardware()`, which `_store_texture_shapes` consults before the variant key is built, because it changes the generated code, and which `Texture3D` consults to choose its storage; `tack.inspect` calls the same hook.

`hip-python` is on PyPI now (it used to be Test-PyPI only), so the `[hip]` extra declares it:

```bash
uv sync --extra hip     # or: pip install tack-core[hip]
```

manylinux x86_64 wheels only, so the dependency carries a platform marker and is skipped elsewhere. Its version tracks the ROCm release it binds to — 7.1.x against ROCm 7.1, 7.2.x against 7.2 — so the extra sets a lower bound rather than a pin, and a mismatched ROCm may want an explicit `hip-python~=7.0.0`.

That mismatch is not always fatal, which is worth knowing before pinning on principle: the lower bound resolves to the newest wheel, so **ROCm 7.0.2 got hip-python 7.2.2 — two minor versions ahead — and the whole suite passed on it**, MI300X, 2026-08-11. Treat the pin as the fix for an actual failure rather than a precaution.

Then: `tack.init(arch=tack.hip)`.

**Known issue**: `hiprtcDestroyProgram` segfaults in hip-python. The backend skips the call (minor leak, mitigated by kernel caching). Recorded against **7.1**; **checked on 2026-08-11 against hip-python 7.2.2 / ROCm 7.0.2 on an MI300X and it still segfaults** — SIGSEGV on the first call, after a successful compile. The documented calling convention is the one that crashes: `hiprtcDestroyProgram(prog)` takes the program directly, and passing a pointer to it is rejected by the binding as a type error. The workaround stays.

**hip-python 7.x struct wrappers are incomplete.** `make_hipExtent`/`make_hipPos`/`make_hipPitchedPtr` are gone (construct `hipExtent(width=...)` etc.), `hipCreateChannelDesc` returns `(err, desc)`, nested struct members are read-only attributes that are views into the parent (set their fields in place), and array handles (`res.array.array`, `hipMemcpy3DParms.dstArray`) and fixed-size arrays (`hipTextureDesc.addressMode`) are not exposed at all. `HIPTextureImage` writes those through ctypes, locating them from neighbouring views. MI300X has no texture hardware, so this path went untested until an MI210 (gfx90a), 2026-10-07; there it can only be run with point filtering.

## Level Zero backend notes

The Level Zero codegen (`opencl_gen.py`) extends `CUDACodeGen` — OpenCL C kernel syntax mirrors CUDA with different qualifiers (`__kernel`/`__global`, `get_group_id(0)*get_local_size(0)+get_local_id(0)`/`blockIdx*blockDim+threadIdx`, `__local`/`__shared__`, `barrier()`/`__syncthreads()`). Math functions are overloaded (no `f` suffix). The runtime (`level_zero_backend.py`) uses ctypes bindings to `libze_loader.so` and `libocloc.so`.

Compilation pipeline: OpenCL C source → `libocloc.so` (in-process, via `oclocInvoke`) → SPIR-V → `zeModuleCreate` → `zeKernelCreate`. The ocloc library is part of the Intel compute runtime (`intel-opencl-icd` package).

Requires: `libze_loader.so` (Level Zero runtime), `libocloc.so` (Intel offline compiler). No Python packages needed.

Then: `tack.init(arch=tack.level_zero)`.

**Sharing memory needs a shared context.** A Level Zero USM pointer is only
meaningful in the context that allocated it, and DLPack carries none.
`tack.init(arch=tack.level_zero, external_context={driver, device, context})`
adopts another library's handles instead of creating a context (all three
are required; `Backend.init_options` declares accepted keywords and
`tack.init` rejects others). `tack.interop.vtk.init_level_zero()` takes them
from VTK's Viskores/Kokkos SYCL queue, and the VTK interop refuses Level Zero
fields from any other context. Intel's driver resolves pointers from any
context, so this check is the only one that catches the mistake. The
backend reports `device_memory_spaces = {"level_zero"}` via
`zeMemGetAllocProperties`. See `test_level_zero_context.py`.

## Do not mention Claude in git commits
