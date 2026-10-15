# Specialization and Caching

A Tack kernel is not compiled once. It is compiled once per **variant**: a
specialization for one combination of the facts that the compiler bakes
into generated code. This page defines what a variant is, lists exactly
what goes into its key and why, describes the cache that holds variants
and how long they live, and gives the user-facing consequences.

The governing requirement is in the contract's
[Specialization and compilation identity](../reference/language-contract.md#specialization-and-compilation-identity)
section: *reusing compiled code must have the same defined behavior as
compiling the current call independently.* Every fact baked into code or
its calling convention must therefore be part of the key. Conversely,
runtime scalar values and the outer launch length alone should not force a
recompile.

## What a variant is

`KernelVariant` in `packages/tack-core/src/tack/runtime/kernel_utils.py`
holds one compiled specialization:

| Attribute | Contents |
|-----------|----------|
| `ir` | the post-pass IR (before GPU scalar packing), kept so the launch range can be evaluated on every dispatch without rerunning passes |
| `payload` | whatever the backend's `build` callback returned: the JIT'd function on the CPU; the compiled kernel, `pack_info` and pack buffers on GPU |
| `requires_full_workgroups` | whether every launch must be a multiple of 256 lanes |
| `atomic_targets` | atomic parameter indices and their alignments, rechecked against each call's storage |
| `written_arrays` | `(index, name)` of each array argument the kernel may store to, so a dispatch can refuse read-only storage without walking IR |
| `has_textures` | whether any argument is a `Texture3D`, so a call without one skips rebinding textures |
| `dispatch_lock` | serializes dispatches of this variant on CUDA and HIP, whose payload holds the scalar pack buffers |

Variants are produced and looked up only by `resolve_variant()`, the
single entry point all five backends share. The
[Compilation Pipeline](compilation-pipeline.md) page describes the steps
that build one.

## Two caches, two keys

Specialization happens at two levels, because some facts change what the
*frontend* emits and others only change what the *passes* resolve. Both
levels are also kept on disk, across processes ([below](#the-disk-cache)),
and in front of them a repeated call skips key derivation altogether
([the dispatch fast path](#the-dispatch-fast-path)).

```mermaid
flowchart LR
    K["Kernel object"] --> IRC["IR template cache<br/>Kernel._ir / Kernel._ir_cache<br/>key: vector widths, texture extents,<br/>template structure"]
    K --> VC["Variant cache, one per backend<br/>backend._cache[kernel][variant_key]<br/>key: param types and categories, vector widths,<br/>template key, shape signature, disjoint bit"]
    IRC -- "template (immutable)" --> VC
```

| Cache | Owner | Key | Built by |
|-------|-------|-----|----------|
| IR template | `Kernel` (`lang/kernel.py`) | `Kernel._make_cache_key(vector_arrays, template_args, texture_arrays, debug)`, or none at all for the plain `Kernel._ir` (no vectors, textures or templates, debug mode off) | template rewrite, source validation, AST transform, `verify_ir('lowered')` |
| Compiled variant | each backend instance (`backend._cache`) | `kernel_variant_key(...)` | clone, IR passes, verification, codegen, native compile |

Every input to the IR template key also reaches the variant key, either
directly (vector widths, template structure) or indirectly (texture
extents arrive as `_texture_shape` on the parameter and, through each
`IRTextureSample`, in the shape signature; debug mode's template has two
more parameters, the hidden errors and extents arrays, so its parameter
signature differs). Two calls that share a variant therefore always share
an IR template.

**Debug mode.** `tack.init(debug=True)` (or `TACK_DEBUG=1`) lowers every
kernel with bounds checks (`lang/ir_bounds.py`): `("debug",)` leads the IR
template key, and the two hidden array parameters follow the kernel's own.
The extents are read from an array at run time rather than resolved, so a
debug kernel compiles no variant per array length. With debug mode off,
the IR and generated code are those of a build without the feature.

## The variant key, element by element

`kernel_variant_key(ir_func, kernel, vector_arrays, template_args,
shape_sig, disjoint)` returns a five-tuple:

```python
(param_sig, vec_sig, tmpl_key, shape_sig, disjoint)
```

The kernel itself is not in the tuple: each kernel has its own slot in the
cache (see [below](#cache-structure-and-lifetime)).

### `param_sig`: types and categories

One entry per IR parameter, computed in a single pass because it runs on
every dispatch:

```python
(p.type_annotation, p._is_array, p._is_texture, p._texture_shape)
```

These come from `infer_param_types` (`lang/type_inference.py`) and the
backend's `store_texture_shapes`, run on a private `_KeyProbe` rather than
on the shared template (see [Concurrency](#concurrency)).

| Component | Values | What it bakes in |
|-----------|--------|------------------|
| `type_annotation` | an array's dtype; for a Python `int`/`np.integer`, `i32`, `i64` or `u64` chosen by `integer_type_for_value` from the *value*; for a `float`/`np.floating`, `f32`, or `f64` when any array argument is `f64` | every expression's type, the arithmetic emitted, the C/LLVM parameter type, the pack buffer an argument lands in |
| `_is_array` | array (or texture) versus scalar | the calling convention: pointer versus value on the CPU; buffer binding versus packed scalar on GPU |
| `_is_texture` | hardware texture versus not | sampler code versus loads; HIP and Level Zero set it to `False` to choose software sampling when the device has no image support or the extent exceeds its 3-D image limit |
| `_texture_shape` | `Texture3D.shape_3d` | the extents used in sampling code |

### `vec_sig`: vector widths

`tuple(sorted(vector_arrays.items()))`: the component count of each
`tack.Array` with a vector element dtype argument. Width determines how many component loads
the frontend emits and the stride between vectors, so two widths need two
lowerings and two compiled kernels. Leaving width out of the key once let a
width-2 variant serve a width-3 call and return wrong norms; that is
contract regression **LC3**.

### `tmpl_key`: template structure and constants

For calls with `@tack.device_class` arguments, `Kernel._make_cache_key`
contributes one part per template parameter:

```python
("tmpl_<index>",
 _ClassToken,                                   # identity of the class
 ((name, type(v), bits_or_value), ...),         # class-level constants
 ((name, dtype, shape, vector_n), ...),         # array attributes
 (runtime_attr_name, ...))                      # instance scalar names
```

(plus a leading `("vec", …)` part when vector arrays are present).

- **Class identity.** Two classes with the same name can have different
  methods, so the actual class is keyed, through a `_ClassToken` (below)
  rather than the class object itself.
- **Constants.** Class-level numeric attributes are inlined as literals,
  so their values belong in the key. Each is stored with its Python type
  (`1` and `1.0` lower differently). Floats are stored as their
  `struct.pack('!d', v)` bit pattern, which distinguishes `0.0` from
  `-0.0` and gives NaN a key equal to itself despite `NaN != NaN`. An
  instance attribute that shadows a class attribute keeps the constant
  role, with the instance's value.
- **Array attributes.** Their dtype, full shape and vector width.
- **Runtime scalar names.** Instance scalars are runtime parameters, so
  their *values* are not keyed. Their *names* are, because adding or
  removing one changes the parameter layout.

!!! note
    Template array attributes are keyed by their full shape, which is
    stricter than plain array arguments (those specialize only on the
    dimensions in the shape signature). Changing the length of a template's
    array therefore lowers a new IR template and compiles a new variant even
    when no dimension is baked in. This over-specializes, which is safe; it
    is a cost, not a correctness issue.

### `shape_sig`: dimensions baked into code

`resolve_ir` substitutes array dimensions as literals: `a[i, j]` becomes
`i * 8 + j` when `a.shape[1] == 8`. That literal is part of the compiled
code's identity. `shape_signature(template, name_to_array)` reports
exactly the dimensions a kernel bakes in, as a tuple of concrete sizes.

It is built from `ir_shape_deps(ir_func)`, which scans the template once
and memoizes the result on it as `_shape_deps` (it depends on the source,
not on the arguments):

- `_dependent_nodes` yields every node whose value reaches generated code,
  **skipping the parallel loop's `end`**.
- Each `IRDimSize(array, d)` contributes `(array, d)`. The array name is
  mapped through `_static_array_aliases`, which follows the
  `__<func>_<param>_<N>__ = param` copies that `@tack.func` inlining creates.
- Each `IRTextureSample` contributes all three dimensions of its texture.

`shape_signature` then looks up each `(name, d)` in this call's arguments,
using `shape_3d` for textures, `_logical_shape` for vector arrays and
`shape` otherwise. A kernel with no baked-in dimensions gets the empty
tuple, so the common 1-D kernel pays nothing for shape keying.

**Why the parallel loop's end is excluded.** It never reaches generated
code. Code generators read it from the `__loop_end__` parameter, and
`_get_loop_range` evaluates it against the arguments on every dispatch.
`resolve_ir` leaves it unresolved for the same reason. Without the
exclusion, `for i in range(x.shape[0])`, the most common line in any
kernel, would compile a new variant for every array length.

| Kernel line | Baked-in dimensions |
|-------------|---------------------|
| `for i in range(x.shape[0]): out[i] = x[i] * 2` | none |
| `for i in range(1, x.shape[0] - 1): …x[i-1] + x[i+1]…` | none (the start moves into the body as a constant) |
| `out[i] = x[x.shape[0] - 1 - i]` | `x.shape[0]` |
| `for i, j in tack.ndrange(a.shape[0], a.shape[1]): out[i, j] = a[i, j]` | `a.shape[1]`, `out.shape[1]` (the row count only sets the launch) |
| `v = tex.sample(u, v, w)` | all three extents of `tex` |

### `disjoint`: the CPU no-overlap bit

Arrays may share storage: the same array passed twice, two views of one
allocation, or imported pointers. Generated code therefore never promises
`noalias`/`restrict` unconditionally (contract regression **LC2**; see
[Memory and Aliasing](memory-and-aliasing.md)). Without that promise,
though, LLVM must reload after every store and cannot vectorize a loop
that accumulates through an array.

The CPU backend recovers that case with a second variant. It calls
`resolve_variant(..., specialize_disjoint=True)` (the module constant
`_SPECIALIZE_DISJOINT` in `runtime/cpu.py`), and on every dispatch
`arrays_disjoint(template, effective_args)` decides whether this call
qualifies:

1. `written_array_params(template)` finds the array parameters the kernel
   may store to, directly, atomically, or through `@tack.func` pointer
   copies, by a fixed point over name copies. If a store goes through a
   name it cannot trace back to a parameter or a local allocation, it
   returns `None` and every array counts as written. The result, and a
   per-parameter flag tuple, are memoized on the template.
2. Each array argument's byte range comes from its buffer's `span`
   (`NumpyBuffer.span`, cached per array). A buffer without a `span` makes
   the answer `False`.
3. The call qualifies when no written range overlaps any other array's
   range. Read-only arrays may overlap each other freely: `dot(x, x)` still
   qualifies.

The answer is the last key element and is stored on the variant's IR as
`disjoint_arrays`. `LLVMCodeGen` then adds `noalias` to array parameters
only in that variant. Both variants must agree on every race-free program,
so overlapping and non-overlapping calls of one kernel each get correct
code; a kernel that is sometimes called with aliased arrays simply has two
variants. GPU backends never pass `specialize_disjoint`, so for them the
element is always `False`.

The check costs a little on every CPU dispatch. Whether that pays for
itself depends on the host and the kernel; the tradeoff is discussed in
[Memory and Aliasing](memory-and-aliasing.md).

## What is deliberately not in the key

| Not keyed | Why that is safe |
|-----------|------------------|
| Scalar argument values (within one inferred type) | scalars are runtime parameters; on GPU they are rewritten into the pack buffers on each dispatch |
| The outer launch length | read from `__loop_end__`, evaluated per dispatch |
| Template instance scalar values | runtime parameters; only their names are keyed |
| Array contents and addresses | never baked in; the only address-derived fact is the CPU `disjoint` bit |
| The kernel's name | each `Kernel` object has its own cache slot |
| The backend | each backend instance has its own cache |
| Debug mode, as a flag | not needed: a debug template has two more parameters, so its `param_sig` differs; the IR template key and the disk keys name it |
| A floating-point math mode | there is one fixed policy (see [Numerical Semantics](numerical-semantics.md)); a selectable mode would have to join the key |
| Device-function bindings | consulted only when an IR template is built; rebinding afterwards is unsupported, so recreate the kernel. The disk cache does record them (below) |

## Cache structure and lifetime

Each backend creates its cache in `__init__` with `new_kernel_cache()`:

```python
backend._cache: WeakKeyDictionary[Kernel, dict[variant_key, KernelVariant]]
```

`kernel_cache_slot(cache, kernel)` returns, or creates, the kernel's inner
dictionary.

**Keyed on the `Kernel` object, held weakly.** An earlier cache was keyed
on `id(kernel)`. A garbage-collected kernel frees its address, and a new
kernel allocated there with the same name and argument types silently ran
the old kernel's code. Keying on the object removes that hazard, and
holding it weakly releases compiled code and its device modules when the
kernel goes away. Most kernels are module-level objects that live as long
as the process, and so do their variants.

**One cache per backend instance.** `tack.init()` constructs a new backend
with a new, empty cache. Variants compiled before the switch stay with the
old backend object, and arrays allocated by the old backend are rejected
with a `RuntimeError` when dispatched on the new one.

**Every cache is registered.** `new_kernel_cache` appends a weak reference
to the module-level `_kernel_caches` list. `drop_variants(kernel, is_stale)`
walks every live cache, not only the active backend's, which matters for
template-class retirement.

### Template-class lifetime: `_ClassToken`

Keying a variant on a template class object would keep the class alive as
long as the kernel. A class defined inside a function and created on every
call would then leave one IR template and one variant behind per call,
forever. Instead, `lang/kernel.py` keys on a `_ClassToken`:

- `_class_token(cls)` returns the class's token from `_class_tokens`, a
  `WeakKeyDictionary` from class to token, creating it on first use and
  registering `weakref.finalize(cls, _retire_class, token)`.
- Each token records the kernels specialized on it (`token.kernels`, a
  `WeakSet`).
- When the class is collected, `_retire_class` removes every IR template
  in those kernels' `_ir_cache` whose key names the token, and calls
  `drop_variants` to remove every variant whose template key (the third
  key element) names it, in every live backend cache.

The token provides identity without a strong reference: distinct classes
never share a variant, and a collected class takes its specializations with
it.

## Concurrency

The warm path takes no lock: the IR template lookup and the variant lookup
are dictionary reads.

- **IR templates** are built under `_transform_lock` with a re-check, so a
  template is lowered once even when several threads miss together.
- **Key derivation** writes argument types onto a per-dispatch `_KeyProbe`
  rather than the shared template, so concurrent dispatches with different
  dtypes cannot read each other's annotations.
- **Variant construction** is not locked. Two threads that miss on the
  same key both build. `kernel_cache_slot` uses `setdefault`, so neither
  replaces the other's slot dictionary, and the second `slot[key] = variant`
  overwrites the first. Each thread returns the variant it built, and both
  are correct, so the only cost is a duplicate compile.
- **Launching a variant** is serialized where a variant owns launch
  state. On CUDA and HIP that is the pack buffers its scalars are written
  into, and they hold the variant's `dispatch_lock` (a slot of
  `KernelVariant`) from the pack update until the launch is queued, so
  dispatches of *different* variants still overlap. Level Zero holds one
  backend-wide `_launch_lock`, because its command list and kernel
  argument state are shared by the whole backend. Metal's launches own
  their argument buffer and scalars and encode under the backend's batch
  lock. The CPU binds arguments per call and takes no lock, though its
  JIT'd pool runs one fan-out at a time. See
  [Backend Implementations](backend-implementations.md#scalar-arguments-on-the-gpu).
- **The fast path's entries** are plain dictionary writes; the slot is
  cleared when it reaches its cap of 256 signatures per kernel and
  backend.

## Hit versus miss

| Step | Hit | Miss |
|------|:---:|:----:|
| classify arguments (templates, vectors, textures) | ✓ | ✓ |
| IR template lookup (a template key is built for template/vector/texture calls) | ✓ | ✓ |
| `check_workgroup_support` (memoized on the template on the CPU, immediate on GPU) | ✓ | ✓ |
| type inference on the `_KeyProbe`, `shape_signature`, `arrays_disjoint` (CPU) | ✓ | ✓ |
| build the key, look up the variant | ✓ | ✓ |
| look up the variant in the disk cache, and load it on a disk hit | | ✓ |
| `clone_ir`, resolve, infer, check, localize, atomic/workgroup analysis, optimize (on a disk miss) | | ✓ |
| `verify_ir` at each boundary | | ✓ |
| GPU clone and scalar packing, `annotate_types`, codegen, native compile | | ✓ |
| `check_workgroup_launch` (collective kernels), `check_atomic_alignment`, `check_writable_arrays` | ✓ | ✓ |
| evaluate the launch range, `check_launch_size` (GPU), update pack buffers and queue the launch under the variant's lock (CUDA, HIP) | ✓ | ✓ |

A hit clones nothing, runs no passes and performs no verification. The
memoized template facts (`_shape_deps`, `_written_flags`,
`_workgroup_features`) and the variant's recorded `atomic_targets` and
`written_arrays` keep the remaining per-dispatch work proportional to
the parameter count rather than to the kernel's size.

### The dispatch fast path

Even a hit walked the arguments half a dozen times: template detection and
expansion, vector and texture detection, the IR lookup, type inference on
the probe, the shape signature, the key. Each took a few microseconds, and
together they were the largest part of a dispatch's host time (90 of about
270 µs per launch of a contour's kernels on a 2 GHz Xeon). So
`resolve_variant` first tries `_fast_resolve`: a call whose *signature*
was seen before reuses what that call resolved.

The signature (`_call_signature`) is read straight off the arguments: a
array's dtype, shape and element layout (memoized on the array as
`_sig`); a texture's extents; a scalar's kind and, for integers, the type
its value selects; a template object's class token, its class constants'
current values, and what the key and the expansion read of each
attribute, nested templates by path. Equal signatures imply an equal
variant key, so an entry, kept per backend and kernel, holds the
expansion plan for template arguments, the variants it resolved by
disjointness, and the array slots for the overlap check. A hit expands the
templates through the plan, decides disjointness, runs the per-call checks
(`_check_call`: full workgroups, atomic alignment, read-only arrays) and
returns. Entries hold no class, and `drop_variants` forgets the kernel's
entries on every backend along with its variants, so a per-call template
class is still collected and a dropped variant is resolved afresh. A call with keyword arguments,
an argument the signature cannot describe, or debug mode (its variants
take hidden arguments, and each launch is noted for the check) takes the
slow path. Together with the CPU passing array addresses rather than
typed pointer objects, a warm CPU dispatch went from 70 to 40 µs.

## The correctness argument

The design rests on one rule: **anything a pass or code generator bakes
into the compiled code must be named by the key.** Leaving something out
causes silent wrong numbers, while keying too much only costs compile time.
The table pairs each baked-in fact with the key element that names it:

| Baked into the variant | By | Named by |
|------------------------|----|----------|
| parameter types, expression types, pack layout | inference, annotation, packing | `param_sig[*].type_annotation` |
| pointer versus value calling convention | codegen, packing | `param_sig[*]._is_array` |
| sampler versus software sampling, texture extents | codegen, `store_texture_shapes` | `param_sig[*]._is_texture`, `_texture_shape`, `shape_sig` |
| vector scalarization | AST transform | `vec_sig` (and the IR template key) |
| inlined template methods, class constants, template parameter layout | template rewrite | `tmpl_key` |
| dimension literals in index arithmetic and loops | `resolve_ir` | `shape_sig` |
| `noalias` on array pointers | `LLVMCodeGen` | `disjoint` |
| everything derived from the source | all stages | the per-kernel slot |

This rule has been learned the hard way. The cache was once keyed on
`id(kernel)` (reused addresses ran another kernel's code); then on types
alone (a new row stride reused code with the old stride baked in); then
without vector widths (**LC3**). Each was a silent wrong-numbers bug. The
regression tests for this area are in
`packages/tack-core/tests/test_variant_cache.py`, `test_kernel_cache.py`,
`test_array_shape.py` and `test_disjoint_arrays.py`.

## Worked examples

The variant counts below were observed on the CPU backend at this release
by inspecting `backend._cache[kernel]` after each call, starting from an
empty cache. All arrays are `f32` unless noted. The disjoint bit is `True`
for every call except the aliased one.

```python
@tack.kernel
def scale(x, out, a):
    for i in range(x.shape[0]):
        out[i] = x[i] * a
```

| Call | Variants | Why |
|------|:-------:|-----|
| `scale(x8, o8, 2.0)`, then lengths 16 and 32 | 1 | the only dimension is the launch bound |
| `scale(x8, o8, 3.5)` | 1 | a new scalar value is a runtime argument |
| `scale(x8, o8, 3)` | 2 | `int` infers `i32`, not `f32` |
| `scale(x8, o8, 2**40)` | 3 | beyond the `i32` range, so `i64` |
| `scale(x8_f64, o8_f64, 2.0)` | 4 | new array dtype, and the float scalar becomes `f64` |
| `scale(x, x, 2.0)` | 5 | the written array overlaps another array, so the disjoint bit is `False` |

| Kernel | Calls | Variants |
|--------|-------|:-------:|
| `out[i] = x[x.shape[0] - 1 - i]` | lengths 8, 16, 32 | 3 |
| the same with `n` passed as a scalar: `out[i] = x[n - 1 - i]` | lengths 8, 16, 32 | 1 |
| `tack.ndrange(a.shape[0], a.shape[1])` over 2-D arrays | shapes `(4, 8)`, `(16, 8)`, `(4, 9)` | 2 |
| `range(1, x.shape[0] - 1)` stencil | lengths 8, 16, 32 | 1 |
| `out[i] = v[i].dot(v[i])` with `tack.Array` with a vector element dtype | widths 2, then 3 | 2 (and 2 IR templates) |

For a `@tack.device_class` class with a class constant `factor`, an
instance scalar `offset` and an instance array `data`:

| Call | Variants | IR templates | Why |
|------|:-------:|:------------:|-----|
| first call | 1 | 1 | |
| new `offset` value | 1 | 1 | instance scalars are runtime parameters |
| `data` of a different length | 2 | 2 | template array shapes are keyed |
| `s.factor = 3.0` on the instance | 3 | 3 | it shadows a class constant, so it stays a constant |
| an equivalent class defined inside a function | 4 | 4 | different class identity |
| ... after that class is garbage-collected | 3 | 3 | `_retire_class` dropped its specializations |

## The disk cache

`runtime/disk_cache.py` keeps both kinds of cached code across processes.
The in-memory caches above stay in front of it: the disk is consulted only
on an in-memory miss, so a warm dispatch never touches it. User-facing
settings are on [The Kernel Cache](../users-guide/17-kernel-cache.md).

**Why the front end is cached too.** A cold contour of a 21³ grid on an M1
Max spent about 1.15 s in 18 new kernels: 0.46 s lowering (template
rewriting included), about 0.25 s in the IR passes and the verifier, 0.19 s
generating MSL and 0.21 s in Metal's compiler. A cache of binaries alone
would save the smallest part.

**Two levels.**

| Entry | Key | Holds |
|---|---|---|
| `.ir` | the kernel's module, qualified name and source; its result type; its closure cells' fingerprints; vector and texture arrays; each template's content; whether debug mode lowered it; Tack's fingerprint | the pickled `IRModule`, and the names lowering resolved |
| `.variant` | the `.ir` entry's key and dependency digest; parameter types and categories, texture shapes, vector widths, `shape_sig`, `disjoint`; the backend and `Backend.cache_target()` | the post-pass IR, `requires_full_workgroups`, the atomic targets, and the backend's artifact |

Debug mode is in the `.ir` key, and the `.variant` key follows from the
`.ir` entry, so a debug kernel and a plain one never share an entry in
either direction: without it, a debug process read the plain kernel from
disk and its launch, missing the hidden arguments, failed. A debug kernel
read from disk still names the access it caught, because its sites travel
in the IR.

A template's content replaces the identities of the in-memory key: each
class by its method resolution order (module and qualified name), its
`@tack.func` methods by source and `__tack_iterate__`; device functions
held as attributes by source; constants by value (floats by bit pattern).

**Names lowering resolved.** The in-memory IR cache relies on the
documented rule that bindings do not change during a process. Across
processes they do: a device function is edited, a constant changes. Every
`CallBindings.resolve` during a lowering is recorded (`_Recorder`) as the
owner function, the dotted name, and a fingerprint of what it resolved to:
a `Func` by its source, a `tack.constant` by value, a module by name, and
"unbound" for a bare intrinsic. On a read, each name is resolved again in
the current owner (`_lookup`, which follows `CallBindings`' rules) and the
entry is used only if every fingerprint matches. Owners are the kernel,
its templates' methods and function attributes, and each device function
a recorded name resolved to, in recording order. A name bound to an object
with no stable fingerprint, or a lowering with an explicit `bindings`
namespace, is not stored. The closure cells are in the key itself, so
kernels a factory makes over different device functions keep separate
entries instead of replacing each other on every run.

**Tack's fingerprint** covers everything else lowering and code generation
depend on: the `RECORD` of each installed Tack package (one read each), or
every source file of a source tree on the path (editable installs: 167
files, 16 ms, once per process), plus the format version and Python's minor
version. It names the directory the entries live in, so another Tack never
reads them.

**Artifacts.** `Backend.cache_target()`, `variant_artifact(payload)` and
`load_variant(ir_func, artifact, effective_args)`; a backend whose
`cache_target()` is None stores only the `.ir` level; every backend
stores variants now. `effective_args` are the call's arguments: a GPU variant's
scalar pack buffers are device memory of the process, so a load makes them
anew from the call's scalars, as `_build_variant` does.

| Backend | Artifact | Loading it |
|---|---|---|
| CPU | the object code MCJIT emitted (`set_object_cache`) | `add_object_file` into a new engine: about 0.4 ms, no LLVM optimization or code generation |
| Metal | MSL source, entry name, parameter layout, scalar pack layout | `newLibraryWithSource`, which macOS's own shader cache makes fast for source it has compiled before |
| CUDA | PTX from NVRTC, entry name, parameter layout, scalar pack layout | `cuModuleLoadData`: the driver compiles the PTX for the device, so the target names the device and compute capability, the driver version, NVRTC's version and its options |
| HIP | the code object from hipRTC, entry name, parameter layout, scalar pack layout | `hipModuleLoadData`: machine code for one architecture, so the target names the device and `gcnArchName` with its feature flags (which also separates devices with and without texture hardware), the ROCm runtime and driver, hipRTC, hip-python and the hipRTC options |
| Level Zero | the native binary IGC made (`zeModuleGetNativeBinary`), the SPIR-V from ocloc, entry name, parameter layout, workgroup size, scalar pack layout | `zeModuleCreate` in the backend's context (its own or the adopted one) as `ZE_MODULE_FORMAT_NATIVE`: about 0.2 ms a kernel. A binary the driver refuses loads from the SPIR-V, which IGC finalizes, about 50 ms a kernel without the driver's own cache. The target names the device and its IP version (stepping), the driver version, ocloc's version, the ocloc and module options, and f64 support |

**File-system rules.** Entries are written to a temporary file in the same
directory and renamed into place, atomic on POSIX file systems including
Lustre and GPFS; concurrent writers of one entry write the same bytes. No
locks. Each entry starts with a magic string and a SHA-256 of its payload,
and holds its own key; anything else is ignored and counted as `bad`. The
directory and the per-version directory must belong to the user and be
writable by nobody else, since loading an entry unpickles it; the cache
creates both with mode 0700, since under a umask of 002 a directory it made
itself was group-writable and then refused. The entries of every Tack
version share one size limit (`TACK_CACHE_SIZE`, 1 GiB by default): past
it, a writing process deletes the least recently used entries until 90%
of the limit is left, scanning the directory at most once an hour across
processes (a `.pruned` mark), never per kernel; a hit refreshes its
entry's time at most once an hour, and bundles are neither counted nor
deleted. Under MPI
(a launcher's rank variable with a world size above one) the default mode
is `bundle`: one file per Tack version, read once per process (or by rank 0
and broadcast, `load_bundle(comm=...)`), and no per-kernel file is looked up
or written. In `write` mode under MPI only rank 0 stores. Nothing compiles
collectively: ranks reach different kernels, so a collective there could
deadlock.

`conftest.py` sets `TACK_CACHE=off` unless the environment chooses, so the
suite compiles from scratch; `test_disk_cache.py` runs its cases in fresh
processes against a temporary directory. `TACK_CACHE=write pytest`, run
twice, serves most of the second run from disk.

## Costs and advice

Variants multiply: every distinct combination of dtypes, scalar types,
vector widths, baked-in dimensions, template constants and (on the CPU)
aliasing produces its own compile. Each cold compile pays for cloning,
the pass pipeline, verification at every boundary, code generation and
the native compiler. The first call of a large kernel is noticeably slower
than a warm one, and a workload that keeps producing new variants never
reaches the warm path.

To keep the number of variants down:

- **Pass lengths as scalars** when a dimension would otherwise appear
  inside the body. `def reverse(x, out, n)` compiles once; reading
  `x.shape[0]` in the index compiles once per length. Dimensions that
  appear only in the outermost `range` never specialize.
- **Keep scalar types stable.** Integer scalars are typed by value: a
  value that crosses the `i32` range selects a new `i64` variant. Float
  scalars follow the precision of the array arguments.
- **Use instance attributes for values that change, class attributes for
  true constants.** Class constants are baked in, and each new value
  compiles a new variant.
- **Define `@tack.device_class` classes once, at module level.** A class
  defined per call is a new identity each time. Its variants are released
  when it is collected, but each call still pays a cold compile.
- **Expect re-specialization when a template's array changes shape.** Its
  full shape is part of the template key.

Specializing on baked-in dimensions is usually worth it: a known row
stride lets LLVM and the vendor compilers fold and vectorize around it.
The advice above is for the cases where the set of shapes is open-ended.

## Related pages

- [Compilation Pipeline](compilation-pipeline.md): what runs on a miss.
- [Memory and Aliasing](memory-and-aliasing.md): the overlap policy behind
  the disjoint bit.
- [Principles](principles.md): specialization over generality, and its
  costs.
- [Template System](../developers-guide/07-templates.md): how templates
  are rewritten.
