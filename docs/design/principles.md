# Design Principles

The [kernel language contract](../reference/language-contract.md) states
Tack's goals as *predictable numerical results, portable CPU/GPU
execution, and a compiler small enough to maintain*. The principles below
are how the code pursues those goals. Each one is stated, justified, traced
to where it shows in the implementation, and paired with what it costs.
They are the result of experience as much as intent: several were adopted
after a silent wrong-numbers bug showed what the alternative did.

## 1. Python syntax, not Python semantics

**Statement.** A kernel is written in Python syntax, but it is a program in
a small, statically typed, statically compiled language. The Python
interpreter never runs it.

**Rationale.** Python's syntax gives users an editor, a parser, and
familiar loop and arithmetic notation for free. Python's *semantics*
(arbitrary-precision integers, dynamic types, closures, exceptions,
objects) cannot run on a GPU and would make CPU and GPU results disagree.
Tack keeps the syntax and defines its own meaning for it.

**Where it shows.**

- `Kernel` (`packages/tack-core/src/tack/lang/kernel.py`) re-parses the
  function's source text; the function object is used only to find that
  source and its global and closure bindings.
- Types are fixed-width and inferred at dispatch: `infer_param_types`
  (`lang/type_inference.py`) gives Python `int` arguments `i32`, `i64` or
  `u64` by value, and Python `float` arguments `f32` unless an `f64` field
  is present.
- Comparisons and `and`/`or` produce `i32` `0`/`1` rather than Python's
  selected operand. Floating `min`/`max` prefer the numeric operand over a
  NaN, unlike Python's order-dependent result.
- Kernels do not capture values from the enclosing scope:
  `KernelTransformer._check_names_bound` (`lang/ast_transform.py`) raises
  `NameError` instead.
- Where Python's semantics are well defined and implementable, Tack adopts
  them deliberately: integer `//` and `%` floor like Python's, and
  floating `//`/`%` follow CPython's algorithm.

**Costs and limits.** Code that looks like Python can behave differently
from Python, for example on integer overflow or on `and` of two numbers.
The contract, not the Python reference manual, is the specification.
Kernels need readable source, so a function created by `exec()` or typed
into a bare REPL cannot be a kernel.

## 2. Translate or reject

**Statement.** Every construct the frontend sees is either translated
according to its defined semantics or rejected with a diagnostic naming
the construct, the function and the source position. Nothing is silently
dropped.

**Rationale.** A dropped statement produces a kernel that compiles, runs
and returns the wrong answer, which is the worst failure a numerical tool
can have. An early version of the transformer did exactly this with
`assert` (contract regression **LC4**).

**Where it shows.**

- `validate_source` (`lang/source_validation.py`) checks the *original*
  kernel and device-function source with an allow-list visitor before
  lowering, including statements in unreachable code. It raises
  `UnsupportedSyntaxError`.
- `KernelTransformer.visit` converts the transformer's own
  `NotImplementedError`s into `UnsupportedSyntaxError` with a location, so
  any case the validator lets through and lowering cannot handle still
  fails loudly.
- `CallBindings` (`lang/call_bindings.py`) resolves device calls statically
  and rejects calls that are neither a bound `@tack.func` nor an intrinsic.
- Unsupported field dtypes are rejected by `check_dispatch_types`, never
  narrowed. Unsupported workgroup primitives and atomic types are rejected
  before compilation.
- `verify_ir` (`lang/ir_verify.py`) applies the same idea to the compiler
  itself: a pass that breaks an invariant stops the variant before code
  generation.

**Costs and limits.** The supported language is visibly smaller than
Python, and some constructs users try first (keyword arguments, `assert`,
comprehensions) are errors. Positions are in the dedented captured source,
not absolute file coordinates. See
[Compilation Pipeline](compilation-pipeline.md#3b-source-validation).

## 3. One kernel source; capabilities declared, not probed

**Statement.** A kernel's source is the same on all five backends. What
differs between backends is described by declared capability attributes,
which callers read instead of probing for methods or class names.

**Rationale.** Probing answers "is this method defined?", which is not the
same question as "is this supported?". `getattr(backend, 'supports_f64',
True)` once reported `True` for Metal, which has no `f64` at all, because
the attribute existed only on Level Zero. A missing
`memory_space()` method made `tack.memory_space()` answer `"cpu"` for a
Level Zero device pointer. A declaration states what *is* supported.

**Where it shows.**

- `Backend` (`packages/tack-core/src/tack/runtime/backend.py`) declares
  `name`, `display_name`/`label`, `supported_dtypes`,
  `supports_device_reductions`, `supports_workgroups`, `init_options`,
  `device_memory_spaces` and `dlpack_refusal_note`.
- Anything derivable is derived, so it cannot disagree with its source:
  `supports_f64` is `f64 in supported_dtypes`, and
  `supported_atomic_dtypes` intersects a per-target table with
  `supported_dtypes`. Level Zero sets `supported_dtypes` in `__init__`
  because `f64` support depends on the device.
- A single shared IR and pass pipeline (`resolve_variant` in
  `runtime/kernel_utils.py`) feeds five code generators. Device-dependent
  lowering choices, such as software texture sampling on HIP and Level
  Zero devices without image support, are made before the variant key is
  built, so they stay visible to the cache.

**Costs and limits.** Portability is bounded by the least capable
backend for any given feature: Metal has no `f64`, the CPU has no
workgroups, and atomic types vary by target. A kernel that uses such a
feature is rejected on backends without it rather than emulated. A few
places still identify backends by class name, notably
`inspect_kernel._generate_source` when choosing a generator. See
[Backend Capabilities](../contracts/backend-capabilities.md).

## 4. Specialize, and key everything you specialize on

**Statement.** Tack compiles a separate variant for each combination of
facts it bakes into generated code, rather than generating generic code
that reads those facts at run time. Every such fact is part of the
variant key.

**Rationale.** Baking in dtypes, vector widths, template constants and
row strides lets LLVM and the vendor compilers fold and vectorize around
known values, and keeps generated code simple. But specialization is only
correct if the cache key names everything that was baked in. The
alternative, a key that misses something, has produced silent wrong
numbers three times: a cache keyed on `id(kernel)`, a key without the
row stride, and a key without vector widths (**LC3**).

**Where it shows.**

- `kernel_variant_key` and `shape_signature` (`runtime/kernel_utils.py`)
  build the key from argument types and categories, vector widths,
  template structure, baked-in dimensions and the CPU disjoint bit.
- `ir_shape_deps` excludes the outermost loop's bound, which is passed at
  launch, so the most common kernel shape compiles once for every length.
- `_ClassToken` (`lang/kernel.py`) keys template classes by identity
  without keeping them alive.
- IR templates are immutable and passes run on a `clone_ir` copy, so one
  variant's resolution cannot leak into another.

**Costs and limits.** Variants multiply, and each pays a cold compile.
A dimension used inside the kernel body specializes on every value, so
open-ended shape sets need the length passed as a scalar. Template field
shapes are keyed in full, which is stricter than necessary. See
[Specialization and Caching](specialization-and-caching.md).

## 5. Correct by default; speed by measured opt-in

**Statement.** The default compilation preserves the defined behavior of
every race-free program. Faster code that depends on an extra assumption
is used only where that assumption has been checked for the call at hand,
and only where it has been measured to help.

**Rationale.** A numerical framework that is fast but occasionally wrong
cannot be trusted for anything. The original Tack made three unsafe
promises: field pointers were marked `noalias`/`restrict` unconditionally,
CUDA and Metal compiled with fast math, and custom LICM and CSE moved
loads without memory or control-flow analysis. Each was removed when a
case was found where it changed results (regressions **LC1** and
**LC2**; the Metal fast-math folds are listed in the contract's
floating-point policy).

**Where it shows.**

- **Aliasing.** Fields may overlap, and no generated parameter carries an
  unconditional no-alias promise (`LLVMCodeGen` in `codegen/llvm_gen.py`,
  the C-family generators, and Metal's argument buffer).
- **Measured opt-in.** The CPU backend compiles a second, `noalias`
  variant used only for calls whose written fields `fields_disjoint`
  proves do not overlap any other field. The check runs on every dispatch,
  and the result is part of the variant key.
- **Floating point.** CUDA compiles with `--ftz=false --prec-div=true
  --prec-sqrt=true --fmad=true` and without `--use_fast_math`; Metal sets
  `fastMathEnabled` to false; the CPU emits no LLVM fast-math flags. See
  the [floating-point policy](../reference/language-contract.md#floating-point-execution-policy).
- **Optimization.** `optimize_ir` (`lang/ir_optimize.py`) performs only
  conservative copy propagation; LICM and CSE stay disabled until they have
  the analyses they need.

**Costs and limits.** Safe defaults cost throughput in some kernels: an
accumulation through a field that may alias cannot be kept in a register,
and the contract notes that leaving fast math off may change throughput
as well as low-order bits. The
disjoint-field check adds a small cost to every CPU dispatch, and whether
it pays depends on the host. Users cannot currently opt into fast math or
declare no-alias themselves. See [Memory and Aliasing](memory-and-aliasing.md)
and [Numerical Semantics](numerical-semantics.md).

## 6. Measure runtime policy instead of hard-coding it

**Statement.** Decisions that depend on the machine and the kernel are
made from measurements taken at run time, not from constants chosen on
one developer's machine.

**Rationale.** The CPU backend's decision to spread a loop over threads
is the clearest case. The crossover between serial and threaded execution
moves by orders of magnitude with a kernel's arithmetic intensity, so no
single element count is right. A fixed threshold made mid-size dispatches
of cheap kernels slower than running them serially.

**Where it shows.**

- Each compiled CPU kernel keeps a smoothed estimate of its serial cost
  per element (`ns_per_elem`), from which the backend derives
  `parallel_min_elems`, so the hot path is one integer comparison.
- The backend measures its own fan-out cost (`_fan_out_ns`, in
  `CPUBackend._calibrate_fan_out`) by dispatching empty ranges through the
  real path, which runs no iterations and so has no side effects.
- `TACK_CPU_THREADS` remains as an explicit override.

**Costs and limits.** Measured policies are harder to reason about and to
test than constants, and they need care to keep measurement off the
critical path and robust to noise. Their behavior can vary from run to
run on a loaded machine. See [CPU Threading Policy](cpu-threading.md).

## 7. Correctness is defined by oracles

**Statement.** A result is correct when it matches an independent oracle,
not when it matches what Tack produced before. Tests compare against
arithmetic computed outside the compiler.

**Rationale.** Recording current output as the expected value freezes
bugs in place. Several of the defects behind the contract's regression
baseline were found only by comparing against an independent computation.

**Where it shows.**

- `packages/tack-core/tests/test_compiler_contract.py` holds the LC1–LC8
  regression baseline. Its expected values come from simple arithmetic or
  NumPy, not from recorded Tack output.
- `test_differential.py` runs the same source as a compiled kernel and as
  serial Python over NumPy arrays, including generated programs with
  branches, `break`, `continue` and varying bounds. It exposed eager
  guarded calls, delayed operand loads and repeated augmented-store
  indices.
- `test_integer_expression_differential.py` checks generated fixed-width
  integer expressions against an oracle written from the contract's rules,
  using nothing from the production compiler.
- `test_float_semantics.py` uses exact rational arithmetic (`Fraction`) to
  accept either the separately rounded or the fused result where
  contraction is permitted, and nothing else.
- Hardware validation is recorded per backend with device, runtime
  versions and pass/fail counts; a skip or deselection is not counted as
  validation.

**Costs and limits.** Oracles cover what they model. The differential
tests exclude atomics, cooperative workgroups, vector methods and open
numerical cases, and CPU validation does not establish GPU correctness.
See [Conformance and Validation](../contracts/conformance.md).

## Where the principles pull against each other

- **Specialization versus cold-start cost.** Baking facts in makes warm
  code faster but multiplies compiles. Verification at every pass boundary
  (principle 2 applied to the compiler) adds to each one. Tack accepts a
  slower first call in exchange for a fast and correct warm path.
- **Correctness versus speed.** The disjoint-field variant shows the
  intended resolution: keep the safe default, and add a faster path only
  where a per-call check proves the extra assumption, with the result in
  the cache key.
- **Portability versus capability.** Declared capabilities let a kernel
  fail clearly on a backend that cannot run it, but they do not make it
  run there. Tack rejects rather than emulates, apart from a few lowering
  fallbacks (software texture sampling) that preserve the same semantics.
