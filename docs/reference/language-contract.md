# Kernel language contract (draft)

This is the draft contract for compiler hardening, updated at stage three on
2026-10-02.
It defines the intended portable kernel model, identifies known violations,
and separates decisions still open for discussion. It is **not a claim that
the current implementation satisfies every requirement below**. The baseline
was reviewed at `05174e1`.

Tack is a language for numerical kernels expressed using Python syntax.
Its goals are predictable numerical results, portable CPU/GPU execution,
and a compiler small enough to maintain. It does not implement arbitrary
Python, provide autograd, or replace NumPy's general-purpose array API.

## Status and interpretation

- **Required** means an intended correctness property. A current violation
  is a compiler defect, including when existing examples happen to work.
- **Current behavior** describes the implementation without making that
  behavior a permanent language guarantee.
- **Proposed** means a policy this draft recommends but that needs agreement
  before becoming a public guarantee.
- **Open** means no portable result is promised by this draft for that case.
  An open decision is not permission to change existing behavior silently.

Known defects are listed under [Regression baseline](#regression-baseline).
This draft governs the hardening work where older guides disagree. The
guides remain useful examples, but are not an exhaustive semantics specification.

## Source and supported constructs

**Required:** the frontend must either translate a construct according to
its defined semantics or reject it with a diagnostic. Silently dropping a
statement, operand, or control-flow effect is not acceptable. Diagnostics
identify the construct, kernel or device function, and source location.
The frontend validates original source before lowering, including
unreachable statements in device functions. Locations currently refer to
the captured, dedented source, rather than absolute file coordinates.

The current language surface includes the following families. Inclusion
does not imply that every Python use of the syntax is supported or that
all corner cases have been validated.

| Family | Kernel surface | Boundary |
|---|---|---|
| Values | Numeric literals, scalar parameters, field loads, local variables | Fixed-width Tack types, not arbitrary Python objects. Assigning to a scalar parameter makes it a per-iteration local (LC6) |
| Arithmetic | Arithmetic, comparisons, Boolean expressions, explicit casts, listed math builtins | Numerical and evaluation rules below |
| Assignments | Local assignment, augmented assignment, field stores, supported tuple unpacking | Storage and ordering rules below |
| Control flow | `range`, `tack.ndrange`, nested sequential loops, `while`, `if`/`elif`/`else`, conditional expressions, `break`, `continue` | One top-level parallel iteration space in the portable baseline |
| Composition | `@tack.func` inlining and `@tack.data_oriented` templates | Static source transformation, not arbitrary Python calls. A `return` ends the function on its path; one inside a loop is rejected (LC7) |
| Storage | Scalar fields, vector fields, local arrays, shared memory, 3D textures | Backend capability restrictions apply |
| Parallel primitives | Atomics, barriers, thread index, block reductions | Workgroup requirements below |

Python exceptions, `try`, context managers, generators, and runtime Python
object manipulation are outside this baseline and rejected. So are imports,
assertions (LC4), nested definitions, comprehensions, annotated assignments,
keyword/starred arguments, and unsupported operators. Docstrings and `pass`
are explicit no-ops. Parameter annotations and decorators remain host
metadata; ordinary positional parameters without defaults are supported.
Supporting device assertions later would require changing this contract
and its rejection test together. Atomics and barriers are currently
statement-only operations; their return values are not supported.

**Current behavior:** kernels are compiled from inspectable source; a bare
REPL or dynamically generated function may not provide that source. Kernel
arguments are positional. Kernel return values are not a host result API;
results are written to fields. Device functions can return supported scalar
or tuple values for inlining.

## Execution and ordering

**Required:** the portable baseline has one top-level parallel `range` or
`ndrange` iteration space. Each iteration owns its local variables. There
is no defined order among different parallel iterations. Algorithms must
avoid conflicting cross-iteration memory accesses unless they use the
appropriate supported synchronization or atomic operation.

Within one iteration, statements and nested sequential loops obey program
order. A load observes an earlier store to the same location by that
iteration, absent a conflicting access from another iteration. This holds
across loop iterations, conditionals, and inlined device-function bodies.
An empty sequential loop does not execute its body. See LC1.

**Required:** scalar operands and function arguments evaluate from left to
right. Inlined calls retain their original execution position. `and` and
`or` short-circuit; a conditional expression evaluates only its selected
arm. Chained comparisons stop at the first false comparison and evaluate
the middle operands' effects once. Ordinary assignment evaluates its value
before its store index. Augmented assignment evaluates its index once and
reads the old value before evaluating the right-hand side. Sequential
`range` arguments are evaluated once before entering the loop, including
when its body changes a bound or step variable.

The top-level `range(start, end)` executes exactly the indices in
`[start, end)`, and no iteration when that interval is empty or reversed.
Backends launch their grids from zero, so the frontend moves a nonzero start
into the body; the length of the interval remains a per-dispatch value and
does not specialize the compiled kernel. See LC5.

Early exits must preserve the defined control flow. `continue` ends the
current iteration of its nearest enclosing loop, including the top-level
parallel one, where it skips the rest of that iteration only (LC8). This
frontend rejects `break` from the parallel loop: there is no ordered prefix
of parallel iterations to stop. Kernel `return` is also rejected; results
are written to fields. Positive `range` steps are supported, and literal
zero or negative steps are rejected. A dynamic step must be positive;
runtime validation of that caller constraint remains open. Multiple
top-level parallel loops and cross-iteration communication require further
validation before joining the portable baseline. Outer launch bounds must
be resolvable from host arguments and field metadata.

**Required:** optimizations preserve the observable behavior of every
defined, race-free program, whether or not a current example uses that
pattern. In particular, an invariant address does not make the stored value
invariant. Moving a computation must also respect whether it executes at
all; moving a load out of a zero-trip loop can introduce an invalid access.
When a pass cannot establish safety, it must leave the computation in place.

**Current implementation:** Tack performs conservative copy propagation,
retaining assignments and replacing only subsequent uses when neither the
copy nor its source is rebound anywhere in the kernel. One kernel-wide
assignment summary, including loop and allocation bindings, is reused in
nested blocks. This deliberately leaves some block-local copies to the
backend compiler instead of recounting every subtree. Custom load hoisting and CSE
are disabled until memory and control-flow analyses can establish their
safety. LLVM and the vendor compilers still perform their own optimizations.

## Memory and aliasing

Fields are typed, shaped storage associated with a backend. A field view or
an imported pointer can share storage with another field. Object identity
does not establish that two fields have different backing allocations.
Callers must keep external storage alive for its use and must use fields
compatible with the active backend.

**Required:** support overlapping field arguments by default, preserving
program order within each iteration. This includes passing the same field
twice and passing distinct views over the same storage. It does not make
cross-iteration data races valid. LC2 tests this policy. LLVM field parameters
carry no `noalias` promise; CUDA/HIP and OpenCL field pointers carry no
`__restrict__` or `restrict` promise. Metal loads field pointers from one
argument buffer, so overlapping fields are not passed as separate device-buffer
kernel arguments. A future opt-in disjoint-storage
specialization would need an explicit contract and a justification based
on storage overlap, rather than Python object identity.

**Required caller constraints for this baseline:** access only in-bounds
elements and initialized values; write only to writable storage. Bounds
checking and enforcement of read-only access inside kernels are not promised
by this draft. In particular, host-side `Field` write checks are not evidence
that generated kernels enforce the same restriction. Negative indices are
outside the portable baseline; Python's wraparound indexing is not promised.

## Types and numerical behavior

Fields have explicit fixed-width integer or floating-point element types.
Backend capabilities restrict which types can execute. Unsupported types
must be rejected rather than silently narrowed to a supported type.

**Current behavior:** Python floating-point arguments default to f32 unless
an f64 field argument establishes an f64 context. Integer scalar arguments
use i32 or widen to i64 based on magnitude. Locals receive one storage type
derived from their assignments. These are Tack rules, not Python's dynamic
typing rules. The hardening work must validate that inferred expression
types and emitted operations agree, including signedness and narrowing.

Comparisons, `not`, `and`, and `or` produce normalized i32 values `0` or `1`.
In particular, Tack's `and` and `or` return Boolean values, rather than the
selected operand that Python returns for numeric operands. Boolean literals
also have numeric values `0` and `1`. Short-circuiting governs execution
independently of this result-type rule. Stage three corrects CPU results
that previously sign-extended a true LLVM i1 to `-1`.

For the initial regression baseline, numerical expectations use small,
representable integers and exact f32 values. The following policies remain
**open** and must be resolved before broader numerical conformance claims:

| Question | Current evidence | Decision needed |
|---|---|---|
| Signed `//` and `%` | CPU integer `-3 // 2` produces `-1`, not Python's `-2` | Python floor semantics or explicitly specified alternative |
| Overflow and conversion | Fixed-width types and promotion rules exist | Overflow, out-of-range casts, mixed signed/unsigned values, invalid shifts, division by zero |
| Floating-point results | Backends use their own arithmetic and math implementations | Rounding, contraction/reassociation, NaNs, infinities, signed zero, denormals, error tolerances |
| Reductions | Parallel implementations may change operation order | Permitted order variation, determinism, and numerical tolerances |

The differential tests cover small exact integer results and an exact
floating-point promotion case; they do not settle the open numerical
policies by recording accidental outputs. Defined integer results should match
exactly across backends. Floating-point comparisons need stated tolerances
and supported input domains rather than a general bitwise-equality promise.

## Workgroups and synchronization

Shared memory, barriers, thread indices, and block reductions require an
explicit workgroup execution model. A barrier orders participating threads
within its workgroup; it is not a global barrier between parallel iterations
on different workgroups. Programs must not rely on divergent participation
or uninitialized shared memory.

**Current limitation:** the CPU generator treats `IRBarrier` as a no-op and
allocates shared arrays on the stack. A CPU result alone therefore cannot
validate cooperative GPU execution. Workgroup size, partial final groups,
barrier participation, atomic ordering/scope, and CPU support or rejection
for cooperative kernels require a separate capability contract and hardware
tests. The first-stage regressions deliberately need none of these features.

## Specialization and compilation identity

**Required:** reusing compiled code must have the same defined behavior as
compiling the current call independently. Every fact baked into code or its
calling convention must participate in specialization identity, including
argument category and dtype, vector width, relevant shape/stride constants,
texture lowering, and template structure/constants. See LC3.

Conversely, changing runtime scalar values or an outer launch length alone
should not force recompilation. A dimension referenced inside the kernel
body can require specialization even when that dimension also sets the
launch length. Class-level template constants and instance-level runtime
scalars retain their distinct roles.

The current key includes argument dtypes and field/scalar/texture categories,
vector widths, texture extents, and resolved shape dependencies. Template
identity includes the actual class, typed constants, field metadata, and
runtime scalar attribute names. Floating-point constants use their bit
patterns, distinguishing signed zeros and making NaN cache keys stable.
Different classes with identical names can have different methods, and
adding a runtime attribute changes the parameter
layout even when its value is not a compilation constant.

The frontend's cached IR template must remain pristine. Mutating passes
operate on a variant's copy. A backend cache must belong to the backend
configuration that compiled its entries; compiled code must not outlive
the resources needed to execute it.

## Regression baseline

`packages/tack-core/tests/test_compiler_contract.py` contains the executable
baseline. Expected results come from simple arithmetic or NumPy calculations,
not from recording Tack's current output.

| ID | Requirement | CPU evidence at the baseline |
|---|---|---|
| LC1 | Sequential field loads observe preceding writes | Three increments produce `1`; bypassing Tack IR optimization produces `3` |
| LC2 | Support for overlapping arguments | Write `1` through `a`, write `2` through aliased `b`, read `a`: returns `1` |
| LC3 | Vector widths specialize independently | Width 2 followed by width 3 reuses the first variant and gives incorrect squared norms |
| LC4 | Unsupported statements are rejected | `assert False` disappears from transformed IR without a diagnostic |
| LC5 | The top-level range honors its start | `range(3, 7)` over eight elements writes indices `0`–`6`; a stencil's `x[i - 1]` reads before the buffer |
| LC6 | Assignment to a scalar parameter takes effect | `value = value + 1` is lost on GPU, where every read of the name is rewritten to the packed scalar; on CPU an assignment inside a branch or loop reaches only the reads emitted after it |
| LC7 | A `return` in an inlined function ends that function | `if a > limit: return limit` followed by `return a` always yields `a` |
| LC8 | `continue` advances its loop | Any `continue` in a `for` loop fails LLVM verification on CPU; in the top-level loop it does not compile on GPU |

The suite includes empty/single-iteration and distinct-buffer controls,
same-field and reshape-view aliases, and both orders of vector-width changes.
Vector inputs have enough allocated storage to keep even the incorrect cached
indexing in bounds. The mutation reference bypasses Tack's IR optimizations
only; LLVM/vendor optimization remains active.

Stage one recorded these defects on CPU; subsequent Linux testing reproduced
LC1–LC3 on CUDA. Stage two removes their expected-failure markers: all
numerical cases are now ordinary assertions on every available backend.

**Metal validation, 2026-10-03:** at `922b642`, four overlap-contract cases
failed numerically on Apple M1 Max: same-field and reshape-view ordered
writes, inner-loop mutation through an alias, and a read after an alias store.
The same failures reproduce at the pre-stage-four `e265e7f`. Direct MSL kernel
buffer arguments must be disjoint under the [Metal language specification,
section 5.2](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
Metal now uses indirect field pointers in one argument buffer, supported since
Metal 2, rather than separate device-buffer arguments. The runtime refreshes
references on each cached dispatch and explicitly declares resource residency.
All four original failures pass on Apple M1 Max, with additional coverage for
changing alias relationships, all supported dtypes through imported buffer
wrappers, mixed scalar packs, and aliased atomic updates. Assertions remain
ordinary tests and vendor optimization remains enabled.
The suite also covers zero-trip local assignments, while-loop mutation,
mutation through aliases, CSE across alias stores, and copy propagation's
statement order, loop steps, and loop-variable bindings.

LC5 was found on CUDA hardware testing of stage two and predates it: every
backend resolved only the end of the top-level range. It is fixed, with
cases for interior, single-element, empty, and reversed intervals, a
dimension-derived bound at two lengths, and an empty `range(n)`. The last
previously failed at launch on CUDA with a driver error rather than a wrong
result; the GPU backends now return before launching an empty grid.

LC6–LC8 were found by reviewing stage two with hand-written kernels and
also predate it. All three are fixed. A scalar parameter the kernel assigns
to is renamed, at variant build, to a local seeded from the parameter at
the top of each iteration. Inlining restructures a function so that every
`return` is the last statement on its path. CPU loops gained a latch block
for `continue`, and a `continue` of the top-level loop leaves the kernel on
GPU. One stage-two case, a parameter read before and after its
reassignment, passed on GPU only because copy propagation rewrote the read;
it now has a twin with Tack's passes bypassed.

There are **33 numerical cases per backend**, plus six host-side cases:
LC4, rejection of `return` inside a loop, and one generated-source check
per GPU generator for the top-level `continue`.
`test_variant_cache.py` covers field/scalar calling conventions, same-named
template classes, typed template constants including signed zero, and changes
to runtime template attribute layouts.

Stage three fixes LC4 and removes its expected-failure marker. All contract
cases are ordinary assertions. `test_source_validation.py` adds rejection
and diagnostic coverage, while `test_differential.py` runs identical source
as a compiled kernel and as serial Python over independent NumPy arrays.
It compares every field, including fields mutated by inlined calls. Its
generated cases use a fixed grammar and reproducible seed IDs, varying
bounds, positive steps, branches, `break`, and `continue`. They exclude
atomics, cooperative workgroups, vector methods, and open numerical cases.

The differential tests exposed and fixed eager guarded calls, discarded
condition-call statements, delayed operand loads, repeated augmented-store
indices, and changing sequential range bounds. CPU lowering now uses
control-flow joins for Boolean and conditional expressions. Native GPU
expressions already short-circuit; their inlined statement effects now
stay inside the corresponding guards in shared IR. CPU validation alone
does not establish hardware correctness; each backend needs the runs below.

Run the baseline on all locally discoverable backends:

```bash
uv run --no-sync pytest packages/tack-core/tests/test_compiler_contract.py -v -rxX
```

Run frontend and differential coverage as ordinary assertions:

```bash
uv run --no-sync pytest packages/tack-core/tests/test_source_validation.py packages/tack-core/tests/test_differential.py -v
```

For a GPU handoff, initialize the requested backend explicitly first, then
select its collected tests. Initialization must succeed and the test output
must show that backend's cases; deselection or a skip is not validation.
For example, after installing the CUDA extra and runtime:

```bash
uv run --no-sync python -c 'import tack; tack.init(arch=tack.cuda)' && \
  uv run --no-sync pytest packages/tack-core/tests/test_compiler_contract.py packages/tack-core/tests/test_differential.py -k cuda -v -rxX
```

Use `tack.hip` / `-k hip` or `tack.level_zero` / `-k level_zero` for those
backends. `--no-sync` preserves the existing environment's backend extras.
The shared-frontend LC4 test runs in the unfiltered command. Record backend,
device, runtime/binding versions, pass/fail/xfail counts, and numerical
differences. Do not waive numerical failures by adding expected-failure markers.

## Subsequent stages

Stage one established the draft and reproductions. Stage two disables unsafe
transformations, supports overlapping arguments, and repairs specialization
identity. Stage three adds strict frontend rejection with source context,
defines guarded expression execution and Boolean values, and brings forward
differential/generated-program tests. Stage four adds shared structural
traversal and verification after lowering, resolution, inference, scalar
localization, optimization, GPU packing, and type annotation. These checks
run at template/variant construction and during inspection, preserving the
cache-hit dispatch path. They check structure, binding existence, loop
targets, and required resolution/type metadata; they do not prove definite
assignment, bounds safety, or barrier uniformity. Verifier role and attribute
tables are precomputed, while all checks still run at their pass boundaries.
Broader testing and the
numerical/capability decisions above remain subsequent work.
