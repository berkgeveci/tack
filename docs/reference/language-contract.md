# Kernel language contract (draft)

This is the draft contract for compiler hardening, updated at stage two on
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
should identify the construct and kernel; source locations are a later
hardening deliverable.

The current language surface includes the following families. Inclusion
does not imply that every Python use of the syntax is supported or that
all corner cases have been validated.

| Family | Kernel surface | Boundary |
|---|---|---|
| Values | Numeric literals, scalar parameters, field loads, local variables | Fixed-width Tack types, not arbitrary Python objects |
| Arithmetic | Arithmetic, comparisons, Boolean expressions, explicit casts, listed math builtins | Numerical and evaluation rules below |
| Assignments | Local assignment, augmented assignment, field stores, supported tuple unpacking | Storage and ordering rules below |
| Control flow | `range`, `tack.ndrange`, nested sequential loops, `while`, `if`/`elif`/`else`, conditional expressions, `break`, `continue` | One top-level parallel iteration space in the portable baseline |
| Composition | `@tack.func` inlining and `@tack.data_oriented` templates | Static source transformation, not arbitrary Python calls |
| Storage | Scalar fields, vector fields, local arrays, shared memory, 3D textures | Backend capability restrictions apply |
| Parallel primitives | Atomics, barriers, thread index, block reductions | Workgroup requirements below |

Python exceptions, `try`, context managers, generators, and runtime Python
object manipulation are outside this baseline. `assert` should be rejected
until Tack defines its device execution and failure-reporting behavior
(LC4). Supporting assertions later would require changing this contract
and its rejection test together.

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

Early exits must preserve the defined control flow. This draft does not
extend Python's sequential outer-loop `break` behavior to a parallel loop.
Multiple top-level parallel loops, outer-loop early exits, negative or
zero `range` steps, and cross-iteration communication need explicit
validation or a specified rejection policy before joining the portable
baseline. Positive step support already exists; its lowering still must
preserve the iteration sequence.

**Required:** optimizations preserve the observable behavior of every
defined, race-free program, whether or not a current example uses that
pattern. In particular, an invariant address does not make the stored value
invariant. Moving a computation must also respect whether it executes at
all; moving a load out of a zero-trip loop can introduce an invalid access.
When a pass cannot establish safety, it must leave the computation in place.

**Current implementation:** Tack performs conservative copy propagation,
retaining assignments and replacing only subsequent uses when neither the
copy nor its source is rebound in the block. Custom load hoisting and CSE
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
`__restrict__` or `restrict` promise. A future opt-in disjoint-storage
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

For the initial regression baseline, numerical expectations use small,
representable integers and exact f32 values. The following policies remain
**open** and must be resolved before broader numerical conformance claims:

| Question | Current evidence | Decision needed |
|---|---|---|
| Signed `//` and `%` | CPU integer `-3 // 2` produces `-1`, not Python's `-2` | Python floor semantics or explicitly specified alternative |
| Boolean and conditional evaluation | CPU Boolean lowering is eager; conditional expressions emit LLVM `select` after computing operands | Whether guards short-circuit and which operands may be evaluated |
| Overflow and conversion | Fixed-width types and promotion rules exist | Overflow, out-of-range casts, mixed signed/unsigned values, invalid shifts, division by zero |
| Floating-point results | Backends use their own arithmetic and math implementations | Rounding, contraction/reassociation, NaNs, infinities, signed zero, denormals, error tolerances |
| Reductions | Parallel implementations may change operation order | Permitted order variation, determinism, and numerical tolerances |

This stage adds no test asserting that an accidental numerical result is
the desired permanent behavior. Defined integer results should match
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

The suite includes empty/single-iteration and distinct-buffer controls,
same-field and reshape-view aliases, and both orders of vector-width changes.
Vector inputs have enough allocated storage to keep even the incorrect cached
indexing in bounds. The mutation reference bypasses Tack's IR optimizations
only; LLVM/vendor optimization remains active.

Stage one recorded these defects on CPU; subsequent Linux testing reproduced
LC1–LC3 on CUDA. Stage two removes their expected-failure markers: all
numerical cases are now ordinary assertions on every available backend.
The suite also covers zero-trip local assignments, while-loop mutation,
mutation through aliases, CSE across alias stores, and copy propagation's
statement order, loop steps, and loop-variable bindings. There are
**17 numerical cases per backend**, plus one frontend case.
`test_variant_cache.py` covers field/scalar calling conventions, same-named
template classes, typed template constants including signed zero, and changes
to runtime template attribute layouts.

LC4 remains a shared-frontend strict expected failure, restricted to failure
to raise the required diagnostic. A fix producing an unexpected pass fails
the normal suite until its marker is removed. CPU validation of stage two
does not establish that the GPU fixes pass on hardware; each backend still
needs the runs below.

Run the baseline on all locally discoverable backends:

```bash
uv run --no-sync pytest packages/tack-core/tests/test_compiler_contract.py -v -rxX
```

Expose the remaining frontend defect as a normal test failure for diagnosis:

```bash
uv run --no-sync pytest packages/tack-core/tests/test_compiler_contract.py --runxfail -v
```

For a GPU handoff, initialize the requested backend explicitly first, then
select its collected tests. Initialization must succeed and the test output
must show that backend's cases; deselection or a skip is not validation.
For example, after installing the CUDA extra and runtime:

```bash
uv run --no-sync python -c 'import tack; tack.init(arch=tack.cuda)' && \
  uv run --no-sync pytest packages/tack-core/tests/test_compiler_contract.py -k cuda -v -rxX
```

Use `tack.hip` / `-k hip` or `tack.level_zero` / `-k level_zero` for those
backends. `--no-sync` preserves the existing environment's backend extras.
The shared-frontend LC4 test runs in the unfiltered command. Record backend,
device, runtime/binding versions, pass/fail/xfail counts, and numerical
differences. Do not waive numerical failures by adding expected-failure markers.

## Subsequent stages

Stage one established the draft and reproductions. Stage two disables unsafe
transformations, supports overlapping arguments, and repairs specialization
identity. Later stages add stage-specific IR verification, source diagnostics,
shared traversal support, and broader differential/generated-program testing.
Language decisions above must be settled explicitly before those tests
encode them as permanent guarantees.
