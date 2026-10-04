# Kernel language contract (draft)

This is the draft contract for compiler hardening, updated for the fourth
numerical-semantics increment on 2026-10-03.
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
Kernels do not capture numerical values from the enclosing Python scope: reading a name
that is not a parameter and is never assigned raises `NameError` with the
kernel (or inlined device function) and source position. Local and shared
arrays are storage, not values: binding another name to one (`view = tmp`)
is rejected, while indexing it or passing it to a device function is
supported.
Supporting device assertions later would require changing this contract
and its rejection test together. Atomics and barriers are currently
statement-only operations; their return values are not supported.

**Required: device-function binding.** A static `@tack.func` call resolves
to the function object bound in the defining kernel or device function's
Python globals or closure. Imported aliases and module-qualified calls are
supported. Each nested callee uses its own defining namespace; arguments
use the caller's namespace. Importing another module with a same-named
function must not change either cold or cached compilation. Local names
and parameters shadow external callable bindings; runtime function values
and ordinary Python callables are rejected rather than resolved by name.
Resolution follows module dictionaries only and does not evaluate object
properties or arbitrary Python expressions. Recursive device calls are
rejected with a source diagnostic.

Template methods retain their original defining bindings after `self`
resolution. Generated method calls use a transformation-local map and a
source marker, so a user function with the same generated spelling remains
distinct. No process-wide device-function registry participates in
validation or lowering. Static bindings are consulted when an IR
specialization is first built; cached IR preserves the inlined bodies.
Rebinding a callable afterward is not a supported way to update compiled
kernels; recreate the kernel to capture changed bindings.

**Required: identifier preservation.** Python-legal parameter, local,
loop, allocation and kernel names retain their binding identity even when
they are target-language keywords, builtins or compiler helper names.
GPU code generation encodes every binding into a dedicated namespace and
all backends use a shared encoding for kernel entry lookup, including
Unicode names. Generated lowering temporaries, vector components,
template parameters and packed scalar buffers allocate fresh spellings
before code generation. These internal spellings are not a public ABI;
argument order and resource binding indices stay unchanged.

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
kernel arguments.

**Disjoint-storage specialization (CPU).** The CPU backend may compile a
second variant whose field pointers carry `noalias`, and uses it only for
calls it has checked. The check runs on every dispatch and compares the byte
ranges of the field arguments, never Python object identity: a call
qualifies when no field the kernel stores to, directly, atomically, or
through an inlined device function, shares a byte with any other field
argument. Fields that are only read may overlap each other. Calls that do
not qualify run the variant without the promise, so the overlap policy above
is unchanged and the two variants must agree on every race-free program.
The qualification is part of the compiled-variant key. `test_disjoint_fields.py`
covers the analysis, partial overlaps of imported storage, the generated
signatures, and switching between overlapping and disjoint calls.

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

CPU field loads and stores promise only byte alignment, since imported
buffers may be unaligned and byte-sized elements do not preserve four-byte
alignment. The numerical regression checks imported buffers at a one-byte
offset for all ten scalar types and verifies that emitted LLVM does not
claim stronger alignment than the storage provides.

**Current behavior:** Python floating-point arguments default to f32 unless
an f64 field argument establishes an f64 context. Integer scalar arguments
use i32, i64, or u64 based on magnitude; values outside the supported
64-bit integer ranges are rejected. Signed integer literals are classified
by their complete value, so `-9223372036854775808` fits i64. Locals receive one storage type
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
representable integers and exact f32 values.

**Required: integer floor division and remainder.** For integer operands,
`a // b` rounds the mathematical quotient toward negative infinity. The
remainder `a % b` satisfies `a == (a // b) * b + (a % b)`, has magnitude
less than `abs(b)`, and is zero or has the divisor's sign. For example,
`-7 // 3 == -3`, `-7 % 3 == 2`, `7 // -3 == -3`, and `7 % -3 == -2`.
This follows [Python's integer arithmetic rules](https://docs.python.org/3/reference/expressions.html#binary-arithmetic-operations).
Both operators use integer arithmetic, including i64/u64 values beyond
floating-point precision. Operands evaluate once in the ordering defined
above. Unsigned operands use unsigned division and remainder.

The operation uses the type annotation pass's promoted integer type; both
operands are converted to that type before arithmetic, and the result has
that type. This increment defines results when those conversions preserve
both input values and the quotient fits the promoted type. Integer promotion
now preserves both full input ranges, as defined below. In particular,
i32 with u32 promotes to i64.

**Required caller constraints:** the evaluated divisor is nonzero. For a
signed promoted type, the minimum representable value with divisor `-1`
is excluded for both `//` and `%`; the quotient is unrepresentable and
[LLVM's signed remainder also excludes this pair](https://llvm.org/docs/LangRef.html#srem-instruction).
Runtime exceptions or defined overflow results for these cases are not
promised. Guarding an operation with `if` or a conditional expression must
avoid executing it on the unselected path. Floating-point `//` and `%` are
outside this integer guarantee.

`test_integer_division.py` compares against Python integer arithmetic on
every available backend, including exhaustive valid i8 pairs, all integer
widths, unsigned high-bit values, boundary and seeded random inputs,
nested/local expressions, literal and lossless mixed-type promotion,
single evaluation of inlined operands, and guarded zero divisors. CUDA,
HIP, Metal, and OpenCL generators share typed C-family helpers; LLVM emits
signed truncating operations with floor correction, or unsigned operations.

**Required: floating floor division and remainder.** With either operand
floating, `a // b` and `a % b` convert both operands to the promoted
floating type: f64 if present, otherwise f32. Both results have that type;
`//` returns an integer-valued **float**, without narrowing to an integer.
An f64 output field alone does not widen f32 field operands; explicit f64
casts do. Mixed integer operands can lose precision in this conversion.

The operations use a truncating floating remainder, corrected to the
divisor's sign. The quotient is reconstructed from that remainder and
snapped to a nearby integral floating value, following
[CPython's floating division/remainder algorithm](https://github.com/python/cpython/blob/3.13/Objects/floatobject.c).
For finite operands this preserves Python-style floor/sign behavior near
rounded division boundaries; merely applying `floor(a / b)` is insufficient.
Floating rounding may make a corrected remainder equal to the divisor's
magnitude or prevent exact reconstruction of `a` from `q*b + r`.
This does not promise exact real arithmetic or general bitwise agreement.

A zero remainder has the divisor's sign. A zero quotient has the sign of
the true quotient, including when the dividend is signed zero. NaN in
either operand, or an infinite dividend, produces NaN for both operations
(NaN sign/payload is unspecified). With a finite dividend and infinite
divisor, zero dividends produce the signed zeros above; a nonzero dividend
of the same sign produces signed-zero quotient and the original dividend
as remainder, while opposite signs produce `-1.0` and the signed infinite
divisor as remainder. Overflow of a finite reconstructed quotient remains
floating infinity; it must not go through an invalid float-to-integer cast.
Floating negation flips zero's sign. Floating `!=` and truth tests treat
NaN as unequal/nonzero, so a nonzero-divisor guard does not accidentally
discard a NaN operation on CPU. Other ordered comparisons with NaN are
false. These expression rules are covered alongside the operator helpers;
the broader GPU math-mode policy for kernels without them remains open.

**Required caller constraint:** the evaluated divisor is nonzero. No
device `ZeroDivisionError` is promised; guards must preserve the stated
evaluation/side-effect ordering. Portable guarantees here exclude denormal
inputs and nonzero denormal intermediate/results: device denormal support
and flush behavior remain part of the open general floating-point policy.
Operands evaluate once, left to right, before the typed arithmetic.

CUDA kernels containing typed floating `//` or `%` compile without
`--use_fast_math`; Metal disables `fastMathEnabled` for those kernels.
This applies to the whole compiled kernel, so other arithmetic within it
also sees the stricter mode. Kernels without these operations retain their
existing settings. LLVM helpers use ordinary floating instructions without
fast-math flags; HIP/OpenCL retain their existing default math settings.
This scoped requirement follows the
[NVRTC fast-math options](https://docs.nvidia.com/cuda/nvrtc/index.html#supported-compile-options)
and [Metal compile options](https://developer.apple.com/documentation/metal/mtlcompileoptions/fastmathenabled).

**Behavior change:** CPU floating `%` previously used dividend-signed
truncating remainder, and its `//` directly floored the rounded division.
GPU floating `%` could emit an invalid integer-only C `%` expression;
GPU `//` forced f32 division and then narrowed to a 64-bit integer, losing
f64 precision and overflowing for large/nonfinite results. All five
emitters now preserve the annotated floating result type.

`test_float_division.py` compares to NumPy's typed `divmod` after explicit
operand conversion. It covers signs, rounded multiples and neighbours,
large/overflowed quotients, seeded finite inputs, NaNs/infinities/signed
zeros, every floating/integer type pair, explicit precision, nested
expressions, single evaluation and guarded zero divisors. Expected zeros
and exceptional classes/signs match exactly; tested finite nonzero results
use a four-ULP tolerance. This is a regression bound for these supported
inputs, not the still-open general arithmetic accuracy guarantee.

**Required: fixed-width arithmetic and integer conversion.** Integer `+`,
`-`, `*`, unary negation, and bitwise operations produce the low N bits at
the annotated result width. Signed types interpret those bits as two's
complement; unsigned types interpret them as nonnegative integers. Thus
i8 `127 + 1` gives `-128`, and u16 `65535 * 65535` gives `1`. Each nested
operation wraps before its result is used by another operation. Integer
`abs` preserves its input type; the signed minimum therefore remains the
minimum under this wrapping policy. Unsigned `abs` is the identity.
Integer `min`, `max`, and comparisons use exact integer values and the
promoted operand type, including unsigned high-bit values.

Promotion chooses the narrowest integer type that preserves both full
operand ranges. Equal signedness chooses the wider type. Mixed signedness
chooses a signed type with sufficient width: i8 with u16 gives i32, and
i16 with u32 gives i64. Mixing any signed type with u64 is rejected with
an explicit-cast diagnostic. This applies to arithmetic, comparisons,
conditional arms, and joining assignments to one local variable.
An explicit cast can express intentional wrapping before promotion. True
division and integer power have the distinct result-type rules below.

Integer-to-integer casts and integer field stores reduce the mathematical
value modulo 2^N at the destination width, then interpret the resulting
bits using the destination signedness. Widening therefore respects the
**source** signedness: i8 `-1` to u64 gives `2^64 - 1`, while u8 `255`
to i64 gives `255`. Narrowing and changing signedness are defined even
when the original value cannot be represented by the destination type.

Shifts preserve the left operand's type, independently of the count type.
Left shifts wrap at that width; signed right shifts fill with the sign bit,
and unsigned right shifts fill with zero. **Required caller constraint:**
the count is an integer in `[0, N)`. Invalid evaluated counts have no
portable result or promised runtime exception. Guarded unselected shifts
must not execute.

Floating-to-integer conversions truncate toward zero. **Required caller
constraints:** the input is finite and its truncated value fits the
destination type. NaNs, infinities, and out-of-range values have no portable
result or promised runtime exception. This increment tests unsigned
conversions and signedness through locals, but does not establish a general
floating-point rounding or optimization policy.

`test_integer_semantics.py` uses Python big integers as the oracle for
wrapping, all 64 integer cast pairs, mixed promotion, comparisons, exact
integer builtins, shifts, and complete-width literals/scalars. LLVM uses
the annotated width, source signedness for extension, and unsigned
operations where appropriate. Its wrapping arithmetic has no `nsw`/`nuw`
flags; see the [LLVM arithmetic specification](https://llvm.org/docs/LangRef.html#add-instruction).
The C-family generators use unsigned carriers of at least 32 bits to avoid
signed overflow caused by implicit integer promotion, then reconstruct
signed values using representable casts. Metal uses `as_type` to reinterpret
the bits. Signed right shift is emitted without relying on C's
implementation-defined behavior. See C11 draft N1570 sections 6.3.1.3,
6.3.1.4, 6.5, and 6.5.7 in the
[language specification](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf).

On the tested M1 Max, pipeline compilation crashes when wrapping 64-bit
additions are optimized inside loops with runtime bounds. Metal emits a
separate `noinline` i64/u64 addition helper for assignments that update a local
using its previous value in those loop bodies. Other operations and
additions remain inline. The dynamic-bound
regression covers empty ranges, overflowing i32 bounds, and steps 1 and 2;
this workaround needs performance and hardware validation on other Apple GPUs.

**Required: true division.** Integer `a / b` converts each operand to f32
and produces an f32 quotient, including signed/u64 pairs. It does not
truncate to an integer. For example, `7 / 2` gives `3.5`, and `(7 / 2) * 2`
gives `7.0`. This follows Python's distinction between true and floor
division, with Tack's default floating-point precision. To request f64 on
a capable backend, cast explicitly: `tack.f64(a) / tack.f64(b)`. An f64
destination field alone does not widen the division. With a floating-point
operand, `/` uses the promoted floating type: f64 if present, otherwise f32.

**Required caller constraint for integer operands:** the evaluated divisor
is nonzero. Guarded unselected divisions must not execute. Storing a
fractional result in an integer field follows the floating-to-integer
conversion constraints above. Large integer inputs may lose precision
during conversion, and this increment does not promise bitwise-identical
floating results or settle the general rounding/optimization policy.

**Required: integer power.** For two integer operands, `a ** e` and
two-argument `pow(a, e)` preserve the **base's** type independently of the
exponent's type. For nonnegative `e`, they produce the exact mathematical
power modulo 2^N, interpreted using the base's signedness. Intermediate
products wrap, without passing through floating-point `pow`. Thus i8
`3 ** 5` gives `-13`; an i8 base with a u64 exponent still produces i8.
`0 ** 0` is `1`. Every nonnegative exponent representable by its integer
type is supported, including u64's full range. Both operands evaluate once,
left to right, and guarded unselected operations must not execute.

**Required caller constraint:** an evaluated integer exponent is
nonnegative. Negative integer literals are rejected with a diagnostic;
dynamic negative integer exponents have no portable result or promised
runtime exception. Cast the base to floating point to use negative powers:
`tack.f32(a) ** e`. With either operand floating, `**` and `pow` convert
both operands to the promoted floating precision and use floating-point
power. Its accuracy and exceptional-input policy remain open.

**Behavior change:** prior integer `/` used truncating C-family division
or a CPU floating result narrowed back to its integer annotation. Prior
integer `**` used floating `pow` and could lose large results; integer
`pow` could also disagree with `**`. Use `//` for integer floor division,
or an explicit integer cast of `/` for truncation toward zero within the
conversion domain. Use a floating cast for floating power. Result types
are now consistent between annotation and all five emitters.

`test_division_and_power.py` uses Python modular exponentiation as an exact
oracle across all 64 integer base/exponent type pairs, exhaustive small
domains, high-bit exponents, and nested expressions. Division checks
fractional results, signedness, explicit f64 precision, side effects,
guards, and large integer inputs with a four-ULP regression tolerance.
That tolerance is for these tested inputs, not a general floating accuracy
guarantee. All GPU generators share unsigned-carrier power helpers; LLVM
uses an internal typed helper. Exponentiation by squaring bounds execution
to at most 64 iterations, including outside-domain negative signed counts.

The following policies remain
**open** and must be resolved before broader numerical conformance claims:

| Question | Current evidence | Decision needed |
|---|---|---|
| Remaining arithmetic domains | Fixed-width wrapping, casts/promotion, valid shifts, true division, integer power, and floating `//`/`%` are defined | Any extension beyond the stated invalid-operation and denormal constraints |
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

Stage five has implemented integer floor division/remainder, fixed-width
arithmetic and conversions, true division, integer power, and floating
floor division/remainder, including their promotion domains and caller
constraints above. The general floating-point policy and reductions
remain within stage five.
The workgroup/capability contract remains a separate stage.
