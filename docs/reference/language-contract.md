# Kernel language contract (draft)

This is the draft contract for compiler hardening, updated for the fifth
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
false. These expression rules apply to ordinary kernels as well as the
operator helpers; see the floating-point execution policy below.

**Required caller constraint:** the evaluated divisor is nonzero. No
device `ZeroDivisionError` is promised; guards must preserve the stated
evaluation/side-effect ordering. Portable guarantees here exclude denormal
inputs and nonzero denormal intermediate/results, as specified by the
floating-point execution policy below. Extending denormal support would
require a capability policy or fallback.
Operands evaluate once, left to right, before the typed arithmetic.

All CUDA kernels compile without `--use_fast_math`; Metal disables
`fastMathEnabled` for all kernels. LLVM helpers use ordinary floating
instructions without fast-math flags; HIP/OpenCL retain default math
settings without unsafe options. The previous increment restricted safe
math to kernels with floating `//` or `%`; the execution policy below
extends it to every kernel, including runtime reduction sources. These
settings follow the
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

### Floating-point execution policy

**Required baseline:** f32 and f64 expressions use their annotated
precision. Floating operands promote to f64 if present, otherwise f32;
an output field does not widen the input computation. Floating math
builtins convert arguments to their annotated result precision before
the call and return that precision before enclosing arithmetic. Integer
arguments to `sqrt`, trigonometric, exponential and logarithmic functions
convert to f32 unless another argument is f64. Integer `abs`, `min`, `max`
and power retain the integer rules above. For example, storing `sqrt(i64)`
in an f64 field widens an f32 result; `sqrt(tack.f64(i64))` requests f64.

The caller must leave the CPU floating-point environment at its default
round-to-nearest, ties-to-even mode, with exceptions untrapped. Kernels do
not change or restore that environment. Basic arithmetic uses the backend's
normal floating operations; f32 division and square root are not promised
to be correctly rounded on every GPU. Overflow remains floating infinity
rather than narrowing through an integer. Floating `/` by signed zero
produces signed infinity for a finite nonzero numerator, and zero/zero
produces NaN; it does not raise a Python exception. This does not extend
the valid divisor domain of integer division or floating `//`/`%`.

**Required exceptional behavior:** ordinary operations preserve the
specified NaN/infinity classes, comparisons, truth tests and signed zeros.
In particular, `NaN != NaN` is true; other comparisons with NaN are false;
NaN is true in a condition; multiplying infinity by zero or subtracting
infinity from itself produces NaN. An optimizer must not replace `x-x`
or `x*0` by positive zero for arbitrary floating inputs. Unary negation
flips zero's sign; `abs`/`fabs` clear it. `floor` and `ceil` preserve signed
zero and nonfinite classes. `sqrt` preserves signed zero and positive
infinity and returns NaN for negative nonzero inputs. NaN payloads/signs,
signaling versus quiet NaNs, exception flags and traps are outside the
portable API.

Floating `min`/`max` prefer the numeric operand when exactly one input is
NaN and return NaN when both are NaN. This is a numerical-kernel rule,
rather than Python's order-dependent handling of NaNs in `min`/`max`.
When both inputs compare equal to zero, either input zero sign is allowed.
Reduction and atomic min/max are separate operations and do not acquire
these guarantees from the scalar builtin rule.

**Permitted contraction:** a backend may fuse adjacent multiply/add or
multiply/subtract operations into one rounding. Contraction can change
low bits, cancellation, intermediate overflow and zero signs. It is
permitted, not required; CPU and GPU need not choose the same result.
General reassociation of addition or multiplication is not permitted:
`(a+b)+c` and `a+(b+c)` retain their grouping. No cross-backend bitwise
reproducibility or rounding of every source intermediate is promised.
Tests use an exact rational oracle to accept either the separate or fused
result for a contraction-sensitive expression, and separately require
parenthesized sums with cancellation to retain their order.

**Compiler configuration:** CUDA explicitly selects `--ftz=false`,
`--prec-div=true`, `--prec-sqrt=true`, and `--fmad=true`, without
`--use_fast_math`. Metal sets `fastMathEnabled` false. Both apply to
ordinary kernels and runtime reduction sources. CPU emits no LLVM
fast-math flags; HIP and OpenCL receive no unsafe/relaxed-math options.
These choices preserve classes and grouping while retaining permitted
backend contraction. There is currently one policy, so no mutable math
mode is omitted from the variant cache key and no per-dispatch scan is
needed. A future selectable mode must participate in specialization
identity and preserve any operator-specific restrictions.

**Portable domain and accuracy:** nonzero denormal inputs, intermediates
and results remain outside this baseline. Safe math does not imply that
all hardware supports gradual underflow. Denormal capability reporting
or a software fallback would be needed to extend this domain. Math
builtins use standard backend routines, rather than deliberately selected
fast/native approximations. Except for the class/sign rules above,
transcendental and floating-power exceptional inputs outside their finite
mathematical domains have no portable result promise. Global error bounds
vary by function, precision and backend; there is no blanket ULP bound.
Algorithms requiring tighter accuracy must establish their own domain and
error budget. See [OpenCL numerical compliance](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html#opencl-numerical-compliance)
and the [CUDA math reference](https://docs.nvidia.com/cuda/cuda-math-api/index.html).

`test_float_semantics.py` checks CPU/GPU execution without introducing
`//` or `%` to select stricter compilation. It covers seeded normal inputs,
exceptional arithmetic, signed zeros, unsafe identity folds, comparisons
and truth, scalar min/max, parenthesized cancellation and permitted
contraction, sqrt edge cases, all fifteen math/power forms on bounded
domains, mixed integer/float arguments, f32 libm result rounding and
explicit widening. Standalone arithmetic/sqrt finite checks use four ULP;
the bounded math smoke checks use eight ULP against typed NumPy. These
are stated regression bounds for their tested domains, not an exhaustive
proof or a new general math-library guarantee. Host-sanitized CUDA/HIP
checks and OpenCL syntax checks supplement hardware testing.

**Behavior change:** CUDA's unconditional fast-math option and Metal's
default fast mode are removed. Programs may observe different low bits,
images and throughput. The prior Metal mode demonstrably replaced
nonfinite identity expressions with zero, folded NaN self-comparisons to
false, and reassociated parenthesized additions. CPU libm calls previously
failed while declaring an empty intrinsic; they now call the precision-
appropriate external `tanf`/`tan`, `asinf`/`asin`, `acosf`/`acos`,
`atanf`/`atan` and `atan2f`/`atan2` symbols. GPU math arguments explicitly
convert to their annotated type, avoiding ambiguous OpenCL overloads for
integer and mixed arguments. No performance improvement is claimed.

### Field and parallel reductions

`Field.sum()`, `min()`, `max()` and `mean()` reduce all logical elements
and return Python `float` values. Reshaping a field does not select an axis
or change the set of elements. CPU uses NumPy; GPU backends reduce eligible
f32 fields on the device and use the shared NumPy fallback for other dtypes.
Allocation and field/backend ownership requirements still apply. Current
CUDA/HIP/Metal field kernels require element counts fitting an unsigned
32-bit integer; larger native reductions are outside this contract.

**Accumulation precision:** f32 sums accumulate in f32, and f64 sums in
f64. Integer sums promote signed inputs to i64 and unsigned inputs to u64,
independently of the host word size, and wrap modulo 2^64 in that accumulator.
The accumulated value, or selected integer extremum, is then converted to
a Python float; integer results beyond binary64's exact range can round.
`mean()` divides that returned sum by the element count in host binary64;
it does not request a wider or compensated accumulator. Integer means
therefore inherit integer-sum wrapping.

**Empty and exceptional inputs:** an empty sum returns positive zero,
an empty mean returns NaN, and empty min/max raise `ValueError`. These
results require no device launch or storage read. This does not promise
that every backend allocator can create a zero-byte buffer. Floating
min/max propagate any NaN, accept both infinities and the full normal
finite range, and resolve zero ties independently of order: minimum prefers
negative zero and maximum positive zero. An all-negative-zero maximum
remains negative zero; an all-positive-zero minimum remains positive zero.
NaN payload/sign preservation is not promised. The selected non-NaN
extremum is exact in the input's storage precision.

Floating sums propagate NaNs and combine infinities using floating addition.
Opposite infinities give NaN; an infinity retains its sign when other
partial sums remain finite. Overflow can depend on grouping, including
whether an intermediate becomes infinite before cancellation. There is
no portable finite-result promise when intermediates overflow. A zero sum
may have either zero sign. Nonzero denormal inputs/intermediates/results
remain outside the floating baseline above.

**Order and reproducibility:** floating summation may reorder and regroup
elements, and may produce different low bits across calls, devices,
backends, compiler versions or implementations. Current GPU field sums
use 256-lane trees followed by atomically accumulated group partials;
group arrival order is unspecified. No deterministic or compensated sum
mode is currently exposed. Repeated equality in a test does not establish
a reproducibility guarantee. Floating extrema are order-independent under
the class/zero rules above, apart from NaN representation. Integer field
reductions have the defined accumulator/conversion behavior independently
of reduction order.

For n finite floating inputs with no overflow or nonzero underflow in any
partial sum, round-to-nearest addition, and n*u < 1, the supported absolute
error budget is `abs(computed_sum - exact_sum) <= gamma_n * sum(abs(x))`,
where `gamma_n = n*u/(1-n*u)`, `u = 2^-24` for f32 and `2^-53` for f64.
This conservative addition bound covers tree and serial accumulation.
It does not promise a small relative or ULP error near cancellation.
The exact sum is over stored input values; input conversion and errors in
expressions producing terms require separate budgets. Mean inherits the
sum error divided by n plus host division rounding. See
[Higham, The Accuracy of Floating Point Summation](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf),
especially the general summation analysis in section 3.

**Kernel reductions:** `block_sum`, `block_min` and `block_max` currently
require f32 arguments and return f32. Integer/f64 arguments require an
explicit `tack.f32(...)` conversion; they are rejected rather than silently
narrowed or given an inconsistent annotation. On correctly participating
GPU workgroups, block extrema use the same NaN/zero rules as field extrema,
and block sums permit order variation with the same addition budget over
their contributed terms. The tested GPU domain here is fully participating
256-lane groups. Full participation, launch sizes and partial groups
remain the separate workgroup contract below. CPU rejects block reductions
because it has no workgroup execution model.

User reductions combining `atomic_add` with block partials likewise have
unspecified accumulation order. Atomic min/max are separate backend
primitives: their NaN and signed-zero handling is outside the portable
atomic-extrema domain, which requires finite nonzero floating operands
and stored values. The field/block extrema guarantees do not imply those
atomic semantics. Atomic scope and supported widths are defined below;
workgroup participation follows its separate contract. The statistical algorithms in
`tack.algorithms.stats` use f32 atomic accumulators for floating statistics;
an f64 input alone does not establish f64 accuracy or determinism for them.

`test_reduction_semantics.py` covers full-range extrema, NaNs and zero ties
across groups and tails, exact and bounded-error sums, allowed cancellation
groupings, nonfinite classes, host fallback types and empty fields, explicit
block precision, and complete GPU groups. Host-sanitized helpers and full
CUDA/HIP/OpenCL syntax checks supplement actual backend execution.

The following extensions remain **open** before broader numerical
conformance claims:

| Question | Current evidence | Decision needed |
|---|---|---|
| Remaining arithmetic domains | Fixed-width wrapping, casts/promotion, valid shifts, true division, integer power, and floating `//`/`%` are defined | Any extension beyond the stated invalid-operation and denormal constraints |
| Floating-point extensions | Safe math, annotated precision, classes/signs, grouping and permitted contraction are defined above | Denormal support and tighter function/domain-specific accuracy or reproducibility guarantees |
| Reduction extensions | Accumulator precision, exceptional classes, extrema ties, permitted order variation and an absolute addition budget are defined above | Optional deterministic/compensated modes and wider statistical accumulators |

The differential tests cover small exact integer results and an exact
floating-point promotion case; they do not settle the open numerical
policies by recording accidental outputs. Defined integer results should match
exactly across backends. Floating-point comparisons need stated tolerances
and supported input domains rather than a general bitwise-equality promise.

## Workgroups and synchronization

Shared memory, barriers, thread indices, and block reductions require an
explicit workgroup execution model. A barrier orders participating threads
within its workgroup, fencing both shared and global field memory; it is not
a global barrier between parallel iterations on different workgroups.
Programs must not rely on divergent participation
or uninitialized shared memory.

**Target support:** backends declare `supports_workgroups` (CPU: false;
Metal/CUDA/HIP/Level Zero: true). CPU rejects kernels containing `shared`,
`shared_like`, `barrier`, `thread_id`, `block_sum`, `block_min` or `block_max`
before compilation or execution. Inspection in every mode and direct LLVM
generation enforce the same restriction. The diagnostic names the kernel,
backend and required primitives. Dispatch wraps the `NotImplementedError`
in its usual kernel `RuntimeError`; inspection and direct generation raise
`NotImplementedError` directly.

This is a structural requirement, including nested branches and inlined
device functions, even if the primitive would be unreachable at runtime.
CPU does not emulate workgroups. Use `local_array` or `local_array_like`
for private scratch arrays; these remain supported on CPU, as do ordinary
scalar kernels, host field reductions and supported atomic operations.

**GPU launch domain:** barriers and block reductions require complete
**256-lane workgroups**. Positive logical iteration counts must be divisible
by 256, including normalized stepped ranges and `ndrange`. Cold dispatch
rejects partial groups before compilation; cached dispatch rechecks every
count before execution without repeating analysis. Zero/negative counts
run nothing. Metal pipeline limits and Level Zero X/total device limits
must admit 256 lanes; smaller groups are rejected. CUDA/HIP launch width
matches the generated reduction tree. Shared-memory/thread-ID kernels
without collectives retain partial-grid support. Bounds and initialized
shared values remain the user's responsibility.

**GPU participation domain:** `workgroup_participation.py` conservatively
proves uniform control flow. Constants, scalar arguments, shape queries,
immutable runtime scalar packs and collective results are uniform across
a workgroup. Lane indices and ordinary memory/texture loads are varying.
Assignments and branch joins propagate this classification; loop fixed
points account for loop-carried conditions and varying `break`/`continue`
paths, including exits after a barrier that affect subsequent iterations.
Uniform scalar branches/loops and varying memory updates that reconverge
before a barrier are supported. Varying loops without collectives may
reconverge before a later barrier. Device functions are checked after inlining.

Collectives must occur inside the parallel body. Reductions in ternary
branches, later short-circuit operands or `while` conditions are rejected
even for uniform predicates: current source generators cannot preserve
their guarded or repeated evaluation. Move the reduction into an explicit
supported statement sequence. Other unproven cases, including field-loaded
conditions and mathematically uniform expressions such as `i // 256`, are
rejected too. This is a conservative domain, not a complete uniformity prover.

Failures raise `ValueError` naming the kernel, primitive and IR location or
launch count/size. Public inspection checks participation and logical counts.
Direct GPU generators check mutable IR afresh; their callers must validate
eventual launch counts and device limits. Variant construction checks before
optimization/packing; packed scalar parameters carry `_is_scalar_pack` to
preserve uniformity. The full-group requirement is cached on `KernelVariant`.

The all-participants requirement follows the
[CUDA synchronization specification](https://docs.nvidia.com/cuda/archive/12.9.1/pdf/CUDA_C_Programming_Guide.pdf)
and [OpenCL C barrier specification](https://registry.khronos.org/OpenCL/specs/3.0-unified/pdf/OpenCL_C.pdf).
Metal can produce smaller final groups with nonuniform dispatch; see
[Apple's grid-size guidance](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes).
The fixed width is Tack's implementation domain, not a general hardware
limit. No automatic padding or CPU workgroup emulator is supplied.

Generated block reductions include a barrier after every lane has read the
result, before the shared array can be reused by another loop iteration.
Callers need no extra barrier to protect the collective's result broadcast.

**Remaining limits:** this analysis does not establish race freedom,
initialization, bounds safety or termination, or synchronize different
workgroups. Hardware confirmation is required per backend; CPU results do
not validate cooperative GPU execution.

## Atomic field updates

`atomic_add`, `atomic_min` and `atomic_max` are statement-only operations on
one scalar element of a global field. They do not return the previous value.
`Backend.supported_atomic_dtypes` declares the implemented add/min/max domain,
independently of the ordinary field types or workgroup capability:

| Target | Atomic field types, for all three operations |
|---|---|
| CPU | i8/u8, i16/u16, i32/u32, i64/u64, f32/f64 |
| CUDA / HIP | i32/u32, i64/u64, f32/f64 |
| Metal / Level Zero | i32/u32, f32 |

Metal/Level Zero 64-bit atomics and GPU 8/16-bit atomics are outside this
implementation domain, even when ordinary fields of those widths are
supported. No reinterpretation as a neighbouring wider field is allowed.
Private/shared arrays, textures, scalar arguments, and targets that cannot
be traced uniquely to a global field parameter are rejected. Field-pointer
copies introduced by device-function inlining are traced to that parameter.
The checker runs before optimization, including structurally unreachable
operations and empty grids. Public inspection in every mode enforces the
same target domain. Direct generators check mutable IR afresh and do not
trust stale atomic dtype annotations. Unsupported operations/types/targets
raise `TypeError` naming the kernel, target, operation and reason.

Atomic targets require natural alignment (1/2/4/8 bytes for their width).
Allocated fields satisfy this; imported field addresses are checked before
cold compilation and on every cached dispatch. Unaligned targets raise
`ValueError` before updating storage. Cached variants retain parameter indices
and alignment requirements, not addresses; changed arguments are checked
without repeating IR analysis. Direct-codegen callers must validate the
addresses used by their eventual launch. Index bounds, allocated extent,
initialization and valid floating-to-integer conversion remain caller
requirements. Ordinary non-atomic field accesses retain their existing
support for unaligned imported storage.

The contributed scalar is converted to the field's dtype once before the
read-modify-write operation, using the existing scalar conversion domain.
Integer addition wraps modulo the target width. Integer extrema compare
with the target's signedness, including unsigned values above the signed
maximum. Floating addition rounds at field precision with unspecified
interleaving; exact reduction order and bitwise reproducibility are not
promised. Normal floating arithmetic constraints, including the denormal
exclusion, apply. Floating atomic extrema's portable domain remains **finite,
nonzero stored values and contributed operands**. NaNs, infinities and
signed-zero ties are outside that extrema domain; field/block extrema's
stronger exceptional-value policy does not apply to user atomic extrema.

**Ordering and scope:** updates to the same element are indivisible among
kernel participants across CPU worker threads or GPU workgroups on the
executing device. The portable memory order is **relaxed**. An atomic update
is not an acquire/release fence for other fields or a publication flag, and
there is no system-scope guarantee spanning concurrent host accesses or
other devices. Do not mix ordinary reads/writes with conflicting atomic
updates during a kernel unless appropriate workgroup synchronization orders
them. Workgroup barriers order shared and global field memory among that
workgroup's lanes; they do not synchronize different workgroups. Current
synchronous dispatch waits for completion before host readback or the next
kernel call, providing the supported way to consume a cross-workgroup result.
Atomics alone do not require a complete workgroup or uniform participation;
partial final groups and lane-dependent atomic branches are permitted.

CPU uses LLVM monotonic atomic read-modify-write instructions, including
unsigned extrema, and integer-bit compare-and-swap loops for floating min/max.
CUDA/HIP 64-bit operations use 64-bit integer CAS, avoiding optional native
f64-add and signed-i64-add overloads; f32 extrema also begin with an atomic
read. Metal uses correctly typed `atomic_int`/`atomic_uint` and f32 atomics.
OpenCL uses explicit relaxed, device-scope C11 atomics, including the final
combine in native field reductions. Its former legacy functions guarantee
only workgroup scope in the specification, which is insufficient for that
combine. Compare-and-swap retries compare integer bits and capture the
contributed value once; they do not repeat the user's value expression.
Explicit Metal/OpenCL user barriers fence global as well as shared memory;
internal shared-only reduction-tree barriers retain their narrower fences.

The ordering/lowering choices follow the
[LLVM atomic instructions](https://www.llvm.org/docs/LangRef.html#cmpxchg-instruction),
[CUDA atomic specification](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#atomic-functions),
[HIP atomic specification](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.1/how-to/hip_cpp_language_extensions.html#atomic-functions),
and [OpenCL atomic specification](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html#atomic-functions).
`test_atomic_contract.py` covers unsigned boundaries, wrapping/conversion,
actual contended CPU workers, GPU updates across groups and tails, aliases
and inlined targets, cached alignment checks, unsupported target rejection,
global-field publication within a workgroup, LLVM verification and GPU
source compilation checks. Device hardware confirmation is required for
CUDA/HIP/Level Zero; host syntax checks do not establish atomic scheduling
or memory-order behavior on those devices.

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

**CPU and HIP validation, 2026-10-04:** at `fea2ad7`, on one AMD Instinct
MI300X VF (gfx942) with ROCm 7.0.2, hip-python 7.2.2, and an Intel Xeon
Platinum 8568Y+ host (one socket, 20 cores, Linux 6.8). Both backends
initialized explicitly. The contract, source-validation, differential,
stage-five semantics, variant-cache, IR-clone and identifier suites gave
**772 passed, 3 skipped** on CPU and **518 passed, 0 skipped** under
`-k hip`. Of the HIP selection, 463 cases executed on the device; 53 are
host-side checks of generated HIP source, and 2 are CPU cases that match
only because `relationships` contains `hip`. With ROCm's clang on `PATH`,
the 40 clang-gated OpenCL/C++ syntax checks also ran; without it they skip.
The three CPU skips are a Metal f64 generator case and two GPU-only block
reduction patterns. There were no failures, expected failures, or
numerical differences. All 45 runnable examples passed on each backend
(four need optional `oidn`, `imgui_bundle`, VTK, or a Metal/CUDA backend),
as did `validate_all.py`.

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
constraints above. The floating-point baseline now defines safe compilation,
precision, exceptional classes, grouping and permitted contraction, with
explicit denormal and accuracy limits. Reduction order/accuracy guarantees
remain within stage five.
The workgroup/capability contract remains a separate stage.
