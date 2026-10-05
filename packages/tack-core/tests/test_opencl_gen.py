"""Tests for OpenCL C code generation — no GPU required.

Level Zero was the only backend with no codegen test at all. CUDA and HIP
each have one that runs host-side on any machine, so their generators were
checked on every commit while `opencl_gen.py` — 238 statements, and the
least similar of the three to its CUDA parent — sat at 10% coverage and
was exercised only by someone with an Intel GPU in hand.

The generator subclasses CUDACodeGen, so what matters here is everything
it *overrides*: OpenCL spells the qualifiers, the thread index, the
barriers and the math functions differently, and gets those wrong
silently — the code still compiles as C, it just addresses the wrong
memory or synchronizes the wrong scope.
"""


import pytest

import tack
from tack.codegen.opencl_gen import generate_opencl_source
from tack.codegen.reductions import field_reduction_source
from tack.lang.inspect_kernel import _prepare_ir
from tack.lang.type_inference import infer_param_types


def _source(kernel_fn, *args):
    """Run the front end for `args` and generate OpenCL C."""
    tack.init(arch=tack.cpu)
    ir_func = kernel_fn.get_ir().functions[0]
    infer_param_types(ir_func, tuple(args))
    return generate_opencl_source(ir_func)


def _annotated_source(kernel_fn, *args):
    """Like `_source`, but through the pipeline the backend actually runs.

    `_source` stops at `infer_param_types`, which carries the structural
    checks above but leaves expression nodes without a `dtype`. Anything the
    generator decides *from* an expression's type is therefore invisible to
    it, and a test written against it can assert on code no device receives.
    `_prepare_ir` is the real path, annotation included.
    """
    tack.init(arch=tack.cpu)
    function, _ = _prepare_ir(kernel_fn, args)
    return generate_opencl_source(function)


def _field(shape=(64,), dtype=tack.f32):
    return tack.field(dtype=dtype, shape=shape)


# ── Kernel signature ─────────────────────────────────────────────────

def test_kernel_qualifier():
    """OpenCL uses __kernel, not CUDA's extern "C" __global__."""

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    src = _source(add, _field(), _field(), _field())
    assert "__kernel void" in src
    assert "__global__" not in src
    assert 'extern "C"' not in src


def test_field_parameters_are_global_pointers():
    """Fields live in __global address space; getting this wrong is silent."""

    @tack.kernel
    def scale(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i] * 2.0

    src = _source(scale, _field(), _field())
    assert "__global float* tack_var_a_x" in src
    assert "__global float* tack_var_a_out" in src
    assert "restrict" not in src


def test_scalar_parameters_are_not_pointers():
    @tack.kernel
    def scale(x, out, alpha):
        for i in range(x.shape[0]):
            out[i] = x[i] * alpha

    src = _source(scale, _field(), _field(), 2.5)
    assert "float tack_var_a_alpha" in src
    assert "__global float* tack_var_a_alpha" not in src


def test_dtype_mapping():
    @tack.kernel
    def copy_ints(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i]

    src = _source(copy_ints, _field(dtype=tack.i32), _field(dtype=tack.i32))
    assert "__global int*" in src


# ── Thread indexing ──────────────────────────────────────────────────

def test_parallel_loop_index_is_64_bit_from_the_group_id():
    """CUDA's blockIdx*blockDim+threadIdx has no meaning in OpenCL C, and
    get_global_id(0) wraps at 2^32 on Intel's driver despite its size_t."""

    @tack.kernel
    def fill(out):
        for i in range(out.shape[0]):
            out[i] = 1.0

    src = _source(fill, _field())
    assert ("long tack_var_a_i = (long)get_group_id(0) * (long)get_local_size(0) "
            "+ (long)get_local_id(0);") in src
    assert "get_global_id" not in src
    assert "blockIdx" not in src


@pytest.mark.parametrize("dtype,opaque_neg,opaque_abs", [
    (tack.i8, True, True), (tack.i16, True, True),
    (tack.i32, False, True), (tack.i64, False, False),
])
def test_widenable_negation_and_abs_are_noinline(dtype, opaque_neg, opaque_abs):
    """IGC 2.7.11 folds the wrapped minimum into its magnitude when it can
    see these helpers' bodies; a call boundary is the one shape that held."""

    @tack.kernel
    def negate(x, neg, mag):
        for i in range(x.shape[0]):
            neg[i] = -x[i]
            mag[i] = abs(x[i])

    src = _annotated_source(negate, _field(dtype=dtype), _field(dtype=dtype),
                            _field(dtype=dtype))
    t = {tack.i8: "char", tack.i16: "short", tack.i32: "int", tack.i64: "long"}[dtype]
    for op, opaque in (("neg", opaque_neg), ("abs", opaque_abs)):
        name = f"__tack_{op}_{dtype.name}"
        assert (f"__attribute__((noinline)) {t} {name}_noinline__(" in src) == opaque
        assert (f"static inline {t} {name}__(" in src) == (not opaque)


@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_native_reduction_index_is_64_bit_from_the_group_id(op):
    src = field_reduction_source("opencl", op)
    assert "long i = (long)get_group_id(0) * (long)get_local_size(0) + tid;" in src
    assert "get_global_id" not in src
    assert "threadIdx" not in src
    assert "blockDim" not in src


def test_bounds_guard_is_emitted():
    """The grid is rounded up to the workgroup size, so the tail must exit."""

    @tack.kernel
    def fill(out):
        for i in range(out.shape[0]):
            out[i] = 1.0

    src = _source(fill, _field())
    assert "return" in src


# ── Shared memory and synchronization ────────────────────────────────

def test_shared_memory_is_local_address_space():
    @tack.kernel
    def reduce_ish(x, out):
        for i in range(x.shape[0]):
            buf = tack.shared(tack.f32, 256)
            buf[tack.thread_id()] = x[i]
            tack.barrier()
            out[i] = buf[tack.thread_id()]

    src = _source(reduce_ish, _field(), _field())
    assert "__local float tack_var_a_buf[256]" in src
    assert "__shared__" not in src


def test_barrier_names_its_fence():
    """OpenCL's barrier() takes a memory-fence flag; __syncthreads() does not."""

    @tack.kernel
    def sync(x, out):
        for i in range(x.shape[0]):
            buf = tack.shared(tack.f32, 64)
            buf[tack.thread_id()] = x[i]
            tack.barrier()
            out[i] = buf[0]

    src = _source(sync, _field(), _field())
    assert "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)" in src
    assert "__syncthreads" not in src


def test_thread_id_is_the_local_id():
    """Shared memory is per-workgroup, so the index must be local, not global."""

    @tack.kernel
    def local_idx(x, out):
        for i in range(x.shape[0]):
            buf = tack.shared(tack.f32, 64)
            buf[tack.thread_id()] = x[i]
            tack.barrier()
            out[i] = buf[tack.thread_id()]

    src = _source(local_idx, _field(), _field())
    assert "get_local_id(0)" in src


# ── Math ─────────────────────────────────────────────────────────────

def test_math_functions_have_no_f_suffix():
    """OpenCL overloads on type; sqrtf/sinf are CUDA spellings."""

    @tack.kernel
    def mathy(x, out):
        for i in range(x.shape[0]):
            out[i] = sqrt(x[i]) + sin(x[i]) + cos(x[i]) + exp(x[i])

    src = _source(mathy, _field(), _field())
    for fn in ("sqrt(", "sin(", "cos(", "exp("):
        assert fn in src, fn
    for fn in ("sqrtf(", "sinf(", "cosf(", "expf("):
        assert fn not in src, fn


def test_power_uses_pow_not_powf():
    @tack.kernel
    def powered(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i] ** 2.5

    src = _source(powered, _field(), _field())
    assert "pow(" in src
    assert "powf(" not in src


# ── min / max ─────────────────────────────────────────────

def test_integer_min_max_never_reach_the_float_family():
    """`fmin(int, int)` is a compile error in OpenCL, not a promotion.

    OpenCL overloads `fmin`/`fmax` over float, double and half, and
    `min`/`max` over the integer types. Neither family takes an argument
    that needs converting, so an integer `fmin` has no unique best match
    and ocloc rejects it: "call to 'fmin' is ambiguous". CUDA's `fminf` is
    a single non-overloaded function, which is why the spelling inherited
    from CUDACodeGen looks right and broke every integer min/max on a
    device -- 18 failures in the Karras BVH builder, found on an Intel Max
    1100.

    What keeps integers away from `fmin` now is `codegen/integer_ops.py`,
    which lowers them to its own helpers for every backend rather than to
    a C library call. This test does not care which mechanism does it; it
    pins the invariant the device cares about, so that routing integers
    back through the float family cannot pass unnoticed again.
    """

    @tack.kernel
    def span(a, b, out):
        for i in range(a.shape[0]):
            out[i] = min(a[i], b[i]) + max(a[i], b[i])

    src = _annotated_source(
        span, _field(dtype=tack.i32), _field(dtype=tack.i32), _field(dtype=tack.i32)
    )
    assert "fmin" not in src
    assert "fmax" not in src
    assert "__tack_min_i32__" in src
    assert "__tack_max_i32__" in src


def test_float_min_max_keep_the_float_family():
    """The integer spelling is just as wrong the other way round."""

    @tack.kernel
    def span(a, b, out):
        for i in range(a.shape[0]):
            out[i] = min(a[i], b[i]) + max(a[i], b[i])

    src = _annotated_source(span, _field(), _field(), _field())
    assert "fmin(" in src
    assert "fmax(" in src
    assert "(float)" in src


def test_min_max_promote_mixed_arguments():
    """An int/float pair is ambiguous unadorned; both sides reach it as float."""

    @tack.kernel
    def span(a, b, out):
        for i in range(a.shape[0]):
            out[i] = min(a[i], b[i])

    src = _annotated_source(
        span, _field(dtype=tack.i32), _field(dtype=tack.f32), _field(dtype=tack.f32)
    )
    call = src.split("fmin(")[1].split(";")[0]
    assert "(int)" not in call, f"an argument reached fmin with its own type: {call}"
    assert call.count("(float)") == 2


# ── Atomics ──────────────────────────────────────────────────────────

def test_integer_atomic_add():
    @tack.kernel
    def count(x, out):
        for i in range(x.shape[0]):
            tack.atomic_add(out, 0, 1)

    src = _source(count, _field(), _field(dtype=tack.i32))
    assert "atomic_fetch_add_explicit" in src
    assert "memory_scope_device" in src


def test_float_atomic_min_uses_compare_and_swap():
    """OpenCL has no float atomics; they must be built from integer CAS."""

    @tack.kernel
    def amin(x, out):
        for i in range(x.shape[0]):
            tack.atomic_min(out, 0, x[i])

    src = _source(amin, _field(), _field())
    assert "atomicMinFloat" in src
    assert "atomic_compare_exchange_weak_explicit" in src
    assert "memory_scope_device" in src
    assert "volatile __global" in src


def test_float_atomic_max_uses_compare_and_swap():
    @tack.kernel
    def amax(x, out):
        for i in range(x.shape[0]):
            tack.atomic_max(out, 0, x[i])

    src = _source(amax, _field(), _field())
    assert "atomicMaxFloat" in src
    assert "atomic_compare_exchange_weak_explicit" in src
    assert "memory_scope_device" in src


# ── Control flow ─────────────────────────────────────────────────────

def test_conditional():
    @tack.kernel
    def clamp_positive(x, out):
        for i in range(x.shape[0]):
            if x[i] > 0.0:
                out[i] = x[i]
            else:
                out[i] = 0.0

    src = _source(clamp_positive, _field(), _field())
    assert "if (" in src
    assert "else" in src


def test_sequential_for_stays_a_c_loop():
    @tack.kernel
    def inner(x, out):
        for i in range(x.shape[0]):
            acc = 0.0
            for j in range(10):
                acc = acc + x[j]
            out[i] = acc

    src = _source(inner, _field(), _field())
    assert "for (" in src


def test_while_loop():
    @tack.kernel
    def countdown(x, out):
        for i in range(x.shape[0]):
            j = 0
            while j < 10:
                j = j + 1
            out[i] = float(j)

    src = _source(countdown, _field(), _field())
    assert "while (" in src


def test_integer_cast():
    @tack.kernel
    def truncate(x, out):
        for i in range(x.shape[0]):
            out[i] = float(int(x[i]))

    src = _source(truncate, _field(), _field())
    assert "(int)" in src


# ── Generated source is well-formed ──────────────────────────────────

def test_braces_balance():
    """A structural smoke test — mismatched braces mean broken codegen."""

    @tack.kernel
    def busy(x, out):
        for i in range(x.shape[0]):
            acc = 0.0
            for j in range(4):
                if x[i] > 0.0:
                    acc = acc + sqrt(x[i])
                else:
                    acc = acc - 1.0
            out[i] = acc

    src = _source(busy, _field(), _field())
    assert src.count("{") == src.count("}")
    assert src.count("(") == src.count(")")


def test_no_cuda_spellings_leak_through():
    """The generator inherits from CUDACodeGen; nothing CUDA-only may survive."""

    @tack.kernel
    def mixed(x, out):
        for i in range(x.shape[0]):
            buf = tack.shared(tack.f32, 64)
            buf[tack.thread_id()] = sqrt(x[i])
            tack.barrier()
            out[i] = buf[tack.thread_id()]

    src = _source(mixed, _field(), _field())
    for cuda_only in ("__global__", "__shared__", "__syncthreads",
                      "blockIdx", "threadIdx", "blockDim", "sqrtf"):
        assert cuda_only not in src, f"CUDA spelling leaked: {cuda_only}"


# ── Conditional operator ─────────────────────────────────────────────

def test_float_ternary_condition_is_compared_against_zero():
    """OpenCL C rejects a float as the condition of `?:`.

    The condition must be of scalar integer type; a float is not converted
    but refused -- "used type 'float' where floating point type is not
    allowed". CUDA C++, HIP and MSL all accept it via implicit conversion to
    bool, and all four generators share one `_expr_ifexp`, so this failed
    only where it was never run: on an Intel device.
    """

    @tack.kernel
    def truth(x, out):
        for i in range(x.shape[0]):
            out[i] = 1 if x[i] else 0

    src = _annotated_source(truth, _field(), _field(dtype=tack.i32))
    assert "!= 0.0f" in src
    assert "?" in src          # still a ternary, not an if/else rewrite


def test_f64_ternary_condition_compares_against_a_double_zero():
    """The f32 literal would silently widen; spell the zero at the right width."""

    @tack.kernel
    def truth(x, out):
        for i in range(x.shape[0]):
            out[i] = 1 if x[i] else 0

    src = _annotated_source(truth, _field(dtype=tack.f64), _field(dtype=tack.i32))
    assert "!= 0.0 " in src or "!= 0.0)" in src
    assert "!= 0.0f" not in src


def test_integer_ternary_condition_is_left_alone():
    """An integer condition is already legal; it must not grow a comparison."""

    @tack.kernel
    def truth(x, out):
        for i in range(x.shape[0]):
            out[i] = 1 if x[i] else 0

    src = _annotated_source(truth, _field(dtype=tack.i32), _field(dtype=tack.i32))
    assert "!= 0" not in src


def test_float_if_statement_keeps_its_bare_condition():
    """`if (x)` on a float is legal OpenCL C, following C99 -- only `?:` is not.

    Verified on an Intel Data Center GPU Max 1100: the `if` form compiles and
    runs, the ternary does not. Rewriting both would be a cost with no cause.
    """

    @tack.kernel
    def truth(x, out):
        for i in range(x.shape[0]):
            if x[i]:
                out[i] = 1
            else:
                out[i] = 0

    src = _annotated_source(truth, _field(), _field(dtype=tack.i32))
    assert "!= 0.0f" not in src


# ── Local address space placement ────────────────────────────────────

def test_nested_shared_allocation_is_declared_at_kernel_scope():
    """OpenCL C admits `__local` only in a kernel's outermost scope.

    A declaration emitted where it is used -- here inside an `if` -- is a
    compile error: "variables in the local address space can only be declared
    in the outermost scope of a kernel function". CUDA and Metal both allow a
    nested `__shared__`/`threadgroup`, so the inherited placement is legal
    there and wrong only here.
    """

    @tack.kernel
    def nested(x, out):
        for i in range(x.shape[0]):
            t = tack.thread_id()
            if x[i] > -1.0:
                buf = tack.shared(tack.f32, 256)
                buf[t] = x[i]
                out[i] = buf[t]

    src = _annotated_source(nested, _field(), _field())
    lines = src.splitlines()
    decl = next(i for i, line in enumerate(lines) if "__local" in line)
    opening = next(i for i, line in enumerate(lines) if line.startswith("__kernel"))
    first_if = next(i for i, line in enumerate(lines) if line.strip().startswith("if ("))
    assert decl < first_if, "the declaration must precede any nested scope"
    assert lines[decl].startswith("    ") and not lines[decl].startswith("        "), \
        f"expected kernel-scope indentation, got {lines[decl]!r}"
    assert opening < decl


def test_block_reduction_scratch_is_declared_at_kernel_scope():
    """The same rule, for the reduction scratch buffer.

    This is the site the workgroup participation tests hit on hardware; a
    reduction inside a loop put its `__local float __breduce_smem_0__[256]`
    inside that loop's braces.
    """

    @tack.kernel
    def reduce_in_loop(data, out):
        for i in range(data.shape[0]):
            for j in range(2):
                out[i] = tack.block_sum(data[i])

    src = _annotated_source(reduce_in_loop, _field(), _field())
    lines = src.splitlines()
    decl = next(i for i, line in enumerate(lines) if "__breduce_smem" in line
                and "__local" in line)
    assert lines[decl].startswith("    ") and not lines[decl].startswith("        "), \
        f"expected kernel-scope indentation, got {lines[decl]!r}"


def test_one_declaration_per_shared_name():
    """Hoisting two scopes into one must not declare the same array twice."""

    @tack.kernel
    def twice(x, out):
        for i in range(x.shape[0]):
            t = tack.thread_id()
            if x[i] > 0.0:
                buf = tack.shared(tack.f32, 256)
                out[i] = buf[t]
            else:
                buf = tack.shared(tack.f32, 256)
                out[i] = buf[t] + 1.0

    src = _annotated_source(twice, _field(), _field())
    assert src.count("__local float") == 1


def test_conflicting_shared_declarations_are_refused():
    """Same name, different size: hoisting cannot keep both, so say so."""

    @tack.kernel
    def conflict(x, out):
        for i in range(x.shape[0]):
            t = tack.thread_id()
            if x[i] > 0.0:
                buf = tack.shared(tack.f32, 256)
                out[i] = buf[t]
            else:
                buf = tack.shared(tack.f32, 128)
                out[i] = buf[t]

    with pytest.raises(NotImplementedError, match="declared twice"):
        _annotated_source(conflict, _field(), _field())


# ── Device deviation: f64 floor/ceil ─────────────────────────────────

def test_f64_floor_and_ceil_restore_the_sign_of_zero():
    """Work around Intel's double `floor`/`ceil` dropping the sign of zero.

    On an Intel Data Center GPU Max 1100 (intel-opencl-icd 25.05.32567.17,
    `-cl-std=CL2.0`, no relaxed-math option requested), `floor(-0.0)` and
    `ceil(-0.1)` both return +0.0, where IEEE-754 roundToIntegral requires
    -0.0. The f32 overload is correct and every other f64 operation preserves
    signed zero, so the workaround is confined to these two at double width.
    Neither function ever changes the sign of its operand, so restoring it
    from the operand is exact -- and a no-op wherever the runtime is already
    right.
    """

    @tack.kernel
    def round_trip(a, out):
        for i in range(a.shape[0]):
            out[i, 0] = floor(a[i])
            out[i, 1] = ceil(a[i])

    src = _annotated_source(round_trip, _field(dtype=tack.f64),
                            _field(shape=(64, 2), dtype=tack.f64))
    assert src.count("copysign(") == 2
    assert "copysign(floor(" in src
    assert "copysign(ceil(" in src


def test_f32_floor_and_ceil_are_left_alone():
    """The f32 overload is correct, so it must not pay for the workaround."""

    @tack.kernel
    def round_trip(a, out):
        for i in range(a.shape[0]):
            out[i, 0] = floor(a[i])
            out[i, 1] = ceil(a[i])

    src = _annotated_source(round_trip, _field(), _field(shape=(64, 2)))
    assert "copysign" not in src


def test_other_f64_math_functions_are_not_wrapped():
    """Only floor and ceil deviate; sqrt and friends stay plain calls."""

    @tack.kernel
    def mixed(a, out):
        for i in range(a.shape[0]):
            out[i] = sqrt(abs(a[i])) + exp(a[i])

    src = _annotated_source(mixed, _field(dtype=tack.f64), _field(dtype=tack.f64))
    assert "copysign" not in src
