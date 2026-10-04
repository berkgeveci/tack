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


import tack
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang.ir_type_annotate import annotate_types
from tack.lang.type_inference import infer_param_types


def _source(kernel_fn, *args):
    """Run the front end for `args` and generate OpenCL C."""
    tack.init(arch=tack.cpu)
    ir_func = kernel_fn.get_ir().functions[0]
    infer_param_types(ir_func, tuple(args))
    return generate_opencl_source(ir_func)


def _annotated_source(kernel_fn, *args):
    """Like `_source`, but runs the type-annotation pass the backend runs.

    `_source` stops at `infer_param_types`, which carries the structural
    checks above but leaves every expression node without a `dtype`. The
    generator then falls back to `float` for all of them, so a test built
    on `_source` can assert on code no device ever receives. The real path
    is `pack_scalars` -> `annotate_types` -> codegen; see
    `LevelZeroBackend._build_variant`. Anything that reads an expression's
    type -- integer `min`/`max` is the first -- has to use this instead.
    """
    tack.init(arch=tack.cpu)
    ir_func = kernel_fn.get_ir().functions[0]
    infer_param_types(ir_func, tuple(args))
    annotate_types(ir_func)
    return generate_opencl_source(ir_func)


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
    assert "__global float* restrict x" in src
    assert "__global float* restrict out" in src


def test_scalar_parameters_are_not_pointers():
    @tack.kernel
    def scale(x, out, alpha):
        for i in range(x.shape[0]):
            out[i] = x[i] * alpha

    src = _source(scale, _field(), _field(), 2.5)
    assert "float alpha" in src
    assert "__global float* restrict alpha" not in src


def test_dtype_mapping():
    @tack.kernel
    def copy_ints(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i]

    src = _source(copy_ints, _field(dtype=tack.i32), _field(dtype=tack.i32))
    assert "__global int* restrict" in src


# ── Thread indexing ──────────────────────────────────────────────────

def test_parallel_loop_uses_get_global_id():
    """CUDA's blockIdx*blockDim+threadIdx has no meaning in OpenCL C."""

    @tack.kernel
    def fill(out):
        for i in range(out.shape[0]):
            out[i] = 1.0

    src = _source(fill, _field())
    assert "get_global_id(0)" in src
    assert "blockIdx" not in src
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
    assert "__local float buf[256]" in src
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
    assert "barrier(CLK_LOCAL_MEM_FENCE)" in src
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

def test_integer_min_max_avoid_the_float_family():
    """`fmin(int, int)` is a compile error in OpenCL, not a promotion.

    OpenCL overloads `fmin`/`fmax` over float, double and half, and
    `min`/`max` over the integer types. Neither family takes an argument
    that needs converting, so the integer call has no unique best match
    and ocloc rejects it: "call to 'fmin' is ambiguous". CUDA's `fminf`
    is a single non-overloaded function, which is why the spelling
    inherited from CUDACodeGen looked right and broke every integer
    min/max on the device -- the Karras BVH builder included.
    """

    @tack.kernel
    def span(a, b, out):
        for i in range(a.shape[0]):
            out[i] = min(a[i], b[i]) + max(a[i], b[i])

    src = _annotated_source(
        span, _field(dtype=tack.i32), _field(dtype=tack.i32), _field(dtype=tack.i32)
    )
    assert "min((int)" in src
    assert "max((int)" in src
    assert "fmin" not in src
    assert "fmax" not in src


def test_float_min_max_keep_the_float_family():
    """The integer spelling is just as wrong the other way round."""

    @tack.kernel
    def span(a, b, out):
        for i in range(a.shape[0]):
            out[i] = min(a[i], b[i]) + max(a[i], b[i])

    src = _annotated_source(span, _field(), _field(), _field())
    assert "fmin((float)" in src
    assert "fmax((float)" in src


def test_min_max_promote_mixed_arguments():
    """An int/float pair is ambiguous unadorned; both sides get the result type."""

    @tack.kernel
    def span(a, b, out):
        for i in range(a.shape[0]):
            out[i] = min(a[i], b[i])

    src = _annotated_source(
        span, _field(dtype=tack.i32), _field(dtype=tack.f32), _field(dtype=tack.f32)
    )
    assert "fmin((float)" in src
    # Neither argument may reach the call with its own type.
    assert "(int)" not in src.split("fmin(")[1].split(";")[0]


# ── Atomics ──────────────────────────────────────────────────────────

def test_integer_atomic_add():
    @tack.kernel
    def count(x, out):
        for i in range(x.shape[0]):
            tack.atomic_add(out, 0, 1)

    src = _source(count, _field(), _field(dtype=tack.i32))
    assert "atomic_add" in src


def test_float_atomic_min_uses_compare_and_swap():
    """OpenCL has no float atomics; they must be built from integer CAS."""

    @tack.kernel
    def amin(x, out):
        for i in range(x.shape[0]):
            tack.atomic_min(out, 0, x[i])

    src = _source(amin, _field(), _field())
    assert "atomicMinFloat" in src
    assert "atomic_cmpxchg" in src
    assert "volatile __global" in src


def test_float_atomic_max_uses_compare_and_swap():
    @tack.kernel
    def amax(x, out):
        for i in range(x.shape[0]):
            tack.atomic_max(out, 0, x[i])

    src = _source(amax, _field(), _field())
    assert "atomicMaxFloat" in src
    assert "atomic_cmpxchg" in src


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
