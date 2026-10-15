"""Tests for CUDA C code generation — no GPU required."""

import pytest

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang.inspect_kernel import _prepare_ir
from tack.lang.type_inference import infer_param_types


def _get_ir(kernel_fn, *args):
    """Helper: define a kernel, create dummy arrays/scalars, run type inference, return IR."""
    tack.set_default_device("cpu")
    ir_func = kernel_fn.get_ir().functions[0]
    infer_param_types(ir_func, tuple(args))
    return ir_func


def _array(shape=(64,), dtype=tack.f32):
    return tack.empty(shape, dtype)


class TestCUDACodeGen:
    """Verify that generated CUDA C is correct without running on a GPU."""

    def test_extern_c_global(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        src = generate_cuda_source(_get_ir(add, _array(), _array(), _array()))
        assert 'extern "C" __global__' in src

    def test_thread_index(self):
        @tack.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 42.0

        src = generate_cuda_source(_get_ir(fill, _array()))
        # Widened before the multiply: blockIdx.x * blockDim.x is 32-bit
        # unsigned and wraps once a launch reaches 2^32 threads.
        assert ('long long tack_var_a_i = (long long)blockIdx.x * blockDim.x '
                '+ threadIdx.x;') in src
        assert 'long long __n__' in src

    def test_bounds_guard(self):
        @tack.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 42.0

        src = generate_cuda_source(_get_ir(fill, _array()))
        assert 'if (tack_var_a_i >= __n__) return;' in src

    def test_array_pointers_allow_aliasing(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        src = generate_cuda_source(_get_ir(add, _array(), _array(), _array()))
        assert 'float* tack_var_a_x' in src
        assert '__restrict__' not in src

    def test_n_parameter(self):
        @tack.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 1.0

        src = generate_cuda_source(_get_ir(fill, _array()))
        assert 'long long __n__' in src

    def test_math_functions(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = sqrt(x[i])

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert 'sqrtf(' in src

    def test_sin_cos(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = sin(x[i]) + cos(x[i])

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert 'sinf(' in src
        assert 'cosf(' in src

    def test_conditional(self):
        @tack.kernel
        def relu(x, out):
            for i in range(x.shape[0]):
                if x[i] > 0.0:
                    out[i] = x[i]
                else:
                    out[i] = 0.0

        src = generate_cuda_source(_get_ir(relu, _array(), _array()))
        assert 'if (' in src
        assert '} else {' in src

    def test_sequential_for(self):
        @tack.kernel
        def kern(a, b):
            for i in range(a.shape[0]):
                for j in range(10):
                    b[i] = a[i]

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert 'for (long long tack_var_a_j = 0; tack_var_a_j < 10; tack_var_a_j++)' in src

    def test_loop_variable_rebound_as_a_local_is_declared(self):
        """The for-header declaration ends with its block; a later plain
        assignment to the same name is a new local and needs its own."""
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                acc = 0
                for d in range(4):
                    acc = acc + d
                d = tack.i32(x[i])
                out[i] = acc + d

        src = generate_cuda_source(_get_ir(kern, _array(dtype=tack.i32), _array(dtype=tack.i32)))
        assert 'for (long long tack_var_a_d = 0; tack_var_a_d < 4; tack_var_a_d++)' in src
        assert 'int tack_var_a_d = ' in src

    def test_while_loop(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                j = 0
                while j < 10:
                    out[i] = out[i] + x[i]
                    j = j + 1

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert 'while (' in src

    def test_scalar_arg_not_pointer(self):
        @tack.kernel
        def saxpy(x, y, out, alpha):
            for i in range(x.shape[0]):
                out[i] = alpha * x[i] + y[i]

        src = generate_cuda_source(_get_ir(saxpy, _array(), _array(), _array(), 2.5))
        # alpha should be a value, not a pointer
        assert 'float tack_var_a_alpha' in src
        assert 'float* tack_var_a_alpha' not in src

    def test_atomic_add(self):
        @tack.kernel
        def kern(x, total):
            for i in range(x.shape[0]):
                tack.atomic_add(total, 0, x[i])

        src = generate_cuda_source(_get_ir(kern, _array(), _array((1,))))
        assert 'atomicAdd(' in src

    def test_float_atomic_min_uses_cas(self):
        @tack.kernel
        def kern(x, min_val):
            for i in range(x.shape[0]):
                tack.atomic_min(min_val, 0, x[i])

        src = generate_cuda_source(_get_ir(kern, _array(), _array((1,))))
        assert 'atomicMinFloat(' in src
        assert '__device__ float atomicMinFloat' in src
        assert 'atomicCAS(' in src

    def test_float_atomic_max_uses_cas(self):
        @tack.kernel
        def kern(x, max_val):
            for i in range(x.shape[0]):
                tack.atomic_max(max_val, 0, x[i])

        src = generate_cuda_source(_get_ir(kern, _array(), _array((1,))))
        assert 'atomicMaxFloat(' in src
        assert '__device__ float atomicMaxFloat' in src
        assert 'atomicCAS(' in src

    def test_shared_memory(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                s = tack.shared(tack.f32, 256)
                tack.barrier()
                out[i] = x[i]

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert '__shared__' in src
        assert '__syncthreads()' in src

    def test_print_emits_printf(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                print("val:", x[i])
                out[i] = x[i]

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert 'printf(' in src

    def test_int_cast(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = float(int(x[i]))

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert '((int)(' in src

    def test_negation(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = -x[i]

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert '(-' in src

    def test_power_uses_powf(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = x[i] ** 2.0

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert 'powf(' in src

    def test_floor_div(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = x[i] // 2.0

        src = generate_cuda_source(_get_ir(kern, _array(), _array()))
        assert 'floorf(' in src

    def test_min_max_builtins(self):
        @tack.kernel
        def kern(x, y, out):
            for i in range(x.shape[0]):
                out[i] = min(x[i], y[i]) + max(x[i], y[i])

        src = generate_cuda_source(_get_ir(kern, _array(), _array(), _array()))
        assert 'fminf(' in src
        assert 'fmaxf(' in src


@tack.kernel
def _negate(x, neg, mag):
    for i in range(x.shape[0]):
        neg[i] = -x[i]
        mag[i] = abs(x[i])


@pytest.mark.parametrize("dtype,in_ptx", [
    (tack.i8, True), (tack.i16, True), (tack.i32, False), (tack.i64, False),
])
def test_narrow_signed_negation_is_inline_ptx(dtype, in_ptx):
    """ptxas 12.0-13.2 widen PTX neg.s16/abs.s16 without sign-extending
    from bit 15, so i16 -(-32768) read 32768 once widened (driver 595 on an
    H100). NVVM narrows any C spelling to those instructions; an inline-PTX
    32-bit subtract keeps them out. HIP and OpenCL keep their own forms."""
    tack.set_default_device("cpu")
    arrays = [_array(dtype=dtype) for _ in range(3)]
    function, _ = _prepare_ir(_negate, tuple(arrays))
    src = generate_cuda_source(function)
    for op in ("neg", "abs"):
        body = src.split(f"__tack_{op}_{dtype.name}__(")[1].split("}")[0]
        assert ("__tack_ptx_neg_u32__(" in body) == in_ptx
    assert ('asm("sub.u32 %0, 0, %1;"' in src) == in_ptx
    for generate in (generate_hip_source, generate_opencl_source):
        function, _ = _prepare_ir(_negate, tuple(arrays))
        other = generate(function)
        assert "asm(" not in other and "__tack_ptx_neg_u32__" not in other
