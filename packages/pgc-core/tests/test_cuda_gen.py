"""Tests for CUDA C code generation — no GPU required."""

import pgc
from pgc.lang.type_inference import infer_param_types
from pgc.codegen.cuda_gen import generate_cuda_source


def _get_ir(kernel_fn, *args):
    """Helper: define a kernel, create dummy fields/scalars, run type inference, return IR."""
    pgc.init(arch=pgc.cpu)
    ir_func = kernel_fn.get_ir().functions[0]
    infer_param_types(ir_func, tuple(args))
    return ir_func


def _field(shape=(64,), dtype=pgc.f32):
    return pgc.field(dtype=dtype, shape=shape)


class TestCUDACodeGen:
    """Verify that generated CUDA C is correct without running on a GPU."""

    def test_extern_c_global(self):
        @pgc.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        src = generate_cuda_source(_get_ir(add, _field(), _field(), _field()))
        assert 'extern "C" __global__' in src

    def test_thread_index(self):
        @pgc.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 42.0

        src = generate_cuda_source(_get_ir(fill, _field()))
        assert 'blockIdx.x' in src
        assert 'blockDim.x' in src
        assert 'threadIdx.x' in src

    def test_bounds_guard(self):
        @pgc.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 42.0

        src = generate_cuda_source(_get_ir(fill, _field()))
        assert 'if (i >= __n__) return;' in src

    def test_restrict_pointers(self):
        @pgc.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        src = generate_cuda_source(_get_ir(add, _field(), _field(), _field()))
        assert 'float* __restrict__' in src
        assert src.count('__restrict__') == 3

    def test_n_parameter(self):
        @pgc.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 1.0

        src = generate_cuda_source(_get_ir(fill, _field()))
        assert 'long long __n__' in src

    def test_math_functions(self):
        @pgc.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = sqrt(x[i])

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert 'sqrtf(' in src

    def test_sin_cos(self):
        @pgc.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = sin(x[i]) + cos(x[i])

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert 'sinf(' in src
        assert 'cosf(' in src

    def test_conditional(self):
        @pgc.kernel
        def relu(x, out):
            for i in range(x.shape[0]):
                if x[i] > 0.0:
                    out[i] = x[i]
                else:
                    out[i] = 0.0

        src = generate_cuda_source(_get_ir(relu, _field(), _field()))
        assert 'if (' in src
        assert '} else {' in src

    def test_sequential_for(self):
        @pgc.kernel
        def kern(a, b):
            for i in range(a.shape[0]):
                for j in range(10):
                    b[i] = a[i]

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert 'for (long long j = 0; j < 10; j++)' in src

    def test_while_loop(self):
        @pgc.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                j = 0
                while j < 10:
                    out[i] = out[i] + x[i]
                    j = j + 1

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert 'while (' in src

    def test_scalar_arg_not_pointer(self):
        @pgc.kernel
        def saxpy(x, y, out, alpha):
            for i in range(x.shape[0]):
                out[i] = alpha * x[i] + y[i]

        src = generate_cuda_source(_get_ir(saxpy, _field(), _field(), _field(), 2.5))
        # alpha should be a value, not a pointer
        assert 'float alpha' in src
        assert 'float* __restrict__ alpha' not in src

    def test_atomic_add(self):
        @pgc.kernel
        def kern(x, total):
            for i in range(x.shape[0]):
                pgc.atomic_add(total, 0, x[i])

        src = generate_cuda_source(_get_ir(kern, _field(), _field((1,))))
        assert 'atomicAdd(' in src

    def test_float_atomic_min_uses_cas(self):
        @pgc.kernel
        def kern(x, min_val):
            for i in range(x.shape[0]):
                pgc.atomic_min(min_val, 0, x[i])

        src = generate_cuda_source(_get_ir(kern, _field(), _field((1,))))
        assert 'atomicMinFloat(' in src
        assert '__device__ float atomicMinFloat' in src
        assert 'atomicCAS(' in src

    def test_float_atomic_max_uses_cas(self):
        @pgc.kernel
        def kern(x, max_val):
            for i in range(x.shape[0]):
                pgc.atomic_max(max_val, 0, x[i])

        src = generate_cuda_source(_get_ir(kern, _field(), _field((1,))))
        assert 'atomicMaxFloat(' in src
        assert '__device__ float atomicMaxFloat' in src
        assert 'atomicCAS(' in src

    def test_shared_memory(self):
        @pgc.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                s = pgc.shared(pgc.f32, 256)
                pgc.barrier()
                out[i] = x[i]

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert '__shared__' in src
        assert '__syncthreads()' in src

    def test_print_emits_printf(self):
        @pgc.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                print("val:", x[i])
                out[i] = x[i]

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert 'printf(' in src

    def test_int_cast(self):
        @pgc.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = float(int(x[i]))

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert '((int)(' in src

    def test_negation(self):
        @pgc.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = -x[i]

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert '(-' in src

    def test_power_uses_powf(self):
        @pgc.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = x[i] ** 2.0

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert 'powf(' in src

    def test_floor_div(self):
        @pgc.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = x[i] // 2.0

        src = generate_cuda_source(_get_ir(kern, _field(), _field()))
        assert 'floorf(' in src

    def test_min_max_builtins(self):
        @pgc.kernel
        def kern(x, y, out):
            for i in range(x.shape[0]):
                out[i] = min(x[i], y[i]) + max(x[i], y[i])

        src = generate_cuda_source(_get_ir(kern, _field(), _field(), _field()))
        assert 'fminf(' in src
        assert 'fmaxf(' in src
