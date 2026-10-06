"""Tests for Metal Shading Language code generation — no GPU required."""

import tack
from tack.codegen.msl_gen import generate_msl_source
from tack.lang.type_inference import infer_param_types


def _get_ir(kernel_fn, *args):
    """Helper: define a kernel, create dummy fields/scalars, run type inference, return IR."""
    tack.init(arch=tack.cpu)
    ir_func = kernel_fn.get_ir().functions[0]
    infer_param_types(ir_func, tuple(args))
    return ir_func


def _field(shape=(64,), dtype=tack.f32):
    return tack.field(dtype=dtype, shape=shape)


class TestMSLCodeGen:
    """Verify that generated MSL is correct without running on a GPU."""

    def test_metal_header(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        src = generate_msl_source(_get_ir(add, _field(), _field(), _field()))
        assert '#include <metal_stdlib>' in src
        assert 'using namespace metal;' in src

    def test_kernel_void(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        src = generate_msl_source(_get_ir(add, _field(), _field(), _field()))
        assert 'kernel void tack_kernel_a_add(' in src

    def test_buffer_bindings(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        src = generate_msl_source(_get_ir(add, _field(), _field(), _field()))
        assert '[[buffer(0)]]' in src
        assert src.count('[[buffer(') == 1
        assert 'device float* tack_var_a_x [[id(0)]]' in src
        assert 'device float* tack_var_a_y [[id(1)]]' in src
        assert 'device float* tack_var_a_out [[id(2)]]' in src

    def test_device_pointers(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        src = generate_msl_source(_get_ir(add, _field(), _field(), _field()))
        assert 'device float*' in src

    def test_thread_position_in_grid(self):
        @tack.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 42.0

        src = generate_msl_source(_get_ir(fill, _field()))
        assert '[[thread_position_in_grid]]' in src
        assert '__tid__' in src

    def test_parallel_for_maps_to_tid(self):
        @tack.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 42.0

        src = generate_msl_source(_get_ir(fill, _field()))
        assert 'long tack_var_a_i = __tid__;' in src

    def test_math_functions(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = sqrt(x[i])

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert 'sqrt(' in src

    def test_sin_cos(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = sin(x[i]) + cos(x[i])

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert 'sin(' in src
        assert 'cos(' in src

    def test_conditional(self):
        @tack.kernel
        def relu(x, out):
            for i in range(x.shape[0]):
                if x[i] > 0.0:
                    out[i] = x[i]
                else:
                    out[i] = 0.0

        src = generate_msl_source(_get_ir(relu, _field(), _field()))
        assert 'if (' in src
        assert '} else {' in src

    def test_sequential_for(self):
        @tack.kernel
        def kern(a, b):
            for i in range(a.shape[0]):
                for j in range(10):
                    b[i] = a[i]

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert 'for (long tack_var_a_j = 0; tack_var_a_j < 10; tack_var_a_j++)' in src

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

        src = generate_msl_source(_get_ir(kern, _field(dtype=tack.i32), _field(dtype=tack.i32)))
        assert 'for (long tack_var_a_d = 0; tack_var_a_d < 4; tack_var_a_d++)' in src
        assert 'int tack_var_a_d = ' in src

    def test_while_loop(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                j = 0
                while j < 10:
                    out[i] = out[i] + x[i]
                    j = j + 1

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert 'while (' in src

    def test_scalar_arg_constant_buffer(self):
        @tack.kernel
        def saxpy(x, y, out, alpha):
            for i in range(x.shape[0]):
                out[i] = alpha * x[i] + y[i]

        src = generate_msl_source(_get_ir(saxpy, _field(), _field(), _field(), 2.5))
        # Scalar passed as constant buffer reference, not device pointer
        assert 'constant float& tack_var_a_alpha' in src
        assert 'device float* tack_var_a_alpha' not in src

    def test_atomic_add_float(self):
        @tack.kernel
        def kern(x, total):
            for i in range(x.shape[0]):
                tack.atomic_add(total, 0, x[i])

        src = generate_msl_source(_get_ir(kern, _field(), _field((1,))))
        assert 'atomic_fetch_add_explicit(' in src
        assert 'atomic_float' in src

    def test_float_atomic_min_cas(self):
        @tack.kernel
        def kern(x, min_val):
            for i in range(x.shape[0]):
                tack.atomic_min(min_val, 0, x[i])

        src = generate_msl_source(_get_ir(kern, _field(), _field((1,))))
        assert 'atomic_compare_exchange_weak_explicit(' in src
        assert 'as_type<float>' in src

    def test_float_atomic_max_cas(self):
        @tack.kernel
        def kern(x, max_val):
            for i in range(x.shape[0]):
                tack.atomic_max(max_val, 0, x[i])

        src = generate_msl_source(_get_ir(kern, _field(), _field((1,))))
        assert 'atomic_compare_exchange_weak_explicit(' in src

    def test_shared_memory(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                s = tack.shared(tack.f32, 256)
                tack.barrier()
                out[i] = x[i]

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert 'threadgroup' in src
        assert 'threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)' in src

    def test_shared_memory_adds_local_tid(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                s = tack.shared(tack.f32, 256)
                tack.barrier()
                out[i] = x[i]

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert '[[thread_position_in_threadgroup]]' in src

    def test_print_is_noop(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                print("val:", x[i])
                out[i] = x[i]

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert 'print not supported on Metal' in src

    def test_int_cast(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = float(int(x[i]))

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert '((int)(' in src

    def test_negation(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = -x[i]

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert '(-' in src

    def test_power(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = x[i] ** 2.0

        src = generate_msl_source(_get_ir(kern, _field(), _field()))
        assert 'pow(' in src

    def test_min_max_builtins(self):
        @tack.kernel
        def kern(x, y, out):
            for i in range(x.shape[0]):
                out[i] = min(x[i], y[i]) + max(x[i], y[i])

        src = generate_msl_source(_get_ir(kern, _field(), _field(), _field()))
        assert 'min(' in src
        assert 'max(' in src
