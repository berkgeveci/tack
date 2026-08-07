"""Tests for HIP C code generation — no GPU required."""

import tack
from tack.codegen.hip_gen import generate_hip_source
from tack.lang.type_inference import infer_param_types


def _get_ir(kernel_fn, *field_shapes):
    """Helper: define a kernel, create dummy fields, run type inference, return IR."""
    tack.init(arch=tack.cpu)
    fields = []
    for shape in field_shapes:
        fields.append(tack.field(dtype=tack.f32, shape=shape))
    ir_func = kernel_fn.get_ir().functions[0]
    infer_param_types(ir_func, tuple(fields))
    return ir_func


class TestHIPCodeGen:
    """Verify that generated HIP C is correct without running on a GPU."""

    def test_includes_hip_header(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        ir_func = _get_ir(add, (64,), (64,), (64,))
        src = generate_hip_source(ir_func)
        assert '#include <hip/hip_runtime.h>' in src

    def test_extern_c_global(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        ir_func = _get_ir(add, (64,), (64,), (64,))
        src = generate_hip_source(ir_func)
        assert 'extern "C" __global__' in src

    def test_thread_index(self):
        @tack.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 42.0

        ir_func = _get_ir(fill, (64,))
        src = generate_hip_source(ir_func)
        assert 'blockIdx.x' in src
        assert 'blockDim.x' in src
        assert 'threadIdx.x' in src

    def test_bounds_guard(self):
        @tack.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 42.0

        ir_func = _get_ir(fill, (64,))
        src = generate_hip_source(ir_func)
        assert 'if (i >= __n__) return;' in src

    def test_restrict_pointers(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        ir_func = _get_ir(add, (64,), (64,), (64,))
        src = generate_hip_source(ir_func)
        assert 'float* __restrict__' in src

    def test_math_functions(self):
        @tack.kernel
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = sqrt(x[i])

        ir_func = _get_ir(kern, (64,), (64,))
        src = generate_hip_source(ir_func)
        assert 'sqrtf(' in src

    def test_conditional(self):
        @tack.kernel
        def relu(x, out):
            for i in range(x.shape[0]):
                if x[i] > 0.0:
                    out[i] = x[i]
                else:
                    out[i] = 0.0

        ir_func = _get_ir(relu, (64,), (64,))
        src = generate_hip_source(ir_func)
        assert 'if (' in src
        assert '} else {' in src

    def test_saxpy_structure(self):
        @tack.kernel
        def saxpy(x, y, out):
            for i in range(x.shape[0]):
                out[i] = 2.0 * x[i] + y[i]

        ir_func = _get_ir(saxpy, (64,), (64,), (64,))
        src = generate_hip_source(ir_func)
        # Should have all three field params as restrict pointers
        assert src.count('__restrict__') == 3
        # Should have the n parameter
        assert 'long long __n__' in src
