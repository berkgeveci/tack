"""Tests for HIP C code generation — no GPU required."""

import tack
from tack.codegen.hip_gen import generate_hip_source
from tack.lang.ir_resolve import resolve_ir
from tack.lang.type_inference import infer_param_types
from tack.runtime.kernel_utils import dispatch_name_to_field


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
        # Widened before the multiply, as on CUDA.
        assert ('long long tack_var_a_i = (long long)blockIdx.x * blockDim.x '
                '+ threadIdx.x;') in src
        assert 'long long __n__' in src

    def test_bounds_guard(self):
        @tack.kernel
        def fill(out):
            for i in range(out.shape[0]):
                out[i] = 42.0

        ir_func = _get_ir(fill, (64,))
        src = generate_hip_source(ir_func)
        assert 'if (tack_var_a_i >= __n__) return;' in src

    def test_field_pointers_allow_aliasing(self):
        @tack.kernel
        def add(x, y, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + y[i]

        ir_func = _get_ir(add, (64,), (64,), (64,))
        src = generate_hip_source(ir_func)
        assert 'float* tack_var_a_x' in src
        assert '__restrict__' not in src

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
        # Field parameters do not promise disjoint storage
        assert '__restrict__' not in src
        # Should have the n parameter
        assert 'long long __n__' in src


def _texture_ir():
    """IR for a texture-sampling kernel, with the texture param marked.

    `infer_param_types` is what sets `_is_texture`, so passing a real
    Texture3D is enough — the hardware-sampling path is selected exactly as
    it would be on a device that has texture units.
    """
    tack.init(arch=tack.cpu)
    data = tack.field(dtype=tack.f32, shape=(64,))
    out = tack.field(dtype=tack.f32, shape=(1,))
    tex = tack.texture3d(data, shape=(4, 4, 4))

    @tack.kernel
    def sample(out, tex, count):
        for i in range(count):
            out[i] = tex.sample(0.5, 0.5, 0.5)

    # The transform has to be told which param is a texture, or `tex.sample`
    # is just an unknown method call.
    ir_func = sample.get_ir(texture_fields={'tex': (4, 4, 4)}).functions[0]
    # resolve fills IRTextureSample.shape, which codegen reads. Build the
    # name map the same way dispatch does, so the texture resolves to its
    # 3D extent rather than the flat field's length.
    args = (out, tex, 1)
    resolve_ir(ir_func, dispatch_name_to_field(ir_func, args))
    infer_param_types(ir_func, args)
    for param, arg in zip(ir_func.params, args):
        if getattr(param, '_is_texture', False):
            param._texture_shape = arg.shape_3d
    return ir_func


class TestHIPTextureHandleType:
    """The texture handle is the one type HIP does not spell like CUDA.

    `hip_gen` inherits the whole signature builder from `CUDACodeGen`, so
    the handle came out as `cudaTextureObject_t` and hipRTC rejected the
    kernel outright: "unknown type name 'cudaTextureObject_t'". No device
    is needed to see it — the string is in the generated source.
    """

    def test_hip_emits_hip_texture_object(self):
        src = generate_hip_source(_texture_ir())
        assert 'hipTextureObject_t' in src
        assert 'cudaTextureObject_t' not in src

    def test_cuda_still_emits_cuda_texture_object(self):
        """The shared constant must not have moved CUDA onto HIP's spelling."""
        from tack.codegen.cuda_gen import generate_cuda_source

        src = generate_cuda_source(_texture_ir())
        assert 'cudaTextureObject_t' in src
        assert 'hipTextureObject_t' not in src
