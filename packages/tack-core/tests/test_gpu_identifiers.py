"""Python identifiers survive GPU keywords, builtins and generated names."""

import copy
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
from compiler_tools import require_clang

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.identifiers import gpu_variable_name, kernel_entry_name
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang.inspect_kernel import _prepare_ir
from tack.lang.ir_pack_scalars import pack_scalars
from tack.lang.ir_traversal import walk_ir
from tack.lang.ir_type_annotate import annotate_types
from tack.lang.ir_verify import verify_ir

NAMES = [
    'default', 'namespace', 'template', 'typename', 'restrict', 'kernel',
    '__global', 'constant', 'half', 'uint', 'sampler_t', 'threadIdx', 'blockIdx',
    'get_global_id', 'sqrtf', 'sqrt', '__n__', '__tid__', '__tack_buffers__',
    '__samp__', '__tack_pow_i32__', '__pack_i32__', '__pack_i32_1__',
    'tack_var_a_default', 'tack_kernel_a_default', 'α', 'tack_var_u_ceb1',
    '_', 'Z', 'Z0', 'Z1',
]
GENERATORS = [generate_cuda_source, generate_hip_source,
              generate_msl_source, generate_opencl_source]


@pytest.fixture(params=NAMES)
def named_kernel(request, tmp_path, monkeypatch):
    name = request.param
    path = tmp_path / 'identifier_kernel.py'
    total = 'namespace_local' if name == 'namespace' else 'namespace'
    array = 'sampler_t_local' if name == 'sampler_t' else 'sampler_t'
    value = 'half_local' if name == 'half' else 'half'
    source = f'''
import tack
@tack.kernel
def {name}({name}, out, scale):
    for enum in range({name}.shape[0]):
        {total} = 0
        for register in range(3):
            {total} = {total} + {name}[enum] + register
        {array} = tack.local_array(tack.i32, 2)
        {array}[0] = {total}
        if enum % 2 == 0:
            {value} = {array}[0] + 1
        else:
            {value} = {array}[0] - 1
        out[enum] = {value} * scale
'''
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('_tack_identifier_test', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return getattr(module, name)


def _inputs():
    values = np.arange(7, dtype=np.int32) - 3
    data = tack.field(tack.i32, values.shape)
    data.from_numpy(values)
    out = tack.field(tack.i32, values.shape)
    return data, out


def _expected(scale):
    values = np.arange(7, dtype=np.int32) - 3
    return (3 * values + 3 + np.where(np.arange(7) % 2 == 0, 1, -1)) * scale


def test_keyword_parameters_locals_loops_allocations_and_entry(backend, named_kernel):
    data, out = _inputs()
    for scale in [2, -3, 2]:
        named_kernel(data, out, scale)
        np.testing.assert_array_equal(out.to_numpy(), _expected(scale))


@tack.kernel
def _helper_collisions(default, out, __pack_i32_1__):
    for __tid__ in range(default.shape[0]):
        __pack_i32__ = tack.i32(7)
        __tack_pow_i32__ = default[__tid__]
        __tack_floordiv_i32__ = tack.i32(3)
        __tack_mod_i32__ = tack.i32(2)
        tack_var_a_default = __tack_pow_i32__ ** 3
        tack_var_u_ceb1 = __tack_pow_i32__ // __tack_floordiv_i32__
        α = __tack_pow_i32__ % __tack_mod_i32__
        sqrtf = tack.f32(4.0)
        __n__ = sqrt(sqrtf)
        if __tid__ % 2 == 0:
            __tack_buffers__ = tack_var_a_default + tack_var_u_ceb1 + α
        else:
            __tack_buffers__ = tack_var_a_default - tack_var_u_ceb1 - α
        out[__tid__] = (__tack_buffers__ + __pack_i32__
                        + __pack_i32_1__ + tack.i32(__n__))


def test_helper_names_encoding_lookalikes_and_packed_local_collision(backend):
    data, out = _inputs()
    x = np.arange(7, dtype=np.int32) - 3
    for scalar in [5, -2, 5]:
        _helper_collisions(data, out, scalar)
        expected = x**3 + np.where(np.arange(7) % 2 == 0, x // 3 + x % 2,
                                   -(x // 3) - x % 2) + 9 + scalar
        np.testing.assert_array_equal(out.to_numpy(), expected)


def test_shared_allocation_and_thread_builtin_names(workgroup_backend):
    @tack.kernel
    def kernel(default, out):
        for __tid__ in range(default.shape[0]):
            threadgroup = tack.shared_like(default, 256)
            __local_tid__ = tack.thread_id()
            threadgroup[__local_tid__] = default[__tid__] + 1
            out[__tid__] = threadgroup[__local_tid__]

    data, out = _inputs()
    kernel(data, out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(7) - 2)


@tack.kernel
def _texture_names(__samp__, default, count):
    for __tid__ in range(count):
        default[__tid__] = __samp__.sample(0.5, 0.5, 0.5)


def _texture_inputs():
    data = tack.field(tack.f32, (8,))
    data.from_numpy(np.full(8, 7.0, dtype=np.float32))
    return tack.texture3d(data, shape=(2, 2, 2)), tack.field(tack.f32, (3,)), 3


def test_texture_and_sampler_names(backend):
    args = _texture_inputs()
    _texture_names(*args)
    np.testing.assert_array_equal(args[1].to_numpy(), np.full(3, 7.0))


@tack.func
def _identifier_bump(x):
    return x + 1


def test_lowered_temporaries_cannot_overwrite_user_bindings(backend):
    @tack.kernel
    def collide(out, scale):
        for i in range(out.shape[0]):
            __unpack_tmp_0_0__ = 100
            __unpack_tmp_0_1__ = 200
            a = 2
            b = 3
            a, b = b, a
            ___identifier_bump_x_1__ = 300
            ___identifier_bump_ret_1__ = 400
            value = _identifier_bump(i)
            __scale_local__ = 500
            scale = scale + i
            out[i] = (a * 10 + b + value + scale + __scale_local__
                      + __unpack_tmp_0_0__ + __unpack_tmp_0_1__
                      + ___identifier_bump_x_1__ + ___identifier_bump_ret_1__)

    out = tack.field(tack.i32, (7,))
    for scale in [5, -2, 5]:
        collide(out, scale)
        np.testing.assert_array_equal(out.to_numpy(), 1533 + scale + 2 * np.arange(7))


def test_vector_components_cannot_overwrite_scalar_bindings(backend):
    @tack.kernel
    def collide(out):
        for i in range(out.shape[0]):
            v__0 = tack.i32(100)
            v__1 = tack.i32(200)
            v = tack.Vector([i, 2, 3])
            v += tack.Vector([1, 2, 3])
            out[i] = v[0] + v[1] + v[2] + v__0 + v__1

    out = tack.field(tack.i32, (7,))
    collide(out)
    np.testing.assert_array_equal(out.to_numpy(), 311 + np.arange(7))


@tack.data_oriented
class _IdentifierModel:
    def __init__(self, field):
        self.default = field
        self.half = 3

    @tack.func
    def value(self, i):
        __tmpl_obj_default__ = 5
        __tmpl_obj_half__ = 7
        return self.default[i] + self.half + __tmpl_obj_default__ + __tmpl_obj_half__


def test_template_expansion_cannot_merge_user_and_attribute_bindings(backend):
    @tack.kernel
    def collide(obj, __tmpl_obj_default__, out, __tmpl_obj_half__):
        for i in range(out.shape[0]):
            out[i] = (obj.default[i] + obj.half + obj.value(i)
                      + __tmpl_obj_default__[i] + __tmpl_obj_half__)

    data, out = _inputs()
    other = tack.field(tack.i32, (7,))
    other.fill(100)
    obj = _IdentifierModel(data)
    for half in [3, 9, 3]:
        obj.half = half
        collide(obj, other, out, 200)
        np.testing.assert_array_equal(out.to_numpy(),
                                      2 * (np.arange(7) - 3 + half) + 312)


def _packed(kernel, args):
    function, effective_args = _prepare_ir(kernel, args)
    pack_scalars(function, effective_args)
    verify_ir(function, 'packed')
    annotate_types(function)
    verify_ir(function, 'typed')
    return function


@pytest.mark.parametrize('generate', GENERATORS, ids=['cuda', 'hip', 'metal', 'opencl'])
def test_generation_preserves_ir_and_parameter_layout(generate):
    tack.init(arch=tack.cpu)
    data, out = _inputs()
    function = _packed(_helper_collisions, (data, out, 5))
    before = copy.deepcopy(function)
    source = generate(function)
    assert generate(function) == source
    assert kernel_entry_name(function.name) + '(' in source
    for param in function.params:
        assert gpu_variable_name(param.name) in source
    assert [(p.name, p.type_annotation, p._is_field) for p in function.params] == [
        (p.name, p.type_annotation, p._is_field) for p in before.params]
    assert function._scalar_pack_info == before._scalar_pack_info
    assert [(type(n), getattr(n, 'name', None), getattr(n, 'target', None),
             getattr(n, 'var', None)) for n in walk_ir(function)] == [
        (type(n), getattr(n, 'name', None), getattr(n, 'target', None),
         getattr(n, 'var', None)) for n in walk_ir(before)]
    verify_ir(function, 'typed')


@pytest.mark.parametrize('generate', GENERATORS, ids=['cuda', 'hip', 'metal', 'opencl'])
def test_texture_reference_slots_are_encoded(generate):
    tack.init(arch=tack.cpu)
    function = _packed(_texture_names, _texture_inputs())
    texture = function.params[0]
    texture._is_texture = True
    source = generate(function)
    encoded = gpu_variable_name('__samp__')
    assert encoded in source
    if generate is generate_msl_source:
        assert encoded + '.sample(__samp__,' in source
    elif generate is generate_opencl_source:
        assert 'read_imagef(' + encoded + ', __samp__,' in source
    else:
        assert 'tex3D<float>(' + encoded + ',' in source


def test_opencl_compiles_keyword_and_internal_collisions(named_kernel, tmp_path):
    clang = require_clang()
    tack.init(arch=tack.cpu)
    data, out = _inputs()
    for kernel in [named_kernel, _helper_collisions]:
        function = _packed(kernel, (data, out, 5))
        path = tmp_path / 'kernel.cl'
        path.write_text(generate_opencl_source(function))
        result = subprocess.run([clang, '-target', 'x86_64-unknown-linux-gnu',
                                 '-x', 'cl', '-cl-std=CL1.2', '-fsyntax-only', str(path)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('generate', [generate_cuda_source, generate_hip_source],
                         ids=['cuda-host', 'hip-host'])
def test_generated_cpp_compiles_and_executes_collisions(generate, named_kernel, tmp_path):
    """Host C++ execution validates naming, independently of GPU availability."""
    clang = require_clang('ubsan')
    tack.init(arch=tack.cpu)
    data, out = _inputs()
    function = _packed(named_kernel, (data, out, 2))
    source = generate(function)
    source = '\n'.join(line for line in source.splitlines()
                       if not line.startswith('#include'))
    entry = kernel_entry_name(function.name)
    source = '''
#include <cstdio>
#define __global__
#define __device__
struct Grid { unsigned int x; };
Grid blockIdx = {0}, blockDim = {1}, threadIdx = {0};
''' + source + f'''
int main() {{
    int data[] = {{-3, -2, -1, 0, 1, 2, 3}};
    int out[7] = {{0}}, scale[] = {{2}};
    for (long long i = 0; i < 7; ++i) {{
        blockIdx.x = i;
        {entry}(data, out, scale, 7);
    }}
    for (int value : out) std::printf("%d ", value);
    return 0;
}}
'''
    cpp, executable = tmp_path / 'kernel.cpp', tmp_path / 'kernel'
    cpp.write_text(source)
    result = subprocess.run([clang, '-std=c++14', '-O2',
                             '-fsanitize=undefined', '-fno-sanitize-recover=all',
                             str(cpp), '-o', str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.fromstring(result.stdout, dtype=np.int32, sep=' '),
                                  _expected(2))
