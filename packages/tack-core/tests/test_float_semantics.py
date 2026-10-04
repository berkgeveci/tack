"""Portable floating expressions use safe math without requiring // or %."""

import itertools
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.identifiers import kernel_entry_name
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang.inspect_kernel import _prepare_ir


def _field(values, dtype):
    values = np.asarray(values, dtype=dtype.numpy_dtype)
    result = tack.field(dtype, values.shape)
    result.from_numpy(values)
    return result


def _assert_float(actual, expected, *, maxulp=4, zero_sign=True):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    zeros = expected == 0
    np.testing.assert_array_equal(actual[zeros], expected[zeros])
    signed = np.isinf(expected) | zeros if zero_sign else np.isinf(expected)
    np.testing.assert_array_equal(np.signbit(actual[signed]), np.signbit(expected[signed]))
    finite = np.isfinite(expected) & ~zeros
    np.testing.assert_array_max_ulp(actual[finite], expected[finite], maxulp=maxulp)


@tack.kernel
def _arithmetic(a, b, out):
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        out[i, 0] = x + y
        out[i, 1] = x - y
        out[i, 2] = x * y
        out[i, 3] = x / y
        out[i, 4] = -x
        out[i, 5] = abs(x)
        out[i, 6] = floor(x)
        out[i, 7] = ceil(x)


def _check_arithmetic(dtype):
    scalar = dtype.numpy_dtype.type
    # Nonzero finite inputs/results are normal; denormal support is separate.
    values = [0.0, -0.0, 1.0, -1.0, 0.1, -0.1, 7.5, -7.5,
              np.inf, -np.inf, np.nan]
    pairs = list(itertools.product(values, repeat=2))
    limit = 100 if dtype is tack.f32 else 900
    pairs += [(2.0**limit, 2.0**limit), (-2.0**limit, 2.0**limit)]
    rng = np.random.default_rng(5707 + dtype.bits)
    pairs += list(rng.uniform(-16, 16, (512, 2)) * np.exp2(rng.integers(-20, 20, (512, 2))))
    a, b = np.asarray(pairs, dtype=dtype.numpy_dtype).T
    out = tack.field(dtype, (len(a), 8))
    _arithmetic(_field(a, dtype), _field(b, dtype), out)
    with np.errstate(all='ignore'):
        expected = np.column_stack([a + b, a - b, a * b, a / b, -a,
                                    np.abs(a), np.floor(a), np.ceil(a)])
    _assert_float(out.to_numpy(), expected)
    # This also checks cached dispatch with different exceptional inputs.
    field_a, field_b = _field(a[::-1].copy(), dtype), _field(b[::-1].copy(), dtype)
    _arithmetic(field_a, field_b, out)
    _assert_float(out.to_numpy(), expected[::-1])


def test_f32_arithmetic_and_exceptional_classes(backend):
    _check_arithmetic(tack.f32)


def test_f64_arithmetic_and_exceptional_classes(f64_backend):
    _check_arithmetic(tack.f64)


@tack.kernel
def _identities(a, out):
    for i in range(a.shape[0]):
        x = a[i]
        out[i, 0] = x * 0.0
        out[i, 1] = x - x
        out[i, 2] = x + 0.0
        out[i, 3] = 0.0 / x


def _check_identities(dtype):
    a = np.asarray([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan],
                   dtype=dtype.numpy_dtype)
    out = tack.field(dtype, (len(a), 4))
    _identities(_field(a, dtype), out)
    with np.errstate(all='ignore'):
        zero = dtype.numpy_dtype.type(0)
        expected = np.column_stack([a * zero, a - a, a + zero, zero / a])
    _assert_float(out.to_numpy(), expected)


def test_f32_unsafe_identity_folds_are_disabled(backend):
    _check_identities(tack.f32)


def test_f64_unsafe_identity_folds_are_disabled(f64_backend):
    _check_identities(tack.f64)


@tack.kernel
def _comparisons(a, b, out):
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        out[i, 0] = x == y
        out[i, 1] = x != y
        out[i, 2] = x < y
        out[i, 3] = x <= y
        out[i, 4] = x > y
        out[i, 5] = x >= y
        out[i, 6] = not x
        out[i, 7] = 1 if x else 0
        out[i, 8] = (x != x) and (y == y)


def _check_comparisons(dtype):
    values = [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan]
    a, b = np.asarray(list(itertools.product(values, repeat=2)), dtype=dtype.numpy_dtype).T
    out = tack.field(tack.i32, (len(a), 9))
    _comparisons(_field(a, dtype), _field(b, dtype), out)
    expected = np.column_stack([a == b, a != b, a < b, a <= b, a > b, a >= b,
                                ~a.astype(bool), a.astype(bool), (a != a) & (b == b)])
    np.testing.assert_array_equal(out.to_numpy(), expected)


def test_f32_comparisons_and_truth(backend):
    _check_comparisons(tack.f32)


def test_f64_comparisons_and_truth(f64_backend):
    _check_comparisons(tack.f64)


@tack.kernel
def _minmax(a, b, out):
    for i in range(a.shape[0]):
        out[i, 0] = min(a[i], b[i])
        out[i, 1] = max(a[i], b[i])


def _check_minmax(dtype):
    values = [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan]
    a, b = np.asarray(list(itertools.product(values, repeat=2)), dtype=dtype.numpy_dtype).T
    out = tack.field(dtype, (len(a), 2))
    _minmax(_field(a, dtype), _field(b, dtype), out)
    expected = np.column_stack([np.fmin(a, b), np.fmax(a, b)])
    # Either input zero sign is permitted for min/max ties only.
    _assert_float(out.to_numpy(), expected, maxulp=0, zero_sign=False)


def test_f32_minmax_prefer_number_to_nan(backend):
    _check_minmax(tack.f32)


def test_f64_minmax_prefer_number_to_nan(f64_backend):
    _check_minmax(tack.f64)


@tack.kernel
def _ordered(a, b, c, out):
    for i in range(a.shape[0]):
        out[i, 0] = (a[i] + b[i]) + c[i]
        out[i, 1] = a[i] + (b[i] + c[i])
        out[i, 2] = a[i] * b[i] + c[i]


def _check_order(dtype):
    scalar = dtype.numpy_dtype.type
    big = scalar(2.0**(25 if dtype is tack.f32 else 54))
    near = scalar(1 + 2.0**(-13 if dtype is tack.f32 else -27))
    a = np.asarray([big, near], dtype=dtype.numpy_dtype)
    b = np.asarray([-big, scalar(1 - (near - 1))], dtype=dtype.numpy_dtype)
    c = np.asarray([1, -1], dtype=dtype.numpy_dtype)
    out = tack.field(dtype, (2, 3))
    _ordered(_field(a, dtype), _field(b, dtype), _field(c, dtype), out)
    result = out.to_numpy()
    np.testing.assert_array_equal(result[:, 0], (a + b) + c)
    np.testing.assert_array_equal(result[:, 1], a + (b + c))
    # Adjacent multiply/add contraction is permitted, but not reassociation.
    separate = a * b + c
    fused = np.asarray([float(Fraction(float(x)) * Fraction(float(y)) + Fraction(float(z)))
                        for x, y, z in zip(a, b, c)], dtype=dtype.numpy_dtype)
    assert np.all((result[:, 2] == separate) | (result[:, 2] == fused))


def test_f32_parentheses_and_permitted_contraction(backend):
    _check_order(tack.f32)


def test_f64_parentheses_and_permitted_contraction(f64_backend):
    _check_order(tack.f64)


@tack.kernel
def _math(a, b, out):
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        out[i, 0] = sqrt(y)
        out[i, 1] = sin(x)
        out[i, 2] = cos(x)
        out[i, 3] = tan(x)
        out[i, 4] = asin(x)
        out[i, 5] = acos(x)
        out[i, 6] = atan(x)
        out[i, 7] = atan2(x, y)
        out[i, 8] = exp(x)
        out[i, 9] = exp2(x)
        out[i, 10] = log(y)
        out[i, 11] = log2(y)
        out[i, 12] = log10(y)
        out[i, 13] = pow(y, x)
        out[i, 14] = y ** x


def _check_math(dtype):
    a = np.asarray([-0.875, -0.5, -0.125, 0.125, 0.5, 0.875], dtype=dtype.numpy_dtype)
    b = np.asarray([0.25, 0.5, 0.75, 1.25, 2.0, 4.0], dtype=dtype.numpy_dtype)
    out = tack.field(dtype, (len(a), 15))
    _math(_field(a, dtype), _field(b, dtype), out)
    expected = np.column_stack([np.sqrt(b), np.sin(a), np.cos(a), np.tan(a),
                                np.arcsin(a), np.arccos(a), np.arctan(a),
                                np.arctan2(a, b), np.exp(a), np.exp2(a),
                                np.log(b), np.log2(b), np.log10(b),
                                np.power(b, a), np.power(b, a)])
    # A bounded-domain smoke test, not a global transcendental ULP contract.
    _assert_float(out.to_numpy(), expected, maxulp=8)


def test_f32_math_builtins_on_bounded_domains(backend):
    _check_math(tack.f32)


def test_f64_math_builtins_on_bounded_domains(f64_backend):
    _check_math(tack.f64)


def test_f32_libm_results_round_before_nested_arithmetic(backend):
    @tack.kernel
    def precision(a, out):
        for i in range(a.shape[0]):
            x = a[i]
            out[i, 0] = tan(x) - tack.f32(tan(x))
            out[i, 1] = asin(x) - tack.f32(asin(x))
            out[i, 2] = acos(x) - tack.f32(acos(x))
            out[i, 3] = atan(x) - tack.f32(atan(x))
            out[i, 4] = atan2(x, 2.0) - tack.f32(atan2(x, 2.0))

    a = _field([0.1, -0.1, 0.3, -0.3, 0.875, -0.875], tack.f32)
    out = tack.field(tack.f32, (6, 5))
    precision(a, out)
    np.testing.assert_array_equal(out.to_numpy(), np.zeros((6, 5), dtype=np.float32))


def test_integer_math_arguments_use_f32_on_f64_destination(f64_backend):
    @tack.kernel
    def integer_math(a, out):
        for i in range(a.shape[0]):
            out[i, 0] = sqrt(a[i])
            out[i, 1] = log(a[i])
            out[i, 2] = atan2(a[i], 3)
            out[i, 3] = sqrt(tack.f64(a[i]))
            out[i, 4] = log(tack.f64(a[i]))
            out[i, 5] = atan2(tack.f64(a[i]), tack.f64(3))

    values = np.asarray([2**24 + 1, 2**30 + 3, 2**50 + 1], dtype=np.int64)
    out = tack.field(tack.f64, (3, 6))
    integer_math(_field(values, tack.i64), out)
    single, double = values.astype(np.float32), values.astype(np.float64)
    expected = np.column_stack([np.sqrt(single), np.log(single),
                                np.arctan2(single, np.float32(3)), np.sqrt(double),
                                np.log(double), np.arctan2(double, np.float64(3))])
    result = out.to_numpy()
    # The first three expressions must be f32 before the f64 field store.
    np.testing.assert_array_equal(result[:, :3], result[:, :3].astype(np.float32).astype(np.float64))
    _assert_float(result[:, :3].astype(np.float32), expected[:, :3].astype(np.float32), maxulp=8)
    _assert_float(result[:, 3:], expected[:, 3:], maxulp=8)


@tack.kernel
def _mixed_math(a, b, out):
    for i in range(a.shape[0]):
        out[i, 0] = sqrt(a[i])
        out[i, 1] = atan2(a[i], b[i])
        out[i, 2] = min(a[i], b[i])
        out[i, 3] = max(a[i], b[i])


def _check_mixed_math(dtype):
    values = np.asarray([2**24 + 1, 2**30 + 3, 2**50 + 1], dtype=np.int64)
    a, b = _field(values, tack.i64), _field([3, 2**30, 2**50], dtype)
    out = tack.field(dtype, (3, 4))
    _mixed_math(a, b, out)
    promoted = values.astype(dtype.numpy_dtype)
    expected = np.column_stack([np.sqrt(values.astype(np.float32)),
                                np.arctan2(promoted, b.to_numpy()),
                                np.fmin(promoted, b.to_numpy()),
                                np.fmax(promoted, b.to_numpy())])
    _assert_float(out.to_numpy(), expected.astype(dtype.numpy_dtype), maxulp=8)


def test_f32_mixed_math_arguments(backend):
    _check_mixed_math(tack.f32)


def test_f64_mixed_math_arguments(f64_backend):
    _check_mixed_math(tack.f64)


def _check_sqrt(dtype):
    @tack.kernel
    def roots(a, out):
        for i in range(a.shape[0]):
            out[i] = sqrt(a[i])

    values = np.asarray([0.0, -0.0, 1.0, 4.0, -1.0, np.inf, -np.inf, np.nan],
                        dtype=dtype.numpy_dtype)
    out = tack.field(dtype, values.shape)
    roots(_field(values, dtype), out)
    with np.errstate(all='ignore'):
        expected = np.sqrt(values)
    _assert_float(out.to_numpy(), expected)


def test_f32_sqrt_exceptional_classes(backend):
    _check_sqrt(tack.f32)


def test_f64_sqrt_exceptional_classes(f64_backend):
    _check_sqrt(tack.f64)


@pytest.mark.parametrize('dtype', [tack.f32, tack.f64], ids=lambda t: t.name)
@pytest.mark.parametrize('generate', [generate_cuda_source, generate_hip_source,
                                     generate_msl_source, generate_opencl_source],
                         ids=['cuda', 'hip', 'metal', 'opencl'])
def test_math_generators_accept_integer_and_mixed_arguments(dtype, generate, tmp_path):
    if dtype is tack.f64 and generate is generate_msl_source:
        pytest.skip('MSL has no f64')
    tack.init(arch=tack.cpu)
    a, b = _field([2**30 + 3], tack.i64), _field([0.5], dtype)
    out = tack.field(dtype, (1, 4))
    function, _ = _prepare_ir(_mixed_math, (a, b, out))
    source = generate(function)
    if generate is generate_opencl_source:
        clang = shutil.which('clang')
        if clang is None:
            pytest.skip('Clang is required for OpenCL syntax validation')
        path = tmp_path / 'math.cl'
        path.write_text(source)
        result = subprocess.run([clang, '-target', 'x86_64-unknown-linux-gnu', '-x', 'cl',
                                 '-cl-std=CL1.2', '-fsyntax-only', str(path)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('dtype', [tack.f32, tack.f64], ids=lambda t: t.name)
@pytest.mark.parametrize('generate', [generate_cuda_source, generate_hip_source],
                         ids=['cuda-host', 'hip-host'])
def test_generated_cpp_executes_safe_identities(dtype, generate, tmp_path):
    clang = shutil.which('clang++')
    if clang is None:
        pytest.skip('Clang++ is required for generated C++ validation')
    tack.init(arch=tack.cpu)
    values = np.asarray([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan], dtype=dtype.numpy_dtype)
    a, out = _field(values, dtype), tack.field(dtype, (len(values), 4))
    function, _ = _prepare_ir(_identities, (a, out))
    source = '\n'.join(line for line in generate(function).splitlines()
                       if not line.startswith('#include'))
    ctype = 'float' if dtype is tack.f32 else 'double'
    source = '''
#include <cmath>
#include <cstdio>
#define __global__
#define __device__
struct Grid { unsigned int x; };
Grid blockIdx = {0}, blockDim = {1}, threadIdx = {0};
''' + source + f'''
int main(int argc, char** argv) {{
    {ctype} a[{len(values)}], out[{len(values) * 4}];
    FILE* input = std::fopen(argv[1], "rb");
    if (!input) return 2;
    if (std::fread(a, sizeof({ctype}), {len(values)}, input) != {len(values)}) return 3;
    std::fclose(input);
    for (unsigned int i = 0; i < {len(values)}; ++i) {{
        blockIdx.x = i;
        {kernel_entry_name(function.name)}(a, out, {len(values)});
    }}
    return std::fwrite(out, sizeof({ctype}), {len(values) * 4}, stdout) == {len(values) * 4} ? 0 : 4;
}}
'''
    cpp, executable, input_path = tmp_path / 'kernel.cpp', tmp_path / 'kernel', tmp_path / 'a.bin'
    cpp.write_text(source)
    result = subprocess.run([clang, '-std=c++14', '-O2', '-fsanitize=undefined',
                             '-fno-sanitize-recover=all', str(cpp), '-o', str(executable)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values.tofile(input_path)
    result = subprocess.run([str(executable), str(input_path)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    actual = np.frombuffer(result.stdout, dtype=dtype.numpy_dtype).reshape(-1, 4)
    with np.errstate(all='ignore'):
        zero = dtype.numpy_dtype.type(0)
        expected = np.column_stack([values * zero, values - values, values + zero, zero / values])
    _assert_float(actual, expected)
