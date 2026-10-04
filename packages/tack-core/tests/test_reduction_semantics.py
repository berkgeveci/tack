"""Reduction classes, precision, order variation and exceptional extrema."""

import math
import subprocess

import numpy as np
import pytest
from compiler_tools import require_clang

import tack
from tack.codegen.reductions import f32_reduction_helpers, field_reduction_source
from tack.lang.field import Field, NumpyBuffer


def _field(values, dtype=tack.f32):
    values = np.asarray(values, dtype=dtype.numpy_dtype)
    result = tack.field(dtype, values.shape)
    result.from_numpy(values)
    return result


def _assert_value(actual, expected):
    assert isinstance(actual, float)
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected
        if expected == 0 or np.isinf(expected):
            assert np.signbit(actual) == np.signbit(expected)


@pytest.mark.parametrize('values', [
    [2e38], [-2e38], [np.inf], [-np.inf], [2e38, 3e38], [-3e38, -2e38],
    [np.inf, -np.inf, 1], [np.nan], [1, np.nan], [np.nan, 1],
    [0.0, -0.0], [-0.0, 0.0], [-0.0], [0.0], [-1, -0.0], [1, -0.0],
])
def test_f32_extrema_classes_and_zero_signs(backend, values):
    data = np.asarray(values, np.float32)
    field = _field(data)
    for op in ('min', 'max'):
        expected = float(getattr(data, op)())
        if expected == 0:
            signs = np.signbit(data[data == 0])
            expected = -0.0 if (signs.any() if op == 'min' else signs.all()) else 0.0
        _assert_value(getattr(field, op)(), expected)


@pytest.mark.parametrize('position', [0, 127, 255, 256, 511, 512, 768])
def test_nan_propagates_across_groups_and_tail(backend, position):
    data = np.linspace(-10, 10, 769, dtype=np.float32)
    field = _field(data)
    assert field.min() == -10 and field.max() == 10
    data[position] = np.nan
    field.from_numpy(data)
    for op in ('sum', 'min', 'max', 'mean'):
        for _ in range(3):
            assert np.isnan(getattr(field, op)())


@pytest.mark.parametrize('position', [0, 256, 512, 768])
def test_zero_ties_across_groups_and_tail(backend, position):
    for base, alternate in ((0.0, -0.0), (-0.0, 0.0)):
        values = np.full(769, base, np.float32)
        values[position] = alternate
        field = _field(values)
        for _ in range(3):
            _assert_value(field.min(), -0.0)
            _assert_value(field.max(), 0.0)


@pytest.mark.parametrize('n', [1, 255, 256, 257, 513, 4099])
def test_exact_sums_and_partial_groups(backend, n):
    field = _field(np.full(n, 0.25, dtype=np.float32))
    assert field.sum() == n / 4
    assert field.mean() == 0.25
    assert field.min() == field.max() == 0.25


def _check_accuracy(dtype):
    rng = np.random.default_rng(5801)
    values = (rng.uniform(-1, 1, 4099) * np.exp2(rng.integers(-8, 9, 4099))).astype(dtype.numpy_dtype)
    field = _field(values, dtype)
    exact = math.fsum(map(float, values))
    absolute_sum = math.fsum(abs(float(x)) for x in values)
    unit_roundoff = 2.0 ** (-24 if dtype is tack.f32 else -53)
    nu = len(values) * unit_roundoff
    bound = nu / (1 - nu) * absolute_sum
    # Test the supported error budget on every run, rather than assuming
    # equal low bits or requiring scheduling to produce a different result.
    for _ in range(5):
        assert abs(field.sum() - exact) <= bound
        assert abs(field.mean() - exact / len(values)) <= bound / len(values) + math.ulp(exact / len(values))


def test_f32_sum_accuracy_budget(backend):
    _check_accuracy(tack.f32)


def test_f64_host_sum_accuracy_budget(f64_backend):
    _check_accuracy(tack.f64)


def test_cancellation_accepts_valid_groupings(backend):
    field = _field([-2**24, 1, 2**24])
    assert field.sum() in (0.0, 1.0)


@pytest.mark.parametrize('values, expected', [
    ([np.inf, 1], np.inf), ([-np.inf, 1], -np.inf),
    ([np.inf, -np.inf], np.nan), ([np.nan, 1], np.nan),
    ([3e38, 3e38], np.inf), ([-3e38, -3e38], -np.inf),
])
def test_f32_sum_nonfinite_classes(backend, values, expected):
    _assert_value(_field(values).sum(), expected)


@pytest.mark.parametrize('dtype', [tack.i8, tack.u8, tack.i16, tack.u16,
                                   tack.i32, tack.u32, tack.i64, tack.u64])
def test_integer_host_reduction_promotion_and_wrap(backend, dtype):
    limits = np.iinfo(dtype.numpy_dtype)
    values = [limits.max, limits.max, 1]
    field = _field(values, dtype)
    total = sum(values) % 2**64
    if limits.min < 0 and total >= 2**63:
        total -= 2**64
    assert field.sum() == float(total)
    assert field.min() == 1.0
    assert field.max() == float(limits.max)
    assert field.mean() == float(total) / len(values)


@pytest.mark.parametrize('values', [[np.nan, 1], [-0.0, 0.0], [-0.0], [0.0]])
def test_f64_host_extrema_rules(f64_backend, values):
    field = _field(values, tack.f64)
    if np.isnan(values).any():
        assert np.isnan(field.min()) and np.isnan(field.max())
    else:
        signs = np.signbit(values)
        _assert_value(field.min(), -0.0 if signs.any() else 0.0)
        _assert_value(field.max(), -0.0 if signs.all() else 0.0)


def test_empty_reductions_skip_backend_and_storage(monkeypatch):
    import tack.runtime.dispatch as dispatch

    def unexpected_backend():
        raise AssertionError('empty reduction must not launch or read storage')

    monkeypatch.setattr(dispatch, 'get_backend', unexpected_backend)
    # Logical empty field: no allocation is needed to establish the result.
    field = Field(tack.f32, (0,), None)
    _assert_value(field.sum(), 0.0)
    assert np.isnan(field.mean())
    for op in ('min', 'max'):
        with pytest.raises(ValueError, match='nonempty field'):
            getattr(field, op)()


def test_empty_cpu_field_and_multidimensional_host_reductions():
    empty = Field(tack.f32, (0, 3), NumpyBuffer(np.float32, (0, 3)))
    assert empty.sum() == 0 and np.isnan(empty.mean())
    # Shared NumPy fallback also serves non-f32 fields on every GPU.
    from tack.runtime.reductions import reduce_numpy
    values = np.array([[0.0, 1.0], [-0.0, 2.0]])
    _assert_value(reduce_numpy(values, 'min'), -0.0)
    assert reduce_numpy(values, 'max') == 2.0


@pytest.mark.parametrize('group_x, total', [(128, 256), (256, 128)])
def test_level_zero_small_workgroup_uses_host_fallback(group_x, total):
    from types import SimpleNamespace

    from tack.runtime.level_zero_backend import LevelZeroBackend
    be = LevelZeroBackend.__new__(LevelZeroBackend)
    be._compute_props = SimpleNamespace(maxGroupSizeX=group_x, maxTotalGroupSize=total)
    values = np.array([2e38, 3e38], np.float32)
    buffer = NumpyBuffer(np.float32, values.shape)
    buffer.from_numpy(values)
    field = Field(tack.f32, values.shape, buffer)
    assert be.reduce_field(field, 'min') == float(values[0])
    assert be.reduce_field(field, 'max') == float(values[1])


@tack.kernel
def _block_extrema(data, out):
    for i in range(data.shape[0]):
        low = tack.block_min(data[i])
        high = tack.block_max(data[i])
        if tack.thread_id() == 0:
            out[i // 256, 0] = low
            out[i // 256, 1] = high


@pytest.mark.parametrize('values', [[np.nan], [np.inf], [-np.inf],
                                   [-0.0], [0.0, -0.0], [2e38, 3e38]])
def test_full_workgroup_f32_extrema(workgroup_backend, values):
    data = np.resize(np.asarray(values, np.float32), 512)
    out = tack.field(tack.f32, (2, 2))
    _block_extrema(_field(data), out)
    actual = out.to_numpy()
    for group in range(2):
        chunk = data[group * 256:(group + 1) * 256]
        for column, op in enumerate(('min', 'max')):
            expected = float(getattr(chunk, op)())
            if expected == 0:
                signs = np.signbit(chunk[chunk == 0])
                expected = -0.0 if (signs.any() if op == 'min' else signs.all()) else 0.0
            _assert_value(float(actual[group, column]), expected)


@tack.kernel
def _block_sum(data, out):
    for i in range(data.shape[0]):
        value = tack.block_sum(data[i])
        if tack.thread_id() == 0:
            out[i // 256] = value


def test_block_integer_input_requires_explicit_f32(backend):
    from tack.runtime.dispatch import get_backend
    message = ('requires f32 input' if get_backend().supports_workgroups
               else 'CPU backend does not support workgroup execution')
    with pytest.raises((TypeError, RuntimeError), match=message):
        _block_sum(_field(np.ones(256), tack.i32), tack.field(tack.f32, (1,)))


def test_block_f64_input_requires_explicit_f32(f64_backend):
    from tack.runtime.dispatch import get_backend
    message = ('requires f32 input' if get_backend().supports_workgroups
               else 'CPU backend does not support workgroup execution')
    with pytest.raises((TypeError, RuntimeError), match=message):
        _block_sum(_field(np.ones(256), tack.f64), tack.field(tack.f32, (1,)))


def test_full_workgroup_sum_accuracy(workgroup_backend):
    rng = np.random.default_rng(5812)
    values = rng.uniform(-64, 64, 512).astype(np.float32)
    out = tack.field(tack.f32, (2,))
    _block_sum(_field(values), out)
    for group, actual in enumerate(out.to_numpy()):
        chunk = values[group * 256:(group + 1) * 256]
        exact = math.fsum(map(float, chunk))
        nu = len(chunk) * 2**-24
        bound = nu / (1 - nu) * math.fsum(abs(float(x)) for x in chunk)
        assert abs(float(actual) - exact) <= bound


def test_block_explicit_f32_cast(workgroup_backend):
    @tack.kernel
    def cast_sum(data, out):
        for i in range(data.shape[0]):
            value = tack.block_sum(tack.f32(data[i]))
            if tack.thread_id() == 0:
                out[0] = value

    out = tack.field(tack.f32, (1,))
    cast_sum(_field(np.ones(256), tack.i32), out)
    assert out.to_numpy()[0] == 256


@pytest.mark.parametrize('dialect', ['cuda', 'hip'])
@pytest.mark.parametrize('op', ['sum', 'min', 'max'])
def test_native_reduction_cpp_syntax(dialect, op, tmp_path):
    clang = require_clang('cxx')
    # Validate full kernel C++ with declared GPU builtins, without pretending
    # this checks the vendor compiler, GPU barriers or atomic scheduling.
    preamble = '''#include <stdint.h>
#define __device__
#define __global__
#define __shared__
struct Dim { unsigned int x; };
extern Dim threadIdx, blockIdx, blockDim;
unsigned int __float_as_uint(float);
float __uint_as_float(unsigned int);
void __syncthreads();
float atomicAdd(float*, float);
unsigned int atomicCAS(unsigned int*, unsigned int, unsigned int);
'''
    source = tmp_path / 'reduce.cpp'
    body = field_reduction_source(dialect, op).replace('#include <hip/hip_runtime.h>\n', '')
    source.write_text(preamble + body)
    result = subprocess.run([clang, '-std=c++17', '-fsyntax-only', str(source)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('op', ['sum', 'min', 'max'])
def test_opencl_native_reduction_syntax(op, tmp_path):
    clang = require_clang()
    source = tmp_path / 'reduce.cl'
    source.write_text(field_reduction_source('opencl', op))
    result = subprocess.run([clang, '-x', 'cl', '-cl-std=CL2.0', '-fsyntax-only', str(source)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('dialect', ['cuda', 'hip', 'metal', 'opencl'])
def test_extrema_helpers_host_sanitized(dialect, tmp_path):
    clang = require_clang('ubsan')
    values = np.array([0.0, -0.0, 1, -1, 2e38, -2e38, np.inf, -np.inf, np.nan], np.float32)
    checks = []
    for a in values:
        for b in values:
            for op in ('min', 'max'):
                if np.isnan(a) or np.isnan(b):
                    expected = np.float32(np.nan)
                elif a == b == 0:
                    negative = (np.signbit(a) or np.signbit(b)) if op == 'min' else (
                        np.signbit(a) and np.signbit(b))
                    expected = np.float32(-0.0 if negative else 0.0)
                else:
                    expected = np.float32(min(a, b) if op == 'min' else max(a, b))
                checks.append(f'{{float v = tack_reduce_{op}_f32(as_float({a.view(np.uint32)}u), '
                              f'as_float({b.view(np.uint32)}u)); '
                              + ('if (v == v) return 1;' if np.isnan(expected) else
                                 f'if (as_uint(v) != {expected.view(np.uint32)}u) return 2;') + '}')
    preamble = '''#include <stdint.h>
#include <string.h>
#define __device__
using uint = unsigned int;
template<class T, class U> T as_type(U x) { T y; memcpy(&y, &x, sizeof(y)); return y; }
inline float as_float(uint x) { return as_type<float>(x); }
inline uint as_uint(float x) { return as_type<uint>(x); }
inline float __uint_as_float(uint x) { return as_float(x); }
inline uint __float_as_uint(float x) { return as_uint(x); }
'''
    source, binary = tmp_path / 'check.cpp', tmp_path / 'check'
    source.write_text(preamble + '\n'.join(f32_reduction_helpers(dialect))
                      + '\nint main() {\n' + '\n'.join(checks) + '\n}\n')
    result = subprocess.run([clang, '-std=c++17', '-O2', '-fsanitize=undefined',
                             '-fno-sanitize-recover=all', str(source), '-o', str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
