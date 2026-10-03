"""True division and exact fixed-width power, with independent numeric oracles."""

import itertools
import random
import shutil
import subprocess

import numpy as np
import pytest

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang import ir
from tack.lang.inspect_kernel import _prepare_ir
from tack.lang.ir_traversal import walk_ir

TYPES = [tack.i8, tack.u8, tack.i16, tack.u16,
         tack.i32, tack.u32, tack.i64, tack.u64]
GENERATORS = [generate_cuda_source, generate_hip_source,
              generate_msl_source, generate_opencl_source]


def _field(values, dtype):
    values = np.asarray(values, dtype=dtype.numpy_dtype)
    result = tack.field(dtype, values.shape)
    result.from_numpy(values)
    return result


def _wrap(value, dtype):
    value %= 1 << dtype.bits
    if dtype.name.startswith('i') and value >= 1 << (dtype.bits - 1):
        value -= 1 << dtype.bits
    return value


def _values(dtype):
    info = np.iinfo(dtype.numpy_dtype)
    values = {0, 1, 2, 3, int(info.min), int(info.max), int(info.max) - 1}
    if info.min < 0:
        values |= {-1, -2, -3, int(info.min) + 1}
    if dtype.bits >= 32:
        values |= {2**24 - 1, 2**24 + 1}
    if dtype.bits == 64:
        values |= {2**53 + 1, 2**62 + 1, 2**62 + 2**38 + 1}
    rng = random.Random(5800 + dtype.bits)
    return sorted(values) + [rng.randint(int(info.min), int(info.max)) for _ in range(24)]


@tack.kernel
def _power(a, b, out):
    for i in range(a.shape[0]):
        x, e = a[i], b[i]
        out[i, 0] = x ** e
        out[i, 1] = pow(x, e)
        out[i, 2] = (x ** e) + x
        out[i, 3] = (x ** e) ** e
        out[i, 4] = x ** 0
        out[i, 5] = pow(x, 1)
        out[i, 6] = (x ** e) < x


@pytest.mark.parametrize('base_type,exponent_type', itertools.product(TYPES, repeat=2),
                         ids=lambda t: t.name)
def test_integer_power_is_exact_and_preserves_base_width(backend, base_type, exponent_type):
    limit = int(np.iinfo(exponent_type.numpy_dtype).max)
    exponents = sorted({e for e in [0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65,
                                   127, 255, 256, 2**32 + 1, 2**63, limit - 1, limit]
                        if 0 <= e <= limit})
    if (base_type, exponent_type) in ((tack.i8, tack.i8), (tack.u8, tack.u8)):
        info = np.iinfo(base_type.numpy_dtype)
        pairs = list(itertools.product(range(int(info.min), int(info.max) + 1),
                                       range(limit + 1)))
    else:
        pairs = list(itertools.product(_values(base_type), exponents))
    a = _field([x for x, _ in pairs], base_type)
    b = _field([e for _, e in pairs], exponent_type)
    out = tack.field(base_type, (len(pairs), 7))
    _power(a, b, out)
    modulus = 1 << base_type.bits
    expected = []
    for x, e in pairs:
        powered = _wrap(pow(x, e, modulus), base_type)
        expected.append([powered, powered, _wrap(powered + x, base_type),
                         _wrap(pow(powered, e, modulus), base_type), 1, x, int(powered < x)])
    np.testing.assert_array_equal(out.to_numpy(),
                                  np.asarray(expected, dtype=base_type.numpy_dtype))


def _rounded_f32_integer(value):
    """Round a Python integer to binary32 without a binary64 intermediate.

    Includes midpoint neighbours that expose double rounding in a naive
    float(value) -> float32 reference for 64-bit inputs.
    """
    magnitude = abs(value)
    shift = max(0, magnitude.bit_length() - 24)
    if shift:
        high, low = divmod(magnitude, 1 << shift)
        half = 1 << (shift - 1)
        if low > half or (low == half and high & 1):
            high += 1
        magnitude = high << shift
    return np.float32(-magnitude if value < 0 else magnitude)


@tack.kernel
def _true_division(a, b, out):
    for i in range(a.shape[0]):
        q = a[i] / b[i]
        out[i, 0] = q
        out[i, 1] = q + q
        out[i, 2] = abs(q)


DIVISION_TYPES = [(t, t) for t in TYPES] + [
    (tack.i8, tack.u8), (tack.u16, tack.i16),
    (tack.i32, tack.u32), (tack.u32, tack.i64),
    (tack.i64, tack.u64), (tack.u64, tack.i32),
]


@pytest.mark.parametrize('left,right', DIVISION_TYPES, ids=lambda t: t.name)
def test_integer_true_division_returns_f32(backend, left, right):
    pairs = [(x, y) for x in _values(left) for y in _values(right) if y]
    a, b = _field([x for x, _ in pairs], left), _field([y for _, y in pairs], right)
    out = tack.field(tack.f32, (len(pairs), 3))
    _true_division(a, b, out)
    expected = []
    for x, y in pairs:
        q = np.float32(float(_rounded_f32_integer(x)) / float(_rounded_f32_integer(y)))
        expected.append([q, np.float32(q + q), abs(q)])
    # Vendor default floating-point options may approximate division; the
    # wider floating-point accuracy/optimization policy remains separate.
    np.testing.assert_array_max_ulp(out.to_numpy(), np.asarray(expected, dtype=np.float32),
                                   maxulp=4)


@tack.kernel
def _division_precision(a, b, narrow, wide):
    for i in range(a.shape[0]):
        narrow[i] = a[i] / b[i]
        wide[i] = tack.f64(a[i]) / tack.f64(b[i])


def test_division_precision_is_explicit(f64_backend):
    a = _field([7, -7, 2**53 + 1, 2**62 + 2**38 + 1], tack.i64)
    b = _field([2, 2, 3, 7], tack.i64)
    default, wide = tack.field(tack.f64, a.shape), tack.field(tack.f64, a.shape)
    _division_precision(a, b, default, wide)
    values = a.to_numpy().tolist()
    divisors = b.to_numpy().tolist()
    expected_default = [np.float32(float(_rounded_f32_integer(x)) /
                                   float(_rounded_f32_integer(y)))
                        for x, y in zip(values, divisors)]
    actual_default = default.to_numpy()
    np.testing.assert_array_equal(actual_default, actual_default.astype(np.float32))
    np.testing.assert_array_max_ulp(actual_default.astype(np.float32),
                                   np.asarray(expected_default, dtype=np.float32), maxulp=4)
    np.testing.assert_allclose(wide.to_numpy(), [float(x) / float(y)
                                                for x, y in zip(values, divisors)], rtol=1e-14)


@tack.kernel
def _fractional(a, b, out, integer):
    for i in range(a.shape[0]):
        out[i, 0] = a[i] / b[i]
        out[i, 1] = (a[i] / b[i]) * b[i]
        integer[i] = a[i] / b[i]


def test_fractional_result_survives_nested_expression_and_truncates_on_store(backend):
    a, b = _field([7, -7, 1, -1], tack.i32), _field([2, 2, 2, 2], tack.i32)
    out, integer = tack.field(tack.f32, (4, 2)), tack.field(tack.i32, (4,))
    _fractional(a, b, out, integer)
    np.testing.assert_array_equal(out.to_numpy(), [[3.5, 7], [-3.5, -7], [0.5, 1], [-0.5, -1]])
    np.testing.assert_array_equal(integer.to_numpy(), [3, -3, 0, 0])


@tack.kernel
def _guarded(a, e, b, power, quotient):
    for i in range(a.shape[0]):
        power[i] = a[i] ** e[i] if e[i] >= 0 else 99
        quotient[i] = a[i] / b[i] if b[i] != 0 else 99


def test_guarded_invalid_operations_are_not_executed(backend):
    a, e, b = [_field(v, tack.i32) for v in ([3, 0, -2, 7], [4, 0, -1, -9], [2, 1, 0, 0])]
    power, quotient = tack.field(tack.i32, (4,)), tack.field(tack.f32, (4,))
    _guarded(a, e, b, power, quotient)
    np.testing.assert_array_equal(power.to_numpy(), [81, 1, 99, 99])
    np.testing.assert_array_equal(quotient.to_numpy(), [1.5, 0, 99, 99])


@tack.func
def _division_power_touch(state, i, value):
    state[i] = state[i] * 10 + value
    return value


@tack.kernel
def _effects(state, powers, quotients):
    for i in range(state.shape[0]):
        powers[i, 0] = _division_power_touch(state, i, 2) ** _division_power_touch(state, i, 3)
        powers[i, 1] = pow(_division_power_touch(state, i, 2), _division_power_touch(state, i, 3))
        quotients[i] = _division_power_touch(state, i, 7) / _division_power_touch(state, i, 2)


def test_operands_execute_once_in_order(backend):
    state = _field([0] * 7, tack.i32)
    powers, quotients = tack.field(tack.i32, (7, 2)), tack.field(tack.f32, (7,))
    _effects(state, powers, quotients)
    np.testing.assert_array_equal(state.to_numpy(), [232372] * 7)
    np.testing.assert_array_equal(powers.to_numpy(), [[8, 8]] * 7)
    np.testing.assert_array_equal(quotients.to_numpy(), [3.5] * 7)


def test_opencl_inlined_effects_preserve_field_address_space(tmp_path):
    clang = shutil.which('clang')
    if clang is None:
        pytest.skip('Clang is required for OpenCL C syntax validation')
    tack.init(arch=tack.cpu)
    state = _field([0], tack.i32)
    powers, quotients = tack.field(tack.i32, (1, 2)), tack.field(tack.f32, (1,))
    func, _ = _prepare_ir(_effects, (state, powers, quotients))
    path = tmp_path / 'effects.cl'
    path.write_text(generate_opencl_source(func))
    result = subprocess.run([clang, '-x', 'cl', '-cl-std=CL1.2', '-fsyntax-only', str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@tack.kernel
def _floating_power(a, b, out):
    for i in range(a.shape[0]):
        out[i, 0] = a[i] ** b[i]
        out[i, 1] = pow(a[i], b[i])


def test_explicit_float_base_supports_negative_exponents(backend):
    a, b = _field([2, 4, 0.5], tack.f32), _field([-3, -2, -1], tack.i64)
    out = tack.field(tack.f32, (3, 2))
    _floating_power(a, b, out)
    np.testing.assert_allclose(out.to_numpy(), [[0.125] * 2, [0.0625] * 2, [2] * 2],
                               rtol=1e-6)


@pytest.mark.parametrize('wide_base', [True, False])
def test_f64_power_preserves_argument_precision(f64_backend, wide_base):
    if wide_base:
        a, b = _field([1.000000000001, 2], tack.f64), _field([3, -3], tack.i32)
    else:
        a, b = _field([2, 3], tack.i32), _field([1.000000000001, -0.5], tack.f64)
    out = tack.field(tack.f64, (2, 2))
    _floating_power(a, b, out)
    expected = [[float(x) ** float(y)] * 2 for x, y in zip(a.to_numpy(), b.to_numpy())]
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-14)


@tack.kernel
def _negative_literal(a, out):
    for i in range(a.shape[0]):
        out[i] = a[i] ** -1


@tack.kernel
def _negative_builtin(a, out):
    for i in range(a.shape[0]):
        out[i] = pow(a[i], -1)


@pytest.mark.parametrize('kernel', [_negative_literal, _negative_builtin])
def test_negative_integer_literal_exponent_is_rejected(kernel):
    tack.init(arch=tack.cpu)
    a, out = _field([2], tack.i32), tack.field(tack.i32, (1,))
    with pytest.raises(TypeError, match='nonnegative exponent.*cast the base'):
        tack.inspect(kernel, a, out)


@pytest.mark.parametrize('generate', GENERATORS)
@pytest.mark.parametrize('dtype', TYPES, ids=lambda t: t.name)
def test_all_generators_use_integer_power_helpers(generate, dtype):
    tack.init(arch=tack.cpu)
    a, e = _field([2], dtype), _field([2**63], tack.u64)
    out = tack.field(dtype, (1, 7))
    func, _ = _prepare_ir(_power, (a, e, out))
    source = generate(func)
    assert 'powf(' not in source and 'pow(' not in source
    definitions = [line for line in source.splitlines()
                   if f' __tack_pow_{dtype.name}__(' in line and line.endswith(') {')]
    assert len(definitions) == 1
    assert 'while (b != 0)' in source
    assert 'b >>= 1;' in source
    # The exponent is not implicitly narrowed to the base's integer type.
    assert 'unsigned long' in source or 'ulong' in source


@pytest.mark.parametrize('generate', GENERATORS)
@pytest.mark.parametrize('dtype', [tack.f32, tack.f64], ids=lambda t: t.name)
def test_float_power_generators_use_promoted_precision(generate, dtype):
    if generate is generate_msl_source and dtype is tack.f64:
        pytest.skip('Metal does not support f64')
    tack.init(arch=tack.cpu)
    a, e = _field([2], tack.i32), _field([1.5], dtype)
    out = tack.field(dtype, (1, 2))
    func, _ = _prepare_ir(_floating_power, (a, e, out))
    source = generate(func)
    name = 'powf' if dtype is tack.f32 and generate in (
        generate_cuda_source, generate_hip_source) else 'pow'
    precision = 'double' if dtype is tack.f64 else 'float'
    assert source.count(f'{name}(({precision})(') == 2
    assert '__tack_pow_' not in source


@pytest.mark.parametrize('base_type,exponent_type', itertools.product(TYPES, repeat=2),
                         ids=lambda t: t.name)
def test_power_annotation_preserves_base_type(base_type, exponent_type):
    tack.init(arch=tack.cpu)
    a, e = _field([2], base_type), _field([3], exponent_type)
    out = tack.field(base_type, (1, 7))
    func, _ = _prepare_ir(_power, (a, e, out))
    expressions = [n for n in walk_ir(func.body)
                   if (isinstance(n, ir.IRBinOp) and n.op == '**')
                   or (isinstance(n, ir.IRCall) and n.func_name == 'pow')]
    assert expressions
    assert all(n.dtype is base_type for n in expressions)


def test_integer_division_annotation_does_not_depend_on_output_type():
    tack.init(arch=tack.cpu)
    a, b = _field([7], tack.i64), _field([2], tack.u64)
    for output_type in (tack.f32, tack.f64):
        out = tack.field(output_type, (1, 3))
        func, _ = _prepare_ir(_true_division, (a, b, out))
        divisions = [n for n in walk_ir(func.body) if isinstance(n, ir.IRBinOp) and n.op == '/']
        assert len(divisions) == 1
        assert divisions[0].dtype is tack.f32
