"""Fixed-width integer conformance, with Python big integers as the oracle."""

import copy
import itertools
import random

import numpy as np
import pytest

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang import ir
from tack.lang.ir_resolve import resolve_ir
from tack.lang.ir_type_annotate import annotate_types
from tack.lang.type_inference import infer_param_types, promote_types

TYPES = [tack.i8, tack.u8, tack.i16, tack.u16,
         tack.i32, tack.u32, tack.i64, tack.u64]
GENERATORS = [generate_cuda_source, generate_hip_source,
              generate_msl_source, generate_opencl_source]


def _wrap(value, dtype):
    value %= 1 << dtype.bits
    if dtype.name.startswith('i') and value >= 1 << (dtype.bits - 1):
        value -= 1 << dtype.bits
    return value


def _values(dtype):
    info = np.iinfo(dtype.numpy_dtype)
    lo, hi = int(info.min), int(info.max)
    boundaries = {lo, lo + 1, hi, hi - 1, 0, 1, hi // 2, hi // 2 + 1}
    if lo < 0:
        boundaries |= {-1, -2, -3}
    rng = random.Random(5700 + dtype.bits)
    return sorted(boundaries) + [rng.randint(lo, hi) for _ in range(64)]


def _field(values, dtype):
    data = np.asarray(values, dtype=dtype.numpy_dtype)
    field = tack.field(dtype, data.shape)
    field.from_numpy(data)
    return field


@tack.kernel
def _arithmetic(a, b, out):
    for i in range(a.shape[0]):
        x = a[i]
        y = b[i]
        out[i, 0] = x + y
        out[i, 1] = x - y
        out[i, 2] = x * y
        out[i, 3] = -x
        out[i, 4] = ~x
        out[i, 5] = (x * y) // y if y else x
        out[i, 6] = (x + y) >> 1
        out[i, 7] = abs(x)
        out[i, 8] = min(x, y)
        out[i, 9] = max(x, y)
        out[i, 10] = x < y
        out[i, 11] = x & y
        out[i, 12] = x | y
        out[i, 13] = x ^ y
        out[i, 14] = (x * y) + (x - y)
        out[i, 15] = x << 1


@pytest.mark.parametrize('dtype', TYPES, ids=lambda t: t.name)
def test_wrapping_arithmetic_and_unsigned_order(backend, dtype):
    values = _values(dtype)
    pairs = list(itertools.product(values[:11], repeat=2)) + list(zip(values, values[::-1]))
    lo = int(np.iinfo(dtype.numpy_dtype).min)
    # Signed min / -1 remains outside the division domain, even after wrapping.
    pairs = [(x, y) for x, y in pairs if not (lo < 0 and _wrap(x * y, dtype) == lo and y == -1)]
    a, b = _field([x for x, _ in pairs], dtype), _field([y for _, y in pairs], dtype)
    out = tack.field(dtype, (len(pairs), 16))
    _arithmetic(a, b, out)
    expected = []
    for x, y in pairs:
        summed, product = _wrap(x + y, dtype), _wrap(x * y, dtype)
        row = [x + y, x - y, x * y, -x, ~x,
               product // y if y else x, summed >> 1, abs(x), min(x, y), max(x, y),
               int(x < y), x & y, x | y, x ^ y,
               product + _wrap(x - y, dtype), x << 1]
        expected.append([_wrap(v, dtype) for v in row])
    np.testing.assert_array_equal(out.to_numpy(), np.asarray(expected, dtype=dtype.numpy_dtype))


@tack.kernel
def _negate_widened(a, out):
    for i in range(a.shape[0]):
        x = a[i]
        out[i, 0] = -x
        out[i, 1] = abs(x)
        n = -x
        out[i, 2] = tack.i64(n)


@pytest.mark.parametrize('dtype', [tack.i8, tack.i16, tack.i32, tack.i64], ids=lambda t: t.name)
def test_wrapped_negation_and_abs_survive_widening(backend, dtype):
    # The minimum negates to itself, and must still be the minimum once
    # sign-extended. Intel's IGC 2.7.11 widened i16 -(-32768) as 32768,
    # and abs of the i8/i16/i32 minimum as its magnitude.
    values = _values(dtype)
    out = tack.field(tack.i64, (len(values), 3))
    _negate_widened(_field(values, dtype), out)
    expected = [[_wrap(-x, dtype), _wrap(abs(x), dtype), _wrap(-x, dtype)] for x in values]
    np.testing.assert_array_equal(out.to_numpy(), np.asarray(expected, dtype=np.int64))


@tack.kernel
def _convert(a, oi8, ou8, oi16, ou16, oi32, ou32, oi64, ou64):
    for i in range(a.shape[0]):
        x = a[i]
        oi8[i, 0] = tack.i8(x)
        oi8[i, 1] = x
        vi8 = tack.i8(x)
        oi8[i, 2] = vi8
        ou8[i, 0] = tack.u8(x)
        ou8[i, 1] = x
        vu8 = tack.u8(x)
        ou8[i, 2] = vu8
        oi16[i, 0] = tack.i16(x)
        oi16[i, 1] = x
        vi16 = tack.i16(x)
        oi16[i, 2] = vi16
        ou16[i, 0] = tack.u16(x)
        ou16[i, 1] = x
        vu16 = tack.u16(x)
        ou16[i, 2] = vu16
        oi32[i, 0] = tack.i32(x)
        oi32[i, 1] = x
        vi32 = tack.i32(x)
        oi32[i, 2] = vi32
        ou32[i, 0] = tack.u32(x)
        ou32[i, 1] = x
        vu32 = tack.u32(x)
        ou32[i, 2] = vu32
        oi64[i, 0] = tack.i64(x)
        oi64[i, 1] = x
        vi64 = tack.i64(x)
        oi64[i, 2] = vi64
        ou64[i, 0] = tack.u64(x)
        ou64[i, 1] = x
        vu64 = tack.u64(x)
        ou64[i, 2] = vu64


@pytest.mark.parametrize('source', TYPES, ids=lambda t: t.name)
def test_all_integer_conversions(backend, source):
    values = _values(source)
    a = _field(values, source)
    outputs = [tack.field(t, (len(values), 3)) for t in TYPES]
    _convert(a, *outputs)
    for target, out in zip(TYPES, outputs):
        expected = [_wrap(x, target) for x in values]
        np.testing.assert_array_equal(out.to_numpy(),
                                      np.asarray([[x] * 3 for x in expected],
                                                 dtype=target.numpy_dtype))


@tack.kernel
def _mixed(a, b, out):
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        out[i, 0] = x + y
        out[i, 1] = x < y
        out[i, 2] = min(x, y)
        out[i, 3] = max(x, y)
        out[i, 4] = x if x < y else y
        out[i, 5] = x // y if y else x


MIXED = [(tack.i8, tack.u16, tack.i32), (tack.u16, tack.i8, tack.i32),
         (tack.u32, tack.i16, tack.i64), (tack.i16, tack.u32, tack.i64),
         (tack.i32, tack.u32, tack.i64), (tack.u8, tack.i64, tack.i64)]


@pytest.mark.parametrize('left,right,result', MIXED, ids=lambda t: t.name)
def test_mixed_operands_preserve_values(backend, left, right, result):
    pairs = list(itertools.product(_values(left)[:11], _values(right)[:11]))
    a, b = _field([x for x, _ in pairs], left), _field([y for _, y in pairs], right)
    out = tack.field(result, (len(pairs), 6))
    _mixed(a, b, out)
    expected = [[_wrap(x + y, result), int(x < y), min(x, y), max(x, y),
                 min(x, y), x // y if y else x] for x, y in pairs]
    np.testing.assert_array_equal(out.to_numpy(), np.asarray(expected, dtype=result.numpy_dtype))


@pytest.mark.parametrize('left,right', itertools.product(TYPES, repeat=2), ids=lambda t: t.name)
def test_promotion_preserves_entire_integer_ranges(left, right):
    info_a, info_b = np.iinfo(left.numpy_dtype), np.iinfo(right.numpy_dtype)
    lo, hi = min(int(info_a.min), int(info_b.min)), max(int(info_a.max), int(info_b.max))
    candidates = [t for t in TYPES if int(np.iinfo(t.numpy_dtype).min) <= lo
                  and int(np.iinfo(t.numpy_dtype).max) >= hi]
    if not candidates:
        with pytest.raises(TypeError, match='explicit integer cast'):
            promote_types(left, right)
        return
    result = promote_types(left, right)
    assert result.bits == min(t.bits for t in candidates)
    assert int(np.iinfo(result.numpy_dtype).min) <= lo
    assert int(np.iinfo(result.numpy_dtype).max) >= hi


@pytest.mark.parametrize('expression', [
    ir.IRBinOp('+', ir.IRName('a'), ir.IRName('b')),
    ir.IRCompare('<', ir.IRName('a'), ir.IRName('b')),
    ir.IRIfExp(ir.IRConstant(1), ir.IRName('a'), ir.IRName('b')),
])
def test_unrepresentable_mixed_types_are_rejected(expression):
    func = ir.IRFunction('mixed', [ir.IRParam('a', tack.i64), ir.IRParam('b', tack.u64)],
                         [ir.IRAssign('value', copy.deepcopy(expression))])
    with pytest.raises(TypeError, match='explicit integer cast'):
        annotate_types(func)


def test_local_join_cannot_swallow_promotion_error():
    func = ir.IRFunction('mixed', [ir.IRParam('a', tack.i64), ir.IRParam('b', tack.u64)],
                         [ir.IRAssign('value', ir.IRName('a')), ir.IRAssign('value', ir.IRName('b'))])
    with pytest.raises(TypeError, match='explicit integer cast'):
        annotate_types(func)


@pytest.mark.parametrize('generate', GENERATORS)
@pytest.mark.parametrize('dtype', TYPES, ids=lambda t: t.name)
def test_gpu_integer_helpers_use_unsigned_arithmetic(generate, dtype):
    tack.init(arch=tack.cpu)
    a, b = _field([1], dtype), _field([2], dtype)
    out = tack.field(dtype, (1, 16))
    func = copy.deepcopy(_arithmetic.get_ir().functions[0])
    resolve_ir(func, {'a': a, 'b': b, 'out': out})
    infer_param_types(func, (a, b, out))
    annotate_types(func)
    source = generate(func)
    assert f'__tack_mul_{dtype.name}__' in source
    assert f'__tack_min_{dtype.name}__' in source
    assert 'fmin' not in source and 'fabs' not in source


@tack.kernel
def _shifts(a, counts, out):
    for i in range(a.shape[0]):
        x = a[i]
        n = counts[i]
        out[i, 0] = x << n
        out[i, 1] = x >> n


@pytest.mark.parametrize('dtype', TYPES, ids=lambda t: t.name)
def test_shift_width_and_signedness(backend, dtype):
    pairs = list(itertools.product(_values(dtype)[:11], [0, 1, dtype.bits // 2, dtype.bits - 1]))
    a = _field([x for x, _ in pairs], dtype)
    counts = _field([n for _, n in pairs], tack.u64)
    out = tack.field(dtype, (len(pairs), 2))
    _shifts(a, counts, out)
    expected = [[_wrap(x << n, dtype), x >> n] for x, n in pairs]
    np.testing.assert_array_equal(out.to_numpy(), np.asarray(expected, dtype=dtype.numpy_dtype))


@tack.kernel
def _as_float(a, out):
    for i in range(a.shape[0]):
        x = a[i]
        out[i, 0] = tack.f32(x)
        out[i, 1] = x
        out[i, 2] = x + 0.0
        y = x + x
        out[i, 3] = tack.f32(y)


@pytest.mark.parametrize('dtype', TYPES, ids=lambda t: t.name)
def test_integer_signedness_survives_locals_and_float_conversion(backend, dtype):
    values = _values(dtype)
    a, out = _field(values, dtype), tack.field(tack.f32, (len(values), 4))
    _as_float(a, out)
    expected = [[float(x)] * 3 + [float(_wrap(x + x, dtype))] for x in values]
    np.testing.assert_array_equal(out.to_numpy(), np.asarray(expected, dtype=np.float32))


@tack.kernel
def _cast_chain(a, out):
    for i in range(a.shape[0]):
        x = a[i]
        s8, u8 = tack.i8(x), tack.u8(x)
        s16, u16 = tack.i16(x), tack.u16(x)
        s32, u32 = tack.i32(x), tack.u32(x)
        s64 = tack.i64(x)
        out[i, 0] = tack.i64(s8)
        out[i, 1] = tack.i64(u8)
        out[i, 2] = tack.i64(s16)
        out[i, 3] = tack.i64(u16)
        out[i, 4] = tack.i64(s32)
        out[i, 5] = tack.i64(u32)
        out[i, 6] = s64 < 0
        out[i, 7] = x < tack.u64(9223372036854775807)


def test_cast_results_keep_their_own_signedness(backend):
    values = _values(tack.u64)
    a, out = _field(values, tack.u64), tack.field(tack.i64, (len(values), 8))
    _cast_chain(a, out)
    expected = [[_wrap(x, t) for t in TYPES[:6]] + [int(_wrap(x, tack.i64) < 0),
                int(x < 2**63 - 1)] for x in values]
    np.testing.assert_array_equal(out.to_numpy(), expected)


@tack.kernel
def _float_to_unsigned(a, out32, out64):
    for i in range(a.shape[0]):
        out64[i, 0] = tack.u64(a[i])
        out64[i, 1] = a[i]
        if i < 4:
            out32[i, 0] = tack.u32(a[i])
            out32[i, 1] = a[i]


def test_representable_float_to_unsigned_conversion(backend):
    values = np.asarray([0.0, 1.9, 2**31, 2**32 - 256, 2**63, 2**64 - 2**40], dtype=np.float32)
    a = _field(values, tack.f32)
    out32, out64 = tack.field(tack.u32, (4, 2)), tack.field(tack.u64, (6, 2))
    _float_to_unsigned(a, out32, out64)
    expected = [[int(x)] * 2 for x in values]
    np.testing.assert_array_equal(out32.to_numpy(), np.asarray(expected[:4], dtype=np.uint32))
    np.testing.assert_array_equal(out64.to_numpy(), np.asarray(expected, dtype=np.uint64))


@tack.kernel
def _wrapping_bounds(starts, a, b, out):
    for i in range(a.shape[0]):
        end = a[i] + b[i]
        total = 0
        for j in range(starts[i], end):
            total = total + j
        out[i, 0] = total
        total_one = 0
        for j in range(starts[i], end, 1):
            total_one = total_one + j
        out[i, 1] = total_one
        total_two = 0
        for j in range(starts[i], end, 2):
            total_two = total_two + j
        out[i, 2] = total_two
        unsigned_total = tack.u64(0)
        for j in range(starts[i], end):
            unsigned_total = unsigned_total + tack.u64(j)
        out[i, 3] = tack.i64(unsigned_total)


def test_wrapping_dynamic_range_bound_and_empty_ranges(backend):
    triples = [(-3, 1, 2), (4, 1, 2), (3, 1, 2), (-5, -3, 2),
               (-2**31, 2**31 - 1, 2), (-2**31, -2**31, 1),
               (2**31 - 2, -2**31, -1)]
    starts, a, b = [_field([row[c] for row in triples], tack.i32) for c in range(3)]
    out = tack.field(tack.i64, (len(triples), 4))
    _wrapping_bounds(starts, a, b, out)
    expected = [[sum(range(start, _wrap(x + y, tack.i32), step)) for step in (1, 1, 2)]
                for start, x, y in triples]
    np.testing.assert_array_equal(out.to_numpy(), [row + row[:1] for row in expected])


@tack.kernel
def _literal_boundaries(out):
    for i in range(out.shape[0]):
        out[i, 0] = tack.i64(-9223372036854775808)
        out[i, 1] = tack.i64(9223372036854775807)
        out[i, 2] = tack.i64(18446744073709551615)
        out[i, 3] = tack.i64(-2147483648)


def test_full_width_integer_literals(backend):
    out = tack.field(tack.i64, (3, 4))
    _literal_boundaries(out)
    np.testing.assert_array_equal(out.to_numpy(),
                                  [[-2**63, 2**63 - 1, -1, -2**31]] * 3)


@tack.kernel
def _unsigned_scalar(a, value, out):
    for i in range(a.shape[0]):
        x = value
        out[i, 0] = x + a[i]
        out[i, 1] = x < a[i]


@pytest.mark.parametrize('value', [2**63, np.uint64(2**64 - 1)])
def test_full_width_unsigned_scalar_arguments(backend, value):
    values = [0, 1, 2**63 - 1, 2**64 - 1]
    a, out = _field(values, tack.u64), tack.field(tack.u64, (4, 2))
    _unsigned_scalar(a, value, out)
    expected = [[_wrap(int(value) + x, tack.u64), int(int(value) < x)] for x in values]
    np.testing.assert_array_equal(out.to_numpy(), np.asarray(expected, dtype=np.uint64))


@pytest.mark.parametrize('value', [-2**63 - 1, 2**64])
def test_unrepresentable_scalar_and_literal_are_rejected(value):
    func = ir.IRFunction('large', [ir.IRParam('value')], [])
    with pytest.raises(TypeError, match='outside Tack.*64-bit range'):
        infer_param_types(func, (value,))
    func = ir.IRFunction('large', [], [ir.IRAssign('x', ir.IRConstant(value))])
    with pytest.raises(TypeError, match='outside Tack.*64-bit range'):
        annotate_types(func)


@tack.kernel
def _guarded_shifts(a, counts, bits, out):
    for i in range(a.shape[0]):
        n = counts[i]
        out[i, 0] = a[i] << n if 0 <= n < bits else tack.u8(99)
        out[i, 1] = a[i] >> n if 0 <= n < bits else tack.u8(99)


@pytest.mark.parametrize('dtype', TYPES, ids=lambda t: t.name)
def test_guarded_invalid_shift_is_not_evaluated(backend, dtype):
    x = int(np.iinfo(dtype.numpy_dtype).min)
    if x == 0:
        x = int(np.iinfo(dtype.numpy_dtype).max)
    counts = [-1, dtype.bits, dtype.bits + 1, 0, dtype.bits - 1]
    a, n = _field([x] * len(counts), dtype), _field(counts, tack.i64)
    out = tack.field(dtype, (len(counts), 2))
    _guarded_shifts(a, n, dtype.bits, out)
    expected = [[_wrap(x << c, dtype), x >> c] if 0 <= c < dtype.bits else [99, 99]
                for c in counts]
    np.testing.assert_array_equal(out.to_numpy(), np.asarray(expected, dtype=dtype.numpy_dtype))


@tack.kernel
def _copy_integer(a, out):
    for i in range(a.shape[0]):
        out[i] = a[i] + a[i]


@pytest.mark.parametrize('dtype', [*TYPES, tack.f32, tack.f64], ids=lambda t: t.name)
def test_fields_do_not_promise_unproved_alignment(dtype):
    tack.init(arch=tack.cpu)
    values = _values(dtype) if dtype in TYPES else [0.0, -0.5, 1.0, 4.0]
    size = len(values) * dtype.numpy_dtype.itemsize
    data = np.ndarray((len(values),), dtype=dtype.numpy_dtype, buffer=bytearray(size + 1), offset=1)
    result = np.ndarray((len(values),), dtype=dtype.numpy_dtype, buffer=bytearray(size + 1), offset=1)
    assert data.ctypes.data % 4 == result.ctypes.data % 4 == 1
    data[:] = values
    a = tack.field_from_ptr(data, dtype, data.shape)
    out = tack.field_from_ptr(result, dtype, result.shape, writable=True)
    _copy_integer(a, out)
    expected = [_wrap(x + x, dtype) if dtype in TYPES else x + x for x in values]
    np.testing.assert_array_equal(result, np.asarray(expected,
                                                    dtype=dtype.numpy_dtype))
    source = tack.inspect(_copy_integer, a, out, mode='source')
    accesses = [line for line in source.splitlines()
                if ('load.ptr' in line or 'store.ptr' in line) and
                (' load ' in line or 'store ' in line)]
    assert accesses
    # A numerical result alone cannot detect undefined over-alignment in IR.
    assert all('align 1' in line for line in accesses)
