"""Integer // and % agree with Python on the defined fixed-width domain.

Expected values use Python integers, avoiding NumPy overflow and recording
no backend's output as a reference. Divisors are nonzero, and signed
minimum / -1 is excluded for both operators. Floating-point division,
out-of-range conversions, and general mixed-sign policy are separate work.
"""

import copy
import random

import numpy as np
import pytest

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang.ir_resolve import resolve_ir
from tack.lang.ir_type_annotate import annotate_types
from tack.lang.type_inference import infer_param_types, promote_types

INTEGER_TYPES = [tack.i8, tack.u8, tack.i16, tack.u16,
                 tack.i32, tack.u32, tack.i64, tack.u64]


@tack.kernel
def _divmod(a, b, quotients, remainders):
    for i in range(a.shape[0]):
        quotients[i] = a[i] // b[i]
        remainders[i] = a[i] % b[i]


@tack.kernel
def _nested_divmod(a, b, quotients, remainders):
    for i in range(a.shape[0]):
        x = a[i]
        y = b[i]
        quotients[i] = (x // y) // y
        remainders[i] = (x % y) % y


def _pairs(dtype):
    info = np.iinfo(dtype.numpy_dtype)
    lo, hi = int(info.min), int(info.max)
    if dtype is tack.i8:
        return [(a, b) for a in range(lo, hi + 1)
                for b in range(lo, hi + 1)
                if b and (a, b) != (lo, -1)]
    values = [lo, lo + 1, 0, 1, 2, 3, hi - 1, hi]
    if lo < 0:
        values += [-3, -2, -1]
    pairs = [(a, b) for a in values for b in values
             if b and (a, b) != (lo, -1)]
    rng = random.Random(5100 + dtype.bits)
    for _ in range(512):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if b and (a, b) != (lo, -1):
            pairs.append((a, b))
    return pairs


def _field(values, dtype):
    data = np.asarray(values, dtype=dtype.numpy_dtype)
    field = tack.field(dtype=dtype, shape=data.shape)
    field.from_numpy(data)
    return field


@pytest.mark.parametrize('dtype', INTEGER_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize('nested', [False, True], ids=['direct', 'nested-locals'])
def test_integer_division_matches_python(backend, dtype, nested):
    pairs = _pairs(dtype)
    # A second division could overflow if its first result is the signed
    # minimum and its second divisor -1; the excluded original pair is the
    # only way that can occur with these identical divisors.
    a = _field([x for x, _ in pairs], dtype)
    b = _field([y for _, y in pairs], dtype)
    q = tack.field(dtype, a.shape)
    r = tack.field(dtype, a.shape)
    (_nested_divmod if nested else _divmod)(a, b, q, r)
    expected_q = [x // y for x, y in pairs]
    expected_r = [x % y for x, y in pairs]
    if nested:
        expected_q = [v // y for v, (_, y) in zip(expected_q, pairs)]
        expected_r = [v % y for v, (_, y) in zip(expected_r, pairs)]
    np.testing.assert_array_equal(q.to_numpy(), np.asarray(expected_q, dtype=dtype.numpy_dtype))
    np.testing.assert_array_equal(r.to_numpy(), np.asarray(expected_r, dtype=dtype.numpy_dtype))
    if not nested:
        for (x, y), quotient, remainder in zip(pairs, expected_q, expected_r):
            assert x == quotient * y + remainder
            assert abs(remainder) < abs(y)
            assert remainder == 0 or (remainder > 0) == (y > 0)


@tack.kernel
def _literal_divmod(a, q, r):
    for i in range(a.shape[0]):
        q[i] = a[i] // 3
        r[i] = a[i] % -3


@pytest.mark.parametrize('dtype', [tack.i8, tack.u8, tack.i16, tack.u16, tack.i32, tack.u32])
def test_integer_literals_use_tack_promotion(backend, dtype):
    info = np.iinfo(dtype.numpy_dtype)
    values = [int(info.min), int(info.min) + 1, 0, 1, int(info.max)]
    a = _field(values, dtype)
    q, r = tack.field(tack.i64, a.shape), tack.field(tack.i64, a.shape)
    _literal_divmod(a, q, r)
    np.testing.assert_array_equal(q.to_numpy(), [x // 3 for x in values])
    np.testing.assert_array_equal(r.to_numpy(), [x % -3 for x in values])


@pytest.mark.parametrize('left_type,right_type', [
    (tack.i8, tack.u8), (tack.u8, tack.i8),
    (tack.i16, tack.u16), (tack.u16, tack.i16),
    (tack.i32, tack.u32), (tack.u32, tack.i32),
    (tack.i64, tack.u32), (tack.u32, tack.i64),
])
def test_lossless_mixed_integer_promotion(backend, left_type, right_type):
    left_info, right_info = np.iinfo(left_type.numpy_dtype), np.iinfo(right_type.numpy_dtype)
    left = [int(left_info.min), int(left_info.min) + 1, 0, 1, int(left_info.max)]
    right = [int(right_info.min), 1, 3, int(right_info.max)]
    if int(right_info.min) < 0:
        right += [-1, -3]
    pairs = [(x, y) for x in left for y in right if y]
    result_type = promote_types(left_type, right_type)
    a, b = _field([x for x, _ in pairs], left_type), _field([y for _, y in pairs], right_type)
    q, r = tack.field(result_type, a.shape), tack.field(result_type, a.shape)
    _divmod(a, b, q, r)
    np.testing.assert_array_equal(q.to_numpy(), np.array([x // y for x, y in pairs],
                                                       dtype=result_type.numpy_dtype))
    np.testing.assert_array_equal(r.to_numpy(), np.array([x % y for x, y in pairs],
                                                       dtype=result_type.numpy_dtype))


@tack.func
def _touch(state, i, value):
    state[i] = state[i] + 1
    return value


@tack.kernel
def _effects(state, q, r):
    for i in range(state.shape[0]):
        q[i] = _touch(state, i, -7) // _touch(state, i, 3)
        r[i] = _touch(state, i, 7) % _touch(state, i, -3)


def test_operands_execute_once(backend):
    state = tack.field(tack.i32, (9,))
    q, r = tack.field(tack.i32, (9,)), tack.field(tack.i32, (9,))
    _effects(state, q, r)
    np.testing.assert_array_equal(state.to_numpy(), np.full(9, 4))
    np.testing.assert_array_equal(q.to_numpy(), np.full(9, -3))
    np.testing.assert_array_equal(r.to_numpy(), np.full(9, -2))


@tack.kernel
def _guarded(a, b, q, r):
    for i in range(a.shape[0]):
        q[i] = a[i] // b[i] if b[i] != 0 else 99
        r[i] = a[i] % b[i] if b[i] != 0 else 99


def test_guarded_zero_divisor_is_not_evaluated(backend):
    a, b = _field([-7, 7, 3, 0], tack.i32), _field([3, -3, 0, 0], tack.i32)
    q, r = tack.field(tack.i32, (4,)), tack.field(tack.i32, (4,))
    _guarded(a, b, q, r)
    np.testing.assert_array_equal(q.to_numpy(), [-3, -3, 99, 99])
    np.testing.assert_array_equal(r.to_numpy(), [2, -2, 99, 99])


@tack.kernel
def _pixel_coordinates(width, x, y):
    for pid in range(x.shape[0]):
        x[pid] = pid % width
        y[pid] = pid // width


@pytest.mark.parametrize('width', [17, 512])
def test_parallel_pixel_coordinates_are_exact(backend, width):
    """Integer indexing used by renderers stays exact across the grid."""
    n = width * width
    x, y = tack.field(tack.i64, (n,)), tack.field(tack.i64, (n,))
    _pixel_coordinates(width, x, y)
    ids = np.arange(n, dtype=np.int64)
    np.testing.assert_array_equal(x.to_numpy(), ids % width)
    np.testing.assert_array_equal(y.to_numpy(), ids // width)


@pytest.mark.parametrize('generate', [generate_cuda_source, generate_hip_source,
                                     generate_msl_source, generate_opencl_source])
@pytest.mark.parametrize('dtype', [tack.i32, tack.i64, tack.u32, tack.u64], ids=lambda t: t.name)
def test_all_gpu_generators_emit_integer_division(generate, dtype):
    tack.init(arch=tack.cpu)
    a, b, q, r = [_field([3], dtype) for _ in range(4)]
    func = copy.deepcopy(_divmod.get_ir().functions[0])
    resolve_ir(func, {'a': a, 'b': b, 'quotients': q, 'remainders': r})
    infer_param_types(func, (a, b, q, r))
    annotate_types(func)
    source = generate(func)
    if dtype.name.startswith('i'):
        assert f'__tack_floordiv_{dtype.name}__' in source
        assert f'__tack_mod_{dtype.name}__' in source
        assert 'return q - adjust;' in source
        assert 'return adjust ? r + b : r;' in source
    else:
        assert '__tack_floordiv_' not in source
    assert 'floor(' not in source and 'floorf(' not in source
