"""Float literals are weakly typed: they take the precision of what they meet.

Every float literal used to be f32, so `x * 0.1` on an f64 field multiplied
by 0.10000000149011612, and even `tack.f64(0.1)` widened that f32 value.
A literal expression now adopts the floating type of its partner, converting
the exact Python value once. Pure f32 expressions are unchanged.
"""

import subprocess

import numpy as np
import pytest
from compiler_tools import require_clang

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.identifiers import kernel_entry_name
from tack.codegen.llvm_gen import generate_llvm_ir
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang.inspect_kernel import _prepare_ir
from tack.runtime.dispatch import get_backend

_VALUES = [0.1, float(np.float32(0.1)), 0.1000000001, 1.0, 7.0, 2.5, 1e-3, 123.456]
_COLUMNS = 10


def _field(values, dtype):
    values = np.asarray(values, dtype=dtype.numpy_dtype)
    result = tack.field(dtype, values.shape)
    result.from_numpy(values)
    return result


@tack.kernel
def _literals(a, out):
    for i in range(a.shape[0]):
        x = a[i]
        out[i, 0] = x * 0.1
        out[i, 1] = x * -0.1
        out[i, 2] = x * (0.1 + 0.2)
        out[i, 3] = x * (1.0 / 3.0)
        out[i, 4] = 1.0 if x < 0.1 else 0.0
        out[i, 5] = 1.0 if x == 0.1 else 0.0
        out[i, 6] = sqrt(x + 0.1)
        out[i, 7] = x if x > 5.0 else 0.1
        out[i, 8] = min(x, 0.1)
        out[i, 9] = 0.1


def _expected(a, scalar):
    """NumPy's result with each literal converted once to `scalar`."""
    tenth = scalar(0.1)
    return np.column_stack([
        a * tenth, a * scalar(-0.1), a * (tenth + scalar(0.2)),
        a * (scalar(1.0) / scalar(3.0)),
        np.where(a < tenth, 1, 0), np.where(a == tenth, 1, 0),
        np.sqrt(a + tenth), np.where(a > 5, a, tenth), np.fmin(a, tenth),
        np.full(len(a), tenth),
    ]).astype(a.dtype)


def test_f64_literals_take_f64_precision(f64_backend):
    a = np.asarray(_VALUES)
    out = tack.field(tack.f64, (len(a), _COLUMNS))
    _literals(_field(a, tack.f64), out)
    # Each operation here is correctly rounded in f64, so equality is exact.
    np.testing.assert_array_equal(out.to_numpy(), _expected(a, np.float64))


def test_f32_literals_are_unchanged(backend):
    a = np.asarray(_VALUES, dtype=np.float32)
    out = tack.field(tack.f32, (len(a), _COLUMNS))
    _literals(_field(a, tack.f32), out)
    result, expected = out.to_numpy(), _expected(a, np.float32)
    # f32 division and square root need not be correctly rounded on a GPU.
    rounded = [3, 6]
    exact = [c for c in range(_COLUMNS) if c not in rounded]
    np.testing.assert_array_equal(result[:, exact], expected[:, exact])
    np.testing.assert_array_max_ulp(result[:, rounded], expected[:, rounded], maxulp=4)


def test_explicit_casts_convert_literals_once(f64_backend):
    @tack.kernel
    def casts(a, out):
        for i in range(a.shape[0]):
            out[i, 0] = tack.f64(0.1)
            out[i, 1] = a[i] * tack.f64(0.1)
            out[i, 2] = tack.f32(0.1)
            out[i, 3] = tack.f64(1.0 / 3.0)

    a = np.asarray([1.0, 3.0, 0.5], dtype=np.float32)
    out = tack.field(tack.f64, (3, 4))
    casts(_field(a, tack.f32), out)
    wide = a.astype(np.float64)
    expected = np.column_stack([np.full(3, 0.1), wide * 0.1,
                                np.full(3, np.float64(np.float32(0.1))), np.full(3, 1.0 / 3.0)])
    np.testing.assert_array_equal(out.to_numpy(), expected)


def test_atomic_literal_takes_the_field_type(f64_backend):
    if tack.f64 not in get_backend().supported_atomic_dtypes:
        pytest.skip('Backend has f64 fields but no f64 atomics')

    @tack.kernel
    def accumulate(total):
        for i in range(1):
            tack.atomic_add(total, 0, 0.1)

    total = _field([1.0], tack.f64)
    accumulate(total)
    assert total.to_numpy()[0] == 1.0 + 0.1


def test_literals_without_a_floating_partner_keep_f32(f64_backend):
    @tack.kernel
    def partners(a, out):
        for i in range(a.shape[0]):
            x = a[i]
            alone = 0.1
            out[i, 0] = x * alone
            out[i, 1] = i * 0.1
            out[i, 2] = x * (1 / 3)
            out[i, 3] = x * sqrt(2.0)
            total = 0.1
            total = total + x
            out[i, 4] = total

    a = np.asarray([1.0, 3.0, 0.5])
    out = tack.field(tack.f64, (3, 5))
    partners(_field(a, tack.f64), out)
    expected = np.column_stack([
        # A local assigned once, to literals, is the literal where it is read.
        a * 0.1,
        # Integer operands do not supply a floating type.
        (np.arange(3, dtype=np.float32) * np.float32(0.1)).astype(np.float64),
        # Integer `/` of integer literals is f32 by the division rule.
        a * np.float64(np.float32(1) / np.float32(3)),
        # A math call on literals is itself a literal expression.
        a * np.sqrt(2.0),
        # A literal stored into a local that settles on f64 converts exactly.
        0.1 + a,
    ])
    np.testing.assert_array_equal(out.to_numpy(), expected)


@tack.kernel
def _long_literal(a, out):
    for i in range(a.shape[0]):
        out[i] = a[i] * 0.30000000000000004 + 1e-300


def _prepare(kernel, dtype):
    """Annotated IR for `kernel` on one-row fields, without a device."""
    tack.init(arch=tack.cpu)
    shape = (1,) if kernel is _long_literal else (1, _COLUMNS)
    function, _ = _prepare_ir(kernel, (_field([1.0], dtype), tack.field(dtype, shape)))
    return function


_C_GENERATORS = [generate_cuda_source, generate_hip_source, generate_opencl_source]
_C_IDS = ['cuda', 'hip', 'opencl']


@pytest.mark.parametrize('generate', _C_GENERATORS, ids=_C_IDS)
def test_c_generators_emit_exact_double_literals(generate, tmp_path):
    source = generate(_prepare(_long_literal, tack.f64))
    assert '0.30000000000000004' in source
    assert '1e-300' in source
    assert '0.30000000000000004f' not in source and '1e-300f' not in source
    if generate is generate_opencl_source:
        clang = require_clang()
        path = tmp_path / 'literal.cl'
        path.write_text(source)
        result = subprocess.run([clang, '-target', 'x86_64-unknown-linux-gnu', '-x', 'cl',
                                 '-cl-std=CL1.2', '-fsyntax-only', str(path)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_llvm_generator_emits_exact_double_literals():
    ll = str(generate_llvm_ir(_prepare(_long_literal, tack.f64)))
    # 0.30000000000000004 and 1e-300 as doubles; f32 would flush the latter.
    assert '0x3fd3333333333334' in ll and '0x1a56e1fc2f8f359' in ll
    assert 'float 0x' not in ll and 'fpext' not in ll


@pytest.mark.parametrize('generate', [*_C_GENERATORS, generate_msl_source],
                         ids=[*_C_IDS, 'metal'])
def test_f32_kernels_emit_no_double_literals(generate):
    source = generate(_prepare(_literals, tack.f32))
    assert '0.1f' in source
    assert 'double' not in source


def test_llvm_f32_kernels_emit_no_double_literals():
    ll = str(generate_llvm_ir(_prepare(_literals, tack.f32)))
    assert 'float 0x3fb99999a0000000' in ll
    assert 'double' not in ll


@pytest.mark.parametrize('generate', [generate_cuda_source, generate_hip_source],
                         ids=['cuda-host', 'hip-host'])
def test_generated_cpp_computes_exact_f64_literals(generate, tmp_path):
    clang = require_clang('ubsan')
    a = np.asarray(_VALUES)
    function = _prepare(_literals, tack.f64)
    source = '\n'.join(line for line in generate(function).splitlines()
                       if not line.startswith('#include'))
    size = len(a) * _COLUMNS
    source = '''
#include <cmath>
#include <cstdio>
#define __global__
#define __device__
struct Grid { unsigned int x; };
Grid blockIdx = {0}, blockDim = {1}, threadIdx = {0};
''' + source + f'''
int main(int argc, char** argv) {{
    double a[{len(a)}], out[{size}];
    FILE* input = std::fopen(argv[1], "rb");
    if (!input) return 2;
    if (std::fread(a, sizeof(double), {len(a)}, input) != {len(a)}) return 3;
    std::fclose(input);
    for (unsigned int i = 0; i < {len(a)}; ++i) {{
        blockIdx.x = i;
        {kernel_entry_name(function.name)}(a, out, {len(a)});
    }}
    return std::fwrite(out, sizeof(double), {size}, stdout) == {size} ? 0 : 4;
}}
'''
    cpp, executable, input_path = tmp_path / 'kernel.cpp', tmp_path / 'kernel', tmp_path / 'a.bin'
    cpp.write_text(source)
    # Contraction would fuse nothing here, but keep the host compiler honest.
    result = subprocess.run([clang, '-std=c++14', '-O2', '-ffp-contract=off',
                             '-fsanitize=undefined', '-fno-sanitize-recover=all',
                             str(cpp), '-o', str(executable)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    a.tofile(input_path)
    result = subprocess.run([str(executable), str(input_path)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    actual = np.frombuffer(result.stdout, dtype=np.float64).reshape(-1, _COLUMNS)
    np.testing.assert_array_equal(actual, _expected(a, np.float64))


# ── Locals holding literals ─────────────────────────────────────────

@tack.func
def _center():
    return tack.Vector([1.0 / 3.0, 0.1, 0.5])


@tack.func
def _pair():
    return 0.1, 0.2


@tack.func
def _branchy(k):
    """Literals returned from two branches: a result local assigned twice."""
    if k > 0:
        return 0.1
    return 0.3


@tack.kernel
def _weak_locals(a, out64, out32, flag):
    for i in range(a.shape[0]):
        pc = _center()                    # a vector of literals, through a device function
        p, q = _pair()                    # several literals, through tuple unpacking
        tenth = 0.1
        chained = tenth                   # a copy of one is one too
        picked = 0.3 if 1 > 0 else 0.7    # a condition of constants
        guarded = 0.3 if flag > 0 else 0.7   # a runtime condition: not replaced, but f64 here
        step = _branchy(flag)             # assigned in two branches: f64 in an f64 kernel
        mixed = 0.1                       # a literal, then not: the join, as before
        total = 0.1                       # assigned twice: its type is its assignments' join
        total = total + a[i]
        if flag > 5:
            mixed = a[i]
        out64[i, 0] = a[i] * pc[0]
        out64[i, 1] = a[i] * pc[1]
        out64[i, 2] = a[i] * chained
        out64[i, 3] = a[i] * picked
        out64[i, 4] = a[i] * guarded
        out64[i, 5] = total
        out64[i, 6] = a[i] * step
        out64[i, 7] = a[i] * mixed
        # No multiply beside the adds: a GPU may fuse a * b + c, which the
        # contract permits, and this compares bit for bit with NumPy.
        out64[i, 8] = a[i] + p - q
        out32[i] = tack.f32(a[i]) * tenth   # the same local, read in f32


def test_locals_holding_literals_take_the_precision_they_meet(f64_backend):
    a = np.asarray([1.0, 3.0, 0.7])
    out64 = tack.field(tack.f64, (3, 9))
    out32 = tack.field(tack.f32, (3,))
    _weak_locals(_field(a, tack.f64), out64, out32, 1)
    f32 = np.float32
    expected = np.column_stack([
        a * (1.0 / 3.0),
        a * 0.1,
        a * 0.1,
        a * 0.3,
        a * 0.3,
        0.1 + a,
        a * 0.1,
        a * 0.1,
        a + 0.1 - 0.2,
    ])
    np.testing.assert_array_equal(out64.to_numpy(), expected)
    np.testing.assert_array_equal(out32.to_numpy(), a.astype(f32) * f32(0.1))


@tack.kernel
def _f32_locals(a, out):
    for i in range(a.shape[0]):
        pc = _center()
        scale = 0.1
        out[i] = a[i] * pc[0] + a[i] * scale


def test_f32_kernels_are_unchanged(backend):
    a = np.asarray([1.0, 3.0, 0.7], np.float32)
    out = tack.field(tack.f32, (3,))
    _f32_locals(_field(a, tack.f32), out)
    f32 = np.float32
    np.testing.assert_allclose(out.to_numpy(), a * f32(1.0 / 3.0) + a * f32(0.1), rtol=1e-6)


def test_which_locals_are_replaced():
    from tack.lang.ir_optimize import _literal
    tack.init(arch=tack.cpu)
    text = tack.inspect(_weak_locals, _field([1.0], tack.f64), tack.field(tack.f64, (1, 8)),
                        tack.field(tack.f32, (1,)), 1, mode="ir")
    assigned = {line.split("=")[0].strip() for line in text.splitlines() if " = " in line
                and "[" not in line.split("=")[0]}
    assert not {"tenth", "chained", "picked", "p", "q"} & assigned
    assert {"guarded", "total"} <= assigned
    ir = tack.lang.ir
    assert _literal(ir.IRBinOp("*", ir.IRConstant(2), ir.IRConstant(0.5))) == (True, True)
    assert _literal(ir.IRBinOp("*", ir.IRConstant(2), ir.IRConstant(3))) == (True, False)
    assert _literal(ir.IRBinOp("*", ir.IRName("x"), ir.IRConstant(0.5)))[0] is False
    assert _literal(ir.IRCast(ir.IRConstant(0.5), tack.f64))[0] is False

