"""Compile generated OpenCL C with Clang, in the shapes a device rejected.

Two defects reached an Intel Max 1100 and failed there, in `3ead45d`: a float
used as the condition of `?:`, and `__local` declared in a nested scope. Both
are OpenCL C rules that C++ does not share, so CUDA, HIP and MSL accepted the
same generated spellings and only ocloc complained.

A host syntax check existed and did not catch either one, for two reasons worth
keeping separate. It skipped wherever `clang` was absent -- which included the
one machine with an Intel GPU -- and it compiled a single kernel shape that had
neither a float ternary nor a nested local, so installing `clang` alone would
not have helped. This module covers the shapes instead, and `clang` rejects both
defects with the same wording ocloc used.

These checks need no GPU. The `_prepare_ir` path matters: it is what the runtime
runs, and `test_opencl_gen._source` stops at type inference, leaving expression
nodes without a `dtype` -- so a float ternary built that way is not recognizably
float and the defect disappears from the test rather than from the code.
"""

import subprocess

import pytest
from compiler_tools import require_clang

import tack
from tack.codegen.opencl_gen import OpenCLCodeGen
from tack.lang.inspect_kernel import _prepare_ir

# What the Level Zero backend passes libocloc.
_CL_STD = "-cl-std=CL2.0"


@pytest.fixture(scope="module")
def clang():
    return require_clang()


def _compile(clang_bin, source, tmp_path):
    """Syntax-check `source` as OpenCL C. Returns (returncode, stderr)."""
    path = tmp_path / "kernel.cl"
    path.write_text(source)
    done = subprocess.run(
        [clang_bin, "-target", "x86_64-unknown-linux-gnu", "-x", "cl",
         _CL_STD, "-fsyntax-only", str(path)],
        capture_output=True, text=True,
    )
    return done.returncode, done.stderr


def _assert_compiles(clang_bin, source, tmp_path):
    code, err = _compile(clang_bin, source, tmp_path)
    assert code == 0, f"Clang rejected generated OpenCL C:\n{err}\n--- source ---\n{source}"


class _PreFixOpenCLCodeGen(OpenCLCodeGen):
    """The generator as it stood before `3ead45d`, for comparison only.

    Both defects are reachable by undoing one override each, which keeps the
    comparison honest: the same IR, the same generator, only the two emission
    decisions restored. Nothing in the production path or in the stage-six
    participation rules is relaxed to build it.
    """

    def _ifexp_condition(self, node):
        return self._expr(node)           # a bare float condition

    def _declare_local(self, name, decl):
        self._emit(decl)                  # at the point of use, not kernel scope


def _source(kernel_fn, *args):
    """Generate OpenCL C the way the backend does, annotation included."""
    tack.init(arch=tack.cpu)
    function, _ = _prepare_ir(kernel_fn, args)
    return OpenCLCodeGen(function).generate()


def _pre_fix_source(kernel_fn, *args):
    tack.init(arch=tack.cpu)
    function, _ = _prepare_ir(kernel_fn, args)
    return _PreFixOpenCLCodeGen(function).generate()


def _field(shape=(256,), dtype=tack.f32):
    return tack.field(dtype=dtype, shape=shape)


# ── Kernels ──────────────────────────────────────────────────────────

@tack.kernel
def _ternary(x, out):
    for i in range(x.shape[0]):
        out[i] = 1 if x[i] else 0


@tack.kernel
def _if_statement(x, out):
    for i in range(x.shape[0]):
        if x[i]:
            out[i] = 1
        else:
            out[i] = 0


@tack.kernel
def _shared_nested(x, out):
    # No barrier: with one, the participation check rejects this before
    # codegen, which is what hid the nested-declaration site.
    for i in range(x.shape[0]):
        t = tack.thread_id()
        if x[i] > -1.0:
            buf = tack.shared(tack.f32, 256)
            buf[t] = x[i]
            out[i] = buf[t]


@tack.kernel
def _shared_like_nested(x, out):
    for i in range(x.shape[0]):
        t = tack.thread_id()
        if x[i] > -1.0:
            buf = tack.shared_like(x, 256)
            buf[t] = x[i]
            out[i] = buf[t]


@tack.kernel
def _reduce_in_uniform_branch(data, out, n):
    for i in range(data.shape[0]):
        if n > 0:
            out[i] = tack.block_sum(data[i])


@tack.kernel
def _reduce_in_loop(data, out):
    for i in range(data.shape[0]):
        for _ in range(2):
            out[i] = tack.block_sum(data[i])


@tack.kernel
def _float_math(a, out):
    for i in range(a.shape[0]):
        out[i] = sqrt(abs(a[i])) + exp(a[i]) + floor(a[i]) + ceil(a[i])


# ── The two shapes a device rejected ─────────────────────────────────

@pytest.mark.parametrize("dtype", [tack.f32, tack.f64], ids=["f32", "f64"])
def test_float_ternary_condition_compiles(clang, tmp_path, dtype):
    """`x ? a : b` with a floating `x`: legal C++, rejected OpenCL C."""
    src = _source(_ternary, _field(dtype=dtype), _field(dtype=tack.i32))
    _assert_compiles(clang, src, tmp_path)


@pytest.mark.parametrize("dtype", [tack.f32, tack.f64], ids=["f32", "f64"])
def test_pre_fix_float_ternary_is_rejected(clang, tmp_path, dtype):
    """The check earns its place only if it fails on the old spelling."""
    src = _pre_fix_source(_ternary, _field(dtype=dtype), _field(dtype=tack.i32))
    code, err = _compile(clang, src, tmp_path)
    assert code != 0, f"Clang accepted the pre-fix float ternary:\n{src}"
    assert "where floating point type is not allowed" in err, err


@pytest.mark.parametrize("kernel", [_shared_nested, _shared_like_nested],
                         ids=["shared", "shared_like"])
def test_nested_local_allocation_compiles(clang, tmp_path, kernel):
    """`__local` belongs in the outermost scope of a kernel."""
    src = _source(kernel, _field(), _field())
    _assert_compiles(clang, src, tmp_path)


@pytest.mark.parametrize("kernel", [_shared_nested, _shared_like_nested],
                         ids=["shared", "shared_like"])
def test_pre_fix_nested_local_is_rejected(clang, tmp_path, kernel):
    src = _pre_fix_source(kernel, _field(), _field())
    code, err = _compile(clang, src, tmp_path)
    assert code != 0, f"Clang accepted a nested __local declaration:\n{src}"
    assert "local address space" in err and "outermost scope" in err, err


def test_block_reduction_in_a_uniform_branch_compiles(clang, tmp_path):
    """Reduction scratch under a uniform scalar condition."""
    src = _source(_reduce_in_uniform_branch, _field(), _field(), 4)
    _assert_compiles(clang, src, tmp_path)


def test_block_reduction_in_a_loop_compiles(clang, tmp_path):
    src = _source(_reduce_in_loop, _field(), _field())
    _assert_compiles(clang, src, tmp_path)


def test_pre_fix_block_reduction_scratch_is_rejected(clang, tmp_path):
    """The reduction site is the one the device actually failed on."""
    src = _pre_fix_source(_reduce_in_loop, _field(), _field())
    code, err = _compile(clang, src, tmp_path)
    assert code != 0, f"Clang accepted nested reduction scratch:\n{src}"
    assert "outermost scope" in err, err


# ── Controls: shapes that were always fine and must stay cheap ────────

def test_integer_ternary_condition_compiles(clang, tmp_path):
    src = _source(_ternary, _field(dtype=tack.i32), _field(dtype=tack.i32))
    _assert_compiles(clang, src, tmp_path)
    assert "!= 0" not in src, "an integer condition does not need a comparison"


def test_float_if_statement_compiles(clang, tmp_path):
    """OpenCL C allows a float `if` condition, following C99; only `?:` differs.

    This is also a negative control for the ternary fix: the pre-fix generator
    emitted the same thing here, so rewriting `if` would be cost without cause.
    """
    src = _source(_if_statement, _field(), _field(dtype=tack.i32))
    _assert_compiles(clang, src, tmp_path)
    assert "!= 0.0f" not in src


@pytest.mark.parametrize("dtype", [tack.f32, tack.f64], ids=["f32", "f64"])
def test_float_math_compiles(clang, tmp_path, dtype):
    """f64 reaches here through `cl_khr_fp64`, and `copysign` wraps floor/ceil."""
    src = _source(_float_math, _field(dtype=dtype), _field(dtype=dtype))
    _assert_compiles(clang, src, tmp_path)
    if dtype is tack.f64:
        assert "cl_khr_fp64" in src
        assert "copysign(" in src          # the Intel floor/ceil workaround
    else:
        assert "copysign" not in src


# ── The skip must not be able to masquerade as a pass ────────────────

def test_absent_clang_fails_when_required(monkeypatch, tmp_path):
    """`TACK_REQUIRE_CLANG=1` turns "no compiler" into a failure, not a skip.

    The original gap was not only the missing shapes: the check skipped on the
    very host that had the device, and a skip reads like success in a summary.
    """
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("TACK_CLANG", raising=False)
    monkeypatch.setenv("TACK_REQUIRE_CLANG", "1")
    # `pytest.fail` raises off BaseException, so name the type rather than
    # catching Exception -- which would let the failure escape the test.
    with pytest.raises(pytest.fail.Exception) as excinfo:
        require_clang()
    assert "TACK_REQUIRE_CLANG" in str(excinfo.value)
    assert not isinstance(excinfo.value, pytest.skip.Exception), "must fail, not skip"


def test_absent_clang_skips_by_default(monkeypatch, tmp_path):
    """A developer without Clang gets a reported skip, not a failure."""
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("TACK_CLANG", raising=False)
    monkeypatch.delenv("TACK_REQUIRE_CLANG", raising=False)
    with pytest.raises(pytest.skip.Exception, match="Clang is required"):
        require_clang()


def test_explicit_clang_path_is_used(monkeypatch, clang):
    """`TACK_CLANG` wins, so a Clang outside PATH still counts."""
    monkeypatch.setenv("TACK_CLANG", clang)
    monkeypatch.setenv("PATH", "")
    assert require_clang() == clang


def test_unusable_explicit_clang_is_not_silently_ignored(monkeypatch, tmp_path):
    """A bad TACK_CLANG must not fall through to a PATH Clang and look fine."""
    missing = tmp_path / "not-a-compiler"
    monkeypatch.setenv("TACK_CLANG", str(missing))
    monkeypatch.setenv("TACK_REQUIRE_CLANG", "1")
    with pytest.raises(pytest.fail.Exception, match="TACK_REQUIRE_CLANG"):
        require_clang()
