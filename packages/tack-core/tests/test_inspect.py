"""Test tack.inspect() — kernel code inspection without execution."""

import pytest

import tack


@tack.kernel
def vector_add(x, y, out):
    for i in range(len(x)):
        out[i] = x[i] + y[i]


def _make_fields():
    n = 64
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    return x, y, out


def test_inspect_ir(backend):
    x, y, out = _make_fields()
    result = tack.inspect(vector_add, x, y, out, mode="ir")
    assert isinstance(result, str)
    assert "Function" in result
    assert "ParallelFor" in result


# What each backend's generated source has to contain: the spelling of its
# kernel entry point. Every backend is listed, and an unknown one is a
# failure rather than a pass -- this test used to check "cpu" and "metal"
# and fall through everywhere else, so on a CUDA machine it asserted only
# that the string was longer than fifty characters.
_ENTRY_POINT = {
    "cpu": "define",              # LLVM IR
    "metal": "kernel void",       # MSL
    "cuda": "__global__",         # CUDA C
    "hip": "__global__",          # HIP C, which extends the CUDA codegen
    "level_zero": "__kernel",     # OpenCL C
}


def test_inspect_source(backend):
    x, y, out = _make_fields()
    result = tack.inspect(vector_add, x, y, out, mode="source")
    assert isinstance(result, str)
    assert len(result) > 50

    assert backend in _ENTRY_POINT, (
        f"no entry point known for {backend!r}; add it rather than let this "
        f"test pass without checking anything")
    assert _ENTRY_POINT[backend] in result


def test_hip_source_carries_its_runtime_header(backend):
    """HIP shares the CUDA codegen and differs by exactly this include."""
    if backend != "hip":
        pytest.skip("HIP only")
    x, y, out = _make_fields()
    result = tack.inspect(vector_add, x, y, out, mode="source")
    assert "#include <hip/hip_runtime.h>" in result


def test_inspect_default_mode(backend):
    x, y, out = _make_fields()
    result = tack.inspect(vector_add, x, y, out)
    assert isinstance(result, str)
    assert len(result) > 50


def test_inspect_invalid_mode(backend):
    x, y, out = _make_fields()
    with pytest.raises(ValueError, match="Unknown inspect mode"):
        tack.inspect(vector_add, x, y, out, mode="bad")


def test_inspect_not_a_kernel(backend):
    with pytest.raises(TypeError, match="Expected a @tack.kernel"):
        tack.inspect(lambda: None, mode="ir")


def test_inspect_scalar_args(backend):
    @tack.kernel
    def scale(x, out, factor):
        for i in range(len(x)):
            out[i] = x[i] * factor

    n = 64
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    result = tack.inspect(scale, x, out, 2.0, mode="source")
    assert isinstance(result, str)
    assert len(result) > 50
