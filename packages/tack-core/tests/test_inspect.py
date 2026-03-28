"""Test tack.inspect() — kernel code inspection without execution."""

import numpy as np
import pytest
import tack

_backends = []
for _arch in ["cpu", "metal"]:
    try:
        tack.init(arch=getattr(tack, _arch))
        _backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=_backends)
def backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


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


def test_inspect_source(backend):
    x, y, out = _make_fields()
    result = tack.inspect(vector_add, x, y, out, mode="source")
    assert isinstance(result, str)
    assert len(result) > 50
    if backend == "cpu":
        # LLVM IR
        assert "define" in result
    elif backend == "metal":
        # MSL
        assert "kernel void" in result


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
