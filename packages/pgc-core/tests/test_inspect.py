"""Test pgc.inspect() — kernel code inspection without execution."""

import numpy as np
import pytest
import pgc

_backends = []
for _arch in ["cpu", "metal"]:
    try:
        pgc.init(arch=getattr(pgc, _arch))
        _backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=_backends)
def backend(request):
    pgc.init(arch=getattr(pgc, request.param))
    return request.param


@pgc.kernel
def vector_add(x, y, out):
    for i in range(len(x)):
        out[i] = x[i] + y[i]


def _make_fields():
    n = 64
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    return x, y, out


def test_inspect_ir(backend):
    x, y, out = _make_fields()
    result = pgc.inspect(vector_add, x, y, out, mode="ir")
    assert isinstance(result, str)
    assert "Function" in result
    assert "ParallelFor" in result


def test_inspect_source(backend):
    x, y, out = _make_fields()
    result = pgc.inspect(vector_add, x, y, out, mode="source")
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
    result = pgc.inspect(vector_add, x, y, out)
    assert isinstance(result, str)
    assert len(result) > 50


def test_inspect_invalid_mode(backend):
    x, y, out = _make_fields()
    with pytest.raises(ValueError, match="Unknown inspect mode"):
        pgc.inspect(vector_add, x, y, out, mode="bad")


def test_inspect_not_a_kernel(backend):
    with pytest.raises(TypeError, match="Expected a @pgc.kernel"):
        pgc.inspect(lambda: None, mode="ir")


def test_inspect_scalar_args(backend):
    @pgc.kernel
    def scale(x, out, factor):
        for i in range(len(x)):
            out[i] = x[i] * factor

    n = 64
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    result = pgc.inspect(scale, x, out, 2.0, mode="source")
    assert isinstance(result, str)
    assert len(result) > 50
