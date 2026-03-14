"""Tests for type inference on kernel arguments."""

import pytest
import numpy as np
import pgc
from pgc.lang.type_inference import infer_param_types, promote_types
from pgc.lang.types import f32, f64, i32, i64
from pgc.lang import ir


def test_infer_field_types():
    x = pgc.field(dtype=pgc.f32, shape=(10,))
    y = pgc.field(dtype=pgc.f32, shape=(10,))
    out = pgc.field(dtype=pgc.f32, shape=(10,))

    func = ir.IRFunction("add", [
        ir.IRParam("x"),
        ir.IRParam("y"),
        ir.IRParam("out"),
    ], [])

    types = infer_param_types(func, (x, y, out))
    assert types == [f32, f32, f32]
    assert func.params[0].type_annotation is f32


def test_infer_scalar_types():
    x = pgc.field(dtype=pgc.f32, shape=(10,))

    func = ir.IRFunction("scale", [
        ir.IRParam("x"),
        ir.IRParam("alpha"),
        ir.IRParam("n"),
    ], [])

    types = infer_param_types(func, (x, 2.5, 10))
    assert types == [f32, f32, i32]


def test_infer_wrong_arg_count():
    func = ir.IRFunction("f", [ir.IRParam("x")], [])
    with pytest.raises(TypeError, match="expects 1 arguments"):
        infer_param_types(func, (1, 2))


def test_promote_same():
    assert promote_types(f32, f32) is f32


def test_promote_int_float():
    assert promote_types(i32, f32) is f32
    assert promote_types(f32, i32) is f32


def test_promote_precision():
    assert promote_types(f32, f64) is f64
    assert promote_types(i32, i64) is i64
