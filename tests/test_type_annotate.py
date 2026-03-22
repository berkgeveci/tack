"""Tests for the IR type annotation pass."""

import pytest
import pgc
from pgc.lang import ir
from pgc.lang.types import f32, i32, i64
from pgc.lang.type_inference import infer_param_types
from pgc.lang.ir_type_annotate import annotate_types


def _make_func(params, body):
    """Helper to create a simple IRFunction."""
    func = ir.IRFunction("test", params, body)
    return func


def test_assign_from_constant_float():
    p = ir.IRParam("x", f32)
    p._is_field = True
    assign = ir.IRAssign(target="a", value=ir.IRConstant(3.14))
    func = _make_func([p], [assign])
    annotate_types(func)
    assert assign._resolved_type is f32


def test_assign_from_constant_int():
    p = ir.IRParam("x", f32)
    p._is_field = True
    assign = ir.IRAssign(target="a", value=ir.IRConstant(42))
    func = _make_func([p], [assign])
    annotate_types(func)
    assert assign._resolved_type is i32


def test_assign_from_field_load():
    """Loading from an f32 field should give f32."""
    p = ir.IRParam("data", f32)
    p._is_field = True
    load = ir.IRFieldLoad(ir.IRName("data"), ir.IRConstant(0))
    assign = ir.IRAssign(target="val", value=load)
    func = _make_func([p], [assign])
    annotate_types(func)
    assert assign._resolved_type is f32


def test_assign_from_binop_promotes():
    """int + float → float."""
    p1 = ir.IRParam("x", f32)
    p1._is_field = True
    p2 = ir.IRParam("n", i32)
    p2._is_field = False
    binop = ir.IRBinOp(op="+", left=ir.IRName("n"), right=ir.IRConstant(1.0))
    assign = ir.IRAssign(target="result", value=binop)
    func = _make_func([p1, p2], [assign])
    annotate_types(func)
    assert assign._resolved_type is f32


def test_assign_from_cast_int():
    cast = ir.IRCast(value=ir.IRConstant(3.14), dtype=i32)
    assign = ir.IRAssign(target="idx", value=cast)
    func = _make_func([], [assign])
    annotate_types(func)
    assert assign._resolved_type is i32


def test_assign_from_cast_float():
    cast = ir.IRCast(value=ir.IRConstant(42), dtype=f32)
    assign = ir.IRAssign(target="val", value=cast)
    func = _make_func([], [assign])
    annotate_types(func)
    assert assign._resolved_type is f32


def test_assign_from_math_call():
    """Math builtins like sqrt return f32."""
    call = ir.IRCall(func_name="sqrt", args=[ir.IRConstant(2.0)])
    assign = ir.IRAssign(target="r", value=call)
    func = _make_func([], [assign])
    annotate_types(func)
    assert assign._resolved_type is f32


def test_field_pointer_not_annotated():
    """Assigning a field (pointer) should get None, not a scalar type."""
    p = ir.IRParam("data", f32)
    p._is_field = True
    assign = ir.IRAssign(target="alias", value=ir.IRName("data"))
    func = _make_func([p], [assign])
    annotate_types(func)
    assert assign._resolved_type is None


def test_type_propagation_chain():
    """Type propagates through a chain of assignments."""
    p = ir.IRParam("data", f32)
    p._is_field = True
    a1 = ir.IRAssign(target="a", value=ir.IRFieldLoad(ir.IRName("data"), ir.IRConstant(0)))
    a2 = ir.IRAssign(target="b", value=ir.IRBinOp(op="*", left=ir.IRName("a"), right=ir.IRConstant(2.0)))
    a3 = ir.IRAssign(target="c", value=ir.IRName("b"))
    func = _make_func([p], [a1, a2, a3])
    annotate_types(func)
    assert a1._resolved_type is f32
    assert a2._resolved_type is f32
    assert a3._resolved_type is f32


def test_if_body_annotated():
    """Assignments inside if-branches get annotated."""
    assign = ir.IRAssign(target="x", value=ir.IRConstant(1.0))
    if_node = ir.IRIf(
        condition=ir.IRCompare(op=">", left=ir.IRConstant(1), right=ir.IRConstant(0)),
        then_body=[assign],
        else_body=[],
    )
    func = _make_func([], [if_node])
    annotate_types(func)
    assert assign._resolved_type is f32
