"""Tests for expression-level dtype annotation (Layer 1 of type system refactor).

Validates that ir_type_annotate sets dtype on every expression IR node,
not just _resolved_type on IRAssign. These tests exercise the annotation
pass directly on IR trees, independent of any backend.
"""

import pytest
from tack.lang import ir
from tack.lang.types import f32, f64, i32, i64, u32
from tack.lang.ir_type_annotate import annotate_types


def _make_func(params, body):
    return ir.IRFunction("test", params, body)


def _field_param(name, dtype):
    p = ir.IRParam(name, dtype)
    p._is_field = True
    return p


def _scalar_param(name, dtype):
    p = ir.IRParam(name, dtype)
    p._is_field = False
    return p


# --- Constants ---

class TestConstantDtype:
    def test_float_constant(self):
        c = ir.IRConstant(3.14)
        func = _make_func([], [ir.IRAssign("x", c)])
        annotate_types(func)
        assert c.dtype is f32

    def test_int_constant(self):
        c = ir.IRConstant(42)
        func = _make_func([], [ir.IRAssign("x", c)])
        annotate_types(func)
        assert c.dtype is i32

    def test_large_int_constant(self):
        c = ir.IRConstant(2**31)
        func = _make_func([], [ir.IRAssign("x", c)])
        annotate_types(func)
        assert c.dtype is i64

    def test_negative_large_int(self):
        c = ir.IRConstant(-(2**31) - 1)
        func = _make_func([], [ir.IRAssign("x", c)])
        annotate_types(func)
        assert c.dtype is i64

    def test_zero(self):
        c = ir.IRConstant(0)
        func = _make_func([], [ir.IRAssign("x", c)])
        annotate_types(func)
        assert c.dtype is i32


# --- Variable references ---

class TestNameDtype:
    def test_scalar_param_i32(self):
        p = _scalar_param("n", i32)
        name = ir.IRName("n")
        func = _make_func([p], [ir.IRAssign("x", name)])
        annotate_types(func)
        assert name.dtype is i32

    def test_scalar_param_f32(self):
        p = _scalar_param("t", f32)
        name = ir.IRName("t")
        func = _make_func([p], [ir.IRAssign("x", name)])
        annotate_types(func)
        assert name.dtype is f32

    def test_field_param_returns_none(self):
        """Field (pointer) params should get dtype=None — they aren't scalars."""
        p = _field_param("data", f32)
        name = ir.IRName("data")
        func = _make_func([p], [ir.IRAssign("alias", name)])
        annotate_types(func)
        assert name.dtype is None

    def test_propagated_from_prior_assign(self):
        """A name should pick up the type from a preceding assignment."""
        p = _field_param("data", f32)
        load = ir.IRFieldLoad(ir.IRName("data"), ir.IRConstant(0))
        a1 = ir.IRAssign("val", load)
        name = ir.IRName("val")
        a2 = ir.IRAssign("copy", name)
        func = _make_func([p], [a1, a2])
        annotate_types(func)
        assert name.dtype is f32


# --- Field loads ---

class TestFieldLoadDtype:
    def test_f32_field(self):
        p = _field_param("data", f32)
        load = ir.IRFieldLoad(ir.IRName("data"), ir.IRConstant(0))
        func = _make_func([p], [ir.IRAssign("v", load)])
        annotate_types(func)
        assert load.dtype is f32

    def test_i32_field(self):
        p = _field_param("idx", i32)
        load = ir.IRFieldLoad(ir.IRName("idx"), ir.IRConstant(0))
        func = _make_func([p], [ir.IRAssign("v", load)])
        annotate_types(func)
        assert load.dtype is i32

    def test_f64_field(self):
        p = _field_param("data", f64)
        load = ir.IRFieldLoad(ir.IRName("data"), ir.IRConstant(0))
        func = _make_func([p], [ir.IRAssign("v", load)])
        annotate_types(func)
        assert load.dtype is f64

    def test_index_gets_annotated(self):
        """The index sub-expression inside a field load should also get dtype."""
        p = _field_param("data", f32)
        idx = ir.IRConstant(5)
        load = ir.IRFieldLoad(ir.IRName("data"), idx)
        func = _make_func([p], [ir.IRAssign("v", load)])
        annotate_types(func)
        assert idx.dtype is i32


# --- Binary ops ---

class TestBinOpDtype:
    def test_int_plus_int(self):
        binop = ir.IRBinOp("+", ir.IRConstant(1), ir.IRConstant(2))
        func = _make_func([], [ir.IRAssign("r", binop)])
        annotate_types(func)
        assert binop.dtype is i32

    def test_float_plus_int(self):
        binop = ir.IRBinOp("+", ir.IRConstant(1.0), ir.IRConstant(2))
        func = _make_func([], [ir.IRAssign("r", binop)])
        annotate_types(func)
        assert binop.dtype is f32

    def test_i32_plus_i64(self):
        binop = ir.IRBinOp("+", ir.IRConstant(1), ir.IRConstant(2**31))
        func = _make_func([], [ir.IRAssign("r", binop)])
        annotate_types(func)
        assert binop.dtype is i64

    def test_f32_times_f32_field(self):
        p = _field_param("data", f32)
        load = ir.IRFieldLoad(ir.IRName("data"), ir.IRConstant(0))
        binop = ir.IRBinOp("*", load, ir.IRConstant(2.0))
        func = _make_func([p], [ir.IRAssign("r", binop)])
        annotate_types(func)
        assert binop.dtype is f32

    def test_nested_binops(self):
        """(1 + 2) * 3.0 → f32"""
        inner = ir.IRBinOp("+", ir.IRConstant(1), ir.IRConstant(2))
        outer = ir.IRBinOp("*", inner, ir.IRConstant(3.0))
        func = _make_func([], [ir.IRAssign("r", outer)])
        annotate_types(func)
        assert inner.dtype is i32
        assert outer.dtype is f32

    def test_bitwise_op_stays_int(self):
        binop = ir.IRBinOp("&", ir.IRConstant(0xFF), ir.IRConstant(0x0F))
        func = _make_func([], [ir.IRAssign("r", binop)])
        annotate_types(func)
        assert binop.dtype is i32


# --- Unary ops ---

class TestUnaryOpDtype:
    def test_negate_float(self):
        unary = ir.IRUnaryOp("-", ir.IRConstant(1.5))
        func = _make_func([], [ir.IRAssign("r", unary)])
        annotate_types(func)
        assert unary.dtype is f32

    def test_negate_int(self):
        unary = ir.IRUnaryOp("-", ir.IRConstant(3))
        func = _make_func([], [ir.IRAssign("r", unary)])
        annotate_types(func)
        assert unary.dtype is i32

    def test_bitwise_not(self):
        unary = ir.IRUnaryOp("~", ir.IRConstant(0xFF))
        func = _make_func([], [ir.IRAssign("r", unary)])
        annotate_types(func)
        assert unary.dtype is i32


# --- Comparisons and boolean ops ---

class TestCompareAndBoolDtype:
    def test_compare_is_i32(self):
        cmp = ir.IRCompare(">", ir.IRConstant(1), ir.IRConstant(0))
        func = _make_func([], [ir.IRAssign("r", cmp)])
        annotate_types(func)
        assert cmp.dtype is i32

    def test_compare_subexprs_annotated(self):
        left = ir.IRConstant(3.14)
        right = ir.IRConstant(2.0)
        cmp = ir.IRCompare("<", left, right)
        func = _make_func([], [ir.IRAssign("r", cmp)])
        annotate_types(func)
        assert left.dtype is f32
        assert right.dtype is f32
        assert cmp.dtype is i32

    def test_boolop_is_i32(self):
        boolop = ir.IRBoolOp("and", [
            ir.IRCompare(">", ir.IRConstant(1), ir.IRConstant(0)),
            ir.IRCompare("<", ir.IRConstant(2), ir.IRConstant(3)),
        ])
        func = _make_func([], [ir.IRAssign("r", boolop)])
        annotate_types(func)
        assert boolop.dtype is i32


# --- Casts ---

class TestCastDtype:
    def test_cast_to_int(self):
        cast = ir.IRCast(ir.IRConstant(3.14), i32)
        assign = ir.IRAssign("r", cast)
        func = _make_func([], [assign])
        annotate_types(func)
        assert assign._resolved_type is i32
        assert cast.dtype is i32

    def test_cast_to_float(self):
        cast = ir.IRCast(ir.IRConstant(42), f32)
        assign = ir.IRAssign("r", cast)
        func = _make_func([], [assign])
        annotate_types(func)
        assert assign._resolved_type is f32
        assert cast.dtype is f32

    def test_cast_to_f64(self):
        cast = ir.IRCast(ir.IRConstant(1.0), f64)
        assign = ir.IRAssign("r", cast)
        func = _make_func([], [assign])
        annotate_types(func)
        assert assign._resolved_type is f64
        assert cast.dtype is f64

    def test_cast_to_i64(self):
        cast = ir.IRCast(ir.IRConstant(42), i64)
        assign = ir.IRAssign("r", cast)
        func = _make_func([], [assign])
        annotate_types(func)
        assert assign._resolved_type is i64
        assert cast.dtype is i64

    def test_cast_to_u32(self):
        cast = ir.IRCast(ir.IRConstant(42), u32)
        assign = ir.IRAssign("r", cast)
        func = _make_func([], [assign])
        annotate_types(func)
        assert assign._resolved_type is u32

    def test_cast_value_annotated(self):
        """The sub-expression inside a cast should also get dtype."""
        inner = ir.IRConstant(42)
        cast = ir.IRCast(inner, f32)
        func = _make_func([], [ir.IRAssign("r", cast)])
        annotate_types(func)
        assert inner.dtype is i32


# --- Function calls (math builtins) ---

class TestCallDtype:
    def test_sqrt_is_f32(self):
        call = ir.IRCall("sqrt", [ir.IRConstant(2.0)])
        func = _make_func([], [ir.IRAssign("r", call)])
        annotate_types(func)
        assert call.dtype is f32

    def test_math_args_annotated(self):
        arg = ir.IRConstant(2.0)
        call = ir.IRCall("sqrt", [arg])
        func = _make_func([], [ir.IRAssign("r", call)])
        annotate_types(func)
        assert arg.dtype is f32

    def test_abs_preserves_int(self):
        call = ir.IRCall("abs", [ir.IRConstant(-5)])
        func = _make_func([], [ir.IRAssign("r", call)])
        annotate_types(func)
        assert call.dtype is i32

    def test_min_promotes(self):
        call = ir.IRCall("min", [ir.IRConstant(1), ir.IRConstant(2.0)])
        func = _make_func([], [ir.IRAssign("r", call)])
        annotate_types(func)
        assert call.dtype is f32

    def test_max_int_stays_int(self):
        call = ir.IRCall("max", [ir.IRConstant(1), ir.IRConstant(2)])
        func = _make_func([], [ir.IRAssign("r", call)])
        annotate_types(func)
        assert call.dtype is i32


# --- Ternary (if-expression) ---

class TestIfExpDtype:
    def test_same_type(self):
        ifexp = ir.IRIfExp(
            ir.IRCompare(">", ir.IRConstant(1), ir.IRConstant(0)),
            ir.IRConstant(1.0),
            ir.IRConstant(2.0),
        )
        func = _make_func([], [ir.IRAssign("r", ifexp)])
        annotate_types(func)
        assert ifexp.dtype is f32

    def test_promotes_branches(self):
        """int ? float : int → f32"""
        ifexp = ir.IRIfExp(
            ir.IRCompare(">", ir.IRConstant(1), ir.IRConstant(0)),
            ir.IRConstant(1.0),
            ir.IRConstant(2),
        )
        func = _make_func([], [ir.IRAssign("r", ifexp)])
        annotate_types(func)
        assert ifexp.dtype is f32

    def test_condition_annotated(self):
        cond = ir.IRCompare(">", ir.IRConstant(1), ir.IRConstant(0))
        ifexp = ir.IRIfExp(cond, ir.IRConstant(1.0), ir.IRConstant(2.0))
        func = _make_func([], [ir.IRAssign("r", ifexp)])
        annotate_types(func)
        assert cond.dtype is i32


# --- ThreadId ---

class TestThreadIdDtype:
    def test_thread_id_is_i32(self):
        tid = ir.IRThreadId()
        func = _make_func([], [ir.IRAssign("t", tid)])
        annotate_types(func)
        assert tid.dtype is i32


# --- Loop variable types ---

class TestLoopVarDtype:
    def test_parallel_for_loop_var(self):
        """Parallel loop variable should be i64 (supports large grids)."""
        name = ir.IRName("i")
        assign = ir.IRAssign("x", name)
        pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(100), [assign])
        func = _make_func([], [pfor])
        annotate_types(func)
        assert name.dtype is i64

    def test_sequential_for_loop_var(self):
        name = ir.IRName("j")
        assign = ir.IRAssign("x", name)
        sfor = ir.IRSequentialFor("j", ir.IRConstant(0), ir.IRConstant(10), [assign])
        func = _make_func([], [sfor])
        annotate_types(func)
        assert name.dtype is i64


# --- Type promotion through complex expressions ---

class TestPromotionChains:
    def test_field_load_plus_int_constant(self):
        """f32_field[i] + 1 → f32"""
        p = _field_param("data", f32)
        load = ir.IRFieldLoad(ir.IRName("data"), ir.IRConstant(0))
        binop = ir.IRBinOp("+", load, ir.IRConstant(1))
        func = _make_func([p], [ir.IRAssign("r", binop)])
        annotate_types(func)
        assert load.dtype is f32
        assert binop.dtype is f32

    def test_i32_field_load_stays_int(self):
        """i32_field[i] + 1 → i32"""
        p = _field_param("idx", i32)
        load = ir.IRFieldLoad(ir.IRName("idx"), ir.IRConstant(0))
        binop = ir.IRBinOp("+", load, ir.IRConstant(1))
        func = _make_func([p], [ir.IRAssign("r", binop)])
        annotate_types(func)
        assert load.dtype is i32
        assert binop.dtype is i32

    def test_multi_step_promotion(self):
        """i32 → (+ i64) → i64 → (* f32) → f32"""
        p = _scalar_param("n", i32)
        # n + large_const → i64
        add = ir.IRBinOp("+", ir.IRName("n"), ir.IRConstant(2**31))
        a1 = ir.IRAssign("big", add)
        # big * 1.0 → f32
        mul = ir.IRBinOp("*", ir.IRName("big"), ir.IRConstant(1.0))
        a2 = ir.IRAssign("result", mul)
        func = _make_func([p], [a1, a2])
        annotate_types(func)
        assert add.dtype is i64
        assert mul.dtype is f32


# --- Expressions inside control flow ---

class TestControlFlowAnnotation:
    def test_if_condition_annotated(self):
        cond = ir.IRCompare(">", ir.IRConstant(1), ir.IRConstant(0))
        if_node = ir.IRIf(cond, [ir.IRAssign("x", ir.IRConstant(1.0))], [])
        func = _make_func([], [if_node])
        annotate_types(func)
        assert cond.dtype is i32

    def test_while_condition_annotated(self):
        cond = ir.IRCompare("<", ir.IRConstant(0), ir.IRConstant(10))
        while_node = ir.IRWhile(cond, [])
        func = _make_func([], [while_node])
        annotate_types(func)
        assert cond.dtype is i32

    def test_nested_if_assigns(self):
        """Assignments in both branches of an if should get annotated."""
        then_val = ir.IRConstant(1.0)
        else_val = ir.IRConstant(2)
        then_assign = ir.IRAssign("x", then_val)
        else_assign = ir.IRAssign("x", else_val)
        if_node = ir.IRIf(
            ir.IRCompare(">", ir.IRConstant(1), ir.IRConstant(0)),
            [then_assign], [else_assign],
        )
        func = _make_func([], [if_node])
        annotate_types(func)
        assert then_val.dtype is f32
        assert else_val.dtype is i32

    def test_field_store_subexprs_annotated(self):
        """Index and value in a field store should get dtype."""
        p = _field_param("out", f32)
        idx = ir.IRConstant(0)
        val = ir.IRBinOp("+", ir.IRConstant(1.0), ir.IRConstant(2.0))
        store = ir.IRFieldStore(ir.IRName("out"), idx, val)
        func = _make_func([p], [store])
        annotate_types(func)
        assert idx.dtype is i32
        assert val.dtype is f32


# --- Atomic ops ---

class TestAtomicOpDtype:
    def test_atomic_add_f32(self):
        p = _field_param("buf", f32)
        atomic = ir.IRAtomicOp("add", ir.IRName("buf"), ir.IRConstant(0), ir.IRConstant(1.0))
        func = _make_func([p], [atomic])
        annotate_types(func)
        assert atomic.dtype is f32

    def test_atomic_add_i32(self):
        p = _field_param("buf", i32)
        atomic = ir.IRAtomicOp("add", ir.IRName("buf"), ir.IRConstant(0), ir.IRConstant(1))
        func = _make_func([p], [atomic])
        annotate_types(func)
        assert atomic.dtype is i32


# --- End-to-end: build a realistic kernel IR and check all dtypes ---

class TestEndToEnd:
    def test_saxpy_pattern(self):
        """y[i] = a * x[i] + y[i]  — all expression nodes should be f32."""
        x = _field_param("x", f32)
        y = _field_param("y", f32)
        a = _scalar_param("a", f32)

        # a * x[i]
        xi_load = ir.IRFieldLoad(ir.IRName("x"), ir.IRName("i"))
        ax = ir.IRBinOp("*", ir.IRName("a"), xi_load)
        # y[i]
        yi_load = ir.IRFieldLoad(ir.IRName("y"), ir.IRName("i"))
        # a*x[i] + y[i]
        add = ir.IRBinOp("+", ax, yi_load)
        # y[i] = ...
        store = ir.IRFieldStore(ir.IRName("y"), ir.IRName("i"), add)
        pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(1000), [store])

        func = _make_func([x, y, a], [pfor])
        annotate_types(func)

        assert xi_load.dtype is f32
        assert ax.dtype is f32
        assert yi_load.dtype is f32
        assert add.dtype is f32

    def test_index_arithmetic_stays_int(self):
        """idx = i * width + j — should stay i64 (loop vars are i64)."""
        p = _field_param("data", f32)
        w = _scalar_param("width", i32)

        i_name = ir.IRName("i")
        j_name = ir.IRName("j")
        w_name = ir.IRName("width")
        mul = ir.IRBinOp("*", i_name, w_name)
        add = ir.IRBinOp("+", mul, j_name)
        assign = ir.IRAssign("idx", add)

        inner_loop = ir.IRSequentialFor("j", ir.IRConstant(0), ir.IRConstant(10), [assign])
        outer_loop = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(100), [inner_loop])

        func = _make_func([p, w], [outer_loop])
        annotate_types(func)

        assert i_name.dtype is i64
        assert mul.dtype is i64  # i64 * i32 → i64
        assert add.dtype is i64  # i64 + i64 → i64
