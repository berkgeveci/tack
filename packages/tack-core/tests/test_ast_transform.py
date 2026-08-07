"""Tests for the AST transformer — validates Python AST → Tack IR conversion."""

import ast
import textwrap

import pytest

from tack.lang import ir
from tack.lang.ast_transform import transform_kernel


def _transform(source: str) -> ir.IRModule:
    """Helper: transform a dedented source string into IR."""
    tree = ast.parse(textwrap.dedent(source))
    return transform_kernel(tree)


# --- Basic structure ---

def test_function_params():
    module = _transform("""
        def add(x, y, out):
            for i in range(10):
                out[i] = x[i] + y[i]
    """)
    assert len(module.functions) == 1
    func = module.functions[0]
    assert func.name == "add"
    assert [p.name for p in func.params] == ["x", "y", "out"]


# --- Parallel vs Sequential for ---

def test_top_level_for_is_parallel():
    module = _transform("""
        def kern(x):
            for i in range(10):
                x[i] = 0.0
    """)
    stmt = module.functions[0].body[0]
    assert isinstance(stmt, ir.IRParallelFor)
    assert stmt.var == "i"


def test_nested_for_is_sequential():
    module = _transform("""
        def kern(x):
            for i in range(10):
                for j in range(10):
                    x[i] = 0.0
    """)
    outer = module.functions[0].body[0]
    assert isinstance(outer, ir.IRParallelFor)
    inner = outer.body[0]
    assert isinstance(inner, ir.IRSequentialFor)
    assert inner.var == "j"


def test_range_two_args():
    module = _transform("""
        def kern(x):
            for i in range(1, 10):
                x[i] = 0.0
    """)
    loop = module.functions[0].body[0]
    assert isinstance(loop.start, ir.IRConstant)
    assert loop.start.value == 1


# --- Binary ops ---

def test_binop_add():
    module = _transform("""
        def kern(x, y, out):
            for i in range(10):
                out[i] = x[i] + y[i]
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store, ir.IRFieldStore)
    assert isinstance(store.value, ir.IRBinOp)
    assert store.value.op == "+"


def test_binop_all_ops():
    module = _transform("""
        def kern(a, b, c):
            for i in range(1):
                c[i] = a[i] - b[i]
                c[i] = a[i] * b[i]
                c[i] = a[i] / b[i]
                c[i] = a[i] // b[i]
                c[i] = a[i] % b[i]
                c[i] = a[i] ** b[i]
    """)
    stmts = module.functions[0].body[0].body
    ops = [s.value.op for s in stmts]
    assert ops == ["-", "*", "/", "//", "%", "**"]


def test_bitwise_ops():
    module = _transform("""
        def kern(a, b, c):
            for i in range(1):
                c[i] = a[i] & b[i]
                c[i] = a[i] | b[i]
                c[i] = a[i] ^ b[i]
                c[i] = a[i] << b[i]
                c[i] = a[i] >> b[i]
    """)
    stmts = module.functions[0].body[0].body
    ops = [s.value.op for s in stmts]
    assert ops == ["&", "|", "^", "<<", ">>"]


# --- Unary ops ---

def test_unary_neg():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                out[i] = -x[i]
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store.value, ir.IRUnaryOp)
    assert store.value.op == "-"


def test_unary_not():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                if not x[i]:
                    out[i] = 1.0
    """)
    if_node = module.functions[0].body[0].body[0]
    assert isinstance(if_node.condition, ir.IRUnaryOp)
    assert if_node.condition.op == "not"


# --- Comparisons ---

def test_compare_ops():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                if x[i] < 0.0:
                    out[i] = 0.0
    """)
    if_node = module.functions[0].body[0].body[0]
    assert isinstance(if_node, ir.IRIf)
    assert isinstance(if_node.condition, ir.IRCompare)
    assert if_node.condition.op == "<"


def test_chained_compare():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                if 0.0 < x[i] < 1.0:
                    out[i] = x[i]
    """)
    if_node = module.functions[0].body[0].body[0]
    cond = if_node.condition
    # chained a < b < c  →  BoolOp(and, [a < b, b < c])
    assert isinstance(cond, ir.IRBoolOp)
    assert cond.op == "and"
    assert len(cond.values) == 2
    assert all(isinstance(v, ir.IRCompare) for v in cond.values)


# --- Boolean ops ---

def test_bool_and():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                if x[i] > 0.0 and x[i] < 1.0:
                    out[i] = x[i]
    """)
    if_node = module.functions[0].body[0].body[0]
    cond = if_node.condition
    assert isinstance(cond, ir.IRBoolOp)
    assert cond.op == "and"


def test_bool_or():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                if x[i] < 0.0 or x[i] > 1.0:
                    out[i] = 0.0
    """)
    cond = module.functions[0].body[0].body[0].condition
    assert isinstance(cond, ir.IRBoolOp)
    assert cond.op == "or"


# --- If/else ---

def test_if_else():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                if x[i] > 0.0:
                    out[i] = x[i]
                else:
                    out[i] = 0.0
    """)
    if_node = module.functions[0].body[0].body[0]
    assert isinstance(if_node, ir.IRIf)
    assert len(if_node.then_body) == 1
    assert len(if_node.else_body) == 1


def test_if_elif_else():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                if x[i] > 1.0:
                    out[i] = 1.0
                elif x[i] > 0.0:
                    out[i] = x[i]
                else:
                    out[i] = 0.0
    """)
    if_node = module.functions[0].body[0].body[0]
    assert isinstance(if_node, ir.IRIf)
    # elif becomes nested if in else_body
    assert len(if_node.else_body) == 1
    assert isinstance(if_node.else_body[0], ir.IRIf)


def test_ifexp_ternary():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                out[i] = x[i] if x[i] > 0.0 else 0.0
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store.value, ir.IRIfExp)


# --- While loop ---

def test_while_loop():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                j = 0
                while j < 10:
                    out[i] = out[i] + x[i]
                    j = j + 1
    """)
    body = module.functions[0].body[0].body
    assert isinstance(body[0], ir.IRAssign)
    assert isinstance(body[1], ir.IRWhile)
    assert isinstance(body[1].condition, ir.IRCompare)


# --- Break / Continue ---

def test_break():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                j = 0
                while j < 100:
                    if x[i] > 0.0:
                        break
                    j = j + 1
    """)
    while_body = module.functions[0].body[0].body[1].body
    if_node = while_body[0]
    assert isinstance(if_node.then_body[0], ir.IRBreak)


def test_continue():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                if x[i] < 0.0:
                    continue
                out[i] = x[i]
    """)
    body = module.functions[0].body[0].body
    if_node = body[0]
    assert isinstance(if_node.then_body[0], ir.IRContinue)


# --- Function calls (math builtins) ---

def test_math_sqrt():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                out[i] = sqrt(x[i])
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store.value, ir.IRCall)
    assert store.value.func_name == "sqrt"
    assert len(store.value.args) == 1


def test_math_module_call():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                out[i] = math.sin(x[i])
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store.value, ir.IRCall)
    assert store.value.func_name == "sin"


def test_math_min_max():
    module = _transform("""
        def kern(x, y, out):
            for i in range(10):
                out[i] = min(x[i], y[i])
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store.value, ir.IRCall)
    assert store.value.func_name == "min"
    assert len(store.value.args) == 2


def test_abs_call():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                out[i] = abs(x[i])
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store.value, ir.IRCall)
    assert store.value.func_name == "abs"


def test_pow_call():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                out[i] = pow(x[i], 2.0)
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store.value, ir.IRCall)
    assert store.value.func_name == "pow"


# --- Type casts ---

def test_int_cast():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                out[i] = int(x[i])
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store.value, ir.IRCast)
    from tack.lang.types import i32 as _i32
    assert store.value.dtype is _i32


def test_float_cast():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                out[i] = float(x[i])
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store.value, ir.IRCast)
    from tack.lang.types import f32 as _f32
    assert store.value.dtype is _f32


# --- Augmented assignment ---

def test_augassign_field():
    module = _transform("""
        def kern(x, out):
            for i in range(10):
                out[i] += x[i]
    """)
    store = module.functions[0].body[0].body[0]
    assert isinstance(store, ir.IRFieldStore)
    assert isinstance(store.value, ir.IRBinOp)
    assert store.value.op == "+"


def test_augassign_variable():
    module = _transform("""
        def kern(x):
            for i in range(10):
                total = 0.0
                total += x[i]
    """)
    body = module.functions[0].body[0].body
    assert isinstance(body[0], ir.IRAssign)
    assert isinstance(body[1], ir.IRAssign)
    assert isinstance(body[1].value, ir.IRBinOp)


# --- Attribute access ---

def test_attribute_shape():
    """x.shape[k] becomes IRDimSize, the node the resolve pass folds."""
    module = _transform("""
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = x[i]
    """)
    loop = module.functions[0].body[0]
    assert isinstance(loop.end, ir.IRDimSize)
    assert loop.end.field_name == "x"
    assert loop.end.dim == 0


def test_attribute_shape_in_the_body():
    """It is the same node wherever it appears — not just the loop bound."""
    module = _transform("""
        def kern(x, out):
            for i in range(x.shape[0]):
                out[i] = x[x.shape[0] - 1 - i]
    """)
    store = module.functions[0].body[0].body[0]
    # x[x.shape[0] - 1 - i] → index is ((DimSize - 1) - i)
    index = store.value.index
    assert isinstance(index.left.left, ir.IRDimSize)
    assert index.left.left.field_name == "x"
    assert index.left.left.dim == 0


def test_len_becomes_dim_size():
    module = _transform("""
        def kern(x, out):
            for i in range(len(x)):
                out[i] = float(len(x))
    """)
    loop = module.functions[0].body[0]
    assert isinstance(loop.end, ir.IRDimSize)
    assert (loop.end.field_name, loop.end.dim) == ("x", 0)
    inner = loop.body[0].value.value    # float(...) wraps the DimSize
    assert isinstance(inner, ir.IRDimSize)


def test_non_constant_shape_index_is_rejected():
    """The dimension has to be known when the kernel is compiled."""
    with pytest.raises(NotImplementedError, match="constant dimension"):
        _transform("""
            def kern(x, out, d):
                for i in range(x.shape[d]):
                    out[i] = x[i]
        """)


# --- Complex kernels ---

def test_mandelbrot_structure():
    """Mandelbrot-like kernel exercises nested loops, if/else, comparisons, math."""
    module = _transform("""
        def mandelbrot(pixels, max_iter):
            for i in range(800):
                for j in range(600):
                    cx = -2.0 + 3.0 * i / 800.0
                    cy = -1.5 + 3.0 * j / 600.0
                    zx = 0.0
                    zy = 0.0
                    count = 0
                    while count < 100:
                        if zx * zx + zy * zy > 4.0:
                            break
                        nx = zx * zx - zy * zy + cx
                        ny = 2.0 * zx * zy + cy
                        zx = nx
                        zy = ny
                        count = count + 1
    """)
    func = module.functions[0]
    assert func.name == "mandelbrot"

    outer = func.body[0]
    assert isinstance(outer, ir.IRParallelFor)
    assert outer.var == "i"

    inner = outer.body[0]
    assert isinstance(inner, ir.IRSequentialFor)
    assert inner.var == "j"

    # Should have assignments then a while loop
    while_node = None
    for stmt in inner.body:
        if isinstance(stmt, ir.IRWhile):
            while_node = stmt
    assert while_node is not None
    assert isinstance(while_node.condition, ir.IRCompare)


# --- IR pretty printer ---

def test_ir_dump():
    module = _transform("""
        def add(x, y, out):
            for i in range(10):
                out[i] = x[i] + y[i]
    """)
    text = ir.dump(module)
    assert "Module:" in text
    assert "Function add(x, y, out):" in text
    assert "ParallelFor i" in text
