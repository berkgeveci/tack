"""Tests for LLVM IR code generation from PGC IR."""

import ast
import textwrap

from pgc.lang.ast_transform import transform_kernel
from pgc.lang import ir
from pgc.lang.types import f32, f64, i32
from pgc.codegen.llvm_gen import generate_llvm_ir


def _gen(source: str, param_types=None) -> str:
    """Helper: transform source to IR, set param types, generate LLVM IR string."""
    tree = ast.parse(textwrap.dedent(source))
    module = transform_kernel(tree)
    func = module.functions[0]

    # Default: all params are f32 fields
    if param_types is None:
        param_types = [f32] * len(func.params)

    for param, ptype in zip(func.params, param_types):
        param.type_annotation = ptype

    llvm_module = generate_llvm_ir(func)
    return str(llvm_module)


def test_vector_add_generates():
    ll = _gen("""
        def add(x, y, out):
            for i in range(10):
                out[i] = x[i] + y[i]
    """)
    assert "define void @\"add\"" in ll
    assert "fadd float" in ll
    assert "getelementptr" in ll
    assert "load float" in ll
    assert "store float" in ll


def test_nested_loops():
    ll = _gen("""
        def kern(a, b):
            for i in range(10):
                for j in range(10):
                    b[i] = a[i]
    """)
    # Should have two loop headers
    assert "for.i.header" in ll
    assert "for.j.header" in ll


def test_if_else():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                if x[i] > 0.0:
                    out[i] = x[i]
                else:
                    out[i] = 0.0
    """)
    assert "if.then" in ll
    assert "if.else" in ll
    assert "if.merge" in ll
    assert "fcmp ogt" in ll or "fcmp olt" in ll or "fcmp" in ll


def test_while_loop():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                j = 0
                while j < 10:
                    out[i] = out[i] + x[i]
                    j = j + 1
    """)
    assert "while.header" in ll
    assert "while.body" in ll


def test_math_sqrt():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = sqrt(x[i])
    """)
    assert "llvm.sqrt.f32" in ll


def test_math_sin_cos():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = sin(x[i])
    """)
    assert "llvm.sin.f32" in ll


def test_min_max():
    ll = _gen("""
        def kern(x, y, out):
            for i in range(10):
                out[i] = min(x[i], y[i])
    """)
    assert "llvm.minnum.f32" in ll


def test_abs_float():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = abs(x[i])
    """)
    assert "llvm.fabs.f32" in ll


def test_negation():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = -x[i]
    """)
    assert "fsub float" in ll


def test_comparison_ops():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                if x[i] < 0.0:
                    out[i] = 0.0
                if x[i] >= 1.0:
                    out[i] = 1.0
    """)
    assert "fcmp" in ll


def test_augmented_assignment():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] += x[i]
    """)
    assert "fadd" in ll
    assert "store" in ll


def test_local_variable():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                tmp = x[i] * 2.0
                out[i] = tmp
    """)
    assert "alloca" in ll
    assert "fmul" in ll


def test_pow_intrinsic():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = x[i] ** 2.0
    """)
    assert "llvm.pow.f32" in ll


def test_floor_div_float():
    ll = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = x[i] // 2.0
    """)
    assert "llvm.floor.f32" in ll


def test_saxpy():
    """SAXPY: out[i] = alpha * x[i] + y[i] — validates scalar + field mixing."""
    ll = _gen("""
        def saxpy(x, y, out):
            for i in range(10):
                out[i] = 2.0 * x[i] + y[i]
    """)
    assert "fmul" in ll
    assert "fadd" in ll


def test_mandelbrot_like():
    """Complex kernel with nested loops, while, if, break, and math."""
    ll = _gen("""
        def mandelbrot(pixels):
            for i in range(800):
                for j in range(600):
                    cx = -2.0 + 3.0 * 0.00125
                    cy = -1.5 + 3.0 * 0.00167
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
    assert "define void" in ll
    assert "while.header" in ll
    assert "if.then" in ll
    # Verify it generates valid LLVM IR (no assertion errors)


def test_generates_valid_llvm_ir():
    """Verify that the generated LLVM IR can be parsed by llvmlite."""
    from llvmlite import binding as llvm

    ll = _gen("""
        def add(x, y, out):
            for i in range(10):
                out[i] = x[i] + y[i]
    """)

    # Parse the LLVM IR — this will raise if invalid
    mod = llvm.parse_assembly(ll)
    mod.verify()
