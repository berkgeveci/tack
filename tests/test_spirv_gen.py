"""Tests for SPIR-V code generation from PGC IR."""

import ast
import shutil
import subprocess
import tempfile
import textwrap

import pytest

from pgc.lang.ast_transform import transform_kernel
from pgc.lang import ir
from pgc.lang.types import f32
from pgc.codegen.spirv_gen import generate_spirv

_has_spirv_val = shutil.which("spirv-val") is not None
_has_spirv_dis = shutil.which("spirv-dis") is not None
_has_spirv_cross = shutil.which("spirv-cross") is not None

_needs_spirv_val = pytest.mark.skipif(not _has_spirv_val, reason="spirv-val not installed")
_needs_spirv_dis = pytest.mark.skipif(not _has_spirv_dis, reason="spirv-dis not installed")
_needs_spirv_cross = pytest.mark.skipif(not _has_spirv_cross, reason="spirv-cross not installed")


def _gen(source: str, param_types=None) -> bytes:
    """Helper: transform source to IR, set param types, generate SPIR-V binary."""
    tree = ast.parse(textwrap.dedent(source))
    module = transform_kernel(tree)
    func = module.functions[0]

    if param_types is None:
        param_types = [f32] * len(func.params)

    for param, ptype in zip(func.params, param_types):
        param.type_annotation = ptype

    return generate_spirv(func)


def _validate_spirv(spirv_bytes: bytes) -> str:
    """Validate SPIR-V binary using spirv-val. Returns validation output."""
    with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as f:
        f.write(spirv_bytes)
        f.flush()
        result = subprocess.run(
            ["spirv-val", f.name],
            capture_output=True, text=True
        )
        return result.returncode, result.stdout + result.stderr


def _disassemble_spirv(spirv_bytes: bytes) -> str:
    """Disassemble SPIR-V binary using spirv-dis."""
    with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as f:
        f.write(spirv_bytes)
        f.flush()
        result = subprocess.run(
            ["spirv-dis", f.name],
            capture_output=True, text=True
        )
        return result.stdout


# --- Basic generation ---

def test_vector_add_generates():
    spirv = _gen("""
        def add(x, y, out):
            for i in range(10):
                out[i] = x[i] + y[i]
    """)
    assert len(spirv) > 0
    # Check SPIR-V magic number
    assert spirv[:4] == b"\x03\x02\x23\x07"


@_needs_spirv_dis
def test_vector_add_disassembles():
    spirv = _gen("""
        def add(x, y, out):
            for i in range(10):
                out[i] = x[i] + y[i]
    """)
    dis = _disassemble_spirv(spirv)
    assert "OpEntryPoint GLCompute" in dis
    assert "OpExecutionMode" in dis
    assert "OpFAdd" in dis
    assert "OpAccessChain" in dis
    print(dis)


@_needs_spirv_val
def test_vector_add_validates():
    spirv = _gen("""
        def add(x, y, out):
            for i in range(10):
                out[i] = x[i] + y[i]
    """)
    rc, output = _validate_spirv(spirv)
    if rc != 0:
        # Print disassembly for debugging
        print(_disassemble_spirv(spirv))
        print("Validation errors:", output)
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_val
def test_saxpy_validates():
    spirv = _gen("""
        def saxpy(x, y, out):
            for i in range(10):
                out[i] = 2.0 * x[i] + y[i]
    """)
    rc, output = _validate_spirv(spirv)
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_val
def test_conditional_validates():
    spirv = _gen("""
        def kern(x, out):
            for i in range(10):
                if x[i] > 0.0:
                    out[i] = x[i]
                else:
                    out[i] = 0.0
    """)
    rc, output = _validate_spirv(spirv)
    if rc != 0:
        print(_disassemble_spirv(spirv))
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_val
def test_negation_validates():
    spirv = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = -x[i]
    """)
    rc, output = _validate_spirv(spirv)
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_dis
@_needs_spirv_val
def test_math_sqrt_validates():
    spirv = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = sqrt(x[i])
    """)
    dis = _disassemble_spirv(spirv)
    assert "OpExtInst" in dis
    rc, output = _validate_spirv(spirv)
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_val
def test_math_sin_cos_validates():
    spirv = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = sin(x[i])
    """)
    rc, output = _validate_spirv(spirv)
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_dis
@_needs_spirv_val
def test_min_max_validates():
    spirv = _gen("""
        def kern(x, y, out):
            for i in range(10):
                out[i] = min(x[i], y[i])
    """)
    dis = _disassemble_spirv(spirv)
    assert "OpExtInst" in dis
    rc, output = _validate_spirv(spirv)
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_val
def test_abs_validates():
    spirv = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = abs(x[i])
    """)
    rc, output = _validate_spirv(spirv)
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_val
def test_pow_validates():
    spirv = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = x[i] ** 2.0
    """)
    rc, output = _validate_spirv(spirv)
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_val
def test_multiple_ops_validates():
    """Tests subtraction, multiplication, division all in one kernel."""
    spirv = _gen("""
        def kern(a, b, c, out):
            for i in range(10):
                out[i] = (a[i] - b[i]) * c[i] / 2.0
    """)
    rc, output = _validate_spirv(spirv)
    assert rc == 0, f"SPIR-V validation failed: {output}"


@_needs_spirv_dis
def test_storage_buffer_bindings():
    """Verify each field gets its own binding."""
    spirv = _gen("""
        def kern(a, b, c):
            for i in range(10):
                c[i] = a[i] + b[i]
    """)
    dis = _disassemble_spirv(spirv)
    assert "Binding 0" in dis
    assert "Binding 1" in dis
    assert "Binding 2" in dis


@_needs_spirv_dis
def test_workgroup_size():
    spirv = _gen("""
        def kern(x, out):
            for i in range(10):
                out[i] = x[i]
    """)
    dis = _disassemble_spirv(spirv)
    # Default workgroup size is 256
    assert "LocalSize 256 1 1" in dis


@_needs_spirv_cross
@_needs_spirv_val
@_needs_spirv_dis
def test_spirv_cross_to_msl():
    """Verify the SPIR-V can be converted to Metal Shading Language."""
    spirv = _gen("""
        def add(x, y, out):
            for i in range(10):
                out[i] = x[i] + y[i]
    """)
    rc, _ = _validate_spirv(spirv)
    assert rc == 0

    with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as f:
        f.write(spirv)
        f.flush()
        result = subprocess.run(
            ["spirv-cross", "--msl", f.name],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("spirv-cross error:", result.stderr)
            print(_disassemble_spirv(spirv))
        assert result.returncode == 0, f"spirv-cross failed: {result.stderr}"
        msl = result.stdout
        assert "kernel void" in msl or "void main" in msl or "compute" in msl.lower()
        print(msl)
