"""IR pass — annotates local variables with resolved types.

Walks the IR after type inference and propagates types from parameters
and expressions to all local variable assignments. Each IRAssign gets
a `_resolved_type` attribute (a ScalarType), so codegen can emit the
correct C type without fragile heuristics.

Must run after type inference (needs _is_field and type_annotation on params).
"""

from pgc.lang import ir
from pgc.lang.types import ScalarType, f32, i32, i64


def annotate_types(ir_func: ir.IRFunction):
    """Annotate all IRAssign nodes with resolved types.

    Mutates ir_func in place. After this pass, every IRAssign has a
    `_resolved_type` attribute (a ScalarType), or None if the type
    cannot be reliably determined (e.g., field pointer assignments).
    """
    # Build initial type environment from parameters
    env = {}  # var_name → ScalarType
    field_params = set()  # names of field (pointer) parameters
    for param in ir_func.params:
        env[param.name] = param.type_annotation
        if getattr(param, '_is_field', False):
            field_params.add(param.name)

    # Walk the body and annotate assignments
    _annotate_body(ir_func.body, env, field_params)


def _infer_expr_type(node, env, field_params) -> ScalarType | None:
    """Infer the ScalarType of an IR expression given the type environment.

    Returns None if the expression is a field reference (pointer type),
    which cannot be represented as a simple ScalarType.
    """
    if isinstance(node, ir.IRConstant):
        if isinstance(node.value, float):
            return f32
        if isinstance(node.value, int):
            val = node.value
            if val > 2**31 - 1 or val < -(2**31):
                return i64
            return i32
        return i32

    if isinstance(node, ir.IRName):
        if node.name in field_params:
            return None  # Field pointer — cannot type as scalar
        return env.get(node.name, i32)

    if isinstance(node, ir.IRFieldLoad):
        # Type comes from the field's element type
        field_name = _get_field_name(node.field)
        if field_name and field_name in env:
            return env[field_name]
        # Recurse into index to handle packed scalar loads
        return f32

    if isinstance(node, ir.IRBinOp):
        lt = _infer_expr_type(node.left, env, field_params)
        rt = _infer_expr_type(node.right, env, field_params)
        if lt is None or rt is None:
            return None
        return _promote(lt, rt)

    if isinstance(node, ir.IRUnaryOp):
        return _infer_expr_type(node.operand, env, field_params)

    if isinstance(node, ir.IRCall):
        # Math builtins (sqrt, sin, etc.) return float
        return f32

    if isinstance(node, ir.IRCast):
        if node.dtype == "int":
            return i32
        if node.dtype == "float":
            return f32
        return f32

    if isinstance(node, ir.IRIfExp):
        tt = _infer_expr_type(node.then_value, env, field_params)
        et = _infer_expr_type(node.else_value, env, field_params)
        if tt is None or et is None:
            return None
        return _promote(tt, et)

    if isinstance(node, ir.IRCompare):
        return i32

    if isinstance(node, ir.IRBoolOp):
        return i32

    if isinstance(node, ir.IRTextureSample):
        return f32

    if isinstance(node, ir.IRThreadId):
        return i32

    if isinstance(node, ir.IRAttribute):
        # e.g., field.shape[0] — integer
        return i32

    # Default fallback
    return f32


def _promote(a: ScalarType, b: ScalarType) -> ScalarType:
    """Type promotion: f32 wins over integers, i64 wins over i32."""
    if a is f32 or b is f32:
        return f32
    if a is i64 or b is i64:
        return i64
    return a


def _get_field_name(node) -> str | None:
    if isinstance(node, ir.IRName):
        return node.name
    return None


def _annotate_body(stmts, env, field_params):
    """Walk a list of statements, annotating assignments and updating env."""
    for stmt in stmts:
        _annotate_stmt(stmt, env, field_params)


def _annotate_stmt(node, env, field_params):
    """Annotate a single statement."""
    if node is None:
        return

    if isinstance(node, ir.IRAssign):
        resolved = _infer_expr_type(node.value, env, field_params)
        node._resolved_type = resolved  # None means "don't override codegen"
        if resolved is not None:
            env[node.target] = resolved
        return

    if isinstance(node, ir.IRParallelFor):
        # Loop variable is always integer (i64 on GPU for large ranges)
        env[node.var] = i64
        _annotate_body(node.body, env, field_params)
        return

    if isinstance(node, ir.IRSequentialFor):
        env[node.var] = i64
        _annotate_body(node.body, env, field_params)
        return

    if isinstance(node, ir.IRWhile):
        _annotate_body(node.body, env, field_params)
        return

    if isinstance(node, ir.IRIf):
        _annotate_body(node.then_body, env, field_params)
        if node.else_body:
            _annotate_body(node.else_body, env, field_params)
        return

    if isinstance(node, ir.IRFieldStore):
        return

    if isinstance(node, ir.IRAtomicOp):
        return

    if isinstance(node, ir.IRPrint):
        return

    if isinstance(node, ir.IRReturn):
        return

    if isinstance(node, ir.IRSharedAlloc):
        if hasattr(node, 'dtype_annotation'):
            env[node.name] = node.dtype_annotation
        return

    if isinstance(node, ir.IRLocalAlloc):
        if hasattr(node, 'dtype_annotation'):
            env[node.name] = node.dtype_annotation
        return

    if isinstance(node, (ir.IRBreak, ir.IRContinue, ir.IRBarrier)):
        return
