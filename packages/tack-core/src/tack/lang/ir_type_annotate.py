"""IR pass — annotates all IR nodes with resolved types.

Walks the IR after type inference and propagates types from parameters
and expressions to all nodes. Every expression node gets a `dtype`
attribute (a ScalarType), and every IRAssign gets a `_resolved_type`.
Codegen backends can then read `node.dtype` directly instead of
re-implementing type inference heuristics.

A local variable gets **one** type for the whole function, computed as the
promotion of every type assigned to it. Backends give each local a single
storage slot — an `alloca` on CPU, a declared C variable elsewhere — so a
type that drifted statement by statement would silently narrow every store
after the first. That is how `total = 0.0` followed by adds from an f64
field used to accumulate in f32.

Must run after type inference (needs _is_field and type_annotation on params).
"""

from tack.lang import ir
from tack.lang.type_inference import promote_types
from tack.lang.types import ScalarType, f32, f64, i32, i64

# The join is monotone (types only widen), so it settles in a couple of
# rounds. The cap is a backstop against a pathological IR, not a budget.
_MAX_JOIN_ROUNDS = 8


def annotate_types(ir_func: ir.IRFunction):
    """Annotate all IR nodes with resolved types.

    Mutates ir_func in place. After this pass:
    - Every expression node has a `dtype` attribute (a ScalarType)
    - Every IRAssign has a `_resolved_type` attribute
    """
    # Build initial type environment from parameters
    base_env = {}  # var_name → ScalarType
    field_params = set()  # names of field (pointer) parameters
    for param in ir_func.params:
        base_env[param.name] = param.type_annotation
        if getattr(param, '_is_field', False):
            field_params.add(param.name)

    # Names whose type is declared elsewhere and must not be widened:
    # parameters, loop variables, and shared/local allocations.
    pinned = set(base_env) | _collect_pinned(ir_func.body)

    # Fixpoint over the assignments: a variable's type is the promotion of
    # every type assigned to it. Assignments can read other locals, so this
    # iterates until nothing widens.
    var_types = {}
    for _ in range(_MAX_JOIN_ROUNDS):
        env = dict(base_env)
        env.update(var_types)
        collected = {}
        _annotate_body(ir_func.body, env, field_params, var_types, collected)
        for name in pinned:
            collected.pop(name, None)
        if collected == var_types:
            break
        var_types = collected

    # Final walk with the settled types, so every read of a variable sees
    # the same type its storage slot will have.
    env = dict(base_env)
    env.update(var_types)
    _annotate_body(ir_func.body, env, field_params, var_types, None)


def _collect_pinned(stmts, out=None):
    """Names bound by a loop or an explicit allocation, not by assignment."""
    if out is None:
        out = set()
    for stmt in stmts:
        if isinstance(stmt, (ir.IRParallelFor, ir.IRSequentialFor)):
            out.add(stmt.var)
            _collect_pinned(stmt.body, out)
        elif isinstance(stmt, ir.IRWhile):
            _collect_pinned(stmt.body, out)
        elif isinstance(stmt, ir.IRIf):
            _collect_pinned(stmt.then_body, out)
            if stmt.else_body:
                _collect_pinned(stmt.else_body, out)
        elif isinstance(stmt, (ir.IRSharedAlloc, ir.IRLocalAlloc)):
            out.add(stmt.name)
    return out


def _join(current, new):
    """Promote two candidate types for one variable, widest wins."""
    if current is None:
        return new
    if new is None or current is new:
        return current
    try:
        return promote_types(current, new)
    except TypeError:
        # Unpromotable pair (shouldn't happen for scalars) — keep the first
        # type rather than guess, matching the old first-assignment-wins.
        return current


def _annotate_expr(node, env, field_params) -> ScalarType | None:
    """Infer and set the dtype of an IR expression node.

    Returns the ScalarType (also sets node.dtype as side effect).
    Returns None for field references (pointer types).
    """
    if isinstance(node, ir.IRConstant):
        if node.dtype is not None:
            # Already annotated (e.g., by AST transform)
            return node.dtype
        if isinstance(node.value, float):
            node.dtype = f32
        elif isinstance(node.value, int):
            val = node.value
            if val > 2**31 - 1 or val < -(2**31):
                node.dtype = i64
            else:
                node.dtype = i32
        else:
            node.dtype = i32
        return node.dtype

    if isinstance(node, ir.IRName):
        if node.name in field_params:
            return None  # Field pointer — not a scalar
        dtype = env.get(node.name, i32)
        node.dtype = dtype
        return dtype

    if isinstance(node, ir.IRFieldLoad):
        # Annotate sub-expressions
        _annotate_expr(node.index, env, field_params)
        # Type comes from the field's element type
        field_name = _get_field_name(node.field)
        if field_name and field_name in env:
            node.dtype = env[field_name]
        else:
            node.dtype = f32  # fallback
        return node.dtype

    if isinstance(node, ir.IRBinOp):
        lt = _annotate_expr(node.left, env, field_params)
        rt = _annotate_expr(node.right, env, field_params)
        if lt is None or rt is None:
            return None
        node.dtype = promote_types(lt, rt)
        return node.dtype

    if isinstance(node, ir.IRUnaryOp):
        t = _annotate_expr(node.operand, env, field_params)
        node.dtype = t
        return t

    if isinstance(node, ir.IRCall):
        # Annotate arguments
        arg_types = []
        for arg in node.args:
            t = _annotate_expr(arg, env, field_params)
            if t is not None:
                arg_types.append(t)
        # Math builtins (sqrt, sin, etc.) preserve float type
        # If any argument is f64, result is f64; otherwise f32
        if any(t is f64 for t in arg_types):
            node.dtype = f64
        else:
            node.dtype = f32
        # Integer-returning builtins
        if node.func_name in ("abs",) and arg_types and arg_types[0] in (i32, i64):
            node.dtype = arg_types[0]
        if node.func_name in ("min", "max") and arg_types:
            node.dtype = arg_types[0]
            for t in arg_types[1:]:
                node.dtype = promote_types(node.dtype, t)
        return node.dtype

    if isinstance(node, ir.IRCast):
        _annotate_expr(node.value, env, field_params)
        # dtype is a ScalarType (i32, f32, f64, etc.)
        if isinstance(node.dtype, ScalarType):
            return node.dtype
        # Legacy string fallback (should not happen after Layer 2)
        if node.dtype == "int":
            return i32
        if node.dtype == "float":
            return f32
        return f32

    if isinstance(node, ir.IRIfExp):
        _annotate_expr(node.condition, env, field_params)
        tt = _annotate_expr(node.then_value, env, field_params)
        et = _annotate_expr(node.else_value, env, field_params)
        if tt is None or et is None:
            return None
        node.dtype = promote_types(tt, et)
        return node.dtype

    if isinstance(node, ir.IRCompare):
        _annotate_expr(node.left, env, field_params)
        _annotate_expr(node.right, env, field_params)
        node.dtype = i32  # comparisons always produce int
        return i32

    if isinstance(node, ir.IRBoolOp):
        for v in node.values:
            _annotate_expr(v, env, field_params)
        node.dtype = i32  # boolean ops always produce int
        return i32

    if isinstance(node, ir.IRTextureSample):
        for c in node.coords:
            _annotate_expr(c, env, field_params)
        node.dtype = f32  # texture samples return float
        return f32

    if isinstance(node, ir.IRThreadId):
        node.dtype = i32
        return i32

    if isinstance(node, ir.IRAttribute):
        # e.g., field.shape[0] — integer
        node.dtype = i32
        return i32

    if isinstance(node, ir.IRAtomicOp):
        _annotate_expr(node.index, env, field_params)
        _annotate_expr(node.value, env, field_params)
        # Atomic ops return the field's element type
        field_name = _get_field_name(node.field)
        if field_name and field_name in env:
            node.dtype = env[field_name]
        else:
            node.dtype = f32
        return node.dtype

    if isinstance(node, ir.IRBlockReduce):
        t = _annotate_expr(node.value, env, field_params)
        node.dtype = t if t is not None else f32
        return node.dtype

    if isinstance(node, ir.IRDimSize):
        # DimSize returns an integer (dimension size)
        return i64

    # Fallback
    return f32


def _get_field_name(node) -> str | None:
    if isinstance(node, ir.IRName):
        return node.name
    return None


def _annotate_body(stmts, env, field_params, var_types=None, collected=None):
    """Walk a list of statements, annotating all nodes and updating env."""
    for stmt in stmts:
        _annotate_stmt(stmt, env, field_params, var_types, collected)


def _annotate_stmt(node, env, field_params, var_types=None, collected=None):
    """Annotate a single statement and all its sub-expressions.

    `var_types` holds the settled per-variable join; when a target is in it,
    that type wins over this statement's own RHS type. `collected` is the
    accumulator for the join pre-pass — None on the final walk.
    """
    if node is None:
        return

    if isinstance(node, ir.IRAssign):
        resolved = _annotate_expr(node.value, env, field_params)
        declared = var_types.get(node.target) if var_types else None
        if collected is not None and resolved is not None:
            collected[node.target] = _join(collected.get(node.target), resolved)
        if declared is not None:
            # One storage slot per variable — every assignment uses its type.
            node._resolved_type = declared
            env[node.target] = declared
        else:
            node._resolved_type = resolved  # None means "don't override codegen"
            if resolved is not None:
                env[node.target] = resolved
        return

    if isinstance(node, ir.IRParallelFor):
        # Loop variable is always integer (i64 on GPU for large ranges)
        env[node.var] = i64
        _annotate_expr(node.start, env, field_params)
        _annotate_expr(node.end, env, field_params)
        _annotate_body(node.body, env, field_params, var_types, collected)
        return

    if isinstance(node, ir.IRSequentialFor):
        env[node.var] = i64
        _annotate_expr(node.start, env, field_params)
        _annotate_expr(node.end, env, field_params)
        if node.step:
            _annotate_expr(node.step, env, field_params)
        _annotate_body(node.body, env, field_params, var_types, collected)
        return

    if isinstance(node, ir.IRWhile):
        _annotate_expr(node.condition, env, field_params)
        _annotate_body(node.body, env, field_params, var_types, collected)
        return

    if isinstance(node, ir.IRIf):
        _annotate_expr(node.condition, env, field_params)
        _annotate_body(node.then_body, env, field_params, var_types, collected)
        if node.else_body:
            _annotate_body(node.else_body, env, field_params, var_types, collected)
        return

    if isinstance(node, ir.IRFieldStore):
        _annotate_expr(node.index, env, field_params)
        _annotate_expr(node.value, env, field_params)
        return

    if isinstance(node, ir.IRAtomicOp):
        _annotate_expr(node, env, field_params)
        return

    if isinstance(node, ir.IRPrint):
        for arg in node.args:
            _annotate_expr(arg, env, field_params)
        return

    if isinstance(node, ir.IRReturn):
        if node.value is not None:
            _annotate_expr(node.value, env, field_params)
        return

    if isinstance(node, ir.IRSharedAlloc):
        if isinstance(node.dtype, ScalarType):
            env[node.name] = node.dtype
        return

    if isinstance(node, ir.IRLocalAlloc):
        if isinstance(node.dtype, ScalarType):
            env[node.name] = node.dtype
        return

    if isinstance(node, (ir.IRBreak, ir.IRContinue, ir.IRBarrier)):
        return
