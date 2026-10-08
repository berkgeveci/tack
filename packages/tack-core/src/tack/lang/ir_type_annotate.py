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

Float literals are weakly typed, like Python scalars under NumPy's NEP 50.
A *literal expression* is built only from numeric literals with unary
`+`/`-`, binary arithmetic, math builtins and conditional-expression arms;
one containing a float literal is *weak*. It is annotated f32, but when it
meets a non-weak floating operand, an explicit floating cast, or a floating
store or assignment target, it takes that type: each float literal in it
converts once, from its exact Python value, and its floating operations run
at that precision. `x_f64 * 0.1` therefore uses the f64 nearest 0.1, while
every f32 expression keeps the types it had before.

Must run after type inference (needs _is_field and type_annotation on params).
"""

from tack.lang import ir
from tack.lang.ir_traversal import walk_ir
from tack.lang.type_inference import promote_types
from tack.lang.types import INTEGER_TYPES, ScalarType, f32, f64, i32, i64, integer_type_for_value

_FLOAT_TYPES = (f32, f64)

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

    # A local assigned only float literals, more than once (a device
    # function returning literals from several branches; once is already
    # the literal, see ir_optimize), takes the kernel's float precision:
    # f64 when a field is f64, as float scalar arguments do.
    float_context = f64 if any(base_env[p] is f64 for p in field_params) else f32
    literal_only = (_literal_only_locals(ir_func.body) - pinned
                    if float_context is f64 else set())

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
        for name in literal_only:
            collected[name] = float_context
        if collected == var_types:
            break
        var_types = collected

    # Final walk with the settled types, so every read of a variable sees
    # the same type its storage slot will have.
    env = dict(base_env)
    env.update(var_types)
    _annotate_body(ir_func.body, env, field_params, var_types, None)


def _literal_only_locals(stmts):
    """Locals every one of whose assignments is a weak literal expression."""
    from tack.lang.ir_optimize import _literal
    weak = {}
    for node in walk_ir(stmts):
        if isinstance(node, ir.IRAssign):
            literal, has_float = _literal(node.value, any_condition=True)
            weak[node.target] = weak.get(node.target, True) and literal and has_float
    return {name for name, only in weak.items() if only}


def _collect_pinned(stmts):
    """Names bound by a loop or an explicit allocation, not by assignment."""
    out = set()
    for node in walk_ir(stmts):
        if isinstance(node, (ir.IRParallelFor, ir.IRSequentialFor)):
            out.add(node.var)
            out.update(getattr(node, 'dims', None) or ())
        elif isinstance(node, (ir.IRSharedAlloc, ir.IRLocalAlloc)):
            out.add(node.name)
    return out


def _join(current, new):
    """Promote two candidate types for one variable, widest wins."""
    if current is None:
        return new
    if new is None or current is new:
        return current
    return promote_types(current, new)


def _annotate_expr(node, env, field_params) -> ScalarType | None:
    """Infer and set the dtype of an IR expression node.

    Returns the ScalarType (also sets node.dtype as side effect).
    Returns None for field references (pointer types).
    """
    if isinstance(node, ir.IRConstant):
        node._literal = isinstance(node.value, (int, float))
        node._weak = isinstance(node.value, float)
        if node._weak:
            # Reset on every walk: an earlier walk may have retyped it.
            node.dtype = f32
            return node.dtype
        if node.dtype is not None:
            # Already annotated (e.g., by AST transform)
            return node.dtype
        if isinstance(node.value, int):
            node.dtype = integer_type_for_value(node.value)
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
        lt, rt = _meet(node.left, lt, node.right, rt)
        _mark_literal(node, (node.left, node.right))
        if node.op == '/' and lt in INTEGER_TYPES and rt in INTEGER_TYPES:
            node.dtype = f32
        elif node.op == '**' and lt in INTEGER_TYPES and rt in INTEGER_TYPES:
            _check_integer_exponent(node.right)
            node.dtype = lt
        else:
            node.dtype = lt if node.op in ('<<', '>>') else promote_types(lt, rt)
        return node.dtype

    if isinstance(node, ir.IRUnaryOp):
        t = _annotate_expr(node.operand, env, field_params)
        node.dtype = i32 if node.op == 'not' else t
        if node.op in ('+', '-'):
            _mark_literal(node, (node.operand,))
        else:
            node._literal = node._weak = False
        return node.dtype

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
        if node.func_name in ("abs",) and arg_types and arg_types[0] in INTEGER_TYPES:
            node.dtype = arg_types[0]
        if node.func_name in ("min", "max") and arg_types:
            node.dtype = arg_types[0]
            for t in arg_types[1:]:
                node.dtype = promote_types(node.dtype, t)
        if node.func_name == 'pow' and len(arg_types) == 2 \
                and all(t in INTEGER_TYPES for t in arg_types):
            _check_integer_exponent(node.args[1])
            node.dtype = arg_types[0]
        _mark_literal(node, node.args)
        if not node._weak and node.dtype in _FLOAT_TYPES:
            for arg in node.args:
                _adopt(arg, node.dtype)
        return node.dtype

    if isinstance(node, ir.IRCast):
        _annotate_expr(node.value, env, field_params)
        # dtype is a ScalarType (i32, f32, f64, etc.)
        if isinstance(node.dtype, ScalarType):
            # tack.f64(0.1) converts the literal directly, never via f32.
            _adopt(node.value, node.dtype)
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
        tt, et = _meet(node.then_value, tt, node.else_value, et)
        _mark_literal(node, (node.then_value, node.else_value))
        node.dtype = promote_types(tt, et)
        return node.dtype

    if isinstance(node, ir.IRCompare):
        lt = _annotate_expr(node.left, env, field_params)
        rt = _annotate_expr(node.right, env, field_params)
        lt, rt = _meet(node.left, lt, node.right, rt)
        node._operand_type = promote_types(lt, rt)
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
            _adopt(node.value, node.dtype)
        else:
            node.dtype = f32
        return node.dtype

    if isinstance(node, ir.IRBlockReduce):
        t = _annotate_expr(node.value, env, field_params)
        if t is not f32:
            raise TypeError(f'block_{node.op} requires f32 input; use an explicit f32 cast')
        node.dtype = f32
        return node.dtype

    if isinstance(node, ir.IRDimSize):
        # DimSize is retained only in the host-evaluated grid bound.
        node.dtype = i64
        return i64

    # Fallback
    return f32


def _mark_literal(node, operands):
    """Record whether node is a literal expression, and whether it is weak."""
    node._literal = all(getattr(o, '_literal', False) for o in operands)
    node._weak = node._literal and any(getattr(o, '_weak', False) for o in operands)


def _meet(left, lt, right, rt):
    """Give a weak operand the floating type of a non-weak partner.

    Returns the operand types after the conversion. Two weak operands, or a
    weak operand beside an integer, keep the f32 default.
    """
    left_weak = getattr(left, '_weak', False)
    right_weak = getattr(right, '_weak', False)
    if left_weak and not right_weak and _adopt(left, rt):
        return rt, rt
    if right_weak and not left_weak and _adopt(right, lt):
        return lt, lt
    return lt, rt


def _adopt(node, dtype) -> bool:
    """Convert a weak literal expression to a floating dtype; True if converted."""
    if not getattr(node, '_weak', False) or dtype not in _FLOAT_TYPES:
        return False
    _retype(node, dtype)
    return True


def _retype(node, dtype):
    """Move every floating node of a literal expression to dtype.

    Integer subexpressions keep their own types; the enclosing operation
    converts them, as it would any integer operand.
    """
    if node.dtype not in _FLOAT_TYPES:
        return
    node.dtype = dtype
    if isinstance(node, ir.IRUnaryOp):
        children = (node.operand,)
    elif isinstance(node, ir.IRBinOp):
        children = (node.left, node.right)
    elif isinstance(node, ir.IRCall):
        children = node.args
    elif isinstance(node, ir.IRIfExp):
        children = (node.then_value, node.else_value)
    else:
        children = ()
    for child in children:
        _retype(child, dtype)


def _get_field_name(node) -> str | None:
    if isinstance(node, ir.IRName):
        return node.name
    return None


def _check_integer_exponent(node):
    """Reject negative integer literals; dynamic exponents are a caller constraint."""
    if isinstance(node, ir.IRConstant) and isinstance(node.value, int) and node.value < 0:
        raise TypeError('Integer power requires a nonnegative exponent; '
                        'cast the base to a floating-point type for negative powers')


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
            _adopt(node.value, declared)
        else:
            node._resolved_type = resolved  # None means "don't override codegen"
            if resolved is not None:
                env[node.target] = resolved
        return

    if isinstance(node, ir.IRParallelFor):
        # Loop variable is always integer (i64 on GPU for large ranges)
        env[node.var] = i64
        for dim in node.dims or ():
            env[dim] = i64
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
        node.dtype = env.get(_get_field_name(node.field))
        _adopt(node.value, node.dtype)
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
        _annotate_expr(node.size, env, field_params)
        if isinstance(node.dtype, ScalarType):
            env[node.name] = node.dtype
        return

    if isinstance(node, ir.IRLocalAlloc):
        _annotate_expr(node.size, env, field_params)
        if isinstance(node.dtype, ScalarType):
            env[node.name] = node.dtype
        return

    if isinstance(node, (ir.IRBreak, ir.IRContinue, ir.IRBarrier)):
        return
