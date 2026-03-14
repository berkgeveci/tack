"""PGC IR optimization passes — LICM and CSE.

These run after type inference and dimension resolution, before codegen.
"""

from pgc.lang import ir


def optimize_ir(ir_func: ir.IRFunction):
    """Run all optimization passes on an IR function (in-place)."""
    _licm_function(ir_func)
    _copy_prop_function(ir_func)
    _cse_function(ir_func)


# ---------------------------------------------------------------------------
# Loop-Invariant Code Motion (LICM)
# ---------------------------------------------------------------------------
#
# Hoist assignments whose RHS depends only on values defined outside the loop.
# This is critical for inlined @pgc.func calls that re-load field values
# every iteration (e.g. domain_min_f[0] loaded inside sample_at()).


def _licm_function(ir_func: ir.IRFunction):
    """Apply LICM to every loop in the function body (recursive)."""
    ir_func.body = _licm_body(ir_func.body)


def _licm_body(body: list) -> list:
    """Process a statement list, hoisting invariants out of any loops found."""
    result = []
    for stmt in body:
        if isinstance(stmt, ir.IRParallelFor):
            # Recurse into parallel for body first
            stmt.body = _licm_body(stmt.body)
            hoisted, stmt.body = _hoist_from_loop(stmt)
            result.extend(hoisted)
            result.append(stmt)
        elif isinstance(stmt, ir.IRSequentialFor):
            stmt.body = _licm_body(stmt.body)
            hoisted, stmt.body = _hoist_from_loop(stmt)
            result.extend(hoisted)
            result.append(stmt)
        elif isinstance(stmt, ir.IRWhile):
            stmt.body = _licm_body(stmt.body)
            hoisted, stmt.body = _hoist_from_loop(stmt)
            result.extend(hoisted)
            result.append(stmt)
        elif isinstance(stmt, ir.IRIf):
            stmt.then_body = _licm_body(stmt.then_body)
            if stmt.else_body:
                stmt.else_body = _licm_body(stmt.else_body)
            result.append(stmt)
        else:
            result.append(stmt)
    return result


def _hoist_from_loop(loop_node) -> tuple[list, list]:
    """Identify and hoist loop-invariant assignments from a loop body.

    Returns (hoisted_stmts, remaining_body).

    An assignment is safe to hoist only if:
    1. Its value is loop-invariant (doesn't depend on loop-modified variables)
    2. The target variable is assigned exactly once in the loop body
       (otherwise hoisting the initial assignment breaks re-assignment semantics)
    """
    # Collect the set of variables modified inside the loop body
    modified = _collect_modified_vars(loop_node.body)

    # Add the loop variable itself
    if hasattr(loop_node, 'var'):
        modified.add(loop_node.var)

    # Count how many times each variable is assigned in the loop
    assign_counts = _count_assignments(loop_node.body)

    # Iteratively hoist: each pass may expose new invariants
    body = loop_node.body
    all_hoisted = []
    changed = True
    while changed:
        changed = False
        new_body = []
        for stmt in body:
            if (isinstance(stmt, ir.IRAssign) and
                    assign_counts.get(stmt.target, 0) == 1 and
                    _is_invariant(stmt.value, modified)):
                # This assignment is loop-invariant and unique — hoist it
                all_hoisted.append(stmt)
                # Remove target from modified set (it's now defined outside)
                modified.discard(stmt.target)
                changed = True
            else:
                new_body.append(stmt)
        body = new_body

    return all_hoisted, body


def _count_assignments(body: list) -> dict[str, int]:
    """Count how many times each variable is assigned in a statement list (recursive)."""
    counts: dict[str, int] = {}
    for stmt in body:
        if isinstance(stmt, ir.IRAssign):
            counts[stmt.target] = counts.get(stmt.target, 0) + 1
        elif isinstance(stmt, ir.IRParallelFor):
            for k, v in _count_assignments(stmt.body).items():
                counts[k] = counts.get(k, 0) + v
        elif isinstance(stmt, ir.IRSequentialFor):
            for k, v in _count_assignments(stmt.body).items():
                counts[k] = counts.get(k, 0) + v
        elif isinstance(stmt, ir.IRWhile):
            for k, v in _count_assignments(stmt.body).items():
                counts[k] = counts.get(k, 0) + v
        elif isinstance(stmt, ir.IRIf):
            for k, v in _count_assignments(stmt.then_body).items():
                counts[k] = counts.get(k, 0) + v
            if stmt.else_body:
                for k, v in _count_assignments(stmt.else_body).items():
                    counts[k] = counts.get(k, 0) + v
    return counts


def _collect_modified_vars(body: list) -> set:
    """Collect all variable names assigned/modified in a statement list (recursive)."""
    modified = set()
    for stmt in body:
        if isinstance(stmt, ir.IRAssign):
            modified.add(stmt.target)
        elif isinstance(stmt, ir.IRParallelFor):
            modified.add(stmt.var)
            modified |= _collect_modified_vars(stmt.body)
        elif isinstance(stmt, ir.IRSequentialFor):
            modified.add(stmt.var)
            modified |= _collect_modified_vars(stmt.body)
        elif isinstance(stmt, ir.IRWhile):
            modified |= _collect_modified_vars(stmt.body)
        elif isinstance(stmt, ir.IRIf):
            modified |= _collect_modified_vars(stmt.then_body)
            if stmt.else_body:
                modified |= _collect_modified_vars(stmt.else_body)
        elif isinstance(stmt, ir.IRFieldStore):
            pass  # Field stores don't define local variables
    return modified


def _is_invariant(expr, modified: set) -> bool:
    """Check if an expression is loop-invariant (doesn't depend on modified vars).

    An expression is invariant if:
    - It's a constant
    - It's a name not in the modified set
    - It's a field load whose index is invariant
    - It's a binary/unary op whose operands are all invariant
    - It's a function call whose args are all invariant
    """
    if isinstance(expr, ir.IRConstant):
        return True

    if isinstance(expr, ir.IRName):
        return expr.name not in modified

    if isinstance(expr, ir.IRFieldLoad):
        return (_is_invariant(expr.field, modified) and
                _is_invariant(expr.index, modified))

    if isinstance(expr, ir.IRBinOp):
        return (_is_invariant(expr.left, modified) and
                _is_invariant(expr.right, modified))

    if isinstance(expr, ir.IRUnaryOp):
        return _is_invariant(expr.operand, modified)

    if isinstance(expr, ir.IRCompare):
        return (_is_invariant(expr.left, modified) and
                _is_invariant(expr.right, modified))

    if isinstance(expr, ir.IRBoolOp):
        return all(_is_invariant(v, modified) for v in expr.values)

    if isinstance(expr, ir.IRCall):
        return all(_is_invariant(a, modified) for a in expr.args)

    if isinstance(expr, ir.IRCast):
        return _is_invariant(expr.value, modified)

    if isinstance(expr, ir.IRIfExp):
        return (_is_invariant(expr.condition, modified) and
                _is_invariant(expr.then_value, modified) and
                _is_invariant(expr.else_value, modified))

    if isinstance(expr, ir.IRDimSize):
        return True  # Dimension sizes are constants

    # Unknown node type — assume not invariant (conservative)
    return False


# ---------------------------------------------------------------------------
# Copy Propagation
# ---------------------------------------------------------------------------
#
# When inlining @pgc.func, each call creates a fresh copy of every parameter:
#   __func_b_0__ = b
#   __func_b_1__ = b
# These prevent CSE from recognizing that func_b_0 and func_b_1 are the
# same value.  Copy propagation replaces uses of the copy with the original,
# enabling CSE to deduplicate subsequent expressions.


def _copy_prop_function(ir_func: ir.IRFunction):
    """Apply copy propagation to the function body."""
    ir_func.body = _copy_prop_body(ir_func.body)


def _copy_prop_body(body: list) -> list:
    """Propagate copies: when x = y (simple name assignment), replace
    subsequent uses of x with y (unless x or y is reassigned later)."""
    # First, count assignments to find single-assignment variables
    assign_counts = _count_assignments(body)

    # Collect simple copies: x = y where x is assigned exactly once
    # AND y is never assigned in this block (it comes from outside — a
    # parameter or outer scope).  This avoids breaking tuple swaps where
    # the source is reassigned later in the same block.
    copies = {}  # target -> source name
    for stmt in body:
        if (isinstance(stmt, ir.IRAssign) and
                isinstance(stmt.value, ir.IRName) and
                assign_counts.get(stmt.target, 0) == 1 and
                assign_counts.get(stmt.value.name, 0) == 0):
            copies[stmt.target] = stmt.value.name

    if not copies:
        # Still recurse into sub-blocks
        return _copy_prop_recurse(body)

    # Resolve transitive copies: if a→b and b→c, then a→c
    resolved = {}
    for target, source in copies.items():
        seen = {target}
        cur = source
        while cur in copies and cur not in seen:
            seen.add(cur)
            cur = copies[cur]
        resolved[target] = cur

    # Replace uses in the body
    result = []
    for stmt in body:
        if (isinstance(stmt, ir.IRAssign) and stmt.target in resolved):
            # Keep the copy assignment but with the resolved source
            result.append(ir.IRAssign(
                stmt.target, ir.IRName(resolved[stmt.target])
            ))
        else:
            result.append(_replace_names(stmt, resolved))

    return _copy_prop_recurse(result)


def _copy_prop_recurse(body: list) -> list:
    """Recurse copy propagation into loops and conditionals."""
    for stmt in body:
        if isinstance(stmt, (ir.IRParallelFor, ir.IRSequentialFor, ir.IRWhile)):
            stmt.body = _copy_prop_body(stmt.body)
        elif isinstance(stmt, ir.IRIf):
            stmt.then_body = _copy_prop_body(stmt.then_body)
            if stmt.else_body:
                stmt.else_body = _copy_prop_body(stmt.else_body)
    return body


def _replace_names(node, mapping: dict):
    """Replace variable names in an IR node according to the mapping."""
    if isinstance(node, ir.IRName):
        if node.name in mapping:
            return ir.IRName(mapping[node.name])
        return node

    if isinstance(node, ir.IRAssign):
        return ir.IRAssign(node.target, _replace_names(node.value, mapping))

    if isinstance(node, ir.IRFieldLoad):
        return ir.IRFieldLoad(
            _replace_names(node.field, mapping),
            _replace_names(node.index, mapping),
        )

    if isinstance(node, ir.IRFieldStore):
        return ir.IRFieldStore(
            _replace_names(node.field, mapping),
            _replace_names(node.index, mapping),
            _replace_names(node.value, mapping),
        )

    if isinstance(node, ir.IRBinOp):
        return ir.IRBinOp(
            node.op,
            _replace_names(node.left, mapping),
            _replace_names(node.right, mapping),
        )

    if isinstance(node, ir.IRUnaryOp):
        return ir.IRUnaryOp(node.op, _replace_names(node.operand, mapping))

    if isinstance(node, ir.IRCompare):
        return ir.IRCompare(
            node.op,
            _replace_names(node.left, mapping),
            _replace_names(node.right, mapping),
        )

    if isinstance(node, ir.IRBoolOp):
        return ir.IRBoolOp(
            node.op, [_replace_names(v, mapping) for v in node.values]
        )

    if isinstance(node, ir.IRCall):
        return ir.IRCall(
            node.func_name,
            [_replace_names(a, mapping) for a in node.args],
        )

    if isinstance(node, ir.IRCast):
        return ir.IRCast(_replace_names(node.value, mapping), node.dtype)

    if isinstance(node, ir.IRIfExp):
        return ir.IRIfExp(
            _replace_names(node.condition, mapping),
            _replace_names(node.then_value, mapping),
            _replace_names(node.else_value, mapping),
        )

    if isinstance(node, ir.IRIf):
        return ir.IRIf(
            _replace_names(node.condition, mapping),
            [_replace_names(s, mapping) for s in node.then_body],
            [_replace_names(s, mapping) for s in node.else_body] if node.else_body else [],
        )

    if isinstance(node, ir.IRParallelFor):
        new_node = ir.IRParallelFor(
            node.var,
            _replace_names(node.start, mapping),
            _replace_names(node.end, mapping),
            [_replace_names(s, mapping) for s in node.body],
        )
        return new_node

    if isinstance(node, ir.IRSequentialFor):
        new_node = ir.IRSequentialFor(
            node.var,
            _replace_names(node.start, mapping),
            _replace_names(node.end, mapping),
            [_replace_names(s, mapping) for s in node.body],
        )
        return new_node

    if isinstance(node, ir.IRWhile):
        return ir.IRWhile(
            _replace_names(node.condition, mapping),
            [_replace_names(s, mapping) for s in node.body],
        )

    if isinstance(node, ir.IRReturn):
        return ir.IRReturn(_replace_names(node.value, mapping))

    if isinstance(node, ir.IRAttribute):
        return ir.IRAttribute(_replace_names(node.obj, mapping), node.attr)

    # Constants, Break, Continue, DimSize — no names to replace
    return node


# ---------------------------------------------------------------------------
# Common Subexpression Elimination (CSE)
# ---------------------------------------------------------------------------
#
# When @pgc.func calls are inlined, the same field loads (e.g. block_cell_dims[b])
# appear multiple times with identical indices.  CSE replaces duplicate
# computations with references to the first result.
#
# Only pure expressions are candidates — field loads with the same field+index
# are safe because fields are read-only within a kernel (stores go to different
# fields).  We invalidate entries when a field store occurs or when the
# variables used in the expression key are reassigned.


def _cse_function(ir_func: ir.IRFunction):
    """Apply CSE to the function body (recursive into loops/ifs)."""
    ir_func.body = _cse_body(ir_func.body)


def _cse_body(body: list) -> list:
    """Run CSE on a flat statement list.

    Maintains a mapping from expression keys to variable names.
    When an assignment's RHS matches a known expression, replace
    RHS with IRName(existing_var).
    """
    available: dict[str, str] = {}  # expr_key -> variable name
    result = []
    for stmt in body:
        if isinstance(stmt, ir.IRAssign):
            key = _expr_key(stmt.value)
            if key is not None and key in available:
                # Replace with reference to existing variable
                stmt = ir.IRAssign(stmt.target, ir.IRName(available[key]))
                result.append(stmt)
            else:
                result.append(stmt)
                if key is not None:
                    available[key] = stmt.target
            # If target was used in any cached expression keys, those
            # are now stale.  Invalidate any entry whose key references
            # this variable.
            _invalidate_var(available, stmt.target)
        elif isinstance(stmt, ir.IRFieldStore):
            # A field store could invalidate field-load CSE entries for
            # the same field.  Conservative: invalidate all field loads
            # on the stored field.
            store_field_key = _expr_key(stmt.field)
            if store_field_key is not None:
                to_remove = [k for k in available
                             if k.startswith(f"load({store_field_key},")]
                for k in to_remove:
                    del available[k]
            result.append(stmt)
        elif isinstance(stmt, ir.IRParallelFor):
            stmt.body = _cse_body(stmt.body)
            result.append(stmt)
        elif isinstance(stmt, ir.IRSequentialFor):
            stmt.body = _cse_body(stmt.body)
            result.append(stmt)
        elif isinstance(stmt, ir.IRWhile):
            stmt.body = _cse_body(stmt.body)
            result.append(stmt)
        elif isinstance(stmt, ir.IRIf):
            stmt.then_body = _cse_body(stmt.then_body)
            if stmt.else_body:
                stmt.else_body = _cse_body(stmt.else_body)
            result.append(stmt)
        else:
            result.append(stmt)
    return result


def _invalidate_var(available: dict[str, str], var: str):
    """Remove entries from available that depend on a reassigned variable."""
    # Remove entries whose value variable is the one being reassigned,
    # and entries whose key contains this variable name as a dependency.
    to_remove = [k for k, v in available.items()
                 if v == var or f"name({var})" in k]
    for k in to_remove:
        del available[k]


def _expr_key(expr) -> str | None:
    """Compute a canonical string key for an expression, or None if not CSE-able.

    Two expressions with the same key compute the same value (assuming
    no intervening mutations).
    """
    if isinstance(expr, ir.IRConstant):
        return f"const({expr.value!r})"

    if isinstance(expr, ir.IRName):
        return f"name({expr.name})"

    if isinstance(expr, ir.IRFieldLoad):
        fk = _expr_key(expr.field)
        ik = _expr_key(expr.index)
        if fk is not None and ik is not None:
            return f"load({fk},{ik})"
        return None

    if isinstance(expr, ir.IRBinOp):
        lk = _expr_key(expr.left)
        rk = _expr_key(expr.right)
        if lk is not None and rk is not None:
            return f"binop({expr.op},{lk},{rk})"
        return None

    if isinstance(expr, ir.IRUnaryOp):
        ok = _expr_key(expr.operand)
        if ok is not None:
            return f"unary({expr.op},{ok})"
        return None

    if isinstance(expr, ir.IRCast):
        vk = _expr_key(expr.value)
        if vk is not None:
            return f"cast({vk},{expr.dtype})"
        return None

    if isinstance(expr, ir.IRCall):
        arg_keys = [_expr_key(a) for a in expr.args]
        if all(k is not None for k in arg_keys):
            args_str = ",".join(arg_keys)
            return f"call({expr.func_name},{args_str})"
        return None

    if isinstance(expr, ir.IRCompare):
        lk = _expr_key(expr.left)
        rk = _expr_key(expr.right)
        if lk is not None and rk is not None:
            return f"cmp({expr.op},{lk},{rk})"
        return None

    if isinstance(expr, ir.IRDimSize):
        return f"dimsize({expr.field_name},{expr.dim})"

    return None
