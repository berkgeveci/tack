"""IR resolution pass — resolves dispatch-time constants in the IR.

Runs after AST transform, before codegen. Replaces:
- IRDimSize(field_name, dim) → IRConstant(shape[dim])
  Uses _logical_shape for vector fields, shape for regular fields.
"""

from pgc.lang import ir


def resolve_ir(ir_func: ir.IRFunction, name_to_field: dict):
    """Resolve dimension sizes in IR using actual field shapes.

    Mutates ir_func.body in place.
    """
    # Build field alias map: inlined @pgc.func parameters create assignments
    # like __func_out_x0_0__ = out_x0, where out_x0 is a field. Track these
    # so shared_like/DimSize can resolve the mangled name to the actual field.
    aliases = _collect_field_aliases(ir_func.body, name_to_field)
    extended = {**name_to_field, **aliases}

    for i, stmt in enumerate(ir_func.body):
        ir_func.body[i] = _resolve(stmt, extended)


def _collect_field_aliases(stmts, known_fields):
    """Scan for IRAssign(target, IRName(field)) and build alias -> Field map."""
    aliases = {}
    for stmt in stmts:
        if isinstance(stmt, ir.IRAssign) and isinstance(stmt.value, ir.IRName):
            src = stmt.value.name
            if src in known_fields:
                aliases[stmt.target] = known_fields[src]
            elif src in aliases:
                aliases[stmt.target] = aliases[src]
        # Recurse into compound statements
        for attr in ('body', 'then_body', 'else_body'):
            children = getattr(stmt, attr, None)
            if isinstance(children, list):
                aliases.update(_collect_field_aliases(children, {**known_fields, **aliases}))
    return aliases


def _resolve(node, fields):
    """Recursively resolve dispatch-time constants in an IR node."""
    if node is None:
        return None

    if isinstance(node, ir.IRDimSize):
        field = fields.get(node.field_name)
        if field is None:
            raise RuntimeError(f"Cannot resolve dimension size: unknown field '{node.field_name}'")
        shape = getattr(field, '_logical_shape', None) or field.shape
        return ir.IRConstant(shape[node.dim])

    if isinstance(node, ir.IRBinOp):
        node.left = _resolve(node.left, fields)
        node.right = _resolve(node.right, fields)
        return node

    if isinstance(node, ir.IRUnaryOp):
        node.operand = _resolve(node.operand, fields)
        return node

    if isinstance(node, ir.IRCompare):
        node.left = _resolve(node.left, fields)
        node.right = _resolve(node.right, fields)
        return node

    if isinstance(node, ir.IRBoolOp):
        node.values = [_resolve(v, fields) for v in node.values]
        return node

    if isinstance(node, ir.IRFieldLoad):
        node.field = _resolve(node.field, fields)
        node.index = _resolve(node.index, fields)
        return node

    if isinstance(node, ir.IRFieldStore):
        node.field = _resolve(node.field, fields)
        node.index = _resolve(node.index, fields)
        node.value = _resolve(node.value, fields)
        return node

    if isinstance(node, ir.IRAtomicOp):
        node.field = _resolve(node.field, fields)
        node.index = _resolve(node.index, fields)
        node.value = _resolve(node.value, fields)
        return node

    if isinstance(node, ir.IRAssign):
        node.value = _resolve(node.value, fields)
        return node

    if isinstance(node, ir.IRParallelFor):
        node.start = _resolve(node.start, fields)
        node.end = _resolve(node.end, fields)
        node.body = [_resolve(s, fields) for s in node.body]
        return node

    if isinstance(node, ir.IRSequentialFor):
        node.start = _resolve(node.start, fields)
        node.end = _resolve(node.end, fields)
        if node.step is not None:
            node.step = _resolve(node.step, fields)
        node.body = [_resolve(s, fields) for s in node.body]
        return node

    if isinstance(node, ir.IRWhile):
        node.condition = _resolve(node.condition, fields)
        node.body = [_resolve(s, fields) for s in node.body]
        return node

    if isinstance(node, ir.IRIf):
        node.condition = _resolve(node.condition, fields)
        node.then_body = [_resolve(s, fields) for s in node.then_body]
        node.else_body = [_resolve(s, fields) for s in node.else_body]
        return node

    if isinstance(node, ir.IRIfExp):
        node.condition = _resolve(node.condition, fields)
        node.then_value = _resolve(node.then_value, fields)
        node.else_value = _resolve(node.else_value, fields)
        return node

    if isinstance(node, ir.IRCall):
        node.args = [_resolve(a, fields) for a in node.args]
        return node

    if isinstance(node, ir.IRCast):
        node.value = _resolve(node.value, fields)
        return node

    if isinstance(node, ir.IRSharedAlloc):
        node.size = _resolve(node.size, fields)
        # Resolve shared_like: fill dtype from the source field's type
        if node.dtype is None and node.field_name is not None:
            field = fields.get(node.field_name)
            if field is None:
                raise RuntimeError(
                    f"Cannot resolve shared_like: unknown field '{node.field_name}'")
            from pgc.lang.types import f32, f64, i32, i64, u32, u64
            _DTYPE_TO_C = {
                f32: "float", f64: "double", i32: "int", i64: "long",
                u32: "uint", u64: "ulong",
            }
            node.dtype = _DTYPE_TO_C.get(field.dtype)
            if node.dtype is None:
                raise RuntimeError(
                    f"Unsupported dtype for shared_like: {field.dtype}")
        return node

    if isinstance(node, ir.IRLocalAlloc):
        node.size = _resolve(node.size, fields)
        return node

    if isinstance(node, ir.IRBlockReduce):
        node.value = _resolve(node.value, fields)
        return node

    if isinstance(node, ir.IRPrint):
        node.args = [_resolve(a, fields) for a in node.args]
        return node

    if isinstance(node, ir.IRReturn):
        if node.value:
            node.value = _resolve(node.value, fields)
        return node

    if isinstance(node, ir.IRTextureSample):
        node.coords = [_resolve(c, fields) for c in node.coords]
        field = fields.get(node.field_name)
        if field is not None:
            from pgc.lang.field import Texture3D
            if isinstance(field, Texture3D):
                node.shape = field.shape_3d
            else:
                node.shape = field.shape
        return node

    # Leaf nodes: IRConstant, IRName, IRAttribute, IRBreak, IRContinue
    return node
