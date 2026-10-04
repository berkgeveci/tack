"""IR resolution pass — resolves dispatch-time constants in the IR.

Runs after AST transform, before codegen. Replaces:
- IRDimSize(field_name, dim) → IRConstant(shape[dim])
  Uses _logical_shape for vector fields, shape for regular fields.
"""

from tack.lang import ir
from tack.lang.ir_traversal import transform_ir, walk_ir


def resolve_ir(ir_func: ir.IRFunction, name_to_field: dict):
    """Resolve dimension sizes in IR using actual field shapes.

    Mutates ir_func.body in place.
    """
    # Build field alias map: inlined @tack.func parameters create assignments
    # like __func_out_x0_0__ = out_x0, where out_x0 is a field. Track these
    # so shared_like/DimSize can resolve the mangled name to the actual field.
    aliases = _collect_field_aliases(ir_func.body, name_to_field)
    extended = {**name_to_field, **aliases}

    for i, stmt in enumerate(ir_func.body):
        if isinstance(stmt, ir.IRParallelFor):
            # Leave the grid bound unresolved. It never reaches generated
            # code — codegen reads the __loop_end__ parameter — so dispatch
            # evaluates it against the actual arguments instead. Folding it
            # to a literal would make the compiled kernel depend on the
            # array's length and force a recompile for every new size.
            stmt.start = _resolve(stmt.start, extended)
            stmt.body = [_resolve(s, extended) for s in stmt.body]
            ir_func.body[i] = stmt
        else:
            ir_func.body[i] = _resolve(stmt, extended)


def _collect_field_aliases(stmts, known_fields):
    """Collect inlined field aliases in structural statement order."""
    aliases = {}
    for node in walk_ir(stmts):
        if isinstance(node, ir.IRAssign) and isinstance(node.value, ir.IRName):
            src = node.value.name
            if src in known_fields:
                aliases[node.target] = known_fields[src]
            elif src in aliases:
                aliases[node.target] = aliases[src]
    return aliases


def _resolve(node, fields):
    """Resolve dispatch-time leaves using the shared structural traversal."""
    def resolve_leaf(node):
        if isinstance(node, ir.IRDimSize):
            field = fields.get(node.field_name)
            if field is None:
                raise RuntimeError(f"Cannot resolve dimension size: unknown field '{node.field_name}'")
            shape = getattr(field, '_logical_shape', None) or field.shape
            return ir.IRConstant(shape[node.dim])
        if isinstance(node, (ir.IRSharedAlloc, ir.IRLocalAlloc)):
            if node.dtype is None and node.field_name is not None:
                field = fields.get(node.field_name)
                if field is None:
                    kind = 'shared_like' if isinstance(node, ir.IRSharedAlloc) else 'local_array_like'
                    raise RuntimeError(f"Cannot resolve {kind}: unknown field '{node.field_name}'")
                node.dtype = field.dtype
        if isinstance(node, ir.IRTextureSample):
            field = fields.get(node.field_name)
            if field is not None:
                node.shape = getattr(field, 'shape_3d', None) or field.shape
        return node

    return transform_ir(node, resolve_leaf)
