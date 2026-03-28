"""IR pass — packs scalar kernel parameters into field buffers.

Solves the Metal 31 buffer binding limit by replacing N scalar parameters
with a small number of typed field parameters (one per scalar type).
After this pass, all former scalar params become field loads from packed
buffers, consuming only 1 buffer binding per type instead of 1 per scalar.

Must run after type inference (needs _is_field and type_annotation).
"""

import copy

from tack.lang import ir
from tack.lang.types import f32, i32, i64


def pack_scalars(ir_func: ir.IRFunction, args: tuple):
    """Pack scalar params into typed field buffers.

    Mutates ir_func.params and ir_func.body in place.
    Returns (new_args, pack_info) where:
      - new_args: rewritten argument tuple (scalars removed, packed fields added)
      - pack_info: list of (pack_param_name, dtype, [(orig_name, index, value)])
        or None if no packing was done.
    """
    # Collect scalar params grouped by type
    groups = {}  # dtype -> [(param_index, param)]
    for i, param in enumerate(ir_func.params):
        if not getattr(param, '_is_field', True):
            dtype = param.type_annotation
            groups.setdefault(dtype, []).append((i, param))

    if not groups:
        return args, None

    # Build the replacement map: old_param_name -> (pack_field_name, index_in_pack)
    replace_map = {}
    pack_info = []
    new_params = []
    new_args = []
    scalar_indices = set()

    for dtype, entries in groups.items():
        pack_name = f"__pack_{dtype.name}__"
        pack_param = ir.IRParam(name=pack_name, type_annotation=dtype)
        pack_param._is_field = True
        new_params.append(pack_param)

        values = []
        for idx_in_pack, (param_idx, param) in enumerate(entries):
            replace_map[param.name] = (pack_name, idx_in_pack)
            scalar_indices.add(param_idx)
            values.append((param.name, param_idx, idx_in_pack))

        pack_info.append((pack_name, dtype, values))

    # Build new params: keep non-scalar params, append pack params
    kept_params = []
    kept_args = []
    for i, (param, arg) in enumerate(zip(ir_func.params, args)):
        if i not in scalar_indices:
            kept_params.append(param)
            kept_args.append(arg)

    ir_func.params = kept_params + new_params

    # Rewrite IR body: IRName(scalar) -> IRFieldLoad(IRName(pack), IRConstant(idx))
    for i, stmt in enumerate(ir_func.body):
        ir_func.body[i] = _rewrite(stmt, replace_map)

    # Build packed field args (numpy arrays created by caller)
    # Store pack_info on ir_func for the dispatch layer
    ir_func._scalar_pack_info = pack_info

    # Return new args (non-scalar args + placeholder Nones for pack fields)
    # The caller must create the actual Field objects from pack_info
    return tuple(kept_args), pack_info


def split_args(args, pack_info):
    """Extract non-scalar args from the original args tuple.

    Uses pack_info (from a previous pack_scalars call) to determine
    which arg indices are scalars and should be excluded.
    """
    scalar_indices = set()
    for _, _, entries in pack_info:
        for _, orig_arg_idx, _ in entries:
            scalar_indices.add(orig_arg_idx)
    return tuple(a for i, a in enumerate(args) if i not in scalar_indices)


def _rewrite(node, replace_map):
    """Recursively rewrite IRName references to packed scalar params."""
    if node is None:
        return None

    if isinstance(node, ir.IRName):
        if node.name in replace_map:
            pack_name, idx = replace_map[node.name]
            return ir.IRFieldLoad(ir.IRName(pack_name), ir.IRConstant(idx))
        return node

    if isinstance(node, ir.IRBinOp):
        node.left = _rewrite(node.left, replace_map)
        node.right = _rewrite(node.right, replace_map)
        return node

    if isinstance(node, ir.IRUnaryOp):
        node.operand = _rewrite(node.operand, replace_map)
        return node

    if isinstance(node, ir.IRCompare):
        node.left = _rewrite(node.left, replace_map)
        node.right = _rewrite(node.right, replace_map)
        return node

    if isinstance(node, ir.IRBoolOp):
        node.values = [_rewrite(v, replace_map) for v in node.values]
        return node

    if isinstance(node, ir.IRFieldLoad):
        node.field = _rewrite(node.field, replace_map)
        node.index = _rewrite(node.index, replace_map)
        return node

    if isinstance(node, ir.IRFieldStore):
        node.field = _rewrite(node.field, replace_map)
        node.index = _rewrite(node.index, replace_map)
        node.value = _rewrite(node.value, replace_map)
        return node

    if isinstance(node, ir.IRAtomicOp):
        node.field = _rewrite(node.field, replace_map)
        node.index = _rewrite(node.index, replace_map)
        node.value = _rewrite(node.value, replace_map)
        return node

    if isinstance(node, ir.IRAssign):
        node.value = _rewrite(node.value, replace_map)
        return node

    if isinstance(node, ir.IRParallelFor):
        node.start = _rewrite(node.start, replace_map)
        node.end = _rewrite(node.end, replace_map)
        node.body = [_rewrite(s, replace_map) for s in node.body]
        return node

    if isinstance(node, ir.IRSequentialFor):
        node.start = _rewrite(node.start, replace_map)
        node.end = _rewrite(node.end, replace_map)
        if node.step is not None:
            node.step = _rewrite(node.step, replace_map)
        node.body = [_rewrite(s, replace_map) for s in node.body]
        return node

    if isinstance(node, ir.IRWhile):
        node.condition = _rewrite(node.condition, replace_map)
        node.body = [_rewrite(s, replace_map) for s in node.body]
        return node

    if isinstance(node, ir.IRIf):
        node.condition = _rewrite(node.condition, replace_map)
        node.then_body = [_rewrite(s, replace_map) for s in node.then_body]
        node.else_body = [_rewrite(s, replace_map) for s in node.else_body]
        return node

    if isinstance(node, ir.IRIfExp):
        node.condition = _rewrite(node.condition, replace_map)
        node.then_value = _rewrite(node.then_value, replace_map)
        node.else_value = _rewrite(node.else_value, replace_map)
        return node

    if isinstance(node, ir.IRCall):
        node.args = [_rewrite(a, replace_map) for a in node.args]
        return node

    if isinstance(node, ir.IRCast):
        node.value = _rewrite(node.value, replace_map)
        return node

    if isinstance(node, ir.IRSharedAlloc):
        node.size = _rewrite(node.size, replace_map)
        return node

    if isinstance(node, ir.IRLocalAlloc):
        node.size = _rewrite(node.size, replace_map)
        return node

    if isinstance(node, ir.IRBlockReduce):
        node.value = _rewrite(node.value, replace_map)
        return node

    if isinstance(node, ir.IRPrint):
        node.args = [_rewrite(a, replace_map) for a in node.args]
        return node

    if isinstance(node, ir.IRReturn):
        if node.value:
            node.value = _rewrite(node.value, replace_map)
        return node

    if isinstance(node, ir.IRTextureSample):
        node.coords = [_rewrite(c, replace_map) for c in node.coords]
        return node

    # Leaf nodes: IRConstant, IRAttribute, IRBreak, IRContinue, etc.
    return node
