"""Identifier slots in the structural IR, excluding intrinsic call names."""

from tack.lang import ir
from tack.lang.ir_traversal import walk_ir

# Definitions and references share a binding spelling in Tack IR. Attributes
# and IRCall.func_name describe operations, rather than variable bindings.
NAME_ARRAYS = {
    ir.IRParam: ('name',),
    ir.IRName: ('name',),
    ir.IRAssign: ('target',),
    ir.IRParallelFor: ('var',),
    ir.IRSequentialFor: ('var',),
    ir.IRSharedAlloc: ('name', 'array_name'),
    ir.IRLocalAlloc: ('name', 'array_name'),
    ir.IRDimSize: ('array_name',),
    ir.IRTextureSample: ('array_name',),
}
# Slots holding a list of names: a multi-dimensional parallel loop's indices.
NAME_LIST_ARRAYS = {
    ir.IRParallelFor: ('dims',),
}


def ir_names(root):
    """Return all variable spellings, including allocation/texture references."""
    names = {name for node in walk_ir(root)
             for attr in NAME_ARRAYS.get(type(node), ())
             if (name := getattr(node, attr)) is not None}
    names.update(name for node in walk_ir(root)
                 for attr in NAME_LIST_ARRAYS.get(type(node), ())
                 for name in getattr(node, attr) or ())
    return names


def fresh_name(preferred, used_names):
    """Reserve a deterministic spelling distinct from all existing bindings."""
    name = preferred
    suffix = 0
    while name in used_names:
        suffix += 1
        name = f'{preferred}_{suffix}'
    used_names.add(name)
    return name
