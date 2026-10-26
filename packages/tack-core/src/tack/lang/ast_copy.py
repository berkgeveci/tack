"""``copy_ast``: a deep copy of a Python syntax tree, as ``copy.deepcopy`` makes it.

Lowering copies a device function's or a template method's tree for every
call it inlines or resolves; ``copy.deepcopy``'s generic protocol made that
a second of a large kernel's cold compile. This walks the nodes' attributes
directly: nodes and lists copied, immutable leaves and the weak references
of ``_tack_origin`` shared (as deepcopy shares them), anything else through
``copy.deepcopy``.
"""

import ast
import copy
import weakref

_SHARED = (str, int, float, bool, type(None), bytes, complex, type(Ellipsis),
           weakref.ReferenceType)


def copy_ast(value):
    """A deep copy of ``value``: a node, a list of nodes, or a leaf."""
    if isinstance(value, ast.AST):
        kind = type(value)
        new = kind.__new__(kind)
        attributes = new.__dict__
        for name, item in value.__dict__.items():
            attributes[name] = copy_ast(item)
        return new
    if type(value) is list:
        return [copy_ast(item) for item in value]
    if isinstance(value, _SHARED):
        return value
    return copy.deepcopy(value)
