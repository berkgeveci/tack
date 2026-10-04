"""Structural children of Tack IR, independent of pass annotations.

Add new node kinds here when extending the IR. Unknown nodes fail loudly;
metadata such as dtype, shape dependencies, and source information is not
part of the tree. The traversal visits each occurrence of a shared node.
"""

import copy

from tack.lang import ir
from tack.lang.types import ScalarType

# (attribute, role). Lists have plural roles; only optional_expr may be None.
CHILD_FIELDS = {
    ir.IRModule: (('functions', 'functions'),),
    ir.IRFunction: (('params', 'params'), ('body', 'stmts')),
    ir.IRParam: (),
    ir.IRParallelFor: (('start', 'expr'), ('end', 'expr'), ('body', 'stmts')),
    ir.IRSequentialFor: (('start', 'expr'), ('end', 'expr'),
                         ('step', 'optional_expr'), ('body', 'stmts')),
    ir.IRWhile: (('condition', 'expr'), ('body', 'stmts')),
    ir.IRBreak: (),
    ir.IRContinue: (),
    ir.IRIf: (('condition', 'expr'), ('then_body', 'stmts'), ('else_body', 'stmts')),
    ir.IRIfExp: (('condition', 'expr'), ('then_value', 'expr'), ('else_value', 'expr')),
    ir.IRBinOp: (('left', 'expr'), ('right', 'expr')),
    ir.IRUnaryOp: (('operand', 'expr'),),
    ir.IRCompare: (('left', 'expr'), ('right', 'expr')),
    ir.IRBoolOp: (('values', 'exprs'),),
    ir.IRFieldLoad: (('field', 'expr'), ('index', 'expr')),
    ir.IRFieldStore: (('field', 'expr'), ('index', 'expr'), ('value', 'expr')),
    ir.IRAtomicOp: (('field', 'expr'), ('index', 'expr'), ('value', 'expr')),
    ir.IRConstant: (),
    ir.IRName: (),
    ir.IRAttribute: (('obj', 'expr'),),
    ir.IRCall: (('args', 'exprs'),),
    ir.IRAssign: (('value', 'expr'),),
    ir.IRReturn: (('value', 'optional_expr'),),
    ir.IRCast: (('value', 'expr'),),
    ir.IRSharedAlloc: (('size', 'expr'),),
    ir.IRLocalAlloc: (('size', 'expr'),),
    ir.IRBlockReduce: (('value', 'expr'),),
    ir.IRBarrier: (),
    ir.IRThreadId: (),
    ir.IRPrint: (('args', 'exprs'),),
    ir.IRDimSize: (),
    ir.IRTextureSample: (('coords', 'exprs'),),
}

LIST_ROLES = {'functions', 'params', 'stmts', 'exprs'}
_COPY_ATOMIC = {type(None), bool, int, float, complex, str, bytes, range,
                type(Ellipsis), type(NotImplemented), ScalarType}


def clone_ir(root):
    """Deep-copy an IR graph, including annotations, with one identity memo.

    Registered nodes are plain attribute containers: avoid their generic
    reconstruction protocol, but copy every attribute, not just structural
    children. Lists/dicts use the same memo to preserve sharing and cycles.
    Other metadata retains Python's deepcopy protocol and ScalarType identity.
    Verification still rejects malformed structural cycles at pass boundaries.
    """
    memo = {}
    keep_alive = []
    memo[id(memo)] = keep_alive

    def clone(value):
        kind = type(value)
        if kind in _COPY_ATOMIC:
            return value
        identity = id(value)
        if identity in memo:
            return memo[identity]
        if kind in CHILD_FIELDS:
            result = object.__new__(kind)
            memo[identity] = result
            keep_alive.append(value)
            result.__dict__ = clone(vars(value))
        elif kind is list:
            result = []
            memo[identity] = result
            keep_alive.append(value)
            result.extend(clone(item) for item in value)
        elif kind is dict:
            result = {}
            memo[identity] = result
            keep_alive.append(value)
            for key, item in value.items():
                result[clone(key)] = clone(item)
        else:
            result = copy.deepcopy(value, memo)
        return result

    try:
        return clone(root)
    finally:
        # Break the recursive closure's self-reference so its memo releases
        # both graphs immediately rather than waiting for cyclic collection.
        clone = None


def child_fields(node):
    """Return the declared structural fields, rejecting unregistered nodes."""
    try:
        return CHILD_FIELDS[type(node)]
    except KeyError:
        raise TypeError(f'Unknown IR node: {type(node).__name__}') from None


def walk_ir(root):
    """Yield nodes in preorder, accepting a node or a list/tuple of nodes."""
    active = set()

    def visit(node):
        if id(node) in active:
            raise ValueError('Cycle in IR tree')
        active.add(id(node))
        try:
            if isinstance(node, (list, tuple)):
                for item in node:
                    yield from visit(item)
                return
            fields = child_fields(node)
            yield node
            for name, role in fields:
                value = getattr(node, name)
                if role == 'optional_expr' and value is None:
                    continue
                yield from visit(value)
        finally:
            active.remove(id(node))

    yield from visit(root)


def transform_ir(root, rewrite, *, copy_nodes=False):
    """Rewrite nodes in postorder, mutating structural children in place.

    ``rewrite(node)`` returns the original or a replacement node. New
    replacement children are not visited again. Node metadata is preserved.
    ``copy_nodes=True`` shallow-copies each occurrence before rewriting its
    children, for substitutions whose mappings depend on statement order.
    """
    active = set()

    def visit(node):
        if node is None:
            return None
        identity = id(node)
        if identity in active:
            raise ValueError('Cycle in IR tree')
        active.add(identity)
        try:
            if isinstance(node, list):
                return [visit(item) for item in node]
            fields = child_fields(node)
            if copy_nodes:
                node = copy.copy(node)
            for name, _ in fields:
                setattr(node, name, visit(getattr(node, name)))
            return rewrite(node)
        finally:
            active.remove(identity)

    return visit(root)
