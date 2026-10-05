"""Resolve static device calls in the defining Python namespace.

Only module attributes are followed: resolving a call must not execute a
user object's property or evaluate arbitrary Python expressions. Numeric
captures and runtime function values remain outside the kernel language.
"""

import ast
import builtins
import math
from types import ModuleType

from tack.lang.func import Func

_UNBOUND = object()
_DYNAMIC = object()
_METHODS = {'dot', 'cross', 'norm', 'norm_sqr', 'normalized', 'sample'}


class CallBindings:
    def __init__(self, function, python_func=None, bindings=None, template_funcs=None):
        self.globals = python_func.__globals__ if python_func is not None else (bindings or {})
        self.nonlocals = {}
        if python_func is not None and python_func.__closure__:
            for name, cell in zip(python_func.__code__.co_freevars, python_func.__closure__):
                try:
                    self.nonlocals[name] = cell.cell_contents
                except ValueError:
                    pass  # an empty closure cell is not a callable binding
        self.locals = {a.arg for a in ast.walk(function.args) if isinstance(a, ast.arg)}
        self.locals.update(n.id for n in ast.walk(function) if isinstance(n, ast.Name)
                           and isinstance(n.ctx, ast.Store))
        self.template_funcs = template_funcs or {}

    def resolve(self, node):
        if isinstance(node, ast.Name):
            if getattr(node, '_tack_template_call', False):
                return self.template_funcs.get(node.id, _UNBOUND)
            if node.id in self.locals:
                return _DYNAMIC
            return self.nonlocals.get(node.id, self.globals.get(node.id, _UNBOUND))
        if isinstance(node, ast.Attribute):
            parent = self.resolve(node.value)
            if isinstance(parent, ModuleType):
                return vars(parent).get(node.attr, _UNBOUND)
        return _DYNAMIC

    def device_func(self, node):
        value = self.resolve(node)
        return value if isinstance(value, Func) else None

    def call_name(self, node):
        value = self.resolve(node)
        if isinstance(value, Func):
            return ''  # not an intrinsic, including allocation/loop syntax
        name = node.id if isinstance(node, ast.Name) else getattr(node, 'attr', '')
        if isinstance(node, ast.Name) and value is _UNBOUND:
            return name  # bare kernel intrinsics need no Python import
        if isinstance(node, ast.Attribute) and name in _METHODS and value is _DYNAMIC:
            return name  # vector/texture methods are checked by lowering
        import tack
        for module in (builtins, math, tack):
            if value is not _UNBOUND and value is vars(module).get(name, _UNBOUND):
                return name
        if isinstance(node, ast.Attribute):
            parent = self.resolve(node.value)
            if parent is math or parent is tack:
                return name
            # AST-only callers have no Python namespace; allow DSL qualifiers.
            if parent is _UNBOUND and isinstance(node.value, ast.Name) \
                    and node.value.id in ('math', 'tack'):
                return name
        raise NotImplementedError(
            f"Call '{ast.unparse(node)}' is not a statically bound @tack.func "
            'or a supported kernel intrinsic')
