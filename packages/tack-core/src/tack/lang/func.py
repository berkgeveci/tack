"""Tack func decorator — captures device-side functions for inlining into kernels.

Functions decorated with @tack.func are inlined at the AST level when called
from within a @tack.kernel.  They are not compiled separately — their body
is substituted at each call site with parameter names replaced.

Supports return values: ``return expr`` becomes an assignment to a synthetic
result variable, and the call expression evaluates to that variable.
"""

import ast
import inspect
import textwrap


class Func:
    """A captured device-side function, ready for AST inlining."""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self._source = textwrap.dedent(inspect.getsource(func))
        self._ast = ast.parse(self._source)
        # Extract the FunctionDef node
        self._funcdef = self._ast.body[0]
        if not isinstance(self._funcdef, ast.FunctionDef):
            raise TypeError(f"@tack.func must decorate a function, got {type(self._funcdef)}")
        # Class methods are collected by @tack.data_oriented and resolved
        # per template transformation. Ordinary functions need no registry.
        self._is_method = (
            len(self._funcdef.args.args) > 0 and
            self._funcdef.args.args[0].arg == 'self'
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"@tack.func '{self.name}' cannot be called from Python. "
            "It can only be called from within a @tack.kernel."
        )

    def __repr__(self):
        return f"Func({self.name})"


def func(f):
    """Decorator that marks a Python function as a device-side inlineable function."""
    return Func(f)
