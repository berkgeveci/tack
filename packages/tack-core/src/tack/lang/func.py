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


def read_source(func, kind: str) -> str:
    """Read a kernel's or device function's source, or explain why not.

    Both are compiled from their source text, so Tack needs to find it.
    `inspect.getsource` cannot when the function has no file behind it —
    `exec()`, a bare REPL, or `python -c` before 3.13. The raw OSError
    says only "could not get source code", which gives no hint that the
    problem is *where the function was defined* rather than its body.
    `kind` names what was decorated ("kernel", "device function").
    """
    try:
        return inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise RuntimeError(
            f"Cannot read the source of {kind} '{func.__name__}'. Tack "
            f"compiles from source text, so it must be defined somewhere "
            f"Python can read back — a module, a script, or a Jupyter "
            f"cell. Defining one with exec(), in a bare REPL, or via "
            f"`python -c` (before Python 3.13) does not work.\n"
            f"  original error: {e}"
        ) from e


class Func:
    """A captured device-side function, ready for AST inlining."""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self._source = textwrap.dedent(read_source(func, "device function"))
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
