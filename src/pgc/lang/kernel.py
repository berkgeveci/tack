"""PGC kernel decorator — captures Python functions for compilation."""

import ast
import inspect
import textwrap

from pgc.lang.ast_transform import transform_kernel
from pgc.lang.type_inference import infer_types


class Kernel:
    """A captured kernel function, ready for AST transformation and compilation."""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self._source = textwrap.dedent(inspect.getsource(func))
        self._ast = ast.parse(self._source)
        self._ir = transform_kernel(self._ast)
        self._compiled = {}  # backend -> compiled kernel

    def __call__(self, *args, **kwargs):
        from pgc.runtime.dispatch import get_backend
        backend = get_backend()
        return backend.execute(self, args, kwargs)

    def __repr__(self):
        return f"Kernel({self.name})"


def kernel(func):
    """Decorator that marks a Python function as a GPU kernel."""
    return Kernel(func)
