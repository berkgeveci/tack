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
        self._funcdef = self._ast.body[0]  # The FunctionDef node
        # Lazy IR: defer transform until first dispatch (vector fields may be needed)
        self._ir = None
        self._ir_cache = {}  # vector_fields key → IRModule
        self._compiled = {}  # backend -> compiled kernel

    def get_ir(self, vector_fields=None):
        """Get IR, re-transforming if vector field metadata is provided."""
        if not vector_fields:
            if self._ir is None:
                self._ir = transform_kernel(self._ast)
            return self._ir
        key = tuple(sorted(vector_fields.items()))
        if key not in self._ir_cache:
            self._ir_cache[key] = transform_kernel(self._ast, vector_fields=vector_fields)
        return self._ir_cache[key]

    def __call__(self, *args, **kwargs):
        from pgc.runtime.dispatch import get_backend
        backend = get_backend()
        return backend.execute(self, args, kwargs)

    def __repr__(self):
        return f"Kernel({self.name})"


def kernel(func):
    """Decorator that marks a Python function as a GPU kernel."""
    return Kernel(func)
