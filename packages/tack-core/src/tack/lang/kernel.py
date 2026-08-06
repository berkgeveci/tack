"""Tack kernel decorator — captures Python functions for compilation."""

import ast
import inspect
import textwrap

from tack.lang.ast_transform import transform_kernel
from tack.lang.type_inference import infer_types


class Kernel:
    """A captured kernel function, ready for AST transformation and compilation."""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self._source = textwrap.dedent(self._read_source(func))
        self._ast = ast.parse(self._source)
        self._funcdef = self._ast.body[0]  # The FunctionDef node
        # Lazy IR: defer transform until first dispatch (vector fields may be needed)
        self._ir = None
        self._ir_cache = {}  # vector_fields key → IRModule
        self._compiled = {}  # backend -> compiled kernel

    @staticmethod
    def _read_source(func) -> str:
        """Read the function's source, or explain why it could not be read.

        A kernel is compiled from its source text, so Tack needs to find it.
        `inspect.getsource` cannot when the function has no file behind it —
        `exec()`, a bare REPL, or `python -c` before 3.13. The raw OSError
        says only "could not get source code", which gives no hint that the
        problem is *where the function was defined* rather than the kernel.
        """
        try:
            return inspect.getsource(func)
        except (OSError, TypeError) as e:
            raise RuntimeError(
                f"Cannot read the source of kernel '{func.__name__}'. Tack "
                f"compiles kernels from their source text, so they must be "
                f"defined somewhere Python can read back — a module, a script, "
                f"or a Jupyter cell. Defining one with exec(), in a bare REPL, "
                f"or via `python -c` (before Python 3.13) does not work.\n"
                f"  original error: {e}"
            ) from e

    def get_ir(self, vector_fields=None, template_args=None, texture_fields=None):
        """Get IR, re-transforming if vector/texture field or template metadata is provided."""
        if template_args:
            key = self._make_cache_key(vector_fields, template_args, texture_fields)
            if key not in self._ir_cache:
                from tack.lang.template_rewrite import rewrite_templates
                from tack.lang.func import _func_registry
                rewritten_ast, registered_keys = rewrite_templates(self._ast, template_args)
                self._ir_cache[key] = transform_kernel(
                    rewritten_ast, vector_fields=vector_fields,
                    texture_fields=texture_fields,
                )
                # Clean up temporary func registry entries from template rewrite
                for rk in registered_keys:
                    _func_registry.pop(rk, None)
            return self._ir_cache[key]
        if not vector_fields and not texture_fields:
            if self._ir is None:
                self._ir = transform_kernel(self._ast)
            return self._ir
        key = self._make_cache_key(vector_fields, None, texture_fields)
        if key not in self._ir_cache:
            self._ir_cache[key] = transform_kernel(
                self._ast, vector_fields=vector_fields,
                texture_fields=texture_fields,
            )
        return self._ir_cache[key]

    def _make_cache_key(self, vector_fields, template_args, texture_fields=None):
        """Build a cache key that distinguishes different specializations."""
        parts = []
        if vector_fields:
            parts.append(("vec", tuple(sorted(vector_fields.items()))))
        if texture_fields:
            parts.append(("tex", tuple(sorted(texture_fields.items()))))
        if template_args:
            for idx in sorted(template_args.keys()):
                param_name, obj = template_args[idx]
                from tack.lang.template_rewrite import classify_template_attrs
                scalars, fields, _runtime = classify_template_attrs(obj)
                cls = type(obj)
                # Only class-level scalars (constants) are part of the cache key.
                # Instance scalars are runtime parameters — changing them does
                # not trigger recompilation.
                parts.append((
                    f"tmpl_{idx}",
                    cls.__qualname__,
                    tuple(sorted(scalars.items())),
                    tuple((k, f.dtype, f.shape) for k, f in sorted(fields.items())),
                ))
        return tuple(parts)

    def __call__(self, *args, **kwargs):
        from tack.runtime.dispatch import get_backend
        backend = get_backend()
        try:
            return backend.execute(self, args, kwargs)
        except TypeError as e:
            raise TypeError(
                f"Kernel '{self.name}': {e}"
            ) from None
        except RuntimeError as e:
            msg = str(e)
            # Shader compilation errors: show a concise message
            if "compilation failed" in msg.lower():
                # Extract just the error lines, not the full source dump
                lines = msg.split("\n")
                error_lines = [l for l in lines if "error:" in l.lower()]
                if error_lines:
                    brief = "\n".join(error_lines[:5])
                    raise RuntimeError(
                        f"Kernel '{self.name}' failed to compile on {type(backend).__name__}:\n"
                        f"{brief}\n"
                        f"(Set TACK_DUMP_MSL=1 to inspect generated source)"
                    ) from None
            raise RuntimeError(
                f"Kernel '{self.name}' failed on {type(backend).__name__}: {e}"
            ) from None

    def __repr__(self):
        return f"Kernel({self.name})"


def kernel(func):
    """Decorator that marks a Python function as a GPU kernel."""
    return Kernel(func)
