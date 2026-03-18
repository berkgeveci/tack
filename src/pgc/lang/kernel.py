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

    def get_ir(self, vector_fields=None, template_args=None, texture_fields=None):
        """Get IR, re-transforming if vector/texture field or template metadata is provided."""
        if template_args:
            key = self._make_cache_key(vector_fields, template_args, texture_fields)
            if key not in self._ir_cache:
                from pgc.lang.template_rewrite import rewrite_templates
                rewritten_ast = rewrite_templates(self._ast, template_args)
                self._ir_cache[key] = transform_kernel(
                    rewritten_ast, vector_fields=vector_fields,
                    texture_fields=texture_fields,
                )
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
                from pgc.lang.template_rewrite import classify_template_attrs
                scalars, fields = classify_template_attrs(obj)
                cls = type(obj)
                parts.append((
                    f"tmpl_{idx}",
                    cls.__qualname__,
                    tuple(sorted(scalars.items())),
                    tuple((k, f.dtype, f.shape) for k, f in sorted(fields.items())),
                ))
        return tuple(parts)

    def __call__(self, *args, **kwargs):
        from pgc.runtime.dispatch import get_backend
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
                        f"(Set PGC_DUMP_MSL=1 to inspect generated source)"
                    ) from None
            raise RuntimeError(
                f"Kernel '{self.name}' failed on {type(backend).__name__}: {e}"
            ) from None

    def __repr__(self):
        return f"Kernel({self.name})"


def kernel(func):
    """Decorator that marks a Python function as a GPU kernel."""
    return Kernel(func)
