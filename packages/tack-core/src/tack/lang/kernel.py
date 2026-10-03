"""Tack kernel decorator — captures Python functions for compilation."""

import ast
import inspect
import struct
import textwrap
import threading

from tack.lang.ast_transform import transform_kernel
from tack.lang.ir_verify import verify_ir


def _verified_transform(*args, **kwargs):
    module = transform_kernel(*args, **kwargs)
    for function in module.functions:
        verify_ir(function, 'lowered')
    return module

# Serialises AST→IR transformation. Two things make this necessary rather
# than tidy.
#
# `rewrite_templates` registers each resolved method in the *module-global*
# `_func_registry` under a name built from `id(obj)`, `transform_kernel`
# reads it back, and the caller pops it afterwards. Two threads sharing one
# @tack.data_oriented object build the same name, so whichever finishes
# first deletes the entry the other is about to look up. The loser fails
# with "Function call '__tmpl_Scaler_apply_4406896896__' not supported in
# kernels", or a bare KeyError from the transform.
#
# It survives on CPython today only because the whole register→transform→pop
# sequence fits inside one GIL slice: measured here, 200 dispatches across 8
# threads are clean at the default 5 ms switch interval and at 1 ms, and 72
# of them fail at 0.1 ms. A free-threaded build removes the slice entirely.
#
# Holding it across the cache check closes the check-then-act on
# `_ir_cache` too, which was benign but is free to fix here. The lock is
# taken only on a miss-shaped path — a warm dispatch never reaches it — so
# it costs nothing in steady state.
_transform_lock = threading.RLock()


def _fields_from_another_backend(args, backend) -> set:
    """Names of backends that allocated the field arguments, minus this one.

    Empty when nothing is out of place, which is the only answer the happy
    path ever needs — this runs after an AttributeError has already been
    raised, never before one.
    """
    from tack.lang.field import Field, Texture3D
    stale = set()
    for arg in args:
        field = arg.field if isinstance(arg, Texture3D) else arg
        if not isinstance(field, Field):
            continue
        origin = getattr(field._buffer, "backend_name", "")
        if origin and origin != backend.name:
            stale.add(origin)
    return stale


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
            cached = self._ir_cache.get(key)
            if cached is not None:
                return cached
            with _transform_lock:
                # Re-check: another thread may have built it while we waited.
                if key not in self._ir_cache:
                    from tack.lang.func import _func_registry
                    from tack.lang.template_rewrite import rewrite_templates
                    rewritten_ast, registered_keys = rewrite_templates(
                        self._ast, template_args)
                    try:
                        self._ir_cache[key] = _verified_transform(
                            rewritten_ast, vector_fields=vector_fields,
                            texture_fields=texture_fields,
                        )
                    finally:
                        # try/finally so a failed transform does not leave its
                        # temporaries behind: the name carries id(obj), which
                        # the allocator reuses, so a leaked entry is a stale
                        # method body waiting for an unrelated object to be
                        # born at the same address.
                        for rk in registered_keys:
                            _func_registry.pop(rk, None)
                return self._ir_cache[key]
        if not vector_fields and not texture_fields:
            if self._ir is None:
                with _transform_lock:
                    if self._ir is None:
                        self._ir = _verified_transform(self._ast)
            return self._ir
        key = self._make_cache_key(vector_fields, None, texture_fields)
        cached = self._ir_cache.get(key)
        if cached is not None:
            return cached
        with _transform_lock:
            if key not in self._ir_cache:
                self._ir_cache[key] = _verified_transform(
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
                scalars, fields, runtime = classify_template_attrs(obj)
                cls = type(obj)
                # Only class-level scalars (constants) are part of the cache key.
                # Instance scalars are runtime parameters — changing them does
                # not trigger recompilation.
                parts.append((
                    f"tmpl_{idx}",
                    cls,
                    # Float bit patterns distinguish signed zero and give
                    # NaN constants a stable key despite NaN != NaN.
                    tuple((k, type(v), struct.pack('!d', v) if isinstance(v, float) else v)
                          for k, v in sorted(scalars.items())),
                    tuple((k, f.dtype, f.shape, getattr(f, '_vector_n', None))
                          for k, f in sorted(fields.items())),
                    tuple(sorted(runtime)),
                ))
        return tuple(parts)

    def __call__(self, *args, **kwargs):
        from tack.runtime.dispatch import get_backend
        backend = get_backend()
        try:
            return backend.execute(self, args, kwargs)
        except AttributeError as e:
            # Almost always one thing: a field allocated before `tack.init()`
            # switched backends, so its buffer is the old backend's type and
            # the new one reaches for an attribute that was never there. The
            # raw error is `'NumpyBuffer' object has no attribute
            # 'metal_buffer'`, which names neither the cause nor the fix.
            # Diagnosed here rather than checked per dispatch: the check
            # costs nothing on the path that works.
            stale = _fields_from_another_backend(args, backend)
            if not stale:
                raise
            raise RuntimeError(
                f"Kernel '{self.name}' was called on the {backend.label} "
                f"backend with {len(stale)} field(s) allocated by another "
                f"backend ({', '.join(sorted(stale))}). `tack.init()` replaces "
                f"the active backend; fields allocated before it still belong "
                f"to the old one and cannot be dispatched on the new one. "
                f"Re-allocate them after switching, and do not re-initialize "
                f"while another thread is dispatching."
            ) from e
        except TypeError as e:
            # `from e`, not `from None`. Naming the kernel is worth doing, but
            # discarding the chain with it means a codegen failure reports
            # only "Kernel 'x' failed" with no way to see where it came from
            # short of editing this file. The chained traceback prints above
            # the clean message, so the readable summary is still last.
            raise TypeError(
                f"Kernel '{self.name}': {e}"
            ) from e
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
                    ) from e
            raise RuntimeError(
                f"Kernel '{self.name}' failed on {type(backend).__name__}: {e}"
            ) from e

    def __repr__(self):
        return f"Kernel({self.name})"


def kernel(func):
    """Decorator that marks a Python function as a GPU kernel."""
    return Kernel(func)
