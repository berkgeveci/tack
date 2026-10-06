"""Tack kernel decorator — captures Python functions for compilation."""

import ast
import struct
import textwrap
import threading
import weakref

from tack.lang.ast_transform import transform_kernel
from tack.lang.func import read_source
from tack.lang.ir_verify import verify_ir
from tack.lang.source_validation import UnsupportedSyntaxError


def _verified_transform(*args, **kwargs):
    module = transform_kernel(*args, **kwargs)
    for function in module.functions:
        verify_ir(function, 'lowered')
    return module

# Serialises AST→IR construction and the IR-cache check/update. Template
# method maps are now local to each transform, but shared kernel cache
# misses still need synchronization. Warm dispatches do not take this lock.
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


class _ClassToken:
    """Stands in for a template class in cache keys.

    Specializations are keyed on the identity of the `@tack.data_oriented`
    class, because two classes with the same name can carry different
    methods. Keying on the class object itself would keep it alive for as
    long as the kernel lives, so a class created per call -- defined inside
    a function, say -- would leave one compiled variant behind each time.
    The token is the identity without the reference: when the class is
    collected, every specialization built for it is dropped.
    """

    __slots__ = ('__weakref__', 'kernels')

    def __init__(self):
        self.kernels = weakref.WeakSet()  # kernels specialized on this class


_class_tokens = weakref.WeakKeyDictionary()  # class → _ClassToken


def _class_token(cls) -> _ClassToken:
    token = _class_tokens.get(cls)
    if token is None:
        fresh = _ClassToken()
        token = _class_tokens.setdefault(cls, fresh)
        if token is fresh:
            weakref.finalize(cls, _retire_class, token)
    return token


def _key_tokens(key):
    """The class tokens named by a `_make_cache_key` result."""
    return [part[1] for part in key
            if len(part) > 1 and isinstance(part[1], _ClassToken)]


def _retire_class(token):
    """Drop every specialization built for a class that no longer exists."""
    from tack.runtime.kernel_utils import drop_variants

    def stale(key):
        return token in _key_tokens(key)

    for kernel in list(token.kernels):
        for key in [k for k in list(kernel._ir_cache) if stale(k)]:
            kernel._ir_cache.pop(key, None)
        # The template key sits third in a compiled variant's key.
        drop_variants(kernel, lambda variant_key: stale(variant_key[2]))


class Kernel:
    """A captured kernel function, ready for AST transformation and compilation."""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self._source = textwrap.dedent(read_source(func, "kernel"))
        self._ast = ast.parse(self._source)
        self._funcdef = self._ast.body[0]  # The FunctionDef node
        # Lazy IR: defer transform until first dispatch (vector fields may be needed)
        self._ir = None
        self._ir_cache = {}  # vector_fields key → IRModule

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
                    from tack.lang.template_rewrite import rewrite_templates
                    rewritten_ast, resolved_funcs = rewrite_templates(
                        self._ast, template_args)
                    self._ir_cache[key] = _verified_transform(
                        rewritten_ast, vector_fields=vector_fields,
                        texture_fields=texture_fields, python_func=self.func,
                        template_funcs=resolved_funcs,
                    )
                    for token in _key_tokens(key):
                        token.kernels.add(self)
                return self._ir_cache[key]
        if not vector_fields and not texture_fields:
            if self._ir is None:
                with _transform_lock:
                    if self._ir is None:
                        self._ir = _verified_transform(self._ast, python_func=self.func)
            return self._ir
        key = self._make_cache_key(vector_fields, None, texture_fields)
        cached = self._ir_cache.get(key)
        if cached is not None:
            return cached
        with _transform_lock:
            if key not in self._ir_cache:
                self._ir_cache[key] = _verified_transform(
                    self._ast, vector_fields=vector_fields,
                    texture_fields=texture_fields, python_func=self.func,
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
                cls = _class_token(type(obj))
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
                    tuple((k, f.dtype, f.shape, getattr(f, '_vector_n', None),
                           getattr(f, '_matrix_shape', None))
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
            # Dispatch checks that already name the kernel (argument count,
            # backend dtypes, atomic targets) keep their message, rather
            # than reading "Kernel 'k': Kernel 'k' ...".
            message = str(e)
            if not message.startswith(f"Kernel '{self.name}'"):
                message = f"Kernel '{self.name}': {message}"
            raise TypeError(message) from e
        except UnsupportedSyntaxError:
            # Already names the kernel and the source position, and callers
            # can catch it by type. It is a RuntimeError by inheritance, so
            # without this it would be re-wrapped below as a backend failure.
            raise
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
            # Verifier errors already name the kernel; say it once.
            msg = msg.removeprefix(f"Kernel '{self.name}': ")
            raise RuntimeError(
                f"Kernel '{self.name}' failed on {type(backend).__name__}: {msg}"
            ) from e

    def __repr__(self):
        return f"Kernel({self.name})"


def kernel(func):
    """Decorator that marks a Python function as a GPU kernel."""
    return Kernel(func)
