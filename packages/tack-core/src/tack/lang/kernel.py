"""Tack kernel decorator — captures Python functions for compilation."""

import ast
import struct
import textwrap
import threading
import types
import weakref

from tack.lang.ast_transform import transform_kernel
from tack.lang.func import read_source
from tack.lang.ir_verify import verify_ir
from tack.lang.source_validation import UnsupportedSyntaxError
from tack.lang.types import ScalarType, f32, i32


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


def _template_key(obj):
    """What of a template object a kernel's IR depends on (after the slot's name)."""
    from tack.lang.template_rewrite import (
        classify_template_attrs,
        template_func_attrs,
        template_structure,
    )
    scalars, fields, runtime = classify_template_attrs(obj)
    # Only class-level scalars (constants) are part of the cache key.
    # Instance scalars are runtime parameters — changing them does
    # not trigger recompilation.
    return (
        _class_token(type(obj)),
        # Float bit patterns distinguish signed zero and give
        # NaN constants a stable key despite NaN != NaN.
        tuple((k, type(v), struct.pack('!d', v) if isinstance(v, float) else v)
              for k, v in sorted(scalars.items())),
        tuple((k, f.dtype, f.shape, getattr(f, '_vector_n', None),
               getattr(f, '_matrix_shape', None))
              for k, f in sorted(fields.items())),
        tuple(sorted(runtime)),
        # Which device function each function-valued attribute
        # holds is compiled in, so it identifies the IR.
        tuple(sorted(template_func_attrs(obj).items(), key=lambda item: item[0])),
        # Templates it holds, by path and class: a different nesting is
        # different code.
        tuple((path, _class_token(cls)) for path, cls in template_structure(obj)),
    )


def _key_tokens(key):
    """The class tokens named by a `_make_cache_key` result: each template's class
    and the classes of the templates it holds."""
    tokens = []
    for part in key:
        if len(part) > 1 and isinstance(part[1], _ClassToken):
            tokens.append(part[1])
            tokens.extend(token for _, token in part[-1])
    return tokens


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

    def __init__(self, func, funcdef=None):
        self.func = func
        if funcdef is None:
            self._source = textwrap.dedent(read_source(func, "kernel"))
            self._ast = ast.parse(self._source)
            self._funcdef = self._ast.body[0]  # The FunctionDef node
        else:
            # A kernel derived from another's AST: the result epilogue below.
            self._source = ast.unparse(funcdef)
            self._ast = ast.Module(body=[funcdef], type_ignores=[])
            self._funcdef = funcdef
        self.name = self._funcdef.name
        # Lazy IR: defer transform until first dispatch (vector fields may be needed)
        self._ir = None
        self._ir_cache = {}  # vector_fields key → IRModule
        # A kernel ending in `return expr` has the expression computed by a
        # one-thread epilogue kernel into a hidden field; see _split_result.
        self._result = None if funcdef is not None else _split_result(self, func)
        self._result_field = None

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
            from tack.lang.template_rewrite import memoized

            for idx in sorted(template_args.keys()):
                _, obj = template_args[idx]
                parts.append((f"tmpl_{idx}",
                              *memoized("key", obj, lambda obj=obj: _template_key(obj))))
        return tuple(parts)

    def _run_result(self, backend, args, kwargs):
        """Run the epilogue kernel and read the result it stored."""
        from tack.lang.field import field as make_field
        dtype, epilogue = self._result
        result = self._result_field
        if result is None or result._buffer.backend_name != backend.name:
            result = make_field(dtype=dtype, shape=(1,))
            self._result_field = result
        backend.execute(epilogue, (*args, result), kwargs)
        return result[0]

    def __get__(self, instance, owner=None):
        """Bind a kernel defined in a class body to the instance it is called on.

        ``model.step(dt)`` passes ``model`` as the kernel's first argument,
        where a ``@tack.data_oriented`` object is a template like any
        other. Reached through the class, the kernel is unbound.
        """
        if instance is None:
            return self
        return types.MethodType(self, instance)

    def __call__(self, *args, **kwargs):
        from tack.runtime.dispatch import get_backend
        backend = get_backend()
        try:
            backend.execute(self, args, kwargs)
            if self._result is None:
                return None
            return self._run_result(backend, args, kwargs)
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


_RESULT_PARAM = "__tack_result__"
_RESULT_TYPES = {"float": f32, "int": i32}


def _split_result(kernel, func):
    """Take a trailing ``return expr`` off a kernel and build its epilogue.

    A kernel's statements outside the parallel loop run once per thread,
    so a value returned there has no single meaning on the device. The
    return is instead computed after the launch by an epilogue kernel of
    one thread: the statements before and after the loop (which may only
    bind locals, load fields and declare arrays) followed by a store of
    the expression into a hidden one-element field, which the host reads.
    So a returned expression may read fields and the kernel's arguments
    and constants, but not a local the loop assigns. The type comes from
    the return annotation: ``-> tack.f32``, ``-> float`` (f32) or ``-> int``
    (i32), which the hidden field is allocated with.

    Returns ``(dtype, epilogue Kernel)``, or None for a kernel without one.
    Other returns are left for the validator to reject.
    """
    funcdef = kernel._funcdef
    body = funcdef.body
    if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return None
    returned = body[-1]
    dtype = _result_dtype(func, funcdef, returned)
    funcdef.body = body[:-1]
    loops = [s for s in funcdef.body if isinstance(s, ast.For)]
    if len(loops) != 1:
        return None       # the validator reports the missing or extra loop
    around = [s for s in funcdef.body if s is not loops[0]]
    bound_outside = {n.id for s in around for n in ast.walk(s)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound_inside = {n.id for n in ast.walk(loops[0])
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    params = {a.arg for a in funcdef.args.args}
    for n in ast.walk(returned.value):
        if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                and n.id in bound_inside and n.id not in bound_outside | params):
            raise UnsupportedSyntaxError(
                f"Kernel '{funcdef.name}': return at line {returned.lineno} reads '{n.id}', "
                f"which the parallel loop assigns and so has a value per thread; a returned "
                f"expression may read fields, arguments, constants and locals assigned "
                f"outside the loop")
    once = ast.For(
        target=ast.Name(id="__tack_once__", ctx=ast.Store()),
        iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                      args=[ast.Constant(1)], keywords=[]),
        body=[*around, ast.Assign(
            targets=[ast.Subscript(value=ast.Name(id=_RESULT_PARAM, ctx=ast.Load()),
                                   slice=ast.Constant(0), ctx=ast.Store())],
            value=returned.value)],
        orelse=[])
    epilogue = ast.FunctionDef(
        name=f"{funcdef.name}__result",
        args=ast.arguments(
            posonlyargs=[], args=[*funcdef.args.args, ast.arg(arg=_RESULT_PARAM)],
            vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[once], decorator_list=[], returns=None, type_params=[])
    for node in ast.walk(epilogue):
        if not hasattr(node, "lineno"):
            ast.copy_location(node, returned)
    ast.fix_missing_locations(epilogue)
    return dtype, Kernel(func, funcdef=epilogue)


def _result_dtype(func, funcdef, returned):
    annotation = getattr(func, "__annotations__", {}).get("return")
    if isinstance(annotation, str):
        annotation = _RESULT_TYPES.get(annotation.rsplit(".", 1)[-1], annotation)
    if annotation is float:
        annotation = f32
    elif annotation is int:
        annotation = i32
    if isinstance(annotation, ScalarType):
        return annotation
    raise UnsupportedSyntaxError(
        f"Kernel '{funcdef.name}': return at line {returned.lineno} needs a return "
        f"annotation naming the result's type (-> tack.f32, -> float, -> int, ...); "
        f"the hidden field it is written to is allocated with that type")
