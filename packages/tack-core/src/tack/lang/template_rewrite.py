"""Tack template rewrite — AST pre-pass that resolves template parameters.

When a @tack.kernel is called with a @tack.data_oriented object, this pass
rewrites the kernel AST before IR transformation:

1. Removes template parameters from the function signature
2. Adds synthetic parameters for the template object's field attributes
3. Rewrites method calls (obj.method(args)) to plain function calls
4. Resolves self.attr references in method bodies:
   - scalar (int/float) → ast.Constant
   - Field → ast.Name referencing synthetic parameter
5. Rewrites ``for c in obj:`` over the object's iteration space (see
   ``iteration_space``)
"""

import ast
import copy
import weakref

from tack.lang.field import Field
from tack.lang.func import Func
from tack.lang.ir_names import fresh_name


def _method_call_name(name):
    """Tag generated method calls so user bindings cannot collide with them."""
    node = ast.Name(id=name, ctx=ast.Load())
    node._tack_template_call = True
    return node


# The names of each data-oriented class's numeric class attributes, found once
# per class: scanning the method resolution order on every launch was most of
# a small dataset filter's time, its views being composed of many mixins.
_CONSTANT_NAMES = weakref.WeakKeyDictionary()


def _class_constants(cls):
    """A class's compile-time constants, by name, with their current values."""
    names = _CONSTANT_NAMES.get(cls)
    if names is None:
        found = {}
        # Base classes before the classes derived from them, so a subclass
        # inherits its bases' constants and may override them.
        for klass in reversed(cls.__mro__):
            for name, val in vars(klass).items():
                if name.startswith('_'):
                    continue
                if isinstance(val, (int, float)):
                    found[name] = val
                else:
                    found.pop(name, None)   # replaced by something that is not a constant
        names = _CONSTANT_NAMES[cls] = tuple(found)
    constants = {}
    for name in names:
        val = getattr(cls, name)
        if isinstance(val, (int, float)):
            constants[name] = val
    return constants


def classify_template_attrs(obj):
    """Classify a template object's attributes into constants, runtime scalars, and fields.

    Returns (scalars, fields, runtime_scalars) where:
    - scalars: dict[str, int|float] — class-level variables, become compile-time constants
    - fields: dict[str, Field] — instance fields, become extra kernel parameters
    - runtime_scalars: dict[str, int|float] — instance scalars, become kernel scalar parameters

    Class variables (defined on the class, not in __init__) are treated as
    compile-time constants and baked into generated code. Instance variables
    that are scalars are passed as runtime parameters — changing them does
    not trigger recompilation. Which class attributes are constants is found
    once per class; a changed value is seen, a numeric attribute added to the
    class after its first launch is not.
    """
    scalars = _class_constants(type(obj))
    fields = {}
    runtime_scalars = {}

    # Scan instance variables (runtime parameters)
    for name in vars(obj):
        if name.startswith('_'):
            continue
        if name in scalars:
            # Instance overrides a class variable — use the instance value
            # but keep it as a compile-time constant (class-level declaration wins)
            scalars[name] = getattr(obj, name)
            continue
        val = getattr(obj, name)
        if isinstance(val, (int, float)):
            runtime_scalars[name] = val
        elif isinstance(val, Field):
            fields[name] = val
    return scalars, fields, runtime_scalars


def template_func_methods(cls) -> dict:
    """The ``@tack.func`` methods of a data-oriented class, inherited ones included.

    Collected along the method resolution order, so an override replaces
    the method it overrides. Looked up when a kernel is lowered rather than
    stored by the decorator, so a subclass that is not decorated itself is
    treated the same. A device function under ``@staticmethod`` is one of
    them: it is called as ``self.name(...)`` and has no ``self`` to resolve.
    """
    methods = {}
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, staticmethod):
                value = value.__func__
            if isinstance(value, Func):
                methods[name] = value
            else:
                methods.pop(name, None)
    return methods


def template_func_attrs(obj) -> dict:
    """Device functions a template object holds as instance attributes.

    ``self.kernel = cubic_kernel`` lets the object's methods, and kernels
    it is passed to, call ``self.kernel(r, h)``. The function is resolved
    when the kernel is lowered, so which function it is belongs to the
    kernel's specialization, like a class-level constant.
    """
    return {name: value for name, value in vars(obj).items()
            if not name.startswith('_') and isinstance(value, Func)}


def iteration_space(cls) -> tuple:
    """The attributes a ``for c in obj`` loop runs over, fastest first.

    A data-oriented class declares them as ``__tack_iterate__``: one to
    three names of its scalar attributes, class constants or instance
    values. One name makes ``for c in obj`` the loop ``for c in
    range(obj.name)``, with ``c`` an integer. Two or three make it an
    ``ndrange`` over them, with ``c`` the vector of indices, ``c[0]``
    running fastest; at the top of a kernel that loop is launched in its
    own shape, so nothing divides to recover the indices. The object's
    methods take ``c`` as it comes, so a kernel written ``for c in obj:``
    runs unchanged over objects that iterate differently.
    """
    names = getattr(cls, '__tack_iterate__', None)
    if names is None:
        raise TypeError(
            f"a {cls.__name__} cannot be iterated in a kernel: its class declares no "
            "__tack_iterate__, the attributes a loop over it runs through")
    names = (names,) if isinstance(names, str) else tuple(names)
    if not 1 <= len(names) <= 3 or not all(isinstance(n, str) for n in names):
        raise TypeError(f"{cls.__name__}.__tack_iterate__ must name one to three "
                        f"attributes, not {names!r}")
    return names


def template_field_param_name(param_name, attr_name):
    """The kernel parameter a template's field attribute becomes.

    The runtime predicts this name before lowering to tell the transform
    which of those parameters are vector fields, so both sides must agree;
    `fresh_name` only departs from it when the kernel's own source already
    uses the name.
    """
    return f"__tmpl_{param_name}_{attr_name}__"


def rewrite_templates(kernel_ast, template_args):
    """Rewrite a kernel AST to resolve template parameters.

    Args:
        kernel_ast: The kernel's Python AST (will be deep-copied)
        template_args: dict of param_index -> (param_name, template_object)

    Returns:
        (rewritten_ast, resolved_funcs) — the rewritten AST and a mapping
        of synthetic call names to resolved device functions. The mapping
        belongs to this transformation and never enters global state.
    """
    rewritten = copy.deepcopy(kernel_ast)
    funcdef = rewritten.body[0]
    resolved_funcs = {}
    # Reserve source names in both the caller and template methods before
    # allocating synthetic parameters. Attribute expansion must not merge a
    # user's binding with a field/scalar reference in the rewritten AST.
    sources = [funcdef]
    for _, obj in template_args.values():
        sources.extend(method._funcdef for method in
                       template_func_methods(type(obj)).values())
    used_names = {n.id for source in sources for n in ast.walk(source)
                  if isinstance(n, ast.Name)}
    used_names.update(n.arg for source in sources for n in ast.walk(source)
                      if isinstance(n, ast.arg))

    # Process each template parameter (reverse order to keep indices stable)
    for idx in sorted(template_args.keys(), reverse=True):
        param_name, obj = template_args[idx]
        scalars, fields, runtime_scalars = classify_template_attrs(obj)

        # Build mapping from field attr name to synthetic parameter name
        field_param_map = {}
        for attr_name in sorted(fields.keys()):
            field_param_map[attr_name] = fresh_name(
                template_field_param_name(param_name, attr_name), used_names)

        # Build mapping from runtime scalar attr name to synthetic parameter name
        runtime_scalar_param_map = {}
        for attr_name in sorted(runtime_scalars.keys()):
            runtime_scalar_param_map[attr_name] = fresh_name(
                f"__tmpl_{param_name}_{attr_name}__", used_names)

        # Resolve the template object's methods in this transformation's map.
        methods = template_func_methods(type(obj))
        method_name_map = {}  # original method name -> resolved func name
        # Device functions held as attributes are called as they are: they
        # take no self and no synthetic parameters. An attribute shadows a
        # method of the same name, as it does in Python.
        func_attr_map = {}
        for attr_name, func_obj in sorted(template_func_attrs(obj).items()):
            resolved_name = fresh_name(f"__tmpl_{param_name}_{attr_name}__", used_names)
            func_attr_map[attr_name] = resolved_name
            resolved_funcs[resolved_name] = func_obj
            methods.pop(attr_name, None)
        # First pass: build the name map so methods can reference siblings
        for method_name in methods:
            method_name_map[method_name] = fresh_name(
                f"__tmpl_{param_name}_{method_name}__", used_names)
        # Second pass: resolve methods with the full sibling name map.
        for method_name, func_obj in methods.items():
            resolved_name = method_name_map[method_name]
            resolved_funcs[resolved_name] = _resolve_method(
                func_obj, resolved_name, scalars, field_param_map,
                method_name_map, runtime_scalar_param_map, func_attr_map,
            )

        # Rewrite the kernel function definition
        rewriter = _KernelTemplateRewriter(
            param_name, idx, scalars, field_param_map, method_name_map,
            runtime_scalar_param_map, func_attr_map,
            template=obj, used_names=used_names,
        )
        rewriter.visit(funcdef)
        ast.fix_missing_locations(funcdef)

    return rewritten, resolved_funcs


def _resolve_method(func_obj, resolved_name, scalars, field_param_map,
                    method_name_map=None, runtime_scalar_param_map=None, func_attr_map=None):
    """Resolve a template method while retaining its defining Python callable.

    The resolved copy has:
    - 'self' parameter removed
    - self.class_scalar replaced with constants
    - self.field_attr replaced with synthetic parameter names
    - self.instance_scalar replaced with synthetic parameter names
    - self.method(args) calls replaced with resolved function calls
    """
    funcdef = copy.deepcopy(func_obj._funcdef)

    # Remove 'self' parameter
    funcdef.args.args = [a for a in funcdef.args.args if a.arg != 'self']

    # Add synthetic field parameters
    for attr_name, synth_name in sorted(field_param_map.items()):
        funcdef.args.args.append(ast.arg(arg=synth_name))

    # Add synthetic runtime scalar parameters
    if runtime_scalar_param_map:
        for attr_name, synth_name in sorted(runtime_scalar_param_map.items()):
            funcdef.args.args.append(ast.arg(arg=synth_name))

    # Resolve self.attr and self.method(args) references in the body
    resolver = _SelfResolver(scalars, field_param_map, method_name_map,
                             runtime_scalar_param_map, func_attr_map)
    for i, stmt in enumerate(funcdef.body):
        funcdef.body[i] = resolver.visit(stmt)

    funcdef.name = resolved_name
    ast.fix_missing_locations(funcdef)

    return _ResolvedFunc(resolved_name, funcdef, func_obj.func)


class _ResolvedFunc(Func):
    """A method body with template attributes resolved and bindings retained."""

    def __init__(self, name, funcdef, python_func):
        self.name = name
        self._funcdef = funcdef
        self.func = python_func


class _SelfResolver(ast.NodeTransformer):
    """Replaces self.attr with constants, synthetic parameter names, or method calls."""

    def __init__(self, scalars, field_param_map, method_name_map=None,
                 runtime_scalar_param_map=None, func_attr_map=None):
        self.scalars = scalars
        self.field_param_map = field_param_map
        self.method_name_map = method_name_map or {}
        self.runtime_scalar_param_map = runtime_scalar_param_map or {}
        self.func_attr_map = func_attr_map or {}

    def _synth_extra_args(self):
        """Build the list of synthetic extra arguments for method calls."""
        extra = [
            ast.Name(id=synth_name, ctx=ast.Load())
            for _, synth_name in sorted(self.field_param_map.items())
        ]
        extra += [
            ast.Name(id=synth_name, ctx=ast.Load())
            for _, synth_name in sorted(self.runtime_scalar_param_map.items())
        ]
        return extra

    def visit_Call(self, node):
        node = self.generic_visit(node)
        # Rewrite self.kernel(args), a device function held as an
        # attribute → that function(args)
        if (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'self' and
                node.func.attr in self.func_attr_map):
            return ast.Call(func=_method_call_name(self.func_attr_map[node.func.attr]),
                            args=node.args, keywords=[])
        # Rewrite self.method(args) → resolved_method(args, *synth_params)
        if (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'self' and
                node.func.attr in self.method_name_map):
            resolved_name = self.method_name_map[node.func.attr]
            return ast.Call(
                func=_method_call_name(resolved_name),
                args=node.args + self._synth_extra_args(),
                keywords=[],
            )
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            if node.attr in self.scalars:
                return ast.Constant(value=self.scalars[node.attr])
            if node.attr in self.field_param_map:
                return ast.Name(
                    id=self.field_param_map[node.attr], ctx=ast.Load()
                )
            if node.attr in self.runtime_scalar_param_map:
                return ast.Name(
                    id=self.runtime_scalar_param_map[node.attr], ctx=ast.Load()
                )
            # Method references used without calling (e.g., passing as arg)
            # are handled by visit_Call; bare attribute access on a method
            # that isn't in scalars/fields is an error
            if node.attr not in self.method_name_map and node.attr not in self.func_attr_map:
                raise ValueError(
                    f"Template method references self.{node.attr} which is "
                    f"neither a class constant, an instance scalar, a tack.Field, "
                    f"a @tack.func method, nor a @tack.func held as an attribute"
                )
        return node


class _KernelTemplateRewriter(ast.NodeTransformer):
    """Rewrites a kernel function to resolve one template parameter."""

    def __init__(self, param_name, param_idx, scalars, field_param_map,
                 method_name_map, runtime_scalar_param_map=None, func_attr_map=None,
                 template=None, used_names=None):
        self.template = template
        self.used_names = used_names if used_names is not None else set()
        self.param_name = param_name
        self.param_idx = param_idx
        self.scalars = scalars
        self.field_param_map = field_param_map
        self.method_name_map = method_name_map
        self.runtime_scalar_param_map = runtime_scalar_param_map or {}
        self.func_attr_map = func_attr_map or {}

    def _synth_extra_args(self):
        """Build the list of synthetic extra arguments for method calls."""
        extra = [
            ast.Name(id=synth_name, ctx=ast.Load())
            for _, synth_name in sorted(self.field_param_map.items())
        ]
        extra += [
            ast.Name(id=synth_name, ctx=ast.Load())
            for _, synth_name in sorted(self.runtime_scalar_param_map.items())
        ]
        return extra

    def visit_FunctionDef(self, node):
        # Remove the template parameter
        node.args.args = [
            a for i, a in enumerate(node.args.args) if i != self.param_idx
        ]
        # Add synthetic field parameters at the end
        for attr_name, synth_name in sorted(self.field_param_map.items()):
            node.args.args.append(ast.arg(arg=synth_name))
        # Add synthetic runtime scalar parameters
        for attr_name, synth_name in sorted(self.runtime_scalar_param_map.items()):
            node.args.args.append(ast.arg(arg=synth_name))

        # Visit the body
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        # for c in obj: becomes a range or ndrange loop over its iteration
        # space; generic_visit then resolves the obj.attr extents.
        if isinstance(node.iter, ast.Name) and node.iter.id == self.param_name:
            node = self._iteration_loop(node)
        return self.generic_visit(node)

    def _iteration_loop(self, node):
        cls = type(self.template)
        names = iteration_space(cls)
        for name in names:
            if name not in self.scalars and name not in self.runtime_scalar_param_map:
                raise TypeError(f"{cls.__name__}.__tack_iterate__ names {name!r}, which is "
                                "not a scalar attribute of the object")
        if not isinstance(node.target, ast.Name):
            raise TypeError(f"a loop over a {cls.__name__} binds one name, the index; "
                            f"not {ast.unparse(node.target)}")

        def extent(name):
            return ast.Attribute(value=ast.Name(id=self.param_name, ctx=ast.Load()),
                                 attr=name, ctx=ast.Load())

        if len(names) == 1:
            node.iter = ast.Call(func=ast.Name(id='range', ctx=ast.Load()),
                                 args=[extent(names[0])], keywords=[])
            return node
        index = node.target.id
        dims = [fresh_name(f"__{index}_{k}__", self.used_names) for k in range(len(names))]
        # ndrange runs slowest first; the index vector is fastest first.
        loop = ast.For(
            target=ast.Tuple(elts=[ast.Name(id=d, ctx=ast.Store()) for d in reversed(dims)],
                             ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id='ndrange', ctx=ast.Load()),
                          args=[extent(n) for n in reversed(names)], keywords=[]),
            body=[ast.Assign(targets=[ast.Name(id=index, ctx=ast.Store())],
                             value=ast.List(elts=[ast.Name(id=d, ctx=ast.Load())
                                                  for d in dims], ctx=ast.Load())),
                  *node.body],
            orelse=[])
        return ast.copy_location(loop, node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        # Rewrite template_obj.method(args) → resolved_func_name(args, *synth_params)
        if (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == self.param_name):
            method_name = node.func.attr
            if method_name in self.func_attr_map:
                return ast.Call(func=_method_call_name(self.func_attr_map[method_name]),
                                args=node.args, keywords=[])
            if method_name in self.method_name_map:
                resolved_name = self.method_name_map[method_name]
                return ast.Call(
                    func=_method_call_name(resolved_name),
                    args=node.args + self._synth_extra_args(),
                    keywords=[],
                )
            raise ValueError(
                f"Template object has no @tack.func method '{method_name}', "
                f"and no @tack.func held as an attribute of that name"
            )
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        # Resolve direct attribute access on the template param
        if (isinstance(node.value, ast.Name) and
                node.value.id == self.param_name):
            if node.attr in self.scalars:
                return ast.Constant(value=self.scalars[node.attr])
            if node.attr in self.field_param_map:
                return ast.Name(
                    id=self.field_param_map[node.attr], ctx=ast.Load()
                )
            if node.attr in self.runtime_scalar_param_map:
                return ast.Name(
                    id=self.runtime_scalar_param_map[node.attr], ctx=ast.Load()
                )
            # Could be a property or method name used without calling — skip
        return node
