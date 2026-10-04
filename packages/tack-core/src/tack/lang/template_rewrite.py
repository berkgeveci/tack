"""Tack template rewrite — AST pre-pass that resolves template parameters.

When a @tack.kernel is called with a @tack.data_oriented object, this pass
rewrites the kernel AST before IR transformation:

1. Removes template parameters from the function signature
2. Adds synthetic parameters for the template object's field attributes
3. Rewrites method calls (obj.method(args)) to plain function calls
4. Resolves self.attr references in method bodies:
   - scalar (int/float) → ast.Constant
   - Field → ast.Name referencing synthetic parameter
"""

import ast
import copy

from tack.lang.field import Field
from tack.lang.func import Func
from tack.lang.ir_names import fresh_name


def _method_call_name(name):
    """Tag generated method calls so user bindings cannot collide with them."""
    node = ast.Name(id=name, ctx=ast.Load())
    node._tack_template_call = True
    return node


def classify_template_attrs(obj):
    """Classify a template object's attributes into constants, runtime scalars, and fields.

    Returns (scalars, fields, runtime_scalars) where:
    - scalars: dict[str, int|float] — class-level variables, become compile-time constants
    - fields: dict[str, Field] — instance fields, become extra kernel parameters
    - runtime_scalars: dict[str, int|float] — instance scalars, become kernel scalar parameters

    Class variables (defined on the class, not in __init__) are treated as
    compile-time constants and baked into generated code. Instance variables
    that are scalars are passed as runtime parameters — changing them does
    not trigger recompilation.
    """
    cls = type(obj)
    class_vars = set(vars(cls)) - {'_data_oriented', '_tack_func_methods'}

    scalars = {}
    fields = {}
    runtime_scalars = {}

    # Scan class-level variables first (compile-time constants)
    for name in class_vars:
        if name.startswith('_'):
            continue
        val = getattr(cls, name)
        if isinstance(val, (int, float)):
            scalars[name] = val

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
                       getattr(type(obj), '_tack_func_methods', {}).values())
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
                f"__tmpl_{param_name}_{attr_name}__", used_names)

        # Build mapping from runtime scalar attr name to synthetic parameter name
        runtime_scalar_param_map = {}
        for attr_name in sorted(runtime_scalars.keys()):
            runtime_scalar_param_map[attr_name] = fresh_name(
                f"__tmpl_{param_name}_{attr_name}__", used_names)

        # Resolve the template object's methods in this transformation's map.
        cls = type(obj)
        method_name_map = {}  # original method name -> resolved func name
        if hasattr(cls, '_tack_func_methods'):
            # First pass: build the name map so methods can reference siblings
            for method_name, func_obj in cls._tack_func_methods.items():
                resolved_name = fresh_name(f"__tmpl_{param_name}_{method_name}__", used_names)
                method_name_map[method_name] = resolved_name
            # Second pass: resolve methods with the full sibling name map.
            for method_name, func_obj in cls._tack_func_methods.items():
                resolved_name = method_name_map[method_name]
                resolved_funcs[resolved_name] = _resolve_method(
                    func_obj, resolved_name, scalars, field_param_map,
                    method_name_map, runtime_scalar_param_map,
                )

        # Rewrite the kernel function definition
        rewriter = _KernelTemplateRewriter(
            param_name, idx, scalars, field_param_map, method_name_map,
            runtime_scalar_param_map,
        )
        rewriter.visit(funcdef)
        ast.fix_missing_locations(funcdef)

    return rewritten, resolved_funcs


def _resolve_method(func_obj, resolved_name, scalars, field_param_map,
                    method_name_map=None, runtime_scalar_param_map=None):
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
                             runtime_scalar_param_map)
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
                 runtime_scalar_param_map=None):
        self.scalars = scalars
        self.field_param_map = field_param_map
        self.method_name_map = method_name_map or {}
        self.runtime_scalar_param_map = runtime_scalar_param_map or {}

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
            if node.attr not in self.method_name_map:
                raise ValueError(
                    f"Template method references self.{node.attr} which is "
                    f"neither a class constant, an instance scalar, a tack.Field, "
                    f"nor a @tack.func method"
                )
        return node


class _KernelTemplateRewriter(ast.NodeTransformer):
    """Rewrites a kernel function to resolve one template parameter."""

    def __init__(self, param_name, param_idx, scalars, field_param_map,
                 method_name_map, runtime_scalar_param_map=None):
        self.param_name = param_name
        self.param_idx = param_idx
        self.scalars = scalars
        self.field_param_map = field_param_map
        self.method_name_map = method_name_map
        self.runtime_scalar_param_map = runtime_scalar_param_map or {}

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

    def visit_Call(self, node):
        node = self.generic_visit(node)
        # Rewrite template_obj.method(args) → resolved_func_name(args, *synth_params)
        if (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == self.param_name):
            method_name = node.func.attr
            if method_name in self.method_name_map:
                resolved_name = self.method_name_map[method_name]
                return ast.Call(
                    func=_method_call_name(resolved_name),
                    args=node.args + self._synth_extra_args(),
                    keywords=[],
                )
            raise ValueError(
                f"Template object method '{method_name}' is not decorated "
                f"with @tack.func"
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
