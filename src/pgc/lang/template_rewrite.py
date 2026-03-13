"""PGC template rewrite — AST pre-pass that resolves template parameters.

When a @pgc.kernel is called with a @pgc.data_oriented object, this pass
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

from pgc.lang.field import Field
from pgc.lang.func import Func, _func_registry


def classify_template_attrs(obj):
    """Classify a template object's attributes into scalars and fields.

    Returns (scalars, fields) where:
    - scalars: dict[str, int|float] — become compile-time constants
    - fields: dict[str, Field] — become extra kernel parameters
    """
    scalars = {}
    fields = {}
    for name in vars(obj):
        if name.startswith('_'):
            continue
        val = getattr(obj, name)
        if isinstance(val, (int, float)):
            scalars[name] = val
        elif isinstance(val, Field):
            fields[name] = val
    return scalars, fields


def rewrite_templates(kernel_ast, template_args):
    """Rewrite a kernel AST to resolve template parameters.

    Args:
        kernel_ast: The kernel's Python AST (will be deep-copied)
        template_args: dict of param_index -> (param_name, template_object)

    Returns:
        Rewritten AST with template params resolved.
    """
    rewritten = copy.deepcopy(kernel_ast)
    funcdef = rewritten.body[0]

    # Process each template parameter (reverse order to keep indices stable)
    for idx in sorted(template_args.keys(), reverse=True):
        param_name, obj = template_args[idx]
        scalars, fields = classify_template_attrs(obj)

        # Build mapping from field attr name to synthetic parameter name
        field_param_map = {}
        for attr_name in sorted(fields.keys()):
            field_param_map[attr_name] = f"__tmpl_{param_name}_{attr_name}__"

        # Register resolved versions of the template object's @pgc.func methods
        cls = type(obj)
        method_name_map = {}  # original method name -> resolved func name
        if hasattr(cls, '_pgc_func_methods'):
            for method_name, func_obj in cls._pgc_func_methods.items():
                resolved_name = f"__tmpl_{cls.__name__}_{method_name}_{id(obj)}__"
                method_name_map[method_name] = resolved_name
                _register_resolved_method(
                    func_obj, resolved_name, scalars, field_param_map
                )

        # Rewrite the kernel function definition
        rewriter = _KernelTemplateRewriter(
            param_name, idx, scalars, field_param_map, method_name_map
        )
        rewriter.visit(funcdef)
        ast.fix_missing_locations(funcdef)

    return rewritten


def _register_resolved_method(func_obj, resolved_name, scalars, field_param_map):
    """Register a resolved copy of a template method in the func registry.

    The resolved copy has:
    - 'self' parameter removed
    - self.scalar_attr replaced with constants
    - self.field_attr replaced with synthetic parameter names
    """
    funcdef = copy.deepcopy(func_obj._funcdef)

    # Remove 'self' parameter
    funcdef.args.args = [a for a in funcdef.args.args if a.arg != 'self']

    # Add synthetic field parameters
    for attr_name, synth_name in sorted(field_param_map.items()):
        funcdef.args.args.append(ast.arg(arg=synth_name))

    # Resolve self.attr references in the body
    resolver = _SelfResolver(scalars, field_param_map)
    for i, stmt in enumerate(funcdef.body):
        funcdef.body[i] = resolver.visit(stmt)

    funcdef.name = resolved_name
    ast.fix_missing_locations(funcdef)

    # Create a Func-like entry in the registry
    resolved_func = _ResolvedFunc(resolved_name, funcdef)
    _func_registry[resolved_name] = resolved_func


class _ResolvedFunc:
    """A resolved template method, compatible with the func registry."""

    def __init__(self, name, funcdef):
        self.name = name
        self._funcdef = funcdef


class _SelfResolver(ast.NodeTransformer):
    """Replaces self.attr with constants or synthetic parameter names."""

    def __init__(self, scalars, field_param_map):
        self.scalars = scalars
        self.field_param_map = field_param_map

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            if node.attr in self.scalars:
                return ast.Constant(value=self.scalars[node.attr])
            if node.attr in self.field_param_map:
                return ast.Name(
                    id=self.field_param_map[node.attr], ctx=ast.Load()
                )
            raise ValueError(
                f"Template method references self.{node.attr} which is "
                f"neither a scalar (int/float) nor a pgc.Field"
            )
        return node


class _KernelTemplateRewriter(ast.NodeTransformer):
    """Rewrites a kernel function to resolve one template parameter."""

    def __init__(self, param_name, param_idx, scalars, field_param_map,
                 method_name_map):
        self.param_name = param_name
        self.param_idx = param_idx
        self.scalars = scalars
        self.field_param_map = field_param_map
        self.method_name_map = method_name_map

    def visit_FunctionDef(self, node):
        # Remove the template parameter
        node.args.args = [
            a for i, a in enumerate(node.args.args) if i != self.param_idx
        ]
        # Add synthetic field parameters at the end
        for attr_name, synth_name in sorted(self.field_param_map.items()):
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
                # Add synthetic field params as extra arguments
                extra_args = [
                    ast.Name(id=synth_name, ctx=ast.Load())
                    for _, synth_name in sorted(self.field_param_map.items())
                ]
                return ast.Call(
                    func=ast.Name(id=resolved_name, ctx=ast.Load()),
                    args=node.args + extra_args,
                    keywords=[],
                )
            raise ValueError(
                f"Template object method '{method_name}' is not decorated "
                f"with @pgc.func"
            )
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        # Resolve direct attribute access on the template param (e.g., cell_set.num_cells)
        if (isinstance(node.value, ast.Name) and
                node.value.id == self.param_name):
            if node.attr in self.scalars:
                return ast.Constant(value=self.scalars[node.attr])
            if node.attr in self.field_param_map:
                return ast.Name(
                    id=self.field_param_map[node.attr], ctx=ast.Load()
                )
            # Could be a property or method name used without calling — skip
        return node
