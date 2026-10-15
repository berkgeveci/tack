"""Tack template rewrite — AST pre-pass that resolves template parameters.

When a @tack.kernel is called with a @tack.device_class object, this pass
rewrites the kernel AST before IR transformation:

1. Removes template parameters from the function signature
2. Adds synthetic parameters for the template object's array attributes
3. Rewrites method calls (obj.method(args)) to plain function calls
4. Resolves self.attr references in method bodies:
   - scalar (int/float) → ast.Constant
   - Array → ast.Name referencing synthetic parameter
5. Rewrites ``for c in obj:`` over the object's iteration space (see
   ``iteration_space``)

A template may hold other templates as instance attributes, to any depth:
``self.base.get(k)`` calls the held object's method, ``self.base.step``
reads its attribute, in methods and (as ``obj.base...``) in kernels. The
whole tree is one template: ``classify_template_attrs`` flattens it, a
nested object's attributes named by their path (``base__step``), so the
runtime expands, detects and keys nested arrays and scalars as it does its
own, and every method of the tree takes the same synthetic parameters.
Each nested class is part of the cache key, as the outer one is.
"""

import ast
import contextlib
import copy
import re
import threading
import weakref

from tack.lang.array import Array
from tack.lang.func import Func
from tack.lang.ir_names import fresh_name


def _method_call_name(name):
    """Tag generated method calls so user bindings cannot collide with them."""
    node = ast.Name(id=name, ctx=ast.Load())
    node._tack_template_call = True
    return node


# The names of each device class's numeric class attributes, found once
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


# Joins a nested template's path to its attribute names when the tree is
# flattened: ``self.base.step`` is the tree's ``base__step``.
_NESTED = "__"
# An instance tuple of numbers, ``self.shape = (nx, ny)``, is one runtime
# scalar per entry, named ``shape__0``, ``shape__1``: read as
# ``obj.shape[k]`` with a literal ``k``.
_TUPLE_ENTRY = re.compile(r"^(.+)__(\d+)$")


def _number_tuple(value):
    return (isinstance(value, tuple) and value
            and all(isinstance(v, (int, float)) for v in value))


def scalar_path(name):
    """The attribute path of an object's own runtime scalar: ``("n",)``, or
    ``("shape", 0)`` for an entry of a tuple attribute."""
    entry = _TUPLE_ENTRY.match(name)
    return (entry.group(1), int(entry.group(2))) if entry else (name,)

# One dispatch reads a template's tree several times -- its cache key, its
# expanded arguments, its vector arrays -- and each pass walked and
# classified the whole tree again, 20 us a level. Within `template_scope`
# each object's answers are found once. The objects are the call's
# arguments, alive throughout, so their ids cannot be reused within it.
_scope = threading.local()


@contextlib.contextmanager
def template_scope():
    """Find each template object's tree and attributes once until this exits."""
    depth = getattr(_scope, "depth", 0)
    if depth == 0:
        _scope.memo = {}
    _scope.depth = depth + 1
    try:
        yield
    finally:
        _scope.depth = depth
        if depth == 0:
            _scope.memo = None


def memoized(kind, obj, compute):
    """``compute()``, found once per template object within ``template_scope``."""
    return _memo(kind, obj, compute)


def _memo(kind, obj, compute):
    memo = getattr(_scope, "memo", None)
    if memo is None:
        return compute()
    key = (kind, id(obj))
    found = memo.get(key)
    if found is None:
        found = memo[key] = compute()
    return found


def _is_template(value):
    return getattr(type(value), "_device_class", False) is True


def template_children(obj) -> dict:
    """The templates an object holds as instance attributes, by name."""
    return {name: value for name, value in sorted(vars(obj).items())
            if not name.startswith('_') and _is_template(value)}


def template_tree(obj, path=(), _seen=None) -> list:
    """``(path, object)`` for ``obj`` and every template nested in it, depth first,
    children by name: the order every flattened list follows."""
    if not path and _seen is None:
        return _memo("tree", obj, lambda: _template_tree(obj, (), set()))
    return _template_tree(obj, path, _seen)


def _template_tree(obj, path, _seen):
    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        raise TypeError(f"a {type(obj).__name__} holds itself, through "
                        f"{'.'.join(path) or 'itself'}: a template is a tree")
    seen.add(id(obj))
    nodes = [(path, obj)]
    for name, child in template_children(obj).items():
        nodes.extend(_template_tree(child, (*path, name), seen))
    seen.discard(id(obj))
    return nodes


def _prefix(path):
    return _NESTED.join(path) + _NESTED if path else ""


def _classify_own(obj):
    """One object's own constants, arrays and runtime scalars (not its children's)."""
    scalars = _class_constants(type(obj))
    arrays = {}
    runtime_scalars = {}

    # Scan instance variables (runtime parameters)
    for name in vars(obj):
        if name.startswith('_'):
            continue
        val = getattr(obj, name)
        if _TUPLE_ENTRY.match(name):
            raise TypeError(f"{type(obj).__name__}.{name}: an attribute name ending in "
                            "'__' and digits is how a tuple's entries are named; rename it")
        if name in scalars:
            if isinstance(val, (int, float)):
                # Instance overrides a class variable — use the instance value
                # but keep it as a compile-time constant (class-level declaration wins)
                scalars[name] = val
                continue
            # Replaced by something that is not a number: no longer a
            # constant, as for a subclass that does the same, and classified
            # like any other attribute. Kept, it reached the cache keys, where
            # a list made dispatch fail with "unhashable type".
            del scalars[name]
        if isinstance(val, (int, float)):
            runtime_scalars[name] = val
        elif isinstance(val, Array):
            arrays[name] = val
        elif _number_tuple(val):
            for k, entry in enumerate(val):
                runtime_scalars[f"{name}{_NESTED}{k}"] = entry
    return scalars, arrays, runtime_scalars


def classify_template_attrs(obj):
    """Classify a template object's attributes into constants, runtime scalars, and arrays.

    Returns (scalars, arrays, runtime_scalars) where:
    - scalars: dict[str, int|float] — class-level variables, become compile-time constants
    - arrays: dict[str, Array] — instance arrays, become extra kernel parameters
    - runtime_scalars: dict[str, int|float] — instance scalars, become kernel scalar parameters

    Class variables (defined on the class, not in __init__) are treated as
    compile-time constants and baked into generated code. Instance variables
    that are scalars are passed as runtime parameters — changing them does
    not trigger recompilation. Which class attributes are constants is found
    once per class; a changed value is seen, a numeric attribute added to the
    class after its first launch is not. Templates held as attributes are
    included, their attributes named by their path (``base__step``).
    """
    return _memo("classes", obj, lambda: _classify_tree(obj))


def _classify_tree(obj):
    flat = ({}, {}, {})
    for path, node in template_tree(obj):
        prefix = _prefix(path)
        for out, own in zip(flat, _classify_own(node)):
            for name, value in own.items():
                key = prefix + name
                if any(key in d for d in flat):
                    raise TypeError(f"{type(obj).__name__}: the attribute {key!r} and a "
                                    "nested template's attribute flatten to the same name")
                out[key] = value
    return flat


def template_func_methods(cls) -> dict:
    """The ``@tack.func`` methods of a device class, inherited ones included.

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


def _own_func_attrs(obj) -> dict:
    return {name: value for name, value in vars(obj).items()
            if not name.startswith('_') and isinstance(value, Func)}


def template_func_attrs(obj) -> dict:
    """Device functions a template object holds as instance attributes.

    ``self.kernel = cubic_kernel`` lets the object's methods, and kernels
    it is passed to, call ``self.kernel(r, h)``. The function is resolved
    when the kernel is lowered, so which function it is belongs to the
    kernel's specialization, like a class-level constant. Nested templates'
    are included under their path.
    """
    return _memo("funcs", obj, lambda: _func_tree(obj))


def _func_tree(obj):
    # Flattened as the other attributes are, and refused on the same collision:
    # a nested function overwriting another under one name kept the second out
    # of the cache key, so replacing it ran stale code.
    taken = set().union(*classify_template_attrs(obj))
    flat = {}
    for path, node in template_tree(obj):
        for name, value in _own_func_attrs(node).items():
            key = _prefix(path) + name
            if key in flat or key in taken:
                raise TypeError(f"{type(obj).__name__}: the attribute {key!r} and a "
                                "nested template's attribute flatten to the same name")
            flat[key] = value
    return flat


def template_structure(obj) -> tuple:
    """The nested templates' paths and classes: part of a kernel's specialization."""
    return _memo("structure", obj, lambda: tuple(
        (path, type(node)) for path, node in template_tree(obj)[1:]))


def iteration_space(cls) -> tuple:
    """The attributes a ``for c in obj`` loop runs over, fastest first.

    A device class declares them as ``__tack_iterate__``: one to
    three names of its scalar attributes, class constants or instance
    values. One name makes ``for c in obj`` the loop ``for c in
    range(obj.name)``, with ``c`` an integer. Two or three make it an
    ``ndindex`` over them, with ``c`` the vector of indices, ``c[0]``
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


def template_array_param_name(param_name, attr_name):
    """The kernel parameter a template's array attribute becomes.

    The runtime predicts this name before lowering to tell the transform
    which of those parameters are vector arrays, so both sides must agree;
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
    # user's binding with an array/scalar reference in the rewritten AST.
    sources = [funcdef]
    for _, obj in template_args.values():
        for _, node in template_tree(obj):
            sources.extend(method._funcdef for method in
                           template_func_methods(type(node)).values())
    used_names = {n.id for source in sources for n in ast.walk(source)
                  if isinstance(n, ast.Name)}
    used_names.update(n.arg for source in sources for n in ast.walk(source)
                      if isinstance(n, ast.arg))

    # Process each template parameter (reverse order to keep indices stable)
    for idx in sorted(template_args.keys(), reverse=True):
        param_name, obj = template_args[idx]
        root, extra = _template_nodes(param_name, obj, used_names, resolved_funcs)

        # obj[i], obj[i] = v and len(obj) as the protocol's method calls
        _desugar_indexing(funcdef, param_name, root, used_names)
        # Rewrite the kernel function definition
        rewriter = _KernelTemplateRewriter(param_name, idx, root, extra, template=obj,
                                           used_names=used_names)
        rewriter.visit(funcdef)
        ast.fix_missing_locations(funcdef)

    return rewritten, resolved_funcs


_INDEXING = ("__getitem__", "__setitem__", "__len__")
_ATOMICS = {"atomic_add", "atomic_min", "atomic_max"}


class _Indexing(ast.NodeTransformer):
    """Python's indexing protocol on device-class objects, as method calls.

    ``obj[i]`` becomes ``obj.__getitem__(i)``, ``obj[i] = v``
    ``obj.__setitem__(i, v)``, ``obj[i] += v`` a get and a set with the
    index evaluated once, and ``len(obj)`` ``obj.__len__()``, wherever
    ``obj`` is ``root`` or a template it holds (``root.base``); the calls
    then resolve as any method's do. Several indices, ``obj[i, j]``, pass
    one vector, as a vector index supplies one dimension per component.
    A stored array, ``root.values[i]``, is left alone.
    """

    def __init__(self, root_name, root_node, used_names):
        self.root_name = root_name
        self.root_node = root_node
        self.used_names = used_names

    def _template(self, expr):
        """The template node ``expr`` names (``root`` or ``root.a.b``), or None."""
        if isinstance(expr, ast.Name):
            return self.root_node if expr.id == self.root_name else None
        names = _chain(expr, self.root_name)
        if not names:
            return None
        node = self.root_node
        for name in names:
            node = node.children.get(name)
            if node is None:
                return None
        return node

    @staticmethod
    def _key(subscript):
        key = subscript.slice
        if isinstance(key, ast.Tuple):
            if not key.elts:
                raise TypeError(f"{ast.unparse(subscript)}: a device-class object takes "
                                "one index or several, not an empty tuple")
            return ast.List(elts=key.elts, ctx=ast.Load())
        return key

    @staticmethod
    def _method(target, name, args):
        return ast.Call(func=ast.Attribute(value=target, attr=name, ctx=ast.Load()),
                        args=args, keywords=[])

    def _require(self, node, name, expr, what):
        if name not in node.methods:
            cls = type(node.obj).__name__
            raise TypeError(f"{ast.unparse(expr)}: {what} a {cls}, whose class defines no "
                            f"@tack.func {name}")

    def _tuple_entries(self, expr):
        """``(owner, name, count)`` if ``expr`` is ``root...name`` for a tuple
        attribute ``name`` of a template, else None."""
        if not isinstance(expr, ast.Attribute):
            return None
        owner = self._template(expr.value)
        if owner is None:
            return None
        prefix = f"{expr.attr}{_NESTED}"
        count = sum(1 for key in owner.runtime
                    if key.startswith(prefix) and key[len(prefix):].isdigit())
        return (owner, expr.attr, count) if count else None

    def visit_Subscript(self, node):
        entries = self._tuple_entries(node.value)
        if entries is not None:
            owner, name, count = entries
            index = node.slice
            literal = None
            if isinstance(index, ast.Constant) and type(index.value) is int:
                literal = index.value
            elif (isinstance(index, ast.UnaryOp) and isinstance(index.op, ast.USub)
                  and isinstance(index.operand, ast.Constant)
                  and type(index.operand.value) is int):
                literal = -index.operand.value
            if literal is None or not -count <= literal < count:
                raise TypeError(f"{ast.unparse(node)}: a tuple attribute takes a literal "
                                f"index from {-count} to {count - 1}")
            if not isinstance(node.ctx, ast.Load):
                raise TypeError(f"{ast.unparse(node)}: a tuple attribute is read-only")
            return ast.copy_location(
                ast.Name(id=owner.runtime[f"{name}{_NESTED}{literal % count}"],
                         ctx=ast.Load()), node)
        node = self.generic_visit(node)
        owner = self._template(node.value)
        if owner is None or not isinstance(node.ctx, ast.Load):
            return node
        self._require(owner, "__getitem__", node, "indexing")
        return ast.copy_location(
            self._method(node.value, "__getitem__", [self._key(node)]), node)

    def visit_Assign(self, node):
        templated = [t for t in node.targets if isinstance(t, ast.Subscript)
                     and self._template(t.value) is not None]
        nested = [e for t in node.targets if isinstance(t, (ast.Tuple, ast.List))
                  for e in t.elts if isinstance(e, ast.Subscript)
                  and self._template(e.value) is not None]
        if nested or (templated and len(node.targets) > 1):
            raise TypeError(f"{ast.unparse(node)}: a store into a device-class object "
                            "takes one target")
        if not templated:
            return self.generic_visit(node)
        target = templated[0]
        owner = self._template(target.value)
        self._require(owner, "__setitem__", target, "storing into")
        value = self.visit(node.value)
        key = self.visit(self._key(target))
        call = self._method(target.value, "__setitem__", [key, value])
        return ast.copy_location(ast.Expr(value=call), node)

    def visit_AugAssign(self, node):
        target = node.target
        owner = self._template(target.value) if isinstance(target, ast.Subscript) else None
        if owner is None:
            return self.generic_visit(node)
        self._require(owner, "__getitem__", target, "updating")
        self._require(owner, "__setitem__", target, "updating")
        value = self.visit(node.value)
        key = self.visit(self._key(target))
        statements = []
        if not isinstance(key, (ast.Name, ast.Constant)):
            temp = fresh_name("__tack_key__", self.used_names)
            self.used_names.add(temp)
            statements.append(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())],
                                         value=key))
            key = ast.Name(id=temp, ctx=ast.Load())
        old = self._method(target.value, "__getitem__", [key])
        new = ast.BinOp(left=old, op=node.op, right=value)
        statements.append(ast.Expr(value=self._method(target.value, "__setitem__",
                                                      [key, new])))
        for statement in statements:
            ast.copy_location(statement, node)
        return statements

    def visit_Call(self, node):
        node = self.generic_visit(node)
        func = node.func
        name = func.id if isinstance(func, ast.Name) else \
            func.attr if isinstance(func, ast.Attribute) else None
        if name == "len" and isinstance(func, ast.Name) and len(node.args) == 1:
            entries = self._tuple_entries(node.args[0])
            if entries is not None:
                return ast.copy_location(ast.Constant(value=entries[2]), node)
            owner = self._template(node.args[0])
            if owner is not None:
                self._require(owner, "__len__", node, "len() of")
                return ast.copy_location(self._method(node.args[0], "__len__", []), node)
        if name in _ATOMICS and node.args and self._template(node.args[0]) is not None:
            cls = type(self._template(node.args[0]).obj).__name__
            raise TypeError(f"{ast.unparse(node)}: an atomic updates an array's memory, "
                            f"and a {cls} has none to update")
        return node


def _desugar_indexing(funcdef, root_name, root_node, used_names):
    """Apply `_Indexing` to a function's body."""
    transformer = _Indexing(root_name, root_node, used_names)
    body = []
    for statement in funcdef.body:
        result = transformer.visit(statement)
        body.extend(result if isinstance(result, list) else [result])
    funcdef.body = body
    ast.fix_missing_locations(funcdef)


class _Node:
    """One object of a template tree, as its methods see it: its own constants, the
    synthetic names of its arrays and runtime scalars, its resolved methods and
    device-function attributes, and its child nodes."""

    def __init__(self, obj, path):
        self.obj = obj
        self.path = path
        self.scalars = {}
        self.arrays = {}
        self.runtime = {}
        self.methods = {}
        self.funcs = {}
        self.children = {}


def _template_nodes(param_name, obj, used_names, resolved_funcs):
    """The resolved tree of one template argument: ``(root, extra)``, ``extra`` the
    synthetic parameters every resolved method takes, in the runtime's order --
    the flattened arrays by name, then the flattened runtime scalars by name."""
    _, arrays, runtime = classify_template_attrs(obj)
    array_names = {key: fresh_name(template_array_param_name(param_name, key), used_names)
                   for key in sorted(arrays)}
    runtime_names = {key: fresh_name(f"__tmpl_{param_name}_{key}__", used_names)
                     for key in sorted(runtime)}
    extra = [array_names[k] for k in sorted(array_names)] + \
        [runtime_names[k] for k in sorted(runtime_names)]

    nodes = {}
    for path, node_obj in template_tree(obj):
        node = nodes[path] = _Node(node_obj, path)
        if path:
            nodes[path[:-1]].children[path[-1]] = node
        prefix = _prefix(path)
        scalars, own_arrays, own_runtime = _classify_own(node_obj)
        node.scalars = scalars
        node.arrays = {name: array_names[prefix + name] for name in own_arrays}
        node.runtime = {name: runtime_names[prefix + name] for name in own_runtime}
        # Device functions held as attributes are called as they are: they
        # take no self and no synthetic parameters. An attribute shadows a
        # method of the same name, as it does in Python.
        methods = template_func_methods(type(node_obj))
        for attr_name, func_obj in sorted(_own_func_attrs(node_obj).items()):
            resolved_name = fresh_name(f"__tmpl_{param_name}_{prefix}{attr_name}__",
                                       used_names)
            node.funcs[attr_name] = resolved_name
            resolved_funcs[resolved_name] = func_obj
            methods.pop(attr_name, None)
        # Names first, so methods can call siblings and nested objects' methods.
        node.methods = {name: fresh_name(f"__tmpl_{param_name}_{prefix}{name}__", used_names)
                        for name in methods}
    for node in nodes.values():
        for method_name, func_obj in template_func_methods(type(node.obj)).items():
            if method_name in node.methods:
                resolved_name = node.methods[method_name]
                resolved_funcs[resolved_name] = _resolve_method(func_obj, resolved_name,
                                                                node, extra)
    return nodes[()], extra


def _chain(expr, root):
    """The attribute names of ``root.a.b.c``, or None if ``expr`` is not one."""
    names = []
    while isinstance(expr, ast.Attribute):
        names.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name) and expr.id == root and names:
        return names[::-1]
    return None


def _walk(node, names, root):
    """The node reached through the child names ``names``."""
    for k, name in enumerate(names):
        if name not in node.children:
            raise ValueError(f"{'.'.join([root, *names[:k + 1]])} is not a template the "
                             f"{type(node.obj).__name__} holds")
        node = node.children[name]
    return node


def _resolve_call(node, call, names, root, extra):
    """``root.a.b.method(args)`` as a call of the resolved method, or None."""
    owner = _walk(node, names[:-1], root)
    last = names[-1]
    if last in owner.funcs:
        return ast.Call(func=_method_call_name(owner.funcs[last]), args=call.args,
                        keywords=[])
    if last in owner.methods:
        return ast.Call(func=_method_call_name(owner.methods[last]),
                        args=call.args + [ast.Name(id=n, ctx=ast.Load()) for n in extra],
                        keywords=[])
    return None


def _own_attribute(owner, name):
    """An object's own constant, array or runtime scalar as an expression, or None."""
    if name in owner.scalars:
        return ast.Constant(value=owner.scalars[name])
    if name in owner.arrays:
        return ast.Name(id=owner.arrays[name], ctx=ast.Load())
    if name in owner.runtime:
        return ast.Name(id=owner.runtime[name], ctx=ast.Load())
    return None


def _resolve_attribute(node, names, root, strict=True):
    """``root.a.b.attr...``: through the held templates ``a`` and ``b``, ``attr`` as a
    constant or a synthetic name, with any further attributes (an array's
    ``.shape``) read from it. None for a method or a template, which only a call
    or a longer chain may use; an unknown name raises when ``strict``."""
    owner = node
    for k, name in enumerate(names):
        if name in owner.children and k < len(names) - 1:
            owner = owner.children[name]
            continue
        base = _own_attribute(owner, name)
        if base is None:
            if name in owner.methods or name in owner.funcs or name in owner.children \
                    or not strict:
                return None
            raise ValueError(
                f"Template method references {'.'.join([root, *names[:k + 1]])} which is "
                f"neither a class constant, an instance scalar, a tack.Array, "
                f"a @tack.func method, a @tack.func held as an attribute, nor a template")
        for rest in names[k + 1:]:
            base = ast.Attribute(value=base, attr=rest, ctx=ast.Load())
        return base
    return None


def _resolve_method(func_obj, resolved_name, node, extra):
    """Resolve a template method while retaining its defining Python callable.

    The resolved copy has:
    - 'self' parameter removed
    - the tree's synthetic parameters appended
    - self.class_scalar replaced with constants
    - self.array_attr and self.instance_scalar replaced with synthetic parameter names
    - self.method(args) calls replaced with resolved function calls
    - and the same through templates held as attributes (self.base.get(k))
    """
    funcdef = copy.deepcopy(func_obj._funcdef)

    # Remove 'self' parameter
    funcdef.args.args = [a for a in funcdef.args.args if a.arg != 'self']
    for synth_name in extra:
        funcdef.args.args.append(ast.arg(arg=synth_name))

    # self[i], self.base[i] and len(self) as the protocol's method calls
    names = {n.id for n in ast.walk(funcdef) if isinstance(n, ast.Name)}
    _desugar_indexing(funcdef, 'self', node, names)

    # Resolve self.attr and self.method(args) references in the body
    resolver = _SelfResolver(node, extra)
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
    """Replaces self.attr with constants, synthetic parameter names, or method calls,
    through any templates the object holds."""

    def __init__(self, node, extra):
        self.node = node
        self.extra = extra

    def visit_Call(self, node):
        node = self.generic_visit(node)
        names = _chain(node.func, 'self')
        if names:
            call = _resolve_call(self.node, node, names, 'self', self.extra)
            if call is not None:
                return call
        return node

    def visit_Attribute(self, node):
        names = _chain(node, 'self')
        if names is None:
            return self.generic_visit(node)
        resolved = _resolve_attribute(self.node, names, 'self')
        return node if resolved is None else resolved


class _KernelTemplateRewriter(ast.NodeTransformer):
    """Rewrites a kernel function to resolve one template parameter."""

    def __init__(self, param_name, param_idx, root, extra, template=None, used_names=None):
        self.template = template
        self.used_names = used_names if used_names is not None else set()
        self.param_name = param_name
        self.param_idx = param_idx
        self.root = root
        self.extra = extra
        self.scalars = root.scalars
        self.runtime_scalar_param_map = root.runtime

    def visit_FunctionDef(self, node):
        # Remove the template parameter
        node.args.args = [
            a for i, a in enumerate(node.args.args) if i != self.param_idx
        ]
        # The tree's synthetic parameters at the end
        for synth_name in self.extra:
            node.args.args.append(ast.arg(arg=synth_name))

        # Visit the body
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        # for c in obj: becomes a range or ndindex loop over its iteration
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
        # ndindex runs slowest first; the index vector is fastest first.
        loop = ast.For(
            target=ast.Tuple(elts=[ast.Name(id=d, ctx=ast.Store()) for d in reversed(dims)],
                             ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id='ndindex', ctx=ast.Load()),
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
        names = _chain(node.func, self.param_name)
        if names:
            call = _resolve_call(self.root, node, names, self.param_name, self.extra)
            if call is not None:
                return call
            raise ValueError(
                f"Template object has no @tack.func method '{'.'.join(names)}', "
                f"and no @tack.func held as an attribute of that name"
            )
        return node

    def visit_Attribute(self, node):
        # Resolve direct attribute access on the template param
        names = _chain(node, self.param_name)
        if names is None:
            return self.generic_visit(node)
        resolved = _resolve_attribute(self.root, names, self.param_name, strict=False)
        # Could be a property or method name used without calling — skip
        return node if resolved is None else resolved
