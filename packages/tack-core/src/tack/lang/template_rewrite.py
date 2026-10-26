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
import re
import threading
import weakref

from tack.lang.array import Array
from tack.lang.ast_copy import copy_ast
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
                if isinstance(val, (int, float)) or _number_tuple(val):
                    found[name] = val
                else:
                    found.pop(name, None)   # replaced by something that is not a constant
        names = _CONSTANT_NAMES[cls] = tuple(found)
    constants = {}
    for name in names:
        val = getattr(cls, name)
        if isinstance(val, (int, float)):
            constants[name] = val
        elif _number_tuple(val):
            # A class tuple of numbers: one constant per entry, read as
            # obj.shape[k] as an instance tuple is, but compiled in.
            for k, entry in enumerate(val):
                constants[f"{name}{_NESTED}{k}"] = entry
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
        entries = [key for key in scalars
                   if scalar_path(key)[0] == name and len(scalar_path(key)) == 2]
        if entries:
            # A class tuple of numbers, one constant per entry: an instance
            # tuple of numbers replacing it is still compiled in; anything
            # else drops it, as for a number above.
            for key in entries:
                del scalars[key]
            if _number_tuple(val):
                for k, entry in enumerate(val):
                    scalars[f"{name}{_NESTED}{k}"] = entry
                continue
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


def rewrite_templates(kernel_ast, template_args, python_func=None):
    """Rewrite a kernel AST to resolve template parameters.

    Args:
        kernel_ast: The kernel's Python AST (will be deep-copied)
        template_args: dict of param_index -> (param_name, template_object)
        python_func: the kernel's Python function, whose namespace resolves the
            device functions a template is passed to

    Returns:
        (rewritten_ast, resolved_funcs) — the rewritten AST and a mapping
        of synthetic call names to resolved device functions. The mapping
        belongs to this transformation and never enters global state.
    """
    rewritten = copy_ast(kernel_ast)
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

    # The source spelling of each held array's synthetic parameter, for
    # diagnostics; lowering keeps it on the IR (`IRFunction.held_arrays`).
    funcdef._tack_held_arrays = {}
    specializer = _Specializer(resolved_funcs, used_names)
    # The templates resolved so far: a call is specialized for all of them at
    # once, and one whose callee is a method of a template still to come waits
    # for it (`b.combine(a, i)` while `a` is being resolved).
    roots = {}
    pending = {name for name, _ in template_args.values()}
    # Process each template parameter (reverse order to keep indices stable)
    for idx in sorted(template_args.keys(), reverse=True):
        param_name, obj = template_args[idx]
        pending.discard(param_name)
        root, extra = _template_nodes(param_name, obj, used_names, resolved_funcs,
                                      funcdef._tack_held_arrays, specializer)

        # obj[i], obj[i] = v and len(obj) as the protocol's method calls
        _desugar_indexing(funcdef, param_name, root, used_names)
        # Rewrite the kernel function definition
        roots[param_name] = root
        rewriter = _KernelTemplateRewriter(param_name, idx, root, extra, template=obj,
                                           used_names=used_names, specializer=specializer,
                                           funcdef=funcdef, python_func=python_func,
                                           roots=roots, pending=frozenset(pending))
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

    def __init__(self, root_name, root_node, used_names, lazy=False):
        self.root_name = root_name
        self.root_node = root_node
        self.used_names = used_names
        # In a method, a refusal waits until the method is used: every method
        # of an object is resolved, and a read-only one's __setitem__ is fine
        # until a kernel stores. The call lowers to the error.
        self.lazy = lazy

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
        """None if ``node``'s class defines ``name``; else, in a method, the call
        that refuses when lowered, and in a kernel the error at once."""
        if name in node.methods:
            return None
        cls = type(node.obj).__name__
        message = (f"{ast.unparse(expr)}: {what} a {cls}, whose class defines no "
                   f"@tack.func {name}")
        if not self.lazy:
            raise TypeError(message)
        return ast.Call(func=ast.Name(id="__tack_refuse__", ctx=ast.Load()),
                        args=[ast.Constant(value=message)], keywords=[])

    def _tuple_entries(self, expr):
        """``(owner, name, count)`` if ``expr`` is ``root...name`` for a tuple
        attribute ``name`` of a template, else None."""
        if not isinstance(expr, ast.Attribute):
            return None
        owner = self._template(expr.value)
        if owner is None:
            return None
        prefix = f"{expr.attr}{_NESTED}"
        count = sum(1 for key in (*owner.runtime, *owner.scalars)
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
            key = f"{name}{_NESTED}{literal % count}"
            entry = ast.Constant(value=owner.scalars[key]) if key in owner.scalars \
                else ast.Name(id=owner.runtime[key], ctx=ast.Load())
            return ast.copy_location(entry, node)
        node = self.generic_visit(node)
        owner = self._template(node.value)
        if owner is None or not isinstance(node.ctx, ast.Load):
            return node
        refusal = self._require(owner, "__getitem__", node, "indexing")
        if refusal is not None:
            return ast.copy_location(refusal, node)
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
        refusal = self._require(owner, "__setitem__", target, "storing into")
        if refusal is not None:
            return ast.copy_location(ast.Expr(value=refusal), node)
        value = self.visit(node.value)
        key = self.visit(self._key(target))
        call = self._method(target.value, "__setitem__", [key, value])
        return ast.copy_location(ast.Expr(value=call), node)

    def visit_AugAssign(self, node):
        target = node.target
        owner = self._template(target.value) if isinstance(target, ast.Subscript) else None
        if owner is None:
            return self.generic_visit(node)
        for name in ("__getitem__", "__setitem__"):
            refusal = self._require(owner, name, target, "updating")
            if refusal is not None:
                return ast.copy_location(ast.Expr(value=refusal), node)
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
                refusal = self._require(owner, "__len__", node, "len() of")
                return ast.copy_location(
                    refusal or self._method(node.args[0], "__len__", []), node)
        if name in _ATOMICS and node.args and self._template(node.args[0]) is not None:
            cls = type(self._template(node.args[0]).obj).__name__
            raise TypeError(f"{ast.unparse(node)}: an atomic updates an array's memory, "
                            f"and a {cls} has none to update")
        return node


def _desugar_indexing(funcdef, root_name, root_node, used_names, lazy=False):
    """Apply `_Indexing` to a function's body."""
    transformer = _Indexing(root_name, root_node, used_names, lazy)
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

    def __init__(self, obj, path, extra=()):
        self.obj = obj
        self.path = path
        # The tree's synthetic parameters, which every method of the tree
        # takes, and which a function given this object takes too.
        self.extra = list(extra)
        self.scalars = {}
        self.arrays = {}
        self.runtime = {}
        self.methods = {}
        self.funcs = {}
        self.children = {}


def _template_nodes(param_name, obj, used_names, resolved_funcs, held_arrays,
                    specializer=None):
    """The resolved tree of one template argument: ``(root, extra)``, ``extra`` the
    synthetic parameters every resolved method takes, in the runtime's order --
    the flattened arrays by name, then the flattened runtime scalars by name.
    Each array's synthetic parameter goes into ``held_arrays`` with the path
    the source writes: ``__tmpl_dst_base__values__`` -> ``dst.base.values``."""
    _, arrays, runtime = classify_template_attrs(obj)
    array_names = {key: fresh_name(template_array_param_name(param_name, key), used_names)
                   for key in sorted(arrays)}
    for key, synthetic in array_names.items():
        held_arrays[synthetic] = f"{param_name}.{key.replace(_NESTED, '.')}"
    runtime_names = {key: fresh_name(f"__tmpl_{param_name}_{key}__", used_names)
                     for key in sorted(runtime)}
    extra = [array_names[k] for k in sorted(array_names)] + \
        [runtime_names[k] for k in sorted(runtime_names)]

    nodes = {}
    for path, node_obj in template_tree(obj):
        node = nodes[path] = _Node(node_obj, path, extra)
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
                                                                node, extra, specializer)
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


def _resolve_method(func_obj, resolved_name, node, extra, specializer=None):
    """Resolve a template method while retaining its defining Python callable.

    The resolved copy has:
    - 'self' parameter removed
    - the tree's synthetic parameters appended
    - self.class_scalar replaced with constants
    - self.array_attr and self.instance_scalar replaced with synthetic parameter names
    - self.method(args) calls replaced with resolved function calls
    - and the same through templates held as attributes (self.base.get(k))
    """
    funcdef = copy_ast(func_obj._funcdef)

    # Remove 'self' parameter
    funcdef.args.args = [a for a in funcdef.args.args if a.arg != 'self']
    for synth_name in extra:
        funcdef.args.args.append(ast.arg(arg=synth_name))

    # self[i], self.base[i] and len(self) as the protocol's method calls
    names = {n.id for n in ast.walk(funcdef) if isinstance(n, ast.Name)}
    _desugar_indexing(funcdef, 'self', node, names, lazy=True)

    # Resolve self.attr and self.method(args) references in the body, and
    # specialize what self (or a template it holds) is passed to
    resolver = _BoundResolver({'self': node}, specializer, funcdef, func_obj.func)
    for i, stmt in enumerate(funcdef.body):
        funcdef.body[i] = resolver.visit(stmt)

    funcdef.name = resolved_name
    ast.fix_missing_locations(funcdef)

    return _ResolvedFunc(resolved_name, funcdef, func_obj.func, {'self': node})


class _ResolvedFunc(Func):
    """A method body with template attributes resolved and bindings retained.

    ``roots`` are the names bound to template nodes when it was resolved (a
    method's ``self``, a copy's bound parameters): a call it could not resolve
    yet, because its callee hangs off a parameter still to be bound, is
    resolved in a later copy, which binds these too."""

    def __init__(self, name, funcdef, python_func, roots=None):
        self.name = name
        self._funcdef = funcdef
        self.func = python_func
        self.roots = dict(roots or {})


def _template_argument(expr, roots):
    """The template node ``expr`` names -- a bound name (``self``, a kernel's or a
    function's template parameter) or a template it holds (``self.base``) --
    or None."""
    if isinstance(expr, ast.Name):
        return roots.get(expr.id)
    names = []
    while isinstance(expr, ast.Attribute):
        names.append(expr.attr)
        expr = expr.value
    if not (isinstance(expr, ast.Name) and expr.id in roots):
        return None
    node = roots[expr.id]
    for name in reversed(names):
        if name not in node.children:
            return None
        node = node.children[name]
    return node


class _Specializer:
    """Copies of device functions with a template parameter bound, for one rewrite.

    A call that passes a device-class object -- or a template it holds -- to a
    ``@tack.func`` or to a template's method gets a copy of the callee in which
    that parameter is bound to the object, as a method's ``self`` is: its
    attributes and methods resolve through the object's node, its own
    template calls are specialized in turn, and the object's tree's synthetic
    parameters are appended, which the call passes. Copies are shared by
    callee and bound objects; the copy keeps the callee's Python function, so
    its body resolves names where the callee was defined.
    """

    def __init__(self, resolved_funcs, used_names):
        self.resolved_funcs = resolved_funcs
        self.used_names = used_names
        self.copies = {}

    def call(self, call, roots, funcdef, python_func):
        """``call`` with any template arguments bound into a copy of its callee."""
        if self is None or not isinstance(call, ast.Call):
            return call
        bound = {}
        for position, arg in enumerate(call.args):
            node = _template_argument(arg, roots)
            if node is not None:
                bound[position] = node
        if not bound:
            return call
        func = call.func
        while isinstance(func, ast.Attribute):
            func = func.value
        if (isinstance(func, ast.Name) and not getattr(func, '_tack_template_call', False)
                and func.id not in roots
                and func.id in {a.arg for a in funcdef.args.args}):
            # The callee hangs off a parameter still to be bound (``mesh.f(self)``
            # in a method given ``mesh``): the copy that binds it resolves this.
            return call
        callee_def, callee_func, inherited = self._callee(call, funcdef, python_func)
        params = [a.arg for a in callee_def.args.args]
        if len(call.args) > len(params):
            raise TypeError(f"'{callee_def.name}' takes {len(params)} arguments, "
                            f"given {len(call.args)}")
        key = (id(callee_def), tuple(sorted((k, id(n)) for k, n in bound.items())))
        if key not in self.copies:
            self._copy(key, callee_def, callee_func, bound, params, inherited)
        name, added = self.copies[key]
        args = [a for k, a in enumerate(call.args) if k not in bound]
        args += [ast.Name(id=n, ctx=ast.Load()) for n in added]
        return ast.copy_location(ast.Call(func=_method_call_name(name), args=args,
                                          keywords=call.keywords), call)

    def _callee(self, call, funcdef, python_func):
        """The callee's definition and Python function: a resolved method or copy,
        or a ``@tack.func`` bound where the calling function was defined."""
        func = call.func
        if getattr(func, '_tack_template_call', False):
            resolved = self.resolved_funcs[func.id]
            return (resolved._funcdef, getattr(resolved, 'func', None),
                    getattr(resolved, 'roots', {}))
        from tack.lang.call_bindings import CallBindings

        bindings = CallBindings(funcdef, python_func=python_func,
                                template_funcs=self.resolved_funcs)
        callee = bindings.device_func(func)
        if callee is None:
            raise TypeError(f"'{ast.unparse(func)}' is given a device-class object, but is "
                            "not a @tack.func or a template's method")
        return callee._funcdef, callee.func, {}

    def _copy(self, key, callee_def, callee_func, bound, params, inherited=None):
        """Make the copy, recorded under ``key`` as ``(name, added)``: its synthetic
        name and the parameters it gained."""
        copied = copy_ast(callee_def)
        bound_names = {params[k]: node for k, node in sorted(bound.items())}
        copied.args.args = [a for k, a in enumerate(copied.args.args) if k not in bound]
        present = {a.arg for a in copied.args.args}
        added = []
        for node in bound_names.values():
            for name in node.extra:
                if name not in present and name not in added:
                    added.append(name)
        copied.args.args += [ast.arg(arg=n) for n in added]
        name = fresh_name(f"__tmpl_{callee_def.name}_{'_'.join(bound_names)}__",
                          self.used_names)
        self.used_names.add(name)
        # Recorded before the body resolves: a copy reached again from inside
        # itself is this one (and lowering then refuses the recursion).
        self.copies[key] = (name, added)
        names = {n.id for n in ast.walk(copied) if isinstance(n, ast.Name)}
        for param, node in bound_names.items():
            _desugar_indexing(copied, param, node, names, lazy=True)
        # The callee's own roots (a method's self) stay bound beside the new ones.
        roots = {**(inherited or {}), **bound_names}
        resolver = _BoundResolver(roots, self, copied, callee_func)
        copied.body = [resolver.visit(stmt) for stmt in copied.body]
        copied.name = name
        ast.fix_missing_locations(copied)
        self.resolved_funcs[name] = _ResolvedFunc(name, copied, callee_func, roots)


def _refuse_kept(node, roots, where):
    """A device-class object bound to another name, or returned, is refused: the
    template is resolved by name, so a second name would not be."""
    value = node.value
    if value is None or _template_argument(value, roots) is None:
        return
    what = ("returns" if isinstance(node, ast.Return)
            else f"binds to '{ast.unparse(node.targets[0])}'")
    raise TypeError(f"{where} {what} the device-class object '{ast.unparse(value)}', "
                    "which can only be used or passed on by its own name")


class _BoundResolver(ast.NodeTransformer):
    """Within one function, names bound to template nodes (a method's ``self``, a
    copy's bound parameters): their attributes become constants or synthetic
    names, their method calls the resolved methods, through the templates they
    hold; and a call passing one of them is specialized."""

    def __init__(self, roots, specializer, funcdef, python_func):
        self.roots = roots
        self.specializer = specializer
        self.funcdef = funcdef
        self.python_func = python_func

    def visit_Call(self, node):
        node = self.generic_visit(node)
        for root, owner in self.roots.items():
            names = _chain(node.func, root)
            if names:
                call = _resolve_call(owner, node, names, root, owner.extra)
                if call is not None:
                    node = call
                break
        if self.specializer is not None:
            node = self.specializer.call(node, self.roots, self.funcdef, self.python_func)
        return node

    def visit_Attribute(self, node):
        for root, owner in self.roots.items():
            names = _chain(node, root)
            if names is not None:
                resolved = _resolve_attribute(owner, names, root)
                return node if resolved is None else resolved
        return self.generic_visit(node)

    def visit_Assign(self, node):
        _refuse_kept(node, self.roots, f"'{self.funcdef.name}'")
        return self.generic_visit(node)

    def visit_Return(self, node):
        _refuse_kept(node, self.roots, f"'{self.funcdef.name}'")
        return self.generic_visit(node)


class _KernelTemplateRewriter(ast.NodeTransformer):
    """Rewrites a kernel function to resolve one template parameter."""

    def __init__(self, param_name, param_idx, root, extra, template=None, used_names=None,
                 specializer=None, funcdef=None, python_func=None, roots=None,
                 pending=frozenset()):
        self.template = template
        self.specializer = specializer
        self.funcdef = funcdef
        self.python_func = python_func
        self.roots = roots if roots is not None else {param_name: root}
        self.pending = pending
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

    def visit_Assign(self, node):
        _refuse_kept(node, self.roots, "the kernel")
        return self.generic_visit(node)

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
            if call is None:
                raise ValueError(
                    f"Template object has no @tack.func method '{'.'.join(names)}', "
                    f"and no @tack.func held as an attribute of that name"
                )
            node = call
        # A template, or one it holds, passed to a function or method -- unless
        # the callee is a method of a template not resolved yet, which binds
        # them when it is
        waiting = isinstance(node, ast.Call) and any(
            _chain(node.func, name) for name in self.pending)
        if self.specializer is not None and not waiting:
            node = self.specializer.call(node, self.roots, self.funcdef, self.python_func)
        return node

    def visit_Attribute(self, node):
        # Resolve direct attribute access on the template param
        names = _chain(node, self.param_name)
        if names is None:
            return self.generic_visit(node)
        resolved = _resolve_attribute(self.root, names, self.param_name, strict=False)
        # Could be a property or method name used without calling — skip
        return node if resolved is None else resolved
