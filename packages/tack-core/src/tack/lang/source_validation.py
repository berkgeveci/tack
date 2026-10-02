"""Reject source constructs before lowering can discard part of their AST.

Validate original device functions before return restructuring or renaming,
so unreachable unsupported syntax is diagnosed too. Locations refer to the
captured, dedented source; annotations and decorators are host metadata.
"""

import ast


class UnsupportedSyntaxError(NotImplementedError):
    """A source construct outside Tack's kernel language."""


class _SourceValidator(ast.NodeVisitor):
    _SUPPORTED = (
        ast.Assign, ast.AugAssign, ast.If, ast.Name, ast.Attribute,
        ast.Tuple, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Load, ast.Store,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor,
        ast.UAdd, ast.USub, ast.Not, ast.Invert, ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )

    def __init__(self, function, kind):
        self.function = function
        self.kind = kind
        self.loops = []
        self.location = function
        self.statement_call = None

    def visit(self, node):
        saved = self.location
        if hasattr(node, 'lineno'):
            self.location = node
        try:
            return super().visit(node)
        finally:
            self.location = saved

    def reject(self, node, detail=None):
        construct = type(node).__name__
        where = node if hasattr(node, 'lineno') else self.location
        location = f"line {where.lineno}, column {where.col_offset + 1}"
        raise UnsupportedSyntaxError(
            f"{self.kind} '{self.function.name}': unsupported {construct} "
            f"at {location}" + (f": {detail}" if detail else ""))

    def generic_visit(self, node):
        if not isinstance(node, self._SUPPORTED):
            self.reject(node)
        super().generic_visit(node)

    def validate(self):
        args = self.function.args
        if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg:
            self.reject(self.function, "only ordinary positional parameters are supported")
        if args.defaults or any(v is not None for v in args.kw_defaults):
            self.reject(self.function, "default parameter values are not supported")
        for stmt in self.function.body:
            self.visit(stmt)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return  # docstrings/standalone strings have no execution effect
        if not isinstance(node.value, ast.Call):
            self.visit(node.value)
            self.reject(node, "expression statements must be supported function calls")
        saved = self.statement_call
        self.statement_call = node.value
        self.visit(node.value)
        self.statement_call = saved

    def visit_Pass(self, node):
        pass  # explicit no-op, rather than NodeVisitor's implicit None

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float)):
            self.reject(node, "only numeric literals are supported here")

    def visit_Subscript(self, node):
        self.visit(node.value)
        if isinstance(node.slice, ast.Constant) and node.slice.value is None:
            return  # scalar field[None]
        self.visit(node.slice)

    def visit_IfExp(self, node):
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Call(self, node):
        if node.keywords:
            self.reject(node, "keyword arguments and **kwargs are not supported")
        self.visit(node.func)
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
        from tack.lang.func import _func_registry
        intrinsic = name not in _func_registry
        if intrinsic and name in ('atomic_add', 'atomic_min', 'atomic_max', 'barrier') \
                and node is not self.statement_call:
            self.reject(node, f"{name}() is only supported as a statement")
        if intrinsic and name in ('barrier', 'thread_id') and node.args:
            self.reject(node, f"{name}() takes no arguments")
        for arg in node.args:
            if intrinsic and name == 'print' and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                continue
            if intrinsic and name == 'Vector' and isinstance(arg, ast.List):
                if not arg.elts:
                    self.reject(arg, "Vector() requires at least one component")
                for element in arg.elts:
                    self.visit(element)
            else:
                self.visit(arg)

    def visit_For(self, node):
        if node.orelse:
            self.reject(node, "for-else is not supported")
        self.visit(node.target)
        self.visit(node.iter)
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) \
                and node.iter.func.id == 'range' and len(node.iter.args) == 3:
            step = node.iter.args[2]
            if isinstance(step, ast.Constant) and isinstance(step.value, int) and step.value <= 0:
                self.reject(step, "range() step must be positive")
            if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub) \
                    and isinstance(step.operand, ast.Constant):
                self.reject(step, "range() step must be positive")
        parallel = self.kind == 'Kernel' and not self.loops
        self.loops.append(parallel)
        for stmt in node.body:
            self.visit(stmt)
        self.loops.pop()

    def visit_While(self, node):
        if node.orelse:
            self.reject(node, "while-else is not supported")
        self.visit(node.test)
        self.loops.append(False)
        for stmt in node.body:
            self.visit(stmt)
        self.loops.pop()

    def visit_Break(self, node):
        if not self.loops:
            self.reject(node, "break must be inside a sequential loop")
        if self.loops[-1]:
            self.reject(node, "break from the parallel loop has no defined execution order")

    def visit_Continue(self, node):
        if not self.loops:
            self.reject(node, "continue must be inside a loop")

    def visit_Return(self, node):
        if self.kind == 'Kernel':
            self.reject(node, "kernels write results to fields; return is only supported in @tack.func")
        if self.loops:
            self.reject(node, "return inside a loop is not supported")
        if node.value is not None:
            self.visit(node.value)


def validate_source(function: ast.FunctionDef, kind='Kernel'):
    """Validate one original kernel or device function, ignoring host metadata."""
    _SourceValidator(function, kind).validate()
