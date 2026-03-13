"""PGC AST transformation — Python AST to PGC IR.

Transforms a @kernel-decorated Python function's AST into PGC's internal IR.
Supports: parallel/sequential for-range loops, while loops, if/else, comparisons,
boolean ops, unary ops, math builtins, field indexing, and type inference.

Inspired by Taichi's AST transformer (Apache 2.0), simplified for PGC's needs.
"""

import ast
import math

from pgc.lang import ir


# Math builtins that map to LLVM intrinsics / libm calls
MATH_BUILTINS = {
    "sqrt", "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2",
    "exp", "log", "log2", "log10",
    "floor", "ceil",
    "abs", "min", "max", "pow",
    "fabs",
}


class KernelTransformer(ast.NodeVisitor):
    """Transforms a Python AST (from a @kernel function) into PGC IR.

    Tracks nesting depth to distinguish parallel for-loops (top-level)
    from sequential for-loops (nested inside another for-loop).
    """

    def __init__(self):
        self._loop_depth = 0

    def visit_Module(self, node: ast.Module) -> ir.IRModule:
        module = ir.IRModule()
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                module.functions.append(self.visit_FunctionDef(stmt))
        return module

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ir.IRFunction:
        params = []
        for arg in node.args.args:
            params.append(ir.IRParam(
                name=arg.arg,
                type_annotation=None,  # resolved during type inference
            ))
        body = self._visit_body(node.body)
        return ir.IRFunction(name=node.name, params=params, body=body)

    def _visit_body(self, stmts: list) -> list:
        """Visit a list of statements, filtering out None results."""
        result = []
        for stmt in stmts:
            node = self.visit(stmt)
            if node is not None:
                result.append(node)
        return result

    # --- Loops ---

    def visit_For(self, node: ast.For) -> ir.IRNode:
        target = node.target
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Only simple loop variables supported")

        if node.orelse:
            raise NotImplementedError("'else' clause on for-loop not supported in kernels")

        # Detect for i in range(...)
        if not self._is_range_call(node.iter):
            raise NotImplementedError("Only range() loops supported in kernels")

        start, end = self._parse_range_args(node.iter)

        self._loop_depth += 1
        body = self._visit_body(node.body)
        self._loop_depth -= 1

        # Top-level for-range is parallel; nested for-range is sequential
        if self._loop_depth == 0:
            return ir.IRParallelFor(var=target.id, start=start, end=end, body=body)
        else:
            return ir.IRSequentialFor(var=target.id, start=start, end=end, body=body)

    def visit_While(self, node: ast.While) -> ir.IRWhile:
        if node.orelse:
            raise NotImplementedError("'else' clause on while-loop not supported in kernels")
        condition = self.visit(node.test)
        body = self._visit_body(node.body)
        return ir.IRWhile(condition=condition, body=body)

    def visit_Break(self, node: ast.Break) -> ir.IRBreak:
        return ir.IRBreak()

    def visit_Continue(self, node: ast.Continue) -> ir.IRContinue:
        return ir.IRContinue()

    # --- Control flow ---

    def visit_If(self, node: ast.If) -> ir.IRIf:
        condition = self.visit(node.test)
        then_body = self._visit_body(node.body)
        else_body = self._visit_body(node.orelse)
        return ir.IRIf(condition=condition, then_body=then_body, else_body=else_body)

    def visit_IfExp(self, node: ast.IfExp) -> ir.IRIfExp:
        return ir.IRIfExp(
            condition=self.visit(node.test),
            then_value=self.visit(node.body),
            else_value=self.visit(node.orelse),
        )

    # --- Assignments ---

    def visit_Assign(self, node: ast.Assign) -> ir.IRNode:
        if len(node.targets) != 1:
            raise NotImplementedError("Multiple assignment targets not supported")

        target = node.targets[0]
        value = self.visit(node.value)

        # field[i] = expr  →  IRFieldStore
        if isinstance(target, ast.Subscript):
            field = self.visit(target.value)
            index = self._visit_subscript_index(target)
            return ir.IRFieldStore(field=field, index=index, value=value)

        # x = expr  →  IRAssign
        if isinstance(target, ast.Name):
            return ir.IRAssign(target=target.id, value=value)

        raise NotImplementedError(f"Unsupported assignment target: {type(target).__name__}")

    def visit_AugAssign(self, node: ast.AugAssign) -> ir.IRNode:
        # x += expr  →  x = x + expr
        target = node.target
        op = self._binop_str(node.op)
        rhs = ir.IRBinOp(op=op, left=self.visit(target), right=self.visit(node.value))

        if isinstance(target, ast.Subscript):
            field = self.visit(target.value)
            index = self._visit_subscript_index(target)
            return ir.IRFieldStore(field=field, index=index, value=rhs)

        if isinstance(target, ast.Name):
            return ir.IRAssign(target=target.id, value=rhs)

        raise NotImplementedError("Unsupported augmented assignment target")

    def visit_Return(self, node: ast.Return) -> ir.IRReturn:
        value = self.visit(node.value) if node.value else None
        return ir.IRReturn(value=value)

    # --- Expressions ---

    def visit_BinOp(self, node: ast.BinOp) -> ir.IRBinOp:
        op = self._binop_str(node.op)
        return ir.IRBinOp(
            op=op,
            left=self.visit(node.left),
            right=self.visit(node.right),
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.IRUnaryOp:
        op = self._unaryop_str(node.op)
        return ir.IRUnaryOp(op=op, operand=self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> ir.IRNode:
        # Desugar chained comparisons: a < b < c  →  (a < b) and (b < c)
        comparisons = []
        left = self.visit(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op = self._cmpop_str(op_node)
            comparisons.append(ir.IRCompare(op=op, left=left, right=right))
            left = right

        if len(comparisons) == 1:
            return comparisons[0]
        return ir.IRBoolOp(op="and", values=comparisons)

    def visit_BoolOp(self, node: ast.BoolOp) -> ir.IRBoolOp:
        op = "and" if isinstance(node.op, ast.And) else "or"
        values = [self.visit(v) for v in node.values]
        return ir.IRBoolOp(op=op, values=values)

    def visit_Subscript(self, node: ast.Subscript) -> ir.IRFieldLoad:
        return ir.IRFieldLoad(
            field=self.visit(node.value),
            index=self._visit_subscript_index(node),
        )

    def visit_Attribute(self, node: ast.Attribute) -> ir.IRAttribute:
        return ir.IRAttribute(
            obj=self.visit(node.value),
            attr=node.attr,
        )

    def visit_Name(self, node: ast.Name) -> ir.IRName:
        return ir.IRName(name=node.id)

    def visit_Constant(self, node: ast.Constant) -> ir.IRConstant:
        return ir.IRConstant(value=node.value)

    def visit_Tuple(self, node: ast.Tuple) -> list:
        """Visit tuple — used for multi-dimensional indexing like field[i, j]."""
        return [self.visit(elt) for elt in node.elts]

    def visit_Call(self, node: ast.Call) -> ir.IRNode:
        func_name = self._resolve_call_name(node)

        # Math builtins from the math module or bare names
        if func_name in MATH_BUILTINS:
            args = [self.visit(arg) for arg in node.args]
            return ir.IRCall(func_name=func_name, args=args)

        # len(field) → attribute access to shape
        if func_name == "len":
            if len(node.args) != 1:
                raise NotImplementedError("len() takes exactly one argument")
            arg = self.visit(node.args[0])
            return ir.IRAttribute(obj=arg, attr="__len__")

        # range() is handled in visit_For; if it appears elsewhere, error
        if func_name == "range":
            raise NotImplementedError("range() outside of for-loop not supported")

        # int(), float() type casts
        if func_name in ("int", "float"):
            if len(node.args) != 1:
                raise NotImplementedError(f"{func_name}() takes exactly one argument")
            return ir.IRCast(value=self.visit(node.args[0]), dtype=func_name)

        raise NotImplementedError(f"Function call '{func_name}' not supported in kernels")

    def visit_Expr(self, node: ast.Expr) -> ir.IRNode:
        """Expression statement (e.g., standalone function call)."""
        return self.visit(node.value)

    # --- Helpers ---

    def _visit_subscript_index(self, node: ast.Subscript):
        """Extract index from a subscript, handling tuples for multi-dim."""
        index = self.visit(node.slice)
        return index

    def _is_range_call(self, node: ast.expr) -> bool:
        """Check if an AST node is a call to range()."""
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "range")

    def _parse_range_args(self, call_node: ast.Call):
        """Parse range(end) or range(start, end) call arguments."""
        args = call_node.args
        if len(args) == 1:
            return ir.IRConstant(0), self.visit(args[0])
        elif len(args) == 2:
            return self.visit(args[0]), self.visit(args[1])
        else:
            raise NotImplementedError("range() with step not yet supported")

    def _resolve_call_name(self, node: ast.Call) -> str:
        """Resolve the function name from a Call node.

        Handles:
          - bare name: sqrt(x) → "sqrt"
          - module attribute: math.sqrt(x) → "sqrt"
          - pgc attribute: pgc.sqrt(x) → "sqrt"
        """
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            # math.sqrt, pgc.sqrt → just "sqrt"
            return func.attr
        raise NotImplementedError(f"Unsupported function call syntax: {ast.dump(func)}")

    def _binop_str(self, op: ast.operator) -> str:
        ops = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
            ast.LShift: "<<",
            ast.RShift: ">>",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.BitAnd: "&",
        }
        op_type = type(op)
        if op_type not in ops:
            raise NotImplementedError(f"Unsupported binary operator: {op_type.__name__}")
        return ops[op_type]

    def _unaryop_str(self, op: ast.unaryop) -> str:
        ops = {
            ast.UAdd: "+",
            ast.USub: "-",
            ast.Not: "not",
            ast.Invert: "~",
        }
        op_type = type(op)
        if op_type not in ops:
            raise NotImplementedError(f"Unsupported unary operator: {op_type.__name__}")
        return ops[op_type]

    def _cmpop_str(self, op: ast.cmpop) -> str:
        ops = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        op_type = type(op)
        if op_type not in ops:
            raise NotImplementedError(f"Unsupported comparison operator: {op_type.__name__}")
        return ops[op_type]


def transform_kernel(kernel_ast: ast.Module) -> ir.IRModule:
    """Transform a kernel's Python AST into PGC IR."""
    transformer = KernelTransformer()
    return transformer.visit(kernel_ast)
