"""What the C-family code generators share.

CUDA C, HIP C, OpenCL C and the Metal Shading Language spell statements,
control flow and most expressions alike. `CUDACodeGen` (and through it
`HIPCodeGen` and `OpenCLCodeGen`) and `MSLCodeGen` derive from
`CFamilyCodeGen`, which holds the methods that are the same for all of
them; each generator keeps what its dialect spells differently (types,
qualifiers, built-ins, kernel signatures, atomics) and sets up, in its own
`__init__`, the state these methods read (`_lines`, `_indent`,
`_local_vars`, the scope stack, ...).
"""

from tack.lang import ir
from tack.lang.types import f64

_CMP_MAP = {
    "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
}


class CFamilyCodeGen:
    """Methods shared by the CUDA-family and Metal code generators."""

    #: The dialect's name in error messages.
    _LABEL = "C"
    #: The 64-bit integer type an array index is cast to.
    _INDEX_TYPE = "long long"
    #: Whether a double-precision constant is spelled without the ``f``
    #: suffix; false for Metal, which has no f64.
    _HAS_F64 = True

    @staticmethod
    def _first_assignments(stmts: list, found: dict | None = None) -> dict:
        """Each name assigned in stmts, nested branches included, mapped to its first
        assignment in statement order (a branch's then before its else), found in
        one walk for every name an `if` hoists."""
        found = {} if found is None else found
        for stmt in stmts:
            if isinstance(stmt, ir.IRAssign):
                found.setdefault(stmt.target, stmt)
            elif isinstance(stmt, ir.IRIf):
                CFamilyCodeGen._first_assignments(stmt.then_body, found)
                if stmt.else_body:
                    CFamilyCodeGen._first_assignments(stmt.else_body, found)
        return found

    def _hoisted_type(self, name: str, branches) -> str | None:
        """The C type of name's first assignment in the first branch that assigns it."""
        for first in branches:
            stmt = first.get(name)
            if stmt is not None:
                found = self._assign_type(stmt)
                if found:
                    return found
        return None

    def _emit(self, line: str):
        self._lines.append("    " * self._indent + line)

    def _emit_array_store(self, node: ir.IRArrayStore):
        array = self._expr(node.array)
        index = self._expr(node.index)
        value = self._expr(node.value)
        value = self._integers.convert(value, getattr(node.value, 'dtype', None), getattr(node, 'dtype', None))
        # Ensure array index is integer
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({self._INDEX_TYPE})({index}))"
        self._emit(f"{array}[{index}] = {value};")

    def _emit_atomic_op(self, node: ir.IRAtomicOp):
        """Emit an atomic as a statement; its old value is dropped."""
        self._emit(f"{self._expr_atomic(node)};")

    def _emit_body(self, stmts: list):
        for stmt in stmts:
            self._emit_stmt(stmt)

    def _emit_if(self, node: ir.IRIf):
        # Pre-declare variables assigned in branches so they are visible at
        # the outer scope in C.
        branches = (self._first_assignments(node.then_body),
                    self._first_assignments(node.else_body or []))
        needs_hoist = {name for first in branches for name in first if name not in self._declared_vars}

        # Hoist with resolved types when available (from ir_type_annotate pass),
        # falling back to the two-pass inference for unannotated IR.
        hoist_types = {}
        for var_name in sorted(needs_hoist):
            if var_name not in self._declared_vars:
                c_type = self._hoisted_type(var_name, branches) or "float"
                hoist_types[var_name] = c_type
                self._local_vars[var_name] = c_type
        # Second pass re-infers non-float types now that referenced vars are
        # registered in _local_vars (needed when _resolved_type is absent).
        for var_name in list(hoist_types):
            if hoist_types[var_name] not in ("float", "double"):
                c_type = self._hoisted_type(var_name, branches) or hoist_types[var_name]
                hoist_types[var_name] = c_type
                self._local_vars[var_name] = c_type
        for var_name in sorted(hoist_types):
            self._emit(f"{hoist_types[var_name]} {var_name};")
            self._declared_vars.add(var_name)

        cond = self._expr(node.condition)
        self._emit(f"if ({cond}) {{")
        self._indent += 1
        self._emit_body(node.then_body)
        self._indent -= 1
        if node.else_body:
            self._emit("} else {")
            self._indent += 1
            self._emit_body(node.else_body)
            self._indent -= 1
        self._emit("}")

    def _emit_while(self, node: ir.IRWhile):
        cond = self._expr(node.condition)
        self._emit(f"while ({cond}) {{")
        self._indent += 1
        self._emit_body(node.body)
        self._indent -= 1
        self._emit("}")

    def _expr_boolop(self, node: ir.IRBoolOp) -> str:
        c_op = "&&" if node.op == "and" else "||"
        parts = [self._expr(v) for v in node.values]
        return "(" + f" {c_op} ".join(parts) + ")"

    def _expr_compare(self, node: ir.IRCompare) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        dtype = node._operand_type
        left = self._integers.convert(left, getattr(node.left, 'dtype', None), dtype)
        right = self._integers.convert(right, getattr(node.right, 'dtype', None), dtype)
        return f"({left} {_CMP_MAP[node.op]} {right})"

    def _get_array_name(self, node) -> str | None:
        if isinstance(node, ir.IRName):
            return node.name
        return None

    def _leave_loop_scope(self, var, outer):
        """Forget a for-header declaration once its block closes."""
        was_declared, outer_type = outer
        if was_declared:
            self._local_vars[var] = outer_type
        else:
            self._declared_vars.discard(var)
            self._local_vars.pop(var, None)

    def _expr_array_load(self, node: ir.IRArrayLoad) -> str:
        array = self._expr(node.array)
        index = self._expr(node.index)
        # Ensure array index is integer — float indices are invalid in C
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({self._INDEX_TYPE})({index}))"
        return f"{array}[{index}]"

    def _expr_attribute(self, node: ir.IRAttribute) -> str:
        raise NotImplementedError(
            f"Attribute access '{node.attr}' should be resolved before {self._LABEL} codegen."
        )

    def _expr_constant(self, node: ir.IRConstant) -> str:
        if isinstance(node.value, float):
            # repr round-trips, so an unsuffixed double literal is exact.
            if self._HAS_F64 and getattr(node, 'dtype', None) is f64:
                return repr(node.value)
            return f"{node.value!r}f"
        if isinstance(node.value, bool):
            return "1" if node.value else "0"
        if isinstance(node.value, int) and -(2**31) < node.value < 2**31:
            return str(node.value)
        if isinstance(node.value, int) and getattr(node, 'dtype', None) is not None:
            ctype = self._integer_type_map[node.dtype]
            # Spell signed minimum using representable tokens.
            literal = f'(-{-(node.value + 1)}LL - 1LL)' if node.value < 0 else f'{node.value}ULL'
            return f'(({ctype})({literal}))'
        return str(node.value)

    def _expr_ifexp(self, node: ir.IRIfExp) -> str:
        cond = self._ifexp_condition(node.condition)
        then = self._expr(node.then_value)
        else_ = self._expr(node.else_value)
        dtype = getattr(node, 'dtype', None)
        then = self._integers.convert(then, getattr(node.then_value, 'dtype', None), dtype)
        else_ = self._integers.convert(else_, getattr(node.else_value, 'dtype', None), dtype)
        return f"({cond} ? {then} : {else_})"

    def _expr_unaryop(self, node: ir.IRUnaryOp) -> str:
        operand = self._expr(node.operand)
        if node.op == "+":
            # Before the integer helpers, which know '+' only as addition
            # and would emit a one-argument call to the two-argument add.
            return operand
        op = 'neg' if node.op == '-' else node.op
        fixed = self._integers.operation(op, getattr(node, 'dtype', None), operand)
        if fixed is not None:
            return fixed
        if node.op == "-":
            return f"(-{operand})"
        if node.op == "not":
            return f"(!{operand})"
        if node.op == "~":
            return f"(~{operand})"
        raise NotImplementedError(f"{self._LABEL} unaryop: {node.op}")

    def _ifexp_condition(self, node) -> str:
        """Render the condition of a ternary.

        C and C++ convert any scalar to bool here, so the condition needs no
        help. OpenCL C does not: see ``OpenCLCodeGen._ifexp_condition``.
        """
        return self._expr(node)
