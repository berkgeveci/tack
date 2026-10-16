"""Device functions (`@tack.func`), inlined at their call sites.

A call is replaced by the callee's body with its locals renamed apart
(`_NameRenamer`), its parameters bound to the arguments in order, and its
returns restructured into assignments of a result variable
(`_structure_returns`). Arrays, textures, vectors and structure-of-arrays
arrays passed in keep what the caller knows about them; shared and local
arrays and textures are aliased rather than bound.
"""

import ast
import copy

from tack.lang import ir
from tack.lang.lowering.values import TupleValue
from tack.lang.source_validation import validate_source


def _has_return(stmts: list) -> bool:
    return any(isinstance(n, ast.Return) for s in stmts for n in ast.walk(s))


def _structure_returns(stmts: list, func_name: str) -> list:
    """Make every `return` the last statement on its path.

    Inlining lowers `return x` to an assignment, which does not leave the
    function: after `if c: return a` the statements that follow would still
    run and overwrite the result. Statements after an `if` that returns are
    moved into its branches, so each path ends at its own return.
    """
    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, ast.Return):
            return stmts[:idx + 1]  # the rest is unreachable
        if isinstance(stmt, (ast.For, ast.While)) and _has_return([stmt]):
            raise NotImplementedError(
                f"@tack.func '{func_name}': 'return' inside a loop is not "
                f"supported. Assign the result to a variable, 'break', and "
                f"return after the loop.")
        if isinstance(stmt, ast.If) and _has_return([stmt]):
            rest = stmts[idx + 1:]
            stmt.body = _structure_returns(
                stmt.body + copy.deepcopy(rest), func_name)
            stmt.orelse = _structure_returns(stmt.orelse + rest, func_name)
            return [*stmts[:idx], stmt]
    return stmts


class _NameRenamer(ast.NodeTransformer):
    """Renames variables in an AST subtree and converts return → assignment.

    result_var can be:
    - str: single return value → one assignment
    - list[str]: multi-return (tuple) → multiple assignments
    """

    def __init__(self, rename_map: dict[str, str], result_var):
        self._rename_map = rename_map
        self._result_var = result_var  # str or list[str]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self._rename_map:
            # Keep the position and the source spelling for diagnostics.
            renamed = ast.copy_location(ast.Name(id=self._rename_map[node.id], ctx=node.ctx), node)
            renamed._source_id = node.id
            return renamed
        return node

    def visit_Return(self, node: ast.Return):
        """Convert 'return expr' → assignment(s).

        Single return: 'return x' → '__func_ret_N__ = x'
        Multi-return:  'return (a, b)' → '__func_ret_0_N__ = a; __func_ret_1_N__ = b'
        """
        value = self.visit(node.value) if node.value else ast.Constant(value=0)

        # Multi-return: result_var is a list, value should be a Tuple
        if isinstance(self._result_var, list):
            if isinstance(value, ast.Tuple):
                stmts = []
                for var, elt in zip(self._result_var, value.elts):
                    stmts.append(ast.copy_location(ast.Assign(
                        targets=[ast.Name(id=var, ctx=ast.Store())],
                        value=elt,
                    ), node))
                return stmts
            # Single expression returned but expected multi — assign to first
            return ast.copy_location(ast.Assign(
                targets=[ast.Name(id=self._result_var[0], ctx=ast.Store())],
                value=value,
            ), node)

        # Keep the return's position for diagnostics in the inlined body.
        return ast.copy_location(ast.Assign(
            targets=[ast.Name(id=self._result_var, ctx=ast.Store())],
            value=value,
        ), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rename function parameters."""
        node = self.generic_visit(node)
        for arg in node.args.args:
            if arg.arg in self._rename_map:
                arg.arg = self._rename_map[arg.arg]
        return node


class Inlining:
    """Inlines device-function calls (`_inline_func_call`, `_inline_func_body`).

    Owns `_active_funcs` (against recursion), `_inline_counter` (unique
    suffixes) and `_inline_stack` (the functions being inlined, which
    error messages name). Extends the array, vector and texture metadata
    of the other mixins for the callee's renamed parameters.
    """

    def _init_inlining(self):
        self._active_funcs = set()      # device functions being inlined, against recursion
        self._inline_counter = 0  # unique suffix for inlined variables
        self._inline_stack: list[str] = []   # names of the device functions being inlined

    def _inline_func_call(self, func_obj, call_node: ast.Call):
        """Inline a @tack.func call, tracking it for diagnostics."""
        # Arguments belong to the caller's source; visiting them first
        # records their positions before the callee's are in play.
        for arg in call_node.args:
            for sub in ast.walk(arg):
                if (isinstance(sub, ast.Name) and hasattr(sub, 'lineno')
                        and sub.id not in self._name_reads):
                    self._name_reads[sub.id] = (
                        sub.lineno, sub.col_offset + 1,
                        self._inline_stack[-1] if self._inline_stack else None)
        # Template methods are renamed; diagnostics use the source name.
        self._inline_stack.append(func_obj.func.__name__)
        try:
            return self._inline_func_body(func_obj, call_node)
        finally:
            self._inline_stack.pop()

    def _inline_func_body(self, func_obj, call_node: ast.Call):
        """Inline a @tack.func call at the call site.

        Produces a list of IR statements (parameter assignments + body),
        and returns the result variable name as an IRName.
        """
        from tack.lang.call_bindings import CallBindings
        func_name = func_obj.name
        funcdef = func_obj._funcdef
        callee_bindings = CallBindings(funcdef, func_obj.func,
                                      template_funcs=self._template_funcs)
        validate_source(funcdef, kind='Device function', call_bindings=callee_bindings)

        # Save and reset _pre_stmts so nested inlining doesn't interfere
        saved_pre_stmts = self._pre_stmts
        self._pre_stmts = []

        # Unique suffix to avoid name collisions across multiple inlines
        suffix = self._inline_counter
        self._inline_counter += 1

        # Map callee params to caller args
        callee_params = [a.arg for a in funcdef.args.args]
        if len(call_node.args) != len(callee_params):
            raise TypeError(
                f"@tack.func '{func_name}' expects {len(callee_params)} args, "
                f"got {len(call_node.args)}"
            )

        # Create a name renamer for the callee's body
        rename_map = {}
        for param_name in callee_params:
            rename_map[param_name] = self._fresh_name(f"__{func_name}_{param_name}_{suffix}__")

        # Detect return count: 0=void, 1=scalar, >1=tuple
        n_returns = self._detect_return_count(funcdef)
        if n_returns == 0:
            result_var = None
        elif n_returns > 1:
            result_var = [self._fresh_name(f"__{func_name}_ret_{c}_{suffix}__")
                          for c in range(n_returns)]
        else:
            result_var = self._fresh_name(f"__{func_name}_ret_{suffix}__")

        # Rename all local variables in the callee body to avoid collisions
        # We collect all assigned names and create renames for them
        assigned_names = self._collect_assigned_names(funcdef.body)
        for name in sorted(assigned_names):
            if name not in rename_map:
                rename_map[name] = self._fresh_name(f"__{func_name}_{name}_{suffix}__")

        # Override rename map for local array / shared memory params:
        # alias the caller's array name directly so the inlined body
        # accesses it without an assignment (arrays can't be assigned in C).
        # Must happen BEFORE the renamer runs.
        for param_name in callee_params:
            if isinstance(call_node.args[callee_params.index(param_name)], ast.Name):
                arg_name = call_node.args[callee_params.index(param_name)].id
                if arg_name in self._shared_vars:
                    rename_map[param_name] = arg_name

        # Rename the callee AST (deep copy first)
        renamed_body = _structure_returns(copy.deepcopy(funcdef.body), func_name)
        renamer = _NameRenamer(rename_map, result_var)
        # Flatten in case visit_Return returns a list (multi-return)
        flat_body = []
        for stmt in renamed_body:
            result = renamer.visit(stmt)
            if isinstance(result, list):
                flat_body.extend(result)
            else:
                flat_body.append(result)
        renamed_body = flat_body

        # Propagate vector-ness and texture metadata for renamed parameters
        # (must happen before parameter assignments so we can skip texture assigns)
        for param_name in callee_params:
            renamed = rename_map[param_name]
            if isinstance(call_node.args[callee_params.index(param_name)], ast.Name):
                arg_name = call_node.args[callee_params.index(param_name)].id
                if arg_name in self._vector_vars:
                    self._bind_vector(renamed, self._variable_value(arg_name))
                if arg_name in self._vector_arrays:
                    # A vector array passed in: `fld[i]` in the body must
                    # expand to its components, as it does in the caller.
                    self._vector_arrays[renamed] = self._vector_arrays[arg_name]
                    if arg_name in self._soa_arrays:
                        # ...from the caller's component parameters, with no
                        # binding of its own (skipped below).
                        self._soa_arrays[renamed] = self._soa_arrays[arg_name]
                    if arg_name in self._matrix_arrays:
                        self._matrix_arrays[renamed] = self._matrix_arrays[arg_name]
                if arg_name in self._texture_arrays:
                    self._texture_arrays[renamed] = self._texture_arrays[arg_name]
                    self._texture_origin[renamed] = self._texture_origin.get(arg_name, arg_name)

        # Emit parameter assignments
        stmts = []
        for param_name, arg_ast in zip(callee_params, call_node.args):
            renamed = rename_map[param_name]
            # Skip assignment for texture params and local array params —
            # these are aliased to the caller's variable directly.
            if renamed in self._texture_arrays:
                continue
            if renamed in self._shared_vars:
                continue
            if renamed in self._soa_arrays:
                continue
            arg_val = self.visit(arg_ast)
            # Capture any stmts from visiting this arg (e.g. nested inlines)
            stmts.extend(self._pre_stmts)
            self._pre_stmts = []
            if isinstance(arg_val, list):
                # Vector argument — assign each component
                ndim = len(arg_val)
                self._bind_vector(renamed, arg_val)
                for c in range(ndim):
                    stmts.append(ir.IRAssign(
                        target=self._component_name(renamed, c), value=arg_val[c]))
            else:
                stmts.append(ir.IRAssign(target=renamed, value=arg_val))

        # Visit the renamed body statements
        if func_obj in self._active_funcs:
            raise NotImplementedError(f"Recursive @tack.func '{func_name}' is not supported")
        caller_bindings = self._call_bindings
        callee_bindings.locals.update(rename_map.values())
        self._call_bindings = callee_bindings
        self._active_funcs.add(func_obj)
        try:
            body_ir = self._visit_body(renamed_body)
        finally:
            self._active_funcs.remove(func_obj)
            self._call_bindings = caller_bindings
        stmts.extend(body_ir)

        # Restore saved pre_stmts and append all inline stmts
        self._pre_stmts = saved_pre_stmts
        self._pre_stmts.extend(stmts)

        # Void function: no return value
        if result_var is None:
            return None

        # Multi-return: one value per element, a vector element as its
        # components (a nested list)
        if isinstance(result_var, list):
            return TupleValue(self._result_value(v) for v in result_var)

        return self._result_value(result_var)

    def _result_value(self, result_var: str):
        """An inlined call's result: a name, or a vector's component names."""
        if result_var in self._vector_vars:
            return self._variable_value(result_var)
        return ir.IRName(result_var)

    def _collect_assigned_names(self, stmts) -> set[str]:
        """Collect all variable names assigned in a list of AST statements.

        Every binding counts, including the names inside a tuple target
        (`t, u = f()`, `for i, j in ndindex(...)`): a callee local that is
        not collected is not renamed, and the inlined body then writes
        the caller's variable of the same name.
        """
        names = set()

        def targets(node):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    targets(elt)

        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        targets(t)
                elif isinstance(node, ast.AugAssign):
                    targets(node.target)
                elif isinstance(node, ast.For):
                    targets(node.target)
        return names

    def _detect_return_count(self, funcdef) -> int:
        """Detect if a function returns a tuple (multi-return).

        Returns the number of return values (0 for void, 1 for scalar, >1 for tuple).
        """
        for node in ast.walk(funcdef):
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Tuple):
                    return len(node.value.elts)
                return 1
        return 0
