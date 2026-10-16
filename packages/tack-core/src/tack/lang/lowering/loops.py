"""Loops, as lowering emits them.

A kernel has one parallel loop, a `for` over `range` or `tack.ndindex`
directly in its body (`IRParallelFor`, multi-dimensional with `dims` and
`extents`); every loop inside it is sequential (`IRSequentialFor`,
`IRWhile`). The module functions finish a lowered function's parallel
loop: they mark its `continue`s and write its bounds in terms of the
arguments, which dispatch evaluates.
"""

import ast
import copy

from tack.lang import ir
from tack.lang.ir_traversal import walk_ir


def _launch_bounds_from_arguments(params: list, body: list) -> list:
    """Write the parallel loop's bounds in terms of the arguments alone.

    Dispatch evaluates the bounds from the arguments, so a local they read,
    bound once before the loop (``n = x.shape[0]``, or the result of an
    inlined ``__len__`` that returns ``self.data.shape[0]``), is replaced
    by its value, and an alias of an array parameter by the parameter.
    Assignments nothing reads any longer are dropped: a dimension read
    outside the bound would specialize the kernel on that length. A bound
    that cannot be written so is left as it was, for dispatch to refuse.
    """
    loop = next((s for s in body if isinstance(s, ir.IRParallelFor)), None)
    if loop is None:
        return body
    names = {p.name for p in params}
    bindings: dict = {}
    for node in walk_ir(body):
        if isinstance(node, ir.IRAssign):
            bindings[node.target] = bindings.get(node.target, 0) + 1
        elif isinstance(node, (ir.IRSequentialFor, ir.IRParallelFor)):
            for var in [node.var, *(getattr(node, 'dims', None) or ())]:
                bindings[var] = 2
    outer = {}
    for stmt in body:
        if stmt is loop:
            break
        if isinstance(stmt, ir.IRAssign) and bindings[stmt.target] == 1:
            outer[stmt.target] = stmt.value
    followed = set()

    def array(name):
        while name not in names and isinstance(outer.get(name), ir.IRName):
            followed.add(name)
            name = outer[name].name
        return name if name in names else None

    def host(node):
        if isinstance(node, ir.IRConstant):
            return node
        if isinstance(node, ir.IRName):
            if node.name in names:
                return node
            value = outer.get(node.name)
            found = host(value) if value is not None else None
            if found is not None:
                followed.add(node.name)
            return found
        if isinstance(node, ir.IRDimSize):
            name = array(node.array_name)
            if name is None:
                return None
            found = copy.copy(node)
            found.array_name = name
            return found
        if isinstance(node, ir.IRBinOp):
            left, right = host(node.left), host(node.right)
            return None if left is None or right is None else ir.IRBinOp(node.op, left, right)
        if isinstance(node, ir.IRUnaryOp):
            operand = host(node.operand)
            return None if operand is None else ir.IRUnaryOp(node.op, operand)
        if isinstance(node, ir.IRCall) and node.func_name in ("min", "max"):
            args = [host(a) for a in node.args]
            return None if None in args else ir.IRCall(node.func_name, args)
        return None

    def rewrite(bound):
        if isinstance(bound, ir.IRConstant):
            return bound
        found = host(bound)
        return bound if found is None else found

    loop.end = rewrite(loop.end)
    if getattr(loop, 'dims', None):
        loop.extents = [rewrite(e) for e in loop.extents]
    # Drop the followed assignments nothing else reads, and then those only
    # they read.
    while True:
        read = set()
        for node in walk_ir(body):
            if isinstance(node, ir.IRName):
                read.add(node.name)
            name = getattr(node, 'array_name', None)
            if name is not None:
                read.add(name)
        for extent in getattr(loop, 'extents', None) or ():
            read.update(n.name if isinstance(n, ir.IRName) else n.array_name
                        for n in walk_ir(extent)
                        if isinstance(n, ir.IRName) or isinstance(n, ir.IRDimSize))
        dead = [s for s in body if isinstance(s, ir.IRAssign) and s.target in followed
                and s.target not in read]
        if not dead:
            return body
        body = [s for s in body if not any(s is d for d in dead)]


def _mark_outermost_continues(stmts: list):
    """Flag each `continue` that belongs to the parallel loop itself."""
    for stmt in stmts:
        if isinstance(stmt, ir.IRContinue):
            stmt.outermost = True
        elif isinstance(stmt, ir.IRIf):
            _mark_outermost_continues(stmt.then_body)
            _mark_outermost_continues(stmt.else_body or [])
        # Nested loops own their continues; do not descend into them.


class Loops:
    """Lowers `for` and `while` loops (`visit_For`, `visit_While`, ...).

    Owns `_loop_depth` (0 outside any loop), `_in_parallel_loop`,
    `_parallel_loop_seen` and `_kernel_body` (the kernel's own statements,
    so the parallel loop can be told from a nested one).
    """

    def _init_loops(self):
        self._loop_depth = 0
        # Statements outside the parallel loop may run any number of times
        # per launch (once per GPU thread, once per CPU chunk or probe), so
        # only effect-free ones are accepted there.
        self._kernel_body: list = []
        self._in_parallel_loop = False
        self._parallel_loop_seen = False

    def visit_For(self, node: ast.For) -> ir.IRNode:
        if not self._in_parallel_loop:
            return self._visit_parallel_for(node)
        return self._visit_for(node)

    def _visit_parallel_for(self, node: ast.For) -> ir.IRParallelFor:
        """Lower the kernel's one parallel loop, a statement of its own body."""
        if self._parallel_loop_seen or not any(node is s for s in self._kernel_body):
            raise self._source_error(
                node, "for loop", "is outside the parallel loop. A kernel has "
                "one parallel loop, a `for` statement directly in its body; "
                "sequential loops belong inside it")
        self._parallel_loop_seen = True
        return self._visit_for(node)

    def _visit_for_body(self, stmts: list) -> list:
        """Lower a for-loop body, which always lies inside the parallel loop.

        Its range arguments were lowered before, where the loop is entered.
        """
        saved = self._in_parallel_loop
        self._in_parallel_loop = True
        self._loop_depth += 1
        body = self._visit_body(stmts)
        self._loop_depth -= 1
        self._in_parallel_loop = saved
        return body

    def _visit_for(self, node: ast.For) -> ir.IRNode:
        target = node.target

        if node.orelse:
            raise NotImplementedError("'else' clause on for-loop not supported in kernels")

        # Handle tuple targets: for i, j in ndindex(w, h)
        if isinstance(target, ast.Tuple):
            return self._visit_ndindex_for(node)

        if not isinstance(target, ast.Name):
            raise NotImplementedError("Only simple loop variables supported")

        # Detect for i in range(...)
        if not self._is_range_call(node.iter):
            raise NotImplementedError("Only range() loops supported in kernels")

        start, end, step = self._parse_range_args(node.iter)
        if self._loop_depth:
            # Python's range evaluates its arguments once, before entering
            # the loop, even when the body rebinds a bound or step variable.
            start = self._capture_value(start, self._pre_stmts, freeze_name=True)
            end = self._capture_value(end, self._pre_stmts, freeze_name=True)
            if step is not None:
                step = self._capture_value(step, self._pre_stmts, freeze_name=True)

        body = self._visit_for_body(node.body)

        # Top-level for-range is parallel; nested for-range is sequential
        if self._loop_depth == 0:
            if step is not None:
                # Transform: for i in range(start, end, step)
                # Into: for __step_idx__ in range(0, (end - start + step - 1) // step):
                #            i = start + __step_idx__ * step
                idx_name = self._fresh_name(f"__step_idx_{self._inline_counter}__")
                self._inline_counter += 1
                total = ir.IRBinOp(
                    op="//",
                    left=ir.IRBinOp(op="+",
                        left=ir.IRBinOp(op="-", left=end, right=start),
                        right=ir.IRBinOp(op="-", left=step, right=ir.IRConstant(1))),
                    right=step,
                )
                decomp = ir.IRAssign(
                    target=target.id,
                    value=ir.IRBinOp(op="+", left=start,
                        right=ir.IRBinOp(op="*", left=ir.IRName(idx_name), right=step)),
                )
                return ir.IRParallelFor(var=idx_name, start=ir.IRConstant(0),
                                        end=total, body=[decomp, *body])
            if not (isinstance(start, ir.IRConstant) and start.value == 0):
                # Every backend launches the grid over [0, n) and never sees
                # the start, so it has to live in the body.
                # Transform: for i in range(start, end)
                # Into: for __start_idx__ in range(0, end - start):
                #            i = start + __start_idx__
                idx_name = self._fresh_name(f"__start_idx_{self._inline_counter}__")
                self._inline_counter += 1
                decomp = ir.IRAssign(
                    target=target.id,
                    value=ir.IRBinOp(op="+", left=start, right=ir.IRName(idx_name)),
                )
                return ir.IRParallelFor(
                    var=idx_name, start=ir.IRConstant(0),
                    end=ir.IRBinOp(op="-", left=end, right=start),
                    body=[decomp, *body])
            return ir.IRParallelFor(var=target.id, start=start, end=end, body=body)
        return ir.IRSequentialFor(var=target.id, start=start, end=end,
                                  body=body, step=step)

    def _visit_ndindex_for(self, node: ast.For):
        """Handle: for i, j in ndindex(w, h) or for i, j in tack.ndindex(w, h).

        Each argument is a size, counting from zero, or a ``(start, end)``
        pair: ``ndindex((1, n - 1), (1, m - 1))`` visits the interior. An
        empty or reversed pair makes the whole loop empty.
        """
        target = node.target
        if not isinstance(target, ast.Tuple):
            raise NotImplementedError("ndindex requires tuple target")

        names = []
        for elt in target.elts:
            if not isinstance(elt, ast.Name):
                raise NotImplementedError("ndindex targets must be simple names")
            names.append(elt.id)

        # Parse ndindex call
        call = node.iter
        if not isinstance(call, ast.Call):
            raise NotImplementedError("Expected ndindex() call")

        func_name = self._resolve_call_name(call)
        if func_name != "ndindex":
            raise NotImplementedError(f"Expected ndindex(), got {func_name}()")

        if len(call.args) != len(names):
            raise NotImplementedError(
                f"ndindex() dimension count ({len(call.args)}) must match "
                f"target count ({len(names)})"
            )

        starts = []
        dim_exprs = []
        for arg in call.args:
            if isinstance(arg, ast.Tuple):
                if len(arg.elts) != 2:
                    raise NotImplementedError(
                        "an ndindex() range is a (start, end) pair")
                start, end = self._visit_ordered(arg.elts)
                starts.append(start)
                # Clamp at zero: a reversed pair must not turn into a count
                # that another reversed pair makes positive again.
                dim_exprs.append(ir.IRCall(func_name="max", args=[
                    ir.IRBinOp(op="-", left=end, right=start), ir.IRConstant(0)]))
            else:
                starts.append(None)
                dim_exprs.append(self.visit(arg))

        # Compute total = dim0 * dim1 * ...
        total = dim_exprs[0]
        for d in dim_exprs[1:]:
            total = ir.IRBinOp(op="*", left=total, right=d)

        if self._loop_depth == 0 and len(names) in (2, 3):
            # The backend binds one index per dimension from a launch of
            # the same shape (see parallel_dims); nothing divides.
            idx_name = self._fresh_name(f"__nd_idx_{self._inline_counter}__")
            dims = [self._fresh_name(f"__nd_{self._inline_counter}_{k}__")
                    for k in range(len(names))]
            self._inline_counter += 1
            offsets = [ir.IRAssign(target=name, value=self._offset_index(start, ir.IRName(dim)))
                       for name, start, dim in zip(names, starts, dims)]
            body_stmts = self._visit_for_body(node.body)
            return ir.IRParallelFor(var=idx_name, start=ir.IRConstant(0), end=total,
                                    body=offsets + body_stmts, dims=dims,
                                    extents=list(dim_exprs))

        # Build body: decompose linear index into multi-dim indices
        # for __nd_idx__ in range(total):
        #     i = __nd_idx__ // dim1
        #     j = __nd_idx__ % dim1
        # For 3D: i = idx // (d1*d2), j = (idx // d2) % d1, k = idx % d2
        idx_name = self._fresh_name(f"__nd_idx_{self._inline_counter}__")
        self._inline_counter += 1

        decomp_stmts = []
        remaining = ir.IRName(idx_name)

        for k in range(len(names) - 1):
            # Product of remaining dimensions
            tail_product = dim_exprs[k + 1]
            for d in dim_exprs[k + 2:]:
                tail_product = ir.IRBinOp(op="*", left=tail_product, right=d)

            decomp_stmts.append(ir.IRAssign(target=names[k], value=self._offset_index(
                starts[k], ir.IRBinOp(op="//", left=remaining, right=tail_product))))
            remaining = ir.IRBinOp(op="%", left=remaining, right=tail_product)

        # Last dimension gets the remainder
        decomp_stmts.append(ir.IRAssign(
            target=names[-1], value=self._offset_index(starts[-1], remaining)))

        body_stmts = self._visit_for_body(node.body)

        full_body = decomp_stmts + body_stmts

        if self._loop_depth == 0:
            return ir.IRParallelFor(
                var=idx_name, start=ir.IRConstant(0), end=total, body=full_body)
        return ir.IRSequentialFor(
            var=idx_name, start=ir.IRConstant(0), end=total, body=full_body)

    @staticmethod
    def _offset_index(start, index):
        """An ndindex index counted from its range's start, if it has one."""
        return index if start is None else ir.IRBinOp(op="+", left=start, right=index)

    def visit_While(self, node: ast.While) -> ir.IRWhile:
        if node.orelse:
            raise NotImplementedError("'else' clause on while-loop not supported in kernels")
        condition, condition_stmts = self._visit_expression(node.test)
        condition = self._scalar_condition(node, "while", condition)
        self._loop_depth += 1
        body = self._visit_body(node.body)
        self._loop_depth -= 1
        if condition_stmts:
            # The condition's inlined calls execute on every test, including
            # after continue, rather than once before entering the loop.
            body = [*condition_stmts,
                    ir.IRIf(ir.IRUnaryOp('not', condition), [ir.IRBreak()], []),
                    *body]
            condition = ir.IRConstant(1)
        return ir.IRWhile(condition=condition, body=body)

    def visit_Break(self, node: ast.Break) -> ir.IRBreak:
        return ir.IRBreak()

    def visit_Continue(self, node: ast.Continue) -> ir.IRContinue:
        return ir.IRContinue()

    def _is_range_call(self, node: ast.expr) -> bool:
        """Check if an AST node is a call to range()."""
        return (isinstance(node, ast.Call) and
                self._resolve_call_name(node) == 'range')

    def _parse_range_args(self, call_node: ast.Call):
        """Parse range(end), range(start, end), or range(start, end, step)."""
        args = self._visit_ordered(call_node.args)
        if len(args) == 1:
            return ir.IRConstant(0), args[0], None
        if len(args) == 2:
            return args[0], args[1], None
        if len(args) == 3:
            return args[0], args[1], args[2]
        raise NotImplementedError("range() takes 1-3 arguments")
