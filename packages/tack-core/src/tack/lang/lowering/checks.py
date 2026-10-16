"""Checks lowering runs on what it has emitted.

Effects outside the parallel loop, a vector or tuple left where a scalar
belongs, names read but never bound, and loop variables used outside
their loops are refused here with the source position of the cause,
before the IR verifier would report them as malformed IR.
"""

from tack.lang import ir
from tack.lang.ir_traversal import LIST_ROLES, child_fields, walk_ir
from tack.lang.lowering.values import _describe_value


def _effect_name(node) -> str | None:
    """Name an observable effect, or None for an effect-free node."""
    if isinstance(node, ir.IRArrayStore):
        return "store"
    if isinstance(node, (ir.IRAtomicOp, ir.IRBlockReduce)):
        prefix = "atomic" if isinstance(node, ir.IRAtomicOp) else "block"
        return f"{prefix}_{node.op}()"
    if isinstance(node, ir.IRBarrier):
        return "barrier()"
    if isinstance(node, ir.IRPrint):
        return "print()"
    return None


class LoweringChecks:
    """The checks on lowered statements and on the finished function.

    Owns `_checked_nodes` (statements `_check_single_values` has seen) and
    `_name_reads` (where each name is first read, for the error that names
    it; `visit_Name` and inlining record into it).
    """

    def _init_checks(self):
        # Lowered nodes already checked by _check_single_values, by id.
        self._checked_nodes = set()
        # Where each name is first read: name -> (line, column, device
        # function it was inlined from or None). Only for diagnostics.
        self._name_reads: dict[str, tuple] = {}

    def _check_outside_parallel_loop(self, stmt, lowered: list):
        """Reject an observable effect lowered outside the parallel loop.

        Runs on each statement's lowered IR, so effects that arrive through
        inlined device functions, template methods, vector stores or tuple
        unpacking are all seen. Nested and inlined bodies are checked first,
        so the position is that of the innermost source statement.
        """
        for node in walk_ir([s for s in lowered if not isinstance(s, ir.IRParallelFor)]):
            effect = _effect_name(node)
            if effect is not None:
                raise self._source_error(
                    stmt, effect, "is outside the parallel loop. Statements "
                    "there may run any number of times per launch, so they "
                    "may only assign local variables and declare arrays; "
                    f"move the {effect} into the loop body")

    def _check_single_values(self, stmt, lowered: list):
        """Reject a vector or a tuple left where the IR needs one value.

        Vectors exist only during lowering, as lists of component
        expressions; every operation that accepts one maps over it. One
        that does not would otherwise put the list into the IR, where the
        verifier reports it as "expected expr node" with a tree path. Here
        the statement's source position is still known. Each lowered node
        is examined once: a nested body was checked when it was lowered.
        """
        pending = list(lowered)
        while pending:
            node = pending.pop()
            if id(node) in self._checked_nodes:
                continue
            self._checked_nodes.add(id(node))
            if isinstance(node, list):
                raise self._source_error(
                    stmt, "statement", f"uses {_describe_value(node)} where a single "
                    f"value is required")
            for name, role in child_fields(node):
                value = getattr(node, name)
                if role in LIST_ROLES:
                    pending.extend(value)
                elif value is not None:
                    pending.append(value)

    def _check_names_bound(self, function: ir.IRFunction):
        """Reject a read of a name the kernel never binds.

        The usual cause is a module-level Python value: a kernel is compiled
        from its own source and captures nothing from the enclosing scope.
        The IR verifier would refuse the same thing a moment later, but only
        as a tree path into the lowered IR; here the source position is still
        known. Existence only -- not definite assignment on every path.
        """
        bound = {p.name for p in function.params}
        used = []
        for n in walk_ir(function.body):
            if isinstance(n, ir.IRAssign):
                bound.add(n.target)
            elif isinstance(n, (ir.IRParallelFor, ir.IRSequentialFor)):
                bound.add(n.var)
                bound.update(getattr(n, 'dims', None) or ())
            elif isinstance(n, (ir.IRSharedAlloc, ir.IRLocalAlloc)):
                bound.add(n.name)
            elif isinstance(n, ir.IRName):
                used.append(n.name)
            elif isinstance(n, (ir.IRDimSize, ir.IRTextureSample)):
                used.append(n.array_name)
        for name in used:
            if name in bound:
                continue
            where = f"Kernel '{function.name}'"
            read = self._name_reads.get(name)
            at = ""
            if read is not None:
                line, column, inlined_from = read
                at = f" at line {line}, column {column}"
                if inlined_from is not None:
                    where = (f"Device function '{inlined_from}' "
                             f"(inlined into kernel '{function.name}')")
            raise NameError(
                f"{where}: name '{name}'{at} is not a parameter and is never "
                f"assigned. Kernels do not capture Python variables from the "
                f"enclosing scope: define it as {name} = tack.constant(...), "
                f"pass the value as an argument, or make it a class-level "
                f"constant of a @tack.device_class template.")

    def _check_loop_variables_stay_in_their_loops(self, function: ir.IRFunction):
        """Reject a read of a loop variable after its loop.

        A `for` loop's variable is a binding of its own that ends with the
        loop, on every backend: afterwards the name means whatever it meant
        before the loop, or nothing. Python would give the last value
        iterated. Rather than compute either silently -- CPU raised a
        NameError at codegen while the GPU backends failed to compile --
        refuse the read here, with its source position. A name whose most
        recent binding in program order is a loop header may not be read
        until it is assigned again. Branches and loops that may not run are
        merged conservatively: the name stays stale if it is stale on any
        path, so `assign it before the loop` is always a valid fix.
        """
        def read(expr, stale):
            for n in walk_ir(expr):
                if isinstance(n, ir.IRName) and n.name in stale:
                    raise self._loop_variable_error(function, n)

        def block(stmts, stale):
            stale = set(stale)
            for s in stmts:
                if isinstance(s, ir.IRAssign):
                    read(s.value, stale)
                    stale.discard(s.target)
                elif isinstance(s, (ir.IRSequentialFor, ir.IRParallelFor)):
                    for bound in (s.start, s.end, getattr(s, "step", None)):
                        if bound is not None:
                            read(bound, stale)
                    inner = block(s.body, stale - {s.var})
                    stale = (stale | inner) | {s.var}
                elif isinstance(s, ir.IRWhile):
                    read(s.condition, stale)
                    stale = stale | block(s.body, stale)
                elif isinstance(s, ir.IRIf):
                    read(s.condition, stale)
                    stale = block(s.then_body, stale) | block(s.else_body or [], stale)
                else:
                    read(s, stale)
            return stale

        block(function.body, set())

    def _loop_variable_error(self, function, name_node):
        where = f"Kernel '{function.name}'"
        at = ""
        read = getattr(name_node, "_source", None) or self._name_reads.get(name_node.name)
        if read is not None:
            line, column, inlined_from = read
            at = f" at line {line}, column {column}"
            if inlined_from is not None:
                where = (f"Device function '{inlined_from}' "
                         f"(inlined into kernel '{function.name}')")
        spelled = getattr(name_node, "_source_id", name_node.name)
        return NameError(
            f"{where}: name '{spelled}'{at} is read after the `for` loop "
            f"that bound it. A loop variable's binding ends with its loop; "
            f"to use the value afterwards, copy it to another name inside the "
            f"loop, or assign the name again before reading it.")
