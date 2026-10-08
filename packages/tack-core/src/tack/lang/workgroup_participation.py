"""Conservative uniformity checks for workgroup barriers and reductions.

Uniform means equal across a workgroup, not necessarily across the grid.
This structured analysis follows scalar assignments, joins branches and
computes loop-carried uniformity to a fixed point. Ordinary memory loads
and lane indices are varying; runtime scalar packs are immutable uniforms.
It proves a restricted domain, not general race freedom or termination.
"""

from tack.lang import ir
from tack.lang.ir_traversal import child_fields, walk_ir

WORKGROUP_SIZE = 256
_COLLECTIVES = (ir.IRBarrier, ir.IRBlockReduce)


def requires_full_workgroups(ir_func):
    return any(isinstance(node, _COLLECTIVES) for node in walk_ir(ir_func.body))


def check_workgroup_launch(kernel_name, loop_end, *, backend_label,
                          workgroup_size=WORKGROUP_SIZE):
    """Check an already identified collective kernel's actual launch."""
    if workgroup_size != WORKGROUP_SIZE:
        raise ValueError(
            f"Kernel '{kernel_name}': {backend_label} workgroup collectives "
            f"require {WORKGROUP_SIZE} lanes; this pipeline/device selects "
            f"{workgroup_size}."
        )
    if loop_end > 0 and loop_end % WORKGROUP_SIZE:
        raise ValueError(
            f"Kernel '{kernel_name}': {backend_label} workgroup collectives "
            f"require complete {WORKGROUP_SIZE}-lane groups; iteration count "
            f"{loop_end} would create a partial workgroup."
        )


def _join(*environments):
    names = set().union(*environments)
    return {name: all(env.get(name, False) for env in environments) for name in names}


class _Uniformity:
    def __init__(self, function):
        self.function = function
        self.in_parallel = False
        self.scalar_packs = {p.name for p in function.params
                             if getattr(p, '_is_scalar_pack', False)}

    def fail(self, node, path, reason):
        primitive = (f'block_{node.op}' if isinstance(node, ir.IRBlockReduce)
                     else 'barrier')
        raise ValueError(
            f"Kernel '{self.function.name}': {primitive} at {path} requires "
            f"uniform workgroup participation: {reason}."
        )

    def reject_collectives(self, node, path, reason):
        for child in walk_ir(node):
            if isinstance(child, _COLLECTIVES):
                self.fail(child, path, reason)

    def expr(self, node, env, uniform, path, validate):
        if node is None:
            return True
        if isinstance(node, ir.IRBlockReduce):
            if validate and (not uniform or not self.in_parallel):
                self.fail(node, path, 'control flow is varying or outside the parallel body')
            self.expr(node.value, env, uniform, path + '.value', validate)
            return True  # Every participating lane receives the same result.
        if isinstance(node, ir.IRIfExp) and validate:
            self.reject_collectives([node.then_value, node.else_value], path,
                                    'collectives in conditional expressions are unsupported')
        if isinstance(node, ir.IRBoolOp) and validate:
            self.reject_collectives(node.values[1:], path,
                                    'collectives in short-circuit operands are unsupported')

        children = []
        for name, role in child_fields(node):
            value = getattr(node, name)
            if role == 'optional_expr' and value is None:
                continue
            items = value if isinstance(value, list) else [value]
            for index, child in enumerate(items):
                children.append(self.expr(child, env, uniform,
                                          f'{path}.{name}[{index}]', validate))
        if isinstance(node, (ir.IRConstant, ir.IRDimSize)):
            return True
        if isinstance(node, ir.IRName):
            return env.get(node.name, False)
        if isinstance(node, ir.IRAttribute):
            return node.attr in ('shape', '__len__')
        if isinstance(node, ir.IRFieldLoad):
            descriptor = isinstance(node.field, ir.IRAttribute) and node.field.attr == 'shape'
            packed = isinstance(node.field, ir.IRName) and node.field.name in self.scalar_packs
            return (descriptor or packed) and children[-1]
        if isinstance(node, (ir.IRThreadId, ir.IRTextureSample)):
            return False
        if isinstance(node, (ir.IRBinOp, ir.IRUnaryOp, ir.IRCompare,
                             ir.IRBoolOp, ir.IRIfExp, ir.IRCast, ir.IRCall)):
            return all(children)
        return False

    def body(self, statements, env, uniform, path, validate):
        env = dict(env)
        escapes = set()
        for index, node in enumerate(statements):
            here = f'{path}[{index}]'
            if isinstance(node, ir.IRAssign):
                value = self.expr(node.value, env, uniform, here + '.value', validate)
                env[node.target] = uniform and value
            elif isinstance(node, ir.IRIf):
                condition = self.expr(node.condition, env, uniform, here + '.condition', validate)
                left, left_exits = self.body(node.then_body, env, uniform and condition,
                                             here + '.then_body', validate)
                right, right_exits = self.body(node.else_body or [], env, uniform and condition,
                                               here + '.else_body', validate)
                env = _join(left, right)
                exits = left_exits | right_exits
                escapes |= exits
                uniform = uniform and not exits
            elif isinstance(node, ir.IRParallelFor):
                env[node.var] = False
                for dim in node.dims or ():
                    env[dim] = False
                previous = self.in_parallel
                self.in_parallel = True
                try:
                    env, exits = self.body(node.body, env, uniform, here + '.body', validate)
                finally:
                    self.in_parallel = previous
                escapes |= exits
                uniform = uniform and not exits
            elif isinstance(node, (ir.IRSequentialFor, ir.IRWhile)):
                env, exits = self.loop(node, env, uniform, here, validate)
                escapes |= exits
                uniform = uniform and not exits
            elif isinstance(node, _COLLECTIVES):
                if validate and (not uniform or not self.in_parallel):
                    self.fail(node, here, 'control flow is varying or outside the parallel body')
            elif isinstance(node, (ir.IRBreak, ir.IRContinue, ir.IRReturn)):
                if isinstance(node, ir.IRReturn) and node.value is not None:
                    self.expr(node.value, env, uniform, here + '.value', validate)
                if not uniform:
                    kernel_exit = (isinstance(node, ir.IRReturn) or
                                   (isinstance(node, ir.IRContinue) and node.outermost))
                    escapes.add('kernel' if kernel_exit else 'loop')
            elif isinstance(node, (ir.IRLocalAlloc, ir.IRSharedAlloc)):
                self.expr(node.size, env, uniform, here + '.size', validate)
                env[node.name] = False
            else:
                # Stores, atomics and prints: visit every expression, even
                # indices and arguments, to discover implicit reductions.
                for name, role in child_fields(node):
                    value = getattr(node, name)
                    if role == 'optional_expr' and value is None:
                        continue
                    items = value if isinstance(value, list) else [value]
                    for child in items:
                        self.expr(child, env, uniform, here + '.' + name, validate)
        return env, escapes

    def loop(self, node, incoming, uniform, path, validate):
        def condition(env, check):
            if isinstance(node, ir.IRWhile):
                if check:
                    self.reject_collectives(node.condition, path + '.condition',
                                            'collectives in while conditions are unsupported')
                return self.expr(node.condition, env, uniform, path + '.condition', check)
            values = [self.expr(getattr(node, name), env, uniform,
                                path + '.' + name, check) for name in ('start', 'end', 'step')]
            return all(values)

        carried = dict(incoming)
        while True:
            entry = dict(carried)
            coherent = uniform and condition(carried, False)
            if isinstance(node, ir.IRSequentialFor):
                entry[node.var] = coherent
            outgoing, exits = self.body(node.body, entry, coherent, path + '.body', False)
            joined = _join(carried, incoming, outgoing)
            if joined == carried:
                break
            carried = joined
        coherent = uniform and condition(carried, validate) and not exits
        if isinstance(node, ir.IRSequentialFor):
            carried[node.var] = coherent
        outgoing, exits = self.body(node.body, carried, coherent, path + '.body', validate)
        # Lane-dependent exits affect subsequent iterations of this loop.
        # Only returns/parallel-continues escape beyond a nested loop.
        return _join(incoming, outgoing), exits & {'kernel'}


def check_workgroup_participation(ir_func):
    """Validate structured collective control flow; return launch requirements.

    This never trusts cached metadata on mutable IR. Call once while building
    a variant, and cache the resulting boolean on the variant for launch checks.
    """
    if not requires_full_workgroups(ir_func):
        return False
    checker = _Uniformity(ir_func)
    env = {p.name: not getattr(p, '_is_field', False) for p in ir_func.params}
    checker.body(ir_func.body, env, True, 'body', True)
    return True
