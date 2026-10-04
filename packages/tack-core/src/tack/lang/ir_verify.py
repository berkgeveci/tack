"""Compiler invariants at the boundaries of Tack's IR pipeline.

This verifies the tree and the annotations needed by the next pass. It is
not a proof of memory safety, definite assignment, or barrier uniformity.
Verification runs when building a template/variant, never on a cache hit.
"""

from tack.lang import ir
from tack.lang.ir_traversal import CHILD_FIELDS, LIST_ROLES, child_fields
from tack.lang.types import ScalarType, i32

STAGES = ('lowered', 'resolved', 'inferred', 'localized', 'optimized', 'packed', 'typed')
EXPRS = {
    ir.IRIfExp, ir.IRBinOp, ir.IRUnaryOp, ir.IRCompare, ir.IRBoolOp,
    ir.IRFieldLoad, ir.IRConstant, ir.IRName, ir.IRAttribute, ir.IRCall,
    ir.IRCast, ir.IRBlockReduce, ir.IRThreadId, ir.IRDimSize, ir.IRTextureSample,
}
STMTS = {
    ir.IRParallelFor, ir.IRSequentialFor, ir.IRWhile, ir.IRBreak, ir.IRContinue,
    ir.IRIf, ir.IRFieldStore, ir.IRAtomicOp, ir.IRAssign, ir.IRReturn,
    ir.IRSharedAlloc, ir.IRLocalAlloc, ir.IRBarrier, ir.IRPrint, ir.IRCall,
}
BINOPS = {'+', '-', '*', '/', '//', '%', '**', '&', '|', '^', '<<', '>>'}
COMPARES = {'==', '!=', '<', '<=', '>', '>='}
OPERATORS = {
    ir.IRBinOp: BINOPS, ir.IRCompare: COMPARES,
    ir.IRUnaryOp: {'-', '+', '~', 'not'}, ir.IRBoolOp: {'and', 'or'},
    ir.IRAtomicOp: {'add', 'min', 'max'}, ir.IRBlockReduce: {'sum', 'min', 'max'},
}
ATTRIBUTES = {
    ir.IRFunction: ('name',), ir.IRParam: ('name', 'type_annotation'),
    ir.IRParallelFor: ('var',), ir.IRSequentialFor: ('var',),
    ir.IRBinOp: ('op',), ir.IRUnaryOp: ('op',), ir.IRCompare: ('op',),
    ir.IRBoolOp: ('op',), ir.IRAtomicOp: ('op',), ir.IRBlockReduce: ('op',),
    ir.IRConstant: ('value',), ir.IRName: ('name',), ir.IRAttribute: ('attr',),
    ir.IRCall: ('func_name',), ir.IRAssign: ('target',), ir.IRCast: ('dtype',),
    ir.IRSharedAlloc: ('name', 'dtype', 'field_name'),
    ir.IRLocalAlloc: ('name', 'dtype', 'field_name'),
    ir.IRDimSize: ('field_name', 'dim'), ir.IRTextureSample: ('field_name', 'shape'),
    ir.IRPrint: ('format_parts',),
}
ROLE_KINDS = {'function': {ir.IRFunction}, 'param': {ir.IRParam},
              'stmt': STMTS, 'expr': EXPRS}
SINGULAR_ROLES = {'params': 'param', 'stmts': 'stmt', 'exprs': 'expr'}
REQUIRED_ATTRIBUTES = {
    kind: ATTRIBUTES.get(kind, ()) + tuple(attr for attr, _ in fields)
    for kind, fields in CHILD_FIELDS.items()
}


class IRVerificationError(RuntimeError):
    """An invalid IR tree or a compiler pass that broke its postconditions."""


def verify_ir(function: ir.IRFunction, stage: str):
    """Check one kernel function without changing its tree or metadata.

    Errors identify the kernel, stage, node kind, and structural path.
    The parallel loop's end is host-evaluated, so unresolved dimension
    queries there are legal even at the final codegen boundary.
    """
    if stage not in STAGES:
        raise ValueError(f'Unknown IR verification stage: {stage}')
    resolved = stage != 'lowered'
    inferred = STAGES.index(stage) >= STAGES.index('inferred')
    typed = stage == 'typed'
    kernel_name = getattr(function, 'name', '<unnamed>')
    active = set()
    nodes = []  # (node, path, host_bound), collected only after shape checks

    def fail(node, path, message):
        raise IRVerificationError(
            f"Kernel '{kernel_name}': invalid IR after {stage} at {path} "
            f'({type(node).__name__}): {message}')

    def require(node, path, condition, message):
        if not condition:
            fail(node, path, message)

    def name(node, path, attr):
        value = getattr(node, attr, None)
        require(node, path, isinstance(value, str) and bool(value),
                f'{attr} must be a nonempty name')

    def visit(node, path, role, loops=(), host_bound=False):
        kind = type(node)
        allowed = ROLE_KINDS[role]
        require(node, path, kind in allowed, f'expected {role} node')
        require(node, path, id(node) not in active, 'cycle in IR tree')
        active.add(id(node))
        nodes.append((node, path, host_bound))
        fields = child_fields(node)
        for attr in REQUIRED_ATTRIBUTES[kind]:
            if not hasattr(node, attr):
                fail(node, path, f'missing attribute {attr}')

        if kind in (ir.IRFunction, ir.IRParam, ir.IRName,
                    ir.IRSharedAlloc, ir.IRLocalAlloc):
            name(node, path, 'name')
        if kind is ir.IRAssign:
            name(node, path, 'target')
        if kind in (ir.IRParallelFor, ir.IRSequentialFor):
            name(node, path, 'var')
        if kind is ir.IRParallelFor:
            require(node, path, path.startswith('function.body[') and path.count('.') == 1,
                    'parallel loop must be at function top level')
            require(node, path, isinstance(node.start, ir.IRConstant)
                    and type(node.start.value) is int and node.start.value == 0,
                    'parallel range must be normalized to start at zero')
        if kind is ir.IRSequentialFor:
            require(node, path, ir.IRParallelFor in loops,
                    'sequential loop must be inside the parallel loop')
        if kind is ir.IRBreak:
            require(node, path, bool(loops) and loops[-1] is not ir.IRParallelFor,
                    'break must target a sequential loop')
        if kind is ir.IRContinue:
            require(node, path, bool(loops), 'continue must target a loop')
            require(node, path, type(node.outermost) is bool and
                    node.outermost == (loops[-1] is ir.IRParallelFor),
                    'outermost flag disagrees with the target loop')
        if kind is ir.IRReturn:
            fail(node, path, 'kernel return must be rejected or inlined before lowering')

        if kind in OPERATORS:
            require(node, path, isinstance(node.op, str) and node.op in OPERATORS[kind],
                    'unsupported operator')
        if kind is ir.IRConstant:
            require(node, path, isinstance(node.value, (int, float)),
                    'constant must be numeric')
        if kind is ir.IRCall:
            name(node, path, 'func_name')
        if kind is ir.IRAttribute:
            name(node, path, 'attr')
            require(node, path, not resolved or host_bound,
                    'unresolved attribute in generated code')
        if kind is ir.IRDimSize:
            name(node, path, 'field_name')
            require(node, path, type(node.dim) is int and node.dim >= 0,
                    'dimension must be a nonnegative integer')
            require(node, path, not resolved or host_bound,
                    'unresolved dimension in generated code')
        if kind in (ir.IRSharedAlloc, ir.IRLocalAlloc):
            require(node, path, isinstance(node.dtype, ScalarType) or
                    (not resolved and node.dtype is None and
                     isinstance(node.field_name, str)), 'allocation dtype is unresolved')
        if kind is ir.IRCast:
            require(node, path, isinstance(node.dtype, ScalarType),
                    'cast target must be a ScalarType')
        if kind is ir.IRTextureSample:
            name(node, path, 'field_name')
            if resolved:
                require(node, path, isinstance(node.shape, tuple) and len(node.shape) == 3
                        and all(type(n) is int and n > 0 for n in node.shape),
                        'texture extent must contain three positive integers')
        if kind is ir.IRParam and inferred:
            require(node, path, isinstance(node.type_annotation, ScalarType),
                    'parameter type must be a ScalarType')
            require(node, path, type(getattr(node, '_is_field', None)) is bool,
                    'parameter field/scalar category is missing')

        try:
            for attr, child_role in fields:
                value = getattr(node, attr)
                child_path = f'{path}.{attr}'
                child_loops = loops
                if attr == 'body' and kind in (ir.IRParallelFor, ir.IRSequentialFor, ir.IRWhile):
                    child_loops += (kind,)
                child_host = host_bound or (kind is ir.IRParallelFor and attr == 'end')
                if child_role in LIST_ROLES:
                    require(node, path, isinstance(value, list), f'{attr} must be a list')
                    if kind is ir.IRBoolOp:
                        require(node, path, len(value) >= 2, 'Boolean operation needs two values')
                    if kind is ir.IRTextureSample:
                        require(node, path, len(value) == 3, 'texture sample needs three coordinates')
                    singular = SINGULAR_ROLES[child_role]
                    for index, child in enumerate(value):
                        visit(child, f'{child_path}[{index}]', singular, child_loops, child_host)
                elif value is not None or child_role != 'optional_expr':
                    visit(value, child_path, 'expr', child_loops, child_host)
        finally:
            active.remove(id(node))

    visit(function, 'function', 'function')
    params = {p.name: p for p in function.params}
    require(function, 'function.params', len(params) == len(function.params),
            'duplicate parameter names')
    parallel = [s for s in function.body if isinstance(s, ir.IRParallelFor)]
    require(function, 'function.body', len(parallel) == 1,
            'kernel must contain exactly one top-level parallel loop')

    # Function-wide binding existence, not definite assignment. Loop-carried
    # and branch-defined variables require a separate control-flow analysis.
    bound = set(params)
    buffers = {p.name for p in function.params if getattr(p, '_is_field', False)}
    for node, _, _ in nodes:
        if isinstance(node, ir.IRAssign):
            bound.add(node.target)
        elif isinstance(node, (ir.IRParallelFor, ir.IRSequentialFor)):
            bound.add(node.var)
        elif isinstance(node, (ir.IRSharedAlloc, ir.IRLocalAlloc)):
            bound.add(node.name)
            buffers.add(node.name)
    # Inlined field arguments can remain as pointer copies in the IR.
    changed = True
    while changed:
        changed = False
        for node, _, _ in nodes:
            if (isinstance(node, ir.IRAssign) and isinstance(node.value, ir.IRName)
                    and node.value.name in buffers and node.target not in buffers):
                buffers.add(node.target)
                changed = True

    for node, path, host_bound in nodes:
        if isinstance(node, ir.IRName):
            require(node, path, node.name in bound, f'unbound name {node.name!r}')
        if isinstance(node, (ir.IRDimSize, ir.IRTextureSample)):
            require(node, path, node.field_name in bound,
                    f'unbound field {node.field_name!r}')
        if inferred and isinstance(node, (ir.IRFieldLoad, ir.IRFieldStore, ir.IRAtomicOp)):
            require(node, path, isinstance(node.field, ir.IRName) and node.field.name in buffers,
                    'field access must name a field parameter or an array allocation')
        if stage == 'packed' and isinstance(node, ir.IRParam):
            require(node, path, node._is_field, 'scalar parameter survived GPU packing')
        if typed and not host_bound:
            pointer = isinstance(node, ir.IRName) and node.name in buffers
            if (type(node) in EXPRS or isinstance(node, ir.IRAtomicOp)) and not pointer:
                require(node, path, isinstance(getattr(node, 'dtype', None), ScalarType),
                        'scalar expression dtype is missing')
            if isinstance(node, (ir.IRCompare, ir.IRBoolOp)) or (
                    isinstance(node, ir.IRUnaryOp) and node.op == 'not'):
                require(node, path, node.dtype is i32, 'logical result must have i32 dtype')
            if isinstance(node, ir.IRAssign):
                pointer_copy = isinstance(node.value, ir.IRName) and node.value.name in buffers
                require(node, path, hasattr(node, '_resolved_type') and
                        (pointer_copy or isinstance(node._resolved_type, ScalarType)),
                        'assignment storage type is missing')
