"""Debug mode's bounds checks: the parts shared by lowering, the passes and dispatch.

With ``tack.init(debug=True)`` (or ``TACK_DEBUG=1``) every index of a field
or array access is checked against its extent, one dimension at a time.
Lowering wraps each access's flat index in an ``IRBoundsCheck``, which is -1
when an index is out of bounds and records the first violation in a hidden
errors field; ``guard_accesses`` then skips every access whose index is
negative, so a bad load reads 0 and a bad store or atomic does nothing.
After the launch, dispatch reads the errors field and raises ``IndexError``
naming the access (``runtime/debug_bounds.py``).

Two hidden field parameters follow the kernel's own: ``ERRORS_PARAM``
(i32, the record below) and ``EXTENTS_PARAM`` (i64, three extents per
parameter: see ``IRFieldExtent``).
"""

from typing import NamedTuple

from tack.lang import ir
from tack.lang.ir_names import fresh_name
from tack.lang.ir_traversal import CHILD_FIELDS, walk_ir
from tack.lang.types import INTEGER_TYPES, i32

ERRORS_PARAM = "__tack_errors__"
EXTENTS_PARAM = "__tack_extents__"
HIDDEN_PARAMS = (ERRORS_PARAM, EXTENTS_PARAM)

# The errors field: a count of violations, then the first one's site,
# dimension, index (low and high 32 bits) and extent (likewise).
COUNT, SITE, DIM, INDEX_LO, INDEX_HI, EXTENT_LO, EXTENT_HI = range(7)
ERRORS_SIZE = 8
EXTENTS_PER_PARAM = 3


class BoundsSite(NamedTuple):
    """Where a checked access is: the kernel, the device function it was
    inlined from (or None), the file and line (or, without a file, the line
    in the function's source) and column, its source text, and how many
    indices it checks (one per dimension, or one flat index)."""
    kernel: str
    function: str | None
    file: str | None
    line: int | None
    column: int
    text: str
    dimensions: int


def guard_accesses(function: ir.IRFunction):
    """Skip each field or array access of a debug kernel whose index is negative.

    Runs once types are annotated: a skipped load or atomic yields a zero
    of its own type. A load becomes ``index >= 0 ? load : 0`` (only the
    chosen arm is evaluated, on every backend); a store or an atomic
    statement runs under ``if index >= 0``, and an atomic whose old value is
    assigned assigns 0 otherwise. The index is evaluated again by the test,
    which is harmless: indices have no effects but a check's record of the
    first violation. Accesses to the hidden fields and to GPU scalar packs
    are Tack's own, and left alone.
    """
    if getattr(function, 'bounds_sites', None) is None or \
            getattr(function, '_bounds_guarded', False):
        return
    function._bounds_guarded = True
    own = set(HIDDEN_PARAMS) | {p.name for p in function.params
                                if getattr(p, '_is_scalar_pack', False)}
    names = {p.name for p in function.params}
    for node in walk_ir(function.body):
        names.update(n for n in (getattr(node, 'name', None), getattr(node, 'target', None),
                                 getattr(node, 'var', None)) if isinstance(n, str))
    function.body = _guard_stmts(function.body, own, names)


def _user_access(node, own) -> bool:
    return not (isinstance(node.field, ir.IRName) and node.field.name in own)


def _guard_stmts(stmts, own, names):
    out = []
    for stmt in stmts:
        if isinstance(stmt, ir.IRAssign) and isinstance(stmt.value, ir.IRAtomicOp):
            atomic = stmt.value
            _guard_children(atomic, own, names)
            if _user_access(atomic, own):
                out.extend(_collective_first(atomic, names))
                otherwise = ir.IRAssign(stmt.target, _zero(atomic.dtype))
                otherwise._resolved_type = stmt._resolved_type
                stmt = ir.IRIf(_nonnegative(atomic.index), [stmt], [otherwise])
            out.append(stmt)
            continue
        _guard_children(stmt, own, names)
        if isinstance(stmt, (ir.IRFieldStore, ir.IRAtomicOp)) and _user_access(stmt, own):
            out.extend(_collective_first(stmt, names))
            stmt = ir.IRIf(_nonnegative(stmt.index), [stmt], [])
        out.append(stmt)
    return out


def _collective_first(access, names) -> list:
    """A store's or atomic's value computed before its guard, if it holds a block
    reduction: every thread of the group must reach a collective."""
    if not any(isinstance(n, ir.IRBlockReduce) for n in walk_ir(access.value)):
        return []
    name = fresh_name('__tack_collective__', names)
    names.add(name)
    value = access.value
    assign = ir.IRAssign(name, value)
    assign._resolved_type = value.dtype
    access.value = ir.IRName(name)
    access.value.dtype = value.dtype
    return [assign]


def _guard_children(node, own, names):
    for attr, role in CHILD_FIELDS[type(node)]:
        if isinstance(node, ir.IRParallelFor) and attr == 'end':
            continue    # evaluated on the host, never generated
        value = getattr(node, attr)
        if role == 'stmts':
            setattr(node, attr, _guard_stmts(value, own, names))
        elif role == 'exprs':
            setattr(node, attr, [_guard_expr(v, own, names) for v in value])
        elif role in ('expr', 'optional_expr') and value is not None:
            setattr(node, attr, _guard_expr(value, own, names))


def _guard_expr(node, own, names):
    _guard_children(node, own, names)
    if isinstance(node, ir.IRFieldLoad) and _user_access(node, own):
        guarded = ir.IRIfExp(_nonnegative(node.index), node, _zero(node.dtype))
        guarded.dtype = node.dtype
        return guarded
    return node


def _nonnegative(index):
    """``index >= 0``, typed, on a copy of the index."""
    from tack.lang.ir_traversal import clone_ir
    copy = clone_ir(index)
    zero = ir.IRConstant(0)
    zero.dtype = copy.dtype
    test = ir.IRCompare(">=", copy, zero)
    test._operand_type = copy.dtype
    test.dtype = i32
    return test


def _zero(dtype):
    zero = ir.IRConstant(0 if dtype in INTEGER_TYPES else 0.0)
    zero.dtype = dtype
    return zero
