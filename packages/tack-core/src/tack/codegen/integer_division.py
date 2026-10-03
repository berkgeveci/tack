"""Typed integer floor division shared by the C-family generators.

The domain excludes zero divisors and signed minimum / -1. Operands are
converted to the annotated result type before arithmetic, so host C integer
promotions cannot override Tack's promotion rules. Helpers evaluate each
operand once; all adjustments stay in integer arithmetic.
"""

from tack.lang.types import i8, i16, i32, i64, u8, u16, u32, u64

INTEGER_TYPES = frozenset((i8, i16, i32, i64, u8, u16, u32, u64))
UNSIGNED_TYPES = frozenset((u8, u16, u32, u64))


def integer_division_expr(node, left, right, type_map, helpers):
    """Emit a typed integer operation, or return None for other expressions."""
    dtype = getattr(node, 'dtype', None)
    if node.op not in ('//', '%') or dtype not in INTEGER_TYPES:
        return None
    if dtype in UNSIGNED_TYPES:
        ctype = type_map[dtype]
        op = '/' if node.op == '//' else '%'
        return f"(({ctype})((({ctype})({left})) {op} (({ctype})({right}))))"
    helpers.add((node.op, dtype))
    return f"{_helper_name(node.op, dtype)}({left}, {right})"


def _helper_name(op, dtype):
    operation = 'floordiv' if op == '//' else 'mod'
    return f"__tack_{operation}_{dtype.name}__"


def integer_division_helpers(helpers, type_map, qualifier):
    """Definitions for only the signed operations used in this kernel."""
    lines = []
    for op, dtype in sorted(helpers, key=lambda item: (item[0], item[1].name)):
        ctype = type_map[dtype]
        name = _helper_name(op, dtype)
        lines.append(f"{qualifier} {ctype} {name}({ctype} a, {ctype} b) {{")
        if op == '//':
            lines.append(f"    {ctype} q = a / b;")
        lines.append(f"    {ctype} r = a % b;")
        lines.append("    int adjust = r != 0 && ((r < 0) != (b < 0));")
        lines.append("    return q - adjust;" if op == '//' else
                     "    return adjust ? r + b : r;")
        lines.extend(("}", ""))
    return lines
