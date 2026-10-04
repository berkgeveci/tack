"""Promoted-precision, Python-style floating floor division and remainder.

Use a truncating remainder, correct its sign to the divisor, and reconstruct
the quotient before snapping it to an integer-valued float. Direct floor(a/b)
can disagree at rounded integer boundaries. Zero divisors are outside the
caller domain; signed zero and nonfinite inputs need safe compiler math.
"""

from tack.lang import ir
from tack.lang.ir_traversal import walk_ir
from tack.lang.types import f32, f64


def uses_float_division(ir_func):
    """Whether a typed kernel needs safe math for floating // or %."""
    return any(isinstance(node, ir.IRBinOp) and node.op in ('//', '%')
               and getattr(node, 'dtype', None) in (f32, f64)
               for node in walk_ir(ir_func))


def float_division_expr(node, left, right, type_map, helpers, *, dtype=None):
    if dtype is None:
        dtype = getattr(node, 'dtype', None)
    if node.op not in ('//', '%') or dtype not in (f32, f64):
        return None
    helpers.add((node.op, dtype))
    ctype = type_map[dtype]
    return f'{_helper_name(node.op, dtype)}(({ctype})({left}), ({ctype})({right}))'


def _helper_name(op, dtype):
    operation = 'floordiv' if op == '//' else 'mod'
    return f'__tack_{operation}_{dtype.name}__'


def float_division_helpers(helpers, type_map, qualifier, *, cuda_math=False):
    """Emit only the operations and precisions used by this kernel."""
    lines = []
    for op, dtype in sorted(helpers, key=lambda item: (item[0], item[1].name)):
        ctype = type_map[dtype]
        suffix = 'f' if cuda_math and dtype is f32 else ''
        zero, one, half = ('0.0f', '1.0f', '0.5f') if dtype is f32 else ('0.0', '1.0', '0.5')
        lines.extend([
            f'{qualifier} {ctype} {_helper_name(op, dtype)}({ctype} a, {ctype} b) {{',
            f'    {ctype} r = fmod{suffix}(a, b);',
        ])
        if op == '//':
            lines.append(f'    {ctype} q = (a - r) / b;')
        lines.extend([
            f'    if (r != {zero}) {{',
            f'        if ((r < {zero}) != (b < {zero})) {{',
            '            r += b;',
        ])
        if op == '//':
            lines.append(f'            q -= {one};')
        lines.extend(['        }', '    }'])
        if op == '%':
            lines.extend([
                '    else {',
                f'        r = copysign{suffix}({zero}, b);',
                '    }',
                '    return r;',
            ])
        else:
            lines.extend([
                f'    if (q == {zero}) return copysign{suffix}({zero}, a / b);',
                f'    {ctype} result = floor{suffix}(q);',
                f'    if (q - result > {half}) result += {one};',
                '    return result;',
            ])
        lines.extend(['}', ''])
    return lines
