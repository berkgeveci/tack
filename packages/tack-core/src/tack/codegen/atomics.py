"""CUDA/HIP 64-bit CAS operations without optional native add overloads.

Each helper returns the element's value before the update, as the native
atomics do, so an atomic can be an expression."""

from tack.lang.types import f64, i64


def cuda_atomic64_helpers(operations):
    lines = []
    for op, dtype in sorted(operations, key=lambda item: (item[0], item[1].name)):
        ctype = 'double' if dtype is f64 else (
            'long long' if dtype is i64 else 'unsigned long long')
        old = ('__longlong_as_double((long long)assumed)' if dtype is f64 else
               '(long long)assumed' if dtype is i64 else 'assumed')
        if op == 'add':
            # Integer addition is modulo 2**64, including signed fields.
            update = ('__double_as_longlong(current + val)' if dtype is f64 else
                      'assumed + (unsigned long long)val')
        else:
            comparison = '<' if op == 'min' else '>'
            selected = f'(val {comparison} current ? val : current)'
            update = f'__double_as_longlong({selected})' if dtype is f64 else selected
        # On exit old == assumed: the value the successful exchange replaced.
        lines.extend([
            f'__device__ inline {ctype} tack_atomic_{op}_{dtype.name}({ctype}* addr, {ctype} val) {{',
            '    unsigned long long* bits = (unsigned long long*)addr;',
            '    unsigned long long old = atomicCAS(bits, 0ull, 0ull), assumed;',
            '    do {',
            '        assumed = old;',
            f'        {ctype} current = {old};',
            f'        old = atomicCAS(bits, assumed, (unsigned long long)({update}));',
            '    } while (old != assumed);',
            f'    return {old};',
            '}', '',
        ])
    return lines
