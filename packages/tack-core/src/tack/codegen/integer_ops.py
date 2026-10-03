"""Portable fixed-width integer expressions for the C-family backends.

Arithmetic uses unsigned carriers (at least 32 bits to avoid C's signed
integer promotions). Signed results are reconstructed with representable
casts only, rather than relying on signed overflow or out-of-range casts.
Helpers take values as parameters so operands execute once.
"""

from tack.codegen.integer_division import INTEGER_TYPES, UNSIGNED_TYPES
from tack.lang.types import u8, u16, u32, u64

_UNSIGNED = {8: u8, 16: u16, 32: u32, 64: u64}
_OPERATIONS = {'+': 'add', '-': 'sub', '*': 'mul', '&': 'and',
               '|': 'or', '^': 'xor', '<<': 'shl', '>>': 'shr',
               'neg': 'neg', '~': 'invert', 'abs': 'abs',
               'min': 'min', 'max': 'max'}


class IntegerCodeGen:
    def __init__(self, type_map, bitcast=False):
        self.type_map = type_map
        self.bitcast = bitcast
        self.wraps = set()
        self.operations = set()

    @staticmethod
    def _wrap_name(dtype):
        return f'__tack_bits_{dtype.name}__'

    def convert(self, expr, source, target):
        if target not in INTEGER_TYPES or source not in INTEGER_TYPES or source is target:
            return expr
        if target in UNSIGNED_TYPES:
            return f'(({self.type_map[target]})({expr}))'
        if source.bits < target.bits or (source not in UNSIGNED_TYPES and source.bits == target.bits):
            return f'(({self.type_map[target]})({expr}))'
        self.wraps.add(target)
        unsigned = self.type_map[_UNSIGNED[target.bits]]
        return f'{self._wrap_name(target)}(({unsigned})({expr}))'

    def operation(self, op, dtype, *args, noinline=False):
        if dtype not in INTEGER_TYPES or op not in _OPERATIONS:
            return None
        self.operations.add((op, dtype, noinline))
        if dtype not in UNSIGNED_TYPES:
            self.wraps.add(dtype)
        suffix = '_noinline' if noinline else ''
        return f'__tack_{_OPERATIONS[op]}_{dtype.name}{suffix}__({", ".join(args)})'

    def definitions(self, qualifier):
        lines = []
        for dtype in sorted(self.wraps, key=lambda t: t.name):
            t, u = self.type_map[dtype], self.type_map[_UNSIGNED[dtype.bits]]
            limit = (1 << (dtype.bits - 1)) - 1
            result = f'as_type<{t}>(x)' if self.bitcast else \
                f'x <= {limit}ULL ? ({t})x : ({t})(-1 - ({t})(({u})~x))'
            lines += [f'{qualifier} {t} {self._wrap_name(dtype)}({u} x) {{',
                      f'    return {result};',
                      '}', '']
        for op, dtype, noinline in sorted(self.operations, key=lambda item: (item[0], item[1].name, item[2])):
            t = self.type_map[dtype]
            u = self.type_map[_UNSIGNED[dtype.bits]]
            carrier = self.type_map[u32 if dtype.bits < 32 else _UNSIGNED[dtype.bits]]
            a, b = f'(({carrier})a)', f'(({carrier})b)'
            binary = op in ('+', '-', '*', '&', '|', '^', '<<', '>>', 'min', 'max')
            # Shift counts have their own domain and do not change the result width.
            bt = self.type_map[u64] if op in ('<<', '>>') else t
            params = f'{t} a, {bt} b' if binary else f'{t} a'
            if op in ('min', 'max'):
                expr = f'a {"<" if op == "min" else ">"} b ? a : b'
            else:
                if op == 'neg' or op == 'abs':
                    raw = f'(({carrier})0 - {a})'
                elif op == '~':
                    raw = f'~{a}'
                elif op == '>>' and dtype not in UNSIGNED_TYPES:
                    # Arithmetic right shift without implementation-defined signed >>.
                    raw = f'(a < 0 ? ~((({carrier})(({u})~a)) >> b) : ({a} >> b))'
                else:
                    raw = f'({a} {op} {b})'
                bits = f'(({u})({raw}))'
                expr = bits if dtype in UNSIGNED_TYPES else f'{self._wrap_name(dtype)}({bits})'
                if op == 'abs':
                    expr = 'a' if dtype in UNSIGNED_TYPES else f'a < 0 ? {expr} : a'
            suffix = '_noinline' if noinline else ''
            name = f'__tack_{_OPERATIONS[op]}_{dtype.name}{suffix}__'
            declaration = '__attribute__((noinline))' if noinline else qualifier
            lines += [f'{declaration} {t} {name}({params}) {{', f'    return {expr};', '}', '']
        return lines
