"""Constant tables for the C-family generators (CUDA, HIP, OpenCL, Metal).

An ``IRTableLoad`` -- a vector or matrix constant read at a runtime index
-- becomes one constant array at program scope and one load from it. It
replaced a chain of conditional expressions, a comparison per entry
inlined at every lookup, which on Metal crashed the compiler service
when a loop that stores did many lookups (M1 Max, macOS 26).
"""

from tack.lang.types import f32, f64

_SUFFIX_64 = {"cuda": ("LL", "ULL"), "opencl": ("L", "UL"), "metal": ("L", "UL")}


class Tables:
    """The tables one kernel reads, each declared once."""

    def __init__(self, dialect: str):
        self._dialect = dialect
        self._names = {}                 # (dtype name, values) -> (name, dtype, values)

    def __bool__(self):
        return bool(self._names)

    def load(self, node, index: str) -> str:
        """The expression reading ``node``'s table at ``index``; past either end, the last value."""
        key = (node.dtype.name, node.values)
        if key not in self._names:
            self._names[key] = (f"__tack_table_{len(self._names)}__", node.dtype, node.values)
        name = self._names[key][0]
        n = len(node.values)
        return f"{name}[(({index}) >= 0 && ({index}) < {n}) ? ({index}) : {n - 1}]"

    def declarations(self, qualifier: str, type_map) -> list:
        lines = []
        for name, dtype, values in self._names.values():
            items = ", ".join(self._literal(v, dtype) for v in values)
            lines.append(f"{qualifier} {type_map[dtype]} {name}[{len(values)}] = {{{items}}};")
        if lines:
            lines.append("")
        return lines

    def _literal(self, value, dtype) -> str:
        if dtype in (f32, f64):
            text = repr(float(value))
            return text + ("f" if dtype == f32 else "")
        signed, unsigned = _SUFFIX_64[self._dialect]
        if -2**31 <= value < 2**31:
            return str(value)
        return f"{value}{unsigned if value >= 2**63 else signed}"
