"""Warp code generation: typed Tack IR to the source of a ``@wp.kernel``.

A prototype. NVIDIA Warp's kernel language is Python parsed by Warp, so
this generator writes Python rather than C. By the time IR reaches a
backend, templates are expanded, vectors are scalarized and every
expression carries its ``dtype``, so none of what Warp's front end lacks
(methods on structs, struct inheritance, untyped parameters) is needed.

Warp does not promote mixed operands, so every operand is converted to the
type the annotator gave its operation (``wp.float32(x)``), and every local
is declared once, with its type, at kernel scope. Arrays are flat
``wp.array`` buffers indexed as Tack linearized them.

Where Warp's semantics differ from Tack's, helpers carry Tack's:
- ``//`` and ``%``, integer and floating (Warp's integer ``%`` truncates
  like C while its ``//`` rounds down; the floating ones follow
  ``float_division.py``, signs of zero included);
- integer ``+``, ``-``, ``*``, ``<<``, negation, ``abs`` and power wrap in
  an unsigned carrier (Warp's C++ leaves signed overflow undefined, and
  clang folded ``(x * y) // y`` to ``x``);
- sequential loops count their trips in 64 bits (Warp's ``range`` is 32-bit);
- floating min/max prefer a number to NaN, and floating negation is a
  product with -1 (Warp's gives +0.0 for 0.0);
- no floating contraction (``fuse_fp``), so a named value is rounded once.

Debug mode's bounds checks (``IRBoundsCheck``) call one helper per kernel,
as ``codegen/bounds.py`` has it in C.

Not yet: textures, workgroups (shared memory, barriers, block
reductions) and ``print``.
"""

from tack.codegen.identifiers import kernel_entry_name
from tack.lang import ir, ir_bounds
from tack.lang.ir_traversal import walk_ir
from tack.lang.types import (
    ScalarType,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
)

_WARP_TYPES = {
    i8: "wp.int8", u8: "wp.uint8", i16: "wp.int16", u16: "wp.uint16",
    i32: "wp.int32", u32: "wp.uint32", i64: "wp.int64", u64: "wp.uint64",
    f32: "wp.float32", f64: "wp.float64",
}

_BINOPS = {"+": "+", "-": "-", "*": "*", "/": "/", "&": "&", "|": "|",
           "^": "^", "<<": "<<", ">>": ">>"}

_MATH = {
    "sqrt": "wp.sqrt", "sin": "wp.sin", "cos": "wp.cos", "tan": "wp.tan",
    "asin": "wp.asin", "acos": "wp.acos", "atan": "wp.atan",
    "atan2": "wp.atan2", "sinh": "wp.sinh", "cosh": "wp.cosh",
    "tanh": "wp.tanh", "exp": "wp.exp", "log": "wp.log", "log2": "wp.log2",
    "log10": "wp.log10", "floor": "wp.floor", "ceil": "wp.ceil",
    "abs": "wp.abs", "fabs": "wp.abs", "pow": "wp.pow",
}

_INDEX = i64
_FLOATS = frozenset((f32, f64))
#: Integer operations that can overflow, done through `_wrapping_helper`.
_WRAPPING = {"+": "add", "-": "sub", "*": "mul", "<<": "shl", "neg": "neg", "abs": "abs"}
_SIGNED = frozenset((i8, i16, i32, i64))


def warp_type(dtype: ScalarType) -> str:
    return _WARP_TYPES[dtype]


def _carrier(dtype: ScalarType) -> str:
    """The unsigned Warp type an integer of ``dtype`` computes in: 32 bits at least."""
    return "wp.uint64" if dtype.bits == 64 else "wp.uint32"


def _v(name: str) -> str:
    """A Tack name as Warp source spells it. Warp copies names into C++
    unchanged, so a parameter called ``short`` failed to compile; the prefix
    makes no keyword and no generated name (those begin ``__``)."""
    return f"v_{name}"


class WarpCodeGen:
    """Generates one Warp kernel (with the helpers it calls) from typed IR."""

    def __init__(self, ir_func: ir.IRFunction):
        self.ir_func = ir_func
        self._lines: list[str] = []
        self._indent = 0
        self._arrays: dict[str, ScalarType] = {}
        self._locals: dict[str, ScalarType] = {}
        self._helpers: dict[str, list[str]] = {}
        self._tables: dict[tuple, str] = {}
        #: ``(dtype, values)`` of each constant table, in parameter order.
        self.tables: list[tuple[ScalarType, tuple]] = []
        self._fresh = 0

    # ── Entry ────────────────────────────────────────────────────────

    def generate(self) -> tuple[str, str]:
        """``(source, entry name)``: Python defining the kernel and its helpers."""
        func = self.ir_func
        params = []
        for param in func.params:
            if param._is_texture:
                raise NotImplementedError("Warp backend: textures are not supported yet")
            t = warp_type(param.type_annotation)
            if param._is_array is not False:
                self._arrays[param.name] = param.type_annotation
                params.append(f"{_v(param.name)}: wp.array(dtype={t})")
            else:
                self._locals[param.name] = param.type_annotation
                params.append(f"{_v(param.name)}: {t}")
        loop = next(s for s in func.body if isinstance(s, ir.IRParallelFor))
        params.append("__n__: wp.int64")
        params.extend(f"__ext_{k}__: wp.int64" for k in range(len(loop.dims or ())))

        entry = kernel_entry_name(func.name)
        self._indent += 1
        self._declare_locals(func.body)
        self._body(func.body)
        self._indent -= 1
        # The tables the body read follow every other parameter.
        params.extend(f"{name}: wp.array(dtype={warp_type(dtype)})"
                      for (dtype, _), name in zip(self.tables, self._tables.values()))

        lines = ["import warp as wp", ""]
        for helper in self._helpers.values():
            lines += [*helper, ""]
        # A module of its own, named by the kernel's content: variants with the
        # same code share one, and Warp's cache finds it in every process.
        # Without contraction: Warp writes every operation as a statement of
        # its own, so NVRTC fused a named product into a later statement's
        # sum, and Tack rounds a named value once (CUDACodeGen's
        # `_rounded_products` keeps it so on Tack's own CUDA backend).
        lines += ['@wp.kernel(module="unique", module_options={"fuse_fp": False})',
                  f"def {entry}({', '.join(params)}):"]
        return "\n".join(lines + self._lines) + "\n", entry

    # ── Statements ───────────────────────────────────────────────────

    def _emit(self, line: str):
        self._lines.append("    " * self._indent + line)

    def _declare_locals(self, body):
        """Give every typed local one type at kernel scope, as a typed zero:
        Warp refuses a later assignment of another type, and a name first
        bound inside a branch or loop must exist after it."""
        for node in walk_ir(body):
            if isinstance(node, ir.IRAssign) and node.target not in self._locals:
                if node._resolved_type is None:
                    # A local naming an array: bound where it is assigned.
                    source = node.value
                    if not (isinstance(source, ir.IRName) and source.name in self._arrays):
                        raise NotImplementedError(
                            f"Warp backend: untyped local '{node.target}'")
                    self._arrays[node.target] = self._arrays[source.name]
                    continue
                self._locals[node.target] = node._resolved_type
                self._emit(f"{_v(node.target)} = {self._zero(node._resolved_type)}")
            elif isinstance(node, ir.IRSequentialFor) and node.var not in self._locals:
                self._locals[node.var] = _INDEX
                self._emit(f"{_v(node.var)} = {self._zero(_INDEX)}")

    def _body(self, stmts):
        if not stmts:
            self._emit("pass")
        for stmt in stmts:
            self._stmt(stmt)

    def _block(self, header, stmts):
        self._emit(header)
        self._indent += 1
        self._body(stmts)
        self._indent -= 1

    def _stmt(self, node):
        if isinstance(node, ir.IRParallelFor):
            self._parallel_for(node)
        elif isinstance(node, ir.IRSequentialFor):
            self._sequential_for(node)
        elif isinstance(node, ir.IRWhile):
            self._block(f"while {self._cond(node.condition)}:", node.body)
        elif isinstance(node, ir.IRIf):
            self._block(f"if {self._cond(node.condition)}:", node.then_body)
            if node.else_body:
                self._block("else:", node.else_body)
        elif isinstance(node, ir.IRAssign):
            if node.target in self._arrays:
                self._emit(f"{_v(node.target)} = {_v(self._array_name(node.value))}")
            else:
                self._emit(f"{_v(node.target)} = {self._typed(node.value, self._locals[node.target])}")
        elif isinstance(node, ir.IRArrayStore):
            array = self._array_name(node.array)
            self._emit(f"{_v(array)}[{self._index(node.index)}] = "
                       f"{self._typed(node.value, self._arrays[array])}")
        elif isinstance(node, ir.IRAtomicOp):
            self._emit(self._atomic(node))
        elif isinstance(node, ir.IRLocalAlloc):
            if not isinstance(node.size, ir.IRConstant):
                raise NotImplementedError("Warp backend: a local array's size must be a constant")
            self._arrays[node.name] = node.dtype
            self._emit(f"{_v(node.name)} = wp.vector(dtype={warp_type(node.dtype)}, "
                       f"length={node.size.value})")
        elif isinstance(node, ir.IRReturn):
            self._emit("return")
        elif isinstance(node, ir.IRBreak):
            self._emit("break")
        elif isinstance(node, ir.IRContinue):
            # The kernel body is one iteration of the parallel loop.
            self._emit("return" if node.outermost else "continue")
        elif isinstance(node, ir.IRCall):
            self._emit(self._expr(node))
        else:
            raise NotImplementedError(f"Warp backend: cannot emit {type(node).__name__}")

    def _parallel_for(self, node: ir.IRParallelFor):
        """One Warp thread per iteration, launched over ``__n__``; a 2D or 3D
        loop recovers its indices from the flat one (x the fastest)."""
        if node.dims:
            self._emit("__rest__ = wp.int64(wp.tid())")
            for k in range(len(node.dims) - 1, 0, -1):
                self._locals[node.dims[k]] = _INDEX
                self._emit(f"{_v(node.dims[k])} = __rest__ % __ext_{k}__")
                self._emit(f"__rest__ = __rest__ // __ext_{k}__")
            self._locals[node.dims[0]] = _INDEX
            self._emit(f"{_v(node.dims[0])} = __rest__")
        else:
            self._locals[node.var] = _INDEX
            self._emit(f"{_v(node.var)} = wp.int64(wp.tid())")
        self._body(node.body)

    def _sequential_for(self, node: ir.IRSequentialFor):
        """Python's ``range`` over 64-bit bounds, captured at entry. Warp's own
        ``range`` takes 32-bit bounds, and a wrapped 64-bit bound truncated to
        32 bits ran nearly forever, so the loop counts its trips (in unsigned
        arithmetic, which cannot overflow) and steps before the body, so a
        ``continue`` steps too."""
        k, n = self._fresh_name("k"), self._fresh_name("trips")
        start = self._fresh_name("start")
        step = self._typed(node.step, _INDEX) if node.step is not None else "wp.int64(1)"
        self._emit(f"{start} = {self._typed(node.start, _INDEX)}")
        stride = self._fresh_name("step")
        self._emit(f"{stride} = {step}")
        self._emit(f"{n} = {self._trip_count_helper()}({start}, "
                   f"{self._typed(node.end, _INDEX)}, {stride})")
        self._emit(f"{k} = wp.uint64(0)")
        self._emit(f"while {k} < {n}:")
        self._indent += 1
        self._emit(f"{_v(node.var)} = wp.int64(wp.uint64({start}) + {k} * wp.uint64({stride}))")
        self._emit(f"{k} = {k} + wp.uint64(1)")
        self._body(node.body)
        self._indent -= 1

    def _trip_count_helper(self):
        name = "tack_trip_count"
        if name not in self._helpers:
            self._helpers[name] = [
                "@wp.func",
                f"def {name}(start: wp.int64, end: wp.int64, step: wp.int64) -> wp.uint64:",
                "    if step > wp.int64(0) and start < end:",
                ("        return (wp.uint64(end) - wp.uint64(start) - wp.uint64(1)) // wp.uint64(step)"
                 " + wp.uint64(1)"),
                "    if step < wp.int64(0) and start > end:",
                ("        return (wp.uint64(start) - wp.uint64(end) - wp.uint64(1)) // "
                 "(wp.uint64(0) - wp.uint64(step)) + wp.uint64(1)"),
                "    return wp.uint64(0)",
            ]
        return name

    # ── Expressions ──────────────────────────────────────────────────

    def _fresh_name(self, stem):
        self._fresh += 1
        return f"__{stem}{self._fresh}__"

    def _zero(self, dtype):
        return f"{warp_type(dtype)}({'0.0' if dtype in _FLOATS else '0'})"

    def _array_name(self, node):
        if not isinstance(node, ir.IRName) or node.name not in self._arrays:
            raise NotImplementedError("Warp backend: an array must be named by a parameter")
        return node.name

    def _index(self, node):
        return self._typed(node, _INDEX)

    def _typed(self, node, dtype):
        """``node`` as a value of ``dtype``: Warp converts nothing implicitly."""
        if isinstance(node, (ir.IRCompare, ir.IRBoolOp)) or (
                isinstance(node, ir.IRUnaryOp) and node.op == "not"):
            return f"({warp_type(dtype)}(1) if {self._cond(node)} else {warp_type(dtype)}(0))"
        if isinstance(node, ir.IRConstant):
            return self._constant(node.value, dtype)
        text = self._expr(node)
        if self._type_of(node) is dtype:
            return text
        return f"{warp_type(dtype)}({text})"

    def _type_of(self, node):
        if isinstance(node, ir.IRName):
            return self._locals.get(node.name)
        return getattr(node, "dtype", None)

    def _constant(self, value, dtype):
        if dtype in _FLOATS:
            return f"{warp_type(dtype)}({float(value)!r})"
        return f"{warp_type(dtype)}({int(value)})"

    def _cond(self, node) -> str:
        """``node`` as a Warp bool."""
        if isinstance(node, ir.IRCompare):
            t = node._operand_type
            return f"({self._typed(node.left, t)} {node.op} {self._typed(node.right, t)})"
        if isinstance(node, ir.IRBoolOp):
            return "(" + f" {node.op} ".join(self._cond(v) for v in node.values) + ")"
        if isinstance(node, ir.IRUnaryOp) and node.op == "not":
            return f"(not {self._cond(node.operand)})"
        t = self._type_of(node)
        return f"({self._expr(node)} != {self._constant(0, t)})"

    def _expr(self, node) -> str:
        if isinstance(node, ir.IRConstant):
            return self._constant(node.value, node.dtype)
        if isinstance(node, ir.IRName):
            return _v(node.name)
        if isinstance(node, ir.IRArrayLoad):
            return f"{_v(self._array_name(node.array))}[{self._index(node.index)}]"
        if isinstance(node, ir.IRBinOp):
            return self._binop(node)
        if isinstance(node, ir.IRUnaryOp):
            if node.op == "+":
                return self._expr(node.operand)
            if node.op == "not":
                return self._typed(node, node.dtype)
            operand = self._typed(node.operand, node.dtype)
            if node.op == "-" and node.dtype not in _FLOATS:
                return f"{self._wrapping_helper('neg', node.dtype)}({operand})"
            if node.op == "-":
                # Warp's float negation gives +0.0 for 0.0 (as 0 - x would);
                # a product with -1 flips the sign of every value exactly.
                return f"({operand} * {self._constant(-1.0, node.dtype)})"
            return f"({node.op}{operand})"
        if isinstance(node, (ir.IRCompare, ir.IRBoolOp)):
            return self._typed(node, node.dtype)
        if isinstance(node, ir.IRIfExp):
            return (f"({self._typed(node.then_value, node.dtype)} if {self._cond(node.condition)} "
                    f"else {self._typed(node.else_value, node.dtype)})")
        if isinstance(node, ir.IRCast):
            return self._typed(node.value, node.dtype)
        if isinstance(node, ir.IRCall):
            return self._call(node)
        if isinstance(node, ir.IRAtomicOp):
            return self._atomic(node)
        if isinstance(node, ir.IRTableLoad):
            return self._table_load(node)
        if isinstance(node, ir.IRBoundsCheck):
            return self._bounds_check(node)
        raise NotImplementedError(f"Warp backend: expression {type(node).__name__}")

    def _binop(self, node: ir.IRBinOp) -> str:
        t = node.dtype
        left = self._typed(node.left, t)
        right = self._typed(node.right, t)
        if node.op in ("//", "%"):
            return f"{self._division_helper(node.op, t)}({left}, {right})"
        if node.op == "**":
            if t in _FLOATS:
                return f"wp.pow({left}, {right})"
            exponent = self._type_of(node.right)
            return (f"{self._power_helper(t, exponent)}({left}, "
                    f"{self._typed(node.right, exponent)})")
        if node.op in _WRAPPING and t not in _FLOATS:
            return f"{self._wrapping_helper(node.op, t)}({left}, {right})"
        return f"({left} {_BINOPS[node.op]} {right})"

    def _call(self, node: ir.IRCall) -> str:
        t = node.dtype
        args = [self._typed(a, t) for a in node.args]
        name = node.func_name
        if name in ("min", "max"):
            pick = self._float_extremum_helper(name, t) if t in _FLOATS else f"wp.{name}"
            out = args[0]
            for arg in args[1:]:
                out = f"{pick}({out}, {arg})"
            return out
        if name == "exp2":
            return f"wp.pow({self._constant(2, t)}, {args[0]})"
        if name == "abs" and t not in _FLOATS:
            return f"{self._wrapping_helper('abs', t)}({args[0]})"
        if name == "pow" and t not in _FLOATS:
            # The exponent keeps its own type, as for ``**``.
            exponent = self._type_of(node.args[1])
            return (f"{self._power_helper(t, exponent)}({args[0]}, "
                    f"{self._typed(node.args[1], exponent)})")
        return f"{_MATH[name]}({', '.join(args)})"

    def _table_load(self, node: ir.IRTableLoad) -> str:
        """A constant table read at a runtime index; past either end, the last
        value (as ``tables.py``). Each table is an array argument the backend
        fills once per variant: a ``wp.constant`` vector became a local
        ``wp::vec_t`` that every thread built, 15 KB for a hexahedron's
        triangles, which made a contour 30 times slower on CUDA."""
        key = (node.dtype.name, node.values)
        if key not in self._tables:
            self._tables[key] = f"__tack_table_{len(self._tables)}__"
            self.tables.append((node.dtype, node.values))
        return f"{self._table_helper(node.dtype)}({self._tables[key]}, {self._index(node.index)})"

    def _table_helper(self, t):
        name = f"tack_table_{t.name}"
        if name not in self._helpers:
            w = warp_type(t)
            self._helpers[name] = [
                "@wp.func",
                f"def {name}(table: wp.array(dtype={w}), i: wp.int64) -> {w}:",
                "    n = wp.int64(table.shape[0])",
                "    k = n - wp.int64(1)",
                "    if i >= wp.int64(0) and i < n:",
                "        k = i",
                "    return table[k]",
            ]
        return name

    def _bounds_check(self, node: ir.IRBoundsCheck) -> str:
        """Debug mode: the flat index, or -1 when an index is out of bounds
        (``codegen/bounds.py``'s rule). The dimensions are tested in order with
        Warp's short-circuiting ``and``, so the first failing one is recorded."""
        helper = self._in_bounds_helper()
        errors = self._expr(node.errors)
        tests = " and ".join(
            f"({helper}({self._typed(index, _INDEX)}, {self._typed(extent, _INDEX)}, "
            f"wp.int32({node.site}), wp.int32({dim}), {errors}) != wp.int32(0))"
            for dim, (index, extent) in enumerate(zip(node.indices, node.extents)))
        return f"({self._typed(node.flat, _INDEX)} if ({tests}) else wp.int64(-1))"

    def _in_bounds_helper(self):
        """Whether ``index`` is in ``[0, extent)``; if not, the first violation's
        site, dimension, index and extent go into the errors array, the first
        elected by an atomic add on its count (``lang/ir_bounds.py``'s layout)."""
        name = "tack_in_bounds"
        if name not in self._helpers:
            b = ir_bounds
            self._helpers[name] = [
                "@wp.func",
                (f"def {name}(index: wp.int64, extent: wp.int64, site: wp.int32, dim: wp.int32, "
                 "errors: wp.array(dtype=wp.int32)) -> wp.int32:"),
                "    inside = wp.int32(0)",
                "    if index >= wp.int64(0) and index < extent:",
                "        inside = wp.int32(1)",
                f"    elif wp.atomic_add(errors, {b.COUNT}, wp.int32(1)) == wp.int32(0):",
                f"        errors[{b.SITE}] = site",
                f"        errors[{b.DIM}] = dim",
                f"        errors[{b.INDEX_LO}] = wp.int32(index)",
                f"        errors[{b.INDEX_HI}] = wp.int32(index >> wp.int64(32))",
                f"        errors[{b.EXTENT_LO}] = wp.int32(extent)",
                f"        errors[{b.EXTENT_HI}] = wp.int32(extent >> wp.int64(32))",
                "    return inside",
            ]
        return name

    def _atomic(self, node: ir.IRAtomicOp) -> str:
        array = self._array_name(node.array)
        value = self._typed(node.value, self._arrays[array])
        return f"wp.atomic_{node.op}({_v(array)}, {self._index(node.index)}, {value})"

    # ── Helpers with Python's semantics ──────────────────────────────

    def _division_helper(self, op, t):
        """Python's ``//`` and ``%``: the remainder takes the divisor's sign."""
        name = f"tack_{'floordiv' if op == '//' else 'mod'}_{t.name}"
        if name not in self._helpers:
            w = warp_type(t)
            if t in _FLOATS:
                # float_division.py's helpers: a truncating remainder corrected
                # to the divisor's sign, the quotient rebuilt and snapped; a
                # zero remainder takes the divisor's sign, a zero quotient the
                # sign of true division.
                zero, one = f"{w}(0.0)", f"{w}(1.0)"
                body = ["    r = wp.mod(a, b)", "    q = (a - r) / b",
                        f"    if r != {zero}:",
                        f"        if (r < {zero}) != (b < {zero}):",
                        "            r = r + b",
                        f"            q = q - {one}",
                        "    else:",
                        f"        r = wp.copysign({zero}, b)"]
                if op == "//":
                    body += [f"    if q == {zero}:",
                             f"        return wp.copysign({zero}, a / b)",
                             "    result = wp.floor(q)",
                             f"    if q - result > {w}(0.5):",
                             f"        result = result + {one}"]
                result = "r" if op == "%" else "result"
            elif t in _SIGNED:
                body = [
                    "    q = a / b",       # C++ truncation in Warp for integer /
                    "    r = a - q * b",
                    f"    if r != {w}(0) and ((r < {w}(0)) != (b < {w}(0))):",
                    f"        q = q - {w}(1)",
                    "        r = r + b",
                ]
                result = "q" if op == "//" else "r"
            else:
                body = ["    q = a / b", "    r = a - q * b"]
                result = "q" if op == "//" else "r"
            self._helpers[name] = ["@wp.func", f"def {name}(a: {w}, b: {w}) -> {w}:",
                                   *body, f"    return {result}"]
        return name

    def _float_extremum_helper(self, name, t):
        """Scalar floating min/max preferring a number to NaN, as C's fmin/fmax."""
        helper = f"tack_f{name}_{t.name}"
        if helper not in self._helpers:
            w, op = warp_type(t), "<" if name == "min" else ">"
            self._helpers[helper] = [
                "@wp.func", f"def {helper}(a: {w}, b: {w}) -> {w}:",
                "    if wp.isnan(a):", "        return b",
                "    if wp.isnan(b):", "        return a",
                f"    return a if a {op} b else b"]
        return helper

    def _power_helper(self, t, exponent_type):
        """Integer power by squaring in the unsigned carrier, so every product
        wraps at the base's width (the exponent is nonnegative, Tack's domain)."""
        name = f"tack_ipow_{t.name}_{exponent_type.name}"
        if name not in self._helpers:
            w, e, c = warp_type(t), warp_type(exponent_type), _carrier(t)
            self._helpers[name] = [
                "@wp.func",
                f"def {name}(base: {w}, exponent: {e}) -> {w}:",
                f"    result = {c}(1)",
                f"    factor = {c}(base)",
                "    n = wp.uint64(exponent)",
                "    while n != wp.uint64(0):",
                "        if (n & wp.uint64(1)) != wp.uint64(0):",
                "            result = result * factor",
                "        factor = factor * factor",
                "        n = n >> wp.uint64(1)",
                f"    return {w}(result)",
            ]
        return name

    def _wrapping_helper(self, op, t):
        """``+``, ``-``, ``*``, ``<<``, negation and ``abs`` wrapping at ``t``'s
        width. Warp's C++ makes signed overflow undefined (clang folded
        ``(x * y) // y`` to ``x``) and promotes 8- and 16-bit operands to a signed
        int, so they run in an unsigned carrier of at least 32 bits and convert
        back, which is modular (``integer_ops.py`` does the same in C)."""
        name = f"tack_{_WRAPPING[op]}_{t.name}"
        if name not in self._helpers:
            w, c = warp_type(t), _carrier(t)
            if op == "neg":
                params, body = f"a: {w}", f"{w}({c}(0) - {c}(a))"
            elif op == "abs":
                params = f"a: {w}"
                body = "a" if t not in _SIGNED else f"({w}({c}(0) - {c}(a)) if a < {w}(0) else a)"
            else:
                params, body = f"a: {w}, b: {w}", f"{w}({c}(a) {op} {c}(b))"
            self._helpers[name] = ["@wp.func", f"def {name}({params}) -> {w}:",
                                   f"    return {body}"]
        return name


def generate_warp_source(ir_func: ir.IRFunction) -> tuple[str, str, list]:
    """``(source, entry name, tables)`` of the Warp kernel for typed
    ``ir_func``: ``tables`` holds ``(dtype, values)`` of each constant table,
    which the kernel takes as array arguments after every other."""
    gen = WarpCodeGen(ir_func)
    source, entry = gen.generate()
    return source, entry, gen.tables
