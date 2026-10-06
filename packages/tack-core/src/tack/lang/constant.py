"""Named constants a kernel may read from its defining scope.

A kernel is compiled from its own source and captures nothing from the
enclosing Python scope: a module-level ``dt = 0.01`` could be rebound
after the kernel was compiled, and the kernel would silently keep the old
value. ``tack.constant`` is the explicit way in. It fixes a value where it
is defined, and a kernel or device function that reads the name gets that
value as a literal.

The result is still an ordinary Python ``int`` or ``float``, so host code
(a NumPy reference, a loop bound, an argument) uses the same name.
"""

import numbers

import numpy as np

from tack.lang.types import INTEGER_TYPES, UNSIGNED_TYPES, ScalarType


class IntConstant(int):
    """An ``int`` that kernels may read by name."""

    dtype = None

    def __new__(cls, value, dtype=None):
        self = super().__new__(cls, value)
        self.dtype = dtype
        return self


class FloatConstant(float):
    """A ``float`` that kernels may read by name."""

    dtype = None

    def __new__(cls, value, dtype=None):
        self = super().__new__(cls, value)
        self.dtype = dtype
        return self


class ArrayConstant:
    """A vector or matrix that kernels may read by name.

    ``SUN = tack.constant((0.5, 0.5, 0.0))`` is the vector
    ``tack.Vector([0.5, 0.5, 0.0])`` in a kernel, and
    ``tack.constant(((1, 0), (0, 1)))`` the matrix with those rows, each
    component a constant with the rules of ``tack.constant``. On the host
    it indexes and iterates like a tuple of its rows and converts to a
    NumPy array: ``np.array(SUN, np.float32)``, ``SUN[1]``, ``len(SUN)``.
    It is not a tuple, so ``+`` does not concatenate it by mistake; do
    host arithmetic on ``np.asarray(SUN)``.
    """

    def __init__(self, components, shape, dtype):
        self.components = tuple(components)      # flat, row-major
        self.shape = tuple(shape)
        self.dtype = dtype

    def _rows(self):
        if len(self.shape) == 1:
            return self.components
        n = self.shape[1]
        return tuple(self.components[i * n:(i + 1) * n] for i in range(self.shape[0]))

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        return iter(self._rows())

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(self.shape) == 2 and len(index) == 2:
            return self._rows()[index[0]][index[1]]
        return self._rows()[index]

    def __array__(self, dtype=None, copy=None):
        if dtype is None and self.dtype is not None:
            dtype = self.dtype.numpy_dtype
        return np.array(self._rows(), dtype=dtype)

    def __eq__(self, other):
        return np.array_equal(np.asarray(self), np.asarray(other))

    def __hash__(self):
        return hash((self.components, self.shape))

    def __repr__(self):
        rows = self._rows()
        body = repr(tuple(rows)) if len(self.shape) == 1 else repr(tuple(tuple(r) for r in rows))
        return f"tack.constant({body}" + (f", {self.dtype})" if self.dtype is not None else ")")


# Neither scalar class changes how the number prints. A constant is used wherever
# a number is, including as a field's shape, and code that formats a number
# into generated source must get its digits.


def constant(value, dtype=None):
    """Declare a named constant that kernels and device functions may read.

    ``DT = tack.constant(0.01)`` at module (or enclosing function) scope
    lets a kernel write ``DT`` as if it were the literal ``0.01``. Without
    ``dtype`` the constant behaves exactly like that literal: an integer
    takes the narrowest of i32, i64 and u64 that holds it, and a float is
    weak, taking the precision of what it meets. With ``dtype`` it has that
    type, as ``tack.u32(747796405)`` or ``tack.f64(0.1)`` written in the
    kernel would: ``tack.constant(747796405, tack.u32)`` multiplies in
    wrapping u32 arithmetic, and ``tack.constant(0.1, tack.f64)`` is the
    exact double.

    The value is fixed here. A kernel reads the constant its name is bound
    to when the kernel is first compiled; binding the name to another
    constant afterwards does not change a compiled kernel, the same rule
    as for device functions.
    """
    if dtype is not None and not isinstance(dtype, ScalarType):
        raise TypeError(f"tack.constant dtype must be a Tack scalar type, not {dtype!r}")
    if isinstance(value, (tuple, list, np.ndarray)):
        return _array_constant(value, dtype)
    if isinstance(value, (bool, numbers.Integral)):
        value = int(value)
        if dtype is None or dtype in INTEGER_TYPES:
            _check_representable(value, dtype)
            return IntConstant(value, dtype)
        return FloatConstant(float(value), dtype)
    if isinstance(value, numbers.Real):
        if dtype in INTEGER_TYPES:
            raise TypeError(
                f"tack.constant({value!r}, {dtype!r}): an integer type needs an integer "
                f"value; convert it explicitly with int() if truncation is intended")
        return FloatConstant(float(value), dtype)
    raise TypeError(
        f"tack.constant takes an int, a float, or a tuple of them (a vector) or of "
        f"tuples (a matrix), not {type(value).__name__}; fields and other objects are "
        f"passed to kernels as arguments")


def _array_constant(value, dtype):
    """A vector from a flat sequence of numbers, a matrix from equal rows of them."""
    rows = [list(row) if isinstance(row, (tuple, list, np.ndarray)) else row for row in value]
    if not rows:
        raise ValueError("tack.constant(()) has no components")
    if all(isinstance(row, list) for row in rows):
        widths = {len(row) for row in rows}
        if len(widths) != 1 or not rows[0]:
            raise ValueError("tack.constant: the rows of a matrix must have the same, "
                             "nonzero length")
        shape = (len(rows), widths.pop())
        flat = [entry for row in rows for entry in row]
    elif any(isinstance(row, list) for row in rows):
        raise TypeError("tack.constant: mix of numbers and rows; a vector is a tuple of "
                        "numbers and a matrix a tuple of equal tuples")
    else:
        shape = (len(rows),)
        flat = rows
    components = []
    for entry in flat:
        if isinstance(entry, np.generic):
            entry = entry.item()
        if isinstance(entry, (tuple, list)):
            raise TypeError("tack.constant: a matrix has two levels of tuples, not more")
        components.append(constant(entry, dtype))
    return ArrayConstant(components, shape, dtype)


def _check_representable(value, dtype):
    if dtype is None:
        low, high = -(2**63), 2**64
    elif dtype in UNSIGNED_TYPES:
        low, high = 0, 2**dtype.bits
    else:
        low, high = -(2**(dtype.bits - 1)), 2**(dtype.bits - 1)
    if not low <= value < high:
        kind = "Tack's 64-bit integers" if dtype is None else repr(dtype)
        raise ValueError(f"tack.constant({value}) does not fit {kind}")


def constant_components(value):
    """The per-component IR of a vector or matrix constant and its shape, or None."""
    if not isinstance(value, ArrayConstant):
        return None
    return [constant_ir(component) for component in value.components], value.shape


def constant_ir(value):
    """The IR for reading a scalar ``value``, or None when it is not a tack.constant."""
    from tack.lang import ir
    if isinstance(value, IntConstant):
        literal = ir.IRConstant(int(value))
    elif isinstance(value, FloatConstant):
        literal = ir.IRConstant(float(value))
    else:
        return None
    return literal if value.dtype is None else ir.IRCast(value=literal, dtype=value.dtype)
