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


# Neither class changes how the number prints. A constant is used wherever
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
        f"tack.constant takes an int or a float, not {type(value).__name__}; "
        f"fields and other objects are passed to kernels as arguments")


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


def constant_ir(value):
    """The IR for reading ``value``, or None when it is not a tack.constant."""
    from tack.lang import ir
    if isinstance(value, IntConstant):
        literal = ir.IRConstant(int(value))
    elif isinstance(value, FloatConstant):
        literal = ir.IRConstant(float(value))
    else:
        return None
    return literal if value.dtype is None else ir.IRCast(value=literal, dtype=value.dtype)
