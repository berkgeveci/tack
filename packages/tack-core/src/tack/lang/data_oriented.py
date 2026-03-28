"""Tack data_oriented decorator — marks classes whose methods can be inlined into kernels.

Classes decorated with @tack.data_oriented can be passed as template arguments
to @tack.kernel functions.  Their @tack.func methods are inlined at compile time
with self references resolved:

- self.scalar_attr (int, float) → compile-time constant
- self.field_attr (tack.Field) → extra kernel parameter
"""

from tack.lang.func import Func


def data_oriented(cls):
    """Mark a class as data-oriented for Tack template dispatch.

    Methods decorated with @tack.func are collected and made available
    for inlining when instances are passed as kernel template arguments.
    """
    cls._data_oriented = True
    cls._tack_func_methods = {}

    for name in list(vars(cls)):
        val = vars(cls)[name]
        if isinstance(val, Func):
            cls._tack_func_methods[name] = val

    return cls
