"""PGC data_oriented decorator — marks classes whose methods can be inlined into kernels.

Classes decorated with @pgc.data_oriented can be passed as template arguments
to @pgc.kernel functions.  Their @pgc.func methods are inlined at compile time
with self references resolved:

- self.scalar_attr (int, float) → compile-time constant
- self.field_attr (pgc.Field) → extra kernel parameter
"""

from pgc.lang.func import Func


def data_oriented(cls):
    """Mark a class as data-oriented for PGC template dispatch.

    Methods decorated with @pgc.func are collected and made available
    for inlining when instances are passed as kernel template arguments.
    """
    cls._data_oriented = True
    cls._pgc_func_methods = {}

    for name in list(vars(cls)):
        val = vars(cls)[name]
        if isinstance(val, Func):
            cls._pgc_func_methods[name] = val

    return cls
