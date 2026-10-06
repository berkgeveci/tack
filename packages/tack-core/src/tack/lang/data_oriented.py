"""Tack data_oriented decorator — marks classes whose methods can be inlined into kernels.

Classes decorated with @tack.data_oriented can be passed as template arguments
to @tack.kernel functions.  Their @tack.func methods are inlined at compile time
with self references resolved:

- self.scalar_attr (int, float) → compile-time constant
- self.field_attr (tack.Field) → extra kernel parameter
"""

def data_oriented(cls):
    """Mark a class as data-oriented for Tack template dispatch.

    Methods decorated with @tack.func are collected and made available
    for inlining when instances are passed as kernel template arguments.
    """
    from tack.lang.template_rewrite import template_func_methods

    cls._data_oriented = True
    # A record of the methods at decoration, inherited ones included.
    # Lowering looks them up again per class, so subclasses defined later,
    # decorated or not, get their own.
    cls._tack_func_methods = template_func_methods(cls)

    return cls
