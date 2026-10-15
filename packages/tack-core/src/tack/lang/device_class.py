"""``@tack.device_class``: a class whose objects kernels can take and specialize on.

An object of a device class is passed to a ``@tack.kernel`` like any
argument, and its class decides the code. Its ``@tack.func`` methods are
inlined, with ``self`` resolved:

- a class-level ``int`` or ``float`` → a compile-time constant
- an instance ``int`` or ``float`` → a runtime kernel parameter
- an instance array → an extra kernel parameter
- an instance device-class object → nested: its methods and attributes too

``data_oriented`` is the same decorator under Taichi's name, kept so code
written for Taichi, or for earlier Tack, still works.
"""


def device_class(cls):
    """Mark a class as a device class: its objects can be kernel arguments.

    Methods decorated with @tack.func are collected and made available
    for inlining when instances are passed to kernels.
    """
    from tack.lang.template_rewrite import template_func_methods

    cls._device_class = True
    # A record of the methods at decoration, inherited ones included.
    # Lowering looks them up again per class, so subclasses defined later,
    # decorated or not, get their own.
    cls._tack_func_methods = template_func_methods(cls)

    return cls


#: Taichi's name for ``device_class``.
data_oriented = device_class
