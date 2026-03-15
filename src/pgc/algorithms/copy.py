"""Field copy and fill utilities.

Usage:
    from pgc import algorithms

    algorithms.copy(src, dst, n)
    algorithms.fill_value(field, value, n)
"""

import pgc


@pgc.kernel
def _copy_f32(src, dst, n):
    for i in range(n):
        dst[i] = src[i]


@pgc.kernel
def _fill_f32(dst, val, n):
    for i in range(n):
        dst[i] = val


def copy(src, dst, n):
    """Copy n elements from src field to dst field on the GPU."""
    _copy_f32(src, dst, n)


def fill_value(dst, value, n):
    """Fill n elements of dst field with a scalar value on the GPU."""
    _fill_f32(dst, value, n)
