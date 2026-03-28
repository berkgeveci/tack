"""Field copy, cast, fill, and concat utilities — all GPU kernel-based.

Usage:
    from tack import algorithms

    algorithms.copy(src, dst, n)
    algorithms.fill_value(field, value, n)
    algorithms.copy_with_offset(src, dst, dst_offset, n)
"""

import tack


@tack.kernel
def _copy_kernel(src, dst, n):
    for i in range(n):
        dst[i] = src[i]


@tack.kernel
def _fill_kernel(dst, val, n):
    for i in range(n):
        dst[i] = val


@tack.kernel
def _copy_offset_kernel(src, dst, dst_offset, n):
    """Copy n elements from src into dst starting at dst_offset."""
    for i in range(n):
        dst[dst_offset + i] = src[i]


def copy(src, dst, n):
    """Copy n elements from src field to dst field on the GPU."""
    _copy_kernel(src, dst, n)


def copy_with_offset(src, dst, dst_offset, n):
    """Copy n elements from src into dst starting at dst_offset on the GPU."""
    _copy_offset_kernel(src, dst, dst_offset, n)


def fill_value(dst, value, n):
    """Fill n elements of dst field with a scalar value on the GPU."""
    _fill_kernel(dst, value, n)
