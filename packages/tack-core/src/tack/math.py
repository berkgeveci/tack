"""Shader-style helpers as device functions: ``fract``, ``mix``, ``clamp``...

GLSL, MSL and Taichi's ``taichi.math`` share a small vocabulary that
graphics and simulation code reaches for constantly, and that Python's
``math`` lacks. These are ordinary ``@tack.func`` device functions, so
they inline into kernels, and because Tack's arithmetic and builtins apply
to each component of a vector, each one works on scalars and on vectors
(and matrices) alike, as the GLSL functions do::

    from tack import math as tm

    @tack.kernel
    def shade(uv, out, n):
        for i in range(n):
            st = tm.fract(uv[i] * 8.0) - 0.5          # a 2-vector
            edge = tm.smoothstep(0.0, 0.05, st.x)     # a scalar
            out[i] = tm.mix(DARK, LIGHT, edge)        # 3-vectors

The definitions are GLSL's, so ``mix(x, y, a)`` is ``x * (1 - a) + y * a``
and ``smoothstep`` is the Hermite polynomial on the clamped ratio. They are
only callable from kernels and device functions; a NumPy reference writes
the same one-liners with ``np.floor``, ``np.clip`` and friends.
"""

import tack


@tack.func
def fract(x):
    """The fractional part, ``x - floor(x)``, in ``[0, 1)``."""
    return x - floor(x)


@tack.func
def mix(x, y, a):
    """The linear blend ``x * (1 - a) + y * a``; ``a`` may be a scalar or match ``x``."""
    return x * (1.0 - a) + y * a


@tack.func
def clamp(x, lo, hi):
    """``x`` limited to ``[lo, hi]``, component by component."""
    return min(max(x, lo), hi)


@tack.func
def saturate(x):
    """``x`` limited to ``[0, 1]``."""
    return min(max(x, 0.0), 1.0)


@tack.func
def smoothstep(edge0, edge1, x):
    """Hermite interpolation: 0 below ``edge0``, 1 above ``edge1``, smooth between.

    ``edge0`` greater than ``edge1`` reverses the step, as in GLSL.
    """
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@tack.func
def length(v):
    """The Euclidean length of a vector."""
    return v.norm()


@tack.func
def distance(a, b):
    """The Euclidean distance between two points."""
    return (a - b).norm()


@tack.func
def normalize(v):
    """``v`` scaled to unit length; a zero vector gives NaNs, as in GLSL."""
    return v / v.norm()
