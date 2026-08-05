"""Local variable typing in kernels.

A local variable's type is fixed by its first assignment, so an
accumulator seeded with a float literal (``total = 0.0``) is declared f32
even when everything added to it comes from an f64 field.  Every add then
round-trips f64 → f32 → f64 and silently loses precision.

The xfail below is the reproducer; the passing tests around it pin the
behaviour that is already correct so a fix can be checked against them.
"""

import numpy as np
import pytest

import tack

f64_backends = []
for _arch in ["cpu", "metal", "cuda", "hip", "level_zero"]:
    try:
        tack.init(arch=getattr(tack, _arch))
    except (ImportError, RuntimeError, OSError):
        continue
    from tack.runtime.dispatch import get_backend as _get_backend
    if _arch != "metal" and getattr(_get_backend(), "supports_f64", True):
        f64_backends.append(_arch)


@pytest.fixture(params=f64_backends)
def f64_backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


# A value that is not representable in f32.
EXACT = np.float64(0.625095466604667)


def _run(kernel, values):
    src = tack.field(dtype=tack.f64, shape=(values.size,))
    src.from_numpy(values)
    out = tack.field(dtype=tack.f64, shape=(values.size,))
    out.fill(0)
    kernel(src, out, values.size)
    return out.to_numpy()


def test_direct_expression_keeps_f64(f64_backend):
    """With no local in the way, the f64 value survives exactly."""

    @tack.kernel
    def direct(src, out, n):
        for i in range(n):
            out[i] = 0.0 + src[i]

    values = np.array([EXACT], dtype=np.float64)
    np.testing.assert_array_equal(_run(direct, values), values)


def test_local_seeded_from_a_field_keeps_f64(f64_backend):
    """Seeding the local from the field types it f64, which is correct."""

    @tack.kernel
    def seeded(src, out, n):
        for i in range(n):
            total = src[i]
            out[i] = total

    values = np.array([EXACT], dtype=np.float64)
    np.testing.assert_array_equal(_run(seeded, values), values)


@pytest.mark.xfail(
    reason="local typed f32 from its float-literal seed; f64 adds truncate",
    strict=True,
)
def test_accumulator_seeded_from_a_literal_keeps_f64(f64_backend):
    """``total = 0.0`` should widen to f64 once an f64 value is added.

    This is the pattern in tack.algorithms.cell_to_point and in any
    hand-written reduction, so the precision loss is easy to hit and
    invisible — the output field is f64 and the values merely wrong in
    the low bits.
    """

    @tack.kernel
    def accumulate(src, out, n):
        for i in range(n):
            total = 0.0
            total = total + src[i]
            out[i] = total

    values = np.array([EXACT], dtype=np.float64)
    np.testing.assert_array_equal(_run(accumulate, values), values)


def test_literal_seeded_accumulator_is_f32_precision(f64_backend):
    """Documents the current behaviour: the result is the f32 rounding.

    Delete this test when the xfail above starts passing.
    """

    @tack.kernel
    def accumulate(src, out, n):
        for i in range(n):
            total = 0.0
            total = total + src[i]
            out[i] = total

    values = np.array([EXACT], dtype=np.float64)
    got = _run(accumulate, values)
    assert got[0] == np.float64(np.float32(EXACT))
