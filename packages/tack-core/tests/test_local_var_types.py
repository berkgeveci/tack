"""Local variable typing in kernels.

Every backend gives a local one storage slot, so the annotation pass gives
it one type: the promotion of everything assigned to it.  Sizing the slot
from the first assignment instead — which is what used to happen — meant an
accumulator seeded with a float literal (``total = 0.0``) was declared f32,
and every later add from an f64 field round-tripped f64 → f32 → f64 and
silently lost precision.
"""

import numpy as np
import pytest

import tack


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


def test_accumulator_seeded_from_a_literal_keeps_f64(f64_backend):
    """``total = 0.0`` widens to f64 once an f64 value is added.

    This is the pattern in tack.algorithms.cell_to_point and in any
    hand-written reduction, so the precision loss was easy to hit and
    invisible — the output field was f64 and the values merely wrong in
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


def test_accumulation_over_a_loop_keeps_f64(f64_backend):
    """A real reduction: the sum must match numpy's f64 sum exactly."""

    @tack.kernel
    def total_of(src, out, n):
        for i in range(n):
            total = 0.0
            for j in range(n):
                total = total + src[j]
            out[i] = total

    rng = np.random.default_rng(3)
    values = rng.random(16)
    got = _run(total_of, values)
    np.testing.assert_allclose(got, values.sum(), rtol=1e-15)


def test_widening_applies_before_the_widening_assignment(f64_backend):
    """Reads of the local ahead of the f64 store also see f64.

    The slot has one type for the whole function, so an early read must
    not observe a narrower one.
    """

    @tack.kernel
    def early_read(src, out, n):
        for i in range(n):
            total = 0.0
            seen = total + src[i]      # reads total before it is widened
            total = total + src[i]
            out[i] = seen + total - src[i]

    values = np.array([EXACT], dtype=np.float64)
    np.testing.assert_array_equal(_run(early_read, values), values)


def test_int_accumulator_is_not_turned_into_a_float(f64_backend):
    """Widening is promotion, not floatification — int stays int."""

    @tack.kernel
    def count_up(src, out, n):
        for i in range(n):
            count = 0
            for j in range(n):
                count = count + 2
            out[i] = float(count)

    values = np.zeros(8, dtype=np.float64)
    np.testing.assert_array_equal(_run(count_up, values), np.full(8, 16.0))


def test_loop_variable_type_is_not_widened(f64_backend):
    """A loop counter stays an integer even next to f64 arithmetic."""

    @tack.kernel
    def index_math(src, out, n):
        for i in range(n):
            acc = 0.0
            for j in range(n):
                acc = acc + src[j] * 2.0
            out[i] = acc / 2.0

    rng = np.random.default_rng(11)
    values = rng.random(8)
    np.testing.assert_allclose(_run(index_math, values), values.sum(), rtol=1e-14)
