"""Kernels that end in ``return expr``: the value is computed after the launch."""

import numpy as np
import pytest

import tack
from tack.lang.source_validation import UnsupportedSyntaxError

SCALE = tack.constant(2.0)


@tack.kernel
def count_live(cells, count, n) -> int:
    for i in range(n):
        if cells[i] > 0:
            tack.atomic_add(count, 0, 1)
    return count[0]


@tack.kernel
def scaled_total(x, acc, n, offset) -> tack.f32:
    half = 0.5                                     # a local bound outside the loop
    for i in range(n):
        tack.atomic_add(acc, 0, x[i])
    return half * acc[0] * SCALE + offset          # fields, constants, arguments, locals


@tack.kernel
def biggest(x, acc, n) -> float:
    for i in range(n):
        tack.atomic_max(acc, 0, x[i])
    return acc[0]


def test_kernels_return_a_value(backend):
    n = 1000
    cells = tack.field(dtype=tack.i32, shape=(n,))
    cells.from_numpy((np.arange(n) % 3 == 0).astype(np.int32))
    count = tack.field(dtype=tack.i32, shape=(1,))
    live = count_live(cells, count, n)
    assert live == 334 and isinstance(live, int)

    x = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.linspace(0, 1, n, dtype=np.float32))
    acc = tack.field(dtype=tack.f32, shape=(1,))
    total = scaled_total(x, acc, n, 1.0)
    assert isinstance(total, float)
    assert total == pytest.approx(0.5 * x.to_numpy().astype(np.float64).sum() * 2.0 + 1.0, rel=1e-5)

    acc.fill(-1.0)
    assert biggest(x, acc, n) == 1.0
    assert count_live(cells, count, n) == 668        # the count field accumulates; the kernel is reusable


@tack.data_oriented
class _Sim:
    def __init__(self, n):
        self.n = n
        self.x = tack.field(dtype=tack.f32, shape=(n,))
        self.x.fill(1.5)
        self.acc = tack.field(dtype=tack.f32, shape=(1,))

    @tack.kernel
    def mass(self) -> tack.f32:
        for i in range(self.n):
            tack.atomic_add(self.acc, 0, self.x[i])
        return self.acc[0]


def test_a_kernel_method_returns_a_value(backend):
    sim = _Sim(200)
    assert sim.mass() == 300.0


def test_a_plain_kernel_still_returns_none(backend):
    @tack.kernel
    def double(x, n):
        for i in range(n):
            x[i] *= 2.0

    x = tack.field(dtype=tack.f32, shape=(4,))
    x.fill(1.0)
    assert double(x, 4) is None
    assert (x.to_numpy() == 2.0).all()


def test_a_return_needs_an_annotation():
    with pytest.raises(UnsupportedSyntaxError, match="needs a return annotation"):
        @tack.kernel
        def bad(x, n):
            for i in range(n):
                x[i] = 1.0
            return x[0]


def test_a_return_may_not_read_a_loop_local():
    with pytest.raises(UnsupportedSyntaxError, match="reads 's', which the parallel loop assigns"):
        @tack.kernel
        def bad(x, n) -> float:
            for i in range(n):
                s = x[i]
            return s


def test_a_return_inside_the_loop_is_rejected():
    @tack.kernel  # noqa: RET503  (the point is that this is rejected)
    def bad(x, n) -> float:
        for i in range(n):
            return x[i]

    with pytest.raises(UnsupportedSyntaxError, match="only as its last statement"):
        bad.get_ir()
