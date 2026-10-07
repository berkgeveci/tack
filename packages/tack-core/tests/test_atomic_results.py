"""Atomics as expressions: the value is the element's value before the update."""

import numpy as np
import pytest

import tack
from tack.runtime.dispatch import get_backend


@tack.kernel
def allocate(counter, slots, n):
    for i in range(n):
        slot = tack.atomic_add(counter, 0, 1)      # a unique slot per thread
        slots[slot] = i


def test_the_old_value_allocates_unique_slots(backend):
    n = 4096
    counter = tack.field(dtype=tack.i32, shape=(1,))
    slots = tack.field(dtype=tack.i32, shape=(n,))
    slots.fill(-1)
    allocate(counter, slots, n)
    np.testing.assert_array_equal(np.sort(slots.to_numpy()), np.arange(n))
    assert counter[0] == n


@tack.kernel
def partial_sums(total, olds, n):
    for i in range(n):
        olds[i] = tack.atomic_add(total, 0, 3)


@pytest.mark.parametrize("dtype", [tack.i32, tack.u32, tack.f32, tack.i64, tack.u64, tack.f64])
def test_add_returns_every_partial_sum_once(backend, dtype):
    """Whatever the order, the old values of n atomic adds of 3 are 0, 3,
    ..., 3(n - 1), each seen by exactly one thread."""
    if dtype not in get_backend().supported_atomic_dtypes:
        pytest.skip(f"{dtype} atomics are not implemented on {get_backend().name}")
    n = 2048
    total = tack.field(dtype=dtype, shape=(1,))
    olds = tack.field(dtype=dtype, shape=(n,))
    partial_sums(total, olds, n)
    np.testing.assert_array_equal(np.sort(olds.to_numpy()), 3 * np.arange(n))
    assert total[0] == 3 * n


@tack.kernel
def extrema_in_order(f, u, out):
    for _ in range(1):          # one thread, so every old value is known exactly
        out[0] = tack.atomic_min(f, 0, 5.0)        # 10 -> 5, old 10
        out[1] = tack.atomic_min(f, 0, 7.0)        # stays 5, old 5
        out[2] = tack.atomic_max(f, 1, -1.0)       # -3 -> -1, old -3
        out[3] = tack.atomic_max(f, 1, -2.0)       # stays -1, old -1
        out[4] = tack.atomic_add(f, 2, 0.5)        # 1 -> 1.5, old 1
        out[5] = tack.f32(tack.atomic_max(u, 0, tack.u32(9)))   # 4 -> 9, old 4
        out[6] = tack.f32(tack.atomic_min(u, 0, tack.u32(2)))   # 9 -> 2, old 9
        out[7] = tack.f32(tack.atomic_add(u, 1, tack.u32(1)))   # 0 -> 1, old 0
        out[8] = tack.atomic_add(f, 2, 1.0) + f[2] * 0.0        # old 1.5, in an expression


def test_min_max_and_float_add_return_the_old_value(backend):
    f = tack.field(dtype=tack.f32, shape=(3,))
    f.from_numpy(np.array([10.0, -3.0, 1.0], np.float32))
    u = tack.field(dtype=tack.u32, shape=(2,))
    u.from_numpy(np.array([4, 0], np.uint32))
    out = tack.field(dtype=tack.f32, shape=(9,))
    extrema_in_order(f, u, out)
    np.testing.assert_array_equal(out.to_numpy(), [10, 5, -3, -1, 1, 4, 9, 0, 1.5])
    np.testing.assert_array_equal(f.to_numpy(), [5.0, -1.0, 2.5])
    np.testing.assert_array_equal(u.to_numpy(), [2, 1])


@tack.kernel
def operands_in_order(f, out, n):
    for i in range(n):
        # The load on the left happens before the atomic on the right.
        out[i] = f[0] - tack.atomic_add(f, 0, 1.0)


def test_an_atomic_operand_keeps_source_order(backend):
    """`f[0] - atomic_add(f, 0, 1)`: the load is evaluated first, so it sees
    the value the atomic then returns as old, and the difference is 0 for
    every thread whatever the interleaving... unless another thread's add
    lands between the two, when it is negative. It is never positive, as
    it would be if the atomic ran first."""
    n = 1024
    f = tack.field(dtype=tack.f32, shape=(1,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    operands_in_order(f, out, n)
    assert np.all(out.to_numpy() <= 0.0)
    assert f[0] == n


@tack.kernel
def vector_olds(vf, out, n):
    for i in range(n):
        old = tack.atomic_add(vf, (0,), [1.0, 10.0])
        out[i] = old.y - 10.0 * old.x


def test_a_vector_atomic_returns_the_components_old_values(backend):
    """One atomic per component, each returning that component's old value;
    the element as a whole is not updated atomically, so the two olds may
    come from different moments, but each is a multiple of its increment."""
    n = 512
    vf = tack.Vector.field(2, dtype=tack.f32, shape=(1,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    vector_olds(vf, out, n)
    np.testing.assert_array_equal(vf.to_numpy(), [n, 10 * n])
    assert np.all(np.abs(out.to_numpy()) % 10.0 == 0.0)


def test_barrier_is_still_a_statement_only():
    @tack.kernel
    def bad(out, n):
        for i in range(n):
            out[i] = tack.barrier()

    with pytest.raises(Exception, match="only supported as a statement"):
        bad.get_ir()
