"""Field elements updated inside sequential loops.

On Metal a loop like

    for k in range(n):
        total[0] += x[k]
        counter[0] += 1

left total at start + x[n - 1]: Apple's compiler read total[0]
once, before the loop. The generated source was right, and one thread ran
it. It happened when the element's address did not depend on the thread,
the loop also stored to another field of the same type, and the trip count
was a runtime value; it began when field pointers became members of one
argument buffer (the commit before gave the right answer in every case
here). msl_gen now compiles a loop that stores to a field in a function
of its own, which the kernel entry calls.

These run on every backend: they are plain statements about what a loop
computes.
"""

import numpy as np
import pytest

import tack
from tack.runtime.dispatch import get_backend


@tack.kernel
def k_00(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[0] += x[k]
            counter[0] += 1

@tack.kernel
def k_tt(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[t] += x[k]
            counter[t] += 1

@tack.kernel
def k_0t(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[0] += x[k]
            counter[t] += 1

@tack.kernel
def k_t0(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[t] += x[k]
            counter[0] += 1

@tack.kernel
def k_01(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[0] += x[k]
            counter[1] += 1

@tack.kernel
def k_10(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[1] += x[k]
            counter[0] += 1

@tack.kernel
def k_zz(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[z] += x[k]
            counter[z] += 1

@tack.kernel
def k_t_times_0(counter, total, x, n, z):
    for t in range(1):
        q = t * z
        for k in range(n):
            total[q] += x[k]
            counter[q] += 1

@tack.kernel
def k_literal_trip(counter, total, x, n, z):
    for t in range(1):
        for k in range(6):
            total[0] += x[k]
            counter[0] += 1

@tack.kernel
def k_counter_first(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            counter[0] += 1
            total[0] += x[k]

@tack.kernel
def k_plain_store_other(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[0] += x[k]
            counter[0] = k + 1

@tack.kernel
def k_other_varies(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[0] += x[k]
            counter[k] = 1

@tack.kernel
def k_total_varies(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[k] += x[k]
            counter[0] += 1

@tack.kernel
def k_three(counter, total, x, n, z):
    for t in range(1):
        for k in range(n):
            total[0] += x[k]
            counter[0] += 1
            counter[1] += 2

@tack.kernel
def k_many_threads(counter, total, x, n, z):
    for t in range(64):
        if t == 5:
            for k in range(n):
                total[0] += x[k]
                counter[0] += 1

CASES = [
    ("total[0], counter[0]", k_00, lambda t, c: (t[0], c[0]), (210, 6)),
    ("total[t], counter[t]   (t the thread index)", k_tt, lambda t, c: (t[0], c[0]), (210, 6)),
    ("total[0], counter[t]", k_0t, lambda t, c: (t[0], c[0]), (210, 6)),
    ("total[t], counter[0]", k_t0, lambda t, c: (t[0], c[0]), (210, 6)),
    ("total[0], counter[1]", k_01, lambda t, c: (t[0], c[1]), (210, 6)),
    ("total[1], counter[0]", k_10, lambda t, c: (t[1], c[0]), (210, 6)),
    ("total[z], counter[z]   (z a scalar argument)", k_zz, lambda t, c: (t[0], c[0]), (210, 6)),
    ("total[t * z], counter[t * z]", k_t_times_0, lambda t, c: (t[0], c[0]), (210, 6)),
    ("literal trip count, range(6)", k_literal_trip, lambda t, c: (t[0], c[0]), (210, 6)),
    ("counter updated first", k_counter_first, lambda t, c: (t[0], c[0]), (210, 6)),
    ("counter[0] = k + 1 (a plain store)", k_plain_store_other, lambda t, c: (t[0], c[0]), (210, 6)),
    ("counter[k] = 1 (other address varies)", k_other_varies, lambda t, c: (t[0], c[5]), (210, 1)),
    ("total[k] += x[k] (total address varies)", k_total_varies, lambda t, c: (t[5], c[0]), (60, 6)),
    ("three stores, two fields", k_three, lambda t, c: (t[0], c[0], c[1]), (210, 6, 12)),
    ("one thread of 64 runs the loop", k_many_threads, lambda t, c: (t[0], c[0]), (210, 6)),
]


@pytest.mark.parametrize("dtype", [tack.i32, tack.f32], ids=["i32", "f32"])
@pytest.mark.parametrize("kernel, pick, want", [case[1:] for case in CASES],
                         ids=[case[0].split("  ")[0] for case in CASES])
def test_field_updates_in_a_sequential_loop(backend, kernel, pick, want, dtype):
    np_dtype = tack.field(dtype=dtype, shape=(1,)).to_numpy().dtype
    counter = tack.field(dtype=dtype, shape=(6,))
    total = tack.field(dtype=dtype, shape=(6,))
    x = tack.field(dtype=dtype, shape=(6,))
    x.from_numpy(np.array([10, 20, 30, 40, 50, 60]).astype(np_dtype))
    kernel(counter, total, x, 6, 0)
    assert tuple(pick(total.to_numpy(), counter.to_numpy())) == want


@tack.kernel
def _drain_queue(counter, total, x, n):
    for _ in range(1):
        while counter[None] < n:
            total[None] += x[counter[None]]
            counter[None] += 1


@pytest.mark.parametrize("dtype", [tack.i32, tack.f32], ids=["i32", "f32"])
def test_while_loop_over_zero_dimensional_fields(backend, dtype):
    """The form it was found in: a queue's head and a running total."""
    counter = tack.field(dtype=dtype, shape=())
    total = tack.field(dtype=dtype, shape=())
    x = tack.field(dtype=dtype, shape=(6,))
    x.from_numpy(np.arange(6).astype(x.to_numpy().dtype))
    _drain_queue(counter, total, x, 6)
    assert counter.to_numpy() == 6 and total.to_numpy() == 15


@tack.kernel
def _block_prefix(x, out, n):
    for i in range(n):
        tid = tack.thread_id()
        smem = tack.shared(tack.f32, 256)
        smem[tid] = x[i]
        tack.barrier()
        if tid == 0:
            for k in range(1, 256):
                smem[k] += smem[k - 1]
            for k in range(256):
                out[i + k] = smem[k]


def test_loop_storing_to_workgroup_memory_and_a_field(backend):
    """A kernel whose loops store to a workgroup array and to a field takes
    the separate body function with the array declared in the entry."""
    if not get_backend().supports_workgroups:
        pytest.skip("workgroup memory needs a GPU backend")
    n = 512
    values = np.arange(n, dtype=np.float32) % 7
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(values)
    _block_prefix(x, out, n)
    np.testing.assert_array_equal(out.to_numpy(), np.cumsum(values.reshape(2, 256), axis=1).ravel())


@tack.kernel
def _no_loop(x, out, n):
    for i in range(n):
        out[i] = x[i] * 2.0


@tack.kernel
def _loop_without_a_store(x, out, n):
    for i in range(n):
        s = 0.0
        for k in range(n):
            s += x[k]
        out[i] = s


@tack.kernel
def _loop_with_a_store(x, out, n):
    for i in range(1):
        for k in range(n):
            out[k] = x[k] * 2.0


def test_only_loops_that_store_get_a_body_function(backend):
    """The body function costs a call per thread, 7-15% for a kernel that
    does one memory operation per thread, so kernels the miscompilation
    cannot reach keep the single function."""
    if get_backend().name != "metal":
        pytest.skip("Metal code generation")
    x = tack.field(dtype=tack.f32, shape=(8,))
    out = tack.field(dtype=tack.f32, shape=(8,))
    for kernel, separate in ((_no_loop, False), (_loop_without_a_store, False),
                             (_loop_with_a_store, True)):
        source = tack.inspect(kernel, x, out, 8, mode="source")
        assert ("__tack_body__(" in source) is separate, kernel.name
        if separate:
            assert "__attribute__((noinline)) static void __tack_body__(" in source
