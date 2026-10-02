"""Behavioral baseline for docs/reference/language-contract.md.

LC1–LC3 are ordinary regressions on every available backend. The remaining
shared-frontend defect LC4 is a strict expected failure until stage three.

Run with --runxfail to expose the current defects as ordinary failures.
"""

import numpy as np
import pytest

import tack


def _int_field(values):
    values = np.asarray(values, dtype=np.int32)
    field = tack.field(dtype=tack.i32, shape=values.shape)
    field.from_numpy(values)
    return field


@tack.kernel
def _increment(state, iterations):
    for i in range(state.shape[0]):
        state[i] = 0
        for j in range(iterations):
            value = state[i]
            state[i] = value + 1


@pytest.mark.parametrize("iterations", [0, 1])
def test_sequential_field_mutation_control(backend, iterations):
    """Empty and single-iteration loops establish the initial-state behavior."""
    state = _int_field([19] * 7)
    _increment(state, iterations)
    np.testing.assert_array_equal(state.to_numpy(), np.full(7, iterations))


def test_sequential_field_read_after_write(backend):
    """An unchanged address does not imply an unchanged value (LC1)."""
    state = _int_field([19] * 7)
    _increment(state, 3)
    np.testing.assert_array_equal(state.to_numpy(), np.full(7, 3))


def test_sequential_field_mutation_without_tack_optimizations(backend, monkeypatch):
    """The same program reaches the reference result without Tack's passes.

    The backend fixture creates a fresh backend, so the dispatch cannot
    reuse the optimized compiled variant from another test. Vendor/LLVM
    optimization remains enabled; this isolates Tack's IR passes only.
    """
    import tack.lang.ir_optimize as opt

    monkeypatch.setattr(opt, "optimize_ir", lambda ir_func: None)
    state = _int_field([19] * 7)
    _increment(state, 3)
    np.testing.assert_array_equal(state.to_numpy(), np.full(7, 3))


@tack.kernel
def _ordered_writes(a, b, out):
    for i in range(out.shape[0]):
        a[i] = 1
        b[i] = 2
        out[i] = a[i]


def test_distinct_field_arguments_control(backend):
    a = _int_field([19] * 7)
    b = _int_field([23] * 7)
    out = _int_field([-1] * 7)
    _ordered_writes(a, b, out)
    np.testing.assert_array_equal(a.to_numpy(), np.full(7, 1))
    np.testing.assert_array_equal(b.to_numpy(), np.full(7, 2))
    np.testing.assert_array_equal(out.to_numpy(), np.full(7, 1))


@pytest.mark.parametrize("alias", ["same_field", "reshape_view"])
def test_overlapping_field_arguments_preserve_program_order(backend, alias):
    """LC2: overlapping arguments preserve each iteration's program order.

    Each parallel iteration owns one element; aliasing here creates no
    inter-iteration race. A distinct Field wrapper must obey the same rule.
    """
    a = _int_field([19] * 7)
    b = a if alias == "same_field" else a.reshape((7,))
    out = _int_field([-1] * 7)
    _ordered_writes(a, b, out)
    np.testing.assert_array_equal(out.to_numpy(), np.full(7, 2))


@tack.kernel
def _squared_norms(vectors, out):
    for i in range(out.shape[0]):
        out[i] = vectors[i].norm_sqr()


@pytest.mark.parametrize("widths", [(2, 3, 2), (3, 2, 3)], ids=["2-3-2", "3-2-3"])
def test_vector_width_changes_preserve_results(backend, widths):
    """LC3: specialize vector lowering in both directions, then revisit it."""
    out = tack.field(dtype=tack.f32, shape=(2,))
    actual = []
    expected = []
    for width in widths:
        # Three logical vectors provide enough storage even when the
        # broken cache uses width=3 to read a width=2 input. Only the first
        # two vectors are requested: no out-of-bounds probe on any backend.
        vectors = tack.Vector.field(width, dtype=tack.f32, shape=(3,))
        values = np.arange(1, 3 * width + 1, dtype=np.float32).reshape(3, width)
        vectors.from_numpy(values.ravel())
        _squared_norms(vectors, out)
        actual.append(out.to_numpy())
        expected.append(np.sum(values[:2] ** 2, axis=1))
    np.testing.assert_array_equal(np.stack(actual), np.stack(expected))


@pytest.mark.parametrize("iterations", [0, 3])
def test_zero_trip_loop_preserves_prior_local_value(backend, iterations):
    @tack.kernel
    def local_value(out, count):
        for i in range(out.shape[0]):
            value = 19
            for j in range(count):
                value = 7
            out[i] = value

    out = _int_field([-1] * 7)
    local_value(out, iterations)
    np.testing.assert_array_equal(out.to_numpy(), np.full(7, 19 if iterations == 0 else 7))


def test_while_loop_observes_field_mutation(backend):
    @tack.kernel
    def increment_while(state):
        for i in range(state.shape[0]):
            state[i] = 0
            j = 0
            while j < 3:
                value = state[i]
                state[i] = value + 1
                j = j + 1

    state = _int_field([19] * 7)
    increment_while(state)
    np.testing.assert_array_equal(state.to_numpy(), np.full(7, 3))


def test_loop_load_observes_store_through_an_alias(backend):
    @tack.kernel
    def increment_alias(a, b):
        for i in range(a.shape[0]):
            a[i] = 0
            for j in range(3):
                value = b[i]
                a[i] = value + 1

    state = _int_field([19] * 7)
    increment_alias(state, state.reshape((7,)))
    np.testing.assert_array_equal(state.to_numpy(), np.full(7, 3))


def test_load_after_alias_store_is_not_a_common_subexpression(backend):
    @tack.kernel
    def read_write_read(a, b, out):
        for i in range(out.shape[0]):
            before = a[i]
            b[i] = before + 2
            after = a[i]
            out[i] = after - before

    state = _int_field([19] * 7)
    out = _int_field([-1] * 7)
    read_write_read(state, state.reshape((7,)), out)
    np.testing.assert_array_equal(out.to_numpy(), np.full(7, 2))


def test_copy_propagation_preserves_reads_before_assignment(backend):
    @tack.kernel
    def late_copy(out, value, replacement):
        for i in range(out.shape[0]):
            before = value
            value = replacement
            out[i] = before + value

    out = _int_field([-1])
    late_copy(out, 3, 7)
    np.testing.assert_array_equal(out.to_numpy(), [10])


def test_copy_propagation_preserves_loop_steps(backend):
    @tack.kernel
    def stepped_sum(data, out):
        for i in range(out.shape[0]):
            alias = data
            total = 0
            for j in range(0, 6, 2):
                total = total + alias[j]
            out[i] = total

    data = _int_field([1, 2, 3, 4, 5, 6])
    out = _int_field([-1])
    stepped_sum(data, out)
    np.testing.assert_array_equal(out.to_numpy(), [9])


def test_copy_propagation_counts_loop_variable_bindings(backend):
    @tack.kernel
    def preserve_scalar(out, cursor):
        for i in range(out.shape[0]):
            saved = cursor
            total = 0
            for cursor in range(3):
                total = total + saved
            out[i] = total

    out = _int_field([-1])
    preserve_scalar(out, 7)
    np.testing.assert_array_equal(out.to_numpy(), [21])


@tack.kernel
def _mark_range(out, start, end):
    for i in range(start, end):
        out[i] = i + 1


@pytest.mark.parametrize("start, end", [(0, 7), (3, 7), (6, 7), (4, 4), (5, 2)])
def test_parallel_range_honors_its_start(backend, start, end):
    """LC5: the outermost range covers [start, end), or nothing when empty."""
    out = _int_field([-1] * 7)
    _mark_range(out, start, end)
    expected = np.full(7, -1)
    expected[start:end] = np.arange(start, end) + 1
    np.testing.assert_array_equal(out.to_numpy(), expected)


def test_parallel_range_start_with_a_dimension_bound(backend):
    """A stencil's interior range must not touch, or read past, the edges."""
    @tack.kernel
    def neighbor_sum(x, out):
        for i in range(1, x.shape[0] - 1):
            out[i] = x[i - 1] + x[i + 1]

    for n in (5, 9):
        values = np.arange(1, n + 1)
        out = _int_field([-1] * n)
        neighbor_sum(_int_field(values), out)
        expected = np.full(n, -1)
        expected[1:-1] = values[:-2] + values[2:]
        np.testing.assert_array_equal(out.to_numpy(), expected)


def test_empty_parallel_range_runs_nothing(backend):
    @tack.kernel
    def fill(out, n):
        for i in range(n):
            out[i] = 7

    out = _int_field([-1] * 7)
    fill(out, 0)
    np.testing.assert_array_equal(out.to_numpy(), np.full(7, -1))


@pytest.mark.xfail(
    strict=True, raises=pytest.fail.Exception,
    reason="LC4: the shared frontend silently discards unsupported assert statements",
)
def test_unsupported_assert_is_rejected():
    """Reject assert at the frontend until its execution semantics are defined."""
    @tack.kernel
    def unsupported_assert(out):
        for i in range(out.shape[0]):
            assert False  # noqa: B011 — deliberately unsupported kernel source
            out[i] = 7

    with pytest.raises((NotImplementedError, SyntaxError, TypeError, RuntimeError),
                       match="(?i)assert"):
        unsupported_assert.get_ir()
