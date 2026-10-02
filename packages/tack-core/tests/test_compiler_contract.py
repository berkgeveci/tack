"""Behavioral baseline for docs/reference/language-contract.md.

Known CPU defects are strict expected failures, limited to numerical
assertions. A compilation failure is still a failure; a fix is an XPASS
that requires removing the marker. Other backends run the numerical
checks without expected-failure markers until measured on hardware.

Run with --runxfail to expose the current defects as ordinary failures.
"""

import numpy as np
import pytest

import tack


def _known_cpu_failure(request, backend, issue):
    if backend == "cpu":
        request.applymarker(pytest.mark.xfail(
            strict=True, raises=AssertionError, reason=issue,
        ))


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


def test_sequential_field_read_after_write(backend, request):
    """An unchanged address does not imply an unchanged value (LC1)."""
    _known_cpu_failure(request, backend, "LC1: LICM hoists a mutable field load")
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
def test_overlapping_field_arguments_preserve_program_order(backend, request, alias):
    """LC2's proposed alias-support policy, not yet a released guarantee.

    Each parallel iteration owns one element; aliasing here creates no
    inter-iteration race. A distinct Field wrapper must obey the same rule.
    """
    _known_cpu_failure(request, backend, "LC2: unconditional noalias on field arguments")
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
def test_vector_width_changes_preserve_results(backend, request, widths):
    """LC3: specialize vector lowering in both directions, then revisit it."""
    _known_cpu_failure(request, backend, "LC3: compiled variant key omits vector width")
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
