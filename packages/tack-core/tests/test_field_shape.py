"""`field.shape[k]` and `len(field)` anywhere in a kernel.

Both are documented as general kernel-language features, but they used to
resolve only as the outermost parallel loop bound — dispatch evaluated
that one expression numerically and every other position reached codegen
as an unresolved attribute, tripping an internal assertion.  Reversing an
array did not work.

They now lower to IRDimSize, which the resolve pass folds wherever it
appears.  The exception is the grid bound itself: it is passed to the
launch rather than compiled in, so it stays a dispatch-time value and
`for i in range(x.shape[0])` still compiles once for every length.
"""

import numpy as np
import pytest

import tack

_backends = []
for _arch in ["cpu", "metal", "cuda", "hip", "level_zero"]:
    try:
        tack.init(arch=getattr(tack, _arch))
        _backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=_backends)
def backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


N = 8


def _run(kernel, n=N):
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    src = np.arange(n, dtype=np.float32)
    x.from_numpy(src)
    out.fill(-1.0)
    kernel(x, out)
    return src, out.to_numpy()


# ── .shape in each position ──────────────────────────────────────────

@tack.kernel
def _bound(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i]


@tack.kernel
def _arith(x, out):
    for i in range(x.shape[0]):
        out[i] = float(x.shape[0])


@tack.kernel
def _condition(x, out):
    for i in range(x.shape[0]):
        if i < x.shape[0] - 1:
            out[i] = x[i + 1]
        else:
            out[i] = 0.0


@tack.kernel
def _reverse(x, out):
    for i in range(x.shape[0]):
        out[i] = x[x.shape[0] - 1 - i]


@tack.kernel
def _inner_loop(x, out):
    for i in range(x.shape[0]):
        acc = 0.0
        for j in range(x.shape[0]):
            acc = acc + x[j]
        out[i] = acc


@tack.kernel
def _while_bound(x, out):
    for i in range(x.shape[0]):
        j = 0
        acc = 0.0
        while j < x.shape[0]:
            acc = acc + x[j]
            j = j + 1
        out[i] = acc


def test_shape_as_loop_bound(backend):
    src, got = _run(_bound)
    np.testing.assert_array_equal(got, src)


def test_shape_in_arithmetic(backend):
    _, got = _run(_arith)
    np.testing.assert_array_equal(got, np.full(N, float(N)))


def test_shape_in_a_condition(backend):
    src, got = _run(_condition)
    np.testing.assert_array_equal(got, np.append(src[1:], 0.0))


def test_shape_in_an_index(backend):
    """Reversing an array — about as basic as kernels get."""
    src, got = _run(_reverse)
    np.testing.assert_array_equal(got, src[::-1])


def test_shape_as_inner_loop_bound(backend):
    src, got = _run(_inner_loop)
    np.testing.assert_allclose(got, np.full(N, src.sum()), rtol=1e-6)


def test_shape_in_a_while_condition(backend):
    src, got = _run(_while_bound)
    np.testing.assert_allclose(got, np.full(N, src.sum()), rtol=1e-6)


# ── len() in each position ───────────────────────────────────────────

@tack.kernel
def _len_bound(x, out):
    for i in range(len(x)):
        out[i] = x[i]


@tack.kernel
def _len_arith(x, out):
    for i in range(len(x)):
        out[i] = x[i] / float(len(x))


@tack.kernel
def _len_reverse(x, out):
    for i in range(len(x)):
        out[i] = x[len(x) - 1 - i]


@tack.kernel
def _len_inner(x, out):
    for i in range(len(x)):
        acc = 0.0
        for j in range(len(x)):
            acc = acc + x[j]
        out[i] = acc


def test_len_as_loop_bound(backend):
    src, got = _run(_len_bound)
    np.testing.assert_array_equal(got, src)


def test_len_in_arithmetic(backend):
    src, got = _run(_len_arith)
    np.testing.assert_allclose(got, src / N, rtol=1e-6)


def test_len_in_an_index(backend):
    src, got = _run(_len_reverse)
    np.testing.assert_array_equal(got, src[::-1])


def test_len_as_inner_loop_bound(backend):
    src, got = _run(_len_inner)
    np.testing.assert_allclose(got, np.full(N, src.sum()), rtol=1e-6)


# ── Multi-dimensional ────────────────────────────────────────────────

@tack.kernel
def _rows_and_cols(a, out):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = a[i, j] + float(a.shape[1])


@tack.kernel
def _ndrange_shape(a, out):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        out[i, j] = a[i, j] * 2.0


@tack.kernel
def _transpose_index(a, out):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        out[i, j] = a[a.shape[0] - 1 - i, a.shape[1] - 1 - j]


def _run2d(kernel, rows, cols):
    a = tack.field(dtype=tack.f32, shape=(rows, cols))
    out = tack.field(dtype=tack.f32, shape=(rows, cols))
    src = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    a.from_numpy(src)
    out.fill(-1.0)
    kernel(a, out)
    return src, out.to_numpy()


@pytest.mark.parametrize("rows,cols", [(4, 8), (8, 4), (3, 5)])
def test_shape_of_each_dimension(backend, rows, cols):
    src, got = _run2d(_rows_and_cols, rows, cols)
    np.testing.assert_array_equal(got, src + cols)


@pytest.mark.parametrize("rows,cols", [(4, 8), (8, 4), (3, 5)])
def test_ndrange_over_shape(backend, rows, cols):
    """The audit's original repro: ndrange(a.shape[0], a.shape[1])."""
    src, got = _run2d(_ndrange_shape, rows, cols)
    np.testing.assert_array_equal(got, src * 2.0)


@pytest.mark.parametrize("rows,cols", [(4, 8), (8, 4)])
def test_shape_in_a_two_dimensional_index(backend, rows, cols):
    src, got = _run2d(_transpose_index, rows, cols)
    np.testing.assert_array_equal(got, src[::-1, ::-1])


# ── Specialization behaviour ─────────────────────────────────────────

def _variants(kernel):
    from tack.runtime.dispatch import get_backend
    slot = get_backend()._cache.get(kernel)
    return len(slot) if slot else 0


def test_shape_in_the_bound_does_not_specialize_on_length(backend):
    """`for i in range(x.shape[0])` is the most common line in any kernel.

    The bound is passed to the launch rather than compiled in, so it must
    not force a recompile for every array size.
    """
    from tack.runtime.dispatch import get_backend
    get_backend()._cache.pop(_bound, None)

    for n in (8, 16, 64, 257, 1000):
        src, got = _run(_bound, n)
        np.testing.assert_array_equal(got, src)
    assert _variants(_bound) == 1


def test_shape_in_the_body_stays_correct_across_lengths(backend):
    """The length is compiled into the index arithmetic here, so this
    specializes — but every specialization must be right."""
    from tack.runtime.dispatch import get_backend
    get_backend()._cache.pop(_reverse, None)

    for n in (8, 1000, 8, 64, 1000):
        src, got = _run(_reverse, n)
        np.testing.assert_array_equal(got, src[::-1])


def test_hoisting_the_length_into_a_scalar_avoids_specializing(backend):
    """The escape hatch when per-size compilation is not wanted."""

    @tack.kernel
    def reverse_n(x, out, n):
        for i in range(n):
            out[i] = x[n - 1 - i]

    for n in (8, 16, 64, 257):
        x = tack.field(dtype=tack.f32, shape=(n,))
        out = tack.field(dtype=tack.f32, shape=(n,))
        src = np.arange(n, dtype=np.float32)
        x.from_numpy(src)
        reverse_n(x, out, n)
        np.testing.assert_array_equal(out.to_numpy(), src[::-1])
    assert _variants(reverse_n) == 1


# ── Through inlining and templates ───────────────────────────────────

@tack.func
def _last_of(f):
    return f[f.shape[0] - 1]


@tack.func
def _mean_of(f):
    total = 0.0
    for j in range(len(f)):
        total = total + f[j]
    return total / float(len(f))


@tack.kernel
def _calls_func(x, out):
    for i in range(x.shape[0]):
        out[i] = _last_of(x)


@tack.kernel
def _calls_func_len(x, out):
    for i in range(len(x)):
        out[i] = _mean_of(x)


def test_shape_inside_an_inlined_func(backend):
    """Inlining renames the field, so the dimension query must follow it."""
    src, got = _run(_calls_func)
    np.testing.assert_array_equal(got, np.full(N, src[-1]))


def test_len_inside_an_inlined_func(backend):
    src, got = _run(_calls_func_len)
    np.testing.assert_allclose(got, np.full(N, src.mean()), rtol=1e-6)


@tack.data_oriented
class _Holder:
    def __init__(self, data):
        self.data = data

    @tack.func
    def scaled(self, i):
        return self.data[i] / float(self.data.shape[0])


@tack.kernel
def _calls_template(h: tack.template(), out):
    for i in range(out.shape[0]):
        out[i] = h.scaled(i)


def test_shape_of_a_template_field(backend):
    """Template rewriting also renames the field."""
    data = tack.field(dtype=tack.f32, shape=(N,))
    out = tack.field(dtype=tack.f32, shape=(N,))
    src = np.arange(N, dtype=np.float32)
    data.from_numpy(src)
    out.fill(-1.0)

    _calls_template(_Holder(data), out)
    np.testing.assert_allclose(out.to_numpy(), src / N, rtol=1e-6)


# ── Diagnostics ──────────────────────────────────────────────────────

def test_non_constant_dimension_index_is_rejected(backend):
    """The dimension has to be known when the kernel is compiled."""
    with pytest.raises((NotImplementedError, RuntimeError), match="constant dimension"):
        @tack.kernel
        def dynamic_dim(x, out, d):
            for i in range(x.shape[d]):
                out[i] = x[i]

        x = tack.field(dtype=tack.f32, shape=(N,))
        out = tack.field(dtype=tack.f32, shape=(N,))
        dynamic_dim(x, out, 0)
