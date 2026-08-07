"""Compiled variants must be keyed on everything baked into the code.

The IR pass pipeline substitutes dispatch-time constants into the IR — a
multi-dimensional index linearizes to ``i * dim1 + j`` with ``dim1`` as a
literal.  So the row stride is part of the compiled code's identity, and
reusing that code for a differently shaped field reads the wrong addresses
and returns wrong numbers with no error.

The other half of the contract is that the key must not be *too* fine: a
flat length is not compiled in, so varying it must not force a recompile.
"""

import numpy as np

import tack


def _variant_count(kernel):
    from tack.runtime.dispatch import get_backend
    slot = get_backend()._cache.get(kernel)
    return len(slot) if slot else 0


@tack.kernel
def _scale2d(a, out, rows, cols):
    for i in range(rows):
        for j in range(cols):
            out[i, j] = a[i, j] * 2.0


@tack.kernel
def _flat(x, out, n):
    for i in range(n):
        out[i] = x[i] + 1.0


def _run2d(rows, cols):
    a = tack.field(dtype=tack.f32, shape=(rows, cols))
    out = tack.field(dtype=tack.f32, shape=(rows, cols))
    src = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    a.from_numpy(src)
    out.fill(-1.0)
    _scale2d(a, out, rows, cols)
    return out.to_numpy(), src * 2.0


def test_row_stride_change_is_not_miscompiled(backend):
    """The bug: (4,8) then (8,4) reused the first shape's row stride."""
    got, want = _run2d(4, 8)
    np.testing.assert_array_equal(got, want)

    got, want = _run2d(8, 4)
    np.testing.assert_array_equal(got, want)


def test_alternating_shapes_stay_correct(backend):
    """Both variants stay live and keep giving the right answer."""
    for _ in range(3):
        for rows, cols in ((4, 8), (8, 4), (2, 16)):
            got, want = _run2d(rows, cols)
            np.testing.assert_array_equal(got, want)


def test_each_row_stride_gets_its_own_variant(backend):
    from tack.runtime.dispatch import get_backend
    get_backend()._cache.pop(_scale2d, None)

    _run2d(4, 8)
    assert _variant_count(_scale2d) == 1
    _run2d(8, 4)
    assert _variant_count(_scale2d) == 2
    _run2d(4, 8)   # back to the first — reuse, not a third
    assert _variant_count(_scale2d) == 2


def test_flat_length_does_not_recompile(backend):
    """Nothing about a 1-D length is baked in, so one variant covers all."""
    from tack.runtime.dispatch import get_backend
    get_backend()._cache.pop(_flat, None)

    for n in (16, 64, 256, 1024, 4099):
        x = tack.field(dtype=tack.f32, shape=(n,))
        out = tack.field(dtype=tack.f32, shape=(n,))
        x.from_numpy(np.arange(n, dtype=np.float32))
        _flat(x, out, n)
        np.testing.assert_allclose(out.to_numpy(),
                                   np.arange(n) + 1.0, rtol=1e-6)
    assert _variant_count(_flat) == 1


def test_dtype_still_separates_variants(backend):
    from tack.runtime.dispatch import get_backend
    get_backend()._cache.pop(_flat, None)

    for dtype, np_dtype in ((tack.f32, np.float32), (tack.i32, np.int32)):
        x = tack.field(dtype=dtype, shape=(32,))
        out = tack.field(dtype=dtype, shape=(32,))
        x.from_numpy(np.arange(32, dtype=np_dtype))
        _flat(x, out, 32)
        np.testing.assert_array_equal(out.to_numpy(),
                                      np.arange(32, dtype=np_dtype) + 1)
    assert _variant_count(_flat) == 2


def test_ir_passes_run_once_per_variant(backend, monkeypatch):
    """The dispatch path must not re-derive the IR it already has."""
    import tack.lang.ir_optimize as opt
    from tack.runtime.dispatch import get_backend
    get_backend()._cache.pop(_flat, None)

    calls = []
    real = opt.optimize_ir
    monkeypatch.setattr(opt, "optimize_ir",
                        lambda f: (calls.append(1), real(f))[1])

    x = tack.field(dtype=tack.f32, shape=(64,))
    out = tack.field(dtype=tack.f32, shape=(64,))
    for _ in range(8):
        _flat(x, out, 64)
    assert len(calls) == 1


def test_template_ir_is_not_mutated_by_dispatch(backend):
    """Resolution runs on a copy, so later shapes can still be resolved."""
    from tack.lang import ir
    from tack.runtime.kernel_utils import _walk_ir

    _run2d(4, 8)
    template = _scale2d.get_ir().functions[0]
    dims = [n for n in _walk_ir(template.body) if isinstance(n, ir.IRDimSize)]
    assert dims, "a dispatch consumed the template's IRDimSize nodes"
