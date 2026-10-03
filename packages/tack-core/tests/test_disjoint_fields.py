"""The CPU backend's proven-disjoint specialization.

Fields may share storage, so the generated code never promises otherwise
unconditionally.  The CPU backend checks each call's field ranges and uses
a `noalias` variant only when no written field overlaps another field.
These tests cover the analysis, the range check, the generated signature,
and that switching between overlapping and disjoint calls stays correct.
"""

import numpy as np
import pytest

import tack
from tack.runtime import cpu
from tack.runtime.kernel_utils import fields_disjoint, written_field_params


@pytest.fixture(autouse=True)
def init_cpu():
    tack.init(arch=tack.cpu)


@tack.func
def _bump(dst, i, v):
    dst[i] = dst[i] + v


@tack.kernel
def _read_only(x, y, out):
    for i in range(out.shape[0]):
        out[i] = x[i] + y[i]


@tack.kernel
def _through_func(x, out):
    for i in range(out.shape[0]):
        _bump(out, i, x[i])


@tack.kernel
def _atomic(x, hist):
    for i in range(x.shape[0]):
        tack.atomic_add(hist, 0, x[i])


@tack.kernel
def _scratch(x, out, n):
    for i in range(n):
        tmp = tack.local_array(tack.f32, 2)
        tmp[0] = x[i]
        out[i] = tmp[0]


@tack.kernel
def _accumulate(x, w, out, n):
    for i in range(n):
        out[i] = 0.0
        for j in range(4):
            out[i] = out[i] + x[i] * w[j]


@tack.kernel
def _ordered(a, b, out):
    for i in range(out.shape[0]):
        a[i] = 1
        b[i] = 2
        out[i] = a[i]


def _template(kernel):
    return kernel.get_ir({}, template_args=None, texture_fields={}).functions[0]


def _f32(n=8, value=0.0):
    f = tack.field(dtype=tack.f32, shape=(n,))
    f.fill(value)
    return f


def _signature(kernel, *args):
    source = tack.inspect(kernel, *args, mode="source")
    return next(line for line in source.splitlines()
                if line.startswith("define"))


# ── Which parameters a kernel writes ────────────────────────────────

@pytest.mark.parametrize("kernel, written", [
    (_read_only, {"out"}),
    (_through_func, {"out"}),
    (_atomic, {"hist"}),
    (_scratch, {"out"}),
    (_ordered, {"a", "b", "out"}),
])
def test_written_parameters(kernel, written):
    assert written_field_params(_template(kernel)) == frozenset(written)


def test_untraceable_store_marks_everything_written():
    from tack.lang import ir
    func = ir.IRFunction("k", [ir.IRParam("x"), ir.IRParam("out")], [
        ir.IRFieldStore(ir.IRName("mystery"), ir.IRConstant(0),
                        ir.IRConstant(1))])
    assert written_field_params(func) is None
    x, out = _f32(), _f32()
    assert fields_disjoint(func, (x, out))
    assert not fields_disjoint(func, (x, x))


def test_a_name_bound_to_two_fields_writes_both():
    from tack.lang import ir
    func = ir.IRFunction("k", [ir.IRParam("a"), ir.IRParam("b")], [
        ir.IRAssign("v", ir.IRName("a")),
        ir.IRAssign("v", ir.IRName("b")),
        ir.IRAssign("u", ir.IRName("v")),
        ir.IRFieldStore(ir.IRName("u"), ir.IRConstant(0), ir.IRConstant(1))])
    assert written_field_params(func) == frozenset({"a", "b"})


# ── The per-call range check ────────────────────────────────────────

def test_distinct_fields_are_disjoint():
    assert fields_disjoint(_template(_read_only), (_f32(), _f32(), _f32()))


def test_read_only_fields_may_overlap():
    x = _f32()
    tmpl = _template(_read_only)
    assert fields_disjoint(tmpl, (x, x, _f32()))
    assert fields_disjoint(tmpl, (x, x.reshape((2, 4)), _f32()))


def test_written_field_overlapping_a_read_is_not_disjoint():
    x = _f32()
    tmpl = _template(_read_only)
    assert not fields_disjoint(tmpl, (x, _f32(), x))
    assert not fields_disjoint(tmpl, (_f32(), x, x.reshape((8,))))


def test_partial_overlap_of_imported_storage():
    base = np.zeros(16, dtype=np.float32)
    tmpl = _template(_read_only)

    def view(lo, hi):
        return tack.field_from_ptr(base[lo:hi], tack.f32, (hi - lo,),
                                   writable=True)

    x = _f32()
    # Adjacent halves of one allocation touch but do not overlap.
    assert fields_disjoint(tmpl, (view(0, 8), x, view(8, 16)))
    assert not fields_disjoint(tmpl, (view(0, 9), x, view(8, 16)))
    assert not fields_disjoint(tmpl, (view(4, 12), x, view(0, 16)))
    # One byte range strictly inside another.
    assert not fields_disjoint(tmpl, (view(0, 16), x, view(6, 7)))


def test_scalars_are_ignored():
    x, w, out = _f32(), _f32(4), _f32()
    assert fields_disjoint(_template(_accumulate), (x, w, out, 8))
    assert not fields_disjoint(_template(_accumulate), (out, w, out, 8))


def test_span_follows_the_array():
    f = _f32()
    start, end = f._buffer.span
    assert end - start == 8 * 4
    assert f._buffer.span is f._buffer.span
    f._buffer._data = np.zeros(4, dtype=np.float32)
    start, end = f._buffer.span
    assert end - start == 4 * 4


# ── Generated code ──────────────────────────────────────────────────

def test_disjoint_call_promises_noalias_on_fields_only():
    sig = _signature(_accumulate, _f32(), _f32(4), _f32(), 8)
    assert sig.count("noalias") == 3
    assert 'noalias %"n"' not in sig


def test_overlapping_call_makes_no_promise():
    out = _f32()
    assert "noalias" not in _signature(_accumulate, out, _f32(4), out, 8)


def test_specialization_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(cpu, "_SPECIALIZE_DISJOINT", False)
    assert "noalias" not in _signature(_accumulate, _f32(), _f32(4), _f32(), 8)


def test_codegen_makes_no_promise_by_default():
    """Direct codegen calls, with no dispatcher to vouch for the storage."""
    import copy

    from tack.codegen.llvm_gen import generate_llvm_ir
    from tack.lang.ir_resolve import resolve_ir
    from tack.lang.ir_type_annotate import annotate_types
    from tack.lang.type_inference import infer_param_types

    args = (_f32(), _f32(), _f32())
    func = copy.deepcopy(_template(_read_only))
    resolve_ir(func, dict(zip(("x", "y", "out"), args)))
    infer_param_types(func, args)
    annotate_types(func)
    assert "noalias" not in str(generate_llvm_ir(func))


# ── Execution ───────────────────────────────────────────────────────

@pytest.mark.parametrize("order", [(False, True, False, True),
                                   (True, False, True, False)])
def test_switching_between_disjoint_and_aliased_calls(order):
    from tack.runtime.dispatch import get_backend

    get_backend()._cache.pop(_ordered, None)
    for alias in order:
        a = tack.field(dtype=tack.i32, shape=(9,))
        b = a.reshape((9,)) if alias else tack.field(dtype=tack.i32, shape=(9,))
        out = tack.field(dtype=tack.i32, shape=(9,))
        _ordered(a, b, out)
        np.testing.assert_array_equal(out.to_numpy(),
                                      np.full(9, 2 if alias else 1))
    variants = get_backend()._cache[_ordered]
    assert sorted(key[-1] for key in variants) == [False, True]


def test_accumulating_into_its_own_input():
    """`out` is also `x`: each element reads itself, then accumulates."""
    w = tack.field(dtype=tack.f32, shape=(4,))
    w.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    xs = np.arange(1, 9, dtype=np.float32)

    x, out = _f32(), _f32()
    x.from_numpy(xs)
    _accumulate(x, w, out, 8)
    np.testing.assert_array_equal(out.to_numpy(), xs * 10)

    # Aliased, the first store zeroes x[i] before it is read.
    x.from_numpy(xs)
    _accumulate(x, w, x, 8)
    np.testing.assert_array_equal(x.to_numpy(), np.zeros(8, dtype=np.float32))

    # And the disjoint variant is still right afterwards.
    x.from_numpy(xs)
    _accumulate(x, w, out, 8)
    np.testing.assert_array_equal(out.to_numpy(), xs * 10)


def test_overlapping_imported_views_run_without_the_promise():
    """A three-point stencil written back into shifted storage."""
    base = np.arange(10, dtype=np.float32)
    src = tack.field_from_ptr(base[1:9], tack.f32, (8,), writable=True)
    dst = tack.field_from_ptr(base[0:8], tack.f32, (8,), writable=True)
    y = _f32()
    _read_only(src, y, dst)
    np.testing.assert_array_equal(base, [1, 2, 3, 4, 5, 6, 7, 8, 8, 9])


def test_disjoint_variant_does_not_respecialize_on_length():
    from tack.runtime.dispatch import get_backend

    get_backend()._cache.pop(_accumulate, None)
    for n in (4, 16, 64):
        _accumulate(_f32(n), _f32(4), _f32(n), n)
    assert len(get_backend()._cache[_accumulate]) == 1


def test_span_is_the_array_byte_range():
    """``span`` must work on whichever NumPy is installed: NumPy 2 keeps
    ``byte_bounds`` in ``numpy.lib.array_utils``, NumPy 1 at top level, and
    the dependency is declared as plain ``numpy``. Every CPU dispatch reads
    it, so a missing import would fail every kernel call."""
    x = tack.field(dtype=tack.f32, shape=(1024,))
    start, end = x._buffer.span
    data = x._buffer._data
    assert start == data.ctypes.data
    assert end == data.ctypes.data + data.nbytes
    view = x._buffer._data[256:512]
    assert np.shares_memory(view, data)
