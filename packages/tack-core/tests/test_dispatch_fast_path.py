"""The dispatch fast path: a call whose signature was seen before reuses the
variant that call resolved, without rebuilding the key. What the signature
must still separate (integer widths, the float context, class constants,
template structure), what a hit must still check (read-only arrays,
disjointness), and what it must not hold on to."""
import gc
import weakref

import numpy as np
import pytest

import tack
from tack.runtime import kernel_utils as ku
from tack.runtime.dispatch import get_backend


def _variants(kernel):
    slot = get_backend()._cache.get(kernel)
    return len(slot) if slot else 0


def _fast_entries(kernel):
    per_backend = ku._fast_caches.get(get_backend())
    slot = per_backend.get(kernel) if per_backend else None
    return len(slot) if slot else 0


@tack.kernel
def _plus(x, out, n):
    for i in range(x.shape[0]):
        out[i] = x[i] + n


@tack.kernel
def _scaled(x, out, a: tack.f64):
    for i in range(x.shape[0]):
        out[i] = x[i] * a


@tack.device_class
class _Gain:
    K = 2

    def __init__(self, bias):
        self.bias = bias

    @tack.func
    def apply(self, v):
        return v * self.K + self.bias


@tack.device_class
class _Inner:
    def __init__(self, values, offset):
        self.values = values
        self.offset = offset


@tack.device_class
class _Outer:
    def __init__(self, inner, scale):
        self.inner = inner
        self.scale = scale


@tack.kernel
def _through_gain(x, out, g):
    for i in range(out.shape[0]):
        out[i] = g.apply(x[i])


@tack.kernel
def _through_tree(out, o):
    for i in range(out.shape[0]):
        out[i] = (o.inner.values[i] + o.inner.offset) * o.scale


@pytest.mark.usefixtures("no_debug")
def test_a_repeated_call_skips_the_key(backend, monkeypatch):
    calls = []
    real = ku.kernel_variant_key
    monkeypatch.setattr(ku, "kernel_variant_key", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    x = tack.empty((16,)); x.fill(1.0)
    out = tack.zeros((16,))
    _plus(x, out, 1)
    _plus(x, out, 2)
    _plus(x, out, 3)
    assert len(calls) == 1, "the key was rebuilt on a repeated call"
    np.testing.assert_array_equal(out.numpy(), np.full(16, 4.0, np.float32))


def test_an_integer_past_i32_gets_its_own_variant(backend):
    x = tack.zeros((8,), tack.i64)
    out = tack.zeros((8,), tack.i64)
    before = _variants(_plus)
    _plus(x, out, 5)
    np.testing.assert_array_equal(out.numpy(), 5)
    _plus(x, out, 1 << 40)
    np.testing.assert_array_equal(out.numpy(), 1 << 40)
    assert _variants(_plus) == before + 2
    _plus(x, out, 6)
    np.testing.assert_array_equal(out.numpy(), 6)
    assert _variants(_plus) == before + 2


def test_the_float_context_follows_the_arrays(f64_backend):
    x32 = tack.empty((4,)); x32.fill(1.0)
    o32 = tack.zeros((4,))
    x64 = tack.empty((4,), tack.f64); x64.fill(1.0)
    o64 = tack.zeros((4,), tack.f64)
    before = _variants(_scaled)
    _scaled(x32, o32, 0.1)
    _scaled(x64, o64, 0.1)
    _scaled(x32, o32, 0.1)
    assert _variants(_scaled) == before + 2
    assert o64.numpy()[0] == 0.1                       # f64 scalar, exact
    assert o32.numpy()[0] == np.float32(0.1)


def test_a_changed_class_constant_is_seen(backend):
    x = tack.empty((4,)); x.fill(3.0)
    out = tack.zeros((4,))
    saved = _Gain.K
    try:
        _through_gain(x, out, _Gain(1.0))
        np.testing.assert_array_equal(out.numpy(), 7.0)
        _Gain.K = 5
        _through_gain(x, out, _Gain(1.0))
        np.testing.assert_array_equal(out.numpy(), 16.0)
        _through_gain(x, out, _Gain(2.0))                  # a runtime scalar: no new variant
        np.testing.assert_array_equal(out.numpy(), 17.0)
    finally:
        _Gain.K = saved


def test_nested_templates_expand_through_the_plan(backend):
    """Fresh objects every call, as the dataset filters make: a hit must read
    the new objects' arrays and scalars, not the first call's."""
    out = tack.zeros((4,))
    a = tack.empty((4,)); a.fill(1.0)
    b = tack.empty((4,)); b.fill(10.0)
    _through_tree(out, _Outer(_Inner(a, 1.0), 2.0))
    np.testing.assert_array_equal(out.numpy(), 4.0)
    before = _variants(_through_tree)
    _through_tree(out, _Outer(_Inner(b, 5.0), 3.0))
    np.testing.assert_array_equal(out.numpy(), 45.0)
    assert _variants(_through_tree) == before


def test_a_dropped_variant_is_resolved_afresh(backend):
    x = tack.empty((4,)); x.fill(1.0)
    out = tack.zeros((4,))
    _plus(x, out, 1)
    ku.drop_variants(_plus, lambda key: True)
    assert _variants(_plus) == 0
    _plus(x, out, 2)
    np.testing.assert_array_equal(out.numpy(), 3.0)
    assert _variants(_plus) == 1


def _per_call_gain(k):
    @tack.device_class
    class Gain:
        K = k

        def __init__(self):
            self.bias = 0.0

        @tack.func
        def apply(self, v):
            return v * self.K + self.bias
    return Gain


def test_the_entries_hold_no_class(backend):
    """A template class defined per call is collected as it was before."""
    x = tack.empty((4,)); x.fill(1.0)
    out = tack.zeros((4,))
    refs = []
    for k in range(3):
        cls = _per_call_gain(k + 1)
        refs.append(weakref.ref(cls))
        _through_gain(x, out, cls())
        np.testing.assert_array_equal(out.numpy(), float(k + 1))
        del cls
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_the_map_is_bounded(backend):
    out = tack.zeros((4,))
    for n in range(ku._FAST_CAP + 20):
        x = tack.empty((n + 1,))                 # every length is a signature
        _plus(x, tack.zeros((n + 1,)), 1)
    assert _fast_entries(_plus) <= ku._FAST_CAP
    assert _variants(_plus) <= 2                           # the length is not compiled in


def test_a_read_only_array_is_refused_on_a_hit():
    tack.set_default_device("cpu")
    base = np.zeros(8, np.float32)
    x = tack.empty((8,)); x.fill(1.0)
    out = tack.array_from_ptr(base, (8,), tack.f32, writable=True)
    _plus(x, out, 1)
    frozen = tack.array_from_ptr(base, (8,), tack.f32, writable=False)
    with pytest.raises(ValueError, match="read-only"):
        _plus(x, frozen, 1)


@tack.kernel
def _shift(x, out):
    for i in range(out.shape[0]):
        out[i] = x[i] + 1.0


def test_overlapping_and_disjoint_calls_alternate_on_a_hit():
    """The CPU compiles a no-overlap variant and an overlapping one; a hit
    must pick by this call's storage, not the first call's."""
    tack.set_default_device("cpu")
    base = np.arange(10, dtype=np.float32)
    whole = tack.array_from_ptr(base, (10,), tack.f32, writable=True)
    head = tack.array_from_ptr(base[:9], (9,), tack.f32, writable=True)
    tail = tack.array_from_ptr(base[1:], (9,), tack.f32, writable=True)
    apart = tack.empty((9,)); apart.fill(0.0)
    sink = tack.zeros((9,))
    for _ in range(3):
        _shift(apart, sink)                                # disjoint
        np.testing.assert_array_equal(sink.numpy(), 1.0)
        base[:] = np.arange(10, dtype=np.float32)
        _shift(head, tail)                                 # overlapping: reads must see stores in order
        np.testing.assert_array_equal(base, [0] + [i + 1 for i in range(9)])
    assert _variants(_shift) == 2
    del whole
