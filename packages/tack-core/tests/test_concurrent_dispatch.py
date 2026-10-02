"""Dispatching one kernel from several threads.

`resolve_variant` has to work out the argument types before it can build a
cache key, and the obvious place to record them is the IR the key is read
back from. That IR is shared by every dispatch of a kernel, so recording
them there hands this call's types to every other thread for the window
between the write and the read.

The window is small and the failure is silent -- a thread computes its key
from another thread's dtypes and runs a variant compiled for them -- so it
is forced here rather than raced for. The GIL does not close it: a switch
may land at any bytecode boundary between the two calls.

Same shape as the C1 bug in test_kernel_cache.py: a cache key that does not
describe the thing it names.
"""

import threading
import time

import numpy as np
import pytest

import tack
from tack.runtime import kernel_utils

pytestmark = pytest.mark.filterwarnings("ignore")

TIMEOUT = 10


@pytest.fixture(autouse=True)
def cpu():
    tack.init(arch=tack.cpu)


@tack.kernel
def _scale(x, out, n):
    for i in range(n):
        out[i] = x[i] * 2.0


class Interleave:
    """Runs two dispatches so the second lands inside the first's window.

    A infers its types, then stops. B runs far enough to overwrite whatever
    A recorded. Only then does A go on to build its key.
    """

    def __init__(self, monkeypatch, hold_b_until_done=False):
        self.a_recorded = threading.Event()
        self.b_recorded = threading.Event()
        self.hold_b_until_done = hold_b_until_done
        self._local = threading.local()
        self._real = kernel_utils.infer_param_types
        monkeypatch.setattr(kernel_utils, "infer_param_types", self._patched)

    def _patched(self, *args):
        # resolve_variant infers twice on a miss -- once for the key, once
        # on its private copy. Only the first is inside the window.
        name = threading.current_thread().name
        first = not getattr(self._local, "seen", False)
        self._local.seen = True

        if name == "B" and first:
            assert self.a_recorded.wait(TIMEOUT), "A never recorded"

        result = self._real(*args)

        if first and name == "B" and not self.hold_b_until_done:
            # B has overwritten whatever A recorded; that is all A waits for.
            self.b_recorded.set()

        if first and name == "A":
            self.a_recorded.set()
            assert self.b_recorded.wait(TIMEOUT), "B never recorded"
        return result

    def run(self, target_a, target_b):
        errors = {}

        def wrap(name, fn, signal_after):
            def run():
                try:
                    fn()
                except Exception as exc:                # pragma: no cover
                    errors[name] = exc
                finally:
                    if signal_after:
                        self.b_recorded.set()
            return run

        a = threading.Thread(target=wrap("A", target_a, False), name="A")
        b = threading.Thread(
            target=wrap("B", target_b, self.hold_b_until_done), name="B")
        a.start(); b.start()
        a.join(TIMEOUT); b.join(TIMEOUT)
        assert not a.is_alive() and not b.is_alive(), "threads did not finish"
        assert not errors, errors
        return errors


def _fields(dtype, np_dtype, n, length=None):
    length = length or n
    x = tack.field(dtype=dtype, shape=(length,))
    out = tack.field(dtype=dtype, shape=(length,))
    x.from_numpy(np.arange(length, dtype=np_dtype))
    return x, out


def _slot():
    from tack.runtime.dispatch import get_backend
    return kernel_utils.kernel_cache_slot(get_backend()._cache, _scale)


def test_concurrent_dtypes_get_their_own_variants(monkeypatch):
    """Two dtypes must produce two entries, not one shared by both."""
    n = 8
    f32_x, f32_out = _fields(tack.f32, np.float32, n)
    f64_x, f64_out = _fields(tack.f64, np.float64, n)

    interleave = Interleave(monkeypatch)
    interleave.run(lambda: _scale(f32_x, f32_out, n),
                   lambda: _scale(f64_x, f64_out, n))

    keys = list(_slot().keys())
    assert len(keys) == 2, f"expected one variant per dtype, got {keys}"
    type_sigs = {k[0][:2] for k in keys}
    assert (tack.f32, tack.f32) in type_sigs
    assert (tack.f64, tack.f64) in type_sigs


def test_a_thread_gets_code_built_for_its_own_dtype(monkeypatch):
    """The consequence, with the numbers to show for it.

    B here finishes its whole dispatch -- compiling and caching an f64
    variant -- before A resumes, so a crossed key is a cache *hit* on
    somebody else's code rather than a redundant compile.

    A's fields are twice the dispatched length so that f64 code reading
    them stays in bounds: at equal length this reads and writes 2x past the
    end, and the test would be a heap corruption rather than an assertion.
    """
    n = 8
    f32_x, f32_out = _fields(tack.f32, np.float32, n, length=2 * n)
    f64_x, f64_out = _fields(tack.f64, np.float64, n)

    interleave = Interleave(monkeypatch, hold_b_until_done=True)
    interleave.run(lambda: _scale(f32_x, f32_out, n),
                   lambda: _scale(f64_x, f64_out, n))

    np.testing.assert_array_equal(
        f32_out.to_numpy()[:n], np.arange(n, dtype=np.float32) * 2.0)
    np.testing.assert_array_equal(
        f64_out.to_numpy(), np.arange(n, dtype=np.float64) * 2.0)


def test_the_template_is_left_alone(monkeypatch):
    """The property the fix rests on, asserted directly.

    If a dispatch writes its types into the shared IR, every other thread
    can read them. Nothing may be recorded there.
    """
    n = 8
    x, out = _fields(tack.f32, np.float32, n)
    template = _scale.get_ir().functions[0]

    _scale(x, out, n)

    recorded = [p.type_annotation for p in template.params]
    assert recorded == [None] * len(recorded), (
        f"dispatch wrote {recorded} into the shared template")
    assert not any(getattr(p, "_is_field", False) for p in template.params)


def test_many_threads_one_kernel():
    """Unforced, both dtypes, plenty of overlap. Nothing may cross."""
    n = 64
    results = {}
    errors = []

    def work(i):
        try:
            if i % 2:
                x, out = _fields(tack.f32, np.float32, n)
                want = np.arange(n, dtype=np.float32) * 2.0
            else:
                x, out = _fields(tack.f64, np.float64, n)
                want = np.arange(n, dtype=np.float64) * 2.0
            for _ in range(20):
                _scale(x, out, n)
            results[i] = np.array_equal(out.to_numpy(), want)
        except Exception as exc:                        # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)

    assert not errors, errors
    assert len(results) == 8
    assert all(results.values()), f"wrong values from {results}"


# ── The template rewrite's shared registry ───────────────────────────
#
# `rewrite_templates` registers each resolved @tack.func method in the
# module-global `_func_registry`, `transform_kernel` reads it back, and
# `get_ir` pops it afterwards. The name is built from `id(obj)`, so two
# threads sharing one @tack.data_oriented object build the *same* name and
# whichever finishes first deletes the entry the other is about to read.
#
# It survived on CPython only because that whole sequence fits in one GIL
# slice. Measured before the fix: 8 threads over 25 kernels were clean at
# the default 5 ms switch interval and at 1 ms, and 72 of 200 dispatches
# failed at 0.1 ms. Rather than depend on that, these force a switch at the
# exact point by making the transform yield.


@tack.data_oriented
class _Scaler:
    factor = 3.0

    def __init__(self, bias):
        self.bias = bias

    @tack.func
    def apply(self, v):
        return v * self.factor + self.bias


@tack.kernel
def _use_template(s, x, out):
    for i in range(x.shape[0]):
        out[i] = s.apply(x[i])


def _yield_during_transform(monkeypatch):
    """Make every transform give up the GIL at its start.

    Deterministic where a switch interval is not: each thread is guaranteed
    to be inside the register→transform→pop window when the next one enters
    it. Post-fix the lock serialises them, so this yields rather than
    deadlocks — the stalled thread is not waiting on anybody.
    """
    import tack.lang.kernel as kernel_mod
    real = kernel_mod.transform_kernel

    def slow_transform(*args, **kwargs):
        time.sleep(0.01)
        return real(*args, **kwargs)

    monkeypatch.setattr(kernel_mod, "transform_kernel", slow_transform)


def test_threads_sharing_a_template_object_all_succeed(monkeypatch):
    """The failure was `Function call '__tmpl_..._<id>__' not supported`."""
    _yield_during_transform(monkeypatch)
    shared = _Scaler(10.0)
    results, errors = {}, []
    lock = threading.Lock()

    def run(tid):
        try:
            x = tack.field(dtype=tack.f32, shape=(8,))
            out = tack.field(dtype=tack.f32, shape=(8,))
            x.from_numpy(np.ones(8, dtype=np.float32))
            _use_template(shared, x, out)
            with lock:
                results[tid] = out.to_numpy()
        except Exception as e:
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)

    assert not errors, f"concurrent template dispatch failed: {errors[:3]}"
    assert len(results) == 8
    for tid, got in results.items():
        np.testing.assert_allclose(got, 1.0 * 3.0 + 10.0)


def test_the_registry_is_left_clean(monkeypatch):
    """Temporaries must not outlive the transform that made them.

    The name carries `id(obj)`, which the allocator reuses, so a leaked
    entry is a stale method body waiting for an unrelated object to be born
    at the same address.

    Unlike the two either side of it, this one **passes on the pre-fix
    code**: distinct objects have distinct ids, so nothing collides and
    every pop finds its own entry. It is here as a guard on the invariant,
    not as a reproducer — worth saying, because a test that never failed is
    evidence of nothing until something changes.
    """
    from tack.lang.func import _func_registry

    _yield_during_transform(monkeypatch)
    before = set(_func_registry)

    def run(bias):
        s = _Scaler(bias)
        x = tack.field(dtype=tack.f32, shape=(4,))
        out = tack.field(dtype=tack.f32, shape=(4,))
        _use_template(s, x, out)

    threads = [threading.Thread(target=run, args=(float(i),)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)

    leaked = set(_func_registry) - before
    assert not leaked, f"template temporaries left in the registry: {leaked}"


@tack.data_oriented
class _ColdScaler:
    """A distinct class-level constant, so its IR cache entry is cold.

    `_make_cache_key` keys on class identity and its class-level scalars,
    so reusing `_Scaler` here would hit the entry the tests above built and
    never reach the transform this one needs to fail.
    """

    factor = 5.0

    def __init__(self, bias):
        self.bias = bias

    @tack.func
    def apply(self, v):
        return v * self.factor + self.bias


def test_a_failed_transform_still_cleans_up(monkeypatch):
    """The pop used to sit after the transform, so a raise skipped it."""
    import tack.lang.kernel as kernel_mod
    from tack.lang.func import _func_registry

    before = set(_func_registry)

    def boom(*args, **kwargs):
        raise ValueError("synthetic transform failure")

    monkeypatch.setattr(kernel_mod, "transform_kernel", boom)

    s = _ColdScaler(1.0)
    x = tack.field(dtype=tack.f32, shape=(4,))
    out = tack.field(dtype=tack.f32, shape=(4,))
    with pytest.raises(Exception, match="synthetic transform failure"):
        _use_template(s, x, out)

    assert set(_func_registry) == before, \
        "a failed transform left its registry temporaries behind"


# ── Switching backends under a live field ────────────────────────────

def test_a_backend_switch_under_a_live_field_says_so():
    """`tack.init()` rebinds a module global; fields do not follow it.

    The raw failure is `'NumpyBuffer' object has no attribute
    'metal_buffer'`, which names neither the cause nor the fix. Any second
    backend will do — the point is the diagnosis, not which one.
    """
    second = None
    for arch in ("metal", "cuda", "hip", "level_zero"):
        try:
            tack.init(arch=getattr(tack, arch))
            second = arch
            break
        except (ImportError, RuntimeError, OSError):
            continue
    if second is None:
        pytest.skip("only one backend on this machine")

    try:
        tack.init(arch=tack.cpu)
        x = tack.field(dtype=tack.f32, shape=(8,))
        out = tack.field(dtype=tack.f32, shape=(8,))
        x.from_numpy(np.arange(8, dtype=np.float32))

        tack.init(arch=getattr(tack, second))
        with pytest.raises(RuntimeError) as excinfo:
            _scale(x, out, 8)

        message = str(excinfo.value)
        assert "another backend" in message
        assert "cpu" in message, "the message must name where the field came from"
        assert "Re-allocate" in message, "and what to do about it"
    finally:
        tack.init(arch=tack.cpu)
