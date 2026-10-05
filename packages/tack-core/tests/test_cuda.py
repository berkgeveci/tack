"""Tests for the CUDA compute backend."""

import numpy as np
import pytest

import tack


@pytest.fixture(autouse=True)
def cuda_backend():
    """Initialize CUDA backend for all tests in this module."""
    try:
        tack.init(arch=tack.cuda)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"CUDA not available: {e}")


# --- Basic correctness ---

def test_vector_add():
    n = 1024
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.arange(n, dtype=np.float32))
    y.from_numpy(np.ones(n, dtype=np.float32) * 2.0)

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)

    result = out.to_numpy()
    expected = np.arange(n, dtype=np.float32) + 2.0
    assert np.allclose(result, expected)


def test_saxpy():
    n = 1024
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    y.from_numpy(np.ones(n, dtype=np.float32) * 1.0)

    @tack.kernel
    def saxpy(x, y, out):
        for i in range(x.shape[0]):
            out[i] = 2.0 * x[i] + y[i]

    saxpy(x, y, out)

    result = out.to_numpy()
    expected = 2.0 * 3.0 + 1.0  # 7.0
    assert np.allclose(result, expected)


def test_subtraction():
    n = 512
    a = tack.field(dtype=tack.f32, shape=(n,))
    b = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    a.from_numpy(np.full(n, 10.0, dtype=np.float32))
    b.from_numpy(np.full(n, 3.0, dtype=np.float32))

    @tack.kernel
    def sub(a, b, out):
        for i in range(a.shape[0]):
            out[i] = a[i] - b[i]

    sub(a, b, out)

    result = out.to_numpy()
    assert np.allclose(result, 7.0)


def test_conditional():
    n = 1024
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    data = np.arange(n, dtype=np.float32) - 512.0
    x.from_numpy(data)

    @tack.kernel
    def relu(x, out):
        for i in range(x.shape[0]):
            if x[i] > 0.0:
                out[i] = x[i]
            else:
                out[i] = 0.0

    relu(x, out)

    result = out.to_numpy()
    expected = np.maximum(data, 0.0)
    assert np.allclose(result, expected)


def test_negation():
    n = 256
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.arange(n, dtype=np.float32))

    @tack.kernel
    def neg(x, out):
        for i in range(x.shape[0]):
            out[i] = -x[i]

    neg(x, out)

    result = out.to_numpy()
    expected = -np.arange(n, dtype=np.float32)
    assert np.allclose(result, expected)


def test_multiple_ops():
    n = 512
    a = tack.field(dtype=tack.f32, shape=(n,))
    b = tack.field(dtype=tack.f32, shape=(n,))
    c = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    a.from_numpy(np.full(n, 10.0, dtype=np.float32))
    b.from_numpy(np.full(n, 3.0, dtype=np.float32))
    c.from_numpy(np.full(n, 2.0, dtype=np.float32))

    @tack.kernel
    def kern(a, b, c, out):
        for i in range(a.shape[0]):
            out[i] = (a[i] - b[i]) * c[i] / 2.0

    kern(a, b, c, out)

    result = out.to_numpy()
    expected = (10.0 - 3.0) * 2.0 / 2.0  # 7.0
    assert np.allclose(result, expected)


def test_math_sqrt():
    n = 256
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    data = np.arange(1, n + 1, dtype=np.float32)
    x.from_numpy(data)

    @tack.kernel
    def kern(x, out):
        for i in range(x.shape[0]):
            out[i] = sqrt(x[i])

    kern(x, out)

    result = out.to_numpy()
    expected = np.sqrt(data)
    assert np.allclose(result, expected, rtol=1e-5)


def test_cached_reuse():
    """Second call should use cached pipeline."""
    n = 256
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32))
    y.from_numpy(np.ones(n, dtype=np.float32) * 2.0)

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)
    assert np.allclose(out.to_numpy(), 3.0)

    # Second call — cached
    x.from_numpy(np.ones(n, dtype=np.float32) * 5.0)
    add(x, y, out)
    assert np.allclose(out.to_numpy(), 7.0)


def test_large_array():
    """Test with a larger array to exercise multiple thread blocks."""
    n = 1_000_000
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    y.from_numpy(np.ones(n, dtype=np.float32) * 4.0)

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)

    result = out.to_numpy()
    assert np.allclose(result, 7.0)


# --- Memory export (cross-API sharing) ---

def test_export_memory():
    """export_memory() must hand back a usable handle on this platform.

    Which kind of handle that is varies -- a POSIX fd on Linux, a Win32 KMT
    handle on Windows -- so this asserts the shape of the result rather than
    one particular kind. Nothing exercised this path before, which is how the
    backend shipped asking for a POSIX fd unconditionally and dying on Windows
    with a bare CUDA_ERROR_INVALID_VALUE.
    """
    n = 256
    f = tack.field(dtype=tack.f32, shape=(n,))
    f.from_numpy(np.arange(n, dtype=np.float32))

    exported = f.export_memory()

    assert exported.backend == "cuda"
    assert exported.handle_type in ("posix_fd", "win32_kmt")
    assert exported.size == n * 4
    assert exported.allocation_size >= exported.size
    assert isinstance(exported.handle, int)
    assert exported.handle != 0
    assert len(exported.device_uuid) == 16

    # The handle is cached, so a second export is the same one and not a leak
    # of a second OS handle per call.
    assert f.export_memory().handle == exported.handle

    # Exporting copies into VMM-backed memory; the field itself is untouched.
    assert np.allclose(f.to_numpy(), np.arange(n, dtype=np.float32))


# --- Context ownership across re-initialization ---

def test_repeated_init_keeps_context_usable():
    """``tack.init(arch=tack.cuda)`` twice in a row must not kill the context.

    ``init()`` builds the new backend before dropping the old one, so for a
    moment two CUDABackend objects hold the same context. The adopting one
    used to record ``_owns_context = False`` -- correct for a context owned
    by an embedding application, wrong for one Tack created -- and the
    outgoing backend's ``__del__`` then destroyed the context out from under
    it. Every later CUDA call failed with CUDA_ERROR_INVALID_CONTEXT.

    The autouse fixture has already initialized CUDA, so the init below is
    the second one.
    """
    import gc

    tack.init(arch=tack.cuda)
    gc.collect()  # force the outgoing backend's __del__ to run now

    n = 64
    f = tack.field(dtype=tack.f32, shape=(n,))
    f.from_numpy(np.arange(n, dtype=np.float32))
    assert np.allclose(f.to_numpy(), np.arange(n, dtype=np.float32))


def test_many_repeated_inits_do_not_leak_contexts():
    """Re-initializing repeatedly reuses one context rather than stacking them.

    The invariant is that the refcount equals the number of live backends,
    however many that happens to be: a plain script settles at one, but a
    test runner can keep a superseded backend alive in a frame or traceback
    for a while, and that is not a leak. What would be a leak is the count
    climbing with the number of inits, or a second context appearing.
    """
    import gc

    from tack.runtime.cuda_backend import _CONTEXTS, CUDABackend

    for _ in range(5):
        tack.init(arch=tack.cuda)
    gc.collect()

    live = sum(1 for o in gc.get_objects() if isinstance(o, CUDABackend))
    assert len(_CONTEXTS) == 1
    assert [t.users for t in _CONTEXTS.values()] == [live]

    f = tack.field(dtype=tack.f32, shape=(32,))
    f.from_numpy(np.ones(32, dtype=np.float32))
    assert f.to_numpy()[0] == 1.0


def test_kernel_runs_after_reinit():
    """A dispatch after re-initialization compiles and runs against a live context."""
    tack.init(arch=tack.cuda)

    n = 128
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32))

    @tack.kernel
    def triple(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i] * 3.0

    triple(x, out)
    assert np.allclose(out.to_numpy(), 3.0)


def test_foreign_context_is_adopted_but_never_destroyed():
    """A context Tack did not create stays alive after every backend is gone.

    This is the case the adoption logic exists for -- an embedding framework
    (AMReX and friends) sets up its own context and expects Tack to share it,
    not to take it away. Refcounting ownership must not regress that: a
    foreign context is never recorded as owned, so nothing ever destroys it.
    """
    import gc

    from cuda.bindings import driver

    from tack.runtime import dispatch
    from tack.runtime.cuda_backend import _CONTEXTS

    # Drop Tack's own context first, so the one we create is the current one.
    tack.init(arch=tack.cpu)
    gc.collect()

    driver.cuInit(0)
    err, dev = driver.cuDeviceGet(0)
    assert err == driver.CUresult.CUDA_SUCCESS
    err, foreign = driver.cuCtxCreate(None, 0, dev)
    assert err == driver.CUresult.CUDA_SUCCESS

    try:
        tack.init(arch=tack.cuda)
        backend = dispatch.get_backend()

        assert int(backend._context) == int(foreign)
        assert backend._owns_context is False
        # Tracked, so buffers get a token to hold, but not owned -- which is
        # what keeps it out of cuCtxDestroy's way.
        assert _CONTEXTS[int(foreign)].owned is False

        # Tear every Tack backend down; the foreign context must survive.
        del backend
        tack.init(arch=tack.cpu)
        gc.collect()

        err, ptr = driver.cuMemAlloc(256)
        assert err == driver.CUresult.CUDA_SUCCESS, "foreign context was destroyed"
        driver.cuMemFree(ptr)
    finally:
        driver.cuCtxDestroy(foreign)


# --- Fields do not outlive their context ---

# These run in a subprocess for two reasons. The scenario used to be a
# SIGSEGV, and a segfault in-process takes the whole test session with it --
# there would be no failure report, just a dead runner. And the root
# conftest holds a CUDABackend in a module global for the life of the
# session, so the CUDA context never actually reaches zero users here and an
# in-process test could not build a stale field even if it were safe to.

_STALE_PRELUDE = """
import gc, sys
import numpy as np
import tack

@tack.kernel
def double(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] * 2.0

tack.init(arch=tack.cuda)
f = tack.field(dtype=tack.f32, shape=(64,))
f.from_numpy(np.arange(64, dtype=np.float32))
double(f, f)                 # compile a variant while the context is alive
tack.init(arch=tack.cpu)     # destroys the CUDA context
gc.collect()
tack.init(arch=tack.cuda)    # fresh context; the old pointers stay dead
"""


def _run_stale_script(tmp_path, body):
    """Run a stale-field access in a subprocess; return the CompletedProcess."""
    import subprocess
    import sys

    script = tmp_path / "stale.py"
    script.write_text(_STALE_PRELUDE + body)
    return subprocess.run([sys.executable, "-u", str(script)],
                          capture_output=True, text=True, timeout=300)


def _assert_no_crash(proc):
    """The process must have died of a Python exception, not a signal."""
    assert proc.returncode >= 0, f"killed by signal {-proc.returncode}"
    assert proc.returncode != 139, "segfaulted"


@pytest.mark.parametrize("verb,action", [
    ("read", "f.to_numpy()"),
    ("write", "f.from_numpy(np.ones(64, dtype=np.float32))"),
    ("fill", "f.fill(1.0)"),
    ("export", "f.export_memory()"),
    ("launch", "double(f, f)"),
])
def test_stale_field_raises_instead_of_segfaulting(tmp_path, verb, action):
    """Touching a field whose context is gone must raise, not crash.

    cuCtxDestroy invalidates every allocation made in the context, and a copy
    through one of those pointers faults inside the driver -- SIGSEGV, with no
    CUresult to check and no traceback to print. The buffer cannot learn this
    from its own pointer, so it holds a token that the context marks dead on
    the way out.
    """
    body = f"""
try:
    {action}
except RuntimeError as e:
    assert "context" in str(e), e
    print("RAISED")
    sys.exit(0)
print("NO ERROR")
sys.exit(1)
"""
    proc = _run_stale_script(tmp_path, body)
    _assert_no_crash(proc)
    assert "RAISED" in proc.stdout, (
        f"{verb} did not raise (rc={proc.returncode})\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr[-2000:]}")


def test_stale_field_can_be_collected_without_crashing(tmp_path):
    """Freeing a dead allocation is the same fault as copying into one.

    The context took the allocation with it, so ``__del__`` has to skip
    cuMemFree rather than call it into a context that no longer exists.
    """
    body = """
del f
gc.collect()
g = tack.field(dtype=tack.f32, shape=(16,))
g.from_numpy(np.ones(16, dtype=np.float32))
assert g.to_numpy()[0] == 1.0
print("OK")
"""
    proc = _run_stale_script(tmp_path, body)
    _assert_no_crash(proc)
    assert proc.returncode == 0, f"stderr: {proc.stderr[-2000:]}"
    assert "OK" in proc.stdout


def test_field_survives_repeated_init():
    """Re-initializing CUDA keeps the context, so existing fields stay valid."""
    import gc

    n = 64
    f = tack.field(dtype=tack.f32, shape=(n,))
    f.from_numpy(np.arange(n, dtype=np.float32))

    tack.init(arch=tack.cuda)
    gc.collect()

    assert np.allclose(f.to_numpy(), np.arange(n, dtype=np.float32))



# --- NVRTC option passing (CX7) ---

def test_nvrtc_accepts_safe_option_set():
    """The backend's safe option set must compile a trivial kernel.

    cuda-python marshals a Python list of bytes itself; handed a ctypes
    ``c_char_p`` array it mis-read the second entry when
    ``--extra-device-vectorization`` came first and rejected the compile
    with "unrecognized option". That broke every kernel that keeps fast
    math, while the precise-math kernels that passed a single option kept
    working. All kernels now use the same safe settings; this regression
    exercises the current multi-option list through the actual binding.
    """
    from tack.runtime.cuda_backend import _compile_ptx
    src = 'extern "C" __global__ void k(float* a) { a[0] = 1.0f; }'
    assert _compile_ptx(src, "k")


def test_ordinary_and_floor_division_kernels_compile_in_sequence():
    """Ordinary and floating floor-division kernels compile and run in
    one process with the same safe settings, preserving the CX7 coverage."""
    x = tack.field(dtype=tack.f32, shape=(8,))
    y = tack.field(dtype=tack.f32, shape=(8,))
    out = tack.field(dtype=tack.f32, shape=(8,))
    x.from_numpy(np.array([6.0, 7.5, -7.5, 1e30, 0.3, -0.0, 5.0, 2.0], dtype=np.float32))
    y.from_numpy(np.array([0.1, -3.0, 3.0, 1e-10, 0.1, 1.0, -2.0, 0.5], dtype=np.float32))

    @tack.kernel
    def plain_first(x, y, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * y[i] + 1.0

    @tack.kernel
    def floor_div(x, y, out):
        for i in range(out.shape[0]):
            out[i] = x[i] // y[i]

    @tack.kernel
    def plain_second(x, y, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * y[i] - 1.0

    plain_first(x, y, out)
    np.testing.assert_allclose(out.to_numpy(), x.to_numpy() * y.to_numpy() + 1.0, rtol=1e-6)
    floor_div(x, y, out)
    with np.errstate(all="ignore"):              # 1e30 // 1e-10 overflows to inf
        expected = np.floor_divide(x.to_numpy(), y.to_numpy())
    np.testing.assert_array_equal(out.to_numpy(), expected)
    plain_second(x, y, out)
    np.testing.assert_allclose(out.to_numpy(), x.to_numpy() * y.to_numpy() - 1.0, rtol=1e-6)
