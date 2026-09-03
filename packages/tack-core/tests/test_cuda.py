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

    from tack.runtime.cuda_backend import _OWNED_CONTEXTS, CUDABackend

    for _ in range(5):
        tack.init(arch=tack.cuda)
    gc.collect()

    live = sum(1 for o in gc.get_objects() if isinstance(o, CUDABackend))
    assert len(_OWNED_CONTEXTS) == 1
    assert list(_OWNED_CONTEXTS.values()) == [live]

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
    from tack.runtime.cuda_backend import _OWNED_CONTEXTS

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
        assert int(foreign) not in _OWNED_CONTEXTS

        # Tear every Tack backend down; the foreign context must survive.
        del backend
        tack.init(arch=tack.cpu)
        gc.collect()

        err, ptr = driver.cuMemAlloc(256)
        assert err == driver.CUresult.CUDA_SUCCESS, "foreign context was destroyed"
        driver.cuMemFree(ptr)
    finally:
        driver.cuCtxDestroy(foreign)
