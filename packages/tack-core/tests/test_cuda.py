"""Tests for the CUDA compute backend."""

import numpy as np
import pytest

import tack

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


@pytest.fixture(autouse=True)
def cuda_backend():
    """Initialize CUDA backend for all tests in this module."""
    try:
        tack.set_default_device("cuda")
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"CUDA not available: {e}")


# --- Basic correctness ---

def test_vector_add():
    n = 1024
    x = tack.empty((n,))
    y = tack.empty((n,))
    out = tack.empty((n,))

    tack.copyto(x, np.arange(n, dtype=np.float32))
    tack.copyto(y, np.ones(n, dtype=np.float32) * 2.0)

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)

    result = out.numpy()
    expected = np.arange(n, dtype=np.float32) + 2.0
    assert np.allclose(result, expected)


def test_saxpy():
    n = 1024
    x = tack.empty((n,))
    y = tack.empty((n,))
    out = tack.empty((n,))

    tack.copyto(x, np.ones(n, dtype=np.float32) * 3.0)
    tack.copyto(y, np.ones(n, dtype=np.float32) * 1.0)

    @tack.kernel
    def saxpy(x, y, out):
        for i in range(x.shape[0]):
            out[i] = 2.0 * x[i] + y[i]

    saxpy(x, y, out)

    result = out.numpy()
    expected = 2.0 * 3.0 + 1.0  # 7.0
    assert np.allclose(result, expected)


def test_subtraction():
    n = 512
    a = tack.empty((n,))
    b = tack.empty((n,))
    out = tack.empty((n,))

    tack.copyto(a, np.full(n, 10.0, dtype=np.float32))
    tack.copyto(b, np.full(n, 3.0, dtype=np.float32))

    @tack.kernel
    def sub(a, b, out):
        for i in range(a.shape[0]):
            out[i] = a[i] - b[i]

    sub(a, b, out)

    result = out.numpy()
    assert np.allclose(result, 7.0)


def test_conditional():
    n = 1024
    x = tack.empty((n,))
    out = tack.empty((n,))

    data = np.arange(n, dtype=np.float32) - 512.0
    tack.copyto(x, data)

    @tack.kernel
    def relu(x, out):
        for i in range(x.shape[0]):
            if x[i] > 0.0:
                out[i] = x[i]
            else:
                out[i] = 0.0

    relu(x, out)

    result = out.numpy()
    expected = np.maximum(data, 0.0)
    assert np.allclose(result, expected)


def test_negation():
    n = 256
    x = tack.empty((n,))
    out = tack.empty((n,))

    tack.copyto(x, np.arange(n, dtype=np.float32))

    @tack.kernel
    def neg(x, out):
        for i in range(x.shape[0]):
            out[i] = -x[i]

    neg(x, out)

    result = out.numpy()
    expected = -np.arange(n, dtype=np.float32)
    assert np.allclose(result, expected)


def test_multiple_ops():
    n = 512
    a = tack.empty((n,))
    b = tack.empty((n,))
    c = tack.empty((n,))
    out = tack.empty((n,))

    tack.copyto(a, np.full(n, 10.0, dtype=np.float32))
    tack.copyto(b, np.full(n, 3.0, dtype=np.float32))
    tack.copyto(c, np.full(n, 2.0, dtype=np.float32))

    @tack.kernel
    def kern(a, b, c, out):
        for i in range(a.shape[0]):
            out[i] = (a[i] - b[i]) * c[i] / 2.0

    kern(a, b, c, out)

    result = out.numpy()
    expected = (10.0 - 3.0) * 2.0 / 2.0  # 7.0
    assert np.allclose(result, expected)


def test_math_sqrt():
    n = 256
    x = tack.empty((n,))
    out = tack.empty((n,))

    data = np.arange(1, n + 1, dtype=np.float32)
    tack.copyto(x, data)

    @tack.kernel
    def kern(x, out):
        for i in range(x.shape[0]):
            out[i] = sqrt(x[i])

    kern(x, out)

    result = out.numpy()
    expected = np.sqrt(data)
    assert np.allclose(result, expected, rtol=1e-5)


def test_cached_reuse():
    """Second call should use cached pipeline."""
    n = 256
    x = tack.empty((n,))
    y = tack.empty((n,))
    out = tack.empty((n,))

    tack.copyto(x, np.ones(n, dtype=np.float32))
    tack.copyto(y, np.ones(n, dtype=np.float32) * 2.0)

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)
    assert np.allclose(out.numpy(), 3.0)

    # Second call — cached
    tack.copyto(x, np.ones(n, dtype=np.float32) * 5.0)
    add(x, y, out)
    assert np.allclose(out.numpy(), 7.0)


def test_large_array():
    """Test with a larger array to exercise multiple thread blocks."""
    n = 1_000_000
    x = tack.empty((n,))
    y = tack.empty((n,))
    out = tack.empty((n,))

    tack.copyto(x, np.ones(n, dtype=np.float32) * 3.0)
    tack.copyto(y, np.ones(n, dtype=np.float32) * 4.0)

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)

    result = out.numpy()
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
    f = tack.empty((n,))
    tack.copyto(f, np.arange(n, dtype=np.float32))

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

    # Exporting copies into VMM-backed memory; the array itself is untouched.
    assert np.allclose(f.numpy(), np.arange(n, dtype=np.float32))


# --- Context ownership across re-initialization ---

def test_repeated_init_keeps_context_usable():
    """``tack.set_default_device("cuda")`` twice in a row must not kill the context.

    Devices are made once now, but a device made with options, or a second
    backend, can still overlap another: for a moment two CUDABackend objects
    hold the same context. The adopting one
    used to record ``_owns_context = False`` -- correct for a context owned
    by an embedding application, wrong for one Tack created -- and the
    outgoing backend's ``__del__`` then destroyed the context out from under
    it. Every later CUDA call failed with CUDA_ERROR_INVALID_CONTEXT.

    The autouse fixture has already initialized CUDA, so the init below is
    the second one.
    """
    import gc

    tack.set_default_device("cuda")
    gc.collect()  # force the outgoing backend's __del__ to run now

    n = 64
    f = tack.empty((n,))
    tack.copyto(f, np.arange(n, dtype=np.float32))
    assert np.allclose(f.numpy(), np.arange(n, dtype=np.float32))


def test_many_repeated_inits_do_not_leak_contexts():
    """Re-initializing repeatedly reuses one context rather than stacking them.

    The invariant is that the refcount equals the number of live backends,
    however many that happens to be: a plain script settles at one, but a
    test runner can keep a superseded backend alive in a frame or traceback
    for a while, and that is not a leak. What would be a leak is the count
    climbing with the number of inits, or a second context appearing on the
    device. Other devices have contexts of their own (one each) once
    something has used them, as test_multi_device does on a multi-GPU node.
    """
    import gc

    from tack.runtime.cuda_backend import _CONTEXTS, CUDABackend

    for _ in range(5):
        tack.set_default_device("cuda")
    gc.collect()

    backend = tack.get_default_device()._backend
    token = backend._token
    assert [t for t in _CONTEXTS.values() if t.device == backend.ordinal] == [token]
    live = sum(1 for o in gc.get_objects()
               if isinstance(o, CUDABackend) and getattr(o, "_token", None) is token)
    assert token.users == live

    f = tack.empty((32,))
    tack.copyto(f, np.ones(32, dtype=np.float32))
    assert f.numpy()[0] == 1.0


def test_kernel_runs_after_reinit():
    """A dispatch after re-initialization compiles and runs against a live context."""
    tack.set_default_device("cuda")

    n = 128
    x = tack.empty((n,))
    out = tack.empty((n,))
    tack.copyto(x, np.ones(n, dtype=np.float32))

    @tack.kernel
    def triple(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i] * 3.0

    triple(x, out)
    assert np.allclose(out.numpy(), 3.0)


_FOREIGN_SCRIPT = """
import gc
from cuda.bindings import driver

# The embedding framework's context, current before Tack touches CUDA.
driver.cuInit(0)
err, dev = driver.cuDeviceGet(0)
assert err == driver.CUresult.CUDA_SUCCESS
err, foreign = driver.cuCtxCreate(None, 0, dev)
assert err == driver.CUresult.CUDA_SUCCESS

import tack
from tack.runtime import dispatch
from tack.runtime.cuda_backend import _CONTEXTS

backend = tack.device("cuda")._backend
assert int(backend._context) == int(foreign), "the current context was not adopted"
assert backend._owns_context is False
# Tracked, so buffers get a token to hold, but not owned -- which is
# what keeps it out of cuCtxDestroy's way.
assert _CONTEXTS[int(foreign)].owned is False

# Let go of every Tack reference to the device; the foreign context must survive.
del backend
dispatch._devices.pop(("cuda", 0))
gc.collect()

err, ptr = driver.cuMemAlloc(256)
assert err == driver.CUresult.CUDA_SUCCESS, "foreign context was destroyed"
driver.cuMemFree(ptr)
driver.cuCtxDestroy(foreign)
print("ok")
"""


def test_foreign_context_is_adopted_but_never_destroyed(tmp_path):
    """A context Tack did not create stays alive after Tack lets go of it.

    This is the case the adoption logic exists for -- an embedding framework
    (AMReX and friends) sets up its own context and expects Tack to share it,
    not to take it away. Refcounting ownership must not regress that: a
    foreign context is never recorded as owned, so nothing ever destroys it.

    In a fresh process, because a device lives as long as the process: this
    one made its CUDA device long ago, with Tack's own context, so the
    context is adopted only where it is current before Tack's first use.
    """
    import subprocess
    import sys

    script = tmp_path / "foreign.py"
    script.write_text(_FOREIGN_SCRIPT)
    proc = subprocess.run([sys.executable, "-u", str(script)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


# A device is a value that lives as long as the process (or as long as
# something holds it), and every array holds its device. So nothing a
# program can do through the API destroys a CUDA context under a live
# array: releasing Tack's own reference to the device, as below, still
# leaves the array's. The stale-pointer token (`_ContextToken`) remains for
# contexts an embedding application destroys behind Tack's back.

_RELEASED_PRELUDE = """
import gc, sys
import numpy as np
import tack
from tack.runtime import dispatch

@tack.kernel
def double(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] * 2.0

tack.set_default_device("cuda")
f = tack.empty((64,), tack.f32)
tack.copyto(f, np.arange(64, dtype=np.float32))
double(f, f)                    # compile a variant while the context is alive
tack.set_default_device("cpu")
dispatch._devices.pop(("cuda", 0))     # Tack forgets the device...
gc.collect()                           # ...but the array still holds it
tack.set_default_device("cuda")        # a new device, with its own context
"""


def _run_released_script(tmp_path, body):
    """Run an access after the release above in a subprocess."""
    import subprocess
    import sys

    script = tmp_path / "released.py"
    script.write_text(_RELEASED_PRELUDE + body)
    return subprocess.run([sys.executable, "-u", str(script)],
                          capture_output=True, text=True, timeout=300)


def _assert_no_crash(proc):
    """The process must have died of a Python exception, not a signal."""
    assert proc.returncode >= 0, f"killed by signal {-proc.returncode}"
    assert proc.returncode != 139, "segfaulted"


@pytest.mark.parametrize("verb,action", [
    ("read", "assert f.numpy()[3] == 6.0"),
    ("write", "tack.copyto(f, np.ones(64, dtype=np.float32))"),
    ("fill", "f.fill(1.0)"),
    ("export", "f.export_memory()"),
    ("launch", "double(f, f)"),
])
def test_an_array_keeps_its_device_and_context_alive(tmp_path, verb, action):
    """An array whose device Tack has let go of still works: it holds the
    device, so its context and its pointers stay valid. A launch runs on
    the array's device, not the new default's."""
    body = f"""
{action}
print("OK")
"""
    proc = _run_released_script(tmp_path, body)
    _assert_no_crash(proc)
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"{verb} failed (rc={proc.returncode})\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr[-2000:]}")


def test_a_released_array_can_be_collected_without_crashing(tmp_path):
    """Dropping the last array of a released device frees it, and its context,
    cleanly; the new device goes on working."""
    body = """
del f
gc.collect()
g = tack.empty((16,), tack.f32)
tack.copyto(g, np.ones(16, dtype=np.float32))
assert g.numpy()[0] == 1.0
print("OK")
"""
    proc = _run_released_script(tmp_path, body)
    _assert_no_crash(proc)
    assert proc.returncode == 0, f"stderr: {proc.stderr[-2000:]}"
    assert "OK" in proc.stdout


def test_array_survives_repeated_init():
    """Re-initializing CUDA keeps the context, so existing arrays stay valid."""
    import gc

    n = 64
    f = tack.empty((n,))
    tack.copyto(f, np.arange(n, dtype=np.float32))

    tack.set_default_device("cuda")
    gc.collect()

    assert np.allclose(f.numpy(), np.arange(n, dtype=np.float32))



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


@pytest.mark.parametrize("dtype", [tack.i8, tack.i16])
def test_narrow_negation_avoids_ptx_neg_s16(dtype):
    """ptxas 12.0-13.2, the JIT of drivers before CUDA 13.3, widen PTX
    neg.s16/abs.s16 without sign-extending from bit 15: i16 -(-32768)
    widened read 32768 on an H100 with driver 595. Correct drivers pass
    test_wrapped_negation_and_abs_survive_widening either way, so check the
    PTX itself."""
    from tack.codegen.cuda_gen import generate_cuda_source
    from tack.lang.inspect_kernel import _prepare_ir
    from tack.runtime.cuda_backend import _compile_ptx

    @tack.kernel
    def negate(x, out):
        for i in range(x.shape[0]):
            out[i, 0] = -x[i]
            out[i, 1] = abs(x[i])
            n = -x[i]
            out[i, 2] = tack.i64(n)

    x = tack.empty((4,), dtype)
    out = tack.empty((4, 3), tack.i64)
    function, _ = _prepare_ir(negate, (x, out))
    ptx = _compile_ptx(generate_cuda_source(function), "negate").decode()
    assert "neg.s16" not in ptx and "abs.s16" not in ptx


def test_nvrtc_leaves_the_thread_locale_alone():
    """Compiling a kernel must not change Python's default text encoding.

    NVRTC 12.0 left the compiling thread in the C locale, so after the
    first CUDA kernel every ``read_text()`` without an encoding was ASCII
    and failed on non-ASCII text. NVRTC 13.2 does not, so on a newer
    toolkit this only checks that restoring the locale does no harm.
    """
    import locale

    from tack.runtime.cuda_backend import _compile_ptx
    before = locale.getencoding()
    src = 'extern "C" __global__ void k(float* a) { a[0] = 2.0f; }'
    assert _compile_ptx(src, "k")
    assert locale.getencoding() == before


def test_old_nvrtc_warns_at_the_first_compile(monkeypatch):
    """NVRTC 12.0 miscompiled tack-examples' lbm on an H100, so compiling
    with NVRTC 12 says so, once per process."""
    import warnings

    from tack.runtime import cuda_backend

    def nvrtc_12():
        return cuda_backend.nvrtc.nvrtcResult.NVRTC_SUCCESS, 12, 0

    src = 'extern "C" __global__ void k(float* a) { a[0] = 2.0f; }'
    monkeypatch.setattr(cuda_backend.nvrtc, "nvrtcVersion", nvrtc_12)
    cuda_backend._warn_if_old_nvrtc.cache_clear()
    try:
        with pytest.warns(UserWarning, match="NVRTC 12.0 is older"):
            cuda_backend._compile_ptx(src, "k")
        with warnings.catch_warnings(record=True) as again:
            warnings.simplefilter("always")
            cuda_backend._compile_ptx(src, "k")
        assert not [w for w in again if "NVRTC" in str(w.message)]
    finally:
        cuda_backend._warn_if_old_nvrtc.cache_clear()


def test_ordinary_and_floor_division_kernels_compile_in_sequence():
    """Ordinary and floating floor-division kernels compile and run in
    one process with the same safe settings, preserving the CX7 coverage."""
    x = tack.empty((8,))
    y = tack.empty((8,))
    out = tack.empty((8,))
    tack.copyto(x, np.array([6.0, 7.5, -7.5, 1e30, 0.3, -0.0, 5.0, 2.0], dtype=np.float32))
    tack.copyto(y, np.array([0.1, -3.0, 3.0, 1e-10, 0.1, 1.0, -2.0, 0.5], dtype=np.float32))

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
    np.testing.assert_allclose(out.numpy(), x.numpy() * y.numpy() + 1.0, rtol=1e-6)
    floor_div(x, y, out)
    with np.errstate(all="ignore"):              # 1e30 // 1e-10 overflows to inf
        expected = np.floor_divide(x.numpy(), y.numpy())
    np.testing.assert_array_equal(out.numpy(), expected)
    plain_second(x, y, out)
    np.testing.assert_allclose(out.numpy(), x.numpy() * y.numpy() - 1.0, rtol=1e-6)


def test_host_element_reads_copy_one_element():
    """array[i] copies just the element on CUDA, from an owned buffer, a
    wrapped pointer in each form, a view and an exportable buffer; a range
    past the allocation is refused rather than read."""
    from tack.runtime.cuda_backend import ExportableCUDABuffer

    f = tack.empty((4, 5), tack.f64)
    tack.copyto(f, np.arange(20, dtype=np.float64).reshape(4, 5))
    f._buffer.to_numpy = None          # a whole-buffer copy would fail here
    assert f[3, 4] == 19.0 and f[-4, 0] == 0.0
    assert f.reshape((20,))[7] == 7.0

    address = f._buffer.address
    for ptr in (address, np.uint64(address)):
        alias = tack.array_from_ptr(ptr, (20,), tack.f64)
        assert alias[13] == 13.0

    v = tack.empty((6,), tack.vec(3, tack.i32))
    tack.copyto(v, np.arange(18, dtype=np.int32).reshape(6, 3))
    v._buffer.to_numpy = None
    np.testing.assert_array_equal(v[5], [15, 16, 17])

    exportable = ExportableCUDABuffer(np.dtype(np.float32), (8,))
    exportable.from_numpy(np.arange(8, dtype=np.float32) * 2)
    np.testing.assert_array_equal(exportable.read_range(6, 2), [12.0, 14.0])

    with pytest.raises(IndexError, match="outside a buffer of 20 elements"):
        f._buffer.read_range(19, 2)
    with pytest.raises(IndexError, match="outside"):
        f._buffer.read_range(-1, 1)


def test_device_memory_imported_with_the_cpu_as_default_stays_on_its_device():
    """As the array API's from_dlpack does: CUDA memory is a CUDA array,
    sharing the allocation, whatever the default device. Asked onto the
    CPU, the refusal names the fix."""
    g = tack.arange(4, dtype=tack.f32)              # on CUDA, the default here
    with tack.device("cpu"):
        wrapped = tack.from_dlpack(g)
        with pytest.raises(RuntimeError, match=r"from_dlpack\(x, device='cuda'\)"):
            tack.from_dlpack(g, device="cpu")
    assert wrapped.device == tack.device("cuda")
    assert int(wrapped._buffer.device_ptr) == int(g._buffer.device_ptr)
    assert np.array_equal(wrapped.numpy(), np.arange(4, dtype=np.float32))
