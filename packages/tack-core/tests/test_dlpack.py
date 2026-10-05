"""DLPack interop — zero-copy exchange with numpy, torch, cupy and friends.

`lang/dlpack.py` had no coverage at all: 80 statements never imported by
any test. It turns out to work, which is not something anyone knew — the
same audit finding on `interop/vtk.py` uncovered a module that had never
worked at all.

What these tests pin is the part that is easy to get subtly wrong and
impossible to notice: that the export is genuinely zero-copy, that the
dtype and shape survive, and that the exported buffer stays alive.
"""

import ctypes
import gc

import numpy as np
import pytest

import tack
from tack.lang import dlpack

ALL_DTYPES = [
    (tack.f32, np.float32), (tack.f64, np.float64),
    (tack.i8, np.int8), (tack.i16, np.int16),
    (tack.i32, np.int32), (tack.i64, np.int64),
    (tack.u8, np.uint8), (tack.u16, np.uint16),
    (tack.u32, np.uint32), (tack.u64, np.uint64),
]


@pytest.fixture(autouse=True)
def cpu():
    """DLPack export needs host-addressable memory."""
    tack.init(arch=tack.cpu)


def _field(values, dtype):
    f = tack.field(dtype=dtype, shape=values.shape)
    f.from_numpy(values)
    return f


# ── Export ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype,np_dtype", ALL_DTYPES,
                         ids=[str(d[0]) for d in ALL_DTYPES])
def test_every_supported_dtype_exports(dtype, np_dtype):
    """Every dtype a field can hold must survive the round trip.

    The narrow ints were missing from the type map — they raised
    "DLPack does not support dtype" despite being perfectly valid fields.
    """
    values = np.arange(4, dtype=np_dtype)
    got = np.from_dlpack(_field(values, dtype))
    assert got.dtype == np_dtype
    np.testing.assert_array_equal(got, values)


def test_export_is_zero_copy():
    """No copy: the consumer's buffer is the field's own memory."""
    values = np.arange(6, dtype=np.float32)
    field = _field(values, tack.f32)
    view = np.from_dlpack(field)

    assert view.__array_interface__["data"][0] == \
        field._buffer._data.__array_interface__["data"][0]


def test_writes_through_the_field_show_up_in_the_view():
    """The consequence that matters: the consumer sees live data."""
    field = _field(np.zeros(4, dtype=np.float32), tack.f32)
    view = np.from_dlpack(field)
    field.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    np.testing.assert_array_equal(view, [1, 2, 3, 4])


def _versioned_tensor(capsule):
    """Read back a versioned capsule tack just produced.

    The caller must keep `capsule` alive: the struct is pinned only until
    the capsule's destructor runs, so a temporary leaves this pointing at
    freed memory.
    """
    ptr = dlpack._PyCapsule_GetPointer(
        ctypes.c_void_p(id(capsule)), b"dltensor_versioned")
    return ctypes.cast(
        ptr, ctypes.POINTER(dlpack.DLManagedTensorVersioned)).contents


def test_the_exported_view_is_writable():
    """A consumer asking for v1.0 gets a writable view.

    The legacy capsule has no writability flag, so numpy has to assume the
    worst and marks what it gets read-only. Exporting the versioned form
    is what makes writing back possible.
    """
    field = _field(np.arange(4, dtype=np.float32), tack.f32)
    view = np.from_dlpack(field)
    assert view.flags.writeable
    view[0] = 99.0
    assert field.to_numpy()[0] == 99.0


def test_an_exported_view_can_be_exported_again():
    """The case that proves it matters.

    numpy refuses to re-export a read-only array over DLPack, so with only
    the legacy capsule any round trip through a consumer that hands the
    array back would fail -- as VTK interop does.
    """
    field = _field(np.arange(6, dtype=np.float32), tack.f32)
    again = np.from_dlpack(np.from_dlpack(field))
    np.testing.assert_array_equal(again, np.arange(6))


def test_a_read_only_field_exports_the_flag():
    """A field over immutable memory must not hand out a writable view."""
    source = np.arange(4, dtype=np.float32)
    source.flags.writeable = False
    field = tack.from_dlpack(source)
    assert not field._writable

    capsule = field.__dlpack__(max_version=(1, 0))
    tensor = _versioned_tensor(capsule)
    assert tensor.flags & dlpack.DLPACK_FLAG_BITMASK_READ_ONLY
    assert not np.from_dlpack(field).flags.writeable


def test_a_writable_field_exports_no_flag():
    field = _field(np.arange(4, dtype=np.float32), tack.f32)
    capsule = field.__dlpack__(max_version=(1, 0))
    tensor = _versioned_tensor(capsule)
    assert not tensor.flags & dlpack.DLPACK_FLAG_BITMASK_READ_ONLY
    assert (tensor.version.major, tensor.version.minor) == (1, 0)


def test_the_capsule_name_matches_the_struct():
    """Reading one layout as the other misreads every field silently."""
    field = _field(np.arange(4, dtype=np.float32), tack.f32)
    legacy = field.__dlpack__()
    assert dlpack._PyCapsule_IsValid(ctypes.c_void_p(id(legacy)), b"dltensor")

    versioned = field.__dlpack__(max_version=(1, 0))
    assert dlpack._PyCapsule_IsValid(
        ctypes.c_void_p(id(versioned)), b"dltensor_versioned")


def test_an_unconsumed_versioned_capsule_is_released():
    """The destructor has to recognise both capsule names."""
    gc.collect()
    before = len(dlpack._prevent_gc)
    for _ in range(25):
        field = _field(np.arange(4, dtype=np.float32), tack.f32)
        capsule = field.__dlpack__(max_version=(1, 0))
        del field, capsule
    gc.collect()
    assert len(dlpack._prevent_gc) == before


@pytest.mark.parametrize("shape", [(6,), (2, 3), (2, 3, 4), (1, 1)])
def test_shape_survives(shape):
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    got = np.from_dlpack(_field(values, tack.f32))
    assert got.shape == shape
    np.testing.assert_array_equal(got, values)


def test_kernel_results_are_visible_through_the_view():
    """End to end: run a kernel, read the answer through DLPack."""

    @tack.kernel
    def double_it(x, out, n):
        for i in range(n):
            out[i] = x[i] * 2.0

    n = 32
    x = _field(np.arange(n, dtype=np.float32), tack.f32)
    out = _field(np.zeros(n, dtype=np.float32), tack.f32)
    view = np.from_dlpack(out)

    double_it(x, out, n)
    np.testing.assert_allclose(view, np.arange(n) * 2.0, rtol=1e-6)


def test_the_field_outlives_its_capsule():
    """Exporting must not hand out a view into freed memory."""
    view = np.from_dlpack(_field(np.arange(8, dtype=np.float32), tack.f32))
    import gc
    gc.collect()
    np.testing.assert_array_equal(view, np.arange(8, dtype=np.float32))


# ── Export lifetime ──────────────────────────────────────────────────
#
# The producer pins the field, the managed tensor and the shape/stride
# arrays for as long as a consumer holds the tensor. Getting this wrong in
# either direction is bad and silent: release too early and the consumer
# reads freed memory; never release and every export leaks — device memory
# included, which is far less forgiving than host memory.

def _pinned():
    return len(dlpack._prevent_gc)


def test_a_consumed_export_is_released():
    """The bug this replaced: nothing was ever released.

    The retained objects were keyed by a counter while the deleter popped
    id(pointer), so the keys never matched. 50 export-and-drop cycles left
    50 fields pinned forever.
    """
    import gc
    before = _pinned()
    for _ in range(25):
        field = _field(np.arange(64, dtype=np.float32), tack.f32)
        view = np.from_dlpack(field)
        del field, view
    gc.collect()
    assert _pinned() == before


def test_an_unconsumed_capsule_is_released():
    """A capsule nobody adopts must still be cleaned up.

    Consumers rename the capsule to "used_dltensor" and take over calling
    the deleter. If it is dropped still named "dltensor", only the capsule
    destructor can free what was pinned.
    """
    import gc
    before = _pinned()
    for _ in range(10):
        field = _field(np.arange(16, dtype=np.float32), tack.f32)
        capsule = field.__dlpack__()
        del capsule, field
    gc.collect()
    assert _pinned() == before


def test_the_field_survives_while_a_consumer_holds_it():
    """Release must not be eager: the view owns the data now."""
    import gc
    field = _field(np.arange(8, dtype=np.float32), tack.f32)
    view = np.from_dlpack(field)
    del field
    for _ in range(3):
        gc.collect()
    np.testing.assert_array_equal(view, np.arange(8, dtype=np.float32))


def test_kernel_output_survives_its_field():
    """The realistic case: compute into a field, hand the result away."""
    import gc

    @tack.kernel
    def triple(out, n):
        for i in range(n):
            out[i] = float(i) * 3.0

    out = _field(np.zeros(6, dtype=np.float32), tack.f32)
    triple(out, 6)
    view = np.from_dlpack(out)
    del out
    for _ in range(3):
        gc.collect()
    np.testing.assert_allclose(view, np.arange(6) * 3.0, rtol=1e-6)


def test_many_live_exports_are_tracked_independently():
    import gc
    before = _pinned()
    views = [np.from_dlpack(_field(np.arange(16, dtype=np.float32), tack.f32))
             for _ in range(25)]
    assert _pinned() == before + 25
    for v in views:
        np.testing.assert_array_equal(v, np.arange(16, dtype=np.float32))
    del v          # the loop variable holds the last view past the loop
    del views
    gc.collect()
    assert _pinned() == before


def test_the_context_key_is_recoverable_from_the_tensor():
    """manager_ctx is the only thing the deleter gets; it must carry the key."""
    import ctypes

    field = _field(np.arange(4, dtype=np.float32), tack.f32)
    capsule = field.__dlpack__()

    ptr = dlpack._PyCapsule_GetPointer(
        ctypes.cast(id(capsule), ctypes.c_void_p), b"dltensor")
    managed = ctypes.cast(
        ptr, ctypes.POINTER(dlpack.DLManagedTensor)).contents

    assert managed.manager_ctx, "manager_ctx is NULL — the deleter cannot find its state"
    assert managed.manager_ctx in dlpack._prevent_gc


def test_two_exports_share_the_same_memory():
    field = _field(np.arange(4, dtype=np.float32), tack.f32)
    a, b = np.from_dlpack(field), np.from_dlpack(field)
    assert a.__array_interface__["data"][0] == b.__array_interface__["data"][0]
    field.from_numpy(np.full(4, 42.0, dtype=np.float32))
    np.testing.assert_array_equal(a, b)
    assert a[0] == 42.0


# ── Device reporting ─────────────────────────────────────────────────

def test_dlpack_device_reports_cpu():
    """kDLCPU is 1; the second element is the device ordinal."""
    device = _field(np.zeros(4, dtype=np.float32), tack.f32).__dlpack_device__()
    assert device == (1, 0)


def test_copy_true_is_refused():
    """Tack's export is a view; it cannot honour a copy request."""
    field = _field(np.zeros(4, dtype=np.float32), tack.f32)
    with pytest.raises(BufferError, match="copy"):
        field.__dlpack__(copy=True)


# ── Import ───────────────────────────────────────────────────────────

def test_import_from_numpy():
    values = np.arange(5, dtype=np.float32)
    field = tack.from_dlpack(values)
    assert field.dtype is tack.f32
    assert field.shape == (5,)
    np.testing.assert_array_equal(field.to_numpy(), values)


@pytest.mark.parametrize("dtype,np_dtype", ALL_DTYPES,
                         ids=[str(d[0]) for d in ALL_DTYPES])
def test_import_preserves_dtype(dtype, np_dtype):
    values = np.arange(4, dtype=np_dtype)
    field = tack.from_dlpack(values)
    assert field.dtype is dtype
    np.testing.assert_array_equal(field.to_numpy(), values)


def test_round_trip_through_both_directions():
    """field → numpy → field preserves dtype, shape and values."""
    original = _field(np.arange(6, dtype=np.float32), tack.f32)
    back = tack.from_dlpack(np.from_dlpack(original))

    assert back.dtype is original.dtype
    assert back.shape == original.shape
    np.testing.assert_array_equal(back.to_numpy(), original.to_numpy())


def test_unsupported_dtype_is_rejected_clearly():
    """A dtype with no DLPack equivalent must say so, not produce garbage."""
    field = _field(np.zeros(4, dtype=np.float32), tack.f32)
    field.dtype = "not-a-dtype"
    with pytest.raises(TypeError, match="DLPack"):
        field.__dlpack__()


# ── Metal ────────────────────────────────────────────────────────────
#
# Everything above runs on the CPU backend, pinned by the autouse fixture
# — which is why none of it ever reached the Metal path. The audit listed
# Metal's zero-copy claim as asserted by `_get_device_info` and never
# tested against a real MTLBuffer. These are that test.

def _metal_or_skip():
    try:
        tack.init(arch=tack.metal)
    except (ImportError, RuntimeError, OSError) as e:
        pytest.skip(f"no Metal backend: {e}")


@pytest.fixture
def metal():
    _metal_or_skip()
    yield
    tack.init(arch=tack.cpu)


def test_a_metal_export_points_into_the_mtlbuffer(metal):
    """The claim: the numpy view is the Metal allocation, not a copy.

    An address comparison is the weaker half — it cannot tell a live
    alias from a stale pointer, which is exactly how D6 slipped past a
    passing suite. The aliasing checks below are the real ones; this
    pins the address because a mismatch here localises the fault.
    """
    f = tack.field(dtype=tack.f32, shape=(8,))
    f.from_numpy(np.arange(8, dtype=np.float32))

    view = np.from_dlpack(f)
    assert view.ctypes.data == f._buffer._view.ctypes.data
    np.testing.assert_array_equal(view, np.arange(8, dtype=np.float32))


def test_a_gpu_kernel_sees_writes_made_through_the_exported_view(metal):
    """Write through the DLPack view, read it back with a real dispatch.

    No `from_numpy` anywhere: if the export were a copy, the kernel would
    read the field's original contents and the values below would be the
    ones it was seeded with.
    """
    @tack.kernel
    def scale(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i] * 10.0

    f = tack.field(dtype=tack.f32, shape=(8,))
    f.from_numpy(np.zeros(8, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(8,))

    view = np.from_dlpack(f)
    assert view.flags.writeable, "the versioned capsule should export writable"
    view[:] = np.arange(100, 108, dtype=np.float32)

    scale(f, out)
    np.testing.assert_array_equal(
        out.to_numpy(), np.arange(100, 108, dtype=np.float32) * 10.0)


def test_the_exported_view_sees_what_a_gpu_kernel_writes(metal):
    """And the other direction, through a view taken before the dispatch."""
    @tack.kernel
    def sevens(out):
        for i in range(out.shape[0]):
            out[i] = 7.0

    f = tack.field(dtype=tack.f32, shape=(8,))
    view = np.from_dlpack(f)

    sevens(f)
    np.testing.assert_array_equal(view, np.full(8, 7.0, dtype=np.float32))


def test_importing_host_memory_on_metal_is_refused_with_a_reason(metal):
    """It used to raise AttributeError from inside wrap_ptr.

    `_DEVICE_BACKENDS` allowed kDLCPU on metal, so the import was
    advertised; it then handed `wrap_ptr` an integer address where an
    MTLBuffer object was expected. Nothing could reach this path from
    the CPU-pinned tests above, so it had never run.
    """
    with pytest.raises(RuntimeError) as excinfo:
        tack.from_dlpack(np.arange(8, dtype=np.float32))

    message = str(excinfo.value)
    assert "page-aligned" in message, \
        "the refusal has to say why, not just that"
    assert "copy=True" in message, "and what to do instead"


def test_a_metal_field_does_not_round_trip_through_dlpack(metal):
    """The same refusal, reached from the field's own capsule.

    A Metal field exports as kDLCPU — correctly, its memory is
    host-addressable — so re-importing it takes the host path and hits
    the same wall. Worth pinning separately: it is the case that looks
    like it obviously ought to work.
    """
    f = tack.field(dtype=tack.f32, shape=(8,))
    with pytest.raises(RuntimeError, match="page-aligned"):
        tack.from_dlpack(f)


def test_copying_is_still_offered_on_metal(metal):
    """The refusal names `copy=True`, so that had better work."""
    values = np.arange(8, dtype=np.float32)
    f = tack.from_dlpack(values, copy=True)
    np.testing.assert_array_equal(f.to_numpy(), values)
    values[0] = 99.0
    assert f.to_numpy()[0] == 0.0, "copy=True must not alias"


def test_wrap_ptr_rejects_an_address_clearly(metal):
    """field_from_ptr is public, and its Metal contract is an object."""
    from tack.runtime.dispatch import get_backend
    with pytest.raises(TypeError, match="MTLBuffer object"):
        get_backend().wrap_ptr(0x1234, tack.f32, (8,))


# ---------------------------------------------------------------------------
# Capsule teardown after the module's globals are gone
# ---------------------------------------------------------------------------
def _without_globals(function):
    """`function` as it runs once its module has been torn down."""
    import types
    return types.FunctionType(function.__code__, {"__builtins__": {}},
                              function.__name__, function.__defaults__)


@pytest.mark.parametrize("versioned", [False, True])
def test_an_unadopted_capsule_is_released_without_module_globals(versioned):
    """A capsule can outlive `tack.lang.dlpack`.

    A consumer that failed part-way through an import was seen holding one
    until interpreter exit, where the destructor's global lookups raised
    NameError and the pinned field was never released. The destructor must
    work from what it captured at definition.
    """
    field = tack.field(dtype=tack.f32, shape=(4,))
    capsule = dlpack.field_to_dlpack(field, versioned=versioned)
    pinned = len(dlpack._prevent_gc)

    _without_globals(dlpack._destroy_capsule)(id(capsule))

    assert len(dlpack._prevent_gc) == pinned - 1
