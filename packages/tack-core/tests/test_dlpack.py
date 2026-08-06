"""DLPack interop — zero-copy exchange with numpy, torch, cupy and friends.

`lang/dlpack.py` had no coverage at all: 80 statements never imported by
any test. It turns out to work, which is not something anyone knew — the
same audit finding on `interop/vtk.py` uncovered a module that had never
worked at all.

What these tests pin is the part that is easy to get subtly wrong and
impossible to notice: that the export is genuinely zero-copy, that the
dtype and shape survive, and that the exported buffer stays alive.
"""

import numpy as np
import pytest

import tack

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


def test_the_exported_view_is_read_only():
    """Documents a real limitation of the current export.

    `__dlpack__` hands out the unversioned "dltensor" capsule. That
    protocol has no writability flag, so numpy cannot tell whether writing
    is safe and marks the result read-only. Consumers can therefore read
    Tack results without a copy, but not write back through them.

    Lifting this means implementing DLPack v1.0 — the versioned
    `DLManagedTensorVersioned` capsule, which carries a read-only flag.
    Until then this test is the record that the limitation is known.
    """
    field = _field(np.arange(4, dtype=np.float32), tack.f32)
    view = np.from_dlpack(field)
    assert not view.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        view[0] = 1.0


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
    from tack.lang import dlpack
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
    from tack.lang import dlpack

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
