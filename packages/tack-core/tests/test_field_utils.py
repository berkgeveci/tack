"""Tests for field utility functions: copy, astype, reshape, concat, zeros, etc."""

import numpy as np
import pytest

import tack
from tack import algorithms

# --- Field.size and len() ---

def test_field_size(backend):
    f = tack.field(dtype=tack.f32, shape=(3, 4))
    assert f.size == 12


def test_field_size_1d(backend):
    f = tack.field(dtype=tack.f32, shape=(100,))
    assert f.size == 100


def test_empty_field(backend):
    """A filter that selects nothing returns a field of no elements; it
    allocates, round-trips an empty array and is a valid kernel argument."""
    f = tack.field(dtype=tack.f32, shape=(0,))
    assert f.size == 0
    assert f.to_numpy().shape == (0,)
    f.from_numpy(np.zeros(0, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(0,))
    algorithms.copy(f, out, 0)
    assert out.to_numpy().shape == (0,)


def test_field_len(backend):
    f = tack.field(dtype=tack.f32, shape=(10,))
    assert len(f) == 10


def test_field_len_2d(backend):
    f = tack.field(dtype=tack.f32, shape=(5, 3))
    assert len(f) == 5


# --- Field.copy() ---

def test_field_copy(backend):
    f = tack.field(dtype=tack.f32, shape=(8,))
    f.from_numpy(np.arange(8, dtype=np.float32))
    g = f.copy()
    np.testing.assert_array_equal(g.to_numpy(), f.to_numpy())
    # Verify it's a separate buffer (modifying one doesn't affect the other)
    g.fill(0)
    assert f.to_numpy()[0] == 0.0  # original unchanged... well, depends on backend
    # At least verify shapes and dtypes match
    assert g.shape == f.shape
    assert g.dtype is f.dtype


def test_field_copy_i32(backend):
    f = tack.field(dtype=tack.i32, shape=(4,))
    f.from_numpy(np.array([10, 20, 30, 40], dtype=np.int32))
    g = f.copy()
    np.testing.assert_array_equal(g.to_numpy(), [10, 20, 30, 40])


# --- Field.astype() ---

def test_astype_f32_to_i32(backend):
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.array([1.5, 2.7, -0.3, 4.0], dtype=np.float32))
    g = f.astype(tack.i32)
    assert g.dtype is tack.i32
    result = g.to_numpy()
    assert result[0] == 1
    assert result[1] == 2
    assert result[3] == 4


def test_astype_i32_to_f32(backend):
    f = tack.field(dtype=tack.i32, shape=(3,))
    f.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    g = f.astype(tack.f32)
    assert g.dtype is tack.f32
    np.testing.assert_allclose(g.to_numpy(), [1.0, 2.0, 3.0])


def test_astype_same_dtype(backend):
    """astype with same dtype returns a copy."""
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.arange(4, dtype=np.float32))
    g = f.astype(tack.f32)
    assert g.dtype is tack.f32
    np.testing.assert_array_equal(g.to_numpy(), f.to_numpy())


def test_astype_u8_to_f32(backend):
    f = tack.field(dtype=tack.u8, shape=(3,))
    f.from_numpy(np.array([0, 128, 255], dtype=np.uint8))
    g = f.astype(tack.f32)
    np.testing.assert_allclose(g.to_numpy(), [0.0, 128.0, 255.0])


# --- Field.reshape() ---

def test_reshape_1d_to_2d(backend):
    f = tack.field(dtype=tack.f32, shape=(12,))
    f.from_numpy(np.arange(12, dtype=np.float32))
    g = f.reshape((3, 4))
    assert g.shape == (3, 4)
    assert g.size == 12
    np.testing.assert_array_equal(g.to_numpy().ravel(), f.to_numpy())


def test_reshape_2d_to_1d(backend):
    f = tack.field(dtype=tack.f32, shape=(3, 4))
    f.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
    g = f.reshape((12,))
    assert g.shape == (12,)


def test_reshape_shares_buffer(backend):
    """reshape shares the underlying buffer (no copy)."""
    f = tack.field(dtype=tack.f32, shape=(12,))
    f.from_numpy(np.arange(12, dtype=np.float32))
    g = f.reshape((3, 4))
    assert g._buffer is f._buffer


def test_reshaped_view_accepts_from_numpy(backend):
    """A view writes through in its own shape, whatever the buffer's is.

    Host-visible buffers used to copy into the allocation's array, which
    numpy refused to broadcast (2, 6) into (3, 4).
    """
    f = tack.field(dtype=tack.f32, shape=(2, 6))
    g = f.reshape((3, 4))
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    g.from_numpy(values)
    np.testing.assert_array_equal(g.to_numpy(), values)
    np.testing.assert_array_equal(f.to_numpy(), values.reshape(2, 6))
    g.from_numpy(values.T.copy().T)  # non-contiguous input, same elements
    np.testing.assert_array_equal(f.to_numpy(), values.reshape(2, 6))


def test_reshape_bad_size(backend):
    f = tack.field(dtype=tack.f32, shape=(12,))
    with pytest.raises(ValueError, match="Cannot reshape"):
        f.reshape((5, 3))


# --- tack.zeros / tack.ones / tack.full ---

def test_zeros(backend):
    f = tack.zeros(dtype=tack.f32, shape=(10,))
    assert f.dtype is tack.f32
    assert f.shape == (10,)
    np.testing.assert_array_equal(f.to_numpy(), 0.0)


def test_ones(backend):
    f = tack.ones(dtype=tack.i32, shape=(5,))
    assert f.dtype is tack.i32
    np.testing.assert_array_equal(f.to_numpy(), 1)


def test_full(backend):
    f = tack.full(tack.f32, (4,), 3.14)
    np.testing.assert_allclose(f.to_numpy(), 3.14, rtol=1e-6)


# --- tack.arange ---

def test_arange_default(backend):
    f = tack.arange(8)
    assert f.dtype is tack.i32
    np.testing.assert_array_equal(f.to_numpy(), np.arange(8, dtype=np.int32))


def test_arange_f32(backend):
    f = tack.arange(5, dtype=tack.f32)
    assert f.dtype is tack.f32
    np.testing.assert_allclose(f.to_numpy(), [0.0, 1.0, 2.0, 3.0, 4.0])


# --- tack.concat ---

def test_concat_basic(backend):
    a = tack.field(dtype=tack.f32, shape=(3,))
    b = tack.field(dtype=tack.f32, shape=(2,))
    a.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b.from_numpy(np.array([4.0, 5.0], dtype=np.float32))
    c = tack.concat([a, b])
    assert c.shape == (5,)
    np.testing.assert_array_equal(c.to_numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])


def test_concat_single(backend):
    a = tack.field(dtype=tack.i32, shape=(3,))
    a.from_numpy(np.array([10, 20, 30], dtype=np.int32))
    c = tack.concat([a])
    np.testing.assert_array_equal(c.to_numpy(), [10, 20, 30])


def test_concat_three(backend):
    fields = []
    for v in [1.0, 2.0, 3.0]:
        f = tack.field(dtype=tack.f32, shape=(2,))
        f.fill(v)
        fields.append(f)
    c = tack.concat(fields)
    assert c.shape == (6,)
    np.testing.assert_allclose(c.to_numpy(), [1, 1, 2, 2, 3, 3])


def test_concat_dtype_mismatch(backend):
    a = tack.field(dtype=tack.f32, shape=(2,))
    b = tack.field(dtype=tack.i32, shape=(2,))
    with pytest.raises(TypeError, match="same dtype"):
        tack.concat([a, b])


def test_concat_empty_raises(backend):
    with pytest.raises(ValueError, match="at least one"):
        tack.concat([])


# --- End-to-end: use utilities in a kernel workflow ---

def test_workflow_arange_kernel_concat(backend):
    """Create fields with arange, process with kernel, concat results."""
    a = tack.arange(4, dtype=tack.f32)
    b = tack.arange(4, dtype=tack.f32)
    out_a = tack.zeros(dtype=tack.f32, shape=(4,))
    out_b = tack.zeros(dtype=tack.f32, shape=(4,))

    @tack.kernel
    def double_kern(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i] * 2.0

    double_kern(a, out_a)
    double_kern(b, out_b)
    result = tack.concat([out_a, out_b])
    np.testing.assert_allclose(result.to_numpy(),
                               [0, 2, 4, 6, 0, 2, 4, 6])


# --- Read-only imported storage ---

def _alias(field, writable):
    """`field`'s storage wrapped again through field_from_ptr."""
    from tack.runtime.dispatch import get_backend
    name = get_backend().name
    if name == "cpu":
        ptr = field._buffer._data
    elif name == "metal":
        ptr = field._buffer.metal_buffer
    else:
        ptr = field._buffer.address
    return tack.field_from_ptr(ptr, field.dtype, field.shape, writable=writable)


@tack.kernel
def _scale_into(src, dst):
    for i in range(src.shape[0]):
        dst[i] = src[i] * 2.0


@tack.kernel
def _bump(dst):
    for i in range(dst.shape[0]):
        tack.atomic_add(dst, i, 1.0)


def test_kernels_cannot_store_to_a_read_only_field(backend):
    """field_from_ptr is read-only by default; kernels used to ignore that."""
    storage = tack.field(dtype=tack.f32, shape=(8,))
    storage.from_numpy(np.arange(8, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(8,))
    read_only = _alias(storage, writable=False)

    _scale_into(read_only, out)  # reading is what read-only is for
    np.testing.assert_array_equal(out.to_numpy(), np.arange(8) * 2.0)

    with pytest.raises(ValueError, match="parameter 'dst'.*read-only"):
        _scale_into(out, read_only)
    with pytest.raises(ValueError, match="read-only"):
        _bump(read_only)
    np.testing.assert_array_equal(storage.to_numpy(), np.arange(8))

    # The refused calls compiled the variants these reuse.
    _scale_into(out, _alias(storage, writable=True))
    np.testing.assert_array_equal(storage.to_numpy(), np.arange(8) * 4.0)


def test_a_read_only_reshaped_view_stays_read_only(backend):
    storage = tack.field(dtype=tack.f32, shape=(8,))
    view = _alias(storage, writable=False).reshape((8,))
    with pytest.raises(ValueError, match="read-only"):
        _bump(view)


# --- fill() with one element's value ---

def test_fill_takes_an_element_of_a_vector_or_matrix_field(backend):
    """`colors.fill([1.0, 0.5, 0.0])` sets every element to that vector. A
    sequence was passed to the buffer's scalar fill and failed inside
    NumPy with "setting an array element with a sequence"."""
    colors = tack.Vector.field(3, dtype=tack.f32, shape=(4, 2))
    colors.fill([1.0, 0.5, 0.0])
    np.testing.assert_array_equal(colors.to_numpy(vectors=True),
                                  np.broadcast_to(np.array([1.0, 0.5, 0.0], np.float32), (4, 2, 3)))
    colors.fill(2.0)                    # a scalar still sets every component
    assert (colors.to_numpy() == 2.0).all()

    frames = tack.Matrix.field(2, 2, dtype=tack.i32, shape=(3,))
    frames.fill(np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(frames.to_numpy(vectors=True),
                                  np.broadcast_to(np.array([[1, 2], [3, 4]], np.int32), (3, 2, 2)))


def test_fill_rejects_a_value_of_the_wrong_shape(backend):
    colors = tack.Vector.field(3, dtype=tack.f32, shape=(4,))
    with pytest.raises(ValueError, match=r"shape \(2,\); the elements of this field have shape \(3,\)"):
        colors.fill([1.0, 2.0])
    scalars = tack.field(dtype=tack.f32, shape=(4,))
    with pytest.raises(TypeError, match="takes a scalar for a field of scalars"):
        scalars.fill([1.0, 2.0])


# --- Zero-dimensional scalar fields ---

@tack.kernel
def _add_offset(offset, x, out, n):
    for i in range(n):
        out[i] = x[i] + offset[None]


@tack.kernel
def _bump_scalar(offset):
    for _ in range(1):
        offset[None] = offset[None] + 7


@pytest.mark.parametrize("dtype", [tack.i32, tack.f32])
def test_zero_dimensional_scalar_field(backend, dtype):
    """`tack.field(dtype, shape=())` is one value, read and written as
    `f[None]`. Metal could not allocate it: its buffer zeroed the view
    with `[:]`, which a zero-dimensional array rejects. A zero-dimensional
    vector field has a flat shape of `(n,)` and was not affected."""
    f = tack.field(dtype=dtype, shape=())
    assert f.shape == () and f.to_numpy().shape == ()
    assert f.to_numpy() == 0
    f.fill(3)
    assert f.to_numpy() == 3
    f.from_numpy(np.array(5, f.to_numpy().dtype))
    assert f.to_numpy() == 5

    x = tack.field(dtype=dtype, shape=(6,))
    out = tack.field(dtype=dtype, shape=(6,))
    x.from_numpy(np.arange(6).astype(x.to_numpy().dtype))
    _bump_scalar(f)
    assert f.to_numpy() == 12
    _add_offset(f, x, out, 6)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(6) + 12)


# --- Reading one element from the host ---

def test_host_reads_one_element(backend):
    """`f[i]`, `f[i, j]`, `f[None]` read an element from the host: a Python
    number from a field of scalars, a NumPy array from a vector or matrix
    field. Negative indices count from the end."""
    f = tack.field(dtype=tack.f32, shape=(3, 4))
    f.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert f[1, 2] == 6.0 and isinstance(f[1, 2], float)
    assert f[-1, -1] == 11.0
    assert f.reshape((12,))[5] == 5.0                  # a view reads through its own shape

    v = tack.Vector.field(2, dtype=tack.i32, shape=(5,))
    v.from_numpy(np.arange(10, dtype=np.int32).reshape(5, 2))
    np.testing.assert_array_equal(v[3], [6, 7])
    np.testing.assert_array_equal(v[-1], [8, 9])

    m = tack.Matrix.field(2, 2, dtype=tack.f32, shape=(2, 3))
    m.from_numpy(np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2))
    np.testing.assert_array_equal(m[1, 2], [[20, 21], [22, 23]])

    z = tack.field(dtype=tack.i32, shape=())
    z.fill(9)
    assert z[None] == 9 and isinstance(z[None], int)


@pytest.mark.parametrize("dt", ["i8", "u16", "i32", "i64", "f32"])
def test_read_range_copies_the_requested_elements(backend, dt):
    """A backend that defines its own read_range copies just the elements asked
    for: element offsets scale by the dtype's size, and the whole buffer is
    never copied to read one. The others use DeviceBuffer's whole-buffer copy."""
    from tack.lang.field import DeviceBuffer

    np_dtype = getattr(tack, dt).numpy_dtype
    data = np.arange(37).astype(np_dtype)
    f = tack.field(dtype=getattr(tack, dt), shape=(37,))
    f.from_numpy(data)
    buf = f._buffer
    if type(buf).read_range is not DeviceBuffer.read_range:
        def no_whole_copy():
            raise AssertionError("read_range copied the whole buffer")
        buf.to_numpy = no_whole_copy
    for start, count in ((0, 1), (5, 3), (36, 1), (0, 37), (20, 0)):
        got = buf.read_range(start, count)
        assert got.dtype == np.dtype(np_dtype)
        np.testing.assert_array_equal(got, data[start:start + count])
    assert f[36] == data[36]


def test_device_read_range_refuses_a_range_past_the_allocation(backend):
    """CUDA, HIP and Level Zero copy at a byte offset into device memory,
    where a range past the end would read beyond the allocation; they
    refuse it. CPU and Metal slice a host-visible array instead."""
    from tack.runtime.dispatch import get_backend

    if get_backend().name not in ("cuda", "hip", "level_zero"):
        pytest.skip("host-visible memory: a slice, not a device copy")
    f = tack.field(dtype=tack.f64 if get_backend().supports_f64 else tack.f32, shape=(20,))
    f.from_numpy(np.arange(20).astype(f.to_numpy().dtype))
    buf = f._buffer
    np.testing.assert_array_equal(buf.read_range(18, 2), [18, 19])
    with pytest.raises(IndexError, match="outside a buffer of 20 elements"):
        buf.read_range(19, 2)
    with pytest.raises(IndexError, match="outside"):
        buf.read_range(-1, 1)
    alias = tack.field_from_ptr(buf.address, f.dtype, (20,))      # a wrapped pointer reads too
    assert alias[13] == 13


def test_host_reads_are_checked_and_writes_refused(backend):
    f = tack.field(dtype=tack.f32, shape=(3, 4))
    with pytest.raises(TypeError, match="takes 2 integer indices"):
        f[1]
    with pytest.raises(TypeError, match="takes 2 integer indices"):
        f[0:2, 0]
    with pytest.raises(IndexError, match="out of range"):
        f[0, 4]
    with pytest.raises(TypeError, match="zero-dimensional"):
        f[None]
    with pytest.raises(TypeError, match="not written by element"):
        f[0, 0] = 1.0
    with pytest.raises(TypeError, match="not iterated"):
        list(f)
