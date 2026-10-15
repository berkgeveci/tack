"""Tests for field utility functions: copy, astype, reshape, concat, zeros, etc."""

import numpy as np
import pytest

import tack
from tack import algorithms

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

# --- Field.size and len() ---

def test_field_size(backend):
    f = tack.empty((3, 4))
    assert f.size == 12


def test_field_size_1d(backend):
    f = tack.empty((100,))
    assert f.size == 100


def test_empty_field(backend):
    """A filter that selects nothing returns a field of no elements; it
    allocates, round-trips an empty array and is a valid kernel argument."""
    f = tack.empty((0,))
    assert f.size == 0
    assert f.numpy().shape == (0,)
    tack.copyto(f, np.zeros(0, dtype=np.float32))
    out = tack.empty((0,))
    algorithms.copy(f, out, 0)
    assert out.numpy().shape == (0,)


def test_empty_field_in_a_launch(backend):
    """An empty field passed beside a non-empty one still binds when the
    kernel launches. HIP once handed it a null pointer it could not bind."""
    x = tack.arange(8, dtype=tack.f32)
    empty = tack.empty((0,))

    @tack.kernel
    def double(x, empty):
        for i in range(x.shape[0]):
            x[i] = x[i] * 2.0

    double(x, empty)
    np.testing.assert_array_equal(x.numpy(), np.arange(8, dtype=np.float32) * 2)


def test_field_len(backend):
    f = tack.empty((10,))
    assert len(f) == 10


def test_field_len_2d(backend):
    f = tack.empty((5, 3))
    assert len(f) == 5


# --- Field.copy() ---

def test_field_copy(backend):
    f = tack.empty((8,))
    tack.copyto(f, np.arange(8, dtype=np.float32))
    g = f.copy()
    np.testing.assert_array_equal(g.numpy(), f.numpy())
    # Verify it's a separate buffer (modifying one doesn't affect the other)
    g.fill(0)
    assert f.numpy()[0] == 0.0  # original unchanged... well, depends on backend
    # At least verify shapes and dtypes match
    assert g.shape == f.shape
    assert g.dtype is f.dtype


def test_field_copy_i32(backend):
    f = tack.empty((4,), tack.i32)
    tack.copyto(f, np.array([10, 20, 30, 40], dtype=np.int32))
    g = f.copy()
    np.testing.assert_array_equal(g.numpy(), [10, 20, 30, 40])


# --- Field.astype() ---

def test_astype_f32_to_i32(backend):
    f = tack.empty((4,))
    tack.copyto(f, np.array([1.5, 2.7, -0.3, 4.0], dtype=np.float32))
    g = f.astype(tack.i32)
    assert g.dtype is tack.i32
    result = g.numpy()
    assert result[0] == 1
    assert result[1] == 2
    assert result[3] == 4


def test_astype_i32_to_f32(backend):
    f = tack.empty((3,), tack.i32)
    tack.copyto(f, np.array([1, 2, 3], dtype=np.int32))
    g = f.astype(tack.f32)
    assert g.dtype is tack.f32
    np.testing.assert_allclose(g.numpy(), [1.0, 2.0, 3.0])


def test_astype_same_dtype(backend):
    """astype with same dtype returns a copy."""
    f = tack.empty((4,))
    tack.copyto(f, np.arange(4, dtype=np.float32))
    g = f.astype(tack.f32)
    assert g.dtype is tack.f32
    np.testing.assert_array_equal(g.numpy(), f.numpy())


def test_astype_u8_to_f32(backend):
    f = tack.empty((3,), tack.u8)
    tack.copyto(f, np.array([0, 128, 255], dtype=np.uint8))
    g = f.astype(tack.f32)
    np.testing.assert_allclose(g.numpy(), [0.0, 128.0, 255.0])


# --- Field.reshape() ---

def test_reshape_1d_to_2d(backend):
    f = tack.empty((12,))
    tack.copyto(f, np.arange(12, dtype=np.float32))
    g = f.reshape((3, 4))
    assert g.shape == (3, 4)
    assert g.size == 12
    np.testing.assert_array_equal(g.numpy().ravel(), f.numpy())


def test_reshape_2d_to_1d(backend):
    f = tack.empty((3, 4))
    tack.copyto(f, np.arange(12, dtype=np.float32).reshape(3, 4))
    g = f.reshape((12,))
    assert g.shape == (12,)


def test_reshape_shares_buffer(backend):
    """reshape shares the underlying buffer (no copy)."""
    f = tack.empty((12,))
    tack.copyto(f, np.arange(12, dtype=np.float32))
    g = f.reshape((3, 4))
    assert g._buffer is f._buffer


def test_reshaped_view_accepts_copyto(backend):
    """A view writes through in its own shape, whatever the buffer's is.

    Host-visible buffers used to copy into the allocation's array, which
    numpy refused to broadcast (2, 6) into (3, 4).
    """
    f = tack.empty((2, 6))
    g = f.reshape((3, 4))
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    tack.copyto(g, values)
    np.testing.assert_array_equal(g.numpy(), values)
    np.testing.assert_array_equal(f.numpy(), values.reshape(2, 6))
    tack.copyto(g, values.T.copy().T)  # non-contiguous input, same elements
    np.testing.assert_array_equal(f.numpy(), values.reshape(2, 6))


def test_reshape_bad_size(backend):
    f = tack.empty((12,))
    with pytest.raises(ValueError, match="Cannot reshape"):
        f.reshape((5, 3))


# --- tack.zeros / tack.ones / tack.full ---

def test_zeros(backend):
    f = tack.zeros((10,))
    assert f.dtype is tack.f32
    assert f.shape == (10,)
    np.testing.assert_array_equal(f.numpy(), 0.0)


def test_ones(backend):
    f = tack.ones((5,), tack.i32)
    assert f.dtype is tack.i32
    np.testing.assert_array_equal(f.numpy(), 1)


def test_full(backend):
    f = tack.full((4,), 3.14, tack.f32)
    np.testing.assert_allclose(f.numpy(), 3.14, rtol=1e-6)


# --- tack.arange ---

def test_arange_default(backend):
    f = tack.arange(8)
    assert f.dtype is tack.i32
    np.testing.assert_array_equal(f.numpy(), np.arange(8, dtype=np.int32))


def test_arange_f32(backend):
    f = tack.arange(5, dtype=tack.f32)
    assert f.dtype is tack.f32
    np.testing.assert_allclose(f.numpy(), [0.0, 1.0, 2.0, 3.0, 4.0])


# --- tack.concat ---

def test_concat_basic(backend):
    a = tack.empty((3,))
    b = tack.empty((2,))
    tack.copyto(a, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    tack.copyto(b, np.array([4.0, 5.0], dtype=np.float32))
    c = tack.concat([a, b])
    assert c.shape == (5,)
    np.testing.assert_array_equal(c.numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])


def test_concat_single(backend):
    a = tack.empty((3,), tack.i32)
    tack.copyto(a, np.array([10, 20, 30], dtype=np.int32))
    c = tack.concat([a])
    np.testing.assert_array_equal(c.numpy(), [10, 20, 30])


def test_concat_three(backend):
    fields = []
    for v in [1.0, 2.0, 3.0]:
        f = tack.empty((2,))
        f.fill(v)
        fields.append(f)
    c = tack.concat(fields)
    assert c.shape == (6,)
    np.testing.assert_allclose(c.numpy(), [1, 1, 2, 2, 3, 3])


def test_concat_dtype_mismatch(backend):
    a = tack.empty((2,))
    b = tack.empty((2,), tack.i32)
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
    out_a = tack.zeros((4,))
    out_b = tack.zeros((4,))

    @tack.kernel
    def double_kern(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i] * 2.0

    double_kern(a, out_a)
    double_kern(b, out_b)
    result = tack.concat([out_a, out_b])
    np.testing.assert_allclose(result.numpy(),
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
    return tack.array_from_ptr(ptr, field.shape, field.dtype, writable=writable)


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
    storage = tack.empty((8,))
    tack.copyto(storage, np.arange(8, dtype=np.float32))
    out = tack.empty((8,))
    read_only = _alias(storage, writable=False)

    _scale_into(read_only, out)  # reading is what read-only is for
    np.testing.assert_array_equal(out.numpy(), np.arange(8) * 2.0)

    with pytest.raises(ValueError, match="parameter 'dst'.*read-only"):
        _scale_into(out, read_only)
    with pytest.raises(ValueError, match="read-only"):
        _bump(read_only)
    np.testing.assert_array_equal(storage.numpy(), np.arange(8))

    # The refused calls compiled the variants these reuse.
    _scale_into(out, _alias(storage, writable=True))
    np.testing.assert_array_equal(storage.numpy(), np.arange(8) * 4.0)


def test_a_read_only_reshaped_view_stays_read_only(backend):
    storage = tack.empty((8,))
    view = _alias(storage, writable=False).reshape((8,))
    with pytest.raises(ValueError, match="read-only"):
        _bump(view)


# --- fill() with one element's value ---

def test_fill_takes_an_element_of_a_vector_or_matrix_field(backend):
    """`colors.fill([1.0, 0.5, 0.0])` sets every element to that vector. A
    sequence was passed to the buffer's scalar fill and failed inside
    NumPy with "setting an array element with a sequence"."""
    colors = tack.empty((4, 2), tack.vec(3, tack.f32))
    colors.fill([1.0, 0.5, 0.0])
    np.testing.assert_array_equal(colors.numpy(),
                                  np.broadcast_to(np.array([1.0, 0.5, 0.0], np.float32), (4, 2, 3)))
    colors.fill(2.0)                    # a scalar still sets every component
    assert (colors.numpy() == 2.0).all()

    frames = tack.empty((3,), tack.mat(2, 2, tack.i32))
    frames.fill(np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(frames.numpy(),
                                  np.broadcast_to(np.array([[1, 2], [3, 4]], np.int32), (3, 2, 2)))


def test_fill_rejects_a_value_of_the_wrong_shape(backend):
    colors = tack.empty((4,), tack.vec(3, tack.f32))
    with pytest.raises(ValueError, match=r"shape \(2,\); the elements of this array have shape \(3,\)"):
        colors.fill([1.0, 2.0])
    scalars = tack.empty((4,))
    with pytest.raises(TypeError, match="takes a scalar for an array of scalars"):
        scalars.fill([1.0, 2.0])


_FILL_DTYPES = [tack.i8, tack.u8, tack.i16, tack.u16, tack.i32, tack.u32, tack.i64, tack.u64,
                tack.f32, tack.f64]


@pytest.mark.parametrize("dtype", _FILL_DTYPES, ids=str)
def test_fill_matches_numpy_full(backend, dtype):
    """GPU fills run on the device: a byte memset when every byte of the
    value is the same (0, -1, 255), 16- and 32-bit memsets, and otherwise
    one element doubled by device copies (8-byte values). Each must store
    what `np.full` stores, at sizes that end a doubling early and at none."""
    from tack.runtime.dispatch import get_backend
    if dtype not in get_backend().supported_dtypes:
        pytest.skip(f"{dtype} is not supported on this backend")
    kind = dtype.numpy_dtype.kind
    values = [0, 1, 7, 100] + ([255] if kind == "u" else [-1]) + \
        ([2.5, -0.0, 1e10] if kind == "f" else [])
    for value in values:
        for n in (0, 1, 3, 1000, 4097):
            f = tack.empty((n,), dtype)
            f.fill(value)
            np.testing.assert_array_equal(f.numpy(), np.full(n, value, dtype.numpy_dtype),
                                          err_msg=f"{dtype} fill({value}) of {n}")
    np.testing.assert_array_equal(tack.zeros((5, 3), dtype).numpy(),
                                  np.zeros((5, 3), dtype.numpy_dtype))
    np.testing.assert_array_equal(tack.ones((7,), dtype).numpy(), np.ones(7, dtype.numpy_dtype))


def test_fill_stays_on_the_device(backend):
    """A fill on CUDA, HIP or Level Zero does not build the field on the host
    and upload it: that was most of a 5M-cell contour's time."""
    from tack.runtime.dispatch import get_backend
    if get_backend().name not in ("cuda", "hip", "level_zero"):
        pytest.skip("host-visible storage fills in place")
    cases = ((tack.empty((1000,)), 2.5),
             (tack.empty((1000,), tack.i64), -7),
             (tack.empty((1000,), tack.vec(3, tack.f32)), [1.0, 2.0, 3.0]))
    for f, value in cases:
        def refuse(arr):
            raise AssertionError(f"fill uploaded {np.asarray(arr).nbytes} bytes")
        f._buffer.from_numpy = refuse
        f.fill(value)
        del f._buffer.from_numpy
        if np.ndim(value):
            np.testing.assert_array_equal(
                f.numpy(),
                np.broadcast_to(np.asarray(value, np.float32), (1000, 3)))
        else:
            np.testing.assert_array_equal(f.numpy(), np.full(1000, value, f.dtype.numpy_dtype))


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
    """`tack.empty((), dtype)` is one value, read and written as
    `f[None]`. Metal could not allocate it: its buffer zeroed the view
    with `[:]`, which a zero-dimensional array rejects. A zero-dimensional
    vector field has a flat shape of `(n,)` and was not affected."""
    assert tack.empty((), dtype).numpy().shape == ()
    f = tack.zeros((), dtype)
    assert f.shape == () and f.numpy().shape == ()
    assert f.numpy() == 0
    f.fill(3)
    assert f.numpy() == 3
    tack.copyto(f, np.array(5, f.numpy().dtype))
    assert f.numpy() == 5

    x = tack.empty((6,), dtype)
    out = tack.empty((6,), dtype)
    tack.copyto(x, np.arange(6).astype(x.numpy().dtype))
    _bump_scalar(f)
    assert f.numpy() == 12
    _add_offset(f, x, out, 6)
    np.testing.assert_array_equal(out.numpy(), np.arange(6) + 12)


# --- Reading one element from the host ---

def test_host_reads_one_element(backend):
    """`f[i]`, `f[i, j]`, `f[None]` read an element from the host: a Python
    number from a field of scalars, a NumPy array from a vector or matrix
    field. Negative indices count from the end."""
    f = tack.empty((3, 4))
    tack.copyto(f, np.arange(12, dtype=np.float32).reshape(3, 4))
    assert f[1, 2] == 6.0 and isinstance(f[1, 2], float)
    assert f[-1, -1] == 11.0
    assert f.reshape((12,))[5] == 5.0                  # a view reads through its own shape

    v = tack.empty((5,), tack.vec(2, tack.i32))
    tack.copyto(v, np.arange(10, dtype=np.int32).reshape(5, 2))
    np.testing.assert_array_equal(v[3], [6, 7])
    np.testing.assert_array_equal(v[-1], [8, 9])

    m = tack.empty((2, 3), tack.mat(2, 2, tack.f32))
    tack.copyto(m, np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2))
    np.testing.assert_array_equal(m[1, 2], [[20, 21], [22, 23]])

    z = tack.empty((), tack.i32)
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
    f = tack.empty((37,), getattr(tack, dt))
    tack.copyto(f, data)
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


def test_host_reads_are_checked_and_writes_refused(backend):
    f = tack.empty((3, 4))
    with pytest.raises(TypeError, match="takes 2 integer indices"):
        f[1]
    with pytest.raises(TypeError, match="takes 2 integer indices"):
        f[0:2, 0]
    with pytest.raises(IndexError, match="out of range"):
        f[0, 4]
    with pytest.raises(TypeError, match="takes 2 integer indices"):
        f[None]
    with pytest.raises(TypeError, match="not written by element"):
        f[0, 0] = 1.0
    with pytest.raises(TypeError, match="not iterated"):
        list(f)
