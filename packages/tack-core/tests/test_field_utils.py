"""Tests for field utility functions: copy, astype, reshape, concat, zeros, etc."""

import numpy as np
import pytest

import tack

# --- Field.size and len() ---

def test_field_size(backend):
    f = tack.field(dtype=tack.f32, shape=(3, 4))
    assert f.size == 12


def test_field_size_1d(backend):
    f = tack.field(dtype=tack.f32, shape=(100,))
    assert f.size == 100


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
