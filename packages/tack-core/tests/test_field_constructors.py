"""Copying NumPy values and allocating matching storage are distinct operations."""

import numpy as np
import pytest

import tack
from tack.lang.field import Field

pytestmark = pytest.mark.quick


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_field_from_numpy_copies_independent_values(backend, shape):
    original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = original.copy()
    result = tack.field_from_numpy(original)
    assert result.shape == shape
    assert result.dtype is tack.f32
    original.fill(-1)
    np.testing.assert_array_equal(result.to_numpy(), expected)
    result.fill(7)
    np.testing.assert_array_equal(original, np.full(shape, -1, dtype=np.float32))


def test_field_from_numpy_converts_noncontiguous_input(backend):
    source = np.arange(24, dtype=np.int32).reshape(4, 6)[:, ::2]
    result = tack.field_from_numpy(source, dtype=tack.f32)
    assert result.dtype is tack.f32
    assert result.shape == source.shape
    np.testing.assert_array_equal(result.to_numpy(), source.astype(np.float32))


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
@pytest.mark.parametrize("use_field", [False, True], ids=["numpy", "field"])
def test_empty_like_matches_scalar_metadata(backend, shape, use_field):
    source = np.ones(shape, dtype=np.float32)
    prototype = tack.field_from_numpy(source) if use_field else source
    result = tack.empty_like(prototype)
    assert result.dtype is tack.f32
    assert result.shape == shape
    result.fill(7)
    np.testing.assert_array_equal(result.to_numpy(), np.full(shape, 7, dtype=np.float32))
    actual = prototype.to_numpy() if use_field else source
    np.testing.assert_array_equal(actual, np.ones(shape, dtype=np.float32))


def test_empty_like_uses_view_shape(backend):
    source = tack.arange(12).reshape((3, 4))
    result = tack.empty_like(source)
    assert result.shape == (3, 4)
    result.fill(9)
    np.testing.assert_array_equal(source.to_numpy(), np.arange(12).reshape(3, 4))


@pytest.mark.parametrize("use_field", [False, True], ids=["numpy", "field"])
def test_empty_like_dtype_override(backend, use_field):
    source = np.ones((2, 3), dtype=np.float32)
    prototype = tack.field_from_numpy(source) if use_field else source
    result = tack.empty_like(prototype, dtype=tack.i32)
    assert result.dtype is tack.i32
    assert result.shape == source.shape
    result.fill(5)
    np.testing.assert_array_equal(result.to_numpy(), np.full(source.shape, 5, dtype=np.int32))


@tack.kernel
def _double_elements(src, dst, n):
    for i in range(n):
        dst[i] = src[i] * 2


@pytest.mark.parametrize("shape", [(), (0,), (4,), (2, 3)])
@pytest.mark.parametrize("components", [(3,), (2, 3)], ids=["vector", "matrix"])
@pytest.mark.parametrize("dtype", [None, tack.i32], ids=["inferred", "override"])
def test_empty_like_preserves_component_fields_in_kernels(backend, shape, components, dtype):
    if len(components) == 1:
        source = tack.Vector.field(*components, dtype=tack.f32, shape=shape)
    else:
        source = tack.Matrix.field(*components, dtype=tack.f32, shape=shape)
    source.fill(3)
    # Each kernel access must still read/write a vector or matrix value.
    output = tack.empty_like(source, dtype=dtype)
    assert output.dtype is (dtype or source.dtype)
    assert output.shape == source.shape
    logical_count = int(np.prod(shape))
    if len(shape) > 1:
        # Use multidimensional indexing for a multidimensional logical field.
        @tack.kernel
        def double_grid(src, dst):
            for row, column in tack.ndrange(src.shape[0], src.shape[1]):
                dst[row, column] = src[row, column] * 2

        double_grid(source, output)
    elif not shape:
        @tack.kernel
        def double_scalar(src, dst):
            for _ in range(1):
                dst[None] = src[None] * 2

        double_scalar(source, output)
    else:
        _double_elements(source, output, logical_count)
    values = output.to_numpy(vectors=True)
    assert values.shape == (*shape, *components)
    np.testing.assert_array_equal(values, np.full(values.shape, 6, output.dtype.numpy_dtype))
    np.testing.assert_array_equal(source.to_numpy(vectors=True),
                                  np.full(values.shape, 3, dtype=np.float32))


def test_empty_like_reads_only_metadata_and_does_not_initialize(backend, monkeypatch):
    source = tack.ones(tack.f32, (4,))

    def refuse(*args, **kwargs):
        raise AssertionError("empty_like must not read, copy or fill values")

    with monkeypatch.context() as patch:
        patch.setattr(Field, "from_numpy", refuse)
        patch.setattr(Field, "to_numpy", refuse)
        patch.setattr(Field, "fill", refuse)
        result = tack.empty_like(source)
        numpy_result = tack.empty_like(np.ones(4, dtype=np.float32))
    result.fill(2)
    numpy_result.fill(3)
    np.testing.assert_array_equal(source.to_numpy(), np.ones(4, dtype=np.float32))


def test_empty_like_of_read_only_storage_is_writable(backend):
    # A NumPy prototype may be read-only; the new field owns its storage.
    source = np.ones(4, dtype=np.float32)
    source.flags.writeable = False
    result = tack.empty_like(source)
    result.fill(2)
    np.testing.assert_array_equal(result.to_numpy(), np.full(4, 2, dtype=np.float32))
    np.testing.assert_array_equal(source, np.ones(4, dtype=np.float32))


@pytest.mark.parametrize("constructor", [tack.field_from_numpy, tack.empty_like])
def test_numpy_dtype_inference_requires_a_supported_type(backend, constructor):
    source = np.ones(4, dtype=np.float16)
    with pytest.raises(TypeError, match="Unsupported numpy dtype: float16"):
        constructor(source)
    result = constructor(source, dtype=tack.f32)
    assert result.dtype is tack.f32


def test_empty_like_requires_a_field_or_numpy_array(backend):
    with pytest.raises(TypeError, match="Tack field or NumPy array"):
        tack.empty_like([1, 2, 3])
