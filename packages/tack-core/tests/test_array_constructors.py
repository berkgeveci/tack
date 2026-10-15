"""The NumPy-style constructors, and copying versus allocating matching storage."""

import numpy as np
import pytest

import tack

pytestmark = pytest.mark.quick


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_array_copies_independent_values(backend, shape):
    original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = original.copy()
    result = tack.array(original)
    assert result.shape == shape
    assert result.dtype is tack.f32
    original.fill(-1)
    np.testing.assert_array_equal(result.numpy(), expected)
    result.fill(7)
    np.testing.assert_array_equal(original, np.full(shape, -1, dtype=np.float32))


def test_array_converts_noncontiguous_input(backend):
    source = np.arange(24, dtype=np.int32).reshape(4, 6)[:, ::2]
    result = tack.array(source, tack.f32)
    assert result.dtype is tack.f32
    assert result.shape == source.shape
    np.testing.assert_array_equal(result.numpy(), source.astype(np.float32))


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
@pytest.mark.parametrize("use_array", [False, True], ids=["numpy", "array"])
def test_empty_like_matches_scalar_metadata(backend, shape, use_array):
    source = np.ones(shape, dtype=np.float32)
    prototype = tack.array(source) if use_array else source
    result = tack.empty_like(prototype)
    assert result.dtype is tack.f32
    assert result.shape == shape
    result.fill(7)
    np.testing.assert_array_equal(result.numpy(), np.full(shape, 7, dtype=np.float32))
    actual = prototype.numpy() if use_array else source
    np.testing.assert_array_equal(actual, np.ones(shape, dtype=np.float32))


def test_empty_like_uses_view_shape(backend):
    source = tack.arange(12).reshape((3, 4))
    result = tack.empty_like(source)
    assert result.shape == (3, 4)
    result.fill(9)
    np.testing.assert_array_equal(source.numpy(), np.arange(12).reshape(3, 4))


@pytest.mark.parametrize("use_array", [False, True], ids=["numpy", "array"])
def test_empty_like_dtype_override(backend, use_array):
    source = np.ones((2, 3), dtype=np.float32)
    prototype = tack.array(source) if use_array else source
    result = tack.empty_like(prototype, dtype=tack.i32)
    assert result.dtype is tack.i32
    assert result.shape == source.shape
    result.fill(5)
    np.testing.assert_array_equal(result.numpy(), np.full(source.shape, 5, dtype=np.int32))


@tack.kernel
def _double_elements(src, dst, n):
    for i in range(n):
        dst[i] = src[i] * 2


@pytest.mark.parametrize("shape", [(), (0,), (4,), (2, 3)])
@pytest.mark.parametrize("components", [(3,), (2, 3)], ids=["vector", "matrix"])
@pytest.mark.parametrize("dtype", [None, tack.i32], ids=["inferred", "override"])
def test_empty_like_preserves_component_arrays_in_kernels(backend, shape, components, dtype):
    source = tack.empty(shape, tack.vec(*components) if len(components) == 1
                        else tack.mat(*components))
    source.fill(3)
    # Each kernel access must still read/write a vector or matrix value.
    output = tack.empty_like(source, dtype=dtype)
    assert output.scalar_dtype is (dtype or tack.f32)
    assert output.dtype.shape == components
    assert output.shape == source.shape
    logical_count = int(np.prod(shape))
    if len(shape) > 1:
        # Use multidimensional indexing for a multidimensional logical array.
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
    values = output.numpy()
    assert values.shape == (*shape, *components)
    np.testing.assert_array_equal(values, np.full(values.shape, 6, output.scalar_dtype.numpy_dtype))
    np.testing.assert_array_equal(source.numpy(),
                                  np.full(values.shape, 3, dtype=np.float32))


def test_empty_like_reads_only_metadata_and_does_not_initialize(backend, monkeypatch):
    source = tack.ones((4,))

    def refuse(*args, **kwargs):
        raise AssertionError("empty_like must not read, copy or fill values")

    with monkeypatch.context() as patch:
        patch.setattr(tack.Array, "_upload", refuse)
        patch.setattr(tack.Array, "numpy", refuse)
        patch.setattr(tack.Array, "fill", refuse)
        result = tack.empty_like(source)
        numpy_result = tack.empty_like(np.ones(4, dtype=np.float32))
    result.fill(2)
    numpy_result.fill(3)
    np.testing.assert_array_equal(source.numpy(), np.ones(4, dtype=np.float32))


def test_empty_like_of_read_only_storage_is_writable(backend):
    # A NumPy prototype may be read-only; the new array owns its storage.
    source = np.ones(4, dtype=np.float32)
    source.flags.writeable = False
    result = tack.empty_like(source)
    result.fill(2)
    np.testing.assert_array_equal(result.numpy(), np.full(4, 2, dtype=np.float32))
    np.testing.assert_array_equal(source, np.ones(4, dtype=np.float32))


@pytest.mark.parametrize("constructor", [tack.array, tack.empty_like])
def test_numpy_dtype_inference_requires_a_supported_type(backend, constructor):
    source = np.ones(4, dtype=np.float16)
    with pytest.raises(TypeError, match="Unsupported numpy dtype: float16"):
        constructor(source)
    result = constructor(source, dtype=tack.f32)
    assert result.dtype is tack.f32


def test_empty_like_requires_a_array_or_numpy_array(backend):
    with pytest.raises(TypeError, match="Tack array or NumPy array"):
        tack.empty_like([1, 2, 3])


# ── NumPy's argument order and defaults ─────────────────────────────

def test_constructors_take_the_shape_first(backend):
    assert tack.empty((3,)).dtype is tack.f32
    np.testing.assert_array_equal(tack.zeros((2, 3)).numpy(), np.zeros((2, 3), np.float32))
    np.testing.assert_array_equal(tack.ones(4, tack.i32).numpy(), np.ones(4, np.int32))
    full = tack.full((3,), 2.5)
    assert full.dtype is tack.f32
    np.testing.assert_array_equal(full.numpy(), [2.5, 2.5, 2.5])
    assert tack.full((2,), 7).dtype is tack.i32
    assert tack.full((2,), 2**40).dtype is tack.i64
    vectors = tack.full((2,), [1, 2, 3], tack.vec3())
    np.testing.assert_array_equal(vectors.numpy(), [[1, 2, 3], [1, 2, 3]])


@pytest.mark.parametrize("dtype", [np.float32, "float32", np.dtype("float32"), tack.f32])
def test_numpy_dtypes_are_accepted(backend, dtype):
    assert tack.empty((2,), dtype).dtype is tack.f32
    assert tack.zeros((2,), dtype).dtype is tack.f32
    assert tack.array([1, 2], dtype).dtype is tack.f32
    assert tack.arange(3, dtype=dtype).dtype is tack.f32
    assert tack.ones((2,)).astype(dtype).dtype is tack.f32


def test_python_values_default_to_f32_and_i32_numpy_arrays_keep_theirs(backend):
    assert tack.array([1.0, 2.0]).dtype is tack.f32
    assert tack.array([1, 2]).dtype is tack.i32
    assert tack.array([1, 2**40]).dtype is tack.i64
    assert tack.array(np.zeros(3)).dtype is tack.f64
    assert tack.array(np.zeros(3, np.int64)).dtype is tack.i64
    with pytest.raises(TypeError, match="boolean"):
        tack.array([True, False])


def test_arange_follows_numpy(backend):
    np.testing.assert_array_equal(tack.arange(5).numpy(), np.arange(5))
    np.testing.assert_array_equal(tack.arange(2, 10, 3).numpy(), [2, 5, 8])
    assert tack.arange(5).dtype is tack.i32
    floats = tack.arange(0, 1, 0.25)
    assert floats.dtype is tack.f32
    np.testing.assert_array_equal(floats.numpy(), [0, 0.25, 0.5, 0.75])
    assert tack.arange(0).shape == (0,)


@pytest.mark.parametrize("make, value", [
    (tack.zeros_like, 0), (tack.ones_like, 1), (lambda p: tack.full_like(p, 4), 4),
])
def test_like_constructors(backend, make, value):
    prototype = tack.empty((2, 3), tack.vec2())
    result = make(prototype)
    assert result.shape == (2, 3) and result.dtype is tack.vec2()
    np.testing.assert_array_equal(result.numpy(), np.full((2, 3, 2), value, np.float32))
    from_host = make(np.empty((4,), np.int32))
    assert from_host.dtype is tack.i32 and from_host.shape == (4,)


def test_asarray_returns_the_same_array_when_nothing_changes(backend):
    values = tack.arange(4)
    assert tack.asarray(values) is values
    assert tack.asarray(values, tack.i32) is values
    converted = tack.asarray(values, tack.f32)
    assert converted is not values and converted.dtype is tack.f32
    host = tack.asarray(np.arange(3, dtype=np.float32))
    assert isinstance(host, tack.Array) and host.dtype is tack.f32


def test_copyto_writes_host_values_and_broadcasts(backend):
    values = tack.empty((2, 3))
    tack.copyto(values, np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(values.numpy(), np.arange(6).reshape(2, 3))
    tack.copyto(values, [1, 2, 3])                       # broadcast over rows
    np.testing.assert_array_equal(values.numpy(), [[1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError, match="cannot broadcast"):
        tack.copyto(values, np.zeros(6))
    vectors = tack.empty((4,), tack.vec3())
    tack.copyto(vectors, np.ones((4, 3)))
    np.testing.assert_array_equal(vectors.numpy(), np.ones((4, 3)))
    with pytest.raises(ValueError, match="cannot broadcast"):
        tack.copyto(vectors, np.ones(12))                # values are (4, 3), not flat


def test_copyto_between_arrays_copies_on_the_device(backend):
    source = tack.arange(5, dtype=tack.f32)
    target = tack.zeros((5,))
    tack.copyto(target, source)
    np.testing.assert_array_equal(target.numpy(), np.arange(5))
    with pytest.raises(ValueError, match="shape"):
        tack.copyto(tack.zeros((4,)), source)


def test_numpy_reads_through_the_array_protocol(backend):
    values = tack.array(np.arange(6, dtype=np.float32).reshape(2, 3))
    np.testing.assert_array_equal(np.asarray(values), np.arange(6).reshape(2, 3))
    assert np.asarray(values, dtype=np.float64).dtype == np.float64
    with pytest.raises(ValueError, match="without a copy"):
        np.asarray(values, copy=False)


def test_reshape_takes_numpy_forms(backend):
    values = tack.arange(12)
    assert values.reshape(3, 4).shape == (3, 4)
    assert values.reshape((3, 4)).shape == (3, 4)
    assert values.reshape(-1, 6).shape == (2, 6)
    with pytest.raises(ValueError):
        values.reshape(5, -1)
    assert values.ndim == 1 and values.nbytes == 48


def test_host_writes_by_index_are_refused(backend):
    values = tack.zeros((3,))
    with pytest.raises(TypeError, match="copyto"):
        values[0] = 1
