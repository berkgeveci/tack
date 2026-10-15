"""Element types describe values; array shapes describe how many values."""

import asyncio
import copy
import pickle

import numpy as np
import pytest

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick


@pytest.mark.parametrize('shape', [(), (0,), (5,), (2, 3)])
@pytest.mark.parametrize('dtype, components', [
    (tack.f32, ()), (tack.vec3(tack.f32), (3,)), (tack.mat(2, 3, tack.f32), (2, 3)),
])
def test_array_element_shape_and_round_trip(backend, shape, dtype, components):
    host_shape = (*shape, *components)
    host = np.arange(np.prod(host_shape), dtype=np.float32).reshape(host_shape)
    values = tack.array(host, dtype=dtype)
    assert isinstance(values, tack.Array)
    assert values.dtype is dtype
    assert values.shape == shape
    assert values.size == int(np.prod(shape))
    if shape:
        assert len(values) == shape[0]
    else:
        with pytest.raises(TypeError, match='zero-dimensional'):
            len(values)
    assert values.ndim == len(shape)
    assert values.scalar_dtype is tack.f32
    np.testing.assert_array_equal(values.numpy(), host)
    np.testing.assert_array_equal(asyncio.run(values.numpy_async()), host)
    host.fill(-10)
    assert not np.any(values.numpy() == -10)


@tack.kernel
def _twice(src, dst):
    for i in range(src.shape[0]):
        dst[i] = src[i] * 2


@pytest.mark.parametrize('dtype, element', [
    (tack.f32, 2), (tack.vec3(), [1, 2, 3]), (tack.mat2(), [[1, 2], [3, 4]]),
])
def test_array_allocations_copy_conversion_and_concat(backend, dtype, element):
    source = tack.empty((5,), dtype)
    source.fill(element)
    result = tack.empty_like(source)
    _twice(source, result)
    expected = np.broadcast_to(element, result.numpy().shape)
    np.testing.assert_array_equal(result.numpy(), expected * 2)
    assert result.dtype is dtype
    for copied in (source.copy(), tack.array(source)):
        assert copied.dtype is dtype and copied.shape == source.shape
        copied.fill(9)
        np.testing.assert_array_equal(source.numpy(), expected)
    converted = source.astype(tack.i32)
    assert converted.scalar_dtype is tack.i32
    assert converted.shape == source.shape
    np.testing.assert_array_equal(converted.numpy(), expected)
    joined = tack.concat([source, result])
    assert joined.shape == (10,) and joined.dtype is dtype
    np.testing.assert_array_equal(joined.numpy(), np.concatenate([expected, expected * 2]))
    for make, fill in ((tack.zeros, 0), (tack.ones, 1)):
        initialized = make(shape=(5,), dtype=dtype)
        np.testing.assert_array_equal(initialized.numpy(), np.full(expected.shape, fill))
    np.testing.assert_array_equal(tack.full((5,), element, dtype).numpy(), expected)


def test_typed_matrix_operations_and_multidimensional_views(backend):
    matrices = tack.empty((2, 3), tack.mat2())
    matrices.fill([[1, 2], [3, 4]])
    vectors = tack.empty((2, 3), tack.vec2())
    vectors.fill([1, 2])
    out = tack.empty_like(vectors)

    @tack.kernel
    def multiply(matrices, vectors, out):
        for i, j in tack.ndindex(out.shape[0], out.shape[1]):
            out[i, j] = matrices[i, j] @ vectors[i, j]

    multiply(matrices, vectors, out)
    np.testing.assert_array_equal(out.numpy(), np.broadcast_to([5, 11], (2, 3, 2)))
    flat = out.reshape((6,))
    assert flat.dtype is tack.vec2() and flat._buffer is out._buffer
    tack.copyto(flat, np.arange(12, dtype=np.float32).reshape(6, 2))
    np.testing.assert_array_equal(out.numpy(), np.arange(12).reshape(2, 3, 2))
    np.testing.assert_array_equal(out[1, -1], [10, 11])
    with pytest.raises(ValueError, match='Cannot reshape'):
        out.reshape((12,))
    with pytest.raises(TypeError, match='component dimensions'):
        out.astype(tack.vec3())


def test_zero_dimensional_vector_in_kernel(backend):
    source = tack.array(np.array([1, 2, 3], dtype=np.float32), dtype=tack.vec3())
    out = tack.empty_like(source)

    @tack.kernel
    def double_one(source, out):
        for _ in range(1):
            out[()] = source[()] * 2

    double_one(source, out)
    assert out.shape == () and out.size == 1
    np.testing.assert_array_equal(out[()], [2, 4, 6])


def test_empty_like_reads_only_metadata(backend, monkeypatch):
    prototype = tack.empty((4,), tack.vec3())

    def refuse(*args, **kwargs):
        raise AssertionError('empty_like must not read, copy or initialize values')

    with monkeypatch.context() as patch:
        for method in ('fill', '_upload', 'numpy', 'copy'):
            patch.setattr(tack.Array, method, refuse)
        result = tack.empty_like(prototype, dtype=tack.i32)
    assert result.shape == (4,) and result.dtype is tack.vec3(tack.i32)
    result.fill(7)
    np.testing.assert_array_equal(result.numpy(), np.full((4, 3), 7))
    changed = tack.empty_like(prototype, dtype=tack.mat2())
    assert changed.shape == (4,) and changed.dtype is tack.mat2()


def test_numpy_dimensions_are_not_guessed_as_vectors(backend):
    host = np.ones((4, 3), dtype=np.float32)
    assert tack.array(host).shape == (4, 3)
    assert tack.array(host).dtype is tack.f32
    assert tack.empty_like(host).shape == (4, 3)
    typed = tack.array(host.T.copy().T, dtype=tack.vec3(tack.i32))
    assert typed.shape == (4,) and typed.dtype is tack.vec3(tack.i32)
    np.testing.assert_array_equal(typed.numpy(), host)
    assert tack.array([1, 2, 3], dtype=tack.i32).shape == (3,)
    with pytest.raises(ValueError, match='must end with element shape'):
        tack.array(host, dtype=tack.mat2())
    assert tack.empty((4,), 'float32').dtype is tack.f32
    with pytest.raises(TypeError, match='dtype'):
        tack.empty((4,), object())
    with pytest.raises(ValueError, match='nonnegative'):
        tack.empty((-1,))


@pytest.mark.parametrize('dtype', [tack.vec3(tack.i32), tack.mat(2, 3, tack.f64)])
def test_element_types_survive_cache_serialization(dtype):
    assert pickle.loads(pickle.dumps(dtype)) == dtype
    assert copy.deepcopy(dtype) == dtype
    assert tack.vec3() is tack.vec(3, tack.f32)
    assert tack.mat3() is tack.mat(3, 3, tack.f32)
    with pytest.raises(TypeError, match='scalar type'):
        tack.vec3(tack.vec2())
    with pytest.raises(ValueError, match='positive integer'):
        tack.vec(0)
    with pytest.raises(ValueError, match='4x4'):
        tack.mat(5, 2)


def test_vector_reductions_count_scalar_components(backend):
    values = tack.array(np.arange(12, dtype=np.float32).reshape(4, 3), dtype=tack.vec3())
    assert values.sum() == 66
    assert values.mean() == 5.5
    assert values.min() == 0 and values.max() == 11
    from tack import algorithms
    assert algorithms.dot(values, values) == 506
    assert algorithms.count_nonzero(values) == 11
    assert algorithms.var(values) == pytest.approx(np.var(np.arange(12)))


def test_typed_geometry_and_dataset_fields(backend):
    mesh = td.uniform_grid((2, 2, 2))
    positions = tack.array(mesh.positions(), dtype=tack.vec3())
    data = td.DataSet(mesh.topology, td.Field(td.H1(mesh), positions))
    assert data.dtype is tack.f32
    assert td.arrays.size_of(positions) == 8
    assert td.arrays.dtype_of(positions) is tack.f32
    temperature = tack.array(data.positions()[:, 0], dtype=tack.f32)
    data.fields['temperature'] = td.Field(td.H1(data), temperature)
    data.fields['velocity'] = td.Field(td.H1(data), positions)
    projected = td.project(data, 'velocity', td.Constant(data))
    np.testing.assert_allclose(td.arrays.to_host(projected.values), [[0.5, 0.5, 0.5]])
    surface = td.contour(data, 'temperature', 0.5)
    assert surface.num_cells > 0
    np.testing.assert_allclose(td.arrays.to_host(surface.fields['velocity'].values)[:, 0], 0.5)


def test_typed_pointer_and_dlpack_views(backend):
    values = tack.array(np.arange(12, dtype=np.float32).reshape(4, 3), dtype=tack.vec3())
    buffer = values._buffer
    pointer = getattr(buffer, '_mtl_buffer', None)
    if pointer is None:
        pointer = buffer.address
    # Metal's public wrap_ptr takes its MTLBuffer rather than a CPU address.
    if backend == 'metal':
        pointer = buffer.metal_buffer
    shared = tack.array_from_ptr(pointer, (4,), tack.vec3())
    assert shared.dtype is tack.vec3() and shared.shape == (4,)
    np.testing.assert_array_equal(shared.numpy(), values.numpy())
    with pytest.raises(RuntimeError, match='read-only'):
        shared.fill(0)
    if backend == 'metal':
        # Metal exports host-addressable memory but cannot wrap arbitrary
        # host pointers as MTLBuffers; the explicit copying import works,
        # and without a device the memory comes back as a CPU array.
        with pytest.raises(RuntimeError, match='cannot wrap without copying'):
            tack.from_dlpack(values, device='metal')
        np.testing.assert_array_equal(tack.from_dlpack(values).numpy(), values.numpy())
        round_trip = tack.from_dlpack(values, copy=True)
    else:
        round_trip = tack.from_dlpack(values)
    assert round_trip.shape == (4, 3)  # DLPack has scalar dtypes and trailing component axes.
    np.testing.assert_array_equal(round_trip.numpy(), values.numpy())


def test_vtk_layout_understands_element_types(backend, monkeypatch):
    from types import SimpleNamespace

    from tack.interop import vtk
    values = tack.empty((4,), tack.mat2())
    values.fill([[1, 2], [3, 4]])
    # The boundary can be checked without installing VTK: it receives a
    # scalar tensor of (tuples, components), sharing the original buffer.
    bridge = SimpleNamespace(dlpack_to_vtk=lambda shaped, name: shaped)
    monkeypatch.setattr(vtk, '_dlpack_support', lambda: bridge)
    monkeypatch.setattr(vtk, '_require_shared_level_zero_context', lambda *args: None)
    shaped = vtk.array_to_vtk(values)
    assert shaped.shape == (4, 4) and shaped.dtype is tack.f32
    assert shaped._buffer is values._buffer
    np.testing.assert_array_equal(shaped.numpy(), np.tile([1, 2, 3, 4], (4, 1)))
    with pytest.raises(ValueError, match='element width'):
        vtk.array_to_vtk(values, n_components=3)


def test_component_precision_controls_scalar_arguments_and_capability_checks(backend):
    from tack.runtime.dispatch import get_backend

    source = tack.array(np.ones((4, 3), dtype=np.float64), dtype=tack.vec3(tack.f64))
    output = tack.empty_like(source)

    @tack.kernel
    def add_epsilon(source, output, epsilon):
        for i in range(source.shape[0]):
            output[i] = source[i] + epsilon

    epsilon = 2.0 ** -40
    if not get_backend().supports_f64:
        with pytest.raises(TypeError, match='not supported'):
            add_epsilon(source, output, epsilon)
    else:
        add_epsilon(source, output, epsilon)
        np.testing.assert_array_equal(output.numpy(), np.full((4, 3), 1.0 + epsilon))
