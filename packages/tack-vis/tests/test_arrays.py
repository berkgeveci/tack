"""Implicit arrays: every kind against NumPy, composed to depth, as a field's values
in filters, and uniform grids against VTK's image data."""

import numpy as np
import pytest

import tack
import tack.data as td
from tack.data import arrays

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    vtk = None


def _field(values, dtype=tack.f32):
    values = np.asarray(values)
    f = (tack.Vector.field(values.shape[1], dtype, shape=(values.shape[0],))
         if values.ndim == 2 else tack.field(dtype, shape=values.shape))
    f.from_numpy(values.astype(dtype.numpy_dtype))
    return f


@tack.func
def _square(k):
    return tack.f32(k) * tack.f32(k)


@tack.func
def _negate(x):
    return -x


@tack.func
def _ramp(k):
    return [tack.f32(k), 2.0 * tack.f32(k), 1.0]


def _cases():
    base = np.arange(10, dtype=np.float32) * 1.5
    vectors = np.arange(30, dtype=np.float32).reshape(10, 3)
    ids = np.array([3, 1, 4, 1, 5, 9, 2, 6], np.int32)
    return {
        "constant": (lambda: td.ConstantArray(2.5, 4), np.full(4, 2.5)),
        "constant vector": (lambda: td.ConstantArray([1.0, 2.0, 3.0], 2),
                            np.tile([1.0, 2.0, 3.0], (2, 1))),
        "counting": (lambda: td.CountingArray(5, 3, 2), 3 + 2 * np.arange(5)),
        "permutation": (lambda: td.Permutation(_field(ids, tack.i32), _field(base)), base[ids]),
        "permutation of counting": (lambda: td.Permutation(ids, td.CountingArray(10, 1, 3)),
                                    1 + 3 * ids),
        "permutation twice": (lambda: td.Permutation(ids[:3], td.Permutation(ids, _field(base))),
                              base[ids][ids[:3]]),
        "concatenate": (lambda: td.Concatenate(_field(base[:3]), td.CountingArray(2, 0.5, 0.5),
                                               _field(base[5:7])),
                        np.r_[base[:3], 0.5, 1.0, base[5:7]]),
        "strided": (lambda: td.Strided(_field(base), 3, 1), base[1::3]),
        "view": (lambda: td.View(td.CountingArray(20, 0, 1), 5, 4), np.arange(5, 9)),
        "extract component": (lambda: td.ExtractComponent(_field(vectors), 2), vectors[:, 2]),
        "components": (lambda: td.Components(td.CountingArray(4, 0.0, 1.0), _field(base[:4]),
                                             td.ConstantArray(7.0, 4)),
                       np.c_[np.arange(4), base[:4], np.full(4, 7.0)]),
        "function": (lambda: td.Function(_square, 6), np.arange(6) ** 2),
        "vector function": (lambda: td.Function(_ramp, 3, width=3),
                            np.c_[np.arange(3), 2 * np.arange(3), np.ones(3)]),
        "transform": (lambda: td.Transform(_negate, td.Permutation(ids, _field(base))),
                      -base[ids]),
        "cast": (lambda: td.Cast(_field(base), tack.i32), base.astype(np.int32)),
        "uniform": (lambda: td.UniformCoordinates((3, 2, 2), (1.0, 2.0, 3.0), (0.5, 1.0, 2.0)),
                    np.array([[1 + 0.5 * i, 2 + j, 3 + 2 * k]
                              for k in range(2) for j in range(2) for i in range(3)])),
    }


CASES = _cases()


@pytest.mark.parametrize("name", CASES)
def test_each_kind_is_its_formula(backend, name):
    make, expected = CASES[name]
    array = make()
    np.testing.assert_allclose(arrays.to_host(arrays.materialize(array)), expected, rtol=1e-6)
    assert arrays.size_of(array) == len(expected)
    assert arrays.width_of(array) == (expected.shape[1] if expected.ndim == 2 else None)


def test_random_arrays_are_tack_random(backend):
    from tack import random

    u = td.RandomUniform(1000, seed=7, low=-1.0, high=1.0).to_numpy()
    expected = [-1.0 + 2.0 * random.np_uniform(random.np_seed(k, 7))[0] for k in range(1000)]
    np.testing.assert_allclose(u, expected, atol=1e-6)
    z = td.RandomNormal(20000, seed=3, mean=5.0, stddev=2.0).to_numpy()
    assert abs(z.mean() - 5.0) < 0.05 and abs(z.std() - 2.0) < 0.05


def test_oriented_uniform_coordinates(backend):
    theta = 0.3
    d = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    grid = td.UniformCoordinates((4, 3, 2), (1.0, -1.0, 0.5), (0.25, 0.5, 1.0), d)
    k, j, i = np.meshgrid(np.arange(2), np.arange(3), np.arange(4), indexing="ij")
    ijk = np.stack([i, j, k], -1).reshape(-1, 3) * [0.25, 0.5, 1.0]
    expected = np.array([1.0, -1.0, 0.5]) + ijk @ d.T
    np.testing.assert_allclose(arrays.to_host(arrays.materialize(grid)), expected, atol=1e-6)
    np.testing.assert_allclose(grid.to_numpy(), expected, atol=1e-6)


def test_composed_values_in_filters(backend):
    """A field whose values are composed arrays, on a uniform grid whose points are
    computed: the filters read both through their views and never store them."""
    data = td.uniform_grid((6, 5, 4), (0.0, 0.0, 0.0), (0.2, 0.25, 1.0 / 3.0))
    n = data.num_points
    x = data.positions()

    @tack.func
    def ramp(k):
        return tack.f32(k % 6) * 0.2

    data.fields["x"] = td.Field(td.H1(data), td.Function(ramp, n))
    data.fields["shifted"] = td.Field(
        td.H1(data), td.Permutation(td.CountingArray(n, 0, 1), td.Transform(_negate,
                                                                            td.Function(ramp, n))))
    np.testing.assert_allclose(arrays.to_host(arrays.materialize(data.fields["x"].values)),
                               x[:, 0], atol=1e-6)
    cut = td.contour(data, "x", 0.5)
    np.testing.assert_allclose(cut.positions()[:, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(cut.fields["shifted"].values.to_numpy(), -0.5, atol=1e-6)
    kept = td.threshold(data, "x", 0.3, 1.0)
    assert kept.num_cells == 3 * 4 * 3
    gradient = td.algorithms.gradients(data, data.fields["x"]).values.to_numpy(vectors=True)
    np.testing.assert_allclose(gradient, np.tile([1.0, 0.0, 0.0], (len(gradient), 1)),
                               atol=1e-5)


def test_refusals():
    with pytest.raises(TypeError, match="width"):
        td.Concatenate(td.CountingArray(3), td.ConstantArray([1.0, 2.0], 2))
    with pytest.raises(TypeError, match="ExtractComponent"):
        td.Strided(tack.Vector.field(3, tack.f32, shape=(4,)), 3)
    with pytest.raises(TypeError, match="tack.func"):
        td.Function(lambda k: k, 3)
    with pytest.raises(ValueError, match="component"):
        td.ExtractComponent(td.CountingArray(3), 0)


@pytest.mark.skipif(vtk is None, reason="needs VTK")
def test_image_data_both_ways(backend):
    image = vtk.vtkImageData()
    image.SetExtent(2, 6, -1, 3, 0, 2)
    image.SetOrigin(0.5, 1.0, -2.0)
    image.SetSpacing(0.5, 0.25, 2.0)
    image.SetDirectionMatrix(0, -1, 0, 1, 0, 0, 0, 0, 1)
    data = vtk_to_dataset(image)
    assert isinstance(data.geometry.values, td.UniformCoordinates)
    expected = np.array([image.GetPoint(p) for p in range(image.GetNumberOfPoints())])
    np.testing.assert_allclose(data.positions(), expected, atol=1e-6)
    back = dataset_to_vtk(data)
    assert back.IsA("vtkImageData")
    np.testing.assert_allclose([back.GetPoint(p) for p in range(back.GetNumberOfPoints())],
                               expected, atol=1e-6)
    wavelet = td.sources.wavelet(3)
    assert isinstance(wavelet.geometry.values, td.UniformCoordinates)
    np.testing.assert_allclose(vtk_to_numpy(dataset_to_vtk(wavelet).GetPointData()
                                            .GetArray("RTData")),
                               wavelet.fields["RTData"].values.to_numpy())
