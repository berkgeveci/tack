"""Field and geometry transforms and vector analysis (tack.data.transforms), in
Viskores' conventions, against VTK where VTK has the filter."""

import numpy as np
import pytest
from test_dataset_api import _scalars, _two_hexes_and_a_pyramid

import tack
import tack.data as td
from tack.data import transforms as tf

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")


def _vectors(rows):
    rows = np.asarray(rows, np.float32)
    field = tack.Vector.field(rows.shape[1], tack.f32, shape=(len(rows),))
    field.from_numpy(rows)
    return field


def _grid():
    source = vtk.vtkCellTypeSource()
    source.SetCellType(10)
    source.SetBlocksDimensions(4, 2, 2)
    source.Update()
    grid = source.GetOutput()
    x = vtk_to_numpy(grid.GetPoints().GetData())
    # Points on every side of the origin, for the coordinate transforms.
    grid.GetPoints().SetData(numpy_to_vtk((x - [2.0, 1.0, 1.0]).astype(np.float32), deep=True))
    return grid


def test_vector_analysis(backend):
    data = _two_hexes_and_a_pyramid()
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, data.num_points, 3)).astype(np.float32)
    fa, fb = td.Field(td.H1(data), _vectors(a)), td.Field(td.H1(data), _vectors(b))
    np.testing.assert_allclose(tf.magnitude(fa).values.to_numpy(), np.linalg.norm(a, axis=1),
                               rtol=1e-5)
    np.testing.assert_allclose(tf.dot(fa, fb).values.to_numpy(), (a * b).sum(1), rtol=1e-4,
                               atol=1e-5)
    np.testing.assert_allclose(tf.cross(fa, fb).values.to_numpy(vectors=True), np.cross(a, b),
                               rtol=1e-4, atol=1e-5)
    both = tf.composite(td.Field(td.H1(data), _scalars(a[:, 0])),
                        td.Field(td.H1(data), _scalars(b[:, 1])))
    np.testing.assert_array_equal(both.values.to_numpy(vectors=True), np.c_[a[:, 0], b[:, 1]])
    assert tf.magnitude(fa).space is td.H1(data)
    with pytest.raises(ValueError, match="one space"):
        tf.dot(fa, td.Field(td.L2(data), _vectors(np.zeros((td.L2(data).size, 3)))))


@pytest.mark.parametrize("base", [np.e, 2, 10])
def test_log_values_clamp_below_the_minimum(backend, base):
    data = _two_hexes_and_a_pyramid()
    values = np.linspace(-1.0, 5.0, data.num_points).astype(np.float32)
    out = tf.log_values(td.Field(td.H1(data), _scalars(values)), base).values.to_numpy()
    tiny = np.finfo(np.float32).tiny
    np.testing.assert_allclose(out, np.log(np.maximum(values, tiny)) / np.log(base), rtol=1e-5)
    out = tf.log_values(td.Field(td.H1(data), _scalars(values)), base, min_value=0.5)
    np.testing.assert_allclose(out.values.to_numpy(),
                               np.log(np.maximum(values, 0.5)) / np.log(base), rtol=1e-5)


def test_ids_are_implicit(backend):
    data = _two_hexes_and_a_pyramid()
    assert isinstance(tf.point_ids(data).values, td.CountingArray)
    np.testing.assert_array_equal(td.arrays.to_host(tf.cell_ids(data).values), [0, 1, 2])
    assert tf.cell_ids(data).space is td.Values(data, "cells")


@needs_vtk
def test_point_elevation_is_vtks(backend):
    grid = _grid()
    elevation = vtk.vtkElevationFilter()
    elevation.SetLowPoint(-1.0, -0.5, 0.2)
    elevation.SetHighPoint(1.5, 0.7, -0.4)
    elevation.SetScalarRange(3.0, 7.0)
    elevation.SetInputData(grid)
    elevation.Update()
    expected = vtk_to_numpy(elevation.GetOutput().GetPointData().GetArray("Elevation"))
    out = tf.point_elevation(vtk_to_dataset(grid), (-1.0, -0.5, 0.2), (1.5, 0.7, -0.4),
                             (3.0, 7.0))
    np.testing.assert_allclose(out.values.to_numpy(), expected, rtol=1e-5, atol=1e-5)
    assert expected.min() == pytest.approx(3.0) and expected.max() == pytest.approx(7.0)


@needs_vtk
def test_warp_is_vtks(backend):
    grid = _grid()
    x = vtk_to_numpy(grid.GetPoints().GetData())
    direction = np.c_[np.sin(x[:, 0]), x[:, 1] ** 2, np.ones(len(x))].astype(np.float32)
    array = numpy_to_vtk(direction, deep=True)
    array.SetName("d")
    grid.GetPointData().SetVectors(array)
    scale = numpy_to_vtk((x[:, 2] + 2).astype(np.float32), deep=True)
    scale.SetName("s")
    grid.GetPointData().SetScalars(scale)
    data = vtk_to_dataset(grid)

    warp = vtk.vtkWarpVector()
    warp.SetScaleFactor(0.3)
    warp.SetInputData(grid)
    warp.Update()
    np.testing.assert_allclose(tf.warp(data, "d", 0.3).positions(),
                               vtk_to_numpy(warp.GetOutput().GetPoints().GetData()), rtol=1e-5,
                               atol=1e-6)
    scalar = vtk.vtkWarpScalar()
    scalar.SetScaleFactor(0.5)
    scalar.UseNormalOn()
    scalar.SetNormal(0.0, 0.6, 0.8)
    scalar.SetInputData(grid)
    scalar.Update()
    out = tf.warp(data, (0.0, 0.6, 0.8), 0.5, scale_by="s")
    np.testing.assert_allclose(out.positions(),
                               vtk_to_numpy(scalar.GetOutput().GetPoints().GetData()), rtol=1e-5,
                               atol=1e-6)
    assert set(out.fields) == set(data.fields) and out.topology is data.topology


@needs_vtk
def test_transform_is_vtks(backend):
    grid = _grid()
    t = vtk.vtkTransform()
    t.Translate(1.0, -2.0, 0.5)
    t.RotateWXYZ(33.0, 0.2, 1.0, 0.4)
    t.Scale(1.5, 0.7, 2.0)
    move = vtk.vtkTransformFilter()
    move.SetTransform(t)
    move.SetInputData(grid)
    move.Update()
    matrix = np.array([[t.GetMatrix().GetElement(i, j) for j in range(4)] for i in range(4)])
    out = tf.transform(vtk_to_dataset(grid), matrix)
    np.testing.assert_allclose(out.positions(),
                               vtk_to_numpy(move.GetOutput().GetPoints().GetData()), rtol=1e-5,
                               atol=1e-5)
    with pytest.raises(ValueError, match="affine"):
        tf.transform(vtk_to_dataset(grid), np.ones((4, 4)))


def _viskores_cylindrical(p):
    r = np.hypot(p[:, 0], p[:, 1])
    theta = np.where(r > 0, np.arcsin(np.divide(p[:, 1], r, out=np.zeros_like(r), where=r > 0)), 0)
    theta = np.where((r > 0) & (p[:, 0] < 0), np.pi - theta, theta)
    return np.c_[r, theta, p[:, 2]]


def _viskores_spherical(p):
    r = np.linalg.norm(p, axis=1)
    theta = np.where(r > 0, np.arccos(np.divide(p[:, 2], r, out=np.zeros_like(r), where=r > 0)),
                     0)
    phi = np.arctan2(p[:, 1], p[:, 0])
    return np.c_[r, theta, np.where(phi < 0, phi + 2 * np.pi, phi)]


@needs_vtk
@pytest.mark.parametrize("which", ["cylindrical", "spherical"])
def test_coordinate_transforms_are_viskores(backend, which):
    data = vtk_to_dataset(_grid())
    x = data.positions()
    assert (x[:, 0] < 0).any() and (np.linalg.norm(x, axis=1) == 0).any()   # both cases
    transform, reference = {"cylindrical": (tf.cylindrical, _viskores_cylindrical),
                            "spherical": (tf.spherical, _viskores_spherical)}[which]
    out = transform(data)
    np.testing.assert_allclose(out.positions(), reference(x.astype(np.float64)), rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(transform(out, inverse=True).positions(), x, atol=1e-5)


def test_an_l2_geometry_moves_every_corner(backend):
    data = _two_hexes_and_a_pyramid()
    corners = td.algorithms.discontinuous(data, data.geometry)
    pulled = td.DataSet(data.topology, corners)
    moved = tf.transform(pulled, [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0]])
    assert isinstance(moved.geometry.space, td.L2)
    np.testing.assert_allclose(moved.geometry.values.to_numpy(vectors=True),
                               corners.values.to_numpy(vectors=True) + np.array([5.0, 0.0, 0.0]))
