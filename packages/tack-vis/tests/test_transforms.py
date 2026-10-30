"""Field and geometry transforms and vector analysis (now in tack.dataset.algorithms), in
Viskores' conventions, against VTK where VTK has the filter."""

import numpy as np
import pytest
from test_dataset_api import _scalars, _two_hexes_and_a_pyramid

import tack
import tack.dataset as td

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")


def _vectors(rows):
    rows = np.asarray(rows, np.float32)
    field = tack.empty((len(rows),), tack.vec(rows.shape[1], tack.f32))
    tack.copyto(field, rows)
    return field


def _grid():
    grid = vtk.vtkCellTypeSource(cell_type=10, blocks_dimensions=(4, 2, 2))()
    x = np.asarray(grid.points.data)
    # Points on every side of the origin, for the coordinate transforms.
    grid.points = vtk.vtkPoints(data=vtk.vtkFloatArray((x - [2.0, 1.0, 1.0]).astype(np.float32)))
    return grid


def test_vector_analysis(backend):
    data = _two_hexes_and_a_pyramid()
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, data.num_points, 3)).astype(np.float32)
    fa, fb = td.Field(td.H1(data), _vectors(a)), td.Field(td.H1(data), _vectors(b))
    np.testing.assert_allclose(td.magnitude(data, fa).values.numpy(), np.linalg.norm(a, axis=1),
                               rtol=1e-5)
    np.testing.assert_allclose(td.dot(data, fa, fb).values.numpy(), (a * b).sum(1), rtol=1e-4,
                               atol=1e-5)
    np.testing.assert_allclose(td.cross(data, fa, fb).values.numpy(), np.cross(a, b),
                               rtol=1e-4, atol=1e-5)
    both = td.composite(data, td.Field(td.H1(data), _scalars(a[:, 0])),
                        td.Field(td.H1(data), _scalars(b[:, 1])))
    np.testing.assert_array_equal(both.values.numpy(), np.c_[a[:, 0], b[:, 1]])
    assert td.magnitude(data, fa).space is td.H1(data)
    with pytest.raises(ValueError, match="one space"):
        td.dot(data, fa, td.Field(td.L2(data), _vectors(np.zeros((td.L2(data).size, 3)))))


@pytest.mark.parametrize("base", [np.e, 2, 10])
def test_log_values_clamp_below_the_minimum(backend, base):
    data = _two_hexes_and_a_pyramid()
    values = np.linspace(-1.0, 5.0, data.num_points).astype(np.float32)
    out = td.log_values(data, td.Field(td.H1(data), _scalars(values)), base).values.numpy()
    tiny = np.finfo(np.float32).tiny
    np.testing.assert_allclose(out, np.log(np.maximum(values, tiny)) / np.log(base), rtol=1e-5)
    out = td.log_values(data, td.Field(td.H1(data), _scalars(values)), base, min_value=0.5)
    np.testing.assert_allclose(out.values.numpy(),
                               np.log(np.maximum(values, 0.5)) / np.log(base), rtol=1e-5)


def test_ids_are_implicit(backend):
    data = _two_hexes_and_a_pyramid()
    assert isinstance(td.point_ids(data).values, tack.arrays.CountingArray)
    np.testing.assert_array_equal(tack.arrays.to_host(td.cell_ids(data).values), [0, 1, 2])
    assert td.cell_ids(data).space is td.Values(data, "cells")


@needs_vtk
def test_point_elevation_is_vtks(backend):
    grid = _grid()
    elevation = vtk.vtkElevationFilter(low_point=(-1.0, -0.5, 0.2), high_point=(1.5, 0.7, -0.4),
                                       scalar_range=(3.0, 7.0))
    expected = np.asarray(elevation(grid).point_data["Elevation"])
    out = td.point_elevation(vtk_to_dataset(grid), (-1.0, -0.5, 0.2), (1.5, 0.7, -0.4),
                             (3.0, 7.0))
    np.testing.assert_allclose(out.values.numpy(), expected, rtol=1e-5, atol=1e-5)
    assert expected.min() == pytest.approx(3.0) and expected.max() == pytest.approx(7.0)


@needs_vtk
def test_warp_is_vtks(backend):
    grid = _grid()
    x = np.asarray(grid.points.data)
    point_data = grid.point_data
    point_data["d"] = np.c_[np.sin(x[:, 0]), x[:, 1] ** 2, np.ones(len(x))].astype(np.float32)
    point_data["s"] = (x[:, 2] + 2).astype(np.float32)
    # The warps read the active vectors and scalars, which have no pythonic setter.
    point_data.SetActiveVectors("d")
    point_data.SetActiveScalars("s")
    data = vtk_to_dataset(grid)

    warped = vtk.vtkWarpVector(scale_factor=0.3)(grid)
    np.testing.assert_allclose(td.warp(data, "d", 0.3).positions(),
                               np.asarray(warped.points.data), rtol=1e-5, atol=1e-6)
    warped = vtk.vtkWarpScalar(scale_factor=0.5, use_normal=True, normal=(0.0, 0.6, 0.8))(grid)
    out = td.warp(data, (0.0, 0.6, 0.8), 0.5, scale_by="s")
    np.testing.assert_allclose(out.positions(), np.asarray(warped.points.data), rtol=1e-5,
                               atol=1e-6)
    assert set(out.fields) == set(data.fields) and out.topology is data.topology


@needs_vtk
def test_transform_is_vtks(backend):
    grid = _grid()
    t = vtk.vtkTransform()
    t.Translate(1.0, -2.0, 0.5)
    t.RotateWXYZ(33.0, 0.2, 1.0, 0.4)
    t.Scale(1.5, 0.7, 2.0)
    moved = vtk.vtkTransformFilter(transform=t)(grid)
    # A vtkMatrix4x4 cannot be indexed.
    matrix = np.array([[t.matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
    out = td.transform(vtk_to_dataset(grid), matrix)
    np.testing.assert_allclose(out.positions(), np.asarray(moved.points.data), rtol=1e-5,
                               atol=1e-5)
    with pytest.raises(ValueError, match="affine"):
        td.transform(vtk_to_dataset(grid), np.ones((4, 4)))


@tack.func
def _wave(x):
    return tack.sin(x[0]) * x[1] + 0.5 * x[2] * x[2]


def _affine(seed, shear=False):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    linear = q @ np.diag([1.5, 0.7, 2.0])
    if shear:
        linear = linear @ np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.2, 0.0, 1.0]])
    return np.vstack([np.c_[linear, rng.normal(size=3)], [0.0, 0.0, 0.0, 1.0]])


@pytest.mark.parametrize("dims, shear", [((5, 4, 3), False), ((5, 4, 3), True), ((6, 5), False)])
def test_transformed_images_stay_images(backend, dims, shear):
    """An image under an affine map is an image: the map folded into its origin and
    direction (any invertible matrix), nothing stored per point, its positions the
    map of the image's -- and every filter's output what it is on the same
    positions stored. A singular map gives the positions themselves."""
    from tack.arrays import UniformCoordinates

    image = td.uniform_grid(dims, origin=(0.5, -1.0, 2.0)[:len(dims)],
                            spacing=(0.3, 0.4, 0.5)[:len(dims)], dtype=tack.f64)
    image.fields["s"] = td.interpolate(image, _wave, td.H1(image))
    m = _affine(len(dims) + shear, shear)
    out = td.transform(image, m)
    assert isinstance(out.geometry.values, UniformCoordinates)
    assert out.topology is image.topology and set(out.fields) == {"s"}
    expected = image.positions() @ m[:3, :3].T + m[:3, 3]
    np.testing.assert_allclose(out.positions(), expected, atol=1e-12)
    stored = td.DataSet(image.topology, expected, dtype=tack.f64)
    stored.fields["s"] = image.fields["s"]
    centers = td.cell_centers(stored).values.numpy()
    np.testing.assert_allclose(td.cell_centers(out).values.numpy(), centers, atol=1e-12)
    np.testing.assert_allclose(td.probe(out, centers).fields["s"].values.numpy(),
                               td.probe(stored, centers).fields["s"].values.numpy(), atol=1e-12)
    # Twice: the directions compose.
    twice = td.transform(out, m)
    assert isinstance(twice.geometry.values, UniformCoordinates)
    np.testing.assert_allclose(twice.positions(), expected @ m[:3, :3].T + m[:3, 3], atol=1e-11)
    flat = np.diag([1.0, 1.0, 0.0, 1.0])
    projected = td.transform(image, flat)
    assert not isinstance(projected.geometry.values, UniformCoordinates)
    np.testing.assert_allclose(projected.positions()[:, 2], 0.0)


def test_a_far_transformed_image_keeps_its_bits(backend):
    """On a dataset far from zero the map moves the origin in double precision and the
    image's direction takes only its linear part, as for stored positions."""
    image = td.uniform_grid((4, 4, 4), spacing=(0.25, 0.25, 0.25))
    image.origin = (1e7, -3e6, 5e5)
    m = _affine(9)
    out = td.transform(image, m)
    np.testing.assert_allclose(out.origin, m[:3, :3] @ np.asarray(image.origin) + m[:3, 3])
    np.testing.assert_allclose(out.positions(), image.positions() @ m[:3, :3].T, atol=1e-6)


@needs_vtk
def test_transformed_images_go_to_vtk_as_images(backend):
    """VTK holds the folded map as an image's origin and direction matrix: the same
    points as vtkTransformFilter makes of the image."""
    from tack.interop.vtk import dataset_to_vtk

    image = td.uniform_grid((5, 4, 3), origin=(0.5, -1.0, 2.0), spacing=(0.3, 0.4, 0.5),
                            dtype=tack.f64)
    m = _affine(4)
    out = dataset_to_vtk(td.transform(image, m))
    assert isinstance(out, vtk.vtkImageData)
    t = vtk.vtkTransform()
    t.SetMatrix(m.reshape(-1))
    theirs = vtk.vtkTransformFilter(transform=t)(dataset_to_vtk(image))
    ours = np.array([out.GetPoint(i) for i in range(out.number_of_points)])
    np.testing.assert_allclose(ours, np.asarray(theirs.points.data), atol=1e-12)
    back = vtk_to_dataset(out, dtype=tack.f64)
    np.testing.assert_allclose(back.positions(), ours, atol=1e-12)


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
    transform, reference = {"cylindrical": (td.cylindrical, _viskores_cylindrical),
                            "spherical": (td.spherical, _viskores_spherical)}[which]
    out = transform(data)
    np.testing.assert_allclose(out.positions(), reference(x.astype(np.float64)), rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(transform(out, inverse=True).positions(), x, atol=1e-5)


def test_an_l2_geometry_moves_every_corner(backend):
    data = _two_hexes_and_a_pyramid()
    corners = td.algorithms.project(data, data.geometry, td.L2(data))
    pulled = td.DataSet(data.topology, corners)
    moved = td.transform(pulled, [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0]])
    assert isinstance(moved.geometry.space, td.L2)
    np.testing.assert_allclose(moved.geometry.values.numpy(),
                               corners.values.numpy() + np.array([5.0, 0.0, 0.0]))


def test_composite_refuses_mixed_types(backend):
    data = td.uniform_grid((2,))
    ints = td.Field(td.H1(data), _scalars([1, 2], tack.i32))
    floats = td.Field(td.H1(data), _scalars([1.75, 2.75]))
    with pytest.raises(TypeError, match="differ in type"):
        td.composite(data, ints, floats)


@pytest.mark.parametrize("algorithm", [td.dot, td.cross])
def test_binary_vector_operations_refuse_mixed_types(backend, algorithm):
    data = td.uniform_grid((2,))
    a = td.Field(td.H1(data), tack.arrays.ConstantArray([1, 2, 3], 2, dtype=tack.i32))
    b = td.Field(td.H1(data), tack.arrays.ConstantArray([0.5, 0.25, 0.75], 2))
    for left, right in ((a, b), (b, a)):
        with pytest.raises(TypeError, match="differ in type"):
            algorithm(data, left, right)
    converted = td.Field(a.space, tack.arrays.Cast(a.values, tack.f32))
    result = algorithm(data, converted, b).values.numpy()
    expected = [3.25, 3.25] if algorithm is td.dot else [[0.75, 0.75, -0.75]] * 2
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_integer_nonlinear_transforms_compute_floating_results(backend):
    data = td.uniform_grid((2,))
    vector = td.Field(td.H1(data), tack.arrays.ConstantArray([1, 1, 0], 2, dtype=tack.i32))
    norm = td.magnitude(data, vector)
    assert norm.values.dtype is tack.f32
    np.testing.assert_allclose(norm.values.numpy(), np.sqrt(2), rtol=1e-6)
    scalar = td.Field(td.H1(data), _scalars([1, 2], tack.i32))
    logarithm = td.log_values(data, scalar)
    assert logarithm.values.dtype is tack.f32
    np.testing.assert_allclose(logarithm.values.numpy(), [0, np.log(2)], atol=1e-6)


def test_l2_face_positions_follow_mutable_geometry(backend):
    data = td.uniform_grid((2, 2, 2))
    data.geometry = td.project(data, data.geometry, td.L2(data))
    points = data.geometry.values.numpy()
    before = td.face_centers(data).values.numpy()
    shift = np.array([10, -3, 2], np.float32)
    tack.copyto(data.geometry.values, points + shift)
    after = td.face_centers(data).values.numpy()
    np.testing.assert_allclose(after, before + shift, atol=1e-6)
    np.testing.assert_allclose(td.face_geometry(data).area.values.numpy(), 1, atol=1e-6)


def test_shrink_linearizes_quadratic_geometry_without_mutating_input(backend):
    data = td.uniform_grid((2, 2, 2))
    data.geometry = td.project(data, data.geometry, td.H1(data, order=2))
    original = data.geometry.values.numpy().copy()
    out = td.shrink(data, factor=0.5)
    assert isinstance(out.geometry.space, td.L2)
    corners = out.geometry.values.numpy()
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.min(axis=0), [0.25] * 3)
    np.testing.assert_allclose(corners.max(axis=0), [0.75] * 3)
    np.testing.assert_array_equal(data.geometry.values.numpy(), original)
