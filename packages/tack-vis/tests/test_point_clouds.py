"""Point clouds: the point locator and the filters on it (``voxel_downsample``,
``outliers``, ``point_normals``), against brute force in NumPy and against VTK's
vtkFiltersPoints where VTK computes the same thing."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.point_locator import MAX_K
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersPoints import (
        vtkPCANormalEstimation,
        vtkRadiusOutlierRemoval,
        vtkStatisticalOutlierRemoval,
    )
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtkPolyData = None

needs_vtk = pytest.mark.skipif(vtkPolyData is None, reason="needs VTK, the reference")

pytestmark = pytest.mark.quick

#: Far from zero, as a survey is: the filters work on offsets from it.
ORIGIN = np.array([4_000_000.0, 500_000.0, 100.0])


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _cloud(n=2000, seed=0, scale=(10.0, 5.0, 1.0)):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) * scale).astype(np.float32)


def _squared(x, q):
    d = x.astype(np.float32) - q.astype(np.float32)
    return (d * d).sum(axis=-1)


# ── The locator ─────────────────────────────────────────────────────

@pytest.mark.parametrize("bin_size", [None, 0.05, 3.0])
def test_neighbors_within_a_radius_are_brute_forces(bin_size):
    x = _cloud()
    locator = td.PointLocator(td.point_cloud(x), bin_size=bin_size)
    offsets, ids = (a.numpy() for a in locator.neighbors(0.3))
    for q in range(len(x)):
        expected = np.flatnonzero(_squared(x, x[q]) <= 0.09)
        expected = expected[expected != q]
        assert sorted(ids[offsets[q]:offsets[q + 1]].tolist()) == expected.tolist()


def test_nearest_are_brute_forces_ties_by_index():
    x = _cloud(1500)
    x[100] = x[200] = x[300] = x[50]                # coincident: ties broken by index
    k = 8
    ids, d2 = (a.numpy() for a in td.PointLocator(td.point_cloud(x)).nearest(k))
    assert ids.shape == (len(x), k)
    for q in range(len(x)):
        d = _squared(x, x[q])
        d[q] = np.inf
        expected = np.lexsort((np.arange(len(x)), d))[:k]
        np.testing.assert_array_equal(np.sort(d2[q]), np.sort(d[expected]))
        np.testing.assert_array_equal(ids[q][d2[q] == d2[q][0]],
                                      np.sort(ids[q][d2[q] == d2[q][0]]))
    assert ids[50][:3].tolist() == [100, 200, 300]


def test_queries_from_elsewhere_are_world_positions():
    x = _cloud(800)
    cloud = td.point_cloud(x.astype(np.float64) + ORIGIN)
    locator = td.PointLocator(cloud)
    q = np.array([[2.5, 2.5, 0.5], [50.0, 50.0, 50.0]])
    ids, _ = locator.nearest(3, q + ORIGIN)
    local = cloud.positions()
    for row, point in zip(ids.numpy(), q - (np.asarray(cloud.origin) - ORIGIN)):
        expected = np.lexsort((np.arange(len(x)), _squared(local, point)))[:3]
        assert row.tolist() == expected.tolist()
    offsets, near = locator.neighbors(0.5, q + ORIGIN)
    assert offsets.numpy()[-1] == offsets.numpy()[1] > 0      # none near the far point


def test_small_and_degenerate_clouds():
    one = td.PointLocator(td.point_cloud(np.zeros((1, 3))))
    ids, d2 = one.nearest(4)
    assert ids.numpy().tolist() == [[-1, -1, -1, -1]] and d2.numpy()[0, 0] == -1
    same = td.PointLocator(td.point_cloud(np.ones((5, 3))))     # no extent at all
    ids, d2 = same.nearest(4)
    assert ids.numpy()[0].tolist() == [1, 2, 3, 4] and not d2.numpy().any()
    flat = _cloud(500)
    flat[:, 2] = 7.0                                # a plane: one bin along z
    offsets, ids = td.PointLocator(td.point_cloud(flat)).neighbors(0.4)
    assert offsets.numpy()[-1] == sum(
        int((_squared(flat, flat[q]) <= 0.16).sum()) - 1 for q in range(len(flat)))
    empty = td.PointLocator(td.point_cloud(np.zeros((0, 3))))
    assert empty.nearest(2)[0].shape == (0, 2)
    assert empty.neighbors(1.0)[0].numpy().tolist() == [0]
    for bad in (0, MAX_K + 1, 2.5):
        with pytest.raises(ValueError, match="whole number"):
            one.nearest(bad)


def test_a_locator_serves_its_geometry_until_written():
    cloud = td.point_cloud(_cloud(300))
    locator = td.PointLocator(cloud)
    assert locator.serves(cloud)
    td.outliers(cloud, radius=0.3, min_neighbors=1, locator=locator)
    cloud.geometry.values.fill(0.0)
    with pytest.raises(ValueError, match="make a new one"):
        td.outliers(cloud, radius=0.3, min_neighbors=1, locator=locator)


# ── voxel_downsample ────────────────────────────────────────────────

def _voxels(x, size):
    key = np.floor((x - x.min(axis=0)) / np.asarray(size, np.float32)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return first, inverse.reshape(-1)


@pytest.mark.parametrize("size", [1.0, (2.0, 0.5, 0.25)])
def test_voxel_centroids_are_the_voxels_means(size):
    rng = np.random.default_rng(4)
    world = ORIGIN + _cloud(3000).astype(np.float64)
    classes = rng.integers(0, 10, len(world)).astype(np.uint8)
    rgb = rng.random((len(world), 3))
    cloud = td.point_cloud(world, fields={"classification": classes, "rgb": rgb})
    x = cloud.positions()
    first, inverse = _voxels(x, size)
    out = td.voxel_downsample(cloud, size)
    assert out.num_points == len(first) and out.origin == cloud.origin
    counts = np.bincount(inverse)
    means = np.zeros((len(first), 3))
    np.add.at(means, inverse, x.astype(np.float64))
    means /= counts[:, None]
    colours = np.zeros((len(first), 3))
    np.add.at(colours, inverse, rgb.astype(np.float32).astype(np.float64))
    colours /= counts[:, None]
    got = out.positions()
    order = np.lexsort(got.T)
    expected = np.lexsort(means.T)
    np.testing.assert_allclose(got[order], means[expected], atol=1e-5)
    np.testing.assert_allclose(out.fields["rgb"].values.numpy()[order], colours[expected],
                               atol=1e-6)
    # Integer fields are the voxel's first point's, not an average.
    by_voxel = dict(zip(map(tuple, np.round(means, 4)), classes[first]))
    assert out.fields["classification"].values.dtype is tack.u8
    for p, c in zip(np.round(got.astype(np.float64), 4), out.fields["classification"]
                    .values.numpy()):
        assert by_voxel.get(tuple(p), c) == c


def test_voxel_first_and_nearest_keep_input_points():
    x = _cloud(2000)
    first, inverse = _voxels(x, 0.8)
    cloud = td.point_cloud(x, fields={"id": np.arange(len(x), dtype=np.int32)})
    kept = td.voxel_downsample(cloud, 0.8, keep="first").fields["id"].values.numpy()
    assert kept.tolist() == sorted(first.tolist())
    means = np.zeros((len(first), 3), np.float32)
    for v in range(len(first)):
        means[v] = x[inverse == v].mean(axis=0)
    nearest = td.voxel_downsample(cloud, 0.8, keep="nearest").fields["id"].values.numpy()
    for v in range(len(first)):
        members = np.flatnonzero(inverse == v)
        d = _squared(x[members], means[v])
        assert d[np.isin(members, nearest)].min() <= d.min() + 1e-6


def test_voxel_downsample_refuses_what_it_cannot_do():
    cloud = td.point_cloud(_cloud(10))
    for size in (0, -1, np.nan, (1, 0, 1)):
        with pytest.raises(ValueError, match="positive voxel side"):
            td.voxel_downsample(cloud, size)
    with pytest.raises(ValueError, match="keep is one of"):
        td.voxel_downsample(cloud, 1.0, keep="median")
    assert td.voxel_downsample(td.point_cloud(np.zeros((0, 3))), 1.0).num_points == 0


def test_voxel_downsample_is_deterministic():
    cloud = td.point_cloud(_cloud(5000))
    a = td.voxel_downsample(cloud, 0.5).positions()
    b = td.voxel_downsample(cloud, 0.5).positions()
    np.testing.assert_array_equal(a, b)


# ── outliers ────────────────────────────────────────────────────────

def _with_strays():
    x = _cloud(2000)
    strays = x[:5] + np.array([0.0, 0.0, 30.0])
    return np.vstack([x, strays]).astype(np.float32)


def test_outliers_by_radius_are_brute_forces():
    x = _with_strays()
    out = td.outliers(td.point_cloud(x), radius=0.4, min_neighbors=4)
    counts = np.array([int((_squared(x, x[q]) <= 0.16).sum()) - 1 for q in range(len(x))])
    np.testing.assert_array_equal(out.fields["neighbors"].values.numpy(), counts)
    np.testing.assert_array_equal(out.fields["outlier"].values.numpy(), counts < 4)
    assert out.fields["outlier"].values.numpy()[-5:].all()


def test_outliers_by_statistics_are_brute_forces():
    x = _with_strays()
    k = 6
    out = td.outliers(td.point_cloud(x), k=k, deviations=2.0)
    means = []
    for q in range(len(x)):
        d = _squared(x, x[q])
        d[q] = np.inf
        means.append(np.sqrt(np.sort(d)[:k]).mean())
    means = np.array(means)
    np.testing.assert_allclose(out.fields["mean_distance"].values.numpy(), means, rtol=1e-5)
    flagged = out.fields["outlier"].values.numpy().astype(bool)
    assert flagged[-5:].all()
    np.testing.assert_array_equal(flagged, means > means.mean() + 2.0 * means.std())
    kept = td.threshold_points(out, "outlier", 0, 0)
    assert kept.num_points == len(x) - flagged.sum()


def test_outliers_takes_one_test():
    cloud = td.point_cloud(_cloud(10))
    for kwargs in ({}, {"radius": 1.0, "k": 3}, {"min_neighbors": 2, "k": 3}):
        with pytest.raises(ValueError, match="radius and min_neighbors, or k"):
            td.outliers(cloud, **kwargs)


# ── point_normals ───────────────────────────────────────────────────

def _plane(n=2000, seed=5, noise=1e-3):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 10.0
    z = 0.3 * xy[:, 0] + 0.2 * xy[:, 1] + rng.normal(0.0, noise, n)
    exact = np.array([-0.3, -0.2, 1.0])
    return np.column_stack([xy, z]), exact / np.linalg.norm(exact)


@pytest.mark.parametrize("kwargs", [{"k": 16}, {"radius": 0.6}])
def test_normals_of_a_plane_far_from_zero(kwargs):
    x, exact = _plane()
    normals = td.point_normals(td.point_cloud(x + ORIGIN), **kwargs).fields["normals"]
    angles = np.degrees(np.arccos(np.clip(normals.values.numpy() @ exact, -1.0, 1.0)))
    assert angles.max() < 2.0                       # up, by the default direction


def test_normals_point_toward_the_viewpoint():
    rng = np.random.default_rng(6)
    u = rng.normal(size=(3000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    centre = ORIGIN + np.array([1.0, 2.0, 3.0])
    cloud = td.point_cloud(2.0 * u + centre)
    inward = td.point_normals(cloud, k=12, viewpoint=centre).fields["normals"].values.numpy()
    assert ((inward * -u).sum(axis=1) > 0.99).all()
    above = centre + np.array([0.0, 0.0, 10.0])
    facing = td.point_normals(cloud, k=12, viewpoint=above).fields["normals"].values.numpy()
    assert ((facing * (above - cloud.world_positions())).sum(axis=1) >= 0).all()


def test_normals_where_there_is_no_plane():
    normals = td.point_normals(td.point_cloud(np.eye(3)), k=8).fields["normals"]
    np.testing.assert_allclose(normals.values.numpy(), np.full((3, 3), 3 ** -0.5), atol=1e-6)
    line = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    assert not td.point_normals(td.point_cloud(line), k=4).fields["normals"].values.numpy().any()
    two = td.point_normals(td.point_cloud(np.eye(3)[:2]), k=8).fields["normals"]
    assert not two.values.numpy().any()             # fewer than three points


# ── Against VTK ─────────────────────────────────────────────────────

def _vtk(x):
    data = vtkPolyData()
    data.points = vtkPoints(data=vtkFloatArray(x))
    return data


def _kept(filter_):
    return {tuple(p) for p in np.asarray(filter_.GetOutput().points.data).tolist()}


@needs_vtk
def test_outliers_keep_vtks_points():
    x = _with_strays()
    radius = vtkRadiusOutlierRemoval(radius=0.5, number_of_neighbors=3)
    radius.SetInputData(_vtk(x))
    radius.Update()
    ours = td.outliers(td.point_cloud(x), radius=0.5, min_neighbors=3)
    assert _kept(radius) == {tuple(p) for p in x[ours.fields["outlier"].values.numpy() == 0]
                             .tolist()}
    statistical = vtkStatisticalOutlierRemoval(sample_size=8, standard_deviation_factor=1.0)
    statistical.SetInputData(_vtk(x))
    statistical.Update()
    ours = td.outliers(td.point_cloud(x), k=8, deviations=1.0)
    assert _kept(statistical) == {tuple(p) for p in x[ours.fields["outlier"].values.numpy()
                                                      == 0].tolist()}
    means = ours.fields["mean_distance"].values.numpy().astype(np.float64)
    assert abs(means.mean() - statistical.GetComputedMean()) < 1e-6
    assert abs(means.std() - statistical.GetComputedStandardDeviation()) < 1e-6


@needs_vtk
def test_normals_are_vtks():
    x = _cloud(3000, scale=(10.0, 10.0, 1.0))
    pca = vtkPCANormalEstimation(sample_size=12)
    pca.SetNormalOrientationToPoint()
    pca.SetOrientationPoint(5.0, 5.0, 100.0)
    pca.SetInputData(_vtk(x))
    pca.Update()
    theirs = np.asarray(pca.GetOutput().point_data["PCANormals"])
    ours = td.point_normals(td.point_cloud(x), k=12, viewpoint=(5.0, 5.0, 100.0))
    cos = (theirs * ours.fields["normals"].values.numpy()).sum(axis=1)
    assert np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max() < 0.1
