"""A dataset's origin: positions are ``origin + value``, the value in the geometry's
type, so coordinates far from zero (a LiDAR survey's eastings and northings) keep
their precision in single precision.

Every algorithm is run on a dataset at zero and on the same values at a large
origin, its absolute arguments (implicit functions, probe points, seeds,
elevation points) given in world coordinates: the results are the same values
bit for bit, at the same origin. Algorithms of absolute position (cylindrical and
spherical coordinates, a function of position) are run against a dataset whose
values include the origin, which integer coordinates make exact.
"""

import numpy as np
import pytest
from test_dataset_devices import CALLS, _hexes

import tack
import tack.dataset as td
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkImageData, vtkUnstructuredGrid

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtkUnstructuredGrid = None

needs_vtk = pytest.mark.skipif(vtkUnstructuredGrid is None, reason="needs VTK")

#: Far from zero, as a UTM survey is, and whole, so ``value + origin`` of the
#: test's integer coordinates is exact in single precision too.
ORIGIN = (4_000_000.0, 500_000.0, 100.0)


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _at(origin, data):
    """``data`` with its values kept and ``origin`` set."""
    return td.DataSet(data.topology, data.geometry, fields=dict(data.fields),
                      sets=dict(data.sets), origin=origin)


def _world(data):
    """``data`` at zero, its values the world positions (exact for the test's
    integer coordinates)."""
    return td.DataSet(data.topology, data.world_positions().astype(np.float32),
                      fields=dict(data.fields), sets=dict(data.sets))


def w(data, p):
    """The world position ``p`` (given relative to the data) for ``data``."""
    return tuple(np.add(p, data.origin))


#: Each algorithm with its absolute arguments given in world coordinates.
SHIFTED = {
    "slice": lambda d: td.slice(d, td.Plane(w(d, (0.5, 0, 0)), (1, 0, 0))),
    "extract_geometry": lambda d: td.extract_geometry(d, td.Sphere(w(d, (0, 0, 0)), 1.5)),
    "extract_points": lambda d: td.extract_points(d, td.Sphere(w(d, (0, 0, 0)), 1.5)),
    "implicit_values": lambda d: td.implicit_values(d, td.Sphere(w(d, (0, 0, 0)), 1.0)),
    "point_elevation": lambda d: td.point_elevation(d, w(d, (0, 0, 0)), w(d, (0, 0, 1))),
    "probe": lambda d: td.probe(d, np.array([w(d, (0.5, 0.5, 0.5))])),
    "advect": lambda d: td.advect(d, "v", np.array([w(d, (0.5, 0.5, 0.5))]), 0.01, 3),
    "streamlines": lambda d: td.streamlines(d, "v", np.array([w(d, (0.5, 0.5, 0.5))]),
                                            0.01, 3),
    "box": lambda d: td.extract_points(d, td.Box(w(d, (-1, -1, -1)), w(d, (1.5, 1.5, 1.5)))),
    "cylinder": lambda d: td.implicit_values(d, td.Cylinder(w(d, (1, 1, 0)), (0, 0, 1), 1.0)),
    "planes": lambda d: td.implicit_values(d, td.Planes([w(d, (1, 0, 0))], [(1, 1, 0)])),
    "clip_boolean": lambda d: td.clip(d, td.Difference(
        td.Sphere(w(d, (1, 1, 1)), 1.2), td.Planes([w(d, (1.5, 0, 0))], [(-1, 0.2, 0)]))),
    "connected_components": td.connected_components,
}
#: What depends on absolute position, compared with ``_world``'s.
ABSOLUTE = {"cylindrical", "spherical", "interpolate"}
#: What moves the origin itself, checked on its own below.
OWN = {"transform"}


def _calls():
    every = {name: call for name, call in CALLS.items() if name not in OWN}
    every.update(SHIFTED)
    return every


def _same(a, b, origin, what):
    """``a`` and ``b`` hold the same values, bit for bit; datasets in ``a`` are at
    ``origin``."""
    if isinstance(a, td.DataSet):
        assert a.origin == origin, f"{what}: origin {a.origin}, expected {origin}"
        _same(a.geometry, b.geometry, origin, f"{what}'s geometry")
        assert sorted(a.fields) == sorted(b.fields), what
        for name in a.fields:
            _same(a.fields[name], b.fields[name], origin, f"{what}'s {name}")
    elif isinstance(a, td.Field):
        np.testing.assert_array_equal(tack.arrays.to_host(a.values),
                                      tack.arrays.to_host(b.values), err_msg=what)
    elif isinstance(a, td.Set):
        np.testing.assert_array_equal(a.ids.numpy(), b.ids.numpy(), err_msg=what)
    elif isinstance(a, (tuple, list)):
        for x, y in zip(a, b):
            _same(x, y, origin, what)
    elif hasattr(a, "__dict__") and not isinstance(a, tack.Array):
        assert vars(a).keys() == vars(b).keys(), what
        for key in vars(a):
            _same(vars(a)[key], vars(b)[key], origin, f"{what}.{key}")
    elif isinstance(a, tack.Array):
        np.testing.assert_array_equal(a.numpy(), b.numpy(), err_msg=what)
    else:
        assert a == b or (a != a and b != b), f"{what}: {a!r} != {b!r}"


def test_an_origin_is_three_finite_numbers():
    data = td.wavelet(2)
    assert data.origin == (0.0, 0.0, 0.0)
    data.origin = np.array([1, 2, 3])
    assert data.origin == (1.0, 2.0, 3.0) and all(type(v) is float for v in data.origin)
    for bad in [(1, 2), (1, 2, float("nan")), "abc", (1, 2, 3, 4)]:
        with pytest.raises(ValueError, match="three finite numbers"):
            data.origin = bad


def test_world_positions_and_rebased():
    data = _at(ORIGIN, _hexes(tack.device("cpu")))
    world = data.world_positions()
    assert world.dtype == np.float64
    np.testing.assert_array_equal(world, data.positions() + np.asarray(ORIGIN))
    moved = data.rebased((3_999_990.0, 499_990.0, 90.0))
    assert moved.origin == (3_999_990.0, 499_990.0, 90.0)
    np.testing.assert_array_equal(moved.world_positions(), world)
    np.testing.assert_array_equal(moved.positions(), data.positions() + 10)
    assert data.rebased(ORIGIN) is data
    assert sorted(moved.fields) == sorted(data.fields) and "corner" in moved.sets


def test_an_image_takes_a_value_per_size():
    flat = td.uniform_grid((4, 3), origin=(1.0, 2.0), spacing=(0.5, 0.25))
    full = td.uniform_grid((4, 3), origin=(1.0, 2.0, 0.0), spacing=(0.5, 0.25, 1.0))
    np.testing.assert_array_equal(flat.positions(), full.positions())
    turned = td.uniform_grid((4, 3), direction=[[0.0, -1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(turned.positions()[1], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(td.uniform_grid((5,), spacing=(0.5,)).positions()[:, 0],
                                  np.arange(5) * 0.5)
    with pytest.raises(ValueError, match="one to three values"):
        td.uniform_grid((4, 3), spacing=(1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="2 x 2"):
        td.uniform_grid((4, 3), direction=np.eye(4))


def test_an_image_rebased_stays_an_image_with_its_points_exact():
    """A 2 m raster at survey coordinates: its own corner in single precision is
    half a metre coarse at 4.58e6; rebased near it, the corner is moved in double
    precision and the points are where they should be."""
    corner = (431_001.3, 4_580_003.3, 0.0)
    image = td.uniform_grid((51, 51), origin=corner[:2], spacing=(2.0, 2.0))
    near = image.rebased((431_000.0, 4_580_000.0, 0.0))
    assert isinstance(near.geometry.values, tack.arrays.UniformCoordinates)
    assert near.num_cells == image.num_cells
    i, j = np.meshgrid(np.arange(51), np.arange(51), indexing="xy")
    exact = np.stack([corner[0] + 2.0 * i.ravel(), corner[1] + 2.0 * j.ravel(),
                      np.zeros(i.size)], axis=1)
    assert np.abs(image.world_positions() - exact).max() > 0.1
    np.testing.assert_allclose(near.world_positions(), exact, rtol=0, atol=1e-5)
    # The image is still one to deposit onto, now in the points' frame.
    inner = (i.ravel() < 50) & (j.ravel() < 50)
    centres = td.point_cloud(exact[inner] + [1.0, 1.0, 0.0], origin=near.origin)
    np.testing.assert_array_equal(td.deposit(centres, near).values.numpy(), 1)
    # Oriented images keep their direction; a rectilinear grid moves its axes.
    turned = td.uniform_grid((3, 3, 3), origin=ORIGIN, direction=[[0, -1, 0], [1, 0, 0],
                                                                    [0, 0, 1]])
    np.testing.assert_allclose(turned.rebased(ORIGIN).world_positions(),
                               turned.world_positions(), rtol=0, atol=0.5)
    assert turned.rebased(ORIGIN).geometry.values.ORIENTED == 1
    lines = td.rectilinear_grid([0.0, 1.0, 3.0], [0.0, 2.0], [5.0, 6.0])
    moved = lines.rebased((-10.0, 0.0, 1.0))
    assert isinstance(moved.geometry.values, tack.arrays.CartesianProduct)
    np.testing.assert_array_equal(moved.world_positions(), lines.world_positions())
    np.testing.assert_array_equal(moved.geometry.values.xs.numpy(), [10.0, 11.0, 13.0])


@pytest.mark.parametrize("name", sorted(_calls()))
def test_every_algorithm_keeps_the_origin_and_its_values(name):
    call = _calls()[name]
    zero = _hexes(tack.device("cpu"))
    far = _at(ORIGIN, zero)
    if name in ABSOLUTE:
        expected, origin = call(_world(far)), (0.0, 0.0, 0.0)
        got = call(far)
        if name == "interpolate":              # a field: compare the values
            origin = None
    else:
        expected, origin = call(zero), ORIGIN
        got = call(far)
    _same(got, expected, origin, name)


def test_transform_moves_the_origin():
    """The map moves the origin in double precision and its linear part acts on the
    values: a rotation of a survey loses nothing to the size of its coordinates."""
    data = _at(ORIGIN, _hexes(tack.device("cpu")))
    c, s = np.cos(0.3), np.sin(0.3)
    m = np.array([[c, -s, 0, 10.0], [s, c, 0, -20.0], [0, 0, 1, 5.0]])
    out = td.transform(data, m)
    expected = data.world_positions() @ m[:, :3].T + m[:, 3]
    assert np.abs(out.world_positions() - expected).max() < 1e-5
    np.testing.assert_allclose(out.origin, m[:, :3] @ np.asarray(ORIGIN) + m[:, 3])
    zero = td.transform(_hexes(tack.device("cpu")), m)
    assert zero.origin == (0.0, 0.0, 0.0)          # at zero the translation is in the values


def test_a_function_of_position_sees_world_positions():
    data = _at(ORIGIN, _hexes(tack.device("cpu")))
    values = td.interpolate(data, _east, td.H1(data)).values.numpy()
    np.testing.assert_array_equal(values, data.positions()[:, 0])


@tack.func
def _east(p):
    return p[0] - 4_000_000.0


@tack.device_class
class _Slab:
    """An implicit function of one's own: evaluated at world positions too."""

    def __init__(self, x):
        self.x = x

    @tack.func
    def value(self, p):
        return p[0] - self.x


def test_a_function_of_ones_own_sees_world_positions():
    zero = _hexes(tack.device("cpu"))
    far = _at((1000.0, 0.0, 0.0), zero)    # small, so world positions are exact in f32
    np.testing.assert_array_equal(td.implicit_values(far, _Slab(1000.5)).values.numpy(),
                                  td.implicit_values(zero, _Slab(0.5)).values.numpy())


def test_pieces_keep_the_origin_and_refuse_two():
    zero = td.wavelet(4)
    far = _at(ORIGIN, zero)
    pieces = td.split(far, 4)
    assert all(p.origin == ORIGIN for p in pieces)
    plain = td.PartitionedDataSet([td.extract_cells(p, np.arange(p.num_cells)) for p in pieces])
    ghosted = td.ghost_cells(plain)
    expected = td.ghost_cells(td.PartitionedDataSet(
        [td.extract_cells(p, np.arange(p.num_cells)) for p in td.split(zero, 4)]))
    for a, b in zip(ghosted, expected):
        _same(a, b, ORIGIN, "ghost_cells")
    assert td.merge_pieces(td.split(far, 3, ghost_layers=1)).origin == ORIGIN
    for a, b in zip(td.generate_global_ids(plain), td.generate_global_ids(
            td.PartitionedDataSet([td.extract_cells(p, np.arange(p.num_cells))
                                   for p in td.split(zero, 4)]))):
        np.testing.assert_array_equal(a.global_ids.points.numpy(), b.global_ids.points.numpy())
    probed = td.probe(td.split(far, 4, ghost_layers=1), np.array([w(far, (0.5, 0.25, 1.0))]))
    alone = td.probe(zero, np.array([(0.5, 0.25, 1.0)]))
    _same(probed, alone, ORIGIN, "probe of pieces")
    mixed = td.PartitionedDataSet([pieces[0], pieces[1].rebased((0, 0, 0))])
    for call in (td.ghost_cells, td.generate_global_ids, td.merge_pieces,
                 lambda d: td.probe(d, np.zeros((1, 3))), td.connected_components):
        with pytest.raises(ValueError, match="one origin"):
            call(mixed)


def test_probe_of_a_target_on_another_origin():
    """A target dataset's points are its own, relative to its origin."""
    source = td.wavelet(4)
    source.origin = ORIGIN
    target = td.point_cloud(np.array([[4_000_000.25, 500_000.5, 101.0]]))
    probed = td.probe(source, target)
    assert probed.origin == target.origin
    expected = td.probe(td.wavelet(4), np.array([[0.25, 0.5, 1.0]]))
    np.testing.assert_array_equal(probed.fields["RTData"].values.numpy(),
                                  expected.fields["RTData"].values.numpy())


def test_a_point_cloud_far_from_zero_keeps_millimetres():
    """A LiDAR-like survey: points a millimetre apart at 4e6 m. In single precision
    without an origin they collapse onto 0.25 m steps; with one they stay apart."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (1000, 3)).astype(np.uint8)
    world = np.array(ORIGIN) + 0.001 * np.arange(1000)[:, None] * [1.0, 0.5, 0.25]
    cls = rng.integers(0, 8, 1000).astype(np.uint8)
    intensity = rng.integers(0, 65535, 1000).astype(np.uint16)
    cloud = td.point_cloud(world, fields={"classification": cls, "intensity": intensity,
                                          "gps_time": np.linspace(0.0, 1.0, 1000),
                                          "rgb": rgb})
    assert cloud.origin == tuple(np.floor(world.min(axis=0)))
    assert cloud.num_cells == cloud.num_points == 1000
    assert isinstance(cloud.topology, td.UnstructuredTopology)
    assert np.abs(cloud.world_positions() - world).max() < 1e-6
    assert len(np.unique(cloud.positions()[:, 0])) == 1000
    assert len(np.unique(world.astype(np.float32)[:, 0])) < 100     # what f32 alone does
    assert cloud.fields["classification"].values.dtype is tack.u8
    assert cloud.fields["intensity"].values.dtype is tack.u16
    assert cloud.fields["gps_time"].values.dtype is tack.f32
    np.testing.assert_array_equal(cloud.fields["rgb"].values.numpy(), rgb)
    assert td.point_cloud(world, origin=(0, 0, 0)).origin == (0.0, 0.0, 0.0)
    assert td.point_cloud(world.astype(np.float32)).origin == (0.0, 0.0, 0.0)
    wide = td.point_cloud(world, dtype=tack.f64)
    assert wide.dtype is tack.f64 and np.abs(wide.world_positions() - world).max() < 1e-9
    kept = td.threshold_points(cloud, "classification", 2, 2)
    assert kept.origin == cloud.origin
    np.testing.assert_array_equal(kept.world_positions(),
                                  cloud.world_positions()[cls == 2])


class _NoDoubles(CPUBackend):
    """A CPU device without f64 arrays, as Metal is."""

    display_name = "no-f64 device"

    @property
    def supported_dtypes(self):
        return frozenset(d for d in super().supported_dtypes if d is not tack.f64)


def test_double_precision_offsets_only_where_the_device_has_them():
    device = dispatch.as_device(_NoDoubles())
    with pytest.raises(TypeError, match="no tack.f64 arrays"):
        td.point_cloud(np.zeros((2, 3)), dtype=tack.f64, device=device)
    assert td.point_cloud(np.zeros((2, 3)), device=device).dtype is tack.f32


@needs_vtk
def test_vtk_points_come_and_go_relative_to_the_origin():
    world = np.array(ORIGIN) + np.array([[0.123, 0.456, 0.789], [1.001, 2.002, 3.003],
                                         [0.5, 0.25, 0.125]])
    grid = vtkUnstructuredGrid()
    grid.points = vtkPoints(data=vtkDoubleArray(world))
    grid.cells = (1, _vertex_cells_vtk(3))
    data = vtk_to_dataset(grid, origin=ORIGIN)
    assert data.origin == ORIGIN
    assert np.abs(data.world_positions() - world).max() < 1e-6
    back = np.asarray(dataset_to_vtk(data).points.data)
    assert back.dtype == np.float64 and np.abs(back - world).max() < 1e-6
    lost = vtk_to_dataset(grid)                     # f32 without an origin: 0.25 m steps
    assert np.abs(lost.world_positions() - world).max() > 1e-3


def _vertex_cells_vtk(n):
    from vtkmodules.vtkCommonDataModel import vtkCellArray

    return vtkCellArray(offsets=np.arange(n + 1, dtype=np.int64),
                        connectivity=np.arange(n, dtype=np.int64))


@needs_vtk
def test_an_image_comes_and_goes_relative_to_the_origin():
    image = vtkImageData(dimensions=(3, 3, 3), origin=ORIGIN, spacing=(0.5, 0.5, 0.5))
    data = vtk_to_dataset(image, origin=ORIGIN)
    np.testing.assert_array_equal(data.positions()[0], [0, 0, 0])
    np.testing.assert_allclose(data.world_positions(),
                               np.array(ORIGIN) + td.uniform_grid((3, 3, 3), spacing=(
                                   0.5, 0.5, 0.5)).positions())
    back = dataset_to_vtk(data)
    assert tuple(back.origin) == ORIGIN
