"""Datasets between Tack and warp-simdata (`tack.interop.warp`'s `to_simdata` and
`from_simdata`), on Warp's default device.

Each of warp-simdata's four VTK-shaped models round-trips through Tack's
topologies with its points, cells and fields; arrays of the types both take
are shared, not copied; and each library's filters on the other's data agree
with its own on the original.
"""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.arrays import UniformCoordinates

wp = pytest.importorskip("warp")
sd = pytest.importorskip("warp_simdata")
from warp_simdata.data_models.vtk import image_data, structured_grid
from warp_simdata.operators import iso_surface
from warp_simdata.operators import slice as simdata_slice

from tack.interop.warp import from_simdata, to_simdata

pytestmark = pytest.mark.quick


@pytest.fixture(scope="module")
def device():
    return tack.device("warp")


@pytest.fixture(scope="module")
def wp_device(device):
    return device._backend.wp_device


def _scalars(values, dtype=tack.f32):
    return tack.array(np.asarray(values, dtype.numpy_dtype))


def _two_hexes_and_a_pyramid():
    def pid(x, y, z):
        return x + 3 * y + 6 * z
    pts = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1, 2)], np.float32)
    pts = np.vstack([pts, [[0.5, 0.5, 2.0]]]).astype(np.float32)
    hexes = [[pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 1, 0), pid(i, 1, 0),
              pid(i, 0, 1), pid(i + 1, 0, 1), pid(i + 1, 1, 1), pid(i, 1, 1)] for i in (0, 1)]
    pyramid = [pid(0, 0, 1), pid(1, 0, 1), pid(1, 1, 1), pid(0, 1, 1), 12]
    topology = td.UnstructuredTopology(np.array([12, 12, 14], np.uint8), [0, 8, 16, 21],
                                       hexes[0] + hexes[1] + pyramid)
    return td.DataSet(topology, tack.array(pts, tack.vec(3, tack.f32)))


def _with_fields(data):
    positions = data.positions()
    data.fields["h"] = td.Field(td.H1(data), _scalars(positions[:, 2]))
    data.fields["v"] = td.Field(td.H1(data), tack.array(positions.astype(np.float32),
                                                        tack.vec(3, tack.f32)))
    data.fields["c"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells), tack.i32))
    return data


def _same_fields(a, b):
    assert sorted(a.fields) == sorted(b.fields)
    for name in a.fields:
        assert type(a.fields[name].space) is type(b.fields[name].space)
        np.testing.assert_array_equal(np.asarray(b.fields[name].values.numpy()),
                                      np.asarray(a.fields[name].values.numpy()))


# ── Round trips, model by model ─────────────────────────────────────

def test_image_data_with_an_offset_extent(device):
    """warp-simdata's origin is where ijk = 0 is, as VTK's; Tack's grid starts at
    its first point, with the extent's start as `offset`."""
    image = image_data.create_dataset(
        origin=wp.vec3f(1.0, 2.0, 3.0), spacing=wp.vec3f(0.5, 1.0, 2.0),
        extent_min=wp.vec3i(2, 0, 1), extent_max=wp.vec3i(5, 2, 4),
        device=str(device._backend.wp_device))
    data = from_simdata(image)
    assert isinstance(data.geometry.values, UniformCoordinates)
    assert data.topology.point_dims == (4, 3, 4) and data.topology.offset == (2, 0, 1)
    i, j, k = np.meshgrid(np.arange(2, 6), np.arange(0, 3), np.arange(1, 5), indexing="ij")
    expected = np.stack([1 + 0.5 * i, 2 + 1.0 * j, 3 + 2.0 * k], -1).transpose(2, 1, 0, 3)
    np.testing.assert_allclose(data.world_positions(), expected.reshape(-1, 3), rtol=1e-6)
    back = to_simdata(data).handle
    assert tuple(back.extent_min) == (2, 0, 1) and tuple(back.extent_max) == (5, 2, 4)
    np.testing.assert_allclose(np.asarray(back.origin), [1, 2, 3], rtol=1e-6)


def test_structured_grids_share_their_points(device, wp_device):
    with device:
        dims = (4, 3, 2)
        i, j, k = np.meshgrid(*(np.arange(d) for d in dims), indexing="ij")
        points = np.stack([i + 0.1 * j, j * j, k + 0.2 * i], -1).transpose(2, 1, 0, 3)
        data = _with_fields(td.DataSet(td.StructuredTopology(dims),
                                       tack.array(points.reshape(-1, 3).astype(np.float32),
                                                  tack.vec(3, tack.f32))))
    grid = to_simdata(data)
    assert grid.data_model.DatasetHandle is structured_grid.DatasetHandle
    assert grid.handle.points.ptr == data.geometry.values._buffer.address     # shared
    back = from_simdata(grid)
    assert back.topology.point_dims == dims
    np.testing.assert_array_equal(back.positions(), data.positions())
    _same_fields(data, back)


def test_unstructured_grids_share_their_arrays(device):
    with device:
        data = _with_fields(_two_hexes_and_a_pyramid())
    grid = to_simdata(data)
    handle = grid.handle
    assert handle.cell_connectivity.ptr == data.topology.connectivity._buffer.address
    assert handle.points.ptr == data.geometry.values._buffer.address
    back = from_simdata(grid)
    for name in ("types", "offsets", "connectivity"):
        np.testing.assert_array_equal(getattr(back.topology, name).numpy(),
                                      getattr(data.topology, name).numpy())
    assert back.topology.connectivity._buffer.address == handle.cell_connectivity.ptr
    np.testing.assert_array_equal(back.positions(), data.positions())
    _same_fields(data, back)


def test_polygons_and_polylines(device):
    with device:
        points = tack.array(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]],
                                     np.float32), tack.vec(3, tack.f32))
        polygons = td.DataSet(td.PolygonalTopology([0, 4, 7], [0, 1, 2, 3, 1, 4, 2]), points)
        lines = td.DataSet(td.PolylineTopology([0, 3, 5], [0, 1, 4, 3, 2]), points)
    for data, offsets_of in ((polygons, lambda t: t.loops()), (lines, lambda t: t.lines())):
        back = from_simdata(to_simdata(data))
        assert type(back.topology) is type(data.topology)
        for a, b in zip(offsets_of(back.topology), offsets_of(data.topology)):
            np.testing.assert_array_equal(a.numpy(), b.numpy())


def _boundary(topology):
    """The boundary faces, each as its sorted point ids: face numbering is the
    constructor's, so faces are compared by their points."""
    face_offsets, face_points = topology.arrays()[:2]
    return {tuple(sorted(face_points[face_offsets[f]:face_offsets[f + 1]]))
            for f in topology.faces().boundary().numpy()}


def test_polyhedra_keep_their_shared_faces(device):
    with device:
        poly = td.as_polyhedra(_with_fields(_two_hexes_and_a_pyramid()))
    grid = to_simdata(poly)
    assert grid.handle.faces_offsets.shape[0] - 1 == poly.topology.num_faces
    back = from_simdata(grid)
    assert isinstance(back.topology, td.PolyhedralTopology)
    assert back.topology.num_faces == poly.topology.num_faces
    assert _boundary(back.topology) == _boundary(poly.topology)   # faces renumbered, same set
    assert len(td.check_winding(back)) == 0
    _same_fields(poly, back)


def test_pieces_and_collections(device):
    with device:
        pieces = td.split(_with_fields(td.uniform_grid((6, 4, 3))), (2, 1, 1))
    collection = to_simdata(pieces)
    assert isinstance(collection, sd.DatasetCollection) and len(collection.datasets) == 2
    back = from_simdata(collection)
    assert isinstance(back, td.PartitionedDataSet) and len(back) == 2
    for a, b in zip(pieces, back):
        np.testing.assert_allclose(b.world_positions(), a.world_positions(), rtol=1e-6)
        assert b.topology.offset == a.topology.offset


def test_what_has_no_counterpart_is_left_behind(device):
    """Face fields and sets stay in Tack; f64 positions and an origin become f32
    world positions."""
    with device:
        data = td.DataSet(_two_hexes_and_a_pyramid().topology,
                          _two_hexes_and_a_pyramid().positions().astype(np.float64),
                          dtype=tack.f64, origin=(1000.0, 0.0, 0.0))
        nf = data.topology.faces().num_faces
        data.fields["flux"] = td.Field(td.Values(data, "faces"), _scalars(np.arange(nf)))
    grid = to_simdata(data)
    assert grid.get_field_names() == []
    np.testing.assert_allclose(grid.handle.points.numpy(), data.world_positions(), rtol=1e-6)


# ── Each library's filters on the other's data ──────────────────────

def _area(data):
    """A surface's area from its polygons or triangles, on the host."""
    p = data.world_positions()
    t = data.topology
    offsets, ids = (a.numpy() for a in (t.loops() if isinstance(t, td.PolygonalTopology)
                                        else (t.offsets, t.connectivity)))
    total = 0.0
    for c in range(len(offsets) - 1):
        ring = p[ids[offsets[c]:offsets[c + 1]]]
        for j in range(1, len(ring) - 1):
            total += 0.5 * np.linalg.norm(np.cross(ring[j] - ring[0], ring[j + 1] - ring[0]))
    return total


def _sphere_grid(device, n=20):
    with device:
        data = td.uniform_grid((n, n, n), spacing=(1.0 / (n - 1),) * 3)
        p = data.positions() - 0.5
        data.fields["r"] = td.Field(td.H1(data), _scalars(np.linalg.norm(p, axis=1)))
    return data


def test_tack_contours_simdata_data_as_simdata_does(device):
    data = _sphere_grid(device)
    image = to_simdata(data)
    theirs = from_simdata(iso_surface.compute(image, "r", 0.35))
    with device:
        ours = td.contour(from_simdata(image), "r", 0.35)
    assert ours.num_cells > 0
    np.testing.assert_allclose(_area(ours), _area(theirs), rtol=1e-4)


def test_simdata_slices_tack_data_as_tack_does(device):
    with device:
        data = td.as_unstructured(_sphere_grid(device)) if hasattr(td, "as_unstructured") \
            else _sphere_grid(device)
        ours = td.slice_plane(data, (0.5, 0.5, 0.4), (0.0, 0.3, 1.0))
    theirs = from_simdata(simdata_slice.compute(to_simdata(data), origin=(0.5, 0.5, 0.4),
                                                     normal=(0.0, 0.3, 1.0)))
    assert ours.num_cells > 0
    np.testing.assert_allclose(_area(theirs), _area(ours), rtol=1e-4)


def test_a_model_that_is_not_vtk_shaped(device, wp_device):
    """warp-simdata's own point cloud goes through its `to_vtk_data_model`
    (polydata of vertices) and becomes vertex cells, its fields with it."""
    from warp_simdata.data_models.custom import point_cloud

    rng = np.random.default_rng(0)
    xyz = rng.random((50, 3)).astype(np.float32)
    cloud = point_cloud.create_dataset(wp.array(xyz, dtype=wp.vec3f, device=wp_device))
    cloud.add_field("m", sd.Field.from_array(wp.array(xyz[:, 0], device=wp_device),
                                             sd.AssociationType.NODE))
    data = from_simdata(cloud)
    assert data.num_points == 50 and data.num_cells == 50
    assert set(np.unique(data.topology.types.numpy())) == {1}             # VTK_VERTEX
    np.testing.assert_array_equal(data.positions(), xyz)
    np.testing.assert_array_equal(data.fields["m"].values.numpy(), xyz[:, 0])
