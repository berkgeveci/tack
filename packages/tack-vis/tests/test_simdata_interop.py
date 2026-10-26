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


def _shares_ids(address, ptr):
    """Whether Tack's id array is warp-simdata's (two addresses), or ids are forced
    to i64 (``--ids=i64``), when Tack converts warp-simdata's int32 ids instead."""
    return address == ptr or td.ids.choose() == tack.i64


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
    assert _shares_ids(handle.cell_connectivity.ptr, data.topology.connectivity._buffer.address)
    assert handle.points.ptr == data.geometry.values._buffer.address
    back = from_simdata(grid)
    for name in ("types", "offsets", "connectivity"):
        np.testing.assert_array_equal(getattr(back.topology, name).numpy(),
                                      getattr(data.topology, name).numpy())
    assert _shares_ids(back.topology.connectivity._buffer.address, handle.cell_connectivity.ptr)
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


def _simdata_polyhedra(poly, loops_of_cell, device):
    """A warp-simdata unstructured grid of polyhedra: each cell's faces from
    ``loops_of_cell(c)`` (a list of (face id, loop)), faces numbered by id."""
    from warp_simdata.data_models.vtk import unstructured_grid
    t = poly.topology
    faces = {}
    locations, location_offsets = [], [0]
    for c in range(t.num_cells):
        for f, loop in loops_of_cell(c):
            faces[f] = loop
            locations.append(f)
        location_offsets.append(len(locations))
    face_offsets = np.concatenate([[0], np.cumsum([len(faces[f]) for f in range(len(faces))])])
    face_points = np.concatenate([faces[f] for f in range(len(faces))])
    point_offsets, point_ids = (a.numpy() for a in t.cell_points())
    def ints(a):
        return wp.array(np.asarray(a, np.int32), dtype=wp.int32, device=device)
    return unstructured_grid.create_dataset(
        wp.array(poly.positions().astype(np.float32), dtype=wp.vec3f, device=device),
        wp.full(t.num_cells, 42, dtype=wp.uint8, device=device), ints(point_offsets),
        ints(point_ids), faces_offsets=ints(face_offsets), faces_connectivity=ints(face_points),
        face_locations_offsets=ints(location_offsets),
        face_locations_connectivity=ints(locations))


def _poly(device):
    with device:
        return td.as_polyhedra(_two_hexes_and_a_pyramid())


def test_shared_polyhedral_faces_convert_on_the_device(device, wp_device):
    """Shared faces (as CGNS NFACE_n and OpenFOAM keep them) become Tack's without a
    copy: warp-simdata's face and cell-face arrays are the topology's."""
    poly = _poly(device)
    grid = to_simdata(poly)
    t = from_simdata(grid).topology
    assert _shares_ids(t.face_points._buffer.address, grid.handle.faces_connectivity.ptr)
    assert _shares_ids(t.cell_faces._buffer.address, grid.handle.face_locations_connectivity.ptr)
    assert _boundary(t) == _boundary(poly.topology) and t.num_faces == poly.topology.num_faces


def test_a_face_seen_only_from_inside_is_reversed(device, wp_device):
    """A boundary face wound into its cell: warp-simdata orients it (-1), and the
    device path reverses its loop, in a new face-points array."""
    poly = _poly(device)
    t = poly.topology
    face_offsets, face_points, cell_offsets, cell_faces, sides = t.arrays()
    boundary = set(t.faces().boundary().numpy().tolist())
    reversed_faces = sorted(boundary)[:3]

    def loops(c):
        out = []
        for f in cell_faces[cell_offsets[c]:cell_offsets[c + 1]]:
            ring = face_points[face_offsets[f]:face_offsets[f + 1]]
            out.append((int(f), ring[::-1] if f in reversed_faces else ring))
        return out
    grid = _simdata_polyhedra(poly, loops, wp_device)
    back = from_simdata(grid)
    assert back.topology.face_points._buffer.address != grid.handle.faces_connectivity.ptr
    assert _shares_ids(back.topology.cell_faces._buffer.address, grid.handle.face_locations_connectivity.ptr)
    assert _boundary(back.topology) == _boundary(t) and len(td.check_winding(back)) == 0


def test_faces_each_cell_has_a_copy_of_are_matched_on_the_host(device, wp_device):
    """As VTK files keep polyhedra: every cell its own face loops, outward. They
    are matched by point set, so shared faces come out once."""
    poly = _poly(device)
    t = poly.topology
    face_offsets, face_points, cell_offsets, cell_faces, sides = t.arrays()
    ids = iter(range(len(cell_faces)))

    def loops(c):
        out = []
        for k in range(cell_offsets[c], cell_offsets[c + 1]):
            f = cell_faces[k]
            ring = face_points[face_offsets[f]:face_offsets[f + 1]]
            out.append((next(ids), ring if sides[k] == 0 else np.roll(ring[::-1], 1)))
        return out
    back = from_simdata(_simdata_polyhedra(poly, loops, wp_device))
    assert back.topology.num_faces == t.num_faces
    assert _boundary(back.topology) == _boundary(t) and len(td.check_winding(back)) == 0


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
    types = data.topology.types
    assert isinstance(types, tack.arrays.ConstantArray)                    # stored nowhere
    assert set(np.unique(tack.arrays.to_host(types))) == {1}               # VTK_VERTEX
    np.testing.assert_array_equal(data.positions(), xyz)
    np.testing.assert_array_equal(data.fields["m"].values.numpy(), xyz[:, 0])


def test_structure_of_arrays_fields_and_positions(device, wp_device):
    """An SOA field goes across as warp-simdata's SoA field, each component shared;
    SOA positions are copied (warp-simdata's points are wp.vec3f)."""
    with device:
        base = _two_hexes_and_a_pyramid()
        p = base.positions()
        xyz = [tack.array(np.ascontiguousarray(p[:, i])) for i in range(3)]
        data = td.DataSet(base.topology, tack.soa(*xyz))
        data.fields["v"] = td.Field(td.H1(data), tack.soa(*xyz))
    grid = to_simdata(data)
    np.testing.assert_array_equal(grid.handle.points.numpy(), p)
    components = grid.get_field("v").get_data()
    assert [c.ptr for c in components] == [x._buffer.address for x in xyz]
    back = from_simdata(grid)
    np.testing.assert_array_equal(back.fields["v"].values.numpy(), p)


# ── Models Tack takes as they are (no to_vtk_data_model) ────────────

@pytest.fixture
def no_vtk_conversion(monkeypatch):
    """The direct imports never go through warp-simdata's VTK conversion."""
    from warp_simdata.data_models.vtk import conversion

    def refused(ds):
        raise AssertionError("to_vtk_data_model was called")
    monkeypatch.setattr(conversion, "to_vtk_data_model", refused)


def _stored_twin(data):
    """``data`` with its topology's arrays stored (materialized), to compute against."""
    t = data.topology
    with data.device:
        twin = td.DataSet(td.UnstructuredTopology(
            np.asarray(tack.arrays.to_host(t.types)), np.asarray(tack.arrays.to_host(t.offsets)),
            np.asarray(tack.arrays.to_host(t.connectivity)), num_points=t.num_points),
            data.positions())
    for name, field in data.fields.items():
        twin.fields[name] = td.Field(type(field.space)(twin), field.values)
    return twin


def _ints(values, device):
    return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=device)


def test_a_surface_mesh_shares_its_faces(device, wp_device, no_vtk_conversion):
    from warp_simdata.data_models.custom import surface_mesh

    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], np.float32)
    mesh = surface_mesh.create_dataset(wp.array(points, dtype=wp.vec3f, device=wp_device),
                                       _ints([0, 1, 2, 3, 1, 4, 2], wp_device),
                                       face_vertex_counts=_ints([4, 3], wp_device))
    mesh.add_field("c", sd.Field.from_array(wp.array(np.array([1.0, 2.0], np.float32),
                                                     device=wp_device),
                                            sd.AssociationType.ELEMENT))
    data = from_simdata(mesh)
    t = data.topology
    assert isinstance(t, td.PolygonalTopology)
    offsets, loops = t.loops()
    assert _shares_ids(loops._buffer.address, mesh.handle.face_vertex_indices.ptr)      # shared
    np.testing.assert_array_equal(offsets.numpy(), [0, 4, 7])
    np.testing.assert_array_equal(data.fields["c"].values.numpy(), [1, 2])


def _openfoam(poly, device):
    """Tack's polyhedra as an OpenFOAM polyMesh: internal faces first, owner below
    neighbour, each face wound out of its owner."""
    from warp_simdata.data_models.openfoam import polymesh

    t = poly.topology
    face_offsets, face_points = t.arrays()[:2]
    sides = t.faces().sides.numpy()
    internal, boundary = [], []
    for f in range(t.num_faces):
        ring = face_points[face_offsets[f]:face_offsets[f + 1]]
        a, b = int(sides[f][0]), int(sides[f][2])
        if b < 0:
            boundary.append((a, -1, ring))
        elif a < b:
            internal.append((a, b, ring))
        else:
            internal.append((b, a, np.roll(ring[::-1], 1)))
    faces = sorted(internal, key=lambda e: (e[0], e[1])) + sorted(boundary, key=lambda e: e[0])
    counts = [len(ring) for _, _, ring in faces]
    return polymesh.create_dataset(
        wp.array(poly.positions().astype(np.float32), dtype=wp.vec3f, device=device),
        _ints(np.concatenate([ring for _, _, ring in faces]), device),
        _ints([o for o, _, _ in faces], device), _ints([n for _, n, _ in internal], device),
        _ints(np.concatenate([[0], np.cumsum(counts)]), device))


def test_an_openfoam_mesh_is_tacks_polyhedra(device, wp_device, no_vtk_conversion):
    """polyMesh's arrays are a PolyhedralTopology's: shared, the sides from owner."""
    with device:
        poly = td.as_polyhedra(_two_hexes_and_a_pyramid())
    mesh = _openfoam(poly, wp_device)
    data = from_simdata(mesh)
    t = data.topology
    assert isinstance(t, td.PolyhedralTopology)
    assert _shares_ids(t.face_points._buffer.address, mesh.handle.faces.ptr)
    assert _shares_ids(t.cell_faces._buffer.address, mesh.handle.cell_faces.ptr)
    assert t.num_faces == poly.topology.num_faces and len(td.check_winding(data)) == 0
    assert _boundary(t) == _boundary(poly.topology)
    np.testing.assert_allclose(td.algorithms.cell_geometry(data)[0].values.numpy(),
                               td.algorithms.cell_geometry(poly)[0].values.numpy(), rtol=1e-6)


def _cgns(points, element_type, connectivity, count, device, start=None):
    from warp_simdata.data_models.sids import unstructured as sids

    return sids.create_dataset(
        wp.array(points.astype(np.float32), dtype=wp.vec3f, device=device), element_type,
        wp.vec2i(1, count), _ints(connectivity, device),
        None if start is None else _ints(start, device))


def test_a_cgns_section_of_one_shape(device, wp_device, no_vtk_conversion):
    """Constant types, counting offsets, and the 1-based connectivity read 0-based by
    a Transform: nothing copied, and filters compute as on stored arrays."""
    base = _two_hexes_and_a_pyramid()
    conn = base.topology.connectivity.numpy()[:16]
    section = _cgns(base.positions(), 17, conn + 1, 2, wp_device)        # HEXA_8
    section.add_field("h", sd.Field.from_array(
        wp.array(base.positions()[:, 2].copy(), device=wp_device), sd.AssociationType.NODE))
    data = from_simdata(section)
    t = data.topology
    assert isinstance(t.types, tack.arrays.ConstantArray)
    assert isinstance(t.offsets, tack.arrays.CountingArray)
    assert isinstance(t.connectivity, tack.arrays.Transform)
    np.testing.assert_array_equal(tack.arrays.to_host(t.connectivity), conn)
    assert data.geometry.values._buffer.address == section.handle.grid_coords.ptr
    twin = _stored_twin(data)
    for call in (lambda d: td.algorithms.cell_geometry(d)[0].values.numpy(),
                 lambda d: td.contour(d, "h", 0.5).positions()):
        np.testing.assert_array_equal(call(data), call(twin))


def test_a_cgns_mixed_section_is_read_lazily(device, wp_device, no_vtk_conversion):
    """MIXED: the type codes inline in the connectivity. Types through two
    permutations (the codes, then CGNS to VTK), offsets less one code per earlier
    element, point ids by a binary search; computes as stored arrays do."""
    base = _two_hexes_and_a_pyramid()
    t0 = base.topology
    offsets, conn = t0.offsets.numpy(), t0.connectivity.numpy()
    codes = {12: 17, 14: 12}
    mixed, start = [], [0]
    for c, vtk in enumerate(t0.types.numpy()):
        mixed += [codes[int(vtk)], *(conn[offsets[c]:offsets[c + 1]] + 1)]
        start.append(len(mixed))
    section = _cgns(base.positions(), 20, mixed, 3, wp_device, start)          # MIXED
    section.add_field("h", sd.Field.from_array(
        wp.array(base.positions()[:, 2].copy(), device=wp_device), sd.AssociationType.NODE))
    data = from_simdata(section)
    t = data.topology
    np.testing.assert_array_equal(tack.arrays.to_host(t.types), [12, 12, 14])
    np.testing.assert_array_equal(tack.arrays.to_host(t.offsets), offsets)
    np.testing.assert_array_equal(tack.arrays.to_host(t.connectivity), conn)
    twin = _stored_twin(data)
    for call in (lambda d: td.algorithms.cell_geometry(d)[0].values.numpy(),
                 lambda d: td.contour(d, "h", 0.5).positions()):
        np.testing.assert_array_equal(call(data), call(twin))


def test_point_clouds_are_implicit_vertex_cells(device, wp_device, no_vtk_conversion):
    from warp_simdata.data_models.custom import point_cloud

    xyz = np.random.default_rng(1).random((20, 3)).astype(np.float32)
    data = from_simdata(point_cloud.create_dataset(wp.array(xyz, dtype=wp.vec3f,
                                                            device=wp_device)))
    t = data.topology
    assert all(isinstance(a, (tack.arrays.ConstantArray, tack.arrays.CountingArray))
               for a in (t.types, t.offsets, t.connectivity))
    np.testing.assert_array_equal(data.positions(), xyz)


# ── FLASH: an AMR hierarchy of revolved boxes ───────────────────────

def _flash(device, angle_range=(0.0, np.pi)):
    """Two root blocks stacked in z, the lower refined into four: 4 x 4 cells each,
    eight sectors over half a turn. Cell values are block * 100 + cell."""
    from warp_simdata.data_models.flash import axisymmetric

    bounds = np.array([[0, 1, 0, 1], [0, 1, 1, 2],
                       [0, .5, 0, .5], [.5, 1, 0, .5], [0, .5, .5, 1], [.5, 1, .5, 1]],
                      np.float32)
    none = -1
    gid = np.array([[none, none, none, 2, none, 3, 4, 5, 6],
                    [none, none, 1, none, none, none, none, none, none],
                    [none, 4, none, 5, 1, none, none, none, none],
                    [3, none, none, 6, 1, none, none, none, none],
                    [none, 6, 3, 2, 1, none, none, none, none],
                    [5, none, 4, 2, 1, none, none, none, none]], np.int32)
    ds = axisymmetric.create_dataset(
        wp.array(bounds, dtype=wp.vec4f, device=device), wp.array(gid, device=device),
        _ints([2, 1, 1, 1, 1, 1], device), _ints([1, 2, 3, 4, 5], device),
        cell_dims=wp.vec2i(4, 4), angular_cells=8, angle_range=angle_range)
    values = (np.arange(6)[:, None] * 100 + np.arange(16)[None, :]).astype(np.float32)
    source = sd.Field.from_array(wp.array(values.reshape(-1), device=device),
                                 sd.AssociationType.ELEMENT)
    ds.add_field("f", axisymmetric.create_element_field(ds, source))
    return ds, values


def test_flash_is_an_amr_hierarchy_of_revolved_boxes(device, wp_device):
    ds, values = _flash(wp_device)
    amr = from_simdata(ds)
    assert isinstance(amr, td.AMRDataSet) and len(amr.levels) == 2
    assert [tuple(r) for r in np.atleast_2d(amr.ratios)] == [(2, 2, 1)]
    assert [b.topology.offset for b in amr.levels[0]] == [(0, 0, 0), (0, 4, 0)]
    assert [b.topology.offset for b in amr.levels[1]] == \
        [(0, 0, 0), (4, 0, 0), (0, 4, 0), (4, 4, 0)]
    # The refined root's cells are marked, the other root's not.
    refined = [(b.ghosts.cells.numpy() & td.Ghosts.REFINEDCELL) != 0 for b in amr.levels[0]]
    assert refined[0].all() and not refined[1].any()


def test_flash_boxes_are_warp_simdatas_points_and_values(device, wp_device):
    """Each leaf box's points are warp-simdata's revolved nodes (same order for a
    partial turn), and each cell holds its meridional cell's value in every
    sector, read from the packed field without a copy."""
    from warp_simdata.data_models.vtk.conversion import to_vtk_data_model

    ds, values = _flash(wp_device)
    amr = from_simdata(ds)
    nodes = to_vtk_data_model(ds).handle.points.numpy()
    per_block = 5 * 5 * 9
    boxes = {0: amr.levels[0][0], 1: amr.levels[0][1], 2: amr.levels[1][0],
             3: amr.levels[1][1], 4: amr.levels[1][2], 5: amr.levels[1][3]}
    for leaf, block in enumerate([1, 2, 3, 4, 5]):
        np.testing.assert_allclose(boxes[block].positions(),
                                   nodes[leaf * per_block:(leaf + 1) * per_block], atol=2e-7)
    for block, box in boxes.items():
        f = box.fields["f"].values
        assert isinstance(f, tack.arrays._Implicit)                # nothing stored
        np.testing.assert_array_equal(tack.arrays.to_host(f), np.tile(values[block], 8))


def test_a_probe_finds_the_finest_flash_cell(device, wp_device):
    ds, _ = _flash(wp_device)
    amr = from_simdata(ds)
    r, z, a = 0.8, 0.8, 0.3                 # block 5, cell (2, 2): 5 * 100 + 10
    out = td.probe(amr, np.array([[r * np.cos(a), z, r * np.sin(a)]]))
    assert out.fields["valid"].values.numpy()[0] == 1
    assert out.fields["f"].values.numpy()[0] == 510


def test_flash_over_a_full_turn_filters_and_reduces(device, wp_device):
    """A full revolution (its seam's points repeated in each box) and the boxes on
    the axis (degenerate hexahedra there): contours run on every box, and the
    hierarchy's statistics count each leaf cell once per sector, refined cells
    left out."""
    ds, values = _flash(wp_device, angle_range=(0.0, 2.0 * np.pi))
    amr = from_simdata(ds)
    for level, box in amr:
        with device:
            box.fields["p"] = td.project(box, "f", td.H1(box))
            iso = td.contour(box, "p", float(np.median(values)))
        assert iso.num_points >= 0
    stats = td.statistics(amr, "f")
    leaves = values[[1, 2, 3, 4, 5]]
    assert stats.n == leaves.size * 8
    np.testing.assert_allclose(stats.mean, leaves.mean(), rtol=1e-6)


# ── Curves, patches, NGON_n and NFACE_n, EnSight, corner-point grids ──

def _cell_faces(poly):
    """Each Tack polyhedron's faces as (face id, outward 0 or inward 1)."""
    t = poly.topology
    face_offsets, face_points, cell_offsets, cell_faces, sides = t.arrays()
    return face_offsets, face_points, cell_offsets, cell_faces, sides


def test_curves_are_polylines(device, wp_device, no_vtk_conversion):
    from warp_simdata.data_models.custom import curves

    xyz = np.random.default_rng(2).random((9, 3)).astype(np.float32)
    ds = curves.create_dataset(wp.array(xyz, dtype=wp.vec3f, device=wp_device),
                               _ints([4, 2, 3], wp_device))
    data = from_simdata(ds)
    t = data.topology
    assert isinstance(t, td.PolylineTopology)
    assert isinstance(t.line_points, tack.arrays.CountingArray)
    np.testing.assert_array_equal(t.line_offsets.numpy(), [0, 4, 6, 9])
    np.testing.assert_array_equal(data.positions(), xyz)


def test_an_openfoam_patch_is_a_view_of_its_mesh(device, wp_device, no_vtk_conversion):
    from warp_simdata.data_models.openfoam import boundary_mesh

    with device:
        poly = td.as_polyhedra(_two_hexes_and_a_pyramid())
    mesh = _openfoam(poly, wp_device)
    h = mesh.handle
    internal = int(h.neighbour.shape[0])
    patch = boundary_mesh.create_dataset(h.points, h.faces, h.face_offsets, internal,
                                         int(h.face_offsets.shape[0]) - 1 - internal)
    data = from_simdata(patch)
    t = data.topology
    assert isinstance(t, td.PolygonalTopology)
    offsets, loops = t.loops()
    assert isinstance(loops, tack.arrays._Implicit) and int(offsets.numpy()[0]) == 0
    expected = h.faces.numpy()[h.face_offsets.numpy()[internal]:]
    np.testing.assert_array_equal(tack.arrays.to_host(loops), expected)
    assert data.num_cells == poly.topology.faces().boundary().numpy().size


def _ngon_nface(poly, device, split=None):
    """Tack's polyhedra as CGNS NGON_n faces (in one block, or two at ``split``)
    and NFACE_n cells, entries signed by side (negative: into the cell)."""
    from warp_simdata.data_models.sids import nface_n
    from warp_simdata.data_models.sids import unstructured as sids

    face_offsets, face_points, cell_offsets, cell_faces, sides = _cell_faces(poly)
    nf = len(face_offsets) - 1
    coords = wp.array(poly.positions().astype(np.float32), dtype=wp.vec3f, device=device)
    ranges = [(0, nf)] if split is None else [(0, split), (split, nf)]
    blocks = []
    for lo, hi in ranges:
        conn = face_points[face_offsets[lo]:face_offsets[hi]] + 1
        blocks.append(sids.create_handle(coords, 22, wp.vec2i(lo + 1, hi), _ints(conn, device),
                                         _ints(face_offsets[lo:hi + 1] - face_offsets[lo], device)))
    entries = np.where(sides == 0, cell_faces + 1, -(cell_faces + 1))
    nface = sids.create_handle(coords, 23, wp.vec2i(nf + 1, nf + len(cell_offsets) - 1),
                               _ints(entries, device), _ints(cell_offsets, device))
    return nface_n.create_dataset(nface, blocks)


@pytest.mark.parametrize("split", [None, 6])
def test_cgns_nface_polyhedra_are_read_lazily(device, wp_device, split, no_vtk_conversion):
    """NGON_n blocks joined lazily, each cell's faces through the blocks' ranges,
    the sides from the signs: the same polyhedra, nothing copied."""
    with device:
        poly = td.as_polyhedra(_two_hexes_and_a_pyramid())
    data = from_simdata(_ngon_nface(poly, wp_device, split))
    t = data.topology
    assert isinstance(t, td.PolyhedralTopology)
    assert all(isinstance(a, tack.arrays._Implicit)
               for a in (t.face_points, t.cell_faces, t.cell_face_sides))
    for a, b in zip(t.arrays(), poly.topology.arrays()):
        np.testing.assert_array_equal(a, b)
    assert len(td.check_winding(data)) == 0
    np.testing.assert_allclose(td.algorithms.cell_geometry(data)[0].values.numpy(),
                               td.algorithms.cell_geometry(poly)[0].values.numpy(), rtol=1e-6)


def test_cgns_ngon_polygons(device, wp_device, no_vtk_conversion):
    from warp_simdata.data_models.sids import unstructured as sids

    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], np.float32)
    ds = sids.create_dataset(wp.array(points, dtype=wp.vec3f, device=wp_device), 22,
                             wp.vec2i(1, 2), _ints([1, 2, 3, 4, 2, 5, 3], wp_device),
                             _ints([0, 4, 7], wp_device))
    t = from_simdata(ds).topology
    assert isinstance(t, td.PolygonalTopology)
    np.testing.assert_array_equal(tack.arrays.to_host(t.loops()[1]), [0, 1, 2, 3, 1, 4, 2])


def test_ensight_parts(device, wp_device, no_vtk_conversion):
    """Pieces of linear shapes joined lazily (0-based), and a part of polygons."""
    from warp_simdata.data_models.ensight_gold import unstructured_part as ensight

    base = _two_hexes_and_a_pyramid()
    conn = base.topology.connectivity.numpy() + 1
    coords = wp.array(base.positions().astype(np.float32), dtype=wp.vec3f, device=wp_device)
    part = ensight.create_dataset(coords, [ensight.create_piece_handle(13, _ints(conn[:16], wp_device)),
                                           ensight.create_piece_handle(9, _ints(conn[16:], wp_device))])
    data = from_simdata(part)
    t = data.topology
    np.testing.assert_array_equal(tack.arrays.to_host(t.types), [12, 12, 14])
    np.testing.assert_array_equal(tack.arrays.to_host(t.offsets), [0, 8, 16, 21])
    np.testing.assert_array_equal(tack.arrays.to_host(t.connectivity), conn - 1)
    twin = _stored_twin(data)
    np.testing.assert_array_equal(td.algorithms.cell_geometry(data)[0].values.numpy(),
                                  td.algorithms.cell_geometry(twin)[0].values.numpy())
    polygons = ensight.create_dataset(coords, [ensight.create_piece_handle(
        15, _ints([1, 2, 5, 4, 2, 3, 6], wp_device), element_node_counts=_ints([4, 3], wp_device))])
    t = from_simdata(polygons).topology
    assert isinstance(t, td.PolygonalTopology)
    np.testing.assert_array_equal(tack.arrays.to_host(t.loops()[1]), [0, 1, 4, 3, 1, 2, 5])


@pytest.mark.parametrize("depth", ["zUp", "zDown"])
def test_a_corner_point_grid_is_computed_where_read(device, wp_device, depth):
    """Hexahedra stored nowhere, points from COORD and ZCORN (a fault: the two
    cells' shared face at different depths), ACTNUM's inactive cell hidden; the
    points are warp-simdata's."""
    from warp_simdata.data_models.reservoir import corner_point_grid
    from warp_simdata.data_models.vtk.conversion import to_vtk_data_model

    nx, ny, nz = 2, 1, 2
    coord = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord += [i * 1.0, j * 1.0 + 0.1 * i, 0.0, i * 1.0 + 0.2, j * 1.0, 10.0]
    zcorn = np.zeros((2 * nz, 2 * ny, 2 * nx), np.float32)
    for k in range(2 * nz):
        zcorn[k] = 1.0 + 2.0 * ((k + 1) // 2)
    zcorn[:, :, 2:] += 0.5                                   # the fault: cells of i = 1 lower
    actnum = np.ones(nx * ny * nz, np.int32)
    actnum[3] = 0
    ds = corner_point_grid.create_dataset(
        wp.vec3i(nx, ny, nz), wp.array(np.asarray(coord, np.float32), device=wp_device),
        wp.array(zcorn.reshape(-1), device=wp_device), _ints(actnum, wp_device),
        depth_direction=depth)
    data = from_simdata(ds)
    t = data.topology
    assert all(isinstance(a, (tack.arrays.ConstantArray, tack.arrays.CountingArray))
               for a in (t.types, t.offsets, t.connectivity))
    expected = to_vtk_data_model(ds).handle.points.numpy()
    np.testing.assert_allclose(data.positions(), expected, rtol=1e-6, atol=1e-6)
    hidden = (data.ghosts.cells.numpy() & td.Ghosts.HIDDENCELL) != 0
    np.testing.assert_array_equal(hidden, actnum == 0)
    volumes = td.algorithms.cell_geometry(data)[0].values.numpy()
    assert (volumes[actnum == 1] > 0).all()
