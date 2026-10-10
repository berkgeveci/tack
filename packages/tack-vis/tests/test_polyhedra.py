"""Tests for tack.data.polyhedra: the polyhedral topology, phase 1.

Shape-based meshes converted to polyhedra must give back exactly what the
shape path derives -- faces, sides, boundary, edges and their numbering --
and wind consistently. VTK checks the export (each cell's own outward copy
of its faces: positive volumes, the same external faces) and the import
(duplicated faces matched). MPAS-like columns -- Voronoi polygons extruded
into thin layers -- give polygonal faces of any size. Polyhedra from VTK's
own test meshes, copied into data/vtk/, are read and checked.
"""

import os
import types

import numpy as np
import pytest
from test_dataset_api import (
    SOLID_TYPES,
    _cell_points,
    _height,
    _scalars,
    _two_hexes_and_a_pyramid,
)

import tack
import tack.data as td
from tack.runtime.dispatch import env_flag

try:
    from vtkmodules import vtkCommonDataModel
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(vtkGeometryFilter=vtkGeometryFilter,
                                vtkCellTypeSource=vtkCellTypeSource,
                                vtkPolyhedron=vtkCommonDataModel.vtkPolyhedron)

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")


def _grids():
    return {"mixed": _two_hexes_and_a_pyramid(),
            "rectilinear": td.rectilinear_grid([0, 1, 3, 4], [0, 2, 3], [0, 1, 1.5])}


def _faces_by_points(topology):
    """Each face as (sorted point ids) -> (cell on side 0, cell on side 1 or -1)."""
    face_offsets, face_points = topology.arrays()[:2]
    sides = topology.faces().sides.to_numpy(vectors=True)
    return {tuple(sorted(face_points[face_offsets[f]:face_offsets[f + 1]])):
            (int(sides[f][0]), int(sides[f][2])) for f in range(topology.num_faces)}


# ── From the shape path ─────────────────────────────────────────────

def _check_same_as_shape_path(data):
    poly = td.as_polyhedra(data)
    t, shape = poly.topology, data.topology
    assert isinstance(t, td.PolyhedralTopology)
    assert (t.num_cells, t.num_points) == (shape.num_cells, shape.num_points)
    assert t.num_faces == shape.faces().num_faces
    np.testing.assert_array_equal(t.faces().sides.to_numpy(vectors=True),
                                  shape.faces().sides.to_numpy(vectors=True))
    np.testing.assert_array_equal(t.faces().boundary().to_numpy(),
                                  shape.faces().boundary().to_numpy())
    np.testing.assert_array_equal(t.edges().rows.to_numpy(vectors=True),
                                  shape.edges().rows.to_numpy(vectors=True))
    offsets, ids = (a.to_numpy() for a in t.cell_points())
    expected = [sorted(set(row)) for row in _cell_points(data)] if hasattr(
        shape, "connectivity") or len(_cell_points(data)) == data.num_cells else None
    for c, want in enumerate(expected):
        assert ids[offsets[c]:offsets[c + 1]].tolist() == want
    assert td.check_winding(poly).size == 0
    return poly


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_shape_meshes_as_polyhedra(backend, make):
    data = _grids()[make]
    data.fields["u"] = _height(data)
    data.fields["c"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    nf, ne = data.topology.faces().num_faces, data.topology.edges().num_edges
    data.fields["flux"] = td.Field(td.Values(data, "faces"), _scalars(np.arange(nf)))
    data.fields["circulation"] = td.Field(td.Values(data, "edges"), _scalars(np.arange(ne)))
    td.algorithms.boundary_faces(data)
    poly = _check_same_as_shape_path(data)
    t = poly.topology
    assert poly.fields["u"].space is td.H1(poly) and poly.fields["c"].space is td.Constant(poly)
    assert poly.fields["flux"].space is td.Values(poly, "faces")
    np.testing.assert_array_equal(poly.fields["flux"].values.to_numpy(), np.arange(nf))
    assert poly.fields["circulation"].space.size == ne
    np.testing.assert_array_equal(poly.sets["boundary"].to_numpy(),
                                  t.faces().boundary().to_numpy())


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_every_solid_as_polyhedra(backend, kind):
    source = vtk.vtkCellTypeSource()
    source.SetCellType(SOLID_TYPES[kind])
    source.SetBlocksDimensions(3, 2, 2)
    source.Update()
    _check_same_as_shape_path(vtk_to_dataset(source.GetOutput()))


@tack.kernel
def _walks(cells, out, width):
    # Every cell's faces as it walks them, point by point, outward.
    for c in cells:
        at = c * width
        for k in range(cells.num_faces(c)):
            for j in range(cells.face_size(cells.face_id(c, k))):
                out[at] = cells.side_point(c, k, j)
                at += 1


def test_cells_walk_their_faces_outward(backend):
    """side_point gives each cell its faces as the shape does: outward, the same
    rings (up to where they start) as the shape's own face tables."""
    data = _two_hexes_and_a_pyramid()
    poly = td.as_polyhedra(data)
    width = 6 * 4
    out = _scalars(np.full(3 * width, -1), tack.i32)
    td.for_each(_walks, poly, "cells", out, width)
    walks = out.to_numpy().reshape(3, width)
    hex_faces = [(0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1),
                 (4, 5, 6, 7)]
    pyramid_faces = [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    for c, (faces, row) in enumerate(zip((hex_faces, hex_faces, pyramid_faces),
                                          _cell_points(data))):
        at = 0
        for face in faces:
            ring = [int(row[i]) for i in face]
            walked = walks[c, at:at + len(ring)].tolist()
            at += len(ring)
            start = walked.index(ring[0])
            assert walked[start:] + walked[:start] == ring


# ── Winding ─────────────────────────────────────────────────────────

def test_check_winding_finds_a_backwards_face(backend):
    poly = td.as_polyhedra(_two_hexes_and_a_pyramid())
    face_offsets, face_points, cell_offsets, cell_faces, sides = poly.topology.arrays()
    # Turn one of the pyramid's triangles (a boundary face) around, without
    # changing its side. (Its first face, the base, is the hexahedron's top.)
    f = cell_faces[cell_offsets[2] + 1]
    face_points = face_points.copy()
    face_points[face_offsets[f]:face_offsets[f + 1]] = face_points[
        face_offsets[f]:face_offsets[f + 1]][::-1]
    bad = td.PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, sides)
    assert td.check_winding(bad).tolist() == [2]
    # ... and orient does not mend it: the face still has its side 0, just wrong.
    assert 2 in td.check_winding(td.orient(bad)).tolist()


def test_orient_turns_inward_boundary_faces(backend):
    poly = td.as_polyhedra(_two_hexes_and_a_pyramid())
    face_offsets, face_points, cell_offsets, cell_faces, sides = poly.topology.arrays()
    boundary = poly.topology.faces().boundary().to_numpy()
    flipped = boundary[::3]
    face_points = face_points.copy()
    for f in flipped:
        a, b = face_offsets[f], face_offsets[f + 1]
        face_points[a:b] = face_points[a:b][::-1]
    sides = np.where(np.isin(cell_faces, flipped), 1, sides).astype(np.uint8)
    inward = td.PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, sides)
    with pytest.raises(ValueError, match="wound into their only cell"):
        inward.faces()
    mended = td.orient(inward)
    np.testing.assert_array_equal(mended.faces().boundary().to_numpy(), boundary)
    assert td.check_winding(mended).size == 0
    assert _faces_by_points(mended) == _faces_by_points(poly.topology)


def test_what_a_polyhedral_topology_refuses(backend):
    poly = td.as_polyhedra(_two_hexes_and_a_pyramid())
    face_offsets, face_points, cell_offsets, cell_faces, sides = poly.topology.arrays()
    twice = sides.copy()
    shared = int(np.flatnonzero(poly.topology.faces().sides.to_numpy(vectors=True)[:, 2]
                                >= 0)[0])
    twice[cell_faces == shared] = 0                  # both cells claim side 0
    with pytest.raises(ValueError, match="same side of two cells"):
        td.PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces,
                              twice).faces()
    wrong = cell_faces.copy()
    wrong[0] = 999
    with pytest.raises(ValueError, match="name no face"):
        td.PolyhedralTopology(face_offsets, face_points, cell_offsets, wrong, sides).faces()
    with pytest.raises(NotImplementedError, match="reference element"):
        td.L2(poly).attributes()
    with pytest.raises(NotImplementedError, match="reference element"):
        td.H1(poly, order=2).attributes()


# ── MPAS-like columns: Voronoi polygons in thin layers ──────────────

def _voronoi_columns(n=40, layers=3, thickness=0.01, seed=7):
    """Voronoi cells of random points, kept where bounded, extruded into ``layers``
    thin layers: prisms with polygonal tops and bottoms, quads around. Skips the
    test without SciPy (in the dev extra)."""
    Voronoi = pytest.importorskip("scipy.spatial").Voronoi

    rng = np.random.default_rng(seed)
    vor = Voronoi(rng.uniform(0, 1, (n, 2)))
    regions = [r for r in (vor.regions[i] for i in vor.point_region)
               if r and -1 not in r and all((0 <= vor.vertices[v]).all()
                                            and (vor.vertices[v] <= 1).all() for v in r)]
    used = sorted({v for r in regions for v in r})
    renumber = {v: i for i, v in enumerate(used)}
    nv = len(used)
    xy = vor.vertices[used]
    points = np.array([[x, y, k * thickness] for k in range(layers + 1) for x, y in xy])
    cells = []
    for r in regions:
        ring = [renumber[v] for v in r]
        area = sum(xy[a][0] * xy[b][1] - xy[b][0] * xy[a][1]
                   for a, b in zip(ring, ring[1:] + ring[:1]))
        if area < 0:
            ring = ring[::-1]                         # counter-clockwise from above
        for k in range(layers):
            lo, hi = k * nv, (k + 1) * nv
            faces = [[lo + v for v in ring[::-1]],    # bottom, facing down
                     [hi + v for v in ring]]          # top, facing up
            faces += [[lo + a, lo + b, hi + b, hi + a]  # sides, facing out
                      for a, b in zip(ring, ring[1:] + ring[:1])]
            cells.append(faces)
    return points, cells, len(regions)


def _from_cell_faces(cells, num_points):
    return td.PolyhedralTopology.from_cell_faces(cells, num_points)


def test_voronoi_columns(backend):
    points, cells, columns = _voronoi_columns()
    t = _from_cell_faces(cells, len(points))
    data = td.DataSet(t, points)
    assert td.check_winding(data).size == 0
    sizes = np.diff(t.arrays()[0])
    assert sizes.max() >= 6                           # polygonal faces, not just quads
    faces = t.faces()
    sides = faces.sides.to_numpy(vectors=True)
    # Each column's layers share their tops and bottoms; neighbours their sides.
    assert (sides[:, 2] >= 0).sum() == faces.num_faces - faces.boundary().shape[0]
    assert t.num_cells == 3 * columns
    # Euler for a closed ball of polyhedra: V - E + F - C = 1.
    assert t.num_points - t.edges().num_edges + t.num_faces - t.num_cells == 1


@needs_vtk
def test_voronoi_columns_through_vtk(backend):
    points, cells, _ = _voronoi_columns()
    volumes = _check_vtk_round_trip(td.DataSet(_from_cell_faces(cells, len(points)), points))
    # Thin cells, all right side out: the case geometric orientation tests get wrong.
    assert len(volumes) == len(cells)


# ── VTK ─────────────────────────────────────────────────────────────

def _check_vtk_round_trip(data):
    """Export: VTK sees each cell right side out (positive volume) and finds our
    boundary. Import: the duplicated faces come back as the same shared faces."""
    grid = dataset_to_vtk(data)
    t = data.topology
    volumes = []
    for c in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(c)
        assert cell.GetCellType() == 42
        volumes.append(cell.ComputeVolume())
    assert min(volumes) > 0
    surface = vtk.vtkGeometryFilter()
    surface.SetInputData(grid)
    surface.Update()
    assert surface.GetOutput().GetNumberOfCells() == t.faces().boundary().shape[0]
    back = vtk_to_dataset(grid)
    assert isinstance(back.topology, td.PolyhedralTopology)
    ours = _faces_by_points(t)
    theirs = _faces_by_points(back.topology)
    assert ours.keys() == theirs.keys()
    assert all(set(ours[k]) == set(theirs[k]) for k in ours)
    assert td.check_winding(back).size == 0
    np.testing.assert_allclose(back.positions(), data.positions(), atol=1e-6)
    return volumes


@needs_vtk
@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_vtk_round_trip(backend, make):
    data = _grids()[make]
    poly = td.as_polyhedra(data)
    poly.fields["c"] = td.Field(td.Constant(poly), _scalars(np.arange(poly.num_cells)))
    volumes = _check_vtk_round_trip(poly)
    if make == "mixed":
        np.testing.assert_allclose(volumes, [1, 1, 1 / 3], rtol=1e-6)
    grid = dataset_to_vtk(poly)
    assert grid.GetCellData().GetArray("c").GetNumberOfTuples() == poly.num_cells


@needs_vtk
def test_vtk_refuses_faces_wound_the_same_way(backend):
    poly = td.as_polyhedra(_two_hexes_and_a_pyramid())
    grid = dataset_to_vtk(poly)
    faces = grid.GetPolyhedronFaces()
    from vtkmodules.util.numpy_support import vtk_to_numpy
    connectivity = vtk_to_numpy(faces.GetConnectivityArray())
    offsets = vtk_to_numpy(faces.GetOffsetsArray())
    # Turn one copy of the shared hex-hex face around: both cells then claim it.
    sides = poly.topology.faces().sides.to_numpy(vectors=True)
    shared = int(np.flatnonzero(sides[:, 2] >= 0)[0])
    cell1, local1 = int(sides[shared][2]), int(sides[shared][3])
    copy = poly.topology.arrays()[2][cell1] + local1
    a, b = offsets[copy], offsets[copy + 1]
    connectivity[a:b] = connectivity[a:b][::-1].copy()
    with pytest.raises(ValueError, match="wound the same way"):
        vtk_to_dataset(grid)


# Copies of VTK test meshes, with VTK's license; see data/vtk/README.md.
_VTK_DATA = os.path.join(os.path.dirname(__file__), "data", "vtk")


@needs_vtk
@pytest.mark.parametrize("name", ["onePolyhedron.vtu", "polyhedron2pieces.vtu",
                                  "polyhedron_mesh.vtu", "concavePolyhedron.vtu",
                                  "sliceOfPolyhedron.vtu", "vtkHDF/polyhedron.vtu",
                                  "nonWatertightPolyhedron.vtu"])
def test_polyhedra_from_vtk_data(backend, name):
    path = os.path.join(_VTK_DATA, name)
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    grid = reader.GetOutput()
    if name == "polyhedron_mesh.vtu":
        # Wound inconsistently: each cell's faces point both ways, and the face the
        # two cells share runs the same way in both. Refused, not guessed at.
        with pytest.raises(ValueError, match="wound the same way"):
            vtk_to_dataset(grid)
        return
    data = vtk_to_dataset(grid)
    assert isinstance(data.topology, td.PolyhedralTopology)
    assert data.num_cells == grid.GetNumberOfCells()
    surface = vtk.vtkGeometryFilter()
    surface.SetInputData(grid)
    surface.Update()
    assert data.topology.faces().boundary().shape[0] == surface.GetOutput().GetNumberOfCells()
    assert td.check_winding(data).size == 0


# ── Phase 2: face-based algorithms on both paths ────────────────────

alg = td.algorithms


def _meshes_for_fv():
    meshes = dict(_grids())
    if vtk is not None:
        for kind in ("tetra", "wedge", "pyramid", "hexahedron", "voxel"):
            source = vtk.vtkCellTypeSource()
            source.SetCellType(SOLID_TYPES[kind])
            source.SetBlocksDimensions(2, 2, 2)
            source.Update()
            meshes[kind] = vtk_to_dataset(source.GetOutput())
    return meshes


def _fv_results(data, flux, cell_values, velocity=(1.0, 0.5, -0.25)):
    normals, areas = alg.face_geometry(data)
    volumes, centroids = alg.cell_geometry(data)
    return {
        "normals": normals.values.to_numpy(vectors=True),
        "areas": areas.values.to_numpy(),
        "centers": alg.face_centers(data).values.to_numpy(vectors=True),
        "volumes": volumes.values.to_numpy(),
        "centroids": centroids.values.to_numpy(vectors=True),
        "divergence": alg.divergence(data, td.Field(td.Values(data, "faces"),
                                                    flux)).values.to_numpy(),
        "jump": alg.jump(data, td.Field(td.Constant(data), cell_values)).values.to_numpy(),
        "upwind": alg.upwind_flux(data, td.Field(td.Constant(data), cell_values),
                                  velocity).values.to_numpy(),
        "perot": alg.perot(data, td.Field(td.Values(data, "faces"),
                                          flux)).values.to_numpy(vectors=True),
        "boundary": alg.boundary_faces(data).to_numpy(),
    }


@pytest.mark.parametrize("name", ["mixed", "rectilinear", "tetra", "wedge", "pyramid",
                                  "hexahedron", "voxel"])
def test_face_based_algorithms_run_on_both_paths(backend, name):
    """The same algorithms, from one source, on a mesh and the same mesh as polyhedra:
    the same answers."""
    meshes = _meshes_for_fv()
    if name not in meshes:
        pytest.skip("needs VTK for this mesh")
    data = meshes[name]
    poly = td.as_polyhedra(data)
    rng = np.random.default_rng(4)
    nf = data.topology.faces().num_faces
    flux = rng.uniform(-1, 1, nf)
    cells = rng.uniform(-1, 1, data.num_cells)
    ours = _fv_results(poly, _scalars(flux), _scalars(cells))
    shape = _fv_results(data, _scalars(flux), _scalars(cells))
    for key in shape:
        np.testing.assert_allclose(ours[key], shape[key], rtol=1e-5, atol=1e-5, err_msg=key)


def test_cell_geometry_of_known_cells(backend):
    poly = td.as_polyhedra(_two_hexes_and_a_pyramid())
    volumes, centroids = alg.cell_geometry(poly)
    np.testing.assert_allclose(volumes.values.to_numpy(), [1, 1, 1 / 3], rtol=1e-6)
    # A pyramid's centroid is a quarter of the way from base to apex.
    np.testing.assert_allclose(centroids.values.to_numpy(vectors=True),
                               [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 0.5, 1.25]],
                               atol=1e-6)
    grid = td.rectilinear_grid([0, 1, 3, 4], [0, 2, 3], [0, 1, 1.5])
    volumes, _ = alg.cell_geometry(td.as_polyhedra(grid))
    dx, dy, dz = np.diff([0, 1, 3, 4]), np.diff([0, 2, 3]), np.diff([0, 1, 1.5])
    np.testing.assert_allclose(volumes.values.to_numpy(),
                               np.einsum("k,j,i->kji", dz, dy, dx).reshape(-1), rtol=1e-6)


def _uniform_flux(data, velocity):
    normals, _ = alg.face_geometry(data)
    return _scalars(normals.values.to_numpy(vectors=True) @ np.asarray(velocity))


@pytest.mark.parametrize("name", ["mixed", "rectilinear", "tetra", "wedge", "pyramid",
                                  "hexahedron", "voxel", "voronoi"])
def test_perot_gives_back_a_uniform_field(backend, name):
    """A uniform velocity's normal components reconstruct to it exactly on cells
    with planar faces -- including thin Voronoi columns, through the stored sides."""
    if name == "voronoi":
        points, cells, _ = _voronoi_columns()
        data = td.DataSet(_from_cell_faces(cells, len(points)), points)
    else:
        meshes = _meshes_for_fv()
        if name not in meshes:
            pytest.skip("needs VTK for this mesh")
        data = meshes[name]
    velocity = (0.3, -1.2, 0.7)
    for mesh in ((data, td.as_polyhedra(data)) if name != "voronoi" else (data,)):
        flux = td.Field(td.Values(mesh, "faces"), _uniform_flux(mesh, velocity))
        got = alg.perot(mesh, flux).values.to_numpy(vectors=True)
        np.testing.assert_allclose(got, np.tile(velocity, (mesh.num_cells, 1)), atol=2e-4)
        # Each closed cell lets as much of a uniform flow out as in.
        _, areas = alg.face_geometry(mesh)
        carried = td.Field(td.Values(mesh, "faces"),
                           _scalars(flux.values.to_numpy() * areas.values.to_numpy()))
        np.testing.assert_allclose(alg.divergence(mesh, carried).values.to_numpy(), 0,
                                   atol=1e-5)


def test_voronoi_columns_measure_up(backend):
    points, cells, columns = _voronoi_columns(layers=3, thickness=0.01)
    data = td.DataSet(_from_cell_faces(cells, len(points)), points)
    volumes, centroids = alg.cell_geometry(data)
    assert (volumes.values.to_numpy() > 0).all()
    # Each column's three layers have one footprint, so one volume, stacked.
    v = volumes.values.to_numpy()
    for c in range(0, len(cells), 3):
        np.testing.assert_allclose(v[c:c + 3], v[c], rtol=1e-4)
    z = centroids.values.to_numpy(vectors=True)[:, 2]
    np.testing.assert_allclose(z, np.tile([0.005, 0.015, 0.025], columns), atol=1e-6)
    # Interior faces, by cell values: jumps are differences of the two sides' values.
    values = np.arange(data.num_cells, dtype=float)
    sides = data.topology.faces().sides.to_numpy(vectors=True)
    jumps = alg.jump(data, td.Field(td.Constant(data), _scalars(values))).values.to_numpy()
    expected = np.where(sides[:, 2] >= 0, values[sides[:, 2]] - values[sides[:, 0]], 0)
    np.testing.assert_allclose(jumps, expected)


@needs_vtk
def test_volumes_agree_with_vtk(backend):
    points, cells, _ = _voronoi_columns()
    for data in (td.as_polyhedra(_two_hexes_and_a_pyramid()),
                 td.DataSet(_from_cell_faces(cells, len(points)), points)):
        grid = dataset_to_vtk(data)
        theirs = [grid.GetCell(c).ComputeVolume() for c in range(grid.GetNumberOfCells())]
        ours = alg.cell_geometry(data)[0].values.to_numpy()
        np.testing.assert_allclose(ours, theirs, rtol=1e-4)


# ── Phase 3a: polygons, the same topology one dimension down ────────

def _plane_mesh():
    """Triangles and quads in the plane z = 0: two quads and two triangles."""
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0],
                       [1, 2, 0]], float)
    loops = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 6], [4, 5, 6]]
    types = np.array([9, 9, 5, 5], np.uint8)
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in loops])])
    return td.DataSet(td.UnstructuredTopology(types, offsets, np.concatenate(loops)), points)


def _shoelace(points, loops):
    out = []
    for loop in loops:
        p = points[loop]
        out.append(0.5 * np.linalg.norm(np.cross(p - p[0], np.roll(p, -1, axis=0) - p[0]).sum(
            axis=0)))
    return np.array(out)


@pytest.mark.parametrize("make", ["plane", "pixels"])
def test_polygons_from_2d_meshes(backend, make):
    data = (_plane_mesh() if make == "plane"
            else vtk_to_dataset(_pixel_grid()) if vtk is not None else None)
    if data is None:
        pytest.skip("needs VTK for this mesh")
    poly = td.as_polygons(data)
    t = poly.topology
    assert isinstance(t, td.PolygonalTopology) and t.dimension == 2
    # Facets are the edges, numbered as the shape path numbers them.
    np.testing.assert_array_equal(
        np.sort(t.arrays()[1].reshape(-1, 2), axis=1),
        data.topology.edges().rows.to_numpy(vectors=True))
    sides = t.faces().sides.to_numpy(vectors=True)
    assert ((sides[:, 2] >= 0).sum() + t.faces().boundary().shape[0]) == t.num_faces
    assert td.check_winding(poly).size == 0
    offsets, loops = (a.to_numpy() for a in t.loops())
    rings = [loops[offsets[c]:offsets[c + 1]] for c in range(t.num_cells)]
    areas, centroids = td.algorithms.cell_geometry(poly)
    np.testing.assert_allclose(areas.values.to_numpy(), _shoelace(poly.positions(), rings),
                               rtol=1e-6)
    np.testing.assert_allclose(centroids.values.to_numpy(vectors=True)[:, 2], 0, atol=1e-6)


def _pixel_grid():
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import vtkAppendFilter

    image = vtkImageData()
    image.SetDimensions(4, 3, 1)
    append = vtkAppendFilter()
    append.AddInputData(image)
    append.Update()
    return append.GetOutput()                         # pixels, as an unstructured grid


def test_polygon_windings_must_agree(backend):
    loops = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 6, 4]]   # the triangle walks 3->4 like its quad
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in loops])])
    with pytest.raises(ValueError, match="walked the same way"):
        td.PolygonalTopology(offsets, np.concatenate(loops))
    with pytest.raises(ValueError, match="not 2D"):
        td.as_polygons(_two_hexes_and_a_pyramid())


def _check_closed_surface(data, surface):
    t = surface.topology
    assert isinstance(t, td.PolygonalTopology)
    assert t.num_cells == data.topology.faces().boundary().shape[0]
    assert t.faces().boundary().shape[0] == 0                 # closed: every edge twice
    assert td.check_winding(surface).size == 0
    used = np.unique(t.loops()[1].to_numpy())
    assert used.size - t.num_faces + t.num_cells == 2         # a sphere's Euler number
    _, areas = td.algorithms.face_geometry(data)
    boundary = data.topology.faces().boundary().to_numpy()
    np.testing.assert_allclose(td.algorithms.cell_geometry(surface)[0].values.to_numpy(),
                               areas.values.to_numpy()[boundary], rtol=1e-5)


def test_a_polyhedral_mesh_has_a_polygonal_surface(backend):
    data = td.as_polyhedra(_two_hexes_and_a_pyramid())
    data.fields["cell"] = td.Field(td.Constant(data), _scalars([1.0, 2.0, 5.0]))
    nf = data.topology.num_faces
    data.fields["flux"] = td.Field(td.Values(data, "faces"), _scalars(np.arange(nf)))
    td.algorithms.boundary_faces(data)
    surface = td.algorithms.extract_surface(data)
    _check_closed_surface(data, surface)
    boundary = data.sets["boundary"].to_numpy()
    sides = data.topology.faces().sides.to_numpy(vectors=True)
    np.testing.assert_array_equal(surface.fields["flux"].values.to_numpy(), boundary)
    np.testing.assert_array_equal(surface.fields["cell"].values.to_numpy(),
                                  np.array([1.0, 2.0, 5.0])[sides[boundary, 0]])
    points, cells, _ = _voronoi_columns()
    columns = td.DataSet(_from_cell_faces(cells, len(points)), points)
    td.algorithms.boundary_faces(columns)
    _check_closed_surface(columns, td.algorithms.extract_surface(columns))


@needs_vtk
def test_polygons_through_vtk(backend):
    data = td.as_polyhedra(_two_hexes_and_a_pyramid())
    td.algorithms.boundary_faces(data)
    surface = td.algorithms.extract_surface(data)
    grid = dataset_to_vtk(surface)
    assert grid.GetNumberOfCells() == surface.num_cells
    assert {grid.GetCellType(c) for c in range(grid.GetNumberOfCells())} == {7}
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    sizes = vtkCellSizeFilter()
    sizes.SetInputData(grid)
    sizes.Update()
    theirs = vtk_to_numpy(sizes.GetOutput().GetCellData().GetArray("Area"))
    np.testing.assert_allclose(td.algorithms.cell_geometry(surface)[0].values.to_numpy(),
                               theirs, rtol=1e-5)


# ── Phase 3b: contour of polyhedra, López face by face ──────────────

def _cycles(points, offsets, loops, decimals=5):
    """Polygons by coordinates, each rotated to start at its smallest point: the same
    polygon, wound the same way, compares equal however it is numbered."""
    out = set()
    for c in range(len(offsets) - 1):
        ring = [tuple(np.round(points[p], decimals) + 0.0)
                for p in loops[offsets[c]:offsets[c + 1]]]
        s = ring.index(min(ring))
        out.add(tuple(ring[s:] + ring[:s]))
    return out


def _radius(data, center=(0.3, 0.2, 0.1)):
    return td.Field(td.H1(data), _scalars(np.linalg.norm(data.positions() - center, axis=1)))


def _solid_meshes():
    meshes = {"mixed": _two_hexes_and_a_pyramid()}
    if vtk is not None:
        for kind in ("tetra", "hexahedron", "wedge", "pyramid", "voxel"):
            source = vtk.vtkCellTypeSource()
            source.SetCellType(SOLID_TYPES[kind])
            source.SetBlocksDimensions(3, 3, 2)
            source.Update()
            meshes[kind] = vtk_to_dataset(source.GetOutput())
    return meshes


@pytest.mark.parametrize("name", ["mixed", "tetra", "hexahedron", "wedge", "pyramid", "voxel"])
def test_contour_of_polyhedra_is_the_shape_paths(backend, name):
    meshes = _solid_meshes()
    if name not in meshes:
        pytest.skip("needs VTK for this mesh")
    data = meshes[name]
    poly = td.as_polyhedra(data)
    for field, iso in ((_radius, 0.9), (_height, 0.7)):
        data.fields["s"] = field(data)
        poly.fields["s"] = td.Field(td.H1(poly), data.fields["s"].values)
        theirs = td.contour(data, "s", iso)
        ours = td.contour(poly, "s", iso)
        assert isinstance(ours.topology, td.PolygonalTopology)
        np.testing.assert_allclose(np.unique(ours.positions().round(6), axis=0),
                                   np.unique(theirs.positions().round(6), axis=0), atol=1e-6)
        np.testing.assert_allclose(ours.fields["s"].values.to_numpy(), iso, atol=1e-5)
        assert td.check_winding(ours).size == 0
        if field is _height:
            # A linear field's iso-polygons are planar: the same area as triangles.
            area = td.algorithms.cell_geometry(ours)[0].values.to_numpy().sum()
            points = theirs.positions()
            tri = theirs.topology.connectivity.to_numpy().reshape(-1, 3)
            theirs_area = 0.5 * np.linalg.norm(np.cross(points[tri[:, 1]] - points[tri[:, 0]],
                                                        points[tri[:, 2]] - points[tri[:, 0]]),
                                               axis=1).sum()
            np.testing.assert_allclose(area, theirs_area, rtol=1e-5)


def _vtk_lopez(data, name, iso):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkContour3DLinearGrid

    grid = dataset_to_vtk(data)
    grid.GetPointData().SetActiveScalars(name)
    contour = vtkContour3DLinearGrid()
    contour.SetInputData(grid)
    contour.SetValue(0, iso)
    contour.GenerateTrianglesOff()
    contour.Update()
    out = contour.GetOutput()
    polys = out.GetPolys()
    return (vtk_to_numpy(out.GetPoints().GetData()) if out.GetNumberOfPoints()
            else np.zeros((0, 3)),
            vtk_to_numpy(polys.GetOffsetsArray()), vtk_to_numpy(polys.GetConnectivityArray()))


def _saddle_pair():
    """Two unit cubes sharing the face x = 1, whose corners alternate high and low:
    a saddle, four crossings on one face."""
    poly = td.as_polyhedra(td.rectilinear_grid([0, 1, 2], [0, 1], [0, 1]))
    p = poly.positions()
    values = np.where(np.isclose(p[:, 0], 1), np.where((p[:, 1] + p[:, 2]) % 2 == 0, 1.0, 0.0),
                      0.3 + 0.1 * p[:, 1] + 0.05 * p[:, 2])
    poly.fields["s"] = td.Field(td.H1(poly), _scalars(values))
    return poly


@needs_vtk
@pytest.mark.parametrize("name", ["mixed", "tetra", "hexahedron", "wedge", "pyramid", "voxel",
                                  "voronoi", "saddle"])
def test_contour_of_polyhedra_is_vtks_lopez(backend, name):
    """VTK's López (vtkPolyhedronContour, through vtkContour3DLinearGrid) on the same
    polyhedra gives the same polygons, wound the same way -- saddle faces included."""
    if name == "voronoi":
        points, cells, _ = _voronoi_columns()
        poly = td.DataSet(_from_cell_faces(cells, len(points)), points)
        poly.fields["s"] = _radius(poly, (0.5, 0.5, 0.0))
        isos = (0.2, 0.35)
    elif name == "saddle":
        poly = _saddle_pair()
        isos = (0.5,)
    else:
        poly = td.as_polyhedra(_solid_meshes()[name])
        poly.fields["s"] = _radius(poly)
        isos = (0.9, 1.4)
    for iso in isos:
        ours = td.contour(poly, "s", iso)
        o, loops = (a.to_numpy() for a in ours.topology.loops())
        theirs = _vtk_lopez(poly, "s", iso)
        mine = _cycles(ours.positions(), o, loops)
        assert len(mine) == len(theirs[1]) - 1 and mine == _cycles(*theirs)


def test_saddle_faces_stay_watertight(backend):
    surface = td.contour(_saddle_pair(), "s", 0.5)
    sizes = np.diff(surface.topology.loops()[0].to_numpy())
    assert surface.num_cells >= 2 and td.check_winding(surface).size == 0
    # The saddle face's four crossings pair up alike in both cells: every edge of
    # the surface on that face is shared, wound oppositely.
    assert surface.topology.faces().boundary().shape[0] < sizes.sum()


@tack.kernel
def _caps(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = cells.MAX_SCRATCH


def test_size_buckets(backend):
    points, cells, _ = _voronoi_columns(n=60)
    data = td.DataSet(_from_cell_faces(cells, len(points)), points)
    t = data.topology
    buckets = td.SizeBuckets(t, caps=(16, 32))
    assert buckets is td.SizeBuckets(t, caps=(16, 32))
    groups = data.launch_groups("cells", [], keys=[buckets])
    assert len(groups) == 2 and sum(g.count for g in groups) == t.num_cells
    out = tack.zeros(tack.i32, (t.num_cells,))
    for group in groups:
        _caps(data.domain_view("cells", group), out)
    face_sizes = np.diff(t.arrays()[0])
    cell_offsets, cell_faces = t.arrays()[2], t.arrays()[3]
    need = [(face_sizes[cell_faces[cell_offsets[c]:cell_offsets[c + 1]]].sum() + 1) // 2
            for c in range(t.num_cells)]
    np.testing.assert_array_equal(out.to_numpy(), np.where(np.array(need) <= 16, 16, 32))
    with pytest.raises(ValueError, match="largest bucket"):
        td.SizeBuckets(t, caps=(4,)).cell_keys()


def test_slice_and_fields_on_polyhedra(backend):
    data = td.as_polyhedra(_two_hexes_and_a_pyramid())
    data.fields["cell"] = td.Field(td.Constant(data), _scalars([1.0, 2.0, 5.0]))
    data.fields["u"] = _height(data)
    origin, normal = np.array([0.7, 0.4, 0.6]), np.array([1.0, 0.3, 0.5])
    cut = td.slice_plane(data, origin, normal)
    assert isinstance(cut.topology, td.PolygonalTopology)
    np.testing.assert_allclose((cut.positions() - origin) @ normal, 0, atol=1e-5)
    p = cut.positions()
    np.testing.assert_allclose(cut.fields["u"].values.to_numpy(), p[:, 0] + 2 * p[:, 1] - p[:, 2],
                               atol=1e-5)
    # Each polygon carries the cell it lies in.
    centroids = td.algorithms.cell_geometry(cut)[1].values.to_numpy(vectors=True)
    cells = cut.fields["cell"].values.to_numpy()
    inside_first = centroids[:, 0] < 1.0
    assert set(cells[inside_first & (centroids[:, 2] < 1)]) <= {1.0}
    assert set(cells[~inside_first]) <= {2.0}


# ── Phase 4: oriented face values, threshold, readers ───────────────

def test_oriented_face_values(backend):
    data = td.as_polyhedra(_two_hexes_and_a_pyramid())
    assert td.Values(data, "faces") is td.Values(data, "faces", oriented=False)
    assert td.Values(data, "faces", oriented=True) is not td.Values(data, "faces")
    with pytest.raises(ValueError, match="only values on faces"):
        td.Values(data, "cells", oriented=True)
    flux = td.algorithms.upwind_flux(data, td.Field(td.Constant(data), _scalars([1.0, 2, 3])),
                                     (1.0, 0.0, 0.0))
    assert flux.space is td.Values(data, "faces", oriented=True)


def _threshold_meshes():
    meshes = _solid_meshes()
    for data in meshes.values():
        centers = td.algorithms.cell_geometry(td.as_polyhedra(data))[1].values.to_numpy(
            vectors=True)
        data.fields["c"] = td.Field(td.Constant(data), _scalars(centers[:, 0]))
        data.fields["p"] = td.Field(td.H1(data), _scalars(data.positions()[:, 0]
                                                          + 0.5 * data.positions()[:, 2]))
    return meshes


@pytest.mark.parametrize("name", ["mixed", "tetra", "hexahedron", "wedge", "pyramid", "voxel"])
@pytest.mark.parametrize("how", ["cells", "all points", "any point"])
def test_threshold_of_polyhedra_is_the_shape_paths(backend, name, how):
    meshes = _threshold_meshes()
    if name not in meshes:
        pytest.skip("needs VTK for this mesh")
    data = meshes[name]
    poly = td.as_polyhedra(data)
    key, lo, hi = ("c", 0.3, 1.2) if how == "cells" else ("p", 0.2, 1.4)
    theirs = td.threshold(data, key, lo, hi, all_points=how != "any point")
    ours = td.threshold(poly, key, lo, hi, all_points=how != "any point")
    assert (ours.num_cells, ours.num_points) == (theirs.num_cells, theirs.num_points)
    if not ours.num_cells:
        return
    np.testing.assert_allclose(ours.positions(), theirs.positions(), atol=1e-6)
    assert td.check_winding(ours).size == 0
    reference = td.as_polyhedra(theirs)
    np.testing.assert_allclose(td.algorithms.cell_geometry(ours)[0].values.to_numpy(),
                               td.algorithms.cell_geometry(reference)[0].values.to_numpy(),
                               rtol=1e-5)
    assert _faces_by_points(ours.topology) == _faces_by_points(reference.topology)
    np.testing.assert_allclose(ours.fields["c"].values.to_numpy(),
                               theirs.fields["c"].values.to_numpy())


def test_threshold_turns_faces_and_their_oriented_values(backend):
    """Keep the upper two layers of the Voronoi columns: the faces between layers 0
    and 1 belonged to the dropped cells (their side 0), so they turn for the cells
    kept. A uniform flow's oriented flux is negated with them, and Perot still gives
    the flow back; the same numbers as plain face values are not, and it does not."""
    points, cells, columns = _voronoi_columns(layers=3)
    data = td.DataSet(_from_cell_faces(cells, len(points)), points)
    velocity = (0.4, -0.3, 0.9)
    values = _uniform_flux(data, velocity)
    data.fields["flux"] = td.Field(td.Values(data, "faces", oriented=True), values)
    data.fields["plain"] = td.Field(td.Values(data, "faces"), values)
    layer = td.algorithms.cell_geometry(data)[1].values.to_numpy(vectors=True)[:, 2]
    data.fields["layer"] = td.Field(td.Constant(data), _scalars(layer))
    upper = td.threshold(data, "layer", 0.01, 1.0)
    assert upper.num_cells == 2 * columns
    assert td.check_winding(upper).size == 0
    np.testing.assert_allclose(
        td.algorithms.perot(upper, upper.fields["flux"]).values.to_numpy(vectors=True),
        np.tile(velocity, (upper.num_cells, 1)), atol=2e-4)
    plain = td.algorithms.perot(upper, upper.fields["plain"]).values.to_numpy(vectors=True)
    assert not np.allclose(plain[:columns], velocity, atol=1e-2)      # the bottom kept layer


def test_threshold_of_polygons(backend):
    data = td.as_polygons(_plane_mesh())
    data.fields["id"] = td.Field(td.Constant(data), _scalars([0.0, 1.0, 2.0, 3.0]))
    kept = td.threshold(data, "id", 0.5, 2.5)
    assert isinstance(kept.topology, td.PolygonalTopology)
    assert kept.num_cells == 2 and td.check_winding(kept).size == 0
    np.testing.assert_allclose(kept.fields["id"].values.to_numpy(), [1, 2])
    np.testing.assert_allclose(td.algorithms.cell_geometry(kept)[0].values.to_numpy(),
                               [1.0, 0.5], rtol=1e-6)


@needs_vtk
def test_threshold_of_polyhedra_is_vtks(backend):
    from vtkmodules.vtkFiltersCore import vtkThreshold

    poly = td.as_polyhedra(_threshold_meshes()["wedge"])
    grid = dataset_to_vtk(poly)
    threshold = vtkThreshold()
    threshold.SetInputData(grid)
    threshold.SetInputArrayToProcess(0, 0, 0, 1, "c")
    threshold.SetLowerThreshold(0.3)
    threshold.SetUpperThreshold(1.2)
    threshold.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
    threshold.Update()
    theirs = threshold.GetOutput()
    ours = td.threshold(poly, "c", 0.3, 1.2)
    assert (ours.num_cells, ours.num_points) == (theirs.GetNumberOfCells(),
                                                 theirs.GetNumberOfPoints())
    np.testing.assert_allclose(sorted(td.algorithms.cell_geometry(ours)[0].values.to_numpy()),
                               sorted(theirs.GetCell(c).ComputeVolume()
                                      for c in range(theirs.GetNumberOfCells())), rtol=1e-5)


def test_orient_repairs_reversed_faces_and_cells(backend):
    points, cells, _ = _voronoi_columns()
    reference = _from_cell_faces(cells, len(points))
    rng = np.random.default_rng(8)
    scrambled = []
    for faces in cells:
        faces = [ring[::-1] if rng.uniform() < 0.3 else ring for ring in faces]
        scrambled.append([r[::-1] for r in faces] if rng.uniform() < 0.5 else faces)
    with pytest.raises(ValueError, match="orient=True"):
        td.PolyhedralTopology.from_cell_faces(scrambled, len(points))
    mended = td.PolyhedralTopology.from_cell_faces(scrambled, len(points), positions=points,
                                                   orient=True)
    assert td.check_winding(mended).size == 0
    data = td.DataSet(mended, points)
    assert (td.algorithms.cell_geometry(data)[0].values.to_numpy() > 0).all()
    ours, theirs = _faces_by_points(mended), _faces_by_points(reference)
    assert ours.keys() == theirs.keys()
    assert all(set(ours[k]) == set(theirs[k]) for k in ours)


def _cgns(name):
    path = os.path.join(_VTK_DATA, name)
    # Not every VTK build has the CGNS reader; the data is always here.
    vtkCGNSReader = pytest.importorskip("vtkmodules.vtkIOCGNSReader").vtkCGNSReader

    reader = vtkCGNSReader()
    reader.SetFileName(path)
    reader.Update()
    blocks = reader.GetOutput().NewIterator()
    blocks.InitTraversal()
    return blocks.GetCurrentDataObject()


@needs_vtk
def test_cgns_in_both_polyhedral_conventions(backend):
    """NFACE_n (a sign per cell -> face reference) and NGON_n with ParentElements
    (owner and neighbour): the same mesh, consistently wound either way."""
    nface = vtk_to_dataset(_cgns("Example_nface_n.cgns"))
    ngon = vtk_to_dataset(_cgns("Example_ngon_pe.cgns"))
    for data in (nface, ngon):
        assert isinstance(data.topology, td.PolyhedralTopology)
        assert td.check_winding(data).size == 0
        assert (td.algorithms.cell_geometry(data)[0].values.to_numpy() > 0).all()

    def by_position(data):
        p = np.round(data.positions(), 6)
        return {frozenset(map(tuple, p[list(k)])) for k in _faces_by_points(data.topology)}

    assert by_position(nface) == by_position(ngon)


@needs_vtk
def test_a_real_cfd_mesh_needs_orienting(backend):
    """EngineSector.cgns, as VTK's reader gives it, is wound inward almost everywhere
    and inconsistently in 65 cells: refused, then repaired by orient."""
    grid = _cgns("EngineSector.cgns")
    with pytest.raises(ValueError, match="orient=True"):
        vtk_to_dataset(grid)
    data = vtk_to_dataset(grid, orient=True)
    assert data.num_cells == 1956 and td.check_winding(data).size == 0
    assert (td.algorithms.cell_geometry(data)[0].values.to_numpy() > 0).all()
    surface = vtk.vtkGeometryFilter()
    surface.SetInputData(dataset_to_vtk(data))
    surface.Update()
    assert data.topology.faces().boundary().shape[0] == surface.GetOutput().GetNumberOfCells()


@pytest.mark.parametrize("make", [td.as_polyhedra, "polygons"])
def test_reference_element_algorithms_are_refused(backend, make):
    """Algorithms that evaluate a basis refuse polyhedra and polygons outright. traces
    used to launch over no cells there (their shape has no faces of its own) and
    return zeros."""
    data = td.as_polygons(_plane_mesh()) if make == "polygons" else make(
        _two_hexes_and_a_pyramid())
    point_data = td.Field(td.H1(data), _scalars(data.positions()[:, 0]))
    cell_data = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells, dtype=float)))
    for call in (lambda: td.algorithms.cell_centers(data),
                 lambda: td.algorithms.values_at_centers(data, point_data),
                 lambda: td.algorithms.gradients(data, point_data),
                 lambda: td.traces(data, point_data),
                 lambda: td.algorithms.jump(data, point_data)):
        with pytest.raises(NotImplementedError, match="needs a reference element"):
            call()
    td.algorithms.jump(data, cell_data)                 # cell data needs none


# ── Point and cell averages ─────────────────────────────────────────

@pytest.mark.parametrize("name", ["mixed", "tetra", "hexahedron", "wedge", "pyramid", "voxel",
                                  "polygons"])
def test_averages_are_the_shape_paths(backend, name):
    if name == "polygons":
        data = _plane_mesh()
        poly = td.as_polygons(data)
    else:
        meshes = _solid_meshes()
        if name not in meshes:
            pytest.skip("needs VTK for this mesh")
        data = meshes[name]
        poly = td.as_polyhedra(data)
    x = data.positions()
    point_values = _scalars(x[:, 0] + 2.0 * x[:, 1] - x[:, 2] ** 2)
    cell_values = _scalars(np.sin(np.arange(data.num_cells, dtype=float)))
    for d in (data, poly):
        d.fields["p"] = td.Field(td.H1(d), point_values)
        d.fields["c"] = td.Field(td.Constant(d), cell_values)
    np.testing.assert_allclose(td.algorithms.to_cells(poly, poly.fields["p"]).values.to_numpy(),
                               td.algorithms.to_cells(data, data.fields["p"]).values.to_numpy(),
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(td.algorithms.to_points(poly, poly.fields["c"]).values.to_numpy(),
                               td.algorithms.to_points(data, data.fields["c"]).values.to_numpy(),
                               rtol=1e-6, atol=1e-6)
    same = td.algorithms.to_cells(poly, poly.fields["c"])
    np.testing.assert_array_equal(same.values.to_numpy(), cell_values.to_numpy())


@needs_vtk
def test_averages_on_voronoi_columns_are_vtks(backend):
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkPointDataToCellData

    points, cells, _ = _voronoi_columns()
    data = td.DataSet(_from_cell_faces(cells, len(points)), points)
    p = np.cos(3 * points[:, 0]) + points[:, 1] * points[:, 2]
    c = np.sin(np.arange(data.num_cells, dtype=float))
    grid = dataset_to_vtk(data)
    array = numpy_to_vtk(p.astype(np.float32), deep=True)
    array.SetName("p")
    grid.GetPointData().AddArray(array)
    array = numpy_to_vtk(c.astype(np.float32), deep=True)
    array.SetName("c")
    grid.GetCellData().AddArray(array)
    to_cells = vtkPointDataToCellData()
    to_cells.SetInputData(grid)
    to_cells.Update()
    to_points = vtkCellDataToPointData()
    to_points.SetInputData(grid)
    to_points.Update()
    ours = td.algorithms.to_cells(data, td.Field(td.H1(data), _scalars(p)))
    np.testing.assert_allclose(ours.values.to_numpy(),
                               vtk_to_numpy(to_cells.GetOutput().GetCellData().GetArray("p")),
                               rtol=1e-5, atol=1e-6)
    ours = td.algorithms.to_points(data, td.Field(td.Values(data, "cells"), _scalars(c)))
    np.testing.assert_allclose(ours.values.to_numpy(),
                               vtk_to_numpy(to_points.GetOutput().GetPointData().GetArray("c")),
                               rtol=1e-5, atol=1e-6)
