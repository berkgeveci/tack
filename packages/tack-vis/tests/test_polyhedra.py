"""Tests for tack.data.polyhedra: the polyhedral topology, phase 1.

Shape-based meshes converted to polyhedra must give back exactly what the
shape path derives -- faces, sides, boundary, edges and their numbering --
and wind consistently. VTK checks the export (each cell's own outward copy
of its faces: positive volumes, the same external faces) and the import
(duplicated faces matched). MPAS-like columns -- Voronoi polygons extruded
into thin layers -- give polygonal faces of any size. Polyhedra from VTK's
data, where present, are read and checked.
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
    thin layers: prisms with polygonal tops and bottoms, quads around."""
    from scipy.spatial import Voronoi

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
    """A polyhedral topology from each cell's outward faces: copies matched by point set,
    the first copy's cell side 0, the other side 1."""
    known, face_points, face_offsets = {}, [], [0]
    cell_offsets, cell_faces, sides = [0], [], []
    for faces in cells:
        for ring in faces:
            key = tuple(sorted(ring))
            if key not in known:
                known[key] = len(face_offsets) - 1
                face_points.extend(ring)
                face_offsets.append(len(face_points))
                sides.append(0)
            else:
                sides.append(1)
            cell_faces.append(known[key])
        cell_offsets.append(len(cell_faces))
    return td.PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces,
                                 np.array(sides, np.uint8), num_points=num_points)


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


_VTK_DATA = os.path.expanduser("~/Data/VTK/Data")


@needs_vtk
@pytest.mark.parametrize("name", ["onePolyhedron.vtu", "polyhedron2pieces.vtu",
                                  "polyhedron_mesh.vtu", "concavePolyhedron.vtu",
                                  "sliceOfPolyhedron.vtu", "vtkHDF/polyhedron.vtu",
                                  "nonWatertightPolyhedron.vtu"])
def test_polyhedra_from_vtk_data(backend, name):
    path = os.path.join(_VTK_DATA, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} is not here")
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
