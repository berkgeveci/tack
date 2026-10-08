"""Tests for the dataset API prototype (``docs/design/dataset-api.md``).

Derived faces and edges are checked against a hand-built mixed mesh and,
where VTK is installed, against VTK's own cells: every face once, with the
cells on its sides, in side 0's outward order, and every edge once, with
the direction each cell's local edge runs. Fields on spaces are checked by
what must hold whatever the layout -- a continuous field converted to the
DG layout and back is unchanged -- and against vtkCellCenters,
vtkCellDataToPointData and vtkGeometryFilter.
"""

import types

import numpy as np
import pytest

import tack
import tack.data as td
from tack.data import algorithms as alg
from tack.runtime.dispatch import env_flag

try:
    from vtkmodules import vtkCommonCore, vtkCommonDataModel, vtkFiltersCore
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkExtractEdges
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(**{name: getattr(module, name)
                                   for module in (vtkCommonCore, vtkCommonDataModel,
                                                  vtkFiltersCore)
                                   for name in dir(module) if name.startswith("vtk")})

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

SOLID_TYPES = {"tetra": 10, "voxel": 11, "hexahedron": 12, "wedge": 13, "pyramid": 14}


def _scalars(values, dtype=tack.f32):
    values = np.asarray(values)
    f = tack.field(dtype, shape=values.shape)
    f.from_numpy(values.astype(dtype.numpy_dtype))
    return f


def _vectors(values, dtype=tack.f32):
    values = np.asarray(values)
    f = tack.Vector.field(values.shape[1], dtype, shape=(values.shape[0],))
    f.from_numpy(values.astype(dtype.numpy_dtype))
    return f


def _two_hexes_and_a_pyramid():
    """Two unit hexahedra side by side, and a pyramid on the first one's top face."""
    def pid(x, y, z):
        return x + 3 * y + 6 * z

    pts = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1, 2)], float)
    pts = np.vstack([pts, [[0.5, 0.5, 2.0]]])
    hexes = [[pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 1, 0), pid(i, 1, 0),
              pid(i, 0, 1), pid(i + 1, 0, 1), pid(i + 1, 1, 1), pid(i, 1, 1)] for i in (0, 1)]
    pyramid = [pid(0, 0, 1), pid(1, 0, 1), pid(1, 1, 1), pid(0, 1, 1), 12]
    topology = td.UnstructuredTopology(np.array([12, 12, 14], np.uint8), [0, 8, 16, 21],
                                       hexes[0] + hexes[1] + pyramid)
    return td.DataSet(topology, pts)


# ── Faces and edges, by hand ────────────────────────────────────────

def test_faces_of_a_mixed_mesh(backend):
    data = _two_hexes_and_a_pyramid()
    faces = data.topology.faces()
    assert faces.num_faces == 6 + 5 + 4             # 16 sides, one shared face of each pair
    assert data.topology.edges().num_edges == 12 + 8 + 4
    sides = faces.sides.to_numpy(vectors=True)
    shared = {tuple(s) for s in sides if s[2] >= 0}
    assert shared == {(0, 1, 1, 0), (0, 5, 2, 0)}   # hex 0's +x and top faces
    assert alg.boundary_faces(data).shape[0] == 13

    jumps = alg.jump(data, td.Field(td.Constant(data), _scalars([1.0, 2.0, 5.0])))
    by_sides = dict(zip(map(tuple, sides), jumps.values.to_numpy()))
    assert by_sides[(0, 1, 1, 0)] == 1.0 and by_sides[(0, 5, 2, 0)] == 4.0
    assert np.count_nonzero(jumps.values.to_numpy()) == 2

    # A face counts plus for its side-0 cell and minus for its side-1 cell.
    _, areas = alg.face_geometry(data)
    slant = np.sqrt(0.5 ** 2 + 1.0 ** 2) / 2         # each pyramid side triangle
    np.testing.assert_allclose(alg.divergence(data, areas).values.to_numpy(),
                               [6, 5 - 1, 4 * slant - 1], rtol=1e-6)


def _closed_cells_check(data):
    """Every cell's outward area vectors sum to zero: the faces' normals point out of
    side 0, and each cell knows which side it is."""
    normals, areas = alg.face_geometry(data)
    area_vectors = normals.values.to_numpy(vectors=True) * areas.values.to_numpy()[:, None]
    flux = td.Field(td.Values(data, "faces"), _vectors(area_vectors))
    sums = alg.divergence(data, flux).values.to_numpy(vectors=True)
    np.testing.assert_allclose(sums, 0, atol=1e-5)
    return areas.values.to_numpy()


def test_faces_point_out_of_side_zero(backend):
    _closed_cells_check(_two_hexes_and_a_pyramid())


def test_rectilinear_faces_and_edges(backend):
    x, y, z = [0, 1, 3, 4], [0, 2, 2.5], [0, 1]
    data = td.rectilinear_grid(x, y, z)
    nx, ny, nz = 4, 3, 2
    assert data.topology.faces().num_faces == (
        nx * (ny - 1) * (nz - 1) + (nx - 1) * ny * (nz - 1) + (nx - 1) * (ny - 1) * nz)
    assert data.topology.edges().num_edges == (
        (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1))
    areas = _closed_cells_check(data)
    # Each cell's faces' areas: two of each pair of opposite faces.
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    expected = 2 * sum(a * b for a, b in ((dx.sum(), dy.sum()), (dy.sum(), dz.sum()),
                                          (dx.sum(), dz.sum())))
    boundary = alg.boundary_faces(data).to_numpy()
    np.testing.assert_allclose(areas[boundary].sum(), expected, rtol=1e-6)
    lengths = alg.edge_lengths(data).values.to_numpy()
    total = (dx.sum() * ny * nz + dy.sum() * nx * nz + dz.sum() * nx * ny)
    np.testing.assert_allclose(lengths.sum(), total, rtol=1e-6)


@tack.kernel
def _edge_incidence(cells, ids, signs):
    for c in cells:
        for e in range(cells.NUM_EDGES):
            ids[cells.entity_id(c) * cells.NUM_EDGES + e] = cells.edge_id(c, e)
            signs[cells.entity_id(c) * cells.NUM_EDGES + e] = cells.edge_sign(c, e)


def test_cells_know_their_edges(backend):
    data = td.rectilinear_grid([0, 1, 2], [0, 1, 2, 3], [0, 1])
    edges = data.topology.edges()
    n = data.num_cells * 12
    ids, signs = tack.field(tack.i32, shape=(n,)), tack.field(tack.i32, shape=(n,))
    td.for_each(_edge_incidence, data, "cells", ids, signs)
    rows = edges.rows.to_numpy(vectors=True)
    assert (rows[:, 0] < rows[:, 1]).all()
    # VTK's hexahedron edges, and the cells' point ids in x-fastest order.
    hex_edges = [(0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7),
                 (0, 4), (1, 5), (3, 7), (2, 6)]
    nx, ny = 3, 4
    ids, signs = ids.to_numpy().reshape(-1, 12), signs.to_numpy().reshape(-1, 12)
    for c in range(data.num_cells):
        i, j, k = c % 2, (c // 2) % 3, c // 6
        b = i + nx * (j + ny * k)
        corners = [b, b + 1, b + 1 + nx, b + nx]
        corners += [p + nx * ny for p in corners]
        for e, (a, d) in enumerate(hex_edges):
            pa, pd = corners[a], corners[d]
            assert tuple(rows[ids[c, e]]) == (min(pa, pd), max(pa, pd))
            assert signs[c, e] == (1 if pa < pd else -1)


# ── Faces and edges, against VTK ────────────────────────────────────

def _vtk_cells(kind, blocks=(3, 2, 2)):
    source = vtkCellTypeSource()
    source.SetCellType(SOLID_TYPES[kind])
    source.SetBlocksDimensions(*blocks)
    source.Update()
    return source.GetOutput()


def _vtk_faces(grid):
    """Every face of every cell: {sorted ids: [(cell, local face, outward ids)]}."""
    faces = {}
    for c in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(c)
        for f in range(cell.GetNumberOfFaces()):
            face = cell.GetFace(f)
            ids = [face.GetPointId(k) for k in range(face.GetNumberOfPoints())]
            if face.GetCellType() == 8:                  # a pixel, as a quad
                ids = [ids[0], ids[1], ids[3], ids[2]]
            faces.setdefault(tuple(sorted(ids)), []).append((c, f, ids))
    return faces


def _same_cycle(a, b):
    return any(list(a) == list(b[k:]) + list(b[:k]) for k in range(len(b)))


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_faces_match_vtk(backend, kind):
    grid = _vtk_cells(kind)
    data = vtk_to_dataset(grid)
    faces = data.topology.faces()
    expected = _vtk_faces(grid)
    assert faces.num_faces == len(expected)
    rows = faces.rows.to_numpy(vectors=True)
    sides = faces.sides.to_numpy(vectors=True)
    kinds = faces.kinds.to_numpy()
    for row, side, k in zip(rows, sides, kinds):
        ids = row[:3] if k == 5 else row
        owners = expected[tuple(sorted(ids))]
        assert len(owners) == (2 if side[2] >= 0 else 1)
        assert {(c, f) for c, f, _ in owners} == {(side[0], side[1]), (side[2], side[3])} - {(-1, -1)}
        outward = next(o for c, f, o in owners if (c, f) == (side[0], side[1]))
        assert _same_cycle(ids, outward)
    surface = vtkGeometryFilter()
    surface.SetInputData(grid)
    surface.Update()
    assert alg.boundary_faces(data).shape[0] == surface.GetOutput().GetNumberOfCells()
    _closed_cells_check(data)


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_edges_match_vtk(backend, kind):
    grid = _vtk_cells(kind)
    data = vtk_to_dataset(grid)
    edges = vtkExtractEdges()
    edges.SetInputData(grid)
    edges.Update()
    lines = edges.GetOutput()
    expected = {tuple(sorted(lines.GetCell(i).GetPointIds().GetId(k) for k in range(2)))
                for i in range(lines.GetNumberOfCells())}
    rows = data.topology.edges().rows.to_numpy(vectors=True)
    assert {tuple(r) for r in rows} == expected
    assert len(rows) == len(expected)


# ── Fields on spaces ────────────────────────────────────────────────

@needs_vtk
@pytest.mark.parametrize("kind", ["tetra", "hexahedron", "wedge", "pyramid"])
def test_cell_centers_match_vtk(backend, kind):
    grid = _vtk_cells(kind, (2, 2, 1))
    centers = vtk.vtkCellCenters()
    centers.SetInputData(grid)
    centers.Update()
    expected = vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
    got = alg.cell_centers(vtk_to_dataset(grid)).values.to_numpy(vectors=True)
    np.testing.assert_allclose(got, expected, atol=1e-5 if kind == "wedge" else 1e-6)


@needs_vtk
def test_cell_to_point_matches_vtk(backend):
    grid = _vtk_cells("tetra", (2, 2, 2))
    values = np.random.default_rng(3).random(grid.GetNumberOfCells())
    array = numpy_to_vtk(values, deep=1)
    array.SetName("v")
    grid.GetCellData().AddArray(array)
    to_points = vtk.vtkCellDataToPointData()
    to_points.SetInputData(grid)
    to_points.Update()
    expected = vtk_to_numpy(to_points.GetOutput().GetPointData().GetArray("v"))
    data = vtk_to_dataset(grid)
    assert data.fields["v"].space is td.Constant(data)
    got = alg.to_points(data, data.fields["v"])
    assert got.space is td.H1(data)
    np.testing.assert_allclose(got.values.to_numpy(), expected, rtol=1e-5)


def _height(data):
    """An H1 field, linear in position: every linear cell interpolates it exactly."""
    x = data.positions()
    return td.Field(td.H1(data), _scalars(x[:, 0] + 2 * x[:, 1] - x[:, 2]))


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_continuous_field_in_dg_layout(backend, make):
    data = (_two_hexes_and_a_pyramid() if make == "mixed"
            else td.rectilinear_grid([0, 1, 3], [0, 2, 3], [0, 1]))
    u = _height(data)
    dg = alg.discontinuous(data, u)
    assert dg.space is td.L2(data)
    corners = sum(g.count * g.shape.NUM_POINTS for g in data.topology.groups())
    assert dg.values.shape == (corners,)
    # The same function: the same values at the centers, and back on the points.
    np.testing.assert_allclose(alg.values_at_centers(data, dg).values.to_numpy(),
                               alg.values_at_centers(data, u).values.to_numpy(), atol=1e-5)
    np.testing.assert_allclose(alg.to_points(data, dg).values.to_numpy(),
                               u.values.to_numpy(), atol=1e-5)
    centers = alg.cell_centers(data).values.to_numpy(vectors=True)
    np.testing.assert_allclose(alg.values_at_centers(data, u).values.to_numpy(),
                               centers @ [1, 2, -1], atol=1e-5)


def test_dg_values_can_disagree(backend):
    """Each cell owns its corner values: add the cell id to cell c's, and every point
    averages its cells' ids on top of the continuous value."""
    data = _two_hexes_and_a_pyramid()
    u = _height(data)
    dg = alg.discontinuous(data, u)
    offsets = td.L2(data).offsets.to_numpy()
    values = dg.values.to_numpy()
    for c in range(data.num_cells):
        values[offsets[c]:offsets[c + 1]] += 10 * c
    dg = td.Field(td.L2(data), _scalars(values))
    connectivity = data.topology.connectivity.to_numpy()
    expected = u.values.to_numpy().copy()
    for p in range(data.num_points):
        cells = [c for c in range(data.num_cells)
                 if p in connectivity[offsets[c]:offsets[c + 1]]]
        expected[p] += 10 * np.mean(cells)
    np.testing.assert_allclose(alg.to_points(data, dg).values.to_numpy(), expected,
                               rtol=1e-6)
    # At a center each cell sees only its own values.
    np.testing.assert_allclose(alg.values_at_centers(data, dg).values.to_numpy(),
                               alg.values_at_centers(data, u).values.to_numpy()
                               + 10 * np.arange(3), rtol=1e-6)


@needs_vtk
def test_surface_of_a_side_set(backend):
    grid = _vtk_cells("wedge")
    data = vtk_to_dataset(grid)
    _, areas = alg.face_geometry(data)
    data.fields["area"] = areas
    alg.boundary_faces(data)
    surface = alg.extract_surface(data)
    assert surface.fields["area"].space is td.Values(surface, "cells")
    reference = vtkGeometryFilter()
    reference.SetInputData(grid)
    reference.Update()
    assert surface.num_cells == reference.GetOutput().GetNumberOfCells()
    # The surface's own faces, as cells of a 2D mesh, cover the same area.
    total = areas.values.to_numpy()[data.sets["boundary"].to_numpy()].sum()
    np.testing.assert_allclose(surface.fields["area"].values.to_numpy().sum(), total)
    out = dataset_to_vtk(surface)
    assert out.GetCellData().GetArray("area").GetNumberOfTuples() == surface.num_cells


@tack.kernel
def _set_areas(faces, out):
    for f in faces:
        out[faces.entity_id(f)] = faces.num_sides(f)


def test_for_each_over_a_side_set(backend):
    data = td.rectilinear_grid([0, 1, 2, 3], [0, 1, 2], [0, 1])
    boundary = alg.boundary_faces(data)
    out = tack.zeros(tack.i32, (data.topology.faces().num_faces,))
    td.for_each(_set_areas, data, "boundary", out)
    out = out.to_numpy()
    np.testing.assert_array_equal(out[boundary.to_numpy()], 1)
    assert np.count_nonzero(out) == boundary.shape[0]


def test_a_field_must_live_where_the_loop_is(backend):
    data = td.rectilinear_grid([0, 1, 2], [0, 1], [0, 1])
    on_faces = td.Field(td.Values(data, "faces"), _scalars(np.zeros(data.topology.faces().num_faces)))
    with pytest.raises(TypeError, match="cannot be viewed while iterating cells"):
        alg.values_at_centers(data, on_faces)


@needs_vtk
def test_rectilinear_round_trip(backend):
    data = td.rectilinear_grid([0, 1, 3], [0, 2], [0, 1, 1.5])
    data.fields["h"] = _height(data)
    data.fields["c"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    grid = dataset_to_vtk(data)
    assert grid.IsA("vtkRectilinearGrid")
    assert grid.GetPointData().GetArray("shape") is None      # the geometry is the grid's
    back = vtk_to_dataset(grid)
    assert back.fields["h"].space is td.H1(back) and back.fields["c"].space is td.Constant(back)
    np.testing.assert_allclose(back.positions(), data.positions())
    np.testing.assert_allclose(back.fields["h"].values.to_numpy(),
                               data.fields["h"].values.to_numpy())


# ── Geometry is a field ─────────────────────────────────────────────

def test_geometry_is_the_shape_field(backend):
    data = _two_hexes_and_a_pyramid()
    assert data.geometry is data.fields["shape"]
    assert data.geometry.space is td.H1(data)
    grid = td.rectilinear_grid([0, 1, 2], [0, 1], [0, 1])
    assert grid.geometry.space is td.H1(grid)
    assert isinstance(grid.geometry.values, td.CartesianProduct)
    with pytest.raises(ValueError, match="is the geometry"):
        td.DataSet(data.topology, data.positions(), fields={"shape": data.geometry})
    with pytest.raises(TypeError, match="H1 or L2"):
        td.DataSet(data.topology, td.Field(td.Values(data, "points"), data.geometry.values))


@tack.kernel
def _jacobian_determinants(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = cells.geometry_jacobian(c, cells.parametric_center()).determinant()


def test_geometry_jacobian(backend):
    x, y, z = [0, 1, 3], [0, 0.5, 2], [0, 4]
    data = td.rectilinear_grid(x, y, z)
    out = tack.field(tack.f32, shape=(data.num_cells,))
    td.for_each(_jacobian_determinants, data, "cells", out)
    volumes = np.multiply.outer(np.multiply.outer(np.diff(z), np.diff(y)), np.diff(x))
    np.testing.assert_allclose(out.to_numpy(), volumes.reshape(-1), rtol=1e-6)


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_gradients_of_a_linear_field(backend, make):
    data = (_two_hexes_and_a_pyramid() if make == "mixed"
            else td.rectilinear_grid([0, 1, 3], [0, 2, 3], [0, 1, 1.5]))
    u = _height(data)
    for field in (u, alg.discontinuous(data, u)):
        got = alg.gradients(data, field).values.to_numpy(vectors=True)
        np.testing.assert_allclose(got, np.tile([1, 2, -1], (data.num_cells, 1)), atol=1e-4)
    constant = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    np.testing.assert_array_equal(alg.gradients(data, constant).values.to_numpy(), 0)


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_gradients_on_every_solid(backend, kind):
    data = vtk_to_dataset(_vtk_cells(kind, (2, 2, 2)))
    got = alg.gradients(data, _height(data)).values.to_numpy(vectors=True)
    np.testing.assert_allclose(got, np.tile([1, 2, -1], (data.num_cells, 1)), atol=1e-4)


def _shrunk(data, s):
    """``data`` with an L2 geometry: each cell's corners pulled toward its center by
    ``s``. The topology, and so the faces, are unchanged."""
    offsets = td.L2(data).offsets.to_numpy()
    connectivity = data.topology.connectivity.to_numpy()
    centers = alg.cell_centers(data).values.to_numpy(vectors=True)
    positions = data.positions()
    corners = np.empty((offsets[-1], 3))
    for c in range(data.num_cells):
        rows = slice(offsets[c], offsets[c + 1])
        corners[rows] = centers[c] + s * (positions[connectivity[rows]] - centers[c])
    geometry = td.Field(td.L2(data), _vectors(corners))
    return td.DataSet(data.topology, geometry, fields={"u": _height(data)})


def test_discontinuous_geometry(backend):
    data = _two_hexes_and_a_pyramid()
    shrunk = _shrunk(data, 0.5)
    assert shrunk.geometry.space is td.L2(shrunk)
    # A cell's center is where its corners were pulled toward: unchanged.
    np.testing.assert_allclose(alg.cell_centers(shrunk).values.to_numpy(vectors=True),
                               alg.cell_centers(data).values.to_numpy(vectors=True), atol=1e-6)
    # The same point values over cells half the size: twice the gradient.
    got = alg.gradients(shrunk, shrunk.fields["u"]).values.to_numpy(vectors=True)
    np.testing.assert_allclose(got, np.tile([2, 4, -2], (3, 1)), atol=1e-4)
    # Faces come from the topology, which still has the cells share points; a face
    # is where its side 0 puts it: half the size each way, facing the same way.
    assert shrunk.topology.faces().num_faces == 15
    normals, areas = alg.face_geometry(shrunk)
    original_normals, original_areas = alg.face_geometry(data)
    np.testing.assert_allclose(areas.values.to_numpy(), 0.25 * original_areas.values.to_numpy(),
                               rtol=1e-5)
    np.testing.assert_allclose(normals.values.to_numpy(vectors=True),
                               original_normals.values.to_numpy(vectors=True), atol=1e-5)
    # Edges have no sides to choose a cell from.
    with pytest.raises(NotImplementedError, match="edges"):
        alg.edge_lengths(shrunk)
    with pytest.raises(ValueError, match="per cell corner"):
        shrunk.positions()


# ── Implicit arrays ─────────────────────────────────────────────────

def test_rectilinear_geometry_is_an_ordinary_field(backend):
    """The geometry's values are a CartesianProduct, an implicit array; any algorithm
    takes it as a field like any other."""
    data = td.rectilinear_grid([0, 1, 3], [0, 2, 3], [0, 1, 1.5])
    np.testing.assert_allclose(
        alg.values_at_centers(data, data.geometry).values.to_numpy(vectors=True),
        alg.cell_centers(data).values.to_numpy(vectors=True), rtol=1e-6)
    np.testing.assert_allclose(data.positions()[:4], [[0, 0, 0], [1, 0, 0], [3, 0, 0],
                                                      [0, 2, 0]])


def test_structured_cells_read_cartesian_values_by_ijk(backend):
    from tack.data import views

    data = td.rectilinear_grid([0, 1, 3], [0, 2, 3], [0, 1])
    cells = data.topology.groups()[0]
    faces = data.topology.faces().groups()[1]
    assert views._StructuredPointAddress in type(data.geometry.view(cells)).__mro__
    assert views._PointAddress in type(data.geometry.view(faces)).__mro__
    explicit = td.Field(td.H1(data), _vectors(data.positions()))
    assert views._PointAddress in type(explicit.view(cells)).__mro__
    if backend == "cpu":
        # No signed division: the structured path never splits a flat point id.
        import re

        out = tack.Vector.field(3, tack.f32, shape=(data.num_cells,))
        src = tack.inspect(alg._at_centers, data.domain_view("cells", cells),
                           data.geometry.view(cells), out, mode="source")
        assert not re.search(r"\bs(?:div|rem)\b", src)


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_implicit_values_match_explicit(backend, make):
    data = (_two_hexes_and_a_pyramid() if make == "mixed"
            else td.rectilinear_grid([0, 1, 3], [0, 2, 3], [0, 1, 1.5]))
    n, m = data.num_points, data.num_cells
    pairs = [
        (td.Field(td.H1(data), td.ConstantArray(3.0, n)), td.Field(td.H1(data), _scalars(np.full(n, 3.0)))),
        (td.Field(td.H1(data), td.CountingArray(n, 0.0, 0.5)),
         td.Field(td.H1(data), _scalars(0.5 * np.arange(n)))),
    ]
    for implicit, explicit in pairs:
        for algorithm in (alg.values_at_centers, alg.gradients):
            np.testing.assert_allclose(algorithm(data, implicit).values.to_numpy(),
                                       algorithm(data, explicit).values.to_numpy(), atol=1e-5)
    ids = td.Field(td.Constant(data), td.CountingArray(m, 0.0, 1.0))
    np.testing.assert_allclose(alg.jump(data, ids).values.to_numpy(),
                               alg.jump(data, td.Field(td.Constant(data), _scalars(np.arange(m))))
                               .values.to_numpy())
    np.testing.assert_allclose(alg.to_points(data, ids).values.to_numpy(),
                               alg.to_points(data, td.Field(td.Constant(data), _scalars(np.arange(m))))
                               .values.to_numpy(), rtol=1e-6)


def test_field_values_must_be_an_array(backend):
    data = td.rectilinear_grid([0, 1], [0, 1])
    with pytest.raises(TypeError, match="implicit array"):
        td.Field(td.H1(data), np.zeros(4))


# ── Spaces ──────────────────────────────────────────────────────────

def test_spaces_are_one_object_per_topology_and_parameters(backend):
    data = _two_hexes_and_a_pyramid()
    other = _two_hexes_and_a_pyramid()
    assert td.H1(data) is td.H1(data.topology) is td.H1(data, order=1)
    assert td.H1(data) is not td.H1(other)
    assert td.Values(data, "faces") is not td.Values(data, "edges")
    assert td.L2(data) is not td.Constant(data)
    assert td.H1(data, order=2) is not td.H1(data)
    with pytest.raises(NotImplementedError, match="order 3"):
        td.H1(data, order=3)
    with pytest.raises(ValueError, match="points, edges, faces or cells"):
        td.Values(data, "corners")


def test_spaces_own_their_layout(backend):
    data = _two_hexes_and_a_pyramid()
    # L2 on an unstructured topology: the connectivity's offsets, shared, not copied.
    assert td.L2(data).offsets is data.topology.offsets
    assert td.L2(data).size == 8 + 8 + 5
    grid = td.rectilinear_grid([0, 1, 2], [0, 1, 2], [0, 1])
    np.testing.assert_array_equal(td.L2(grid).offsets.to_numpy(), 8 * np.arange(5))
    assert td.L2(grid).size == 32
    # Sizes follow the entities the values live on.
    assert td.H1(data).size == 13 and td.Constant(data).size == 3
    assert td.Values(data, "faces").size == 15 and td.Values(data, "edges").size == 24
    # Two fields on one space share it, and its layout, entirely.
    a = td.Field(td.L2(data), _scalars(np.zeros(21)))
    b = td.Field(td.L2(data), td.ConstantArray(1.0, 21))
    assert a.space is b.space


def test_a_field_holds_its_space_size(backend):
    data = _two_hexes_and_a_pyramid()
    with pytest.raises(ValueError, match="holds 13 values, not 12"):
        td.Field(td.H1(data), _scalars(np.zeros(12)))
    with pytest.raises(TypeError, match="a Space on a topology"):
        td.Field("H1", _scalars(np.zeros(13)))


def test_points_no_cell_uses_still_count(backend):
    """VTK lets a dataset have points no cell uses; the topology is told how many."""
    data = _two_hexes_and_a_pyramid()
    t = data.topology
    wider = td.UnstructuredTopology(t.types, t.offsets, t.connectivity, num_points=15)
    positions = np.vstack([data.positions(), [[9, 9, 9], [8, 8, 8]]])
    wide = td.DataSet(wider, positions)
    assert wide.num_points == td.H1(wide).size == 15
    np.testing.assert_allclose(alg.cell_centers(wide).values.to_numpy(vectors=True),
                               alg.cell_centers(data).values.to_numpy(vectors=True))


def test_fields_stay_on_their_topology(backend):
    data = _two_hexes_and_a_pyramid()
    other = _two_hexes_and_a_pyramid()
    u = _height(other)
    with pytest.raises(ValueError, match="another topology"):
        alg.values_at_centers(data, u)
    with pytest.raises(ValueError, match="another topology"):
        td.DataSet(data.topology, data.geometry.values, fields={"u": u})


# ── Variable order: groups from the fields' spaces ──────────────────

def _cell_points(data):
    """Each cell's point ids, host lists: the connectivity, or a hexahedral grid's."""
    t = data.topology
    if hasattr(t, "connectivity"):
        conn, offs = t.connectivity.to_numpy(), t.offsets.to_numpy()
        return [conn[offs[c]:offs[c + 1]] for c in range(t.num_cells)]
    nx, ny, nz = t.point_dims
    rows = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                b = i + nx * (j + ny * k)
                bottom = [b, b + 1, b + 1 + nx, b + nx]
                rows.append(np.array(bottom + [p + nx * ny for p in bottom]))
    return rows


def _grids():
    return {"mixed": _two_hexes_and_a_pyramid(),
            "rectilinear": td.rectilinear_grid([0, 1, 3, 4], [0, 2, 3], [0, 1, 1.5])}


def _variable_height(data, orders):
    """The linear height in an L2 space of an order per cell: order-1 cells hold
    it at their corners, order-0 cells its value at their center."""
    space = td.L2(data, order=orders)
    height = _height(data).values.to_numpy()
    centers = alg.cell_centers(data).values.to_numpy(vectors=True) @ [1, 2, -1]
    offsets = space.offsets.to_numpy()
    values = np.empty(space.size)
    for c, points in enumerate(_cell_points(data)):
        if orders[c] == 1:
            values[offsets[c]:offsets[c + 1]] = height[points]
        else:
            values[offsets[c]] = centers[c]
    return td.Field(space, _scalars(values)), centers


@tack.kernel
def _orders_seen(cells, u, v, out):
    for c in cells:
        out[cells.entity_id(c)] = u.ORDER * 10 + v.ORDER


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_each_cell_launches_with_its_own_order(backend, make):
    data = _grids()[make]
    n = data.num_cells
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
    a[0], a[-1] = 0, 1                                  # both orders present
    u = td.Field(td.L2(data, order=a), _scalars(np.zeros(td.L2(data, order=a).size)))
    v = td.Field(td.L2(data, order=b), _scalars(np.zeros(td.L2(data, order=b).size)))
    out = tack.zeros(tack.i32, (n,))
    td.for_each(_orders_seen, data, "cells", u, v, out)
    np.testing.assert_array_equal(out.to_numpy(), 10 * a + b)
    # One launch per (shape, order of u, order of v) present.
    groups = data.launch_groups("cells", [u, v])
    expected = {(g.shape.__name__, int(x), int(y))
                for g in data.topology.groups()
                for x, y in zip(a[_group_ids(g)], b[_group_ids(g)])}
    assert {(g.shape.__name__, g.keys[u.space], g.keys[v.space]) for g in groups} == expected
    assert sum(g.count for g in groups) == n


def _group_ids(group):
    if hasattr(group.args[1], "to_numpy"):
        return group.args[1].to_numpy()                 # an explicit group's cell ids
    return np.arange(group.count)                       # a structured group's


@tack.kernel
def _face_ids_seen(cells, u, out):
    for c in cells:
        total = 0
        for f in range(cells.NUM_FACES):
            total += cells.face_id(c, f) * (f + 1)
        out[cells.entity_id(c)] = total + 0 * u.ORDER


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_subgroups_keep_their_cells_incidence(backend, make):
    data = _grids()[make]
    data.topology.faces()
    n = data.num_cells
    orders = np.arange(n) % 2
    varying = td.Field(td.L2(data, order=orders), td.ConstantArray(0.0, td.L2(data, order=orders).size))
    uniform = td.Field(td.L2(data), td.ConstantArray(0.0, td.L2(data).size))
    got, expected = tack.zeros(tack.i32, (n,)), tack.zeros(tack.i32, (n,))
    td.for_each(_face_ids_seen, data, "cells", varying, got)
    td.for_each(_face_ids_seen, data, "cells", uniform, expected)
    np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_variable_order_values(backend, make):
    data = _grids()[make]
    n = data.num_cells
    orders = np.zeros(n, int)
    orders[::2] = 1
    u, centers = _variable_height(data, orders)
    assert u.space.size == sum(len(p) if o else 1 for p, o in zip(_cell_points(data), orders))
    np.testing.assert_allclose(alg.values_at_centers(data, u).values.to_numpy(), centers,
                               atol=1e-5)
    gradient = alg.gradients(data, u).values.to_numpy(vectors=True)
    np.testing.assert_allclose(gradient, np.outer(orders, [1, 2, -1]), atol=1e-4)
    # Each point averages what its cells give it: the height from order-1 cells,
    # the cell's one value from order-0 cells.
    height = _height(data).values.to_numpy()
    given = [[] for _ in range(data.num_points)]
    for c, points in enumerate(_cell_points(data)):
        for p in points:
            given[p].append(height[p] if orders[c] else centers[c])
    np.testing.assert_allclose(alg.to_points(data, u).values.to_numpy(),
                               [np.mean(g) for g in given], atol=1e-5)


def test_variable_order_spaces(backend):
    data = _two_hexes_and_a_pyramid()
    orders = np.array([1, 0, 1])
    assert td.L2(data, order=orders) is td.L2(data, order=orders)
    assert td.L2(data, order=orders) is not td.L2(data, order=orders.copy())
    assert td.L2(data, order=orders).varies and not td.L2(data).varies
    np.testing.assert_array_equal(td.L2(data, order=orders).offsets.to_numpy(), [0, 8, 9, 14])
    np.testing.assert_array_equal(td.L2(data, order=0).offsets.to_numpy(), [0, 1, 2, 3])
    with pytest.raises(NotImplementedError, match=r"orders \[2\]"):
        td.L2(data, order=np.array([1, 2, 1])).cell_keys()
    with pytest.raises(ValueError, match="3 orders, not 2"):
        td.L2(data, order=np.array([1, 0])).cell_keys()
    u = td.Field(td.L2(data, order=orders), _scalars(np.zeros(14)))
    with pytest.raises(ValueError, match="for_each makes"):
        u.view(data.topology.groups()[0])


# ── Face orientation and per-side traces ────────────────────────────

@tack.kernel
def _face_positions(cells, points, slots, reflected):
    for c in cells:
        for f in range(cells.NUM_FACES):
            face = cells.face_id(c, f)
            for k in range(cells.face_num_points(f)):
                at = (cells.entity_id(c) * 6 + f) * 4 + k
                points[at] = cells.point_id(c, cells.face_corner(f, k))
                slots[at] = face * 4 + cells.face_position(c, f, k)
            reflected[cells.entity_id(c) * 6 + f] = (cells.face_orientation(c, f) & 1) \
                - cells.face_side(c, f)


def _check_orientation(data):
    """Every cell's face points land, through its orientation, on the same points of
    the face's own row; and a cell goes round its face backwards exactly when it is
    side 1, its outward normal being the other way."""
    faces = data.topology.faces()
    n = data.num_cells * 6 * 4
    points = _scalars(np.full(n, -1), tack.i32)
    slots = _scalars(np.full(n, -1), tack.i32)
    reflected = _scalars(np.zeros(data.num_cells * 6), tack.i32)
    td.for_each(_face_positions, data, "cells", points, slots, reflected)
    points, slots = points.to_numpy(), slots.to_numpy()
    used = slots >= 0
    rows = faces.rows.to_numpy()
    np.testing.assert_array_equal(rows[slots[used]], points[used])
    assert used.sum() == sum(4 if k == 9 else 3 for k in faces.kinds.to_numpy()
                             for _ in range(2)) - sum(
        4 if k == 9 else 3 for k, s in zip(faces.kinds.to_numpy(),
                                           faces.sides.to_numpy(vectors=True)) if s[2] < 0)
    np.testing.assert_array_equal(reflected.to_numpy(), 0)


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_face_orientation(backend, make):
    _check_orientation(_grids()[make])


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_face_orientation_on_every_solid(backend, kind):
    _check_orientation(vtk_to_dataset(_vtk_cells(kind)))


def _with_cell_offsets(data, field, step):
    """``field`` in the DG layout with ``step`` times each cell's id added to its
    own values: cells disagree by known amounts wherever they meet."""
    dg = alg.discontinuous(data, field)
    offsets = td.L2(data).offsets.to_numpy()
    values = dg.values.to_numpy()
    for c in range(data.num_cells):
        values[offsets[c]:offsets[c + 1]] += step * c
    return td.Field(dg.space, _scalars(values))


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_traces_line_up(backend, make):
    data = _grids()[make]
    faces = data.topology.faces()
    u = _height(data)
    t = alg.traces(data, u)
    assert t.space is td.SideTraces(data)
    values = t.values.to_numpy().reshape(-1, 2, 4)
    rows = faces.rows.to_numpy(vectors=True)
    sides = faces.sides.to_numpy(vectors=True)
    height = u.values.to_numpy()
    for f, (row, side) in enumerate(zip(rows, sides)):
        n = 3 if faces.kinds[f] == 5 else 4
        np.testing.assert_allclose(values[f, 0, :n], height[row[:n]], atol=1e-5)
        if side[2] >= 0:
            np.testing.assert_allclose(values[f, 1, :n], height[row[:n]], atol=1e-5)
    # A continuous field has no jumps; a DG one has its own.
    np.testing.assert_allclose(alg.jump(data, u).values.to_numpy(), 0, atol=1e-5)
    dg = _with_cell_offsets(data, u, 10.0)
    expected = np.where(sides[:, 2] >= 0, 10.0 * (sides[:, 2] - sides[:, 0]), 0.0)
    np.testing.assert_allclose(alg.jump(data, dg).values.to_numpy(), expected, atol=1e-4)


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_jumps_on_every_solid(backend, kind):
    data = vtk_to_dataset(_vtk_cells(kind, (2, 2, 2)))
    sides = data.topology.faces().sides.to_numpy(vectors=True)
    dg = _with_cell_offsets(data, _height(data), 10.0)
    expected = np.where(sides[:, 2] >= 0, 10.0 * (sides[:, 2] - sides[:, 0]), 0.0)
    # Values reach 10 * cells (about 500 here): a difference of two f32 traces
    # that size is good to a few units in their last place.
    np.testing.assert_allclose(alg.jump(data, dg).values.to_numpy(), expected,
                               atol=1e-6 * 10.0 * data.num_cells * 8)


def test_traces_of_a_variable_order_field(backend):
    data = _two_hexes_and_a_pyramid()
    orders = np.array([1, 0, 1])
    u, centers = _variable_height(data, orders)
    values = alg.traces(data, u).values.to_numpy().reshape(-1, 2, 4)
    sides = data.topology.faces().sides.to_numpy(vectors=True)
    kinds = data.topology.faces().kinds.to_numpy()
    for f, side in enumerate(sides):
        n = 3 if kinds[f] == 5 else 4
        for s in range(2 if side[2] >= 0 else 1):
            if orders[side[2 * s]] == 0:            # an order-0 cell: its one value
                np.testing.assert_allclose(values[f, s, :n], centers[side[2 * s]], atol=1e-5)


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_upwind_flux(backend, make):
    data = _grids()[make]
    velocity = (1.0, 0.5, -0.25)
    # A constant is carried through every closed cell unchanged: no net outflow.
    constant = td.Field(td.Constant(data), td.ConstantArray(2.0, data.num_cells))
    net = alg.divergence(data, alg.upwind_flux(data, constant, velocity))
    np.testing.assert_allclose(net.values.to_numpy(), 0, atol=1e-4)
    # Each face carries the value of the cell the flow leaves.
    values = np.arange(data.num_cells, dtype=float) + 1
    flux = alg.upwind_flux(data, td.Field(td.Constant(data), _scalars(values)), velocity)
    normals, areas = alg.face_geometry(data)
    vn = normals.values.to_numpy(vectors=True) @ velocity
    sides = data.topology.faces().sides.to_numpy(vectors=True)
    upwind = np.where((vn < 0) & (sides[:, 2] >= 0), sides[:, 2], sides[:, 0])
    np.testing.assert_allclose(flux.values.to_numpy(),
                               vn * areas.values.to_numpy() * values[upwind], rtol=1e-5,
                               atol=1e-6)


# ── Quadratic bases ─────────────────────────────────────────────────

@tack.kernel
def _quadratic_at(shape: tack.template(), pcs, values, gradients, nodes, n):
    for i in range(n):
        for k in range(shape.NUM_QUADRATIC):
            values[i * 27 + k] = shape.quadratic_function(k, pcs[i])
            gradients[i * 27 + k] = shape.quadratic_gradient(k, pcs[i])
            if i == 0:
                nodes[k] = shape.quadratic_node(k)


@pytest.mark.parametrize("name", ["Tetra", "Hexahedron", "Voxel", "Wedge"])
def test_quadratic_bases(backend, name):
    """Each shape's quadratic Lagrange functions: one at their own node and zero at
    the others, summing to one, with gradients their finite differences."""
    from tack.data import shapes

    cls = getattr(shapes, name)
    n = cls.NUM_QUADRATIC
    nodes = tack.Vector.field(3, tack.f32, shape=(27,))

    def evaluate(points):
        pcs = _vectors(points)
        values = tack.field(tack.f32, shape=(27 * len(points),))
        gradients = tack.Vector.field(3, tack.f32, shape=(27 * len(points),))
        _quadratic_at(cls(), pcs, values, gradients, nodes, len(points))
        return (values.to_numpy().reshape(-1, 27)[:, :n],
                gradients.to_numpy(vectors=True).reshape(-1, 27, 3)[:, :n])

    evaluate(np.zeros((1, 3)))
    at_nodes = nodes.to_numpy(vectors=True)[:n]
    assert len(np.unique(at_nodes.round(5), axis=0)) == n
    values, _ = evaluate(at_nodes)
    np.testing.assert_allclose(values, np.eye(n), atol=1e-6)
    points = np.random.default_rng(3).uniform(0.05, 0.3, (6, 3))
    values, gradients = evaluate(points)
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-5)
    h = 1e-2
    for d in range(3):
        up, _ = evaluate(points + h * np.eye(3)[d])
        down, _ = evaluate(points - h * np.eye(3)[d])
        np.testing.assert_allclose((up - down) / (2 * h), gradients[:, :, d], atol=1e-3)
