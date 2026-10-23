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
import tack.dataset as td
from tack.dataset import algorithms as alg
from tack.dataset.algorithms.values_at_centers import _at_centers
from tack.runtime.dispatch import env_flag

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

try:
    from vtkmodules import vtkCommonCore, vtkCommonDataModel, vtkFiltersCore
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
    f = tack.empty(values.shape, dtype)
    tack.copyto(f, values.astype(dtype.numpy_dtype))
    return f


def _vectors(values, dtype=tack.f32):
    values = np.asarray(values)
    f = tack.empty((values.shape[0],), tack.vec(values.shape[1], dtype))
    tack.copyto(f, values.astype(dtype.numpy_dtype))
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
    sides = faces.sides.numpy()
    shared = {tuple(s) for s in sides if s[2] >= 0}
    assert shared == {(0, 1, 1, 0), (0, 5, 2, 0)}   # hex 0's +x and top faces
    assert len(alg.boundary_faces(data)) == 13

    jumps = alg.jump(data, td.Field(td.Constant(data), _scalars([1.0, 2.0, 5.0])))
    by_sides = dict(zip(map(tuple, sides), jumps.values.numpy()))
    assert by_sides[(0, 1, 1, 0)] == 1.0 and by_sides[(0, 5, 2, 0)] == 4.0
    assert np.count_nonzero(jumps.values.numpy()) == 2

    # A face counts plus for its side-0 cell and minus for its side-1 cell.
    _, areas = alg.face_geometry(data)
    slant = np.sqrt(0.5 ** 2 + 1.0 ** 2) / 2         # each pyramid side triangle
    np.testing.assert_allclose(alg.net_flux(data, areas).values.numpy(),
                               [6, 5 - 1, 4 * slant - 1], rtol=1e-6)


def _closed_cells_check(data):
    """Every cell's outward area vectors sum to zero: the faces' normals point out of
    side 0, and each cell knows which side it is."""
    normals, areas = alg.face_geometry(data)
    area_vectors = normals.values.numpy() * areas.values.numpy()[:, None]
    flux = td.Field(td.Values(data, "faces"), _vectors(area_vectors))
    sums = alg.net_flux(data, flux).values.numpy()
    np.testing.assert_allclose(sums, 0, atol=1e-5)
    return areas.values.numpy()


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
    boundary = alg.boundary_faces(data).numpy()
    np.testing.assert_allclose(areas[boundary].sum(), expected, rtol=1e-6)
    lengths = alg.edge_lengths(data).values.numpy()
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
    ids, signs = tack.empty((n,), tack.i32), tack.empty((n,), tack.i32)
    td.for_each(_edge_incidence, data, "cells", ids, signs)
    rows = edges.rows.numpy()
    assert (rows[:, 0] < rows[:, 1]).all()
    # VTK's hexahedron edges, and the cells' point ids in x-fastest order.
    hex_edges = [(0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7),
                 (0, 4), (1, 5), (3, 7), (2, 6)]
    nx, ny = 3, 4
    ids, signs = ids.numpy().reshape(-1, 12), signs.numpy().reshape(-1, 12)
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
    return vtkCellTypeSource(cell_type=SOLID_TYPES[kind], blocks_dimensions=blocks)()


def _vtk_faces(grid):
    """Every face of every cell: {sorted ids: [(cell, local face, outward ids)]}."""
    faces = {}
    for c in range(grid.number_of_cells):
        cell = grid.GetCell(c)
        for f in range(cell.number_of_faces):
            face = cell.GetFace(f)
            ids = [face.GetPointId(k) for k in range(face.number_of_points)]
            if face.cell_type == 8:                  # a pixel, as a quad
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
    rows = faces.rows.numpy()
    sides = faces.sides.numpy()
    kinds = faces.kinds.numpy()
    for row, side, k in zip(rows, sides, kinds):
        ids = row[:3] if k == 5 else row
        owners = expected[tuple(sorted(ids))]
        assert len(owners) == (2 if side[2] >= 0 else 1)
        assert {(c, f) for c, f, _ in owners} == {(side[0], side[1]), (side[2], side[3])} - {(-1, -1)}
        outward = next(o for c, f, o in owners if (c, f) == (side[0], side[1]))
        assert _same_cycle(ids, outward)
    assert len(alg.boundary_faces(data)) == vtkGeometryFilter()(grid).number_of_cells
    _closed_cells_check(data)


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_edges_match_vtk(backend, kind):
    grid = _vtk_cells(kind)
    data = vtk_to_dataset(grid)
    lines = vtkExtractEdges()(grid)
    expected = {tuple(sorted(lines.GetCell(i).point_ids.GetId(k) for k in range(2)))
                for i in range(lines.number_of_cells)}
    rows = data.topology.edges().rows.numpy()
    assert {tuple(r) for r in rows} == expected
    assert len(rows) == len(expected)


# ── Fields on spaces ────────────────────────────────────────────────

@needs_vtk
@pytest.mark.parametrize("kind", ["tetra", "hexahedron", "wedge", "pyramid"])
def test_cell_centers_match_vtk(backend, kind):
    grid = _vtk_cells(kind, (2, 2, 1))
    expected = np.asarray(vtk.vtkCellCenters()(grid).points.data)
    got = alg.cell_centers(vtk_to_dataset(grid)).values.numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5 if kind == "wedge" else 1e-6)


@needs_vtk
def test_cell_to_point_matches_vtk(backend):
    grid = _vtk_cells("tetra", (2, 2, 2))
    grid.cell_data["v"] = np.random.default_rng(3).random(grid.number_of_cells)
    expected = np.asarray(vtk.vtkCellDataToPointData()(grid).point_data["v"])
    data = vtk_to_dataset(grid)
    assert data.fields["v"].space is td.Constant(data)
    got = alg.project(data, data.fields["v"], td.H1(data))
    assert got.space is td.H1(data)
    np.testing.assert_allclose(got.values.numpy(), expected, rtol=1e-5)


def _height(data):
    """An H1 field, linear in position: every linear cell interpolates it exactly."""
    x = data.positions()
    return td.Field(td.H1(data), _scalars(x[:, 0] + 2 * x[:, 1] - x[:, 2]))


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_continuous_field_in_dg_layout(backend, make):
    data = (_two_hexes_and_a_pyramid() if make == "mixed"
            else td.rectilinear_grid([0, 1, 3], [0, 2, 3], [0, 1]))
    u = _height(data)
    dg = alg.project(data, u, td.L2(data))
    assert dg.space is td.L2(data)
    corners = sum(g.count * g.shape.NUM_POINTS for g in data.topology.groups())
    assert dg.values.shape == (corners,)
    # The same function: the same values at the centers, and back on the points.
    np.testing.assert_allclose(alg.values_at_centers(data, dg).values.numpy(),
                               alg.values_at_centers(data, u).values.numpy(), atol=1e-5)
    np.testing.assert_allclose(alg.project(data, dg, td.H1(data)).values.numpy(),
                               u.values.numpy(), atol=1e-5)
    centers = alg.cell_centers(data).values.numpy()
    np.testing.assert_allclose(alg.values_at_centers(data, u).values.numpy(),
                               centers @ [1, 2, -1], atol=1e-5)


def test_dg_values_can_disagree(backend):
    """Each cell owns its corner values: add the cell id to cell c's, and every point
    averages its cells' ids on top of the continuous value."""
    data = _two_hexes_and_a_pyramid()
    u = _height(data)
    dg = alg.project(data, u, td.L2(data))
    offsets = td.L2(data).offsets.numpy()
    values = dg.values.numpy()
    for c in range(data.num_cells):
        values[offsets[c]:offsets[c + 1]] += 10 * c
    dg = td.Field(td.L2(data), _scalars(values))
    connectivity = data.topology.connectivity.numpy()
    expected = u.values.numpy().copy()
    for p in range(data.num_points):
        cells = [c for c in range(data.num_cells)
                 if p in connectivity[offsets[c]:offsets[c + 1]]]
        expected[p] += 10 * np.mean(cells)
    np.testing.assert_allclose(alg.project(data, dg, td.H1(data)).values.numpy(), expected,
                               rtol=1e-6)
    # At a center each cell sees only its own values.
    np.testing.assert_allclose(alg.values_at_centers(data, dg).values.numpy(),
                               alg.values_at_centers(data, u).values.numpy()
                               + 10 * np.arange(3), rtol=1e-6)


@needs_vtk
def test_surface_of_a_side_set(backend):
    grid = _vtk_cells("wedge")
    data = vtk_to_dataset(grid)
    _, areas = alg.face_geometry(data)
    data.fields["area"] = areas
    data.sets["boundary"] = alg.boundary_faces(data)
    surface = alg.extract_surface(data, "boundary")
    assert surface.fields["area"].space is td.Values(surface, "cells")
    assert surface.num_cells == vtkGeometryFilter()(grid).number_of_cells
    # The surface's own faces, as cells of a 2D mesh, cover the same area.
    total = areas.values.numpy()[data.sets["boundary"].numpy()].sum()
    np.testing.assert_allclose(surface.fields["area"].values.numpy().sum(), total)
    out = dataset_to_vtk(surface)
    assert out.cell_data["area"].number_of_tuples == surface.num_cells


@tack.kernel
def _set_areas(faces, out):
    for f in faces:
        out[faces.entity_id(f)] = faces.num_sides(f)


def test_for_each_over_a_side_set(backend):
    data = td.rectilinear_grid([0, 1, 2, 3], [0, 1, 2], [0, 1])
    boundary = data.sets["boundary"] = alg.boundary_faces(data)
    out = tack.zeros((data.topology.faces().num_faces,), tack.i32)
    td.for_each(_set_areas, data, "boundary", out)
    out = out.numpy()
    np.testing.assert_array_equal(out[boundary.numpy()], 1)
    assert np.count_nonzero(out) == len(boundary)


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
    assert isinstance(grid, vtk.vtkRectilinearGrid)
    assert "shape" not in grid.point_data.keys()      # the geometry is the grid's
    back = vtk_to_dataset(grid)
    assert back.fields["h"].space is td.H1(back) and back.fields["c"].space is td.Constant(back)
    np.testing.assert_allclose(back.positions(), data.positions())
    np.testing.assert_allclose(back.fields["h"].values.numpy(),
                               data.fields["h"].values.numpy())



@needs_vtk
@pytest.mark.parametrize("make", ["as_polygons", "polyhedral_contour"])
def test_polygonal_round_trip(backend, make):
    """A polygonal topology exports as VTK_POLYGON cells, which the import read
    as linear shapes and refused ("cell types [7] are not linear shapes")."""
    if make == "as_polygons":
        data = td.as_polygons(td.extract_surface(td.uniform_grid((3, 3, 3))))
    else:
        poly = td.as_polyhedra(td.uniform_grid((4, 4, 4)))
        data = td.contour(poly, _height(poly), 3.0)
    assert isinstance(data.topology, td.PolygonalTopology) and data.num_cells
    data.fields["c"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    grid = dataset_to_vtk(data)
    back = vtk_to_dataset(grid)
    assert isinstance(back.topology, td.PolygonalTopology)
    assert back.num_cells == data.num_cells
    for a, b in zip(back.topology.loops(), data.topology.loops()):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    np.testing.assert_allclose(back.positions(), data.positions())
    np.testing.assert_array_equal(back.fields["c"].values.numpy(), np.arange(data.num_cells))


@needs_vtk
@pytest.mark.parametrize("make", ["warp", "transform"])
def test_explicit_structured_round_trip(backend, make):
    """A warped or transformed grid keeps its structured topology over explicit
    coordinates, which the export refused ("unsupported topology
    StructuredTopology with Array"); it is a vtkStructuredGrid."""
    data = td.rectilinear_grid([0, 1, 3], [0, 2], [0, 1, 1.5])
    if make == "warp":
        data = td.warp(data, (0, 0, 1), scale=0.5)
    else:
        data = td.transform(data, np.array([[0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 2],
                                            [0, 0, 0, 1]], np.float64))
    assert isinstance(data.topology, td.StructuredTopology)
    data.fields["h"] = _height(data)
    grid = dataset_to_vtk(data)
    assert isinstance(grid, vtk.vtkStructuredGrid) and grid.number_of_cells == data.num_cells
    back = vtk_to_dataset(grid)
    assert isinstance(back.topology, td.StructuredTopology)
    assert back.topology.point_dims == data.topology.point_dims
    np.testing.assert_allclose(back.positions(), data.positions(), atol=1e-6)
    np.testing.assert_allclose(back.fields["h"].values.numpy(),
                               data.fields["h"].values.numpy(), atol=1e-6)


# ── Geometry is a field, kept apart ─────────────────────────────────

def test_geometry_is_a_field_kept_apart(backend):
    data = _two_hexes_and_a_pyramid()
    assert data.geometry.space is td.H1(data)
    assert not any(f is data.geometry for f in data.fields.values())
    grid = td.rectilinear_grid([0, 1, 2], [0, 1], [0, 1])
    assert grid.geometry.space is td.H1(grid)
    assert isinstance(grid.geometry.values, tack.arrays.CartesianProduct)
    # "shape" is an ordinary name now, and the geometry can be replaced in place.
    named = td.DataSet(data.topology, data.positions(),
                       fields={"shape": td.Field(td.Constant(data), tack.arrays.ConstantArray(1.0, 3))})
    assert named.fields["shape"].space is td.Constant(data)
    moved = data.positions() + 1.0
    named.geometry = moved
    np.testing.assert_allclose(named.positions(), moved)
    with pytest.raises(TypeError, match="H1 or L2"):
        named.geometry = td.Field(td.Values(data, "points"), data.geometry.values)
    with pytest.raises(TypeError, match="H1 or L2"):
        td.DataSet(data.topology, td.Field(td.Values(data, "points"), data.geometry.values))


@tack.kernel
def _jacobian_determinants(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = cells.geometry_jacobian(c, cells.parametric_center()).determinant()


def test_geometry_jacobian(backend):
    x, y, z = [0, 1, 3], [0, 0.5, 2], [0, 4]
    data = td.rectilinear_grid(x, y, z)
    out = tack.empty((data.num_cells,))
    td.for_each(_jacobian_determinants, data, "cells", out)
    volumes = np.multiply.outer(np.multiply.outer(np.diff(z), np.diff(y)), np.diff(x))
    np.testing.assert_allclose(out.numpy(), volumes.reshape(-1), rtol=1e-6)


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_gradients_of_a_linear_field(backend, make):
    data = (_two_hexes_and_a_pyramid() if make == "mixed"
            else td.rectilinear_grid([0, 1, 3], [0, 2, 3], [0, 1, 1.5]))
    u = _height(data)
    for field in (u, alg.project(data, u, td.L2(data))):
        got = alg.gradients(data, field).values.numpy()
        np.testing.assert_allclose(got, np.tile([1, 2, -1], (data.num_cells, 1)), atol=1e-4)
    constant = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    np.testing.assert_array_equal(alg.gradients(data, constant).values.numpy(), 0)


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_gradients_on_every_solid(backend, kind):
    data = vtk_to_dataset(_vtk_cells(kind, (2, 2, 2)))
    got = alg.gradients(data, _height(data)).values.numpy()
    np.testing.assert_allclose(got, np.tile([1, 2, -1], (data.num_cells, 1)), atol=1e-4)


def _shrunk(data, s):
    """``data`` with an L2 geometry: each cell's corners pulled toward its center by
    ``s``. The topology, and so the faces, are unchanged."""
    offsets = td.L2(data).offsets.numpy()
    connectivity = data.topology.connectivity.numpy()
    centers = alg.cell_centers(data).values.numpy()
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
    np.testing.assert_allclose(alg.cell_centers(shrunk).values.numpy(),
                               alg.cell_centers(data).values.numpy(), atol=1e-6)
    # The same point values over cells half the size: twice the gradient.
    got = alg.gradients(shrunk, shrunk.fields["u"]).values.numpy()
    np.testing.assert_allclose(got, np.tile([2, 4, -2], (3, 1)), atol=1e-4)
    # Faces come from the topology, which still has the cells share points; a face
    # is where its side 0 puts it: half the size each way, facing the same way.
    assert shrunk.topology.faces().num_faces == 15
    normals, areas = alg.face_geometry(shrunk)
    original_normals, original_areas = alg.face_geometry(data)
    np.testing.assert_allclose(areas.values.numpy(), 0.25 * original_areas.values.numpy(),
                               rtol=1e-5)
    np.testing.assert_allclose(normals.values.numpy(),
                               original_normals.values.numpy(), atol=1e-5)
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
        alg.values_at_centers(data, data.geometry).values.numpy(),
        alg.cell_centers(data).values.numpy(), rtol=1e-6)
    np.testing.assert_allclose(data.positions()[:4], [[0, 0, 0], [1, 0, 0], [3, 0, 0],
                                                      [0, 2, 0]])


def test_structured_cells_read_cartesian_values_by_ijk(backend):
    from tack.dataset import views

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

        out = tack.empty((data.num_cells,), tack.vec(3, tack.f32))
        src = tack.inspect(_at_centers, data.domain_view("cells", cells),
                           data.geometry.view(cells), out, mode="source")
        assert not re.search(r"\bs(?:div|rem)\b", src)


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_implicit_values_match_explicit(backend, make):
    data = (_two_hexes_and_a_pyramid() if make == "mixed"
            else td.rectilinear_grid([0, 1, 3], [0, 2, 3], [0, 1, 1.5]))
    n, m = data.num_points, data.num_cells
    pairs = [
        (td.Field(td.H1(data), tack.arrays.ConstantArray(3.0, n)), td.Field(td.H1(data), _scalars(np.full(n, 3.0)))),
        (td.Field(td.H1(data), tack.arrays.CountingArray(n, 0.0, 0.5)),
         td.Field(td.H1(data), _scalars(0.5 * np.arange(n)))),
    ]
    for implicit, explicit in pairs:
        for algorithm in (alg.values_at_centers, alg.gradients):
            np.testing.assert_allclose(algorithm(data, implicit).values.numpy(),
                                       algorithm(data, explicit).values.numpy(), atol=1e-5)
    ids = td.Field(td.Constant(data), tack.arrays.CountingArray(m, 0.0, 1.0))
    np.testing.assert_allclose(alg.jump(data, ids).values.numpy(),
                               alg.jump(data, td.Field(td.Constant(data), _scalars(np.arange(m))))
                               .values.numpy())
    np.testing.assert_allclose(alg.project(data, ids, td.H1(data)).values.numpy(),
                               alg.project(data, td.Field(td.Constant(data), _scalars(np.arange(m))), td.H1(data))
                               .values.numpy(), rtol=1e-6)


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
    np.testing.assert_array_equal(td.L2(grid).offsets.numpy(), 8 * np.arange(5))
    assert td.L2(grid).size == 32
    # Sizes follow the entities the values live on.
    assert td.H1(data).size == 13 and td.Constant(data).size == 3
    assert td.Values(data, "faces").size == 15 and td.Values(data, "edges").size == 24
    # Two fields on one space share it, and its layout, entirely.
    a = td.Field(td.L2(data), _scalars(np.zeros(21)))
    b = td.Field(td.L2(data), tack.arrays.ConstantArray(1.0, 21))
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
    np.testing.assert_allclose(alg.cell_centers(wide).values.numpy(),
                               alg.cell_centers(data).values.numpy())


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
        conn, offs = t.connectivity.numpy(), t.offsets.numpy()
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
    height = _height(data).values.numpy()
    centers = alg.cell_centers(data).values.numpy() @ [1, 2, -1]
    offsets = space.offsets.numpy()
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
    out = tack.zeros((n,), tack.i32)
    td.for_each(_orders_seen, data, "cells", u, v, out)
    np.testing.assert_array_equal(out.numpy(), 10 * a + b)
    # One launch per (shape, order of u, order of v) present.
    groups = data.launch_groups("cells", [u, v])
    expected = {(g.shape.__name__, int(x), int(y))
                for g in data.topology.groups()
                for x, y in zip(a[_group_ids(g)], b[_group_ids(g)])}
    assert {(g.shape.__name__, g.keys[u.space], g.keys[v.space]) for g in groups} == expected
    assert sum(g.count for g in groups) == n


def _group_ids(group):
    if hasattr(group.args[1], "numpy"):
        return group.args[1].numpy()                 # an explicit group's cell ids
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
    varying = td.Field(td.L2(data, order=orders), tack.arrays.ConstantArray(0.0, td.L2(data, order=orders).size))
    uniform = td.Field(td.L2(data), tack.arrays.ConstantArray(0.0, td.L2(data).size))
    got, expected = tack.zeros((n,), tack.i32), tack.zeros((n,), tack.i32)
    td.for_each(_face_ids_seen, data, "cells", varying, got)
    td.for_each(_face_ids_seen, data, "cells", uniform, expected)
    np.testing.assert_array_equal(got.numpy(), expected.numpy())


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_variable_order_values(backend, make):
    data = _grids()[make]
    n = data.num_cells
    orders = np.zeros(n, int)
    orders[::2] = 1
    u, centers = _variable_height(data, orders)
    assert u.space.size == sum(len(p) if o else 1 for p, o in zip(_cell_points(data), orders))
    np.testing.assert_allclose(alg.values_at_centers(data, u).values.numpy(), centers,
                               atol=1e-5)
    gradient = alg.gradients(data, u).values.numpy()
    np.testing.assert_allclose(gradient, np.outer(orders, [1, 2, -1]), atol=1e-4)
    # Each point averages what its cells give it: the height from order-1 cells,
    # the cell's one value from order-0 cells.
    height = _height(data).values.numpy()
    given = [[] for _ in range(data.num_points)]
    for c, points in enumerate(_cell_points(data)):
        for p in points:
            given[p].append(height[p] if orders[c] else centers[c])
    np.testing.assert_allclose(alg.project(data, u, td.H1(data)).values.numpy(),
                               [np.mean(g) for g in given], atol=1e-5)


def test_variable_order_spaces(backend):
    data = _two_hexes_and_a_pyramid()
    orders = np.array([1, 0, 1])
    assert td.L2(data, order=orders) is td.L2(data, order=orders)
    assert td.L2(data, order=orders) is not td.L2(data, order=orders.copy())
    assert td.L2(data, order=orders).varies and not td.L2(data).varies
    np.testing.assert_array_equal(td.L2(data, order=orders).offsets.numpy(), [0, 8, 9, 14])
    np.testing.assert_array_equal(td.L2(data, order=0).offsets.numpy(), [0, 1, 2, 3])
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
    points, slots = points.numpy(), slots.numpy()
    used = slots >= 0
    rows = faces.rows.numpy().reshape(-1)              # slots index components
    np.testing.assert_array_equal(rows[slots[used]], points[used])
    assert used.sum() == sum(4 if k == 9 else 3 for k in faces.kinds.numpy()
                             for _ in range(2)) - sum(
        4 if k == 9 else 3 for k, s in zip(faces.kinds.numpy(),
                                           faces.sides.numpy()) if s[2] < 0)
    np.testing.assert_array_equal(reflected.numpy(), 0)


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
    dg = alg.project(data, field, td.L2(data))
    offsets = td.L2(data).offsets.numpy()
    values = dg.values.numpy()
    for c in range(data.num_cells):
        values[offsets[c]:offsets[c + 1]] += step * c
    return td.Field(dg.space, _scalars(values))


@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_traces_line_up(backend, make):
    data = _grids()[make]
    faces = data.topology.faces()
    u = _height(data)
    t = alg.project(data, u, td.SideTraces(data))
    assert t.space is td.SideTraces(data)
    values = t.values.numpy().reshape(-1, 2, 4)
    rows = faces.rows.numpy()
    sides = faces.sides.numpy()
    height = u.values.numpy()
    for f, (row, side) in enumerate(zip(rows, sides)):
        n = 3 if faces.kinds[f] == 5 else 4
        np.testing.assert_allclose(values[f, 0, :n], height[row[:n]], atol=1e-5)
        if side[2] >= 0:
            np.testing.assert_allclose(values[f, 1, :n], height[row[:n]], atol=1e-5)
    # A continuous field has no jumps; a DG one has its own.
    np.testing.assert_allclose(alg.jump(data, u).values.numpy(), 0, atol=1e-5)
    dg = _with_cell_offsets(data, u, 10.0)
    expected = np.where(sides[:, 2] >= 0, 10.0 * (sides[:, 2] - sides[:, 0]), 0.0)
    np.testing.assert_allclose(alg.jump(data, dg).values.numpy(), expected, atol=1e-4)


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_jumps_on_every_solid(backend, kind):
    data = vtk_to_dataset(_vtk_cells(kind, (2, 2, 2)))
    sides = data.topology.faces().sides.numpy()
    dg = _with_cell_offsets(data, _height(data), 10.0)
    expected = np.where(sides[:, 2] >= 0, 10.0 * (sides[:, 2] - sides[:, 0]), 0.0)
    # Values reach 10 * cells (about 500 here): a difference of two f32 traces
    # that size is good to a few units in their last place.
    np.testing.assert_allclose(alg.jump(data, dg).values.numpy(), expected,
                               atol=1e-6 * 10.0 * data.num_cells * 8)


def test_traces_of_a_variable_order_field(backend):
    data = _two_hexes_and_a_pyramid()
    orders = np.array([1, 0, 1])
    u, centers = _variable_height(data, orders)
    values = alg.project(data, u, td.SideTraces(data)).values.numpy().reshape(-1, 2, 4)
    sides = data.topology.faces().sides.numpy()
    kinds = data.topology.faces().kinds.numpy()
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
    constant = td.Field(td.Constant(data), tack.arrays.ConstantArray(2.0, data.num_cells))
    net = alg.net_flux(data, alg.upwind_flux(data, constant, velocity))
    np.testing.assert_allclose(net.values.numpy(), 0, atol=1e-4)
    # Each face carries the value of the cell the flow leaves.
    values = np.arange(data.num_cells, dtype=float) + 1
    flux = alg.upwind_flux(data, td.Field(td.Constant(data), _scalars(values)), velocity)
    normals, areas = alg.face_geometry(data)
    vn = normals.values.numpy() @ velocity
    sides = data.topology.faces().sides.numpy()
    upwind = np.where((vn < 0) & (sides[:, 2] >= 0), sides[:, 2], sides[:, 0])
    np.testing.assert_allclose(flux.values.numpy(),
                               vn * areas.values.numpy() * values[upwind], rtol=1e-5,
                               atol=1e-6)


# ── Quadratic bases ─────────────────────────────────────────────────

@tack.kernel
def _quadratic_at(shape, pcs, values, gradients, nodes, n):
    for i in range(n):
        for k in range(shape.NUM_QUADRATIC):
            values[i * 27 + k] = shape.quadratic_function(k, pcs[i])
            gradients[i * 27 + k] = shape.quadratic_gradient(k, pcs[i])
            if i == 0:
                nodes[k] = shape.quadratic_node(k)


@pytest.mark.parametrize("name", ["Tetra", "Hexahedron", "Voxel", "Wedge", "Triangle", "Quad",
                                  "Pixel"])
def test_quadratic_bases(backend, name):
    """Each shape's quadratic Lagrange functions: one at their own node and zero at
    the others, summing to one, with gradients their finite differences."""
    from tack.dataset import shapes

    cls = getattr(shapes, name)
    n = cls.NUM_QUADRATIC
    nodes = tack.empty((27,), tack.vec(3, tack.f32))

    def evaluate(points):
        pcs = _vectors(points)
        values = tack.empty((27 * len(points),))
        gradients = tack.empty((27 * len(points),), tack.vec(3, tack.f32))
        _quadratic_at(cls(), pcs, values, gradients, nodes, len(points))
        return (values.numpy().reshape(-1, 27)[:, :n],
                gradients.numpy().reshape(-1, 27, 3)[:, :n])

    evaluate(np.zeros((1, 3)))
    at_nodes = nodes.numpy()[:n]
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


# ── Point to cell averaging ─────────────────────────────────────────

@pytest.mark.parametrize("make", ["mixed", "rectilinear"])
def test_to_cells_averages_each_cells_points(backend, make):
    data = _grids()[make]
    rng = np.random.default_rng(1)
    scalar = rng.uniform(-1, 1, data.num_points)
    vector = rng.uniform(-1, 1, (data.num_points, 3))
    rows = _cell_points(data)
    out = alg.project(data, td.Field(td.H1(data), _scalars(scalar)), td.Constant(data))
    assert out.space is td.Constant(data)
    np.testing.assert_allclose(out.values.numpy(), [scalar[r].mean() for r in rows],
                               atol=1e-6)
    out = alg.project(data, td.Field(td.Values(data, "points"), _vectors(vector)), td.Constant(data))
    np.testing.assert_allclose(out.values.numpy(),
                               [vector[r].mean(axis=0) for r in rows], atol=1e-6)


def test_to_cells_of_every_space(backend):
    """A DG field averages its own corners; a cell constant is itself; an
    order-2 field averages its corner values; an order per cell is followed."""
    data = _two_hexes_and_a_pyramid()
    u = _height(data)
    expected = alg.project(data, u, td.Constant(data)).values.numpy()
    np.testing.assert_allclose(alg.project(data, alg.project(data, u, td.L2(data)), td.Constant(data)).values.numpy(),
                               expected, atol=1e-6)
    dg = _with_cell_offsets(data, u, 10.0)
    np.testing.assert_allclose(alg.project(data, dg, td.Constant(data)).values.numpy(),
                               expected + 10.0 * np.arange(3), atol=1e-5)
    constant = td.Field(td.Constant(data), _scalars([1.0, 2.0, 5.0]))
    np.testing.assert_allclose(alg.project(data, constant, td.Constant(data)).values.numpy(), [1, 2, 5])
    orders = np.array([1, 0, 1])
    varying, centers = _variable_height(data, orders)
    np.testing.assert_allclose(alg.project(data, varying, td.Constant(data)).values.numpy(),
                               np.where(orders == 1, expected, centers), atol=1e-5)
    grid = td.rectilinear_grid([0, 1, 3], [0, 2, 3], [0, 1])
    quadratic = td.H1(grid, order=2)
    values = np.zeros(quadratic.size)
    values[:grid.num_points] = _height(grid).values.numpy()       # the points' values
    values[grid.num_points:] = 100.0                                 # the rest: not read
    np.testing.assert_allclose(
        alg.project(grid, td.Field(quadratic, _scalars(values)), td.Constant(grid)).values.numpy(),
        alg.project(grid, _height(grid), td.Constant(grid)).values.numpy(), atol=1e-5)


def test_to_cells_refuses(backend):
    data = _two_hexes_and_a_pyramid()
    with pytest.raises(TypeError, match="floating-point"):
        alg.project(data, td.Field(td.H1(data), tack.arange(data.num_points)), td.Constant(data))
    on_faces = td.Field(td.Values(data, "faces"), _scalars(np.zeros(15)))
    with pytest.raises(TypeError, match="on the points or cells"):
        alg.project(data, on_faces, td.Constant(data))


@needs_vtk
@pytest.mark.parametrize("kind", sorted(SOLID_TYPES))
def test_to_cells_is_vtks(f64_backend, kind):
    grid = _vtk_cells(kind, (2, 2, 2))
    grid.point_data["p"] = np.random.default_rng(5).uniform(-1, 1, grid.number_of_points)
    want = np.asarray(vtk.vtkPointDataToCellData()(grid).cell_data["p"])
    data = vtk_to_dataset(grid, dtype=tack.f64)
    np.testing.assert_allclose(alg.project(data, data.fields["p"], td.Constant(data)).values.numpy(), want,
                               atol=1e-12)


@needs_vtk
def test_vtk_edge_cases(backend):
    """From the outside review (1-b, 1-c): a vtkUnstructuredGrid as VTK makes it
    empty, with no vtkPoints at all, imports as an empty dataset; and an order-2
    field exports its values at the points, one tuple per point of the linear
    grid, so the export imports again."""
    empty = vtk_to_dataset(vtk.vtkUnstructuredGrid())
    assert (empty.num_cells, empty.num_points) == (0, 0)
    data = td.uniform_grid((3, 3, 3))
    space = td.H1(data, order=2)
    nodes = td.project(data, data.geometry, space).values.numpy()
    data.fields["u"] = td.Field(space, _scalars(nodes[:, 0] + 2.0 * nodes[:, 1]))
    grid = dataset_to_vtk(data)
    assert grid.point_data["u"].number_of_tuples == data.num_points
    back = vtk_to_dataset(grid)
    x = back.positions()
    np.testing.assert_allclose(back.fields["u"].values.numpy(), x[:, 0] + 2.0 * x[:, 1],
                               rtol=1e-6)


def test_view_mixins_may_not_share_a_name():
    """Stacked into one view, a name two mixins define would hide one of them."""
    from tack.dataset import shapes, views

    class First:
        @tack.func
        def value(self, c):
            return 1

    class Second:
        @tack.func
        def value(self, c):
            return 2

    class Third(First):
        pass

    with pytest.raises(TypeError, match="First and Second both define 'value'"):
        views.view_class(views._Cells, shapes.Tetra, First, Second)
    # The same definition reached twice is one name, not two.
    assert views.view_class(views._Cells, shapes.Tetra, Third, First)


@needs_vtk
def test_vtk_char_arrays_come_across_as_i8():
    """A vtkCharArray (vtkProbeFilter's valid mask, readers' flags) has no
    numpy-compatible wrapper in VTK 9.7.1, and NumPy reads it as bytes; its
    values are signed 8-bit integers."""
    from vtkmodules.vtkCommonCore import vtkCharArray

    image = vtk.vtkImageData(dimensions=(2, 2, 1))
    mask = vtkCharArray()
    mask.SetNumberOfTuples(4)
    for i, v in enumerate((0, 1, -1, 1)):
        mask.SetTuple1(i, v)
    mask.SetName("mask")
    image.GetPointData().AddArray(mask)      # point_data[...] = refuses a vtkCharArray
    data = vtk_to_dataset(image)
    assert data.fields["mask"].values.dtype == tack.i8
    np.testing.assert_array_equal(data.fields["mask"].values.numpy(), [0, 1, -1, 1])


@needs_vtk
def test_vtk_constant_and_affine_fields_stay_implicit():
    """VTK's implicit arrays in point and cell data are Tack's implicit arrays, and
    back: nothing stored on either side."""
    from vtkmodules.vtkCommonCore import vtkAffineArray, vtkConstantArray

    from tack.arrays import ConstantArray, CountingArray

    image = vtk.vtkImageData(dimensions=(3, 3, 1))
    image.point_data["c"] = vtkConstantArray[np.float32](9, 2.5)
    image.cell_data["r"] = vtkAffineArray[np.int32](4, 3, 10)
    data = vtk_to_dataset(image)
    c, r = data.fields["c"].values, data.fields["r"].values
    assert isinstance(c, ConstantArray) and isinstance(r, CountingArray)
    np.testing.assert_array_equal(np.asarray(c), np.full(9, 2.5, np.float32))
    np.testing.assert_array_equal(np.asarray(r), [10, 13, 16, 19])
    back = dataset_to_vtk(data)
    assert "Constant" in back.point_data["c"].class_name
    assert "Affine" in back.cell_data["r"].class_name
    np.testing.assert_array_equal(np.asarray(back.cell_data["r"]), [10, 13, 16, 19])


# ── Structured faces and edges by arithmetic ────────────────────────

@tack.kernel
def _corners_of(cells, out):
    for c in cells:
        for j in range(cells.NUM_POINTS):
            out[cells.entity_id(c) * cells.NUM_POINTS + j] = cells.point_id(c, j)


def _as_unstructured(topology):
    """A structured topology's cells, the same ids and corners, as unstructured."""
    group = topology.groups()[0]
    n, k = topology.num_cells, group.shape.NUM_POINTS
    corners = tack.zeros((n * k,), tack.i64)
    _corners_of(group.view(), corners)
    return td.UnstructuredTopology(np.full(n, int(group.shape.ID), np.uint8),
                                   np.arange(n + 1) * k, corners.numpy(),
                                   num_points=topology.num_points)


@pytest.mark.parametrize("dims", [(4, 3, 5), (2, 2, 2), (2, 7, 3), (6, 2, 2), (3, 4), (5, 2)])
def test_structured_faces_and_edges_are_the_general_derivations(backend, dims):
    """A structured grid derives its faces and edges by arithmetic, in one pass over
    its cells with no sort, and gets what the general derivation (emit, sort,
    match) gives the same cells as unstructured: every array, id for id."""
    grid = td.StructuredTopology(dims)
    other = _as_unstructured(grid)
    ours, theirs = grid.faces(), other.faces()
    assert ours.num_faces == theirs.num_faces
    for name in ("rows", "kinds", "sides", "side_face", "side_slot", "side_orientation"):
        np.testing.assert_array_equal(getattr(ours, name).numpy(), getattr(theirs, name).numpy(),
                                      err_msg=f"faces.{name}")
    ours, theirs = grid.edges(), other.edges()
    assert ours.num_edges == theirs.num_edges
    for name in ("rows", "side_edge", "side_sign"):
        np.testing.assert_array_equal(getattr(ours, name).numpy(), getattr(theirs, name).numpy(),
                                      err_msg=f"edges.{name}")


# ── A space's nodes as a for_each domain ────────────────────────────

@tack.kernel
def _visit_nodes(nodes, visits, where):
    for i in nodes:
        for j in range(nodes.node_count(i)):
            d = nodes.node_dof(i, j)
            tack.atomic_add(visits, d, 1)
            where[d] = nodes.node_position(i, j)


def _node_meshes():
    curved = td.uniform_grid((3, 3, 3))
    curved.geometry = td.project(curved, curved.geometry, td.H1(curved, order=2))
    return {"hexes": td.uniform_grid((4, 3, 3)), "quads": td.uniform_grid((4, 3)),
            "tets": td.tetrahedralize(td.uniform_grid((3, 3, 3))), "curved": curved}


@pytest.mark.parametrize("mesh", ["hexes", "quads", "tets", "curved"])
def test_for_each_over_a_space_visits_each_node_once(backend, mesh):
    """for_each over a space runs a kernel over its nodes: every value of the space
    is one node's, visited once, at the position the space's nodal interpolant
    puts it (interpolate of the identity gives the same positions)."""
    data = _node_meshes()[mesh]

    @tack.func
    def identity(x):
        return x

    spaces = [td.H1(data), td.H1(data, order=2), td.L2(data), td.L2(data, order=0),
              td.Constant(data), td.Values(data, "points"), td.Values(data, "cells")]
    for space in spaces:
        visits = tack.zeros((space.size,), tack.i32)
        where = tack.zeros((space.size,), tack.vec(3, data.dtype))
        td.for_each(_visit_nodes, data, space, visits, where)
        np.testing.assert_array_equal(visits.numpy(), 1, err_msg=repr(space))
        expected = td.interpolate(data, identity, space, dtype=tack.vec(3, data.dtype))
        np.testing.assert_allclose(where.numpy(), expected.values.numpy(), atol=1e-6,
                                   err_msg=repr(space))


def test_what_a_space_domain_refuses(backend):
    data = td.uniform_grid((3, 3, 3))
    space = td.H1(data, order=2)
    visits = tack.zeros((space.size,), tack.i32)
    with pytest.raises(TypeError, match="takes arrays, not fields"):
        td.for_each(_visit_nodes, data, space, td.Field(space, visits), visits)
    with pytest.raises(TypeError, match="no nodes to visit as such"):
        data.node_views(td.Values(data, "faces"))
    poly = td.as_polyhedra(data)
    assert len(poly.node_views(td.H1(poly))) == 1                  # its points
    with pytest.raises(NotImplementedError, match="needs a reference element"):
        poly.node_views(td.Constant(poly))
    broken = td.uniform_grid((3, 3, 3))
    broken.geometry = td.project(broken, broken.geometry, td.L2(broken))
    with pytest.raises(NotImplementedError, match="no single position"):
        broken.node_views(td.H1(broken))
