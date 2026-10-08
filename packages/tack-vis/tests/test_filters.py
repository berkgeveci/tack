"""Tests for tack.data.filters.

Each filter is checked against NumPy on every backend -- averages, links,
face counts, boundary faces facing outward, cell data carried along -- and,
where VTK is installed, against VTK's own filter: vtkCellCenters,
vtkPointDataToCellData, vtkCellDataToPointData and vtkGeometryFilter.
"""

import types

import numpy as np
import pytest
from test_shapes import EXPECTED

import tack
import tack.data as td
from tack.data import shapes as sh
from tack.runtime.dispatch import env_flag

try:
    from vtkmodules import vtkCommonCore, vtkCommonDataModel, vtkFiltersCore
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(**{name: getattr(module, name)
                                   for module in (vtkCommonCore, vtkCommonDataModel,
                                                  vtkFiltersCore)
                                   for name in dir(module) if name.startswith("vtk")})
    vtk.vtkGeometryFilter = vtkGeometryFilter

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

SOLIDS = [sh.Tetra, sh.Voxel, sh.Hexahedron, sh.Wedge, sh.Pyramid]


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


def _mixed(rng, num_cells, num_points, shapes=sh.SHAPES):
    """A mixed explicit cell set whose cells share points (no point twice in a cell)."""
    kinds = rng.choice(shapes, num_cells)
    rows = [rng.choice(num_points, cls.NUM_POINTS, replace=False) for cls in kinds]
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    types_ = np.array([cls.ID for cls in kinds], np.uint8)
    return types_, offsets, np.concatenate(rows), rows


def _hex_grid(n):
    """Points and x-fastest hexahedra of a unit grid with n points per side."""
    pts = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64)] * 3, indexing="ij"),
                   -1)[..., ::-1].reshape(-1, 3)
    rows = []
    for k in range(n - 1):
        for j in range(n - 1):
            for i in range(n - 1):
                b = i + j * n + k * n * n
                rows.append([b, b + 1, b + 1 + n, b + n,
                             b + n * n, b + 1 + n * n, b + 1 + n + n * n, b + n + n * n])
    return pts, np.array(rows)


# ── Cell centers ────────────────────────────────────────────────────

def test_cell_centers_of_a_grid(backend):
    pts, rows = _hex_grid(4)
    data = td.DataSet(pts, td.SingleTypeCellSet(sh.Hexahedron, rows),
                      cell_data={"id": _scalars(np.arange(27))})
    out = td.cell_centers(data)
    assert out.num_points == out.num_cells == 27
    np.testing.assert_allclose(out.points.to_numpy(vectors=True),
                               pts[rows].mean(axis=1), atol=1e-6)
    np.testing.assert_array_equal(out.point_data["id"].to_numpy(), np.arange(27))
    assert out.cells.shapes() == [sh.Vertex]


def test_cell_centers_of_a_structured_grid(backend):
    pts, rows = _hex_grid(5)
    out = td.cell_centers(td.DataSet(pts, td.StructuredCellSet((5, 5, 5))))
    np.testing.assert_allclose(out.points.to_numpy(vectors=True), pts[rows].mean(axis=1),
                               atol=1e-6)


# ── Point data to cell data ─────────────────────────────────────────

def test_point_data_to_cell_data_averages(backend):
    rng = np.random.default_rng(1)
    types_, offsets, connectivity, rows = _mixed(rng, 300, 200)
    scalar = rng.uniform(-1, 1, 200)
    vector = rng.uniform(-1, 1, (200, 3))
    data = td.DataSet(rng.uniform(size=(200, 3)), td.ExplicitCellSet(types_, offsets, connectivity),
                      point_data={"s": _scalars(scalar), "v": _vectors(vector)})
    out = td.point_data_to_cell_data(data)
    np.testing.assert_allclose(out.cell_data["s"].to_numpy(),
                               [scalar[r].mean() for r in rows], atol=1e-6)
    np.testing.assert_allclose(out.cell_data["v"].to_numpy(vectors=True),
                               [vector[r].mean(axis=0) for r in rows], atol=1e-6)
    assert set(out.point_data) == {"s", "v"}


def test_averaging_needs_floating_point(backend):
    data = td.DataSet(np.zeros((8, 3)), td.SingleTypeCellSet(sh.Hexahedron, [list(range(8))]),
                      point_data={"labels": tack.zeros(tack.i32, (8,))})
    with pytest.raises(TypeError, match="'labels' is i32"):
        td.point_data_to_cell_data(data)


# ── Links and cell data to point data ───────────────────────────────

def test_point_links_group_each_points_cells(backend):
    rng = np.random.default_rng(2)
    types_, offsets, connectivity, rows = _mixed(rng, 120, 90)
    data = td.DataSet(np.zeros((90, 3)), td.ExplicitCellSet(types_, offsets, connectivity))
    points, cells, run_offsets, count = td.point_links(data)
    used = sorted(set(connectivity.tolist()))
    assert count == len(used) and points.to_numpy().tolist() == used
    cells, run_offsets = cells.to_numpy(), run_offsets.to_numpy()
    for r, p in enumerate(used):
        want = sorted(c for c, row in enumerate(rows) if p in row)
        assert sorted(cells[run_offsets[r]:run_offsets[r + 1]].tolist()) == want
    assert td.point_links(data) is td.point_links(data)


def test_cell_data_to_point_data_averages(backend):
    rng = np.random.default_rng(3)
    types_, offsets, connectivity, rows = _mixed(rng, 250, 220)
    scalar = rng.uniform(-1, 1, 250)
    vector = rng.uniform(-1, 1, (250, 2))
    data = td.DataSet(np.zeros((220, 3)), td.ExplicitCellSet(types_, offsets, connectivity),
                      cell_data={"s": _scalars(scalar), "v": _vectors(vector)})
    out = td.cell_data_to_point_data(data)
    want_s, want_v = np.zeros(220), np.zeros((220, 2))
    for p in range(220):
        users = [c for c, row in enumerate(rows) if p in row]
        if users:
            want_s[p] = scalar[users].mean()
            want_v[p] = vector[users].mean(axis=0)
    np.testing.assert_allclose(out.point_data["s"].to_numpy(), want_s, atol=1e-6)
    np.testing.assert_allclose(out.point_data["v"].to_numpy(vectors=True), want_v, atol=1e-6)
    again = td.cell_data_to_point_data(data).point_data["s"].to_numpy()
    assert np.array_equal(again, out.point_data["s"].to_numpy())   # bit for bit


def test_cell_data_to_point_data_on_a_structured_grid(backend):
    pts, rows = _hex_grid(4)
    values = np.arange(27, dtype=np.float32)
    data = td.DataSet(pts, td.StructuredCellSet((4, 4, 4)), cell_data={"c": _scalars(values)})
    out = td.cell_data_to_point_data(data).point_data["c"].to_numpy()
    want = [values[[c for c in range(27) if p in rows[c]]].mean() for p in range(64)]
    np.testing.assert_allclose(out, want, atol=1e-6)


# ── External faces ──────────────────────────────────────────────────

def _faces_of(surface):
    """The surface's faces as point-id lists, and its cell types."""
    offsets = surface.cells.offsets.to_numpy()
    conn = surface.cells.connectivity.to_numpy()
    return ([conn[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)],
            surface.cells.types.to_numpy())


def _normal(points):
    """Newell's normal of a polygon."""
    n = np.zeros(3)
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        n += np.cross(a, b)
    return n


def _assert_outward(faces, owners, pts, cell_rows):
    for face, owner in zip(faces, owners):
        outward = pts[face].mean(axis=0) - pts[cell_rows[owner]].mean(axis=0)
        assert _normal(pts[face]) @ outward > 0, (face, owner)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_external_faces_of_a_hexahedral_grid(backend, n):
    pts, rows = _hex_grid(n)
    for cells in (td.SingleTypeCellSet(sh.Hexahedron, rows), td.StructuredCellSet((n, n, n))):
        data = td.DataSet(pts, cells, cell_data={"id": _scalars(np.arange(len(rows)))})
        surface = td.external_faces(data)
        faces, kinds = _faces_of(surface)
        assert len(faces) == 6 * (n - 1) ** 2 and set(kinds.tolist()) == {sh.QUAD}
        for face in faces:                     # every face lies on the cube's boundary
            on = pts[face]
            assert any(np.allclose(on[:, a], 0) or np.allclose(on[:, a], n - 1) for a in range(3))
        owners = surface.cell_data["id"].to_numpy().astype(int)
        _assert_outward(faces, owners, pts, rows)


@pytest.mark.parametrize("shape", SOLIDS, ids=lambda cls: cls.__name__)
def test_a_single_cell_is_all_external(backend, shape):
    ref = np.array(EXPECTED[shape]["points"], float) * [1.0, 2.0, 3.0]
    data = td.DataSet(ref, td.SingleTypeCellSet(shape, [list(range(len(ref)))]))
    faces, kinds = _faces_of(td.external_faces(data))
    want = EXPECTED[shape]["faces"]
    assert len(faces) == len(want)
    assert sorted(sorted(f) for f in faces) == sorted(sorted(f) for f in want)
    assert set(kinds.tolist()) <= {sh.TRIANGLE, sh.QUAD}
    _assert_outward(faces, [0] * len(faces), ref, [list(range(len(ref)))])


def test_shared_faces_are_internal(backend):
    # Two tetrahedra on one triangle, and a hexahedron with a wedge on top.
    pts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)], float)
    tets = td.DataSet(pts, td.SingleTypeCellSet(sh.Tetra, [[0, 1, 2, 3], [1, 2, 3, 4]]))
    assert td.external_faces(tets).num_cells == 6
    # A hexahedron cut along the diagonal plane through 0 2 6 4 into two
    # wedges, which share that quad.
    pts = np.array(EXPECTED[sh.Hexahedron]["points"], float)
    data = td.DataSet(pts, td.SingleTypeCellSet(sh.Wedge, [[0, 1, 2, 4, 5, 6],
                                                           [0, 2, 3, 4, 6, 7]]))
    faces, kinds = _faces_of(td.external_faces(data))
    assert sorted(kinds.tolist()) == [sh.TRIANGLE] * 4 + [sh.QUAD] * 4
    assert sorted(sorted(f) for f in faces if len(f) == 4) == [
        [0, 1, 4, 5], [0, 3, 4, 7], [1, 2, 5, 6], [2, 3, 6, 7]]
    # Where the faces do not match -- a quad against two triangles -- both are external.
    hex_pts = np.array(EXPECTED[sh.Hexahedron]["points"], float)
    top = np.array([(0, 0, 2), (1, 0, 2), (1, 1, 2), (0, 1, 2)], float)
    types_ = np.array([sh.HEXAHEDRON, sh.WEDGE, sh.WEDGE], np.uint8)
    connectivity = [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 8, 9, 10, 4, 6, 7, 8, 10, 11]
    data = td.DataSet(np.vstack([hex_pts, top]),
                      td.ExplicitCellSet(types_, [0, 8, 14, 20], connectivity))
    assert td.external_faces(data).num_cells == 6 + 2 + 2 + 4


def test_lower_dimensional_cells_have_no_faces(backend):
    data = td.DataSet(np.zeros((4, 3)), td.SingleTypeCellSet(sh.Quad, [[0, 1, 2, 3]]))
    surface = td.external_faces(data)
    assert surface.num_cells == 0 and surface.cells.offsets.to_numpy().tolist() == [0]


# ── VTK ─────────────────────────────────────────────────────────────

def _to_vtk(data):
    from tack.interop.vtk import dataset_to_vtk
    return dataset_to_vtk(data)


def _run(filter_, grid):
    filter_.SetInputData(grid)
    filter_.Update()
    return filter_.GetOutput()


def _mixed_dataset(rng, num_cells=300, num_points=250, shapes=sh.SHAPES):
    """Cells of every shape over a shared, random point cloud, with data."""
    types_, offsets, connectivity, _ = _mixed(rng, num_cells, num_points, shapes)
    return td.DataSet(rng.uniform(-1, 1, (num_points, 3)),
                      td.ExplicitCellSet(types_, offsets, connectivity),
                      point_data={"p": _scalars(rng.uniform(size=num_points), tack.f64)},
                      cell_data={"c": _scalars(rng.uniform(size=num_cells), tack.f64)},
                      dtype=tack.f64)


@needs_vtk
def test_cell_centers_are_vtks(f64_backend):
    # Random points: no pixels or voxels, which VTK evaluates as axis-aligned.
    others = [s for s in sh.SHAPES if s not in (sh.Pixel, sh.Voxel)]
    data = _mixed_dataset(np.random.default_rng(4), shapes=others)
    want = vtk_to_numpy(_run(vtk.vtkCellCenters(), _to_vtk(data)).GetPoints().GetData())
    # shape_function returns its literals through branches: a local assigned
    # more than once, only literals, is f32, so the weights are f32 on f64 points.
    np.testing.assert_allclose(td.cell_centers(data).points.to_numpy(vectors=True), want,
                               atol=1e-6)


@needs_vtk
def test_point_data_to_cell_data_is_vtks(f64_backend):
    data = _mixed_dataset(np.random.default_rng(5))
    want = vtk_to_numpy(_run(vtk.vtkPointDataToCellData(), _to_vtk(data))
                        .GetCellData().GetArray("p"))
    np.testing.assert_allclose(td.point_data_to_cell_data(data).cell_data["p"].to_numpy(),
                               want, atol=1e-12)


@needs_vtk
def test_cell_data_to_point_data_is_vtks(f64_backend):
    data = _mixed_dataset(np.random.default_rng(6))
    want = vtk_to_numpy(_run(vtk.vtkCellDataToPointData(), _to_vtk(data))
                        .GetPointData().GetArray("c"))
    np.testing.assert_allclose(td.cell_data_to_point_data(data).point_data["c"].to_numpy(),
                               want, atol=1e-12)


def _face_set(points, faces):
    """Faces by coordinates, each as its cyclic point order from its smallest point."""
    out = set()
    for face in faces:
        coords = [tuple(np.round(points[p], 9)) for p in face]
        start = coords.index(min(coords))
        out.add(tuple(coords[start:] + coords[:start]))
    return out


def _vtk_faces(poly):
    pts = vtk_to_numpy(poly.GetPoints().GetData())
    polys = poly.GetPolys()
    offsets = vtk_to_numpy(polys.GetOffsetsArray())
    conn = vtk_to_numpy(polys.GetConnectivityArray())
    return pts, [conn[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]


def _delaunay_tets(rng, n=60):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(rng.uniform(size=(n, 3)), deep=1))
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(points)
    return _run(vtk.vtkDelaunay3D(), cloud)


def _wedge_columns(n=4):
    """A hexahedral grid with every other column split into two wedges: conforming."""
    pts, rows = _hex_grid(n)
    types_, conn = [], []
    for c, row in enumerate(rows):
        i, j = c % (n - 1), (c // (n - 1)) % (n - 1)
        if (i + j) % 2:
            a, b, cc, d, e, f, g, h = row
            for wedge in ([a, b, cc, e, f, g], [a, cc, d, e, g, h]):
                types_.append(sh.WEDGE)
                conn.append(wedge)
        else:
            types_.append(sh.HEXAHEDRON)
            conn.append(list(row))
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in conn])])
    return td.DataSet(pts, td.ExplicitCellSet(np.array(types_, np.uint8), offsets,
                                              np.concatenate(conn)), dtype=tack.f64)


@needs_vtk
@pytest.mark.parametrize("mesh", ["delaunay", "hexahedra", "hexahedra and wedges"])
def test_external_faces_are_vtks(f64_backend, mesh):
    from tack.interop.vtk import vtk_to_dataset
    if mesh == "delaunay":
        data = vtk_to_dataset(_delaunay_tets(np.random.default_rng(7)), dtype=tack.f64)
    elif mesh == "hexahedra":
        pts, rows = _hex_grid(4)
        data = td.DataSet(pts, td.SingleTypeCellSet(sh.Hexahedron, rows), dtype=tack.f64)
    else:
        data = _wedge_columns()
    faces, _ = _faces_of(td.external_faces(data))
    pts = data.points.to_numpy(vectors=True)
    want_pts, want_faces = _vtk_faces(_run(vtk.vtkGeometryFilter(), _to_vtk(data)))
    assert len(faces) == len(want_faces)
    assert _face_set(pts, faces) == _face_set(want_pts, want_faces)
