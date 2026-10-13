"""Polylines (``PolylineTopology``): chains of points, the polyhedral path two
dimensions down. Their faces are their ends; their geometry, extraction,
averaging, boundary and VTK round trip; streamlines are tested with advection.
And contour lines of 2D meshes, shape-based and polygonal."""

import numpy as np
import pytest
from test_dataset_api import _scalars

import tack
import tack.data as td
from tack.data.algorithms.cell_geometry import _line_geometry

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    vtk = None

# Two chains meeting end to end at point 3, a closed loop, and a lone segment.
LINES = [[0, 1, 2, 3], [3, 4, 5], [6, 7, 8, 6], [9, 10]]
X = np.array([[i, (i % 3) * 0.5, 0.0] for i in range(11)], np.float32)


def _lines(lines=LINES, x=X):
    offsets = np.cumsum([0] + [len(c) for c in lines])
    return td.DataSet(td.PolylineTopology(offsets, np.concatenate(lines)), x)


def _chain_lengths(lines=LINES, x=X):
    return [np.linalg.norm(np.diff(x[c], axis=0), axis=1).sum() for c in lines]


def test_construction_and_points(backend):
    data = _lines()
    t = data.topology
    assert (data.num_cells, data.num_points, t.dimension) == (4, 11, 1)
    offsets, points = (a.to_numpy() for a in t.lines())
    np.testing.assert_array_equal(points, np.concatenate(LINES))
    offsets, points = (a.to_numpy() for a in t.cell_points())
    np.testing.assert_array_equal(offsets, [0, 4, 7, 10, 12])      # the loop's 6 once
    np.testing.assert_array_equal(points, [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError, match="fewer than two points"):
        td.PolylineTopology([0, 2, 3], [0, 1, 2])


def test_faces_are_the_ends(backend):
    data = _lines()
    faces = data.topology.faces()
    np.testing.assert_array_equal(data.topology.face_points.to_numpy(), [0, 3, 5, 6, 9, 10])
    # Chains 0 and 1 share point 3; the loop's two ends are one interior face.
    assert faces.sides.to_numpy(vectors=True).tolist() == [
        [0, 0, -1, -1], [0, 1, 1, 0], [1, 1, -1, -1], [2, 0, 2, 1], [3, 0, -1, -1],
        [3, 1, -1, -1]]
    np.testing.assert_array_equal(td.boundary_faces(data).to_numpy(), [0, 2, 4, 5])
    ends = td.external_faces(data)
    assert set(ends.topology.types.to_numpy()) == {1}                  # VTK_VERTEX
    np.testing.assert_array_equal(ends.topology.connectivity.to_numpy(), [0, 5, 9, 10])


def test_branches_have_no_faces(backend):
    # Three chains ending at point 0: the topology works, its faces do not.
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], np.float32)
    data = _lines([[1, 0], [2, 0], [3, 0]], x)
    np.testing.assert_allclose(td.cell_geometry(data).measure.values.to_numpy(),
                               [1, 1, np.sqrt(2)], rtol=1e-6)
    with pytest.raises(ValueError, match="more than two polylines end at 1 points"):
        data.topology.faces()


def test_geometry(backend):
    data = _lines()
    lengths, centroids = td.cell_geometry(data)
    np.testing.assert_allclose(lengths.values.to_numpy(), _chain_lengths(), rtol=1e-6)
    want = []
    for c in LINES:
        a, b = X[c[:-1]], X[c[1:]]
        w = np.linalg.norm(b - a, axis=1)
        want.append((w[:, None] * (a + b) / 2).sum(0) / w.sum())
    np.testing.assert_allclose(centroids.values.to_numpy(vectors=True), want, rtol=1e-5)
    normals, areas = td.face_geometry(data)
    np.testing.assert_array_equal(areas.values.to_numpy(), 1)
    n = normals.values.to_numpy(vectors=True)
    # Out of side 0, along its chain: the first chain's start points backwards.
    np.testing.assert_allclose(n[0], (X[0] - X[1]) / np.linalg.norm(X[0] - X[1]), rtol=1e-6)
    np.testing.assert_allclose(n[1], (X[3] - X[2]) / np.linalg.norm(X[3] - X[2]), rtol=1e-6)
    np.testing.assert_allclose(td.face_centers(data).values.to_numpy(vectors=True),
                               X[[0, 3, 5, 6, 9, 10]])


def test_a_straight_flow_has_no_divergence(backend):
    x = np.array([[k, 0, 0] for k in range(6)], np.float32)
    data = _lines([[0, 1, 2], [2, 3], [3, 4, 5]], x)
    normals, areas = td.face_geometry(data)
    flux = normals.values.to_numpy(vectors=True) @ np.array([2.0, 0, 0])
    flux_field = td.Field(td.Values(data, "faces", oriented=True), _scalars(flux))
    divergence = td.divergence(data, flux_field).values.to_numpy()
    # Interior cells pass the flow on; nothing leaves the chain but at its ends.
    np.testing.assert_allclose(divergence, 0, atol=1e-6)


def test_averages_and_extraction(backend):
    data = _lines()
    data.fields["s"] = td.Field(td.H1(data), _scalars(X[:, 0]))
    data.fields["c"] = td.Field(td.Constant(data), _scalars([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_allclose(td.to_cells(data, "s").values.to_numpy(), [1.5, 4, 7, 9.5])
    point_values = td.to_points(data, "c").values.to_numpy()
    assert point_values[3] == pytest.approx(15.0)                     # shared by 0 and 1
    kept = td.threshold(data, "c", 25, 45)
    assert isinstance(kept.topology, td.PolylineTopology)
    np.testing.assert_array_equal(kept.topology.lines()[1].to_numpy(), [0, 1, 2, 0, 3, 4])
    np.testing.assert_array_equal(kept.fields["c"].values.to_numpy(), [30, 40])
    np.testing.assert_array_equal(kept.fields["s"].values.to_numpy(), [6, 7, 8, 9, 10])


class _Halves:
    """A launch key: cells split into two subgroups by parity."""

    def __init__(self, topology):
        self.topology = topology
        n = topology.num_cells
        self.keys = tack.field(tack.i32, shape=(n,))
        self.keys.from_numpy((np.arange(n) % 2).astype(np.int32))

    def cell_keys(self):
        return self.keys


def test_subgroups_select_their_cells(backend):
    data = _lines()
    groups = data.launch_groups("cells", [], keys=[_Halves(data.topology)])
    assert [g.count for g in groups] == [2, 2]
    lengths = tack.field(tack.f32, shape=(data.num_cells,))
    centroids = tack.Vector.field(3, tack.f32, shape=(data.num_cells,))
    for group in groups:
        _line_geometry(data.domain_view("cells", group), lengths, centroids)
    np.testing.assert_allclose(lengths.to_numpy(), _chain_lengths(), rtol=1e-6)


@pytest.mark.skipif(vtk is None, reason="needs VTK")
def test_vtk_round_trip(backend):
    data = _lines()
    data.fields["c"] = td.Field(td.Constant(data), _scalars([10.0, 20.0, 30.0, 40.0]))
    grid = dataset_to_vtk(data)
    assert {grid.GetCellType(c) for c in range(grid.GetNumberOfCells())} == {vtk.VTK_POLY_LINE}
    back = vtk_to_dataset(grid)
    assert isinstance(back.topology, td.PolylineTopology)
    for a, b in zip(back.topology.lines(), data.topology.lines()):
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    np.testing.assert_array_equal(back.fields["c"].values.to_numpy(), [10, 20, 30, 40])
    # A line among polylines is a two-point polyline; a triangle is refused.
    grid.GetCellData().RemoveArray("c")
    grid.InsertNextCell(vtk.VTK_LINE, 2, [9, 10])
    assert vtk_to_dataset(grid).num_cells == 5
    grid.InsertNextCell(vtk.VTK_TRIANGLE, 3, [0, 1, 2])
    with pytest.raises(ValueError, match="only lines and polylines"):
        vtk_to_dataset(grid)


def test_what_polylines_refuse(backend):
    data = _lines()
    data.fields["s"] = td.Field(td.H1(data), _scalars(X[:, 0]))
    with pytest.raises(NotImplementedError, match="contour of lines"):
        td.contour(data, "s", 1.5)
    with pytest.raises(NotImplementedError, match="not 1D"):
        td.triangulate(data)
    with pytest.raises(NotImplementedError, match="edges"):
        data.topology.edges()


# ── Contour lines ───────────────────────────────────────────────────

def _circle_meshes():
    xs = np.linspace(-2, 2, 21).astype(np.float32)
    grid = td.rectilinear_grid(xs, xs)
    meshes = {"quads": grid, "polygons": td.as_polygons(grid), "triangles": td.triangulate(grid)}
    for data in meshes.values():
        x = data.positions()
        data.fields["r2"] = td.Field(td.H1(data), _scalars(x[:, 0] ** 2 + x[:, 1] ** 2))
        data.fields["cell"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    return meshes


@pytest.mark.parametrize("mesh", ["quads", "polygons", "triangles"])
def test_contour_lines_of_a_2d_mesh(backend, mesh):
    data = _circle_meshes()[mesh]
    out = td.contour(data, "r2", 1.1)       # off the grid points: no zero-length lines
    assert set(out.topology.types.to_numpy()) == {3}                  # VTK_LINE
    # A closed curve: every point is shared by two lines.
    assert out.num_points == out.num_cells
    np.testing.assert_array_equal(
        np.bincount(out.topology.connectivity.to_numpy(), minlength=out.num_points), 2)
    x = out.positions()
    radius = np.linalg.norm(x[:, :2], axis=1)
    np.testing.assert_allclose(radius, np.sqrt(1.1), atol=0.02)
    np.testing.assert_allclose(out.fields["r2"].values.to_numpy(), 1.1, atol=1e-5)
    # Each line is in the cell it came from, and runs with the inside on its left
    # (the higher values on its right) as the cells wind counterclockwise.
    rows = out.topology.connectivity.to_numpy().reshape(-1, 2)
    a, b = x[rows[:, 0]], x[rows[:, 1]]
    turning = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    assert (turning > 0).all()
    _, centroids = td.cell_geometry(data)
    cells = out.fields["cell"].values.to_numpy().astype(int)
    source = centroids.values.to_numpy(vectors=True)[cells]
    mid = 0.5 * (x[rows[:, 0]] + x[rows[:, 1]])
    assert (np.abs(source - mid).max(1) < 0.2 + 1e-6).all()


def test_contour_lines_agree_across_paths(backend):
    meshes = _circle_meshes()
    a = td.contour(meshes["quads"], "r2", 1.1).positions()
    b = td.contour(meshes["polygons"], "r2", 1.1).positions()
    np.testing.assert_allclose(np.sort(a, axis=0), np.sort(b, axis=0), atol=1e-6)


def test_a_plane_slices_a_2d_mesh_into_a_line(backend):
    data = _circle_meshes()["quads"]
    cut = td.slice_plane(data, (0.1, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert cut.num_cells == 20 and cut.num_points == 21
    x = cut.positions()
    np.testing.assert_allclose(x[:, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose([x[:, 1].min(), x[:, 1].max()], [-2, 2], atol=1e-6)


def test_cell_geometry_of_lines_and_vertices(backend):
    """A shape-based line's length and midpoint, and a vertex's point: neither was
    written, only polylines and 2D and 3D cells."""
    data = td.DataSet(td.UnstructuredTopology([3, 1], [0, 2, 3], [0, 1, 1]),
                      [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    out = td.cell_geometry(data)
    np.testing.assert_allclose(out.measure.values.to_numpy(), [5.0, 0.0])
    np.testing.assert_allclose(out.centroid.values.to_numpy(vectors=True),
                               [[1.5, 2.0, 0.0], [3.0, 4.0, 0.0]])
