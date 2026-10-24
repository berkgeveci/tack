"""CellLocator and probe, against VTK's vtkProbeFilter.

vtkProbeFilter is the reference for which points lie in the mesh and, on
hexahedra, tetrahedra and pyramids, for the values there. On skewed wedges it
is not: vtkWedge::EvaluatePosition reports a distance of ~1e-31 for points
outside the wedge (r + s > 1), and the probe's cell search takes that for a
hit in the wrong neighbour, while the same EvaluatePosition puts the point
inside the cell Tack finds. There, and everywhere, a linear field -- which any
cell holding the point reproduces exactly -- checks the values instead.
"""

import numpy as np
import pytest
from test_dataset_api import _scalars, _two_hexes_and_a_pyramid, _vectors

import tack
import tack.dataset as td

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

SHAPES = {"hexahedron": 12, "tetra": 10, "wedge": 13, "pyramid": 14, "voxel": 11}


def _grid(kind):
    grid = vtk.vtkCellTypeSource(cell_type=SHAPES[kind], blocks_dimensions=(3, 3, 2))()
    x = np.asarray(grid.points.data).astype(np.float64)
    if kind != "voxel":                       # a voxel is axis-aligned by definition
        x = x + 0.12 * np.c_[np.sin(2 * x[:, 1]), np.cos(x[:, 2]), np.sin(x[:, 0])]
        grid.points = vtk.vtkPoints(data=vtk.vtkFloatArray(x.astype(np.float32)))
    x = np.asarray(grid.points.data).astype(np.float64)
    for name, values in (("s", np.sin(x[:, 0]) * x[:, 1] + x[:, 2] ** 2),
                         ("lin", 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2])):
        grid.point_data[name] = values.astype(np.float32)
    return grid


def _vtk_probe(grid, points, name="s"):
    # vtkPoints' default storage, f32, as InsertNextPoint filled it.
    cloud = vtk.vtkPolyData(
        points=vtk.vtkPoints(data=vtk.vtkFloatArray(np.asarray(points, np.float32))))
    point_data = vtk.vtkProbeFilter(source_data=grid)(cloud).point_data
    # The mask is a vtkCharArray, which NumPy reads as bytes (S1): view it as the int8
    # numpy_support gave.
    return (np.asarray(point_data["vtkValidPointMask"]).view(np.int8),
            np.asarray(point_data[name]))


@needs_vtk
@pytest.mark.parametrize("kind", SHAPES)
def test_probe_is_vtks(backend, kind):
    grid = _grid(kind)
    data = vtk_to_dataset(grid)
    points = np.random.default_rng(3).uniform(-0.4, 3.4, (500, 3))
    out = td.probe(data, points)
    valid, expected = _vtk_probe(grid, points)
    np.testing.assert_array_equal(out.fields["valid"].values.numpy(), valid)
    inside = valid == 1
    assert 50 < inside.sum() < 500
    lin = 2 * points[:, 0] - points[:, 1] + 0.5 * points[:, 2]
    np.testing.assert_allclose(out.fields["lin"].values.numpy()[inside], lin[inside],
                               atol=2e-4)
    assert (out.fields["s"].values.numpy()[~inside] == 0).all()
    if kind != "wedge":
        np.testing.assert_allclose(out.fields["s"].values.numpy()[inside],
                                   expected[inside], atol=1e-3)


@needs_vtk
def test_probe_of_a_structured_grid_is_vtks(backend):
    wavelet = vtk.vtkRTAnalyticSource(whole_extent=(-6, 6, -6, 6, -6, 6))()
    data = td.sources.wavelet(6)
    points = np.random.default_rng(5).uniform(-7, 7, (800, 3))
    out = td.probe(data, points)
    valid, expected = _vtk_probe(wavelet, points, "RTData")
    np.testing.assert_array_equal(out.fields["valid"].values.numpy(), valid)
    inside = valid == 1
    np.testing.assert_allclose(out.fields["RTData"].values.numpy()[inside],
                               expected[inside], rtol=2e-4)


@needs_vtk
def test_locator_finds_the_cell_and_its_coordinates(backend):
    grid = _grid("hexahedron")
    data = vtk_to_dataset(grid)
    points = np.random.default_rng(4).uniform(0.2, 2.8, (100, 3)) * [1, 1, 0.6]
    cells, pcs, tets = td.CellLocator(data).find(points)
    assert tets is None
    cells, pcs = cells.numpy(), pcs.numpy()
    for q in range(0, 100, 9):
        cell = grid.GetCell(int(cells[q]))
        pc, weights = [0.0] * 3, [0.0] * 8
        inside = cell.EvaluatePosition(points[q], [0.0] * 3, vtk.reference(0), pc,
                                       vtk.reference(0.0), weights)
        assert inside == 1
        np.testing.assert_allclose(pcs[q], pc, atol=1e-4)


def test_probe_keeps_the_probed_dataset_and_evaluates_every_kind(backend):
    data = _two_hexes_and_a_pyramid()
    x = data.positions()
    data.fields["h"] = td.Field(td.H1(data), _scalars(3 * x[:, 0] + x[:, 2]))
    data.fields["c"] = td.Field(td.Constant(data), _scalars([1.0, 2.0, 3.0]))
    data.fields["dg"] = td.algorithms.project(data, data.fields["h"], td.L2(data))
    where = td.sources.tangle((4, 3, 3))
    out = td.probe(data, where)
    assert out.topology is where.topology and "tangle" in out.fields
    p = where.positions()
    valid = out.fields["valid"].values.numpy() == 1
    np.testing.assert_allclose(out.fields["h"].values.numpy()[valid],
                               (3 * p[:, 0] + p[:, 2])[valid], atol=1e-5)
    np.testing.assert_allclose(out.fields["dg"].values.numpy()[valid],
                               (3 * p[:, 0] + p[:, 2])[valid], atol=1e-5)
    assert set(np.unique(out.fields["c"].values.numpy()[valid])) <= {1.0, 2.0, 3.0}
    assert out.fields["h"].space is td.H1(out)


def _two_tets():
    """Two tetrahedra sharing the face x + y + z = 1, and a field that is 0 on the
    first and rises to 1 at the second's far corner: a kink at the face."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    topology = td.UnstructuredTopology(np.array([10, 10], np.uint8), [0, 4, 8],
                                       [0, 1, 2, 3, 1, 2, 3, 4])
    data = td.DataSet(topology, pts)
    p = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    data.fields["p"] = td.Field(td.H1(data), _scalars(p))
    data.fields["v"] = td.Field(td.H1(data), _vectors(np.c_[p, np.zeros(5), np.zeros(5)]))
    return data


# Points a little inside the second tetrahedron: within the tolerance (0.1% of
# the first's box diagonal, 1.7e-3) of the first, which has the lower id.
_DEPTH = 5e-4
_NEAR_THE_FACE = (np.array([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.25, 0.25], [0.2, 0.5, 0.3]])
                  + _DEPTH * np.ones(3) / np.sqrt(3))


def test_a_point_in_a_cell_is_found_there_not_in_a_neighbour_within_tolerance(backend):
    """The cell that holds the point must win over a lower-numbered one that only
    comes within the tolerance: there the field would be extrapolated across the
    kink (0 here, where the second gives the true value)."""
    data = _two_tets()
    cells = td.CellLocator(data).find(_NEAR_THE_FACE).cells
    np.testing.assert_array_equal(cells.numpy(), [1, 1, 1])
    out = td.probe(data, _NEAR_THE_FACE)
    # The far corner is 2 / sqrt(3) from the face, so p there is the depth over that.
    np.testing.assert_allclose(out.fields["p"].values.numpy(), _DEPTH * np.sqrt(3) / 2,
                               atol=1e-6)


def test_advection_samples_the_cell_that_holds_the_particle(backend):
    """Advection locates as the locator does: the velocity is zero in the first
    tetrahedron and not in the second, so a particle there must move."""
    data = _two_tets()
    out = td.advect(data, "v", _NEAR_THE_FACE, step_size=0.1, steps=1)
    status = out.fields["status"].values.numpy()
    assert not (status & td.ParticleStatus.ZERO_VELOCITY).any(), status
    assert (out.positions()[:, 0] > _NEAR_THE_FACE[:, 0]).all()


def test_polyhedra_rank_the_cell_that_holds_the_point_first(backend):
    """The same two tetrahedra as polyhedra: located in sub-tetrahedra, the point
    is only near the first (rank past the ids) and in the second, which wins, for
    probe and for advection alike."""
    data = td.as_polyhedra(_two_tets())
    cells, _, tets = td.CellLocator(data).find(_NEAR_THE_FACE)
    np.testing.assert_array_equal(cells.numpy(), [1, 1, 1])
    assert (tets.numpy() >= 0).all()
    out = td.probe(data, _NEAR_THE_FACE)
    np.testing.assert_allclose(out.fields["p"].values.numpy(), _DEPTH * np.sqrt(3) / 2,
                               atol=1e-6)
    moved = td.advect(data, "v", _NEAR_THE_FACE, step_size=0.1, steps=1)
    assert not (moved.fields["status"].values.numpy() & td.ParticleStatus.ZERO_VELOCITY).any()
    assert (moved.positions()[:, 0] > _NEAR_THE_FACE[:, 0]).all()


def test_points_outside_every_cell(backend):
    data = _two_hexes_and_a_pyramid()
    out = td.probe(data, [[10.0, 10.0, 10.0], [-1.0, 0.5, 0.5]])
    np.testing.assert_array_equal(out.fields["valid"].values.numpy(), [0, 0])
    ring = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 2], [3, 2], [3, 3], [0, 3]], float)
    notch = td.DataSet(td.PolygonalTopology([0, 8], list(range(8))),
                       np.c_[ring, np.zeros(8)].astype(np.float32))
    with pytest.raises(NotImplementedError, match="1 of the 1 cells are not star-shaped"):
        td.CellLocator(notch)                    # a C: its mean point is in the notch


def test_probe_keeps_target_topology_ids_and_quadratic_geometry(backend):
    source = td.uniform_grid((2, 2, 2))
    source.fields["x"] = td.Field(td.H1(source), _scalars(source.positions()[:, 0]))
    top = td.StructuredTopology((2, 2, 2), id_dtype=tack.i64)
    target = td.DataSet(top, tack.arrays.UniformCoordinates((2, 2, 2)))
    target.geometry = td.project(target, target.geometry, td.H1(target, order=2))
    out = td.probe(source, target)
    assert out.topology is target.topology
    assert out.geometry is target.geometry
    assert out.id_dtype is tack.i64
    np.testing.assert_allclose(out.fields["x"].values.numpy(), target.positions()[:, 0])
    np.testing.assert_array_equal(out.fields["valid"].values.numpy(), 1)


def test_probe_refuses_a_locator_for_another_geometry(backend):
    source = td.uniform_grid((2, 2, 2))
    moved = td.transform(source, [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]])
    with pytest.raises(ValueError, match="another topology or geometry"):
        td.probe(source, [[0.5, 0.5, 0.5]], locator=td.CellLocator(moved))


@tack.kernel
def _shift_x(positions, dx):
    for i in range(positions.shape[0]):
        p = positions[i]
        positions[i] = [p[0] + dx, p[1], p[2]]


def test_probe_reuses_the_locator_its_topology_keeps_until_the_geometry_changes(backend):
    """Without a locator, probe uses the one the topology keeps for its geometry:
    built once, and taken by any dataset of that topology and geometry. Moving
    the positions in place -- a kernel, or copyto -- makes it stale: probe
    builds a new one, and refuses a stale one passed in."""
    grid = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25))
    positions = tack.array(grid.positions().astype(np.float32), tack.vec3(tack.f32))
    data = td.DataSet(td.tetrahedralize(grid).topology, positions)
    data.fields["x"] = td.Field(td.H1(data), tack.array(grid.positions()[:, 0].astype(np.float32)))
    query = [[0.3, 0.4, 0.5]]
    td.probe(data, query)
    kept = data.topology._locator
    assert kept is not None
    td.probe(data, query)
    assert data.topology._locator is kept                       # built once
    again = td.DataSet(data.topology, data.geometry)              # same cells and points
    again.fields["x"] = data.fields["x"]
    td.probe(again, query, locator=kept)
    assert kept.serves(again)

    _shift_x(positions, 10.0)                                     # moved in place
    assert not kept.serves(data)
    with pytest.raises(ValueError, match="written the positions since"):
        td.probe(data, query, locator=kept)
    out = td.probe(data, [[10.3, 0.4, 0.5], [0.3, 0.4, 0.5]])
    assert data.topology._locator is not kept                     # rebuilt
    np.testing.assert_array_equal(out.fields["valid"].values.numpy(), [1, 0])
    np.testing.assert_allclose(out.fields["x"].values.numpy()[0], 0.3, atol=1e-5)

    tack.copyto(positions, grid.positions().astype(np.float32))   # moved back, by copyto
    out = td.probe(data, query)
    np.testing.assert_allclose(out.fields["x"].values.numpy(), [0.3], atol=1e-5)
    # Another geometry on the same cells takes the topology's one locator over.
    moved = td.transform(data, [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0]])
    out = td.probe(moved, [[5.3, 0.4, 0.5]])
    np.testing.assert_allclose(out.fields["x"].values.numpy(), [0.3], atol=1e-5)
    assert data.topology._locator.serves(moved)


def test_probe_preserves_f64_point_coordinates(f64_backend):
    source = td.uniform_grid((2, 2, 2), dtype=tack.f64)
    points = np.array([[0.5 + 1e-9, 0.5, 0.5]], np.float64)
    out = td.probe(source, points)
    assert out.dtype is tack.f64
    np.testing.assert_array_equal(out.positions(), points)


# ── Polyhedra: located in their sub-tetrahedra ──────────────────────

def _linear(x):
    return 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2]


@needs_vtk
@pytest.mark.parametrize("kind", SHAPES)
def test_probe_of_polyhedra_is_the_shape_paths(backend, kind):
    """The shape meshes as polyhedra hold the same points -- all of them where the
    faces are planar (tetrahedra, voxels), all but a few near the bilinear faces
    a fan of triangles replaces -- and reproduce a linear field exactly; cell
    data is the holding cell's."""
    data = vtk_to_dataset(_grid(kind))
    poly = td.as_polyhedra(data)
    poly.fields["c"] = td.Field(td.Constant(poly), _scalars(np.arange(poly.num_cells) + 1.0))
    points = np.random.default_rng(3).uniform(-0.4, 3.4, (500, 3))
    ours, theirs = td.probe(poly, points), td.probe(data, points)
    valid = ours.fields["valid"].values.numpy() == 1
    same = valid == (theirs.fields["valid"].values.numpy() == 1)
    assert same.all() if kind in ("tetra", "voxel") else same.mean() > 0.98
    assert 50 < valid.sum() < 500
    np.testing.assert_allclose(ours.fields["lin"].values.numpy()[valid],
                               _linear(points)[valid], atol=2e-4)
    assert (ours.fields["s"].values.numpy()[~valid] == 0).all()
    cells, pcs, tets = td.CellLocator(poly).find(points)
    cells, tets = cells.numpy(), tets.numpy()
    np.testing.assert_array_equal(cells >= 0, valid)
    np.testing.assert_array_equal(tets >= 0, valid)
    np.testing.assert_array_equal(ours.fields["c"].values.numpy()[valid], cells[valid] + 1.0)


@needs_vtk
def test_probe_of_polyhedra_at_their_points(backend):
    """At the mesh's own points -- on faces, edges and corners of several cells, and
    on the boundary -- every point is found and takes its own value."""
    data = vtk_to_dataset(_grid("hexahedron"))
    poly = td.as_polyhedra(data)
    x = poly.positions().astype(np.float64)
    out = td.probe(poly, x)
    np.testing.assert_array_equal(out.fields["valid"].values.numpy(), 1)
    np.testing.assert_allclose(out.fields["s"].values.numpy(),
                               poly.fields["s"].values.numpy(), atol=1e-5)


def test_probe_of_voronoi_polyhedra_holds_the_points_inside(backend):
    """Voronoi cells are convex, so a point is in one exactly when it is in the convex
    hull of its points: the probe finds those points, and no others, and a linear
    field there exactly.

    Not vtkProbeFilter, on VTK 9.7.1: on these cells it accepts 55 of 800 points
    lying outside every cell (up to 3.8% of the mesh's diagonal), and for 83 of
    the rest its linear field is off by up to 0.18, because
    vtkPolyhedron::EvaluatePosition reports points outside a convex cell as
    inside (its own IsInside disagrees)."""
    from scipy.spatial import Delaunay
    from test_polyhedra import _from_cell_faces, _voronoi_columns

    points, cells, _ = _voronoi_columns(thickness=0.4)
    data = td.DataSet(_from_cell_faces(cells, len(points)), points)
    x = data.positions().astype(np.float64)
    data.fields["lin"] = td.Field(td.H1(data), _scalars(_linear(x)))
    lo, hi = x.min(axis=0), x.max(axis=0)
    where = lo + np.random.default_rng(6).uniform(-0.05, 1.05, (800, 3)) * (hi - lo)
    out = td.probe(data, where)
    offsets, ids = (a.numpy() for a in data.topology.cell_points())
    hulls = [Delaunay(x[ids[offsets[c]:offsets[c + 1]]]) for c in range(data.num_cells)]
    inside = np.array([any(h.find_simplex(q) >= 0 for h in hulls) for q in where])
    np.testing.assert_array_equal(out.fields["valid"].values.numpy(), inside)
    assert 100 < inside.sum() < 800
    np.testing.assert_allclose(out.fields["lin"].values.numpy()[inside],
                               _linear(where)[inside], atol=2e-5)


# ── From the points and from the cells ──────────────────────────────

def _both_ways(data, points):
    """``find`` and ``probe`` from the points (by_id's search over the cell bins)
    and from the cells (each group's view), on fresh locators."""
    by_points = td.CellLocator(data)
    by_points._search()                                # build the search: from the points
    by_points._found_before = True
    by_cells = td.CellLocator(data)
    by_cells._mesh_search = None                       # no search: from the cells
    return [(loc.find(points), td.probe(data, points, locator=loc))
            for loc in (by_points, by_cells)]


def _polyline():
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 1], [0, 2, 0], [1, 3, 1]],
                      np.float32)
    return td.DataSet(td.PolylineTopology([0, 4, 6], [0, 1, 2, 3, 4, 5]), points)


def _meshes():
    meshes = {"mixed": _two_hexes_and_a_pyramid(),
              "polygons": td.as_polygons(td.uniform_grid((5, 4))),
              "polyline": _polyline(),
              "structured": td.wavelet(4),
              "two tets as polyhedra": td.as_polyhedra(_two_tets())}
    if vtk is not None:
        meshes.update({kind: vtk_to_dataset(_grid(kind)) for kind in SHAPES})
        meshes["wedges as polyhedra"] = td.as_polyhedra(vtk_to_dataset(_grid("wedge")))
    return meshes


@pytest.mark.parametrize("name", ["mixed", "structured", "two tets as polyhedra", "polygons",
                                  "polyline", "hexahedron", "tetra", "wedge", "pyramid", "voxel",
                                  "wedges as polyhedra"])
def test_searching_from_the_points_finds_what_searching_from_the_cells_does(backend, name):
    """The two searches rank candidates alike -- holding the point first, then the
    smaller id -- so they find the same cells, coordinates and sub-tetrahedra,
    and probe the same values, points outside the mesh included."""
    meshes = _meshes()
    if name not in meshes:
        pytest.skip("needs VTK for this mesh")
    data = meshes[name]
    x = data.positions().astype(np.float64)
    data.fields["lin"] = td.Field(td.H1(data), _scalars(_linear(x)))
    data.fields["ids"] = td.Field(td.Constant(data),
                                  _scalars(np.arange(data.num_cells, dtype=float)))
    lo, hi = x.min(axis=0), x.max(axis=0)
    points = np.r_[lo + np.random.default_rng(8).uniform(-0.1, 1.1, (400, 3)) * (hi - lo), x]
    (a, pa), (b, pb) = _both_ways(data, points)
    np.testing.assert_array_equal(a.cells.numpy(), b.cells.numpy())
    inside = a.cells.numpy() >= 0
    assert inside.any() and not inside.all()
    np.testing.assert_allclose(a.pcs.numpy()[inside], b.pcs.numpy()[inside], atol=1e-5)
    if a.tets is not None:
        np.testing.assert_array_equal(a.tets.numpy(), b.tets.numpy())
    for f in ("lin", "ids", "valid"):
        np.testing.assert_allclose(pa.fields[f].values.numpy(), pb.fields[f].values.numpy(),
                                   atol=1e-5)


@needs_vtk
@pytest.mark.parametrize("kind", SHAPES)
def test_a_point_shared_by_several_cells_goes_to_the_smallest_id(backend, kind):
    """A mesh point lies on the boundary of every cell that has it: each holds it,
    and the smallest id wins, from both searches. Held without a margin, which
    cells held it was the sign of rounding noise in their parametric
    coordinates; on CUDA the two searches, the same functions inlined into two
    kernels that NVRTC fused differently, disagreed at 5 of the tetra grid's
    mesh points."""
    data = vtk_to_dataset(_grid(kind))
    offsets = data.topology.offsets.numpy().reshape(-1)
    connectivity = data.topology.connectivity.numpy().reshape(-1)
    smallest = np.full(data.num_points, data.num_cells)
    for c in range(data.num_cells):
        ids = connectivity[offsets[c]:offsets[c + 1]]
        smallest[ids] = np.minimum(smallest[ids], c)
    points = np.arange(data.num_points)        # pyramids' apexes too
    x = data.positions().astype(np.float64)[points]
    for found, _ in _both_ways(data, x):
        np.testing.assert_array_equal(found.cells.numpy(), smallest[points])


@needs_vtk
def test_a_point_at_a_pyramids_apex_is_found(backend):
    """The pyramid grid's apexes, one per block shared by its six pyramids, were
    found by no cell (Newton cannot converge where the pyramid's map is
    singular). Both searches find them now, and a linear field there is exact."""
    data = vtk_to_dataset(_grid("pyramid"))
    offsets = data.topology.offsets.numpy().reshape(-1)
    connectivity = data.topology.connectivity.numpy().reshape(-1)
    apexes = sorted({int(connectivity[offsets[c] + 4]) for c in range(data.num_cells)})
    x = data.positions().astype(np.float64)[apexes]
    assert len(apexes) == 18
    for found, probed in _both_ways(data, x):
        cells = found.cells.numpy()
        assert (cells >= 0).all(), cells
        for c, p in zip(cells, apexes):
            assert p in connectivity[offsets[c]:offsets[c + 1]]
        np.testing.assert_allclose(probed.fields["lin"].values.numpy(), _linear(x), atol=2e-4)


def test_a_first_query_of_few_points_runs_from_the_cells():
    """The search from the points needs the cell bins, about eight entries a cell:
    a fresh locator's first query of few points runs from the cells, and builds
    them only from its second query on, or for many points."""
    tack.set_default_device("cpu")
    data = td.wavelet(6)
    locator = td.CellLocator(data)
    locator.find([[0.0, 0.0, 0.0]])
    assert "_cell_bins" not in locator.__dict__
    locator.find([[0.0, 0.0, 0.0]])
    assert "_cell_bins" in locator.__dict__
    many = td.CellLocator(data)
    many.find(np.zeros((data.num_cells, 3)))
    assert "_cell_bins" in many.__dict__


def test_what_by_id_cannot_serve_is_searched_from_the_cells():
    """A curved (order-2) geometry and line cells have no search from the points:
    find and probe run from the cells, as before."""
    tack.set_default_device("cpu")
    curved = td.uniform_grid((3, 3, 3))
    curved.geometry = td.project(curved, curved.geometry, td.H1(curved, order=2))
    lines = td.DataSet(td.UnstructuredTopology(np.array([3, 3], np.uint8), [0, 2, 4],
                                               [0, 1, 1, 2]),
                       np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32))
    for data, point in ((curved, [0.6, 0.6, 0.6]), (lines, [1.5, 0.0, 0.0])):
        locator = td.CellLocator(data)
        locator._found_before = True
        assert locator._search() is None
        found = locator.find([point])
        assert found.cells.numpy()[0] >= 0


# ── Polygons and polylines ──────────────────────────────────────────

def test_polygons_hold_what_their_quads_do(backend):
    """A quad grid as polygons, each fanned into triangles from its points' mean:
    the same points found (near the plane too) and a linear field exact."""
    quads = td.uniform_grid((5, 4))
    x = quads.positions().astype(np.float64)
    quads.fields["lin"] = td.Field(td.H1(quads), _scalars(_linear(x)))
    polygons = td.as_polygons(quads)
    rng = np.random.default_rng(9)
    points = np.c_[rng.uniform(-0.2, 4.2, (300, 2)), rng.uniform(-1e-4, 1e-4, 300)]
    ours, theirs = td.probe(polygons, points), td.probe(quads, points)
    valid = ours.fields["valid"].values.numpy() == 1
    np.testing.assert_array_equal(valid, theirs.fields["valid"].values.numpy() == 1)
    assert 100 < valid.sum() < 300
    projected = np.c_[points[:, :2], np.zeros(len(points))]
    np.testing.assert_allclose(ours.fields["lin"].values.numpy()[valid],
                               _linear(projected)[valid], atol=1e-5)


def test_a_polyline_is_linear_along_its_segments(backend):
    """A point is located on the nearest segment within the tolerance, held when its
    projection falls on it, and point data is linear along the segment."""
    data = _polyline()
    data.fields["u"] = td.Field(td.H1(data), _scalars(np.array([0, 1, 3, 6, 10, 20.0])))
    points = [[0.5, 0, 0], [1.5, 0.5, 0], [2.5, 1, 0.5], [1, 0, 0], [0.5, 2.5, 0.5], [5, 5, 5]]
    out = td.probe(data, points)
    np.testing.assert_array_equal(out.fields["valid"].values.numpy(), [1, 1, 1, 1, 1, 0])
    np.testing.assert_allclose(out.fields["u"].values.numpy(), [0.5, 2, 4.5, 1, 15, 0],
                               atol=1e-5)
    cells, pcs, tets = td.CellLocator(data).find(points)
    np.testing.assert_array_equal(cells.numpy(), [0, 0, 0, 0, 1, -1])
    np.testing.assert_array_equal(tets.numpy()[[0, 1, 2, 4]], [0, 1, 2, 0])

