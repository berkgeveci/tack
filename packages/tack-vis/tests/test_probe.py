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
    cells, pcs = td.CellLocator(data).find(points)
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
    cells, _ = td.CellLocator(data).find(_NEAR_THE_FACE)
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


def test_points_outside_every_cell(backend):
    data = _two_hexes_and_a_pyramid()
    out = td.probe(data, [[10.0, 10.0, 10.0], [-1.0, 0.5, 0.5]])
    np.testing.assert_array_equal(out.fields["valid"].values.numpy(), [0, 0])
    with pytest.raises(NotImplementedError, match="polygons or polylines"):
        td.CellLocator(td.as_polygons(td.uniform_grid((3, 3, 1))))


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
    with pytest.raises(ValueError, match="another dataset"):
        td.probe(source, [[0.5, 0.5, 0.5]], locator=td.CellLocator(moved))


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
    cells, tets, pcs = td.CellLocator(poly).find(points)
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
    the rest its linear field is off by up to 0.18 -- vtkPolyhedron's
    InterpolateFunctions does not map the parametric coordinates
    EvaluatePosition returns back to the point."""
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
