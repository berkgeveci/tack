"""``feature_edges``: a surface's boundary, feature, non-manifold and manifold edges,
against VTK's ``vtkFeatureEdges`` (compared by their ends' positions, since VTK
renumbers points)."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkFiltersCore import vtkFeatureEdges
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    from tack.interop.vtk import dataset_to_vtk
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtkFeatureEdges = None

needs_vtk = pytest.mark.skipif(vtkFeatureEdges is None, reason="needs VTK, the reference")

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    """The surfaces are in f64, so on every backend that has f64 -- the CPU, and
    CUDA, HIP or Level Zero where present -- whatever the run's default device.
    Metal has none; a device without f64 is checked by test_dataset_devices."""
    yield f64_backend
    tack.set_default_device("cpu")

KINDS = {"boundary": td.EdgeKind.BOUNDARY, "feature": td.EdgeKind.FEATURE,
         "non_manifold": td.EdgeKind.NON_MANIFOLD, "manifold": td.EdgeKind.MANIFOLD}


def _ours(surface, kind, angle):
    flags = {name: name == kind for name in KINDS}
    out = td.feature_edges(surface, angle=angle, **flags)
    lines = tack.arrays.materialize(out.topology.connectivity).numpy().reshape(-1, 2)
    assert (out.fields["kind"].values.numpy() == KINDS[kind]).all()
    return {tuple(sorted(r)) for r in lines.tolist()}


def _pairs(x, lines, points):
    """Each edge as the ids of the input points nearest its ends, ordered: a set."""
    nearest = np.array([np.argmin(((points - p) ** 2).sum(1)) for p in x], int)
    return {tuple(sorted(r)) for r in nearest[lines].tolist()}


def _vtk(surface, kind, angle):
    polys = vtkGeometryFilter()
    polys.SetInputData(dataset_to_vtk(surface))
    edges = vtkFeatureEdges()
    edges.SetInputConnection(polys.GetOutputPort())
    edges.SetBoundaryEdges(kind == "boundary")
    edges.SetFeatureEdges(kind == "feature")
    edges.SetNonManifoldEdges(kind == "non_manifold")
    edges.SetManifoldEdges(kind == "manifold")
    edges.SetFeatureAngle(angle)
    edges.ColoringOff()
    edges.Update()
    out = edges.GetOutput()
    x = np.asarray(out.points.data, np.float64) if out.points is not None else np.zeros((0, 3))
    lines = np.asarray(out.lines.connectivity_array).reshape(-1, 2) if out.number_of_lines \
        else np.zeros((0, 2), int)
    return _pairs(x, lines, surface.world_positions())


def _bumpy_closed_surface():
    """A closed triangle surface with corners of many angles."""
    rng = np.random.default_rng(4)
    grid = td.tetrahedralize(td.uniform_grid((4, 4, 4), dtype=tack.f64))
    x = grid.positions() + rng.normal(0.0, 0.12, (grid.num_points, 3))
    moved = td.DataSet(grid.topology, x, dtype=tack.f64)
    return td.extract_surface(moved, td.boundary_faces(moved))


def _open_surface():
    """A warped grid of quads (as an unstructured mesh) with a boundary."""
    grid = td.uniform_grid((6, 5), spacing=(1.0, 1.0), dtype=tack.f64)
    x = grid.positions().copy()
    x[:, 2] = 0.6 * np.sin(x[:, 0]) * np.cos(x[:, 1])
    return td.DataSet(grid.topology, x, dtype=tack.f64)


def _book():
    """Three quads sharing one edge, and a fourth alone: non-manifold."""
    x = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [-1, 0, 0], [-1, 0, 1],
                  [0, 1, 0], [0, 1, 1], [5, 0, 0], [6, 0, 0], [6, 1, 0], [5, 1, 0]], float)
    quads = [[0, 2, 3, 1], [0, 1, 5, 4], [0, 6, 7, 1], [8, 9, 10, 11]]
    topology = td.UnstructuredTopology(np.full(4, 9, np.uint8), np.arange(0, 17, 4),
                                       np.array(quads).reshape(-1), num_points=12)
    return td.DataSet(topology, x, dtype=tack.f64)


@needs_vtk
@pytest.mark.parametrize("make", [_bumpy_closed_surface, _open_surface, _book])
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("angle", [30.0, 60.0])
def test_feature_edges_are_vtks(make, kind, angle):
    surface = make()
    assert _ours(surface, kind, angle) == _vtk(surface, kind, angle)


def test_kinds_counts_and_fields():
    cube = td.uniform_grid((3, 3, 3))
    surface = td.extract_surface(cube, td.boundary_faces(cube))
    out = td.feature_edges(surface)
    assert out.num_cells == 24 and (out.fields["kind"].values.numpy() == 2).all()
    plane = td.uniform_grid((3, 3))
    plane.fields["h"] = td.Field(td.H1(plane), tack.array(np.arange(9, dtype=np.float32)), "m")
    out = td.feature_edges(plane, manifold=True)
    assert np.bincount(out.fields["kind"].values.numpy()).tolist() == [0, 8, 0, 0, 4]
    assert out.num_points == plane.num_points and out.fields["h"].unit == "m"
    lines = tack.arrays.materialize(out.topology.connectivity).numpy().reshape(-1, 2)
    assert (lines[:, 0] < lines[:, 1]).all()
    assert [tuple(r) for r in lines] == sorted(tuple(r) for r in lines)   # by point ids
    nothing = td.feature_edges(plane, boundary=False, feature=False, non_manifold=False)
    assert nothing.num_cells == 0


def test_polygons_and_a_cell_wound_the_wrong_way():
    plane = td.uniform_grid((4, 4))
    polygons = td.as_polygons(plane)
    assert td.feature_edges(polygons).num_cells == 12              # the boundary
    t = td.tetrahedralize(td.uniform_grid((3, 3, 2)))
    surface = td.extract_surface(t, td.boundary_faces(t))
    conn = tack.arrays.materialize(surface.topology.connectivity).numpy().copy()
    conn[[0, 1]] = conn[[1, 0]]                                     # flip one triangle
    flipped = td.DataSet(td.UnstructuredTopology(
        tack.arrays.materialize(surface.topology.types), surface.topology.offsets,
        tack.array(conn), num_points=surface.num_points), surface.geometry.values)
    before = td.feature_edges(surface).num_cells
    assert td.feature_edges(flipped).num_cells > before            # its edges turn ~180


def test_refusals():
    with pytest.raises(ValueError, match="extract_surface first"):
        td.feature_edges(td.uniform_grid((3, 3, 3)))
    with pytest.raises(ValueError, match="0 to 180"):
        td.feature_edges(td.uniform_grid((3, 3)), angle=200.0)
