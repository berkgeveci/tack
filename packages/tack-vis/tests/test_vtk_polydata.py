"""``vtk_to_dataset`` of a ``vtkPolyData``: VTK's surfaces, lines and points, as the
unstructured grid of their cells -- the shapes of each kind, strips as their
triangles with their cell data, attributes carried as a grid's are."""

import numpy as np
import pytest

import tack
import tack.dataset as td

vtk = pytest.importorskip("vtkmodules.all")
from tack.interop.vtk import vtk_to_dataset

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    yield f64_backend
    tack.set_default_device("cpu")


def _measure(data):
    return float(td.cell_geometry(data).measure.values.numpy().astype(np.float64).sum())


def _vtk_measure(output):
    sizes = vtk.vtkCellSizeFilter()(output).cell_data
    return sum(float(np.asarray(sizes[n]).sum()) for n in ("Length", "Area", "Volume")
               if n in sizes.keys())


def _polydata(points, verts=(), lines=(), polys=(), strips=()):
    out = vtk.vtkPolyData()
    out.points = vtk.vtkPoints(data=np.asarray(points, np.float64))
    for name, cells in (("verts", verts), ("lines", lines), ("polys", polys),
                        ("strips", strips)):
        if cells:
            offsets = np.concatenate([[0], np.cumsum([len(c) for c in cells])])
            setattr(out, name, vtk.vtkCellArray(offsets=offsets,
                                                connectivity=np.concatenate(cells)))
    return out


def test_a_surface_of_triangles_and_quads_is_unstructured():
    sphere = vtk.vtkSphereSource(theta_resolution=16, phi_resolution=12)()
    sphere.cell_data["id"] = np.arange(sphere.number_of_cells, dtype=np.float64)
    data = vtk_to_dataset(sphere, dtype=tack.f64)
    assert isinstance(data.topology, td.UnstructuredTopology)
    assert data.num_cells == sphere.number_of_cells
    np.testing.assert_array_equal(data.fields["id"].values.numpy(),
                                  np.arange(sphere.number_of_cells))
    assert _measure(data) == pytest.approx(_vtk_measure(sphere), rel=1e-12)
    disk = vtk_to_dataset(vtk.vtkDiskSource(circumferential_resolution=7)(), dtype=tack.f64)
    assert set(np.unique(disk.topology.types.numpy())) == {9}         # VTK_QUAD


def test_larger_polygons_make_a_polygonal_topology():
    hexagon = [[np.cos(a), np.sin(a), 0] for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
    surface = _polydata([*hexagon, [0, 0, 1]], polys=[[0, 1, 2, 3, 4, 5], [1, 0, 6]])
    data = vtk_to_dataset(surface, dtype=tack.f64)
    assert isinstance(data.topology, td.PolygonalTopology)
    assert _measure(data) == pytest.approx(_vtk_measure(surface), rel=1e-12)


def test_strips_are_their_triangles_wound_alike_with_their_cell_data():
    sphere = vtk.vtkSphereSource(theta_resolution=16, phi_resolution=12)()
    strips = vtk.vtkStripper()(sphere)
    assert strips.strips.number_of_cells and not strips.polys.number_of_cells
    strips.cell_data["strip"] = np.arange(strips.number_of_cells, dtype=np.float64)
    label = vtk.vtkDataArray.UNITS_LABEL()
    strips.cell_data["strip"].GetInformation().Set(label, "m")
    data = vtk_to_dataset(strips, dtype=tack.f64)
    assert data.num_cells == sphere.number_of_cells
    assert _measure(data) == pytest.approx(_vtk_measure(sphere), rel=1e-12)
    x = data.positions().astype(np.float64)
    rows = data.topology.connectivity.numpy().reshape(-1, 3)
    normals = np.cross(x[rows[:, 1]] - x[rows[:, 0]], x[rows[:, 2]] - x[rows[:, 0]])
    assert (np.einsum("ij,ij->i", normals, x[rows].mean(1)) > 0).all()     # all outward
    ids = data.fields["strip"].values.numpy()
    assert (np.diff(ids) >= 0).all() and len(np.unique(ids)) == strips.number_of_cells
    assert data.fields["strip"].unit == "m"


def test_lines_vertices_and_their_mixtures():
    points = np.array([[i, 0, 0] for i in range(5)], float)
    segments = vtk_to_dataset(_polydata(points, lines=[[0, 1], [1, 2]]), dtype=tack.f64)
    assert set(segments.topology.types.numpy()) == {3}                  # VTK_LINE
    chain = vtk_to_dataset(_polydata(points, lines=[[0, 1, 2, 3], [3, 4]]), dtype=tack.f64)
    assert isinstance(chain.topology, td.PolylineTopology) and chain.num_cells == 2
    assert _measure(chain) == pytest.approx(4.0)
    cloud = vtk_to_dataset(_polydata(points, verts=[[0], [2], [4]]), dtype=tack.f64)
    assert set(cloud.topology.types.numpy()) == {1}                     # VTK_VERTEX
    both = vtk_to_dataset(_polydata(points, verts=[[4]], polys=[[0, 1, 2]]))
    assert sorted(both.topology.types.numpy().tolist()) == [1, 5]       # VTK's order
    with pytest.raises(ValueError, match="polylines"):
        vtk_to_dataset(_polydata(points, lines=[[0, 1, 2]], polys=[[0, 1, 3]]))


def test_point_data_ghosts_and_time_come_across():
    sphere = vtk.vtkSphereSource()()
    x = np.asarray(sphere.points.data)
    sphere.point_data["z"] = x[:, 2].copy()
    marks = np.zeros(sphere.number_of_cells, np.uint8)
    marks[:3] = 2                                           # HIDDENCELL
    sphere.cell_data["vtkGhostType"] = marks
    sphere.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(), 2.5)
    data = vtk_to_dataset(sphere, dtype=tack.f64)
    np.testing.assert_allclose(data.fields["z"].values.numpy(), x[:, 2])
    np.testing.assert_array_equal(data.ghosts.cells.numpy(), marks)
    assert data.time == 2.5
    assert td.contour(data, "z", 0.1).num_cells > 0                    # lines on the surface


def test_an_empty_polydata():
    data = vtk_to_dataset(vtk.vtkPolyData())
    assert (data.num_cells, data.num_points) == (0, 0)
