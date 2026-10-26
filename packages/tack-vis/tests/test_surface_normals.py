"""``surface_normals``: a surface's polygons wound alike and its normals at points and
cells, held to VTK's vtkPolyDataNormals (without splitting)."""

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


def _sphere():
    sphere = vtk.vtkSphereSource(theta_resolution=24, phi_resolution=16)()
    # VTK's normals of a surface that already has a Normals array differ from
    # those of the same surface without one; the source's are dropped.
    sphere.point_data.RemoveArray("Normals")
    return sphere


def _scrambled(surface, fraction=0.4):
    """``surface`` with some polygons turned round (never the first)."""
    offsets = np.asarray(surface.polys.offsets_array)
    connectivity = np.asarray(surface.polys.connectivity_array).copy()
    turned = np.random.default_rng(0).random(len(offsets) - 1) < fraction
    turned[0] = False
    for c in np.flatnonzero(turned):
        a, b = offsets[c], offsets[c + 1]
        connectivity[a + 1:b] = connectivity[a + 1:b][::-1]
    out = vtk.vtkPolyData()
    out.points = surface.points
    out.polys = vtk.vtkCellArray(offsets=offsets, connectivity=connectivity)
    return out


def _wavy_quads():
    plane = vtk.vtkPlaneSource(x_resolution=6, y_resolution=5)()
    x = np.asarray(plane.points.data).copy()
    x[:, 2] = 0.2 * np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1])
    out = vtk.vtkPolyData()
    out.points = vtk.vtkPoints(data=x)
    out.polys = plane.polys
    return out


def _polygons():
    disk = vtk.vtkRegularPolygonSource(number_of_sides=7, generate_polygon=True,
                                       generate_polyline=False)()
    ring = vtk.vtkDiskSource(circumferential_resolution=7, radial_resolution=1,
                             inner_radius=1.0, outer_radius=2.0)()
    both = vtk.vtkAppendPolyData()
    both.AddInputData(disk)
    both.AddInputData(ring)
    both.Update()
    return vtk.vtkCleanPolyData()(both.GetOutput())


SURFACES = {"sphere": _sphere, "scrambled sphere": lambda: _scrambled(_sphere()),
            "wavy quads": _wavy_quads, "polygons": _polygons}


@pytest.mark.parametrize("name", SURFACES)
def test_normals_are_vtks(name):
    surface = SURFACES[name]()
    theirs = vtk.vtkPolyDataNormals(splitting=False, compute_cell_normals=True,
                                    auto_orient_normals=False)(surface)
    data = vtk_to_dataset(surface, dtype=tack.f64)
    ours = td.surface_normals(data)
    np.testing.assert_allclose(ours.fields["normals"].values.numpy(),
                               np.asarray(theirs.point_data["Normals"]), atol=1e-6)
    np.testing.assert_allclose(ours.fields["cell_normals"].values.numpy(),
                               np.asarray(theirs.cell_data["Normals"]), atol=1e-6)
    assert "normals" not in data.fields                       # the input is left as it was
    assert ours.fields["normals"].unit == "dimensionless"


def test_consistency_turns_the_cells_that_wind_against_their_region():
    """A sphere with 40% of its triangles turned, and a second sphere apart: each
    region winds as its first cell does, and a polygonal topology (which refuses
    an inconsistent winding) takes the result."""
    one = _sphere()
    two = vtk.vtkSphereSource(center=(3, 0, 0), theta_resolution=10, phi_resolution=8)()
    two.point_data.RemoveArray("Normals")
    both = vtk.vtkAppendPolyData()
    both.AddInputData(_scrambled(one))
    both.AddInputData(_scrambled(two))
    both.Update()
    data = vtk_to_dataset(both.GetOutput(), dtype=tack.f64)
    with pytest.raises(ValueError, match="walked the same way"):
        td.as_polygons(data)
    ours = td.surface_normals(data)
    polygons = td.as_polygons(ours)                            # consistent now
    assert polygons.num_cells == data.num_cells
    x = ours.positions().astype(np.float64)
    centers = np.where(x[:, :1] > 1.5, [3.0, 0, 0], [0.0, 0, 0])
    outward = np.einsum("ij,ij->i", ours.fields["normals"].values.numpy(), x - centers)
    assert (outward > 0).all() or (outward[x[:, 0] < 1.5] > 0).all()
    unturned = td.surface_normals(data, consistent=False)
    assert unturned.topology is not ours.topology


def test_refusals():
    strip = vtk.vtkPolyData()                                  # a Moebius strip of quads
    n = 12
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = []
    for t in (-0.3, 0.3):
        twist = angles / 2
        rows.append(np.c_[(1 + t * np.cos(twist)) * np.cos(angles),
                          (1 + t * np.cos(twist)) * np.sin(angles), t * np.sin(twist)])
    strip.points = vtk.vtkPoints(data=np.vstack(rows))
    quads = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n - 1)]
    quads.append([n - 1, n, 0, 2 * n - 1])                    # the half twist closes it
    strip.polys = vtk.vtkCellArray(offsets=np.arange(0, 4 * n + 1, 4),
                                   connectivity=np.ravel(quads))
    with pytest.raises(ValueError, match="not orientable"):
        td.surface_normals(vtk_to_dataset(strip, dtype=tack.f64))
    with pytest.raises(ValueError, match="2D cells"):
        td.surface_normals(td.uniform_grid((3, 3, 3)))
