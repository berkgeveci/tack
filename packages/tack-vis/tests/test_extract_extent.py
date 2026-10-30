"""``extract_extent``: a box of a structured grid's points, every ``rate``-th, as a
structured grid of the same kind, held to VTK's own extractors (vtkExtractVOI for
images, vtkExtractRectilinearGrid, vtkExtractGrid) point for point and cell for
cell."""

import numpy as np
import pytest

import tack
import tack.dataset as td

vtk = pytest.importorskip("vtkmodules.all")
from tack.interop.vtk import vtk_to_dataset

pytestmark = pytest.mark.quick

EXTENT = (-10, 10, -8, 8, -6, 6)


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    yield f64_backend
    tack.set_default_device("cpu")


def _grids():
    image = vtk.vtkRTAnalyticSource(whole_extent=EXTENT)()
    image.cell_data["cid"] = np.arange(image.number_of_cells, dtype=np.float64)
    curvilinear = vtk.vtkImageDataToPointSet()(image)
    x = np.asarray(curvilinear.points.data)
    curvilinear.points = vtk.vtkPoints(data=x + 0.1 * np.sin(x[:, [1, 2, 0]]))
    rectilinear = vtk.vtkRectilinearGrid(dimensions=(21, 17, 13))
    for name, n in (("x_coordinates", 21), ("y_coordinates", 17), ("z_coordinates", 13)):
        setattr(rectilinear, name, vtk.vtkDoubleArray(np.cumsum(np.linspace(0.5, 1.5, n))))
    rectilinear.extent = EXTENT
    rectilinear.point_data["RTData"] = np.asarray(image.point_data["RTData"]).copy()
    rectilinear.cell_data["cid"] = np.arange(rectilinear.number_of_cells, dtype=np.float64)
    return {"image": (image, vtk.vtkExtractVOI),
            "curvilinear": (curvilinear, vtk.vtkExtractGrid),
            "rectilinear": (rectilinear, vtk.vtkExtractRectilinearGrid)}


KINDS = {"image": "UniformCoordinates", "curvilinear": "Array",
         "rectilinear": "CartesianProduct"}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("voi, rate", [
    ((-5, 7, -8, 3, -2, 4), 1),            # a box
    ((-10, 10, -8, 8, 0, 0), 1),           # a k-plane: a 2D grid of faces
    ((-10, 10, -8, 8, 6, 6), 1),           # the far side's: cells from the layer before
    ((-9, 9, -7, 7, -5, 5), 3),            # every third point
    ((-3, 3, 2, 2, -6, 6), 2),             # a j-plane, every other point
])
def test_extents_are_vtks(kind, voi, rate):
    grid, extractor = _grids()[kind]
    theirs = extractor(voi=voi, sample_rate=(rate,) * 3)(grid)
    ours = td.extract_extent(vtk_to_dataset(grid, dtype=tack.f64), voi, rate=rate)
    assert type(ours.geometry.values).__name__ == KINDS[kind]           # its kind kept
    assert isinstance(ours.topology, td.StructuredTopology)
    if rate == 1:                                                 # VTK's extent, kept axes
        assert ours.topology.offset == tuple(lo for lo, hi in zip(voi[0::2], voi[1::2])
                                             if hi > lo)
    np.testing.assert_allclose(ours.positions(),
                               [theirs.GetPoint(i) for i in range(theirs.number_of_points)],
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(ours.fields["RTData"].values.numpy(),
                                  np.asarray(theirs.point_data["RTData"]))
    assert ours.num_cells == theirs.number_of_cells
    np.testing.assert_array_equal(ours.fields["cid"].values.numpy(),
                                  np.asarray(theirs.cell_data["cid"]))


@pytest.mark.parametrize("kind", ["image", "rectilinear"])
def test_a_flat_grid_takes_vtks_three_axes(kind):
    """A flat image or rectilinear grid (here in the xz plane) is a 2D grid in
    Tack; VTK still gives its extent and rate in three axes."""
    voi, rate = (1, 7, 0, 0, 2, 6), (2, 1, 2)
    if kind == "image":
        grid = vtk.vtkImageData(dimensions=(9, 1, 7), spacing=(0.5, 1.0, 2.0),
                                origin=(1.0, 2.0, 3.0))
    else:
        grid = vtk.vtkRectilinearGrid(dimensions=(9, 1, 7))
        grid.x_coordinates = vtk.vtkDoubleArray(np.cumsum(np.linspace(0.5, 1.5, 9)))
        grid.y_coordinates = vtk.vtkDoubleArray(np.array([2.0]))
        grid.z_coordinates = vtk.vtkDoubleArray(np.cumsum(np.linspace(1.0, 3.0, 7)))
    grid.point_data["v"] = np.arange(grid.number_of_points, dtype=np.float64)
    extractor = vtk.vtkExtractVOI if kind == "image" else vtk.vtkExtractRectilinearGrid
    theirs = extractor(voi=voi, sample_rate=rate)(grid)
    ours = td.extract_extent(vtk_to_dataset(grid, dtype=tack.f64), voi, rate)
    np.testing.assert_allclose(ours.positions(),
                               [theirs.GetPoint(i) for i in range(theirs.number_of_points)],
                               rtol=1e-12)
    np.testing.assert_array_equal(ours.fields["v"].values.numpy(),
                                  np.asarray(theirs.point_data["v"]))


def test_a_box_is_clipped_to_the_grid_and_keeps_its_marks():
    data = td.uniform_grid((5, 4, 3), dtype=tack.f64)
    marks = np.zeros(data.num_cells, np.uint8)
    marks[[0, 5]] = td.Ghosts.DUPLICATECELL
    data.ghosts = td.Ghosts(cells=tack.array(marks))
    data.global_ids = td.GlobalIds(points=tack.array(np.arange(data.num_points)),
                                   cells=tack.array(np.arange(data.num_cells)))
    out = td.extract_extent(data, [(-3, 2), (0, 9), (1, 2)])
    assert out.topology.point_dims == (3, 4, 2) and out.topology.offset == (0, 0, 1)
    cells = out.global_ids.cells.numpy()
    np.testing.assert_array_equal(out.ghosts.cells.numpy(), marks[cells])
    np.testing.assert_array_equal(out.positions(),
                                  data.positions()[out.global_ids.points.numpy()])
    flat = td.extract_extent(td.uniform_grid((5, 4), dtype=tack.f64), (1, 3, 2, 2))
    assert flat.topology.point_dims == (3,) and flat.num_cells == 2      # a line of a 2D grid


def test_refusals():
    with pytest.raises(TypeError, match="structured grid"):
        td.extract_extent(td.tetrahedralize(td.uniform_grid((3, 3, 3))), (0, 1, 0, 1, 0, 1))
    grid = td.uniform_grid((3, 3, 3))
    with pytest.raises(ValueError, match="misses"):
        td.extract_extent(grid, (5, 6, 0, 1, 0, 1))
    with pytest.raises(ValueError, match="one point"):
        td.extract_extent(grid, (1, 1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="pairs"):
        td.extract_extent(grid, (0, 1, 0, 1))
    with pytest.raises(ValueError, match="at least 1"):
        td.extract_extent(grid, (0, 2, 0, 2, 0, 2), rate=0)
