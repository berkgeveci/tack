"""``vtk_to_dataset(..., copy=False)`` and ``dataset_to_vtk(..., copy=False)``: VTK's
and Tack's datasets sharing their arrays on the CPU instead of copying them --
the points, data arrays, cell types, offsets and connectivity, ghost marks and
global ids -- each holding the other's memory alive, and giving what the copies
give."""

import gc

import numpy as np
import pytest

import tack
import tack.dataset as td

vtk = pytest.importorskip("vtkmodules.all")
from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _address(values):
    """A Tack array's or a VTK array's first byte."""
    if isinstance(values, tack.Array):
        return values._buffer.address
    return np.asarray(values).ctypes.data


def _shares(a, b):
    return _address(a) == _address(b)


def _tetrahedra():
    """A small grid of tetrahedra with f32 points, f32, f64 and integer point and
    cell data, ghost marks and global ids."""
    grid = vtk.vtkCellTypeSource(cell_type=vtk.VTK_TETRA, blocks_dimensions=(3, 2, 2))()
    x = np.asarray(grid.points.data)
    # vtkPoints(data=) stores a NumPy array as doubles: a typed array keeps f32.
    grid.points = vtk.vtkPoints(data=vtk.vtkFloatArray(np.ascontiguousarray(x, np.float32)))
    n, m = grid.number_of_points, grid.number_of_cells
    grid.point_data["s"] = np.sin(x[:, 0]).astype(np.float32)
    grid.point_data["v"] = x.astype(np.float64) * 2.0
    grid.cell_data["c"] = np.arange(m, dtype=np.int32)
    grid.cell_data["vtkGhostType"] = np.zeros(m, np.uint8)
    # Global ids have no pythonic setter in VTK 9.7.1.
    ids = vtk.vtkIdTypeArray(np.arange(n, dtype=np.int64) + 100)
    ids.SetName("GlobalPointIds")
    grid.point_data.SetGlobalIds(ids)
    return grid


def test_an_unstructured_grid_is_shared_not_copied():
    grid = _tetrahedra()
    data = vtk_to_dataset(grid, dtype=None, copy=False)
    topology = data.topology
    assert _shares(data.geometry.values, grid.points.data)
    assert _shares(data.fields["s"].values, grid.point_data["s"])
    assert _shares(data.fields["v"].values, grid.point_data["v"])
    assert _shares(data.fields["c"].values, grid.cell_data["c"])
    assert _shares(topology.offsets, grid.cells.offsets_array)
    assert _shares(topology.connectivity, grid.cells.connectivity_array)
    assert _shares(topology.types, grid.cell_types)
    assert _shares(data.ghosts.cells, grid.cell_data["vtkGhostType"])
    assert _shares(data.global_ids.points, grid.point_data.GetGlobalIds())
    # Each array in its own type: the ids in VTK's width, f64 kept f64.
    assert topology.connectivity.dtype == tack.i64
    assert data.fields["v"].values.dtype == tack.vec(3, tack.f64)
    assert data.geometry.values.dtype == tack.vec(3, tack.f32)


def test_shared_arrays_give_what_copies_give():
    grid = _tetrahedra()
    shared = vtk_to_dataset(grid, dtype=None, copy=False)
    copied = vtk_to_dataset(grid, dtype=None)
    np.testing.assert_array_equal(shared.positions(), copied.positions())
    for name in copied.fields:
        np.testing.assert_array_equal(shared.fields[name].values.numpy(),
                                      copied.fields[name].values.numpy())
    for a, b in ((shared, copied),):
        np.testing.assert_array_equal(np.asarray(a.topology.connectivity),
                                      np.asarray(b.topology.connectivity))
    s = td.contour(shared, "s", 0.3)
    c = td.contour(copied, "s", 0.3)
    np.testing.assert_array_equal(s.positions(), c.positions())


def test_tack_does_not_write_into_vtks_memory():
    grid = _tetrahedra()
    data = vtk_to_dataset(grid, dtype=None, copy=False)

    @tack.kernel
    def scale(x: tack.Array):
        for i in range(x.shape[0]):
            x[i] = 2.0 * x[i]

    with pytest.raises(ValueError):
        scale(data.fields["s"].values)


def test_the_dataset_keeps_vtks_arrays_alive():
    grid = _tetrahedra()
    expected = np.asarray(grid.point_data["s"]).copy()
    data = vtk_to_dataset(grid, dtype=None, copy=False)
    del grid
    gc.collect()
    np.testing.assert_array_equal(data.fields["s"].values.numpy(), expected)


def test_a_conversion_that_must_convert_copies():
    """A type asked for that VTK's arrays are not in, or an origin to subtract,
    copies those arrays, as before."""
    grid = _tetrahedra()
    data = vtk_to_dataset(grid, dtype=tack.f32, copy=False)
    assert not _shares(data.fields["v"].values, grid.point_data["v"])       # f64 -> f32
    assert _shares(data.fields["s"].values, grid.point_data["s"])           # already f32
    moved = vtk_to_dataset(grid, dtype=None, copy=False, origin=(1.0, 0.0, 0.0))
    assert not _shares(moved.geometry.values, grid.points.data)
    np.testing.assert_allclose(moved.world_positions(), np.asarray(grid.points.data), atol=1e-6)


def test_polydata_of_one_kind_shares_its_cells():
    sphere = vtk.vtkSphereSource(theta_resolution=12, phi_resolution=8)()
    data = vtk_to_dataset(sphere, dtype=None, copy=False)
    assert _shares(data.topology.connectivity, sphere.polys.connectivity_array)
    assert _shares(data.geometry.values, sphere.points.data)
    copied = vtk_to_dataset(sphere, dtype=None)
    np.testing.assert_array_equal(np.asarray(data.topology.connectivity),
                                  np.asarray(copied.topology.connectivity))
    np.testing.assert_array_equal(np.asarray(data.topology.types),
                                  np.asarray(copied.topology.types))


def test_triangle_strips_are_copied_as_their_triangles():
    strips = vtk.vtkStripper()(vtk.vtkSphereSource(theta_resolution=8, phi_resolution=6)())
    assert strips.strips.number_of_cells
    shared = vtk_to_dataset(strips, dtype=None, copy=False)
    copied = vtk_to_dataset(strips, dtype=None)
    assert shared.num_cells == copied.num_cells
    np.testing.assert_array_equal(np.asarray(shared.topology.connectivity),
                                  np.asarray(copied.topology.connectivity))


def test_an_image_shares_its_data_arrays():
    image = vtk.vtkRTAnalyticSource(whole_extent=(-4, 4, -4, 4, -4, 4))()
    data = vtk_to_dataset(image, dtype=None, copy=False)
    assert _shares(data.fields["RTData"].values, image.point_data["RTData"])
    assert data.geometry.values.dtype == tack.f64           # VTK's image positions
    np.testing.assert_allclose(data.positions(), np.asarray(
        [image.GetPoint(i) for i in range(image.number_of_points)]))


def test_a_structured_grid_shares_its_points():
    image = vtk.vtkRTAnalyticSource(whole_extent=(0, 3, 0, 3, 0, 2))()
    grid = vtk.vtkStructuredGrid(dimensions=image.dimensions)
    x = np.asarray([image.GetPoint(i) for i in range(image.number_of_points)], np.float32)
    grid.points = vtk.vtkPoints(data=vtk.vtkFloatArray(x))
    data = vtk_to_dataset(grid, dtype=None, copy=False)
    assert _shares(data.geometry.values, grid.points.data)


def test_a_tack_dataset_is_shared_with_vtk():
    data = vtk_to_dataset(_tetrahedra())            # Tack's own arrays: i32 ids
    grid = dataset_to_vtk(data, copy=False)
    topology = data.topology
    assert _shares(data.geometry.values, grid.points.data)
    assert _shares(data.fields["s"].values, grid.point_data["s"])
    assert _shares(data.fields["c"].values, grid.cell_data["c"])
    assert _shares(data.ghosts.cells, grid.cell_data["vtkGhostType"])
    assert _shares(topology.connectivity, grid.cells.connectivity_array)
    assert _shares(topology.offsets, grid.cells.offsets_array)
    # i32 ids stay i32: VTK keeps 32-bit cell arrays too.
    assert topology.connectivity.dtype == tack.i32 and not grid.cells.IsStorage64Bit()
    copied = dataset_to_vtk(data)
    np.testing.assert_array_equal(np.asarray(grid.points.data), np.asarray(copied.points.data))
    np.testing.assert_array_equal(np.asarray(grid.cells.connectivity_array),
                                  np.asarray(copied.cells.connectivity_array))
    np.testing.assert_array_equal(np.asarray(grid.cell_types), np.asarray(copied.cell_types))
    assert vtk.vtkCellSizeFilter()(grid).cell_data["Volume"] is not None


def test_vtk_keeps_tacks_arrays_alive():
    data = vtk_to_dataset(_tetrahedra())
    expected = data.positions().copy()
    grid = dataset_to_vtk(data, copy=False)
    del data
    gc.collect()
    np.testing.assert_array_equal(np.asarray(grid.points.data), expected)


def test_pieces_are_shared_both_ways():
    pieces = vtk.vtkPartitionedDataSet()
    grids = [_tetrahedra(), _tetrahedra()]
    for g in grids:
        pieces.append(g)
    pds = vtk_to_dataset(pieces, dtype=None, copy=False)
    assert isinstance(pds, td.PartitionedDataSet) and len(pds) == 2
    for piece, g in zip(pds, grids):
        assert _shares(piece.geometry.values, g.points.data)
    out = dataset_to_vtk(pds, copy=False)
    for piece, block in zip(pds, out):
        assert _shares(piece.fields["s"].values, block.point_data["s"])


def test_a_round_trip_through_a_filter_shares_both_ways():
    grid = _tetrahedra()
    surface = td.extract_surface(vtk_to_dataset(grid, dtype=None, copy=False))
    out = dataset_to_vtk(surface, copy=False)
    assert _shares(surface.geometry.values, out.points.data)
    reference = vtk.vtkGeometryFilter()(grid)
    assert out.number_of_cells == reference.number_of_cells
