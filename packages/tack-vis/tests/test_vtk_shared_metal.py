"""Metal arrays shared with VTK, with any VTK: on Apple silicon a Metal buffer is
host memory, so ``dataset_to_vtk(copy=False)`` hands VTK views of the buffers
themselves (after one wait), and ``vtk_to_dataset(copy=False)`` with Metal the
default device takes them back as the same buffers -- a pipeline of Tack filters
on Metal through VTK copies nothing. VTK's own memory is wrapped in place where it
lies on whole pages, and copied to Metal otherwise, so a dataset is on Metal
throughout. ``test_vtk_shared.py`` holds the CPU's sharing."""

import gc
import mmap

import numpy as np
import pytest

import tack
import tack.dataset as td

vtk = pytest.importorskip("vtkmodules.all")

from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
from tack.runtime.metal import _handed_out

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def metal():
    try:
        tack.set_default_device("metal")
    except (ImportError, RuntimeError):
        pytest.skip("no Metal device")
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
    """A grid of tetrahedra with f32 points and f32 point and i32 cell data."""
    grid = vtk.vtkCellTypeSource(cell_type=vtk.VTK_TETRA, blocks_dimensions=(3, 2, 2))()
    x = np.asarray(grid.points.data)
    grid.points = vtk.vtkPoints(data=vtk.vtkFloatArray(np.ascontiguousarray(x, np.float32)))
    grid.point_data["s"] = np.sin(x[:, 0]).astype(np.float32)
    grid.cell_data["c"] = np.arange(grid.number_of_cells, dtype=np.int32)
    return grid


def _on_metal(data):
    """A dataset of ``_tetrahedra`` made by Tack on Metal (its own buffers)."""
    tack.set_default_device("cpu")
    data = vtk_to_dataset(_tetrahedra(), dtype=None).to_device(tack.device("metal"))
    tack.set_default_device("metal")
    return data


def test_a_metal_dataset_is_shared_with_vtk():
    """VTK's arrays are views of the Metal buffers, and read what copies read."""
    data = _on_metal(None)
    assert data.device.type == "metal"
    grid = dataset_to_vtk(data, copy=False)
    assert _shares(data.geometry.values, grid.points.data)
    assert _shares(data.fields["s"].values, grid.point_data["s"])
    assert _shares(data.fields["c"].values, grid.cell_data["c"])
    assert _shares(data.topology.connectivity, grid.cells.connectivity_array)
    copied = dataset_to_vtk(data)
    np.testing.assert_array_equal(np.asarray(grid.points.data), np.asarray(copied.points.data))
    np.testing.assert_array_equal(np.asarray(grid.point_data["s"]),
                                  np.asarray(copied.point_data["s"]))
    assert vtk.vtkCellSizeFilter()(grid).cell_data["Volume"] is not None


def test_vtk_keeps_metal_arrays_alive():
    data = _on_metal(None)
    expected = data.positions().copy()
    grid = dataset_to_vtk(data, copy=False)
    del data
    gc.collect()
    np.testing.assert_array_equal(np.asarray(grid.points.data), expected)


def test_metal_buffers_come_back_as_themselves():
    """A VTK grid over Tack's Metal buffers, read back with ``copy=False`` and Metal
    the default device, is the same buffers: nothing copied either way."""
    data = _on_metal(None)
    grid = dataset_to_vtk(data, copy=False)
    back = vtk_to_dataset(grid, dtype=None, copy=False)
    assert back.device.type == "metal"
    assert _shares(back.geometry.values, data.geometry.values)
    assert _shares(back.fields["s"].values, data.fields["s"].values)
    assert _shares(back.fields["c"].values, data.fields["c"].values)
    assert back.geometry.values._buffer.metal_buffer is data.geometry.values._buffer.metal_buffer


def test_host_memory_on_whole_pages_is_wrapped_in_place():
    """Memory on page boundaries, a whole number of pages long, becomes a Metal array
    over it (Metal's no-copy buffer); other host memory is refused, for a copy."""
    pages = mmap.mmap(-1, 2 * mmap.PAGESIZE)
    host = np.frombuffer(pages, dtype=np.float32)
    host[:] = np.arange(host.size)
    wrapped = tack.array_from_ptr(host, (host.size,), tack.f32, device="metal")
    assert wrapped._buffer.address == host.ctypes.data
    host[3] = 42.0                              # the same memory, seen both ways
    assert wrapped.numpy()[3] == 42.0
    odd = np.arange(1000, dtype=np.float32)     # not a whole number of pages
    with pytest.raises(TypeError, match="whole pages"):
        tack.array_from_ptr(odd, (odd.size,), tack.f32, device="metal")


def test_vtk_memory_that_cannot_be_wrapped_is_copied_to_metal():
    """A grid VTK made, read with ``copy=False`` and Metal the default device, is on
    Metal throughout (its small arrays copied there), and filters it as the CPU
    does."""
    grid = _tetrahedra()
    data = vtk_to_dataset(grid, dtype=None, copy=False)
    assert data.device.type == "metal"
    assert data.geometry.values.device.type == "metal"
    assert data.topology.connectivity.device.type == "metal"
    surface = td.extract_surface(data)
    tack.set_default_device("cpu")
    reference = td.extract_surface(vtk_to_dataset(grid, dtype=None))
    np.testing.assert_allclose(np.sort(surface.positions(), axis=0),
                               np.sort(reference.positions(), axis=0))


def test_a_pipeline_through_vtk_on_metal_copies_nothing():
    """Tack on Metal, VTK, Tack on Metal: the second Tack filter reads the first's
    output buffers, and the answer is the CPU's."""
    data = _on_metal(None)
    surface = td.extract_surface(data)
    grid = dataset_to_vtk(surface, copy=False)
    assert surface.geometry.values._buffer.address in _handed_out
    sizes = vtk.vtkCellSizeFilter()(grid)                 # VTK reads Metal memory
    back = vtk_to_dataset(sizes, dtype=None, copy=False)
    assert _shares(back.geometry.values, surface.geometry.values)
    area = td.cell_geometry(back).measure.values.numpy()
    np.testing.assert_allclose(area, np.asarray(sizes.cell_data["Area"]), rtol=1e-6)
