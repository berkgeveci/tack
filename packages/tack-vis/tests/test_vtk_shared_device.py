"""``vtk_to_dataset(..., copy=False)`` and ``dataset_to_vtk(..., copy=False)`` with the
data on a GPU: Tack's device arrays and VTK's Viskores arrays sharing one
allocation each way -- the points, data arrays, offsets and connectivity -- with the
dataset made on the device that holds them, and giving what the copies give.

Needs a VTK with DLPack and Viskores' arrays on a device Tack has (the arrays-only
build of VTK's Viskores is enough), so it skips with the VTK wheel."""

import gc

import numpy as np
import pytest

import tack
import tack.dataset as td

vtk = pytest.importorskip("vtkmodules.all")
from tack.interop import vtk as interop
from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

pytestmark = pytest.mark.quick

_SPACES = {"cuda": "CudaDeviceMemory", "hip": "HipDeviceMemory"}


def _arches():
    """Devices whose memory this VTK holds in Viskores arrays."""
    found = []
    try:
        from vtkmodules.util import dlpack_support  # noqa: F401
        from vtkmodules.vtkAcceleratorsVTKmCore import vtkmDataArrayFactory  # noqa: F401
    except ImportError:
        return found
    for arch, space in _SPACES.items():
        try:
            device = tack.device(arch)
            probe = interop.array_to_vtk(tack.zeros((4,), device=device))
        except (ImportError, RuntimeError, OSError, ValueError):
            continue
        if probe.GetMemorySpace() == getattr(vtk.vtkDataArray, space):
            found.append(arch)
    return found


@pytest.fixture(params=_arches() or [pytest.param(None, marks=pytest.mark.skip(
    reason="needs a VTK with DLPack and Viskores arrays on a GPU Tack has"))])
def device(request):
    tack.set_default_device("cpu")
    yield tack.device(request.param)
    tack.set_default_device("cpu")


def _address(values):
    """A Tack array's first byte, or the memory a VTK array describes."""
    if isinstance(values, tack.Array):
        return values._buffer.address
    return values.NewMemoryDescriptors().GetItemAsObject(0).GetPointer()


def _shares(a, b):
    return _address(a) == _address(b)


def _on_device(values, device):
    """A VTK array over ``values``' device memory, as ``array_to_vtk`` makes one."""
    return interop.array_to_vtk(tack.array(values, device=device),
                                n_components=None if values.ndim == 2 else 1)


def _host_tetrahedra():
    grid = vtk.vtkCellTypeSource(cell_type=vtk.VTK_TETRA, blocks_dimensions=(4, 3, 2))()
    x = np.ascontiguousarray(np.asarray(grid.points.data), np.float32)
    grid.points = vtk.vtkPoints(data=vtk.vtkFloatArray(x))
    grid.point_data["s"] = np.sin(x[:, 0] * 3).astype(np.float32)
    grid.point_data["v"] = x.astype(np.float64) * 2.0
    grid.cell_data["c"] = np.arange(grid.number_of_cells, dtype=np.int32)
    return grid


def _device_tetrahedra(device):
    """The same grid with its points, cells and data arrays in ``device``'s memory."""
    host = _host_tetrahedra()
    grid = vtk.vtkUnstructuredGrid()
    grid.points = vtk.vtkPoints(data=_on_device(np.asarray(host.points.data), device))
    cells = host.cells
    grid.cells = (vtk.VTK_TETRA, vtk.vtkCellArray(
        offsets=_on_device(np.asarray(cells.offsets_array, np.int64), device),
        connectivity=_on_device(np.asarray(cells.connectivity_array, np.int64), device)))
    for attributes, source in ((grid.point_data, host.point_data),
                               (grid.cell_data, host.cell_data)):
        for name in source.keys():
            array = _on_device(np.asarray(source[name]), device)
            array.SetName(name)
            attributes.AddArray(array)
    return grid, host


def test_a_device_grid_is_shared_on_its_device(device):
    grid, _ = _device_tetrahedra(device)
    data = vtk_to_dataset(grid, dtype=None, copy=False)
    topology = data.topology
    assert data.device == device                # though the default is the CPU
    assert _shares(data.geometry.values, grid.points.data)
    assert _shares(topology.offsets, grid.cells.offsets_array)
    assert _shares(topology.connectivity, grid.cells.connectivity_array)
    for attributes, names in ((grid.point_data, ("s", "v")), (grid.cell_data, ("c",))):
        for name in names:
            assert _shares(data.fields[name].values, attributes.GetArray(name))
    # Each in its own type, the ids in VTK's width.
    assert topology.connectivity.dtype == tack.i64
    assert data.fields["v"].values.dtype == tack.vec(3, tack.f64)
    assert data.geometry.values.dtype == tack.vec(3, tack.f32)


def test_tack_does_not_write_into_vtks_device_memory(device):
    grid, _ = _device_tetrahedra(device)
    data = vtk_to_dataset(grid, dtype=None, copy=False)
    with pytest.raises(RuntimeError, match="read-only"):
        data.fields["s"].values.fill(0)


def test_host_arrays_beside_them_are_copied_to_the_device(device):
    grid, _ = _device_tetrahedra(device)
    grid.point_data["h"] = np.arange(grid.number_of_points, dtype=np.float32)
    data = vtk_to_dataset(grid, dtype=None, copy=False)
    assert data.fields["h"].values.device == device
    np.testing.assert_array_equal(data.fields["h"].values.numpy(),
                                  np.arange(grid.number_of_points, dtype=np.float32))


def test_shared_device_arrays_give_what_copies_give(device):
    grid, host = _device_tetrahedra(device)
    shared = vtk_to_dataset(grid, dtype=None, copy=False)
    with device:
        copied = vtk_to_dataset(host, dtype=None)
    for name in ("s", "v", "c"):
        np.testing.assert_array_equal(shared.fields[name].values.numpy(),
                                      copied.fields[name].values.numpy())
    a, b = td.contour(shared, "s", 0.3), td.contour(copied, "s", 0.3)
    assert a.num_cells == b.num_cells > 0
    np.testing.assert_array_equal(a.geometry.values.numpy(), b.geometry.values.numpy())


def test_the_dataset_keeps_vtks_device_arrays_alive(device):
    grid, host = _device_tetrahedra(device)
    data = vtk_to_dataset(grid, dtype=None, copy=False)
    del grid
    gc.collect()
    np.testing.assert_array_equal(data.fields["s"].values.numpy(),
                                  np.asarray(host.point_data["s"]))


def _device_dataset(device):
    with device:
        return vtk_to_dataset(_host_tetrahedra(), dtype=None)


def test_a_device_dataset_is_shared_with_vtk(device):
    data = _device_dataset(device)
    grid = dataset_to_vtk(data, copy=False)
    space = getattr(vtk.vtkDataArray, _SPACES[device.type])
    for vtk_array, values in ((grid.points.data, data.geometry.values),
                              (grid.cells.offsets_array, data.topology.offsets),
                              (grid.cells.connectivity_array, data.topology.connectivity),
                              (grid.point_data.GetArray("s"), data.fields["s"].values),
                              (grid.point_data.GetArray("v"), data.fields["v"].values),
                              (grid.cell_data.GetArray("c"), data.fields["c"].values)):
        assert vtk_array.GetMemorySpace() == space
        assert _shares(values, vtk_array)
    # VTK reads them on the host as it would its own.
    np.testing.assert_array_equal(np.asarray(grid.point_data["s"]),
                                  data.fields["s"].values.numpy())


def test_vtk_filters_read_shared_device_arrays(device):
    data = _device_dataset(device)
    shared, copied = dataset_to_vtk(data, copy=False), dataset_to_vtk(data)
    outputs = []
    for grid in (shared, copied):
        contour = vtk.vtkContourFilter(input_data=grid)
        contour.SetInputArrayToProcess(0, 0, 0, 0, "s")
        contour.SetValue(0, 0.3)
        contour.Update()
        out = contour.GetOutput()
        outputs.append((np.asarray(out.points.data), np.asarray(out.polys.connectivity_array)))
    (p1, c1), (p2, c2) = outputs
    assert len(c1) > 0
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(c1, c2)


def test_vtk_keeps_tacks_device_arrays_alive(device):
    data = _device_dataset(device)
    expected = data.fields["s"].values.numpy().copy()
    grid = dataset_to_vtk(data, copy=False)
    del data
    gc.collect()
    np.testing.assert_array_equal(np.asarray(grid.point_data["s"]), expected)


def test_a_round_trip_shares_both_ways(device):
    data = _device_dataset(device)
    back = vtk_to_dataset(dataset_to_vtk(data, copy=False), dtype=None, copy=False)
    assert back.device == device
    assert _shares(back.geometry.values, data.geometry.values)
    assert _shares(back.topology.connectivity, data.topology.connectivity)
    assert _shares(back.fields["v"].values, data.fields["v"].values)


def test_a_chain_of_tack_algorithms_stays_on_the_device(device):
    from tack.interop.vtk.algorithms import Clip, Contour

    space = getattr(vtk.vtkDataArray, _SPACES[device.type])
    grid, host = _device_tetrahedra(device)
    outputs = []
    for source in (grid, host):
        clip, contour = Clip(by="s", value=0.0), Contour(field="s", value=0.3)
        clip.SetInputDataObject(source)
        contour.SetInputConnection(clip.GetOutputPort())
        contour.Update()
        outputs.append((clip.GetOutputDataObject(0), contour.GetOutputDataObject(0)))
    (clipped, surface), (_, host_surface) = outputs
    assert clipped.points.data.GetMemorySpace() == space
    assert clipped.point_data.GetArray("s").GetMemorySpace() == space
    assert surface.points.data.GetMemorySpace() == space
    assert surface.number_of_cells == host_surface.number_of_cells > 0
    # The filters ran on the GPU in one and the CPU in the other: f32 rounding apart.
    np.testing.assert_allclose(np.asarray(surface.points.data),
                               np.asarray(host_surface.points.data), atol=1e-6)


def test_a_filter_given_a_device_moves_its_input_there_once(device):
    """A reader's output is on the host: device= puts the filter on the GPU, and
    the filters after it find their input there."""
    from tack.interop.vtk import algorithms as ta

    space = getattr(vtk.vtkDataArray, _SPACES[device.type])
    host = _host_tetrahedra()
    contour = ta.Contour(field="s", value=0.3, device=device.type)
    contour.SetInputDataObject(host)
    contour.Update()
    surface = contour.GetOutputDataObject(0)
    assert contour.error is None and surface.points.data.GetMemorySpace() == space
    moved = ta._as_dataset(host, device)
    assert moved.device == device and ta._as_dataset(host, device) is moved
    on_cpu = ta.Contour(field="s", value=0.3)(host)
    assert surface.number_of_cells == on_cpu.number_of_cells > 0
    np.testing.assert_allclose(np.asarray(surface.points.data),
                               np.asarray(on_cpu.points.data), atol=1e-6)
    probe = ta.Probe(fields=["v"])
    probe.SetInputConnection(contour.GetOutputPort())
    probe.SetSourceData(host)
    probe.device = device.type
    probe.Update()
    assert probe.error is None
    assert probe.GetOutputDataObject(0).point_data.GetArray("v").GetMemorySpace() == space


def test_marks_and_ids_go_to_the_host(device):
    """VTK reads ghost marks and global ids as its own concrete arrays."""
    data = _device_dataset(device)
    data = td.generate_global_ids(td.PartitionedDataSet([data]))[0]
    grid = dataset_to_vtk(data, copy=False)
    ids = grid.point_data.GetGlobalIds()
    assert isinstance(ids, vtk.vtkIdTypeArray)
    np.testing.assert_array_equal(np.asarray(ids), data.global_ids.points.numpy())


def test_without_dlpack_device_arrays_are_copied(device, monkeypatch):
    def missing():
        raise RuntimeError("no dlpack_support")

    data = _device_dataset(device)
    monkeypatch.setattr(interop, "_dlpack_support", missing)
    grid = dataset_to_vtk(data, copy=False)
    assert isinstance(grid.point_data.GetArray("s"), vtk.vtkFloatArray)
    np.testing.assert_array_equal(np.asarray(grid.point_data["s"]),
                                  data.fields["s"].values.numpy())
