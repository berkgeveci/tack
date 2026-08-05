"""Tests for tack.interop.vtk.

VTK is an optional dependency and is not installed in CI, so these tests
drive vtk_to_field through a stand-in that implements only the four
vtkDataArray methods the converter actually calls.  That is enough to
cover the type mapping, the pointer parsing, and the zero-copy wrap —
which is where the bugs live.  field_to_vtk needs real VTK and skips.
"""

import numpy as np
import pytest

import tack
from tack.interop.vtk import _parse_vtk_pointer, vtk_to_field

# VTK type enum values used by the converter
VTK_FLOAT, VTK_DOUBLE, VTK_INT, VTK_LONG_LONG, VTK_ID_TYPE = 10, 11, 6, 8, 12

_HOST_MEMORY = 0
_CUDA_DEVICE_MEMORY = 1


class FakeVTKArray:
    """The slice of vtkDataArray that vtk_to_field depends on."""

    def __init__(self, array, data_type, memory_space=_HOST_MEMORY,
                 null_pointer=False):
        self._array = array           # keeps the buffer alive
        self._data_type = data_type
        self._memory_space = memory_space
        self._null = null_pointer

    def GetDataType(self):
        return self._data_type

    def GetDataTypeAsString(self):
        return f"type{self._data_type}"

    def GetNumberOfValues(self):
        return self._array.size

    def GetMemorySpace(self):
        return self._memory_space

    def _ptr_string(self):
        if self._null:
            return "_0000000000000000_p_void"
        return "_%016x_p_void" % self._array.ctypes.data

    def GetVoidPointer(self, _i):
        return self._ptr_string()

    def GetDeviceVoidPointer(self, _i):
        return self._ptr_string()


# ================================================================
# Pointer parsing
# ================================================================

def test_parse_pointer():
    assert _parse_vtk_pointer("_00007f9a1c2d3e40_p_void") == 0x7F9A1C2D3E40


def test_parse_pointer_none_is_null():
    assert _parse_vtk_pointer(None) == 0


def test_parse_pointer_garbage_is_null():
    assert _parse_vtk_pointer("not a vtk pointer") == 0
    assert _parse_vtk_pointer("_zzzz_p_void") == 0
    assert _parse_vtk_pointer("") == 0


# ================================================================
# vtk_to_field
# ================================================================

def _cpu_only(backend):
    if backend != "cpu":
        pytest.skip("host-pointer wrapping needs the CPU backend")


@pytest.mark.parametrize("vtk_type,np_dtype,tack_dtype", [
    (VTK_FLOAT, np.float32, tack.f32),
    (VTK_DOUBLE, np.float64, tack.f64),
    (VTK_INT, np.int32, tack.i32),
    (VTK_LONG_LONG, np.int64, tack.i64),
    (VTK_ID_TYPE, np.int64, tack.i64),
])
def test_host_array_round_trip(backend, vtk_type, np_dtype, tack_dtype):
    """Each supported VTK type maps to the right tack dtype and values."""
    _cpu_only(backend)
    values = np.arange(6, dtype=np_dtype)
    field = vtk_to_field(FakeVTKArray(values, vtk_type))
    assert field.dtype is tack_dtype
    assert field.shape == (6,)
    np.testing.assert_array_equal(field.to_numpy(), values)


def test_wrap_is_zero_copy(backend):
    """The field is a view: writes through VTK show up in the field."""
    _cpu_only(backend)
    values = np.arange(4, dtype=np.float32)
    field = vtk_to_field(FakeVTKArray(values, VTK_FLOAT))
    values[2] = 99.0
    assert field.to_numpy()[2] == 99.0


def test_multi_component_array_is_flattened(backend):
    """GetNumberOfValues counts tuples × components, so the field is flat."""
    _cpu_only(backend)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    field = vtk_to_field(FakeVTKArray(values, VTK_FLOAT))
    assert field.shape == (12,)
    np.testing.assert_array_equal(field.to_numpy(), values.ravel())


def test_unsupported_type_raises(backend):
    values = np.zeros(4, dtype=np.uint8)
    with pytest.raises(TypeError, match="Unsupported VTK data type"):
        vtk_to_field(FakeVTKArray(values, 3))  # VTK_UNSIGNED_CHAR


def test_null_host_pointer_raises(backend):
    _cpu_only(backend)
    values = np.zeros(4, dtype=np.float32)
    with pytest.raises(RuntimeError, match="GetVoidPointer returned null"):
        vtk_to_field(FakeVTKArray(values, VTK_FLOAT, null_pointer=True))


def test_null_device_pointer_raises(backend):
    values = np.zeros(4, dtype=np.float32)
    with pytest.raises(RuntimeError, match="GetDeviceVoidPointer returned null"):
        vtk_to_field(FakeVTKArray(values, VTK_FLOAT,
                                  memory_space=_CUDA_DEVICE_MEMORY,
                                  null_pointer=True))


def test_device_memory_space_takes_the_device_path(backend):
    """A non-host memory space is routed through GetDeviceVoidPointer.

    On the CPU backend the address is still host memory, so the wrap
    succeeds — the point is that the device branch was taken at all.
    """
    _cpu_only(backend)
    values = np.arange(4, dtype=np.float32)

    calls = []

    class Tracking(FakeVTKArray):
        def GetVoidPointer(self, i):
            calls.append("host")
            return super().GetVoidPointer(i)

        def GetDeviceVoidPointer(self, i):
            calls.append("device")
            return super().GetDeviceVoidPointer(i)

    field = vtk_to_field(Tracking(values, VTK_FLOAT,
                                  memory_space=_CUDA_DEVICE_MEMORY))
    assert calls == ["device"]
    np.testing.assert_array_equal(field.to_numpy(), values)


def test_empty_array(backend):
    _cpu_only(backend)
    values = np.zeros(0, dtype=np.float32)
    field = vtk_to_field(FakeVTKArray(values, VTK_FLOAT))
    assert field.shape == (0,)


# ================================================================
# field_to_vtk — needs real VTK
# ================================================================

def test_field_to_vtk_round_trip(backend):
    pytest.importorskip("vtkmodules.util.numpy_support")
    _cpu_only(backend)
    from tack.interop.vtk import field_to_vtk

    values = np.arange(9, dtype=np.float32)
    field = tack.field(dtype=tack.f32, shape=(9,))
    field.from_numpy(values)

    vtk_array = field_to_vtk(field, n_components=3)
    assert vtk_array.GetNumberOfComponents() == 3
    assert vtk_array.GetNumberOfTuples() == 3
    assert vtk_array.GetTuple3(1) == pytest.approx((3.0, 4.0, 5.0))
