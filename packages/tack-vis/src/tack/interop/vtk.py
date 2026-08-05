"""tack.interop.vtk — Zero-copy interop between VTK arrays and Tack fields.

Supports both host (vtkDataArray) and device (vtkmDataArray) arrays.
vtkmDataArray is not directly exposed to Python, but virtual dispatch
on vtkDataArray's GetDeviceVoidPointer() and GetMemorySpace() works
transparently through the vtable.

Usage
-----
::

    from tack.interop.vtk import vtk_to_field, field_to_vtk

    # VTK → Tack (zero-copy for host and device arrays)
    field = vtk_to_field(vtk_data_array)

    # Tack → VTK (zero-copy for host arrays)
    vtk_array = field_to_vtk(field, n_components=3)
"""

import numpy as np
import tack


# VTK MemorySpace enum values (from vtkDataArray.h)
_HOST_MEMORY = 0
_CUDA_DEVICE_MEMORY = 1
_HIP_DEVICE_MEMORY = 2

# VTK type enum → (numpy dtype, tack dtype)
_VTK_TYPE_MAP = {
    10: (np.float32, tack.f32),   # VTK_FLOAT
    11: (np.float64, tack.f64),   # VTK_DOUBLE
    6:  (np.int32,   tack.i32),   # VTK_INT
    8:  (np.int64,   tack.i64),   # VTK_LONG_LONG (or VTK_ID_TYPE on 64-bit)
    12: (np.int64,   tack.i64),   # VTK_ID_TYPE
}


def _parse_vtk_pointer(ptr_str):
    """Extract integer address from VTK's mangled pointer string.

    VTK wraps void* returns as '_HEXADDR_p_void' strings.
    Returns 0 if ptr_str is None or cannot be parsed.
    """
    if ptr_str is None:
        return 0
    try:
        return int(ptr_str.split('_')[1], 16)
    except (IndexError, ValueError):
        return 0


def vtk_to_field(vtk_array):
    """Convert a vtkDataArray to a tack.field (zero-copy when possible).

    For host arrays (vtkFloatArray, etc.): wraps the host pointer via
    field_from_ptr.  The VTK array must outlive the returned field.

    For device arrays (vtkmDataArray): wraps the device pointer via
    field_from_ptr.  Works for CUDA, HIP, and any backend where
    vtkmDataArray stores data on-device.

    Args:
        vtk_array: A vtkDataArray instance.

    Returns:
        tack.field wrapping the array's memory (zero-copy).

    Raises:
        TypeError: If the array type is not supported.
        RuntimeError: If the pointer cannot be extracted.
    """
    data_type = vtk_array.GetDataType()
    if data_type not in _VTK_TYPE_MAP:
        raise TypeError(
            f"Unsupported VTK data type: {data_type} "
            f"({vtk_array.GetDataTypeAsString()})")

    np_dtype, tack_dtype = _VTK_TYPE_MAP[data_type]
    n_values = int(vtk_array.GetNumberOfValues())  # tuples * components

    memory_space = vtk_array.GetMemorySpace()

    if memory_space != _HOST_MEMORY:
        # Device array — use GetDeviceVoidPointer
        ptr_str = vtk_array.GetDeviceVoidPointer(0)
        addr = _parse_vtk_pointer(ptr_str)
        if addr == 0:
            raise RuntimeError(
                "vtk_to_field: GetDeviceVoidPointer returned null. "
                "Array reports device memory but pointer is unavailable.")
        return tack.field_from_ptr(addr, tack_dtype, (n_values,))

    # Host array — use GetVoidPointer
    ptr_str = vtk_array.GetVoidPointer(0)
    addr = _parse_vtk_pointer(ptr_str)
    if addr == 0:
        raise RuntimeError(
            "vtk_to_field: GetVoidPointer returned null.")
    return tack.field_from_ptr(addr, tack_dtype, (n_values,))


def field_to_vtk(field, n_components=1):
    """Create a vtkDataArray wrapping a tack.field's memory (zero-copy).

    The field must be on the host (CPU backend or after to_numpy()).
    The returned VTK array shares memory with the field — the field
    must outlive the VTK array.

    Args:
        field: A tack.field.
        n_components: Number of components per tuple (default 1).

    Returns:
        A vtkDataArray (e.g. vtkFloatArray) sharing the field's memory.
    """
    try:
        import vtkmodules.vtkCommonCore as vtk_core
    except ImportError:
        import vtk as vtk_core

    arr_np = field.to_numpy()
    n_tuples = arr_np.shape[0] // n_components

    # Map dtype to VTK array type
    dtype = arr_np.dtype
    if dtype == np.float32:
        vtk_arr = vtk_core.vtkFloatArray()
    elif dtype == np.float64:
        vtk_arr = vtk_core.vtkDoubleArray()
    elif dtype == np.int32:
        vtk_arr = vtk_core.vtkIntArray()
    elif dtype == np.int64:
        vtk_arr = vtk_core.vtkLongLongArray()
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

    vtk_arr.SetNumberOfComponents(n_components)
    vtk_arr.SetNumberOfTuples(n_tuples)

    # Copy data (VTK Python doesn't support SetVoidArray from Python easily)
    from vtkmodules.util.numpy_support import numpy_to_vtk
    result = numpy_to_vtk(arr_np, deep=False)
    result.SetNumberOfComponents(n_components)
    return result
