"""tack.interop.vtk — Zero-copy interop between VTK arrays and Tack fields.

    from tack.interop.vtk import vtk_to_field, field_to_vtk

    field = vtk_to_field(vtk_array)              # VTK  -> Tack
    array = field_to_vtk(field, n_components=3)  # Tack -> VTK

Both directions share memory rather than copying, on the host and on the
device, and each side keeps the other alive for as long as it is needed.

What this module is for
-----------------------
The transport is DLPack, which VTK and Tack both speak, so almost nothing
is left here: no pointer parsing, no memory-space tables, no dtype maps.
What remains is the one thing DLPack cannot know, which is how the two
libraries disagree about shape.

A vtkDataArray is *tuples x components* -- 1000 points of 3 floats is
1000 tuples of 3. A Tack field is flat: the same data is 3000 values, and
vector-ness is a property of the kernel reading it. So every exchange has
to decide whether to flatten, and that decision is what this module owns.

Requirements
------------
VTK with ``vtkmodules.util.dlpack_support``. Device arrays additionally
need VTK built with Viskores; host arrays work with any VTK that has the
DLPack module.
"""

import tack

__all__ = ["vtk_to_field", "field_to_vtk"]


def _dlpack_support():
    """Import VTK's DLPack module, or explain what is missing."""
    try:
        from vtkmodules.util import dlpack_support
    except ImportError as exc:
        raise RuntimeError(
            "tack.interop.vtk needs vtkmodules.util.dlpack_support, which is "
            "not in this VTK. It provides the zero-copy exchange both "
            "directions rely on."
        ) from exc
    return dlpack_support


def vtk_to_field(vtk_array, flatten=True):
    """Wrap a vtkDataArray as a Tack field, without copying.

    Args:
        vtk_array: any vtkDataArray, on the host or on a device.
        flatten: if True (default) the field is 1-D of
            ``tuples * components`` values, which is how Tack kernels index
            interleaved data and what ``tack.Vector`` fields expect. If
            False the field keeps VTK's ``(tuples, components)`` shape.

    Returns:
        A tack.field sharing the array's memory. The VTK array is held for
        as long as the field lives, so it may be dropped immediately.

    Raises:
        RuntimeError: if the array lives somewhere the active Tack backend
            cannot address -- a CUDA array under the CPU backend, say.
        ValueError: for an array with no memory to share (implicit and
            computed arrays have none) or a non-contiguous layout.
    """
    dlpack_support = _dlpack_support()

    field = tack.from_dlpack(dlpack_support.vtk_to_dlpack(vtk_array))
    if flatten and len(field.shape) > 1:
        return field.reshape((field.size,))
    return field


def field_to_vtk(field, n_components=1, name=None):
    """Wrap a Tack field as a vtkDataArray, without copying.

    Args:
        field: the tack.field to share.
        n_components: components per tuple. A flat field of 3000 values
            with n_components=3 becomes 1000 tuples of 3 -- which is what
            VTK wants for points, vectors and colours.
        name: array name, as VTK uses to identify it in a dataset.

    Returns:
        A vtkDataArray sharing the field's memory. The field is held for as
        long as the array lives.

    Raises:
        ValueError: if the field's size is not a multiple of n_components.
    """
    # Validate before reaching for VTK, so a shape mistake reports itself
    # rather than being masked by a missing-VTK error.
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}")
    if field.size % n_components:
        raise ValueError(
            f"a field of {field.size} values does not divide into tuples of "
            f"{n_components}")

    dlpack_support = _dlpack_support()

    # VTK reads a DLPack tensor as (tuples, components), so reshape rather
    # than leaving it to guess -- a flat field would arrive as N tuples of 1.
    shaped = field
    if n_components > 1 or len(field.shape) != 2:
        shaped = field.reshape((field.size // n_components, n_components))

    return dlpack_support.dlpack_to_vtk(shaped, name=name)
