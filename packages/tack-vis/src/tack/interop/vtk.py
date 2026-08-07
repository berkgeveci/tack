"""tack.interop.vtk — Zero-copy interop between VTK arrays and Tack fields.

    from tack.interop.vtk import vtk_to_field, field_to_vtk

    field = vtk_to_field(vtk_array)   # VTK  -> Tack
    array = field_to_vtk(field)       # Tack -> VTK

Both directions share memory rather than copying, on the host and on the
device, and each side keeps the other alive for as long as it is needed.

Shape
-----
A vtkDataArray is *tuples x components*, and a Tack field is n-dimensional,
so the two line up directly: a 2-D field of shape ``(n, 3)`` is n tuples of
3 components, and ``field.shape[1]`` is the component count. Nothing needs
to be declared -- the shape already says it.

    (1000, 3) field  <->  1000 tuples x 3 components
    (1000,)   field  <->  1000 tuples x 1 component

The one wrinkle is that Tack's own visualization algorithms work on *flat
interleaved* arrays -- ``flying_edges`` returns points as ``(n*3,)``, and
``compute_normals`` indexes ``points[i * 3 + c]``. For those, ask for the
flat form explicitly:

    points = vtk_to_field(vtk_points, flatten=True)   # (n*3,)
    array  = field_to_vtk(points, n_components=3)     # back to n x 3

Requirements
------------
VTK with ``vtkmodules.util.dlpack_support``. Device arrays additionally
need VTK built with Viskores; host arrays work with any VTK that has the
DLPack module.
"""

import tack

__all__ = ["field_to_vtk", "vtk_to_field"]


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


def vtk_to_field(vtk_array, flatten=False):
    """Wrap a vtkDataArray as a Tack field, without copying.

    Args:
        vtk_array: any vtkDataArray, on the host or on a device.
        flatten: return a 1-D field of ``tuples * components`` values
            instead of a 2-D one. Tack's visualization algorithms take
            their point and vector inputs in this interleaved form.

    Returns:
        A tack.field sharing the array's memory, shaped
        ``(tuples, components)`` -- or ``(tuples,)`` when there is one
        component, since a scalar array is naturally 1-D. The VTK array is
        held for as long as the field lives, so it may be dropped
        immediately.

    Raises:
        RuntimeError: if the array lives somewhere the active Tack backend
            cannot address -- a CUDA array under the CPU backend, say.
        ValueError: for an array with no memory to share (implicit and
            computed arrays have none) or a non-contiguous layout.
    """
    dlpack_support = _dlpack_support()

    # VTK always exports 2-D, (tuples, components).
    field = tack.from_dlpack(dlpack_support.vtk_to_dlpack(vtk_array))

    if flatten:
        return field.reshape((field.size,))
    if len(field.shape) == 2 and field.shape[1] == 1:
        return field.reshape((field.shape[0],))
    return field


def field_to_vtk(field, n_components=None, name=None):
    """Wrap a Tack field as a vtkDataArray, without copying.

    Args:
        field: the tack.field to share. A 2-D field maps straight over --
            ``(1000, 3)`` becomes 1000 tuples of 3.
        n_components: only needed for a *flat* field holding interleaved
            data, which is the form Tack's own algorithms produce. A
            ``(3000,)`` field with n_components=3 becomes 1000 tuples of 3.
            Leave it unset for a 2-D field, whose shape already says.
        name: array name, as VTK uses to identify it in a dataset.

    Returns:
        A vtkDataArray sharing the field's memory. The field is held for as
        long as the array lives.

    Raises:
        ValueError: if n_components contradicts the field's shape, does not
            divide its size, or the field has more than 2 dimensions.
    """
    # Work out the layout before reaching for VTK, so a shape mistake
    # reports itself rather than being masked by a missing-VTK error.
    ndim = len(field.shape)

    if n_components is None:
        if ndim == 2:
            n_components = field.shape[1]
        elif ndim <= 1:
            n_components = 1
        else:
            raise ValueError(
                f"a {ndim}-dimensional field has no obvious VTK layout; "
                f"reshape it to (tuples, components), or pass n_components "
                f"if it holds flat interleaved data")
    else:
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        if ndim == 2 and field.shape[1] != n_components:
            raise ValueError(
                f"n_components={n_components} contradicts the field's shape "
                f"{field.shape}, which already says {field.shape[1]}")
        if field.size % n_components:
            raise ValueError(
                f"a field of {field.size} values does not divide into tuples "
                f"of {n_components}")

    dlpack_support = _dlpack_support()

    # VTK reads the tensor as (tuples, components), so give it that shape.
    shaped = field
    if field.shape != (field.size // n_components, n_components):
        shaped = field.reshape((field.size // n_components, n_components))

    return dlpack_support.dlpack_to_vtk(shaped, name=name)
