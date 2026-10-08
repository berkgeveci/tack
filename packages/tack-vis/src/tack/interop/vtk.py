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

Level Zero
----------
CUDA and HIP pointers identify themselves: the runtime keeps one context per
device for the whole process. A Level Zero pointer does not -- it means
something only inside the context it was allocated from, and DLPack has no
field for a context. So the two libraries have to be in the same one before
any array is exchanged:

    from tack.interop.vtk import init_level_zero
    init_level_zero()        # instead of tack.init(arch=tack.level_zero)

That starts Tack's Level Zero backend inside the context VTK's Viskores
device already uses. It needs VTK built with Viskores on Kokkos' SYCL
backend. Fields made before the call belong to another context and cannot
be shared.
"""

import tack

__all__ = ["dataset_to_vtk", "field_to_vtk", "init_level_zero", "vtk_to_dataset", "vtk_to_field"]


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


def init_level_zero():
    """Start Tack's Level Zero backend in the context VTK's device memory uses.

    Raises RuntimeError if this VTK cannot report one: it predates
    ``dlpack_support.level_zero_handles``, was built without Viskores on
    Kokkos' SYCL backend, or is not running on a Level Zero device.
    """
    dlpack_support = _dlpack_support()
    handles = getattr(dlpack_support, "level_zero_handles", lambda: None)()
    if not handles:
        raise RuntimeError(
            "this VTK has no Level Zero context to share. Device interop on "
            "Level Zero needs VTK built with Viskores on Kokkos' SYCL "
            "backend, running on a Level Zero device.")
    tack.init(arch=tack.level_zero, external_context=handles)


def _require_shared_level_zero_context(dlpack_support):
    """Refuse to exchange Level Zero memory across contexts.

    Nothing downstream can catch this. The driver answers for a pointer from
    any context, so VTK's own check passes, and on some drivers the memory
    even reads correctly -- which makes it a mistake that works until it
    does not. Tack knows which context it allocates in, so it is checked
    here.
    """
    from tack.runtime.dispatch import get_backend

    backend = get_backend()
    if backend.name != "level_zero":
        return
    handles = getattr(dlpack_support, "level_zero_handles", lambda: None)()
    if not handles or backend._context.value != handles["context"]:
        raise RuntimeError(
            "Tack's Level Zero backend and VTK are not in the same Level "
            "Zero context, so a device pointer from one means nothing to "
            "the other. Start Tack with tack.interop.vtk.init_level_zero() "
            "instead of tack.init(arch=tack.level_zero).")


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
    _require_shared_level_zero_context(dlpack_support)

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
    _require_shared_level_zero_context(dlpack_support)

    # VTK reads the tensor as (tuples, components), so give it that shape.
    shaped = field
    if field.shape != (field.size // n_components, n_components):
        shaped = field.reshape((field.size // n_components, n_components))

    return dlpack_support.dlpack_to_vtk(shaped, name=name)


# ── Datasets ────────────────────────────────────────────────────────
#
# Copied through NumPy, so these work with any VTK, with or without the
# DLPack module. Point data becomes ``H1`` fields and cell data
# ``Constant`` fields (``docs/design/dataset-api.md`` §5).

def _array_to_field(array, float_dtype):
    import numpy as np
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from tack.lang.types import from_numpy_dtype

    values = vtk_to_numpy(array)
    dtype = float_dtype if values.dtype.kind == "f" else from_numpy_dtype(values.dtype)
    n = array.GetNumberOfComponents()
    if n == 1:
        field = tack.field(dtype, shape=(values.shape[0],))
    else:
        field = tack.Vector.field(n, dtype, shape=(values.shape[0],))
    if values.size:
        field.from_numpy(np.ascontiguousarray(values).astype(dtype.numpy_dtype))
    return field


def _attributes_to_fields(attributes, space, float_dtype):
    from tack.data import Field

    fields = {}
    for i in range(attributes.GetNumberOfArrays()):
        array = attributes.GetArray(i)
        if array is not None and array.GetName():
            fields[array.GetName()] = Field(space, _array_to_field(array, float_dtype))
    return fields


def vtk_to_dataset(grid, dtype=tack.f32):
    """Copy a ``vtkUnstructuredGrid`` or ``vtkRectilinearGrid`` into a
    ``tack.data.DataSet``.

    Points and floating-point arrays become fields of ``dtype``; integer
    arrays keep their type. Named point data become ``H1`` fields and named
    cell data ``Constant`` fields. An unstructured grid's cells must all be
    linear shapes.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from tack.data import (
        H1,
        Constant,
        DataSet,
        RectilinearCoordinates,
        StructuredTopology,
        UnstructuredTopology,
    )

    if grid.IsA("vtkUnstructuredGrid"):
        cell_array = grid.GetCells()
        # The per-cell types are GetCellTypes() from VTK 9.6 on; through 9.5 that
        # takes an argument and the array is GetCellTypesArray() (removed on master).
        try:
            types = grid.GetCellTypes()
        except TypeError:
            types = grid.GetCellTypesArray()
        topology = UnstructuredTopology(vtk_to_numpy(types),
                                        vtk_to_numpy(cell_array.GetOffsetsArray()),
                                        vtk_to_numpy(cell_array.GetConnectivityArray()))
        geometry = vtk_to_numpy(grid.GetPoints().GetData())
    elif grid.IsA("vtkRectilinearGrid"):
        geometry = RectilinearCoordinates(vtk_to_numpy(grid.GetXCoordinates()),
                                          vtk_to_numpy(grid.GetYCoordinates()),
                                          vtk_to_numpy(grid.GetZCoordinates()), dtype=dtype)
        dims = [0, 0, 0]
        grid.GetDimensions(dims)
        if dims != list(geometry.dims):
            raise ValueError(f"grid dimensions {dims} do not match its coordinates")
        if 1 in geometry.point_dims:
            raise ValueError(f"a grid of dimensions {tuple(dims)} is not supported: "
                             "only trailing dimensions may be 1")
        topology = StructuredTopology(geometry.point_dims)
    else:
        raise TypeError(f"expected a vtkUnstructuredGrid or vtkRectilinearGrid, "
                        f"not {grid.GetClassName()}")
    fields = _attributes_to_fields(grid.GetPointData(), H1(), dtype)
    fields.update(_attributes_to_fields(grid.GetCellData(), Constant(), dtype))
    return DataSet(topology, geometry, fields=fields, dtype=dtype)


def _field_to_array(name, field):
    from vtkmodules.util.numpy_support import numpy_to_vtk

    values = field.to_numpy(vectors=True) if getattr(field, "_vector_n", None) else field.to_numpy()
    array = numpy_to_vtk(values, deep=1)
    array.SetName(name)
    return array


def dataset_to_vtk(data):
    """Copy a ``tack.data.DataSet`` into a new VTK dataset.

    A structured topology over rectilinear coordinates gives a
    ``vtkRectilinearGrid``; an unstructured one a ``vtkUnstructuredGrid``.
    ``H1`` fields and ``Values("points")`` become point data, ``Constant``
    and ``Values("cells")`` cell data. ``L2`` fields have no VTK array to go
    to without duplicating points, and fields on faces or edges none at
    all; they are left out. The geometry (the ``"shape"`` field) becomes the
    points, so it must be ``H1``.
    """
    import numpy as np
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import (
        vtkCellArray,
        vtkRectilinearGrid,
        vtkUnstructuredGrid,
    )

    from tack.data import (
        H1,
        Constant,
        RectilinearCoordinates,
        StructuredTopology,
        UnstructuredTopology,
        Values,
    )

    topology = data.topology
    coordinates = data.geometry.values
    if isinstance(topology, StructuredTopology) and isinstance(coordinates,
                                                               RectilinearCoordinates):
        grid = vtkRectilinearGrid()
        grid.SetDimensions(*coordinates.dims)
        grid.SetXCoordinates(numpy_to_vtk(coordinates.x.to_numpy(), deep=1))
        grid.SetYCoordinates(numpy_to_vtk(coordinates.y.to_numpy(), deep=1))
        grid.SetZCoordinates(numpy_to_vtk(coordinates.z.to_numpy(), deep=1))
    elif isinstance(topology, UnstructuredTopology):
        cell_array = vtkCellArray()
        cell_array.SetData(
            numpy_to_vtkIdTypeArray(topology.offsets.to_numpy().astype(np.int64), deep=1),
            numpy_to_vtkIdTypeArray(topology.connectivity.to_numpy().astype(np.int64), deep=1))
        grid = vtkUnstructuredGrid()
        grid.SetCells(numpy_to_vtk(topology.types.to_numpy().astype(np.uint8), deep=1),
                      cell_array)
        points = vtkPoints()
        points.SetData(numpy_to_vtk(data.positions(), deep=1))
        grid.SetPoints(points)
    else:
        raise TypeError(f"unsupported topology {type(topology).__name__} with "
                        f"{type(coordinates).__name__}")
    for name, field in data.fields.items():
        if name == "shape":
            continue                                  # the geometry: the points above
        if field.space in (H1(), Values("points")):
            grid.GetPointData().AddArray(_field_to_array(name, field.values))
        elif field.space in (Constant(), Values("cells")):
            grid.GetCellData().AddArray(_field_to_array(name, field.values))
    return grid
