"""tack.interop.vtk — Zero-copy interop between VTK arrays and Tack arrays.

    from tack.interop.vtk import vtk_to_array, array_to_vtk

    field = vtk_to_array(vtk_array)   # VTK  -> Tack
    array = array_to_vtk(field)       # Tack -> VTK

Both directions share memory rather than copying, on the host and on the
device, and each side keeps the other alive for as long as it is needed.

Shape
-----
A vtkDataArray is *tuples x components*, and a Tack array is n-dimensional,
so the two line up directly: a 2-D array of shape ``(n, 3)`` is n tuples of
3 components, and ``field.shape[1]`` is the component count. Nothing needs
to be declared -- the shape already says it.

    (1000, 3) field  <->  1000 tuples x 3 components
    (1000,)   field  <->  1000 tuples x 1 component

The one wrinkle is that Tack's own visualization algorithms work on *flat
interleaved* arrays -- ``flying_edges`` returns points as ``(n*3,)``, and
``compute_normals`` indexes ``points[i * 3 + c]``. For those, ask for the
flat form explicitly:

    points = vtk_to_array(vtk_points, flatten=True)   # (n*3,)
    array  = array_to_vtk(points, n_components=3)     # back to n x 3

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
    init_level_zero()        # instead of tack.set_default_device("level_zero")

That starts Tack's Level Zero backend inside the context VTK's Viskores
device already uses. It needs VTK built with Viskores on Kokkos' SYCL
backend. Fields made before the call belong to another context and cannot
be shared.
"""

import tack

__all__ = ["array_to_vtk", "dataset_to_vtk", "init_level_zero", "vtk_to_array", "vtk_to_dataset"]


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
    """Make a Level Zero device in the context VTK's device memory uses the
    default device, and return it.

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
    return tack.set_default_device("level_zero", external_context=handles)


def _require_shared_level_zero_context(dlpack_support, backend=None):
    """Refuse to exchange Level Zero memory across contexts.

    Nothing downstream can catch this. The driver answers for a pointer from
    any context, so VTK's own check passes, and on some drivers the memory
    even reads correctly -- which makes it a mistake that works until it
    does not. Tack knows which context it allocates in, so it is checked
    here.
    """
    if backend is None:
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
            "instead of tack.set_default_device('level_zero').")


def vtk_to_array(vtk_array, flatten=False):
    """Wrap a vtkDataArray as a Tack array, without copying.

    Args:
        vtk_array: any vtkDataArray, on the host or on a device.
        flatten: return a 1-D array of ``tuples * components`` values
            instead of a 2-D one. Tack's visualization algorithms take
            their point and vector inputs in this interleaved form.

    Returns:
        A Tack array sharing the VTK array's memory, shaped
        ``(tuples, components)`` -- or ``(tuples,)`` when there is one
        component, since a scalar array is naturally 1-D. The VTK array is
        held for as long as the array lives, so it may be dropped
        immediately.

    The array is on the default device when that can address the memory,
    else on the memory's own device: a CUDA array with the CPU as default is
    a CUDA array.

    Raises:
        RuntimeError: for Level Zero memory from another context than
            Tack's (see `init_level_zero`).
        ValueError: for an array with no memory to share (implicit and
            computed arrays have none) or a non-contiguous layout.
    """
    dlpack_support = _dlpack_support()

    # VTK always exports 2-D, (tuples, components). The array lands on the
    # default device if that can wrap the memory, else on the memory's own
    # device, whose context is the one to check.
    array = tack.from_dlpack(dlpack_support.vtk_to_dlpack(vtk_array))
    _require_shared_level_zero_context(dlpack_support, array._backend)

    if flatten:
        return array.reshape((array.size,))
    if len(array.shape) == 2 and array.shape[1] == 1:
        return array.reshape((array.shape[0],))
    return array


def array_to_vtk(array, n_components=None, name=None):
    """Wrap a Tack array as a vtkDataArray, without copying.

    Args:
        array: the Tack array to share. A 2-D array maps straight over --
            ``(1000, 3)`` becomes 1000 tuples of 3.
        n_components: only needed for a *flat* array holding interleaved
            data, which is the form Tack's own algorithms produce. A
            ``(3000,)`` array with n_components=3 becomes 1000 tuples of 3.
            Leave it unset for a 2-D array, whose shape already says.
        name: array name, as VTK uses to identify it in a dataset.

    Returns:
        A vtkDataArray sharing the array's memory. The array is held for as
        long as the array lives.

    Raises:
        ValueError: if n_components contradicts the array's shape, does not
            divide its size, or the array has more than 2 dimensions.
    """
    # Work out the layout before reaching for VTK, so a shape mistake
    # reports itself rather than being masked by a missing-VTK error.
    if getattr(array, 'layout', 'aos') == 'soa':
        raise TypeError(
            "array_to_vtk() shares one buffer with VTK, and an SOA array has one per "
            "component; share each of its components, or copy it to one buffer with "
            "tack.array(a, layout='aos')")
    if hasattr(array, '_vector_n'):
        width = array._vector_n
        if n_components is not None and n_components != width:
            raise ValueError(
                f"n_components={n_components} contradicts the element width {width}")
        if len(array.shape) > 1:
            raise ValueError("reshape a typed array to one array dimension for VTK tuples")
        array = array._scalar_view().reshape((array.size, width))
        n_components = width
    ndim = len(array.shape)

    if n_components is None:
        if ndim == 2:
            n_components = array.shape[1]
        elif ndim <= 1:
            n_components = 1
        else:
            raise ValueError(
                f"a {ndim}-dimensional array has no obvious VTK layout; "
                f"reshape it to (tuples, components), or pass n_components "
                f"if it holds flat interleaved data")
    else:
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        if ndim == 2 and array.shape[1] != n_components:
            raise ValueError(
                f"n_components={n_components} contradicts the array's shape "
                f"{array.shape}, which already says {array.shape[1]}")
        if array.size % n_components:
            raise ValueError(
                f"an array of {array.size} values does not divide into tuples "
                f"of {n_components}")

    dlpack_support = _dlpack_support()
    _require_shared_level_zero_context(dlpack_support, array._backend)

    # VTK reads the tensor as (tuples, components), so give it that shape.
    shaped = array
    if array.shape != (array.size // n_components, n_components):
        shaped = array.reshape((array.size // n_components, n_components))

    return dlpack_support.dlpack_to_vtk(shaped, name=name)


# ── Datasets ────────────────────────────────────────────────────────
#
# Through VTK's pythonic API (VTK 9.7.1 or later): properties for the grid's
# parameters, ``points``, ``cells``, ``cell_types`` and dict-like
# ``point_data``/``cell_data``, numpy-compatible arrays, and implicit arrays
# both ways -- VTK's constant and affine arrays are Tack's ``ConstantArray``
# and ``CountingArray``. Values are copied (the DLPack path above shares
# them). Point data becomes ``H1`` fields and cell data ``Constant`` fields
# (``docs/design/dataset-api.md`` §5).

def _from_vtk(array, dtype=None):
    """A VTK array as Tack takes it: a one-component constant array as a
    ``ConstantArray``, an affine one as a ``CountingArray`` (computed where read,
    nothing copied), anything else as a NumPy array of its values."""
    import numpy as np
    from vtkmodules.numpy_interface.vtk_affine_array import VTKAffineArray
    from vtkmodules.numpy_interface.vtk_constant_array import VTKConstantArray

    from tack.arrays import ConstantArray, CountingArray

    n = array.number_of_tuples
    if array.number_of_components == 1:
        if isinstance(array, VTKConstantArray):
            return ConstantArray(array.value, n, dtype=dtype)
        if isinstance(array, VTKAffineArray):
            return CountingArray(n, start=array.intercept, step=array.slope, dtype=dtype)
    return np.asarray(array)


def _array_to_field(array, float_dtype):
    """A VTK data array's values as a field's: floating point in ``float_dtype``,
    integers in their own type; a constant or affine array stays implicit."""
    import numpy as np

    from tack.arrays import ConstantArray, CountingArray
    from tack.lang.types import from_numpy_dtype

    host = np.asarray(array)
    if host.dtype.kind == "S":
        # A vtkCharArray has no numpy-compatible wrapper in 9.7.1, and NumPy reads
        # it as bytes; its values are signed 8-bit integers, as vtk_to_numpy gave.
        host = host.view(np.int8)
    dtype = float_dtype if host.dtype.kind == "f" else from_numpy_dtype(host.dtype)
    values = _from_vtk(array, dtype)
    if isinstance(values, (ConstantArray, CountingArray)):
        return values
    values = np.ascontiguousarray(host)
    n = array.number_of_components
    field = tack.empty((array.number_of_tuples,), dtype if n == 1 else tack.vec(n, dtype))
    if values.size:
        tack.copyto(field, values.reshape((-1,) if n == 1 else (-1, n)).astype(dtype.numpy_dtype))
    return field


def _attributes_to_fields(attributes, space, float_dtype):
    from vtkmodules.vtkCommonCore import vtkDataArray

    from tack.dataset import Field

    fields = {}
    for name in attributes.keys():
        array = attributes[name]
        # Named data arrays: string and other abstract arrays have no values to compute on.
        if name and isinstance(array, vtkDataArray):
            fields[name] = Field(space, _array_to_field(array, float_dtype))
    return fields


def _polyhedral_topology(grid, orient=False, id_dtype=None):
    """A ``vtkUnstructuredGrid`` as a ``PolyhedralTopology``, every cell a polyhedron.

    A polyhedron's faces are its own (``polyhedron_faces``/``_face_locations``);
    any other cell's are its cell class's, outward as VTK winds them (a voxel's
    pixels as quads). VTK keeps a face once per cell, each copy outward for its
    cell, so copies are matched by their point sets: the first found is the face,
    its cell side 0, and a second copy -- which must go round the other way -- is
    side 1. Read on the host, cell by cell.
    """
    import numpy as np

    from tack.dataset import PolyhedralTopology

    types = np.asarray(grid.cell_types)
    locations, faces = grid.polyhedron_face_locations, grid.polyhedron_faces
    if locations is not None and faces is not None:
        loc_offsets = np.asarray(locations.offsets_array)
        loc_faces = np.asarray(locations.connectivity_array)
        face_offsets = np.asarray(faces.offsets_array)
        face_conn = np.asarray(faces.connectivity_array)
    cells = []
    for c in range(grid.number_of_cells):
        if types[c] == 42:
            listed = [face_conn[face_offsets[f]:face_offsets[f + 1]]
                      for f in loc_faces[loc_offsets[c]:loc_offsets[c + 1]]]
        else:
            cell = grid.GetCell(c)                     # no pythonic form in 9.7.1
            if cell.cell_dimension != 3:
                raise ValueError(f"cell {c} is not 3D: a polyhedral topology is of 3D cells")
            listed = []
            for k in range(cell.number_of_faces):
                face = cell.GetFace(k)
                ids = [face.GetPointId(i) for i in range(face.number_of_points)]
                if face.cell_type == 8:                        # a pixel, as a quad
                    ids = [ids[0], ids[1], ids[3], ids[2]]
                listed.append(ids)
        cells.append(listed)
    positions = _vtk_points(grid) if orient else None
    try:
        return PolyhedralTopology.from_cell_faces(cells, grid.number_of_points,
                                                  positions=positions, orient=orient,
                                                  id_dtype=id_dtype)
    except ValueError as error:
        if "wound the same way" in str(error) and not orient:
            raise ValueError(f"{error}. VTK winds every polyhedron face outward for its "
                             "cell; vtk_to_dataset(..., orient=True) repairs the input") from None
        raise


def vtk_to_dataset(grid, dtype=tack.f32, polyhedral=None, orient=False, id_dtype=None):
    """Copy a ``vtkUnstructuredGrid``, ``vtkStructuredGrid``, ``vtkRectilinearGrid`` or
    ``vtkImageData`` into a ``tack.dataset.DataSet``: an image's points as
    ``UniformCoordinates`` (with its direction matrix), a rectilinear grid's as a
    ``CartesianProduct``, a structured grid's as its explicit points.

    Points and floating-point arrays become fields of ``dtype``; integer
    arrays keep their type. Named point data become ``H1`` fields and named
    cell data ``Constant`` fields; a constant or affine array (VTK's implicit
    arrays) stays implicit, as a ``ConstantArray`` or ``CountingArray``. An
    unstructured grid's cells must all be linear shapes -- unless it holds
    polyhedra (or ``polyhedral`` is true), when it becomes a
    ``PolyhedralTopology``, every cell a polyhedron of its faces, polylines,
    when it becomes a ``PolylineTopology`` (its lines two-point polylines, and
    no other cells allowed), or polygons (``VTK_POLYGON``), when it becomes a
    ``PolygonalTopology`` (its triangles and quads polygons too, and no other
    cells allowed). Its types, offsets and connectivity stay implicit where
    VTK's are (a cell type given as a number, affine offsets).
    With ``orient``, inconsistently wound polyhedra are repaired first
    (``PolyhedralTopology.from_cell_faces``); without it they are refused.
    VTK's ids are 64-bit; the topology keeps them in ``id_dtype``, by default
    ``i32`` unless the grid is too large for it (``tack.dataset.ids``).

    A ``vtkPartitionedDataSet`` becomes a ``PartitionedDataSet``, each piece
    converted as above (null pieces skipped), and a
    ``vtkPartitionedDataSetCollection`` a ``dict`` of them, keyed by each
    partitioned dataset's ``vtkCompositeDataSet.NAME()`` (``"block<i>"`` where
    it has none, ``"<name>#<i>"`` for a name already taken).
    """
    import numpy as np
    from vtkmodules.vtkCommonDataModel import (
        vtkImageData,
        vtkPartitionedDataSet,
        vtkPartitionedDataSetCollection,
        vtkRectilinearGrid,
        vtkStructuredGrid,
        vtkUnstructuredGrid,
    )

    from tack.arrays import CartesianProduct, ConstantArray, UniformCoordinates
    from tack.dataset import (
        H1,
        Constant,
        DataSet,
        PartitionedDataSet,
        PolygonalTopology,
        PolylineTopology,
        UnstructuredTopology,
    )
    from tack.dataset.dataset import grid_topology

    options = {"dtype": dtype, "polyhedral": polyhedral, "orient": orient, "id_dtype": id_dtype}
    if isinstance(grid, vtkPartitionedDataSetCollection):
        return _collection_to_datasets(grid, options)
    if isinstance(grid, vtkPartitionedDataSet):
        # Not iterated: that walks the leaves and skips empty partitions.
        pieces = (grid.GetPartition(i) for i in range(grid.number_of_partitions))
        return PartitionedDataSet(vtk_to_dataset(piece, **options)
                                  for piece in pieces if piece is not None)
    if isinstance(grid, vtkUnstructuredGrid):
        # VTK's constant and affine arrays come across as Tack's implicit ones.
        types = _from_vtk(grid.cell_types, tack.u8)
        present = ({int(types.value.numpy().reshape(-1)[0])} if isinstance(types, ConstantArray)
                   else {int(t) for t in np.unique(types)})
        if polyhedral is None:
            polyhedral = 42 in present
    if isinstance(grid, vtkUnstructuredGrid) and polyhedral:
        topology = _polyhedral_topology(grid, orient=orient, id_dtype=id_dtype)
        geometry = _vtk_points(grid)
    elif isinstance(grid, vtkUnstructuredGrid):
        cells = grid.cells
        offsets = _from_vtk(cells.offsets_array)
        connectivity = _from_vtk(cells.connectivity_array)
        num_points = grid.number_of_points
        if 4 in present:
            # Polylines: their lines are two-point chains; nothing else mixes in.
            if not present <= {3, 4}:
                raise ValueError("a grid with polylines (VTK_POLY_LINE) may hold only lines "
                                 "and polylines")
            topology = PolylineTopology(np.asarray(offsets), np.asarray(connectivity),
                                        num_points=num_points, id_dtype=id_dtype)
        elif 7 in present:
            # Polygons, as dataset_to_vtk writes a PolygonalTopology: triangles
            # and quads are polygons too; nothing else mixes in.
            if not present <= {5, 7, 9}:
                raise ValueError("a grid with polygons (VTK_POLYGON) may hold only "
                                 "triangles, quads and polygons")
            topology = PolygonalTopology(np.asarray(offsets), np.asarray(connectivity),
                                         num_points=num_points, id_dtype=id_dtype)
        else:
            topology = UnstructuredTopology(types, offsets, connectivity, num_points=num_points,
                                            id_dtype=id_dtype)
        geometry = _vtk_points(grid)
    elif isinstance(grid, vtkImageData):
        dims = list(grid.dimensions)
        matrix = grid.direction_matrix              # a vtkMatrix3x3: no indexing in 9.7.1
        direction = np.array([[matrix.GetElement(r, c) for c in range(3)] for r in range(3)])
        spacing = np.asarray(grid.spacing)
        # Points are origin + direction @ (ijk * spacing), ijk from the extent's start.
        origin = np.asarray(grid.origin) + direction @ (np.asarray(grid.extent[0::2]) * spacing)
        geometry = UniformCoordinates(dims, origin, spacing, direction, dtype=dtype)
        topology = grid_topology(dims, id_dtype=id_dtype)
    elif isinstance(grid, vtkRectilinearGrid):
        geometry = CartesianProduct(np.asarray(grid.x_coordinates),
                                    np.asarray(grid.y_coordinates),
                                    np.asarray(grid.z_coordinates), dtype=dtype)
        dims = list(grid.dimensions)
        if dims != list(geometry.dims):
            raise ValueError(f"grid dimensions {dims} do not match its coordinates")
        topology = grid_topology(dims, id_dtype=id_dtype)
    elif isinstance(grid, vtkStructuredGrid):
        dims = [0, 0, 0]
        grid.GetDimensions(dims)        # not a readable property on a structured grid in 9.7.1
        topology = grid_topology(dims, id_dtype=id_dtype)
        geometry = _vtk_points(grid)
    else:
        raise TypeError(f"expected a vtkUnstructuredGrid, vtkStructuredGrid, "
                        f"vtkRectilinearGrid, vtkImageData, vtkPartitionedDataSet or "
                        f"vtkPartitionedDataSetCollection, not {grid.class_name}")
    fields = _attributes_to_fields(grid.point_data, H1(topology), dtype)
    fields.update(_attributes_to_fields(grid.cell_data, Constant(topology), dtype))
    return DataSet(topology, geometry, fields=fields, dtype=dtype)


def _to_vtk_types(types):
    """Cell types for ``vtkUnstructuredGrid.cells``: a ``ConstantArray``'s value as
    a number (VTK keeps it as a vtkConstantArray), else a vtkUnsignedCharArray."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray

    from tack.arrays import ConstantArray

    if isinstance(types, ConstantArray):
        return int(types.value.numpy().reshape(-1)[0])
    return vtkUnsignedCharArray(np.ascontiguousarray(np.asarray(types), np.uint8))


def _to_vtk_ids(values):
    """Offsets or connectivity for ``vtkCellArray``: a ``CountingArray`` as a
    vtkAffineArray, anything else as int64 values."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkAffineArray

    from tack.arrays import CountingArray

    if isinstance(values, CountingArray):
        return vtkAffineArray[np.int64](values.size, values.step, values.start)
    return np.ascontiguousarray(np.asarray(values), np.int64)


def _collection_to_datasets(collection, options):
    """A ``vtkPartitionedDataSetCollection`` as a ``dict`` of name to
    ``PartitionedDataSet``."""
    from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet

    from tack.dataset import PartitionedDataSet

    out = {}
    for i in range(collection.number_of_partitioned_data_sets):
        name = None
        if collection.HasMetaData(i):
            meta = collection.GetMetaData(i)
            if meta.Has(vtkCompositeDataSet.NAME()):
                name = meta.Get(vtkCompositeDataSet.NAME())
        name = name or f"block{i}"
        if name in out:
            name = f"{name}#{i}"
        pieces = collection.GetPartitionedDataSet(i)
        out[name] = (vtk_to_dataset(pieces, **options) if pieces is not None
                     else PartitionedDataSet())
    return out


def _vtk_points(grid):
    """A point set's positions; a grid with no vtkPoints at all (an empty one, as
    VTK makes it) has none."""
    import numpy as np

    points = grid.points
    return np.zeros((0, 3)) if points is None else np.asarray(points.data)


def _vtk_points_of(data):
    """``data``'s positions as vtkPoints of the geometry's own precision:
    ``vtkPoints(data=...)`` given a NumPy array stores doubles, so give it a
    typed array (f32 positions as a vtkFloatArray)."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkFloatArray, vtkPoints

    positions = data.positions()
    typed = vtkFloatArray if positions.dtype == np.float32 else vtkDoubleArray
    return vtkPoints(data=typed(np.ascontiguousarray(positions)))


def _to_vtk_values(values):
    """A field's values for ``point_data``/``cell_data``: a scalar ``ConstantArray``
    as a vtkConstantArray and a ``CountingArray`` as a vtkAffineArray (nothing
    stored), anything else copied to the host."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkAffineArray, vtkConstantArray

    from tack.arrays import ConstantArray, CountingArray, to_host

    if isinstance(values, ConstantArray) and values.width is None:
        dtype = values.dtype.numpy_dtype
        return vtkConstantArray[dtype](values.size, values.value.numpy().reshape(-1)[0])
    if isinstance(values, CountingArray):
        return vtkAffineArray[values.dtype.numpy_dtype](values.size, values.step, values.start)
    return np.array(to_host(values))            # a copy: the grid owns its values


def dataset_to_vtk(data):
    """Copy a ``tack.dataset.DataSet`` into a new VTK dataset; a ``PartitionedDataSet``
    into a ``vtkPartitionedDataSet`` of its pieces, and a ``dict`` of names to
    ``PartitionedDataSet``s into a ``vtkPartitionedDataSetCollection`` whose
    partitioned datasets carry their names (``vtkCompositeDataSet.NAME()``).

    A structured topology over uniform coordinates gives a ``vtkImageData`` (with
    the direction matrix), over rectilinear ones a ``vtkRectilinearGrid``, over
    explicit ones (a warped or transformed grid) a ``vtkStructuredGrid``; an
    unstructured one a ``vtkUnstructuredGrid``, a polygonal or polyline one a
    ``vtkUnstructuredGrid`` of ``VTK_POLYGON`` or ``VTK_POLY_LINE`` cells, and a
    polyhedral one a ``vtkUnstructuredGrid`` of ``VTK_POLYHEDRON`` cells, each
    keeping its own copy of its faces, outward for it (VTK's layout has no
    orientation bit, so a shared face cannot be stored once).
    ``H1`` fields and values on points become point data -- an order-2 field
    its values at the points, the grid being linear -- and ``Constant`` fields
    and values on cells cell data. ``L2`` fields have no VTK array to go
    to without duplicating points, and fields on faces or edges none at
    all; they are left out. The geometry (``data.geometry``) becomes the
    points, so it must be ``H1``. Implicit types, offsets and connectivity,
    and constant and counting field values, become VTK's constant and affine
    arrays.
    """
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkAffineArray, vtkUnsignedCharArray
    from vtkmodules.vtkCommonDataModel import (
        vtkCellArray,
        vtkImageData,
        vtkRectilinearGrid,
        vtkStructuredGrid,
        vtkUnstructuredGrid,
    )

    from tack.arrays import CartesianProduct, UniformCoordinates
    from tack.dataset import (
        H1,
        Constant,
        PartitionedDataSet,
        PolygonalTopology,
        PolyhedralTopology,
        PolylineTopology,
        StructuredTopology,
        UnstructuredTopology,
        Values,
    )

    if isinstance(data, dict):
        return _datasets_to_collection(data)
    if isinstance(data, PartitionedDataSet):
        from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet

        out = vtkPartitionedDataSet()
        out.SetNumberOfPartitions(len(data))
        for i, piece in enumerate(data):
            out.SetPartition(i, dataset_to_vtk(piece))
        return out
    topology = data.topology
    coordinates = data.geometry.values
    # A grid of one point or none has a topology without cells (grid_topology),
    # but is still the image or rectilinear grid its coordinates describe.
    grid = isinstance(topology, StructuredTopology) or (
        topology.num_cells == 0 and isinstance(coordinates, (UniformCoordinates,
                                                             CartesianProduct)))
    if grid and isinstance(coordinates, UniformCoordinates):
        grid = vtkImageData(dimensions=coordinates.dims, origin=coordinates._origin,
                            spacing=coordinates._spacing,
                            direction_matrix=tuple(coordinates._matrix.reshape(-1)))
    elif grid and isinstance(coordinates, CartesianProduct):
        grid = vtkRectilinearGrid(dimensions=coordinates.dims)
        grid.x_coordinates = np.array(coordinates.x.numpy())
        grid.y_coordinates = np.array(coordinates.y.numpy())
        grid.z_coordinates = np.array(coordinates.z.numpy())
    elif isinstance(topology, StructuredTopology):
        # Explicit coordinates, as a warp or transform leaves; both number the
        # points x fastest.
        grid = vtkStructuredGrid(dimensions=(*topology.point_dims, 1, 1)[:3])
        grid.points = _vtk_points_of(data)
    elif isinstance(topology, UnstructuredTopology):
        # Implicit types, offsets and connectivity go across as VTK's constant
        # and affine arrays: a cell type given as a number is a vtkConstantArray.
        grid = vtkUnstructuredGrid()
        grid.points = _vtk_points_of(data)
        grid.cells = (_to_vtk_types(topology.types),
                      vtkCellArray(offsets=_to_vtk_ids(topology.offsets),
                                   connectivity=_to_vtk_ids(topology.connectivity)))
    elif isinstance(topology, (PolygonalTopology, PolylineTopology)):
        polygons = isinstance(topology, PolygonalTopology)
        offsets, loops = (a.numpy() for a in (topology.loops() if polygons
                                                  else topology.lines()))
        grid = vtkUnstructuredGrid()
        grid.points = _vtk_points_of(data)
        grid.cells = (7 if polygons else 4,
                      vtkCellArray(offsets=offsets.astype(np.int64),
                                   connectivity=loops.astype(np.int64)))
    elif isinstance(topology, PolyhedralTopology):
        # VTK keeps a face per cell, outward for that cell: side 1's copy is
        # the face walked backwards from the same first point.
        face_offsets, face_points, cell_offsets, cell_faces, sides = topology.arrays()
        point_offsets, point_ids = (a.numpy() for a in topology.cell_points())
        copies, copy_offsets = [], [0]
        for f, side in zip(cell_faces, sides):
            ring = face_points[face_offsets[f]:face_offsets[f + 1]]
            copies.extend(ring if side == 0 else np.roll(ring[::-1], 1))
            copy_offsets.append(len(copies))
        grid = vtkUnstructuredGrid()
        grid.points = _vtk_points_of(data)
        # No pythonic form for polyhedral cells in 9.7.1. Each cell lists its own
        # face copies in order, so the face locations' connectivity is a ramp.
        grid.SetPolyhedralCells(
            vtkUnsignedCharArray(np.full(topology.num_cells, 42, np.uint8)),
            vtkCellArray(offsets=np.asarray(point_offsets, np.int64),
                         connectivity=np.asarray(point_ids, np.int64)),
            vtkCellArray(offsets=np.asarray(cell_offsets, np.int64),
                         connectivity=vtkAffineArray[np.int64](len(cell_faces), 1, 0)),
            vtkCellArray(offsets=np.asarray(copy_offsets, np.int64),
                         connectivity=np.asarray(copies, np.int64)))
    else:
        raise TypeError(f"unsupported topology {type(topology).__name__} with "
                        f"{type(coordinates).__name__}")
    from tack.dataset.carry import _point_values

    for name, field in data.fields.items():
        space = field.space
        if isinstance(space, H1) or (isinstance(space, Values) and space.on == "points"):
            # An order-2 field's values at the points, its first ones: the grid
            # is linear, as the filters read order-2 data.
            grid.point_data[name] = _to_vtk_values(_point_values(field.values, data.num_points))
        elif isinstance(space, Constant) or (isinstance(space, Values) and space.on == "cells"):
            grid.cell_data[name] = _to_vtk_values(field.values)
    return grid


def _datasets_to_collection(datasets):
    """A ``dict`` of name to ``PartitionedDataSet`` as a
    ``vtkPartitionedDataSetCollection``, each partitioned dataset named."""
    from vtkmodules.vtkCommonDataModel import (
        vtkCompositeDataSet,
        vtkPartitionedDataSetCollection,
    )

    from tack.dataset import PartitionedDataSet

    out = vtkPartitionedDataSetCollection()
    out.SetNumberOfPartitionedDataSets(len(datasets))
    for i, (name, pieces) in enumerate(datasets.items()):
        if not isinstance(pieces, PartitionedDataSet):
            raise TypeError(f"collection entry {name!r} is a {type(pieces).__name__}, not "
                            "a PartitionedDataSet")
        out.SetPartitionedDataSet(i, dataset_to_vtk(pieces))
        out.GetMetaData(i).Set(vtkCompositeDataSet.NAME(), str(name))
    return out
