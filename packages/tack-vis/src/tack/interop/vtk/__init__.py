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
# (``docs/vis-guide/design/dataset-api.md`` §5).

def _shared(host, element):
    """``host``, a NumPy view of VTK's memory, as a read-only CPU array of ``element``s
    (a scalar type, or a vector of one), sharing it: the array holds the view and
    the view holds VTK's array. ``None`` where it cannot be shared -- a layout
    that is not one contiguous block, or values of another type -- for the
    caller to copy."""
    import numpy as np

    scalar = element.scalar_dtype if hasattr(element, "scalar_dtype") else element
    if (not isinstance(host, np.ndarray) or not host.flags.c_contiguous
            or host.dtype != np.dtype(scalar.numpy_dtype)):
        return None
    from tack.runtime.dispatch import get_backend

    if get_backend().name == "metal":
        # Unified memory: a Metal array over VTK's, where Metal can wrap it in
        # place (a buffer Tack handed VTK, or memory on whole pages); else the
        # caller copies it to the device, so the dataset is on Metal throughout.
        try:
            return tack.array_from_ptr(host, (host.shape[0],), element, device="metal")
        except TypeError:
            return None
    return tack.array_from_ptr(host, (host.shape[0],), element, device="cpu")


class _Held:
    """A NumPy view of a Tack array's memory that holds the array: NumPy keeps this
    object as the view's base, so a VTK array over the view keeps the Tack array,
    and its buffer, alive."""

    def __init__(self, view, array):
        self.__array_interface__ = view.__array_interface__
        self._view, self._array = view, array


def _host_values(array):
    """A Tack array's values as a NumPy array: a view of its memory where the array
    is on the CPU (DLPack) or on Metal (unified memory: its buffer's host view,
    after the device is done with it), a copy otherwise. VTK arrays made from it
    share the view and hold it, and so the Tack array."""
    import numpy as np

    from tack.arrays import to_host

    if isinstance(array, tack.Array) and array.device.type == "cpu" and \
            getattr(array, "layout", "aos") == "aos":
        return np.from_dlpack(array)
    if _on_metal(array):
        view = array._buffer.host_view()
        width = getattr(array, "_vector_n", None)
        shape = (*array.shape, width) if width else tuple(array.shape)
        return np.asarray(_Held(view.reshape(shape), array))
    return np.array(to_host(array))


def _on_metal(values):
    """Whether ``values`` is a stored interleaved array on Metal, whose memory is
    host memory VTK can share."""
    return (isinstance(values, tack.Array) and values.device.type == "metal"
            and getattr(values, "layout", "aos") == "aos"
            and hasattr(getattr(values, "_buffer", None), "host_view"))


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


def _array_to_field(array, float_dtype, share=False):
    """A VTK data array's values as a field's: floating point in ``float_dtype`` (in
    its own type when that is ``None``), integers in their own type, bits as u8 0
    and 1; a constant or affine array stays implicit. With ``share``, VTK's memory
    where it can be (``_shared``)."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkBitArray

    from tack.arrays import ConstantArray, CountingArray
    from tack.lang.types import from_numpy_dtype

    if isinstance(array, vtkBitArray):
        # NumPy reads a vtkBitArray as its packed bytes, eight values to a byte.
        host = _bits(array).reshape(array.number_of_tuples, -1)
    else:
        host = np.asarray(array)
    if host.dtype.kind == "S":
        # A vtkCharArray has no numpy-compatible wrapper in 9.7.1, and NumPy reads
        # it as bytes; its values are signed 8-bit integers, as vtk_to_numpy gave.
        host = host.view(np.int8)
    dtype = (float_dtype if host.dtype.kind == "f" and float_dtype is not None
             else from_numpy_dtype(host.dtype))
    values = _from_vtk(array, dtype)
    if isinstance(values, (ConstantArray, CountingArray)):
        return values
    n = array.number_of_components
    if share:
        shared = _shared(host, dtype if n == 1 else tack.vec(n, dtype))
        if shared is not None:
            return shared
    values = np.ascontiguousarray(host)
    field = tack.empty((array.number_of_tuples,), dtype if n == 1 else tack.vec(n, dtype))
    if values.size:
        tack.copyto(field, values.reshape((-1,) if n == 1 else (-1, n)).astype(dtype.numpy_dtype))
    return field


def _units_label(array):
    """The unit a VTK array's ``UNITS_LABEL`` names, in unyt's spelling, or
    ``None``: no label, or one that is no unit (VTK's labels are free text;
    a label that does not parse is dropped with a warning, not guessed at)."""
    from vtkmodules.vtkCommonCore import vtkDataArray

    from tack.dataset import units

    if array is None or not isinstance(array, vtkDataArray):
        return None
    information = array.GetInformation()
    key = vtkDataArray.UNITS_LABEL()
    if not information.Has(key):
        return None
    label = information.Get(key)
    if not label or not label.strip():
        return None
    try:
        return units.canonical(label)
    except ValueError:
        import warnings

        warnings.warn(f"VTK array {array.GetName()!r} has UNITS_LABEL {label!r}, which is "
                      "no unit Tack reads: its unit is unknown", stacklevel=3)
        return None


def _label(array, unit):
    """Set a VTK array's ``UNITS_LABEL`` to ``unit``, if known; returns the array."""
    from vtkmodules.vtkCommonCore import vtkDataArray

    if unit is not None and array is not None:
        array.GetInformation().Set(vtkDataArray.UNITS_LABEL(), unit)
    return array


#: VTK's ghost array, ``vtkDataSetAttributes::GhostArrayName()``: it becomes
#: ``data.ghosts``, not a field.
_GHOST_ARRAY = "vtkGhostType"


def _attributes_to_fields(attributes, space, float_dtype, share=False):
    from vtkmodules.vtkCommonCore import vtkDataArray

    from tack.dataset import Field

    fields = {}
    global_ids = attributes.GetGlobalIds()          # no pythonic form in 9.7.1
    skipped = {_GHOST_ARRAY, global_ids.GetName() if global_ids is not None else None}
    for name in attributes.keys():
        array = attributes[name]
        # Named data arrays: string and other abstract arrays have no values to compute on,
        # the ghost array becomes the ghost marks and the global ids the global ids.
        if name and name not in skipped and isinstance(array, vtkDataArray):
            fields[name] = Field(space, _array_to_field(array, float_dtype, share),
                                 _units_label(array))
    return fields


def _global_ids(attributes, share=False):
    """An attributes object's ``GlobalIds`` array as i64 ids, or ``None``; with
    ``share``, VTK's memory where it is i64 already."""
    import numpy as np

    array = attributes.GetGlobalIds()               # no pythonic form in 9.7.1
    if array is None:
        return None
    host = np.asarray(array).reshape(-1)
    shared = _shared(host, tack.i64) if share else None
    if shared is not None:
        return shared
    return tack.array(np.ascontiguousarray(host, np.int64), tack.i64)


def _ghost_marks(attributes, share=False):
    """An attributes object's ``vtkGhostType`` array as u8 marks, or ``None``; with
    ``share``, VTK's memory."""
    import numpy as np

    if _GHOST_ARRAY not in attributes.keys():
        return None
    host = np.asarray(attributes[_GHOST_ARRAY]).reshape(-1)
    shared = _shared(host, tack.u8) if share else None
    if shared is not None:
        return shared
    return tack.array(np.ascontiguousarray(host, np.uint8), tack.u8)


#: Per VTK 3D shape: three corners whose edges from corner 0 span a positive frame in
#: a cell wound as VTK winds it, and the corner order that turns the cell inside out
#: (its own inverse), checked against VTK's cells (``test_vtk_interop``).
_ORIENTATION = {10: ((1, 2, 3), (0, 2, 1, 3)),                      # tetrahedron
                12: ((1, 3, 4), (0, 3, 2, 1, 4, 7, 6, 5)),          # hexahedron
                13: ((1, 2, 3), (0, 2, 1, 3, 5, 4)),                # wedge
                14: ((1, 3, 4), (0, 3, 2, 1, 4))}                   # pyramid


@tack.kernel
def _suspect_frames(types, offsets, connectivity, positions, flags):
    # The screen of _oriented_connectivity, a cell a thread: 1 where a tetrahedron's,
    # hexahedron's, wedge's or pyramid's frame at corner 0 (_ORIENTATION's edges) is
    # negative or nearly flat, which only the volume then decides.
    for c in range(flags.shape[0]):
        kind = types[c]
        flag = 0
        if kind == 10 or kind == 12 or kind == 13 or kind == 14:
            s = offsets[c]
            second = 2
            third = 3
            if kind == 12 or kind == 14:
                second = 3
                third = 4
            origin = positions[connectivity[s]]
            a = positions[connectivity[s + 1]] - origin
            b = positions[connectivity[s + second]] - origin
            d = positions[connectivity[s + third]] - origin
            frame = a.dot(b.cross(d))
            if frame < 1e-4 * tack.sqrt(a.dot(a) * b.dot(b) * d.dot(d)):
                flag = 1
        flags[c] = flag


def _suspects(types, offsets, connectivity, x):
    """``_suspect_frames`` over every cell, on the CPU device, reading VTK's arrays where
    they lie: each cell's flag."""
    import numpy as np

    cpu = tack.device("cpu")

    def wrapped(host, dtype, shape=None):
        host = np.ascontiguousarray(host)
        return tack.array_from_ptr(host, shape or host.shape, dtype, device=cpu)

    from tack.lang.types import from_numpy_dtype

    flags = tack.empty((types.size,), tack.u8, device=cpu)
    _suspect_frames(wrapped(types.astype(np.uint8, copy=False), tack.u8),
                    wrapped(offsets, tack.i64),
                    wrapped(connectivity, from_numpy_dtype(connectivity.dtype)),
                    wrapped(x, tack.vec(3, from_numpy_dtype(x.dtype)), (x.shape[0],)), flags)
    return flags.numpy().astype(bool)


def _oriented_connectivity(grid, orient, cells=None):
    """The grid's connectivity with its inside-out cells turned outward, or ``None``
    where there is nothing to change. A cell is inside out when its corners are in
    the reverse of VTK's order for its shape (as some converters write prisms): the
    determinant at corner 0 is negative -- cheap, over every tetrahedron,
    hexahedron, wedge and pyramid -- and so is its volume summed over its faces,
    computed for those alone; a tangled cell, negative at some corners but not
    inside out, is left. Without ``orient`` they are counted in a warning that
    names ``orient=True``, and nothing changes. ``cells`` (ids) limits the check to
    those cells."""
    import numpy as np

    chosen = (np.arange(grid.number_of_cells, dtype=np.int64) if cells is None
              else np.asarray(cells, np.int64))
    types = np.asarray(grid.cell_types)[chosen] if chosen.size else np.zeros(0, np.uint8)
    kinds = [k for k in _ORIENTATION if (types == k).any()] if types.size else []
    if not kinds:
        return None
    stored = grid.GetCells()
    offsets = np.asarray(stored.offsets_array).astype(np.int64)
    connectivity = np.asarray(stored.connectivity_array)
    x = _vtk_points(grid)
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float64)
    flagged = _suspects(np.asarray(grid.cell_types), offsets, connectivity, x)
    inverted = {}
    for kind in kinds:
        cells = chosen[types == kind]
        corners = int(offsets[cells[0] + 1] - offsets[cells[0]])
        suspect = cells[flagged[cells]]
        if not suspect.size:
            continue
        rows = connectivity[offsets[suspect][:, None] + np.arange(corners)]
        corner = x[rows].astype(np.float64)                         # (cells, corners, 3)
        centre = corner.mean(axis=1)
        volume = np.zeros(suspect.size)
        for ring in _face_tables(kind, corners, int(suspect[0])):
            ring_x = corner[:, ring] - centre[:, None]
            middle = ring_x.mean(axis=1)
            for j in range(len(ring)):
                volume += np.einsum("ij,ij->i", middle,
                                    np.cross(ring_x[:, j], ring_x[:, (j + 1) % len(ring)]))
        inside_out = suspect[volume < 0]
        if inside_out.size:
            inverted[kind] = (inside_out, corners)
    if not inverted:
        return None
    count = sum(cells.size for cells, _ in inverted.values())
    if not orient:
        import warnings

        names = {10: "tetrahedra", 12: "hexahedra", 13: "wedges", 14: "pyramids"}
        listed = ", ".join(f"{cells.size} {names[k]}" for k, (cells, _) in inverted.items())
        warnings.warn(f"{count} cells are inside out ({listed}): their corners are in the "
                      "reverse of VTK's order, so their volumes are negative and their faces "
                      "point inward; vtk_to_dataset(..., orient=True) turns them outward",
                      stacklevel=4)
        return None
    repaired = np.array(connectivity)
    for kind, (cells, corners) in inverted.items():
        rows = offsets[cells][:, None] + np.arange(corners)
        repaired[rows] = repaired[rows][:, list(_ORIENTATION[kind][1])]
    return repaired


#: Cells whose face lists are placed at once: bounds the index arrays.
_CELLS_AT_ONCE = 1 << 22


def _polyhedral_topology(grid, orient=False, id_dtype=None, cells=None):
    """A ``vtkUnstructuredGrid`` as a ``PolyhedralTopology``, every cell a polyhedron.

    A polyhedron's faces are its own (``polyhedron_faces``/``_face_locations``);
    any other cell's are its cell class's, outward as VTK winds them (a voxel's
    pixels as quads). VTK keeps a face once per cell, each copy outward for its
    cell, so copies are matched by their point sets: the first found is the face,
    its cell side 0, and a second copy -- which must go round the other way -- is
    side 1. Read on the host in whole-array steps: a polyhedron's faces gathered by
    ranges, every other cell type's through its class's face table applied to all
    its cells at once (``_face_tables``). ``cells`` (ids, in order) takes those cells
    alone, over all of the grid's points.
    """
    import numpy as np

    from tack.dataset import PolyhedralTopology
    from tack.dataset.polyhedra import _ranges

    chosen = (np.arange(grid.number_of_cells, dtype=np.int64) if cells is None
              else np.asarray(cells, np.int64))
    n = chosen.size
    types = np.asarray(grid.cell_types)[chosen] if n else np.zeros(0, np.uint8)
    stored = grid.GetCells()
    offsets = np.asarray(stored.offsets_array).astype(np.int64)
    # Point ids in 32 bits where they fit: the face lists are the largest arrays.
    ids = np.int32 if grid.number_of_points < 2**31 else np.int64
    connectivity = _oriented_connectivity(grid, orient, cells=chosen)
    if connectivity is None:
        connectivity = np.asarray(stored.connectivity_array)
    faces_of = np.zeros(n, np.int64)                       # each cell's ring count
    listed_parts = []                                      # polyhedra: (cells, count, sizes, points)
    fixed_parts = []                                       # one shape: (cells, face sizes, block)
    for kind in np.unique(types):
        local = np.flatnonzero(types == kind)              # among the chosen cells
        members = chosen[local]                            # the grid's ids
        if kind == 42:                                     # VTK_POLYHEDRON: its own faces
            locations, faces = grid.polyhedron_face_locations, grid.polyhedron_faces
            loc_offsets = np.asarray(locations.offsets_array).astype(np.int64)
            loc_faces = np.asarray(locations.connectivity_array).astype(np.int64)
            face_offsets = np.asarray(faces.offsets_array).astype(np.int64)
            face_conn = np.asarray(faces.connectivity_array)
            count = loc_offsets[members + 1] - loc_offsets[members]
            listed = loc_faces[_ranges(loc_offsets[members], count)]
            sizes = face_offsets[listed + 1] - face_offsets[listed]
            points = face_conn[_ranges(face_offsets[listed], sizes)].astype(ids)
            faces_of[local] = count
            listed_parts.append((local, count, sizes, points))
        else:
            # A cell type's faces read from its corners by columns, a block of one
            # row a cell: no index arrays as long as all the ring points.
            corners = int(offsets[members[0] + 1] - offsets[members[0]])
            tables = _face_tables(int(kind), corners, int(members[0]))
            column = [connectivity[offsets[members] + k].astype(ids) for k in range(corners)]
            block = np.stack([column[i] for table in tables for i in table], axis=1)
            del column
            faces_of[local] = len(tables)
            fixed_parts.append((local, [len(table) for table in tables], block))
    cell_offsets = np.concatenate([[0], np.cumsum(faces_of)]).astype(np.int64)
    ring_sizes = np.zeros(int(cell_offsets[-1]), np.int64)
    for local, count, sizes, _ in listed_parts:
        ring_sizes[_ranges(cell_offsets[local], count)] = sizes
    for local, face_sizes, _ in fixed_parts:
        ring_sizes[cell_offsets[local][:, None] + np.arange(len(face_sizes))] = face_sizes
    ring_offsets = np.concatenate([[0], np.cumsum(ring_sizes)]).astype(np.int64)
    del ring_sizes
    ring_points = np.zeros(int(ring_offsets[-1]), ids)
    for local, count, sizes, points in listed_parts:
        rings = _ranges(cell_offsets[local], count)
        ring_points[_ranges(ring_offsets[rings], sizes)] = points
    for local, _, block in fixed_parts:
        starts = ring_offsets[cell_offsets[local]]         # each cell's first ring point
        width = np.arange(block.shape[1])
        for i in range(0, local.size, _CELLS_AT_ONCE):     # bounded index arrays
            ring_points[starts[i:i + _CELLS_AT_ONCE, None] + width] = block[i:i + _CELLS_AT_ONCE]
    del fixed_parts, listed_parts
    try:
        try:
            # With orient, the shaped cells are outward already: the faces usually agree,
            # and the cell-by-cell walk below is needed only where polyhedra do not.
            return PolyhedralTopology._from_rings(cell_offsets, ring_offsets, ring_points,
                                                  grid.number_of_points, id_dtype=id_dtype)
        except ValueError as error:
            if not orient or "wound the same way" not in str(error):
                raise
        listed = [[ring_points[ring_offsets[r]:ring_offsets[r + 1]].tolist()
                   for r in range(cell_offsets[c], cell_offsets[c + 1])] for c in range(n)]
        return PolyhedralTopology.from_cell_faces(listed, grid.number_of_points,
                                                  positions=_vtk_points(grid), orient=True,
                                                  id_dtype=id_dtype)
    except ValueError as error:
        if "wound the same way" in str(error) and not orient:
            raise ValueError(f"{error}. VTK winds every polyhedron face outward for its "
                             "cell; vtk_to_dataset(..., orient=True) repairs the input") from None
        raise


def _face_tables(kind, corners, first):
    """VTK cell type ``kind``'s faces as lists of local corner numbers, from a cell of
    that type whose point ids are its corners' numbers (a pixel face as a quad);
    ``first`` names the grid's first cell of the type in the error a cell that is
    not 3D raises."""
    from vtkmodules.vtkCommonDataModel import vtkGenericCell

    cell = vtkGenericCell()
    cell.SetCellType(kind)
    cell.point_ids.SetNumberOfIds(corners)
    for i in range(corners):
        cell.point_ids.SetId(i, i)
    if cell.cell_dimension != 3:
        raise ValueError(f"cell {first} is not 3D: a polyhedral topology is of 3D cells")
    tables = []
    for k in range(cell.number_of_faces):
        face = cell.GetFace(k)
        ids = [face.GetPointId(i) for i in range(face.number_of_points)]
        if face.cell_type == 8:                            # a pixel, as a quad
            ids = [ids[0], ids[1], ids[3], ids[2]]
        tables.append(ids)
    return tables


def _split_polyhedra(grid, options):
    """``vtk_to_dataset(grid, polyhedral="split")``: a ``PartitionedDataSet`` of the
    grid's shaped cells (an ``UnstructuredTopology``) and its polyhedra (a
    ``PolyhedralTopology``), both over all of the grid's points, so a mesh mixing
    them is read without making every hexahedron a polyhedron. The pieces share
    their positions and point fields (one array each); the shaped cells' piece
    reads VTK's points and point data as they are (with ``copy=False``, without
    copying), and takes its cells' rows of the cell data. A point is owned by the
    shaped cells' piece if one of its cells uses it, else by the polyhedra's: the
    other piece marks it ``DUPLICATEPOINT``, so reductions count it once, and a
    point's global id is its index (VTK's ``GlobalIds`` where it has them), a
    cell's its id in the grid. A grid of one kind gives one piece."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkCellData, vtkUnstructuredGrid

    from tack.dataset import H1, Constant, DataSet, Field, PartitionedDataSet
    from tack.dataset.ghosts import Ghosts
    from tack.dataset.global_ids import GlobalIds
    from tack.dataset.polyhedra import _ranges

    whole = {**options, "polyhedral": None}
    types = np.asarray(grid.cell_types)
    polyhedra = np.flatnonzero(types == 42)
    shaped = np.flatnonzero(types != 42)
    if not polyhedra.size or not shaped.size:
        return PartitionedDataSet([_vtk_to_dataset(grid, **whole)])
    stored = grid.GetCells()
    offsets = np.asarray(stored.offsets_array).astype(np.int64)
    connectivity = np.asarray(stored.connectivity_array)
    sizes = offsets[shaped + 1] - offsets[shaped]
    corners = connectivity[_ranges(offsets[shaped], sizes)]
    part = vtkUnstructuredGrid()
    part.SetPoints(grid.GetPoints())
    part.GetPointData().ShallowCopy(grid.GetPointData())
    part.GetFieldData().ShallowCopy(grid.GetFieldData())
    kinds = types[shaped]
    part.cells = (int(kinds[0]) if (kinds == kinds[0]).all() else vtkUnsignedCharArray(kinds),
                  vtkCellArray(offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
                               connectivity=corners))
    _take_cell_data(grid.GetCellData(), part.GetCellData(), shaped)
    first = _vtk_to_dataset(part, **{**whole, "polyhedral": False})

    topology = _polyhedral_topology(grid, orient=options["orient"],
                                    id_dtype=options["id_dtype"], cells=polyhedra)
    # The point fields are the shaped piece's arrays; a cell field may share a point
    # field's name, as VTK's arrays do (two fields of one name in two spaces).
    fields = [(name, Field(H1(topology), f.values, f.unit))
              for name, f in first.fields.entries() if isinstance(f.space, H1)]
    rows = vtkCellData()
    _take_cell_data(grid.GetCellData(), rows, polyhedra)
    share = not options["copy"]
    fields += list(_attributes_to_fields(rows, Constant(topology), options["dtype"],
                                         share).items())

    # Ownership: the shaped cells' points are theirs, the rest the polyhedra's.
    n = grid.number_of_points
    by_shapes = np.zeros(n, bool)
    by_shapes[corners] = True
    held = (first.ghosts.points.numpy() if first.ghosts is not None
            and first.ghosts.points is not None else np.zeros(n, np.uint8))
    elsewhere = np.uint8(Ghosts.DUPLICATEPOINT)
    first_cells = first.ghosts.cells if first.ghosts is not None else None
    first.ghosts = Ghosts(cells=first_cells,
                          points=tack.array(held | np.where(by_shapes, 0, elsewhere).astype(np.uint8)))
    marks = _ghost_marks(rows, share)
    second_ghosts = Ghosts(cells=marks, points=tack.array(
        held | np.where(by_shapes, elsewhere, 0).astype(np.uint8)))
    points = (first.global_ids.points if first.global_ids is not None
              and first.global_ids.points is not None else tack.arange(n, dtype=tack.i64))
    first.global_ids = GlobalIds(cells=tack.array(shaped, tack.i64), points=points)
    second = DataSet(topology, Field(H1(topology), first.geometry.values, first.unit),
                     fields=fields, ghosts=second_ghosts,
                     global_ids=GlobalIds(cells=tack.array(polyhedra, tack.i64), points=points),
                     origin=first.origin, unit=first.unit)
    return PartitionedDataSet([first, second])


class _DefaultFloat:
    """``vtk_to_dataset``'s default ``dtype``, the default float type when it is
    called (``dtype=None`` already means VTK's own types)."""

    def __repr__(self):
        return "tack.get_default_float()"


_DEFAULT_FLOAT = _DefaultFloat()


def vtk_to_dataset(grid, dtype=_DEFAULT_FLOAT, polyhedral=None, orient=False, id_dtype=None,
                   origin=None, copy=True):
    """Copy a ``vtkUnstructuredGrid``, ``vtkPolyData``, ``vtkStructuredGrid``,
    ``vtkRectilinearGrid`` or ``vtkImageData`` into a ``tack.dataset.DataSet``: an image's points as
    ``UniformCoordinates`` (with its direction matrix), a rectilinear grid's as a
    ``CartesianProduct``, a structured grid's as its explicit points.

    Points and floating-point arrays become fields of ``dtype``, by default the
    default float type (``tack.get_default_float()``, f32 unless set); integer
    arrays keep their type. Named point data become ``H1`` fields and named
    cell data ``Constant`` fields (a point array and a cell array of one name
    are two fields of that name, ``data.fields[name, td.Constant]`` the cell
    one); a ``vtkGhostType`` array becomes the ghost marks
    (``data.ghosts``), not a field; a constant or affine array (VTK's implicit
    arrays) stays implicit, as a ``ConstantArray`` or ``CountingArray``. An
    unstructured grid's cells must all be linear shapes -- or all VTK Lagrange
    curves, quadrilaterals or hexahedra of one order ``p``, which become linear
    cells on their corner points, with the geometry and point data in
    ``H1(order=p, basis="equispaced")`` (VTK's Lagrange nodes) -- unless it holds
    polyhedra (or ``polyhedral`` is true), when it becomes a
    ``PolyhedralTopology``, every cell a polyhedron of its faces (with
    ``polyhedral="split"``, a ``PartitionedDataSet`` instead: its shaped cells one
    piece, its polyhedra another, over the same points; see ``_split_polyhedra``), polylines,
    when it becomes a ``PolylineTopology`` (its lines two-point polylines, and
    no other cells allowed), or polygons (``VTK_POLYGON``), when it becomes a
    ``PolygonalTopology`` (its triangles and quads polygons too, and no other
    cells allowed). Its types, offsets and connectivity stay implicit where
    VTK's are (a cell type given as a number, affine offsets). Hexahedra with
    corners repeated as finite-element files write other shapes (Tecplot's
    bricks, degenerate Exodus and Patran hexahedra) become those shapes: a
    wedge ``[a, b, c, c, d, e, f, f]`` (in any rotation), a pyramid ``[a, b, c,
    d, e, e, e, e]``, a tetrahedron ``[a, b, c, c, d, d, d, d]``.
    A ``vtkPolyData`` is taken as the unstructured grid of its cells in VTK's
    order -- verts (a poly vertex as its vertices), lines, polygons (of three or
    four points triangles and quads), and triangle strips, each strip as its
    triangles, which take its cell data -- so the same rules hold: a surface of triangles and quads is
    an ``UnstructuredTopology``, one with larger polygons a ``PolygonalTopology``,
    lines a ``PolylineTopology``, and polylines or polygons beside other cells
    are refused.
    Tetrahedra, hexahedra, wedges and pyramids whose corners are in the reverse of
    VTK's order (inside out: a negative volume, faces pointing inward, as some
    converters write prisms) are found on every read, by a kernel, and named in a
    warning; with ``orient`` they are turned outward (``_oriented_connectivity``).
    A tangled cell, negative at some corners but not inside out, is left.
    With ``orient``, inconsistently wound polyhedra are repaired too
    (``PolyhedralTopology.from_cell_faces``), where faces still disagree once the
    shaped cells are outward; without it they are refused.
    VTK's ids are 64-bit; the topology keeps them in ``id_dtype``, by default
    ``i32`` unless the grid is too large for it (``tack.dataset.ids``).

    ``origin`` (three numbers) becomes the dataset's ``origin``: VTK's points,
    an image's origin and a rectilinear grid's lines are taken relative to it,
    subtracted in double precision before they become ``dtype``, so points far
    from zero keep their precision. Without one, converting to ``f32`` takes
    the lower corner of VTK's bounds, rounded down, where the data are more
    than 100 times farther from zero than they are wide
    (``tack.dataset.dataset.automatic_origin``; a composite's pieces one
    origin), and zero otherwise.

    A ``vtkOverlappingAMR`` becomes an ``AMRDataSet``: each block (an image or a
    rectilinear grid) converted as above, placed in its level by its
    ``vtkAMRBox``, the levels' refinement ratios kept; the refined cells are
    marked afresh (VTK's ``vtkGhostType`` bits come in and are recomputed).
    Blocks with ghost layers are refused.

    A ``vtkCellGrid`` (VTK's discontinuous-Galerkin cells, read from ``.dg``
    files by ``vtkCellGridReader``) becomes a ``DataSet`` of its cells on its
    shape attribute's corners: ``C1`` attributes copied (shared ones as ``H1``,
    per-cell ones as ``L2``), ``constant`` ones as ``Constant``, and every other
    ``HGRAD`` or Bezier basis resampled by VTK onto Tack's equispaced space of
    its order (``tack.interop.vtk._cellgrid``); ``HCURL`` and ``HDIV`` are left
    out, with a warning. It needs VTK's cell-type accessors on
    ``vtkCellAttribute``, newer than 9.7.1.

    A ``vtkPartitionedDataSet`` becomes a ``PartitionedDataSet``, each piece
    converted as above (null pieces skipped), and a
    ``vtkPartitionedDataSetCollection`` a ``dict`` of them, keyed by each
    partitioned dataset's ``vtkCompositeDataSet.NAME()`` (``"block<i>"`` where
    it has none, ``"<name>#<i>"`` for a name already taken).

    VTK's ``DATA_TIME_STEP`` becomes the step's time (``time``), its unit
    unknown.

    ``dtype=None`` keeps each floating-point array in its own type (f32 or f64),
    the points too; an image's geometry is then f64, as VTK's is. On a default
    device without f64 (Metal), it is f32, which such a device's arrays take.

    With ``copy=False`` the dataset shares VTK's memory instead of copying it,
    as CPU arrays (whatever the default device): the points, data arrays, cell
    types, offsets and connectivity, ghost marks and global ids, wherever
    their values are one contiguous block of the type Tack keeps them in, and
    a polygonal dataset's cells where it holds one kind (verts, lines or
    polygons). Ids then keep VTK's width (``id_dtype`` follows it unless
    given), and floating-point arrays are shared at their own precision with
    ``dtype=None``. The shared arrays are read-only to Tack and hold VTK's
    arrays alive; what VTK changes in them afterwards, Tack sees. What cannot
    be shared -- an origin to subtract, a type to convert, triangle strips,
    collapsed hexahedra, polyhedra -- is copied as before.

    With Metal the default device, ``copy=False`` gives Metal arrays instead, as
    Apple silicon's unified memory allows with any VTK: the buffers a Metal
    dataset handed VTK (``dataset_to_vtk(copy=False)``) come back as themselves,
    memory on whole pages is wrapped in place, and anything else is copied to
    Metal, so the dataset is on Metal throughout."""
    from tack.runtime.dispatch import get_backend

    if dtype is _DEFAULT_FLOAT:
        dtype = tack.get_default_float()
    if dtype is None and not get_backend().supports_f64:
        dtype = tack.f32
    if origin is None and dtype == tack.f32:
        origin = _far_origin(grid)
    out = _vtk_to_dataset(grid, dtype=dtype, polyhedral=polyhedral, orient=orient,
                          id_dtype=id_dtype, origin=origin, copy=copy)
    if grid.GetInformation().Has(_time_key()):
        _stamp(out, grid.GetInformation().Get(_time_key()))
    return out


def _strip_triangles(offsets, connectivity):
    """``(rows, strip of each)``: triangle strips as triangles, each strip's
    triangles in order, every other one turned so all wind as the first does."""
    import numpy as np

    sizes = np.diff(offsets)
    counts = np.maximum(sizes - 2, 0)
    strip = np.repeat(np.arange(len(sizes)), counts)
    t = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    first = offsets[:-1][strip] + t
    odd = (t % 2).astype(bool)
    rows = np.stack([connectivity[first], connectivity[first + 1], connectivity[first + 2]], 1)
    rows[odd, 0], rows[odd, 1] = rows[odd, 1], rows[odd, 0].copy()
    return rows, strip


def _far_origin(grid):
    """The origin data far from zero for their size take in f32
    (``automatic_origin`` of VTK's bounds), or None; none for a hyper tree grid,
    whose octree keeps its roots' own lattice."""
    import numpy as np
    from vtkmodules.vtkCommonDataModel import vtkHyperTreeGrid

    from tack.dataset.dataset import automatic_origin

    if isinstance(grid, vtkHyperTreeGrid) or not hasattr(grid, "GetBounds"):
        return None
    bounds = [0.0] * 6           # filled: composites' GetBounds() returns no tuple
    grid.GetBounds(bounds)
    bounds = np.asarray(bounds, np.float64)
    return automatic_origin(bounds[0::2], bounds[1::2])


#: A VTK hexahedron's edges, as pairs of its corners.
_HEXAHEDRON_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7))


def _collapsed_hexahedra(types, offsets, connectivity):
    """Hexahedra with corners repeated as finite-element files write other shapes
    (Tecplot's bricks, Patran's and Exodus' degenerate hexahedra) as those
    shapes: ``[a, b, c, c, d, e, f, f]`` (any rotation of the quads) a wedge
    ``[a, b, c, d, e, f]``, ``[a, b, c, d, e, e, e, e]`` a pyramid,
    ``[a, b, c, c, d, d, d, d]`` a tetrahedron, VTK's corner orders keeping the
    volume's sign. ``(types, offsets, connectivity)`` as NumPy arrays, or None
    where there is none; any other collapse stays a hexahedron."""
    import numpy as np

    from tack.arrays import to_host

    def host(values):
        return np.asarray(values if isinstance(values, np.ndarray) else to_host(values))

    t = host(types).reshape(-1)
    hexes = np.flatnonzero(t == 12)
    if not len(hexes):
        return None
    o = host(offsets).astype(np.int64)
    c = host(connectivity)
    # Every collapse converted below joins an edge's two corners, so only rows with
    # a collapsed edge need the sort -- in most meshes none, and the 21M rows of a
    # large mesh are not sorted for nothing.
    corner = [c[o[hexes] + k] for k in range(8)]
    collapsed = np.zeros(len(hexes), bool)
    for a, b in _HEXAHEDRON_EDGES:
        collapsed |= corner[a] == corner[b]
    if not collapsed.any():
        return None
    hexes = hexes[collapsed]
    rows = np.stack([column[collapsed] for column in corner], axis=1)
    del corner
    ordered = np.sort(rows, axis=1)
    distinct = 1 + (np.diff(ordered, axis=1) != 0).sum(axis=1)
    if (distinct == 8).all():
        return None
    # Each converted hexahedron's new type and its corners' places in its row.
    kinds = np.full(len(hexes), 12, np.uint8)
    places = np.tile(np.arange(8), (len(hexes), 1))
    top = (rows[:, 4] == rows[:, 5]) & (rows[:, 5] == rows[:, 6]) & (rows[:, 6] == rows[:, 7])
    tetra = top & (rows[:, 2] == rows[:, 3]) & (distinct == 4)
    pyramid = top & (distinct == 5)
    kinds[tetra], places[tetra, :4] = 10, [0, 1, 2, 4]
    kinds[pyramid], places[pyramid, :5] = 14, [0, 1, 2, 3, 4]
    for turn in range(4):
        bottom = [(turn + i) % 4 for i in range(4)]
        wedge = ((kinds == 12) & (distinct == 6)
                 & (rows[:, bottom[2]] == rows[:, bottom[3]])
                 & (rows[:, 4 + bottom[2]] == rows[:, 4 + bottom[3]]))
        kinds[wedge] = 13
        places[wedge, :6] = bottom[:3] + [4 + b for b in bottom[:3]]
    if (kinds == 12).all():
        return None
    t = t.copy()
    t[hexes] = kinds
    sizes = np.diff(o)
    sizes[hexes] = np.select([kinds == 13, kinds == 14, kinds == 10], [6, 5, 4], 8)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    within = np.arange(starts[-1]) - np.repeat(starts[:-1], sizes)
    local = within.copy()
    rank = np.full(len(t), -1)
    rank[hexes] = np.arange(len(hexes))
    entry_rank = np.repeat(rank, sizes)
    moved = entry_rank >= 0
    local[moved] = places[entry_rank[moved], within[moved]]
    return t, starts, c[np.repeat(o[:-1], sizes) + local]


def _polydata_as_grid(polydata):
    """A ``vtkPolyData`` as the ``vtkUnstructuredGrid`` of its cells, on the same
    points and point data: its verts, lines, polygons and strips in VTK's cell
    order (a polygon of three or four points a triangle or quad, as
    ``vtkPolyData.GetCellType`` says), each strip as its triangles and each
    poly vertex as its vertices, which take its cell data."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkIdList, vtkUnsignedCharArray
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid

    kinds = [(kind, cells) for kind, cells in (("verts", polydata.verts),
                                                ("lines", polydata.lines),
                                                ("polys", polydata.polys),
                                                ("strips", polydata.strips))
             if cells.number_of_cells]
    def poly_vertices(cells):
        return (np.diff(np.asarray(cells.offsets_array)) > 1).any()

    if len(kinds) == 1 and kinds[0][0] != "strips" and not (kinds[0][0] == "verts"
                                                           and poly_vertices(kinds[0][1])):
        # One kind of cell: the grid takes the polydata's cell array itself, so
        # its offsets and connectivity are not copied, and the cells (and their
        # cell data) are the same in the same order.
        kind, cells = kinds[0]
        size = np.diff(np.asarray(cells.offsets_array))
        types = {"verts": np.where(size == 1, 1, 2),
                 "lines": np.where(size == 2, 3, 4),
                 "polys": np.select([size == 3, size == 4], [5, 9], 7)}[kind].astype(np.uint8)
        grid = vtkUnstructuredGrid()
        grid.SetPoints(polydata.GetPoints())
        grid.GetPointData().ShallowCopy(polydata.GetPointData())
        grid.GetFieldData().ShallowCopy(polydata.GetFieldData())
        grid.GetCellData().ShallowCopy(polydata.GetCellData())
        # One type is a number (a vtkConstantArray), nothing stored per cell.
        grid.cells = (int(types[0]) if (types == types[0]).all()
                      else vtkUnsignedCharArray(types), cells)
        return grid
    types, sizes, members, sources = [], [], [], []
    first = 0
    for kind, cells in (("verts", polydata.verts), ("lines", polydata.lines),
                        ("polys", polydata.polys), ("strips", polydata.strips)):
        n = cells.number_of_cells
        if not n:
            continue
        offsets = np.asarray(cells.offsets_array, np.int64)
        connectivity = np.asarray(cells.connectivity_array, np.int64)
        size = np.diff(offsets)
        if kind == "verts":
            # A poly vertex is its vertices.
            types.append(np.ones(len(connectivity), np.uint8))
            sizes.append(np.ones(len(connectivity), np.int64))
            members.append(connectivity)
            sources.append(first + np.repeat(np.arange(n), size))
        elif kind == "strips":
            rows, strip = _strip_triangles(offsets, connectivity)
            types.append(np.full(len(rows), 5, np.uint8))
            sizes.append(np.full(len(rows), 3))
            members.append(rows.reshape(-1))
            sources.append(first + strip)
        else:
            types.append({"lines": np.where(size == 2, 3, 4),
                          "polys": np.select([size == 3, size == 4], [5, 9], 7)}[kind]
                         .astype(np.uint8))
            sizes.append(size)
            members.append(connectivity)
            sources.append(first + np.arange(n))
        first += n
    grid = vtkUnstructuredGrid()
    grid.SetPoints(polydata.GetPoints())
    grid.GetPointData().ShallowCopy(polydata.GetPointData())
    grid.GetFieldData().ShallowCopy(polydata.GetFieldData())
    if not types:
        return grid
    offsets = np.concatenate([[0], np.cumsum(np.concatenate(sizes))]).astype(np.int64)
    grid.cells = (vtkUnsignedCharArray(np.concatenate(types)),
                  vtkCellArray(offsets=offsets, connectivity=np.concatenate(members)))
    source = np.concatenate(sources)
    if len(source) == polydata.number_of_cells:  # no strips or poly vertices: the same cells
        grid.GetCellData().ShallowCopy(polydata.GetCellData())
    else:
        _take_cell_data(polydata.GetCellData(), grid.GetCellData(), source)
    return grid


def _take_cell_data(source, target, rows):
    """``target`` (empty) given ``source``'s arrays at ``rows``, each its own class,
    name, component names and information (``UNITS_LABEL``), and active where
    ``source``'s was: a NumPy gather per array, where ``CopyData`` takes a
    ``vtkIdList`` filled one id at a time. An array that is not numeric (strings) is
    copied by ``InsertTuples`` through such a list."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkIdList

    listed = None
    for i in range(source.GetNumberOfArrays()):
        array = source.GetAbstractArray(i)
        out = array.NewInstance()
        out.SetName(array.GetName())
        out.SetNumberOfComponents(array.GetNumberOfComponents())
        for k in range(array.GetNumberOfComponents()):
            if array.HasAComponentName() and array.GetComponentName(k):
                out.SetComponentName(k, array.GetComponentName(k))
        out.SetNumberOfTuples(len(rows))
        out.CopyInformation(array.GetInformation(), 1)
        if array.IsNumeric():
            values, taken = np.asarray(array), np.asarray(out)
            if values.dtype.kind == "S":            # a vtkCharArray, read as bytes
                values, taken = values.view(np.int8), taken.view(np.int8)
            taken[...] = values[rows]
        else:
            if listed is None:
                listed = vtkIdList()
                listed.SetNumberOfIds(len(rows))
                for k, row in enumerate(rows.tolist()):
                    listed.SetId(k, row)
                into = vtkIdList()
                into.SetNumberOfIds(len(rows))
                for k in range(len(rows)):
                    into.SetId(k, k)
            out.InsertTuples(into, listed, array)
        target.AddArray(out)
    for kind in range(source.NUM_ATTRIBUTES):
        active = source.GetAbstractAttribute(kind)
        if active is not None and active.GetName():
            target.SetActiveAttribute(active.GetName(), kind)


def _vtk_to_dataset(grid, dtype=tack.f32, polyhedral=None, orient=False, id_dtype=None,
                   origin=None, copy=True):
    """``vtk_to_dataset`` before the time is stamped."""
    import numpy as np
    from vtkmodules.vtkCommonDataModel import (
        vtkImageData,
        vtkOverlappingAMR,
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
    from tack.dataset.ghosts import Ghosts
    from tack.dataset.global_ids import GlobalIds

    options = {"dtype": dtype, "polyhedral": polyhedral, "orient": orient, "id_dtype": id_dtype,
               "origin": origin, "copy": copy}
    share = not copy
    from tack.dataset.dataset import as_origin

    shift = np.asarray(as_origin(origin))
    if isinstance(grid, vtkOverlappingAMR):
        return _amr_to_dataset(grid, options)
    from vtkmodules.vtkCommonDataModel import vtkPolyData

    if isinstance(grid, vtkPolyData):
        return _vtk_to_dataset(_polydata_as_grid(grid), **options)
    geometry_dtype = dtype
    from vtkmodules.vtkCommonDataModel import vtkHyperTreeGrid

    if isinstance(grid, vtkHyperTreeGrid):
        return _htg_to_octree(grid, dtype)
    from vtkmodules.vtkCommonDataModel import vtkCellGrid

    if isinstance(grid, vtkCellGrid):
        from tack.interop.vtk import _cellgrid

        return _cellgrid.to_dataset(grid, dtype=dtype or tack.f64, id_dtype=id_dtype,
                                    shift=shift)
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
        if polyhedral == "split":
            return _split_polyhedra(grid, options)
        if polyhedral is None:
            polyhedral = 42 in present
        from tack.interop.vtk import _lagrange

        if present & (_lagrange.HIGHER_ORDER | set(_lagrange.REFUSED)):
            linear = present - _lagrange.HIGHER_ORDER - set(_lagrange.REFUSED)
            if linear - set(_lagrange._LINEAR_SHAPES):
                raise NotImplementedError(
                    f"VTK cell types {sorted(linear - set(_lagrange._LINEAR_SHAPES))} beside "
                    "higher-order cells")

            def float_fields(values):
                values = values.reshape(values.shape[0]) if values.shape[1] == 1 else values
                if np.issubdtype(values.dtype, np.floating):
                    kind = dtype or (tack.f64 if values.dtype == np.float64 else tack.f32)
                    values = values.astype(kind.numpy_dtype)
                return tack.array(values) if values.ndim == 1 else _vectors_of(values)

            return _lagrange.to_dataset(grid, present, dtype or tack.f64, id_dtype, shift,
                                        _units_label, float_fields)
    if isinstance(grid, vtkUnstructuredGrid) and polyhedral:
        topology = _polyhedral_topology(grid, orient=orient, id_dtype=id_dtype)
        geometry, geometry_dtype = _points_geometry(grid, shift, dtype, False)
    elif isinstance(grid, vtkUnstructuredGrid):
        cells = grid.cells
        offsets = _from_vtk(cells.offsets_array)
        connectivity = _from_vtk(cells.connectivity_array)
        if share:
            offsets, connectivity = _shared_ids(offsets), _shared_ids(connectivity)
            shared = _shared(types, tack.u8)
            types = types if shared is None else shared
            widths = {a.dtype for a in (offsets, connectivity) if isinstance(a, tack.Array)}
            if id_dtype is None and len(widths) == 1:
                id_dtype, = widths                  # VTK's width, so nothing converts
        num_points = grid.number_of_points
        if 4 in present:
            # Polylines: their lines are two-point chains; nothing else mixes in.
            if not present <= {3, 4}:
                raise ValueError("a grid with polylines (VTK_POLY_LINE) may hold only lines "
                                 "and polylines")
            topology = PolylineTopology(_ids(offsets), _ids(connectivity),
                                        num_points=num_points, id_dtype=id_dtype)
        elif 7 in present:
            # Polygons, as dataset_to_vtk writes a PolygonalTopology: triangles
            # and quads are polygons too; nothing else mixes in.
            if not present <= {5, 7, 9}:
                raise ValueError("a grid with polygons (VTK_POLYGON) may hold only "
                                 "triangles, quads and polygons")
            topology = PolygonalTopology(_ids(offsets), _ids(connectivity),
                                         num_points=num_points, id_dtype=id_dtype)
        else:
            from tack.dataset.shapes import _BY_ID

            # Refused here, by VTK's names, rather than by the first algorithm.
            unknown = sorted(present - set(_BY_ID))
            if unknown:
                from vtkmodules.vtkCommonDataModel import vtkCellTypeUtilities

                names = ", ".join(vtkCellTypeUtilities.GetClassNameFromTypeId(t)
                                  for t in unknown)
                raise ValueError(f"VTK cell types {unknown} ({names}) are not shapes Tack "
                                 "has")
            repaired = _oriented_connectivity(grid, orient)
            if repaired is not None:
                connectivity = repaired                     # turned outward: VTK's own stays
            if 12 in present:
                collapsed = _collapsed_hexahedra(types, offsets, connectivity)
                if collapsed is not None:
                    types, offsets, connectivity = collapsed
            topology = UnstructuredTopology(types, offsets, connectivity, num_points=num_points,
                                            id_dtype=id_dtype)
        geometry, geometry_dtype = _points_geometry(grid, shift, dtype, share)
    elif isinstance(grid, vtkImageData):
        dims = list(grid.dimensions)
        matrix = grid.direction_matrix              # a vtkMatrix3x3: no indexing in 9.7.1
        direction = np.array([[matrix.GetElement(r, c) for c in range(3)] for r in range(3)])
        spacing = np.asarray(grid.spacing)
        # Points are origin + direction @ (ijk * spacing), ijk from the extent's start.
        start = np.asarray(grid.origin) + direction @ (np.asarray(grid.extent[0::2]) * spacing)
        geometry_dtype = dtype if dtype is not None else tack.f64      # VTK's double
        geometry = UniformCoordinates(dims, start - shift, spacing, direction,
                                      dtype=geometry_dtype)
        topology = grid_topology(dims, id_dtype=id_dtype, offset=grid.extent[0::2])
    elif isinstance(grid, vtkRectilinearGrid):
        if dtype is None:
            geometry_dtype = (tack.f32 if np.asarray(grid.x_coordinates).dtype == np.float32
                              else tack.f64)
        geometry = CartesianProduct(np.asarray(grid.x_coordinates, np.float64) - shift[0],
                                    np.asarray(grid.y_coordinates, np.float64) - shift[1],
                                    np.asarray(grid.z_coordinates, np.float64) - shift[2],
                                    dtype=geometry_dtype)
        dims = list(grid.dimensions)
        if dims != list(geometry.dims):
            raise ValueError(f"grid dimensions {dims} do not match its coordinates")
        topology = grid_topology(dims, id_dtype=id_dtype, offset=grid.extent[0::2])
    elif isinstance(grid, vtkStructuredGrid):
        dims = [0, 0, 0]
        grid.GetDimensions(dims)        # not a readable property on a structured grid in 9.7.1
        topology = grid_topology(dims, id_dtype=id_dtype, offset=grid.extent[0::2])
        geometry, geometry_dtype = _points_geometry(grid, shift, dtype, share)
    else:
        raise TypeError(f"expected a vtkUnstructuredGrid, vtkPolyData, vtkStructuredGrid, "
                        f"vtkRectilinearGrid, vtkImageData, vtkPartitionedDataSet or "
                        f"vtkPartitionedDataSetCollection, not {grid.class_name}")
    fields = _attributes_to_fields(grid.point_data, H1(topology), dtype, share)
    for name, field in _attributes_to_fields(grid.cell_data, Constant(topology), dtype,
                                             share).items():
        # A point array and a cell array of one name are two fields of that name.
        fields[name if name not in fields else (name, field.space)] = field
    if isinstance(grid, vtkRectilinearGrid):
        unit = _units_label(grid.x_coordinates)
    elif isinstance(grid, vtkImageData):
        unit = None                     # VTK 9.7.1 has nowhere to keep an image's
    else:
        unit = _units_label(grid.points.data) if grid.points is not None else None
    cells, points = _ghost_marks(grid.cell_data, share), _ghost_marks(grid.point_data, share)
    ghosts = None if cells is None and points is None else Ghosts(cells=cells, points=points)
    cells, points = _global_ids(grid.cell_data, share), _global_ids(grid.point_data, share)
    global_ids = None if cells is None and points is None else GlobalIds(cells=cells,
                                                                         points=points)
    return DataSet(topology, geometry, fields=fields, dtype=geometry_dtype, ghosts=ghosts,
                   global_ids=global_ids, origin=origin, unit=unit)


def _vectors_of(values):
    """A host ``(n, width)`` array as a device array of vectors of its type."""
    from tack.lang.types import from_numpy_dtype

    out = tack.empty((values.shape[0],), tack.vec(values.shape[1], from_numpy_dtype(values.dtype)))
    if values.shape[0]:
        tack.copyto(out, values)
    return out


def _points_geometry(grid, shift, dtype, share):
    """``(geometry, its type)``: a point set's positions relative to the origin, as a
    NumPy array the dataset converts to ``dtype`` (the points' own floating type
    when ``None``); with ``share``, VTK's points themselves, where no origin
    moves them and their type is that one (``_shared``)."""
    import numpy as np

    from tack.lang.types import from_numpy_dtype

    host = _vtk_points(grid)
    if dtype is None:
        dtype = from_numpy_dtype(host.dtype) if host.dtype.kind == "f" else tack.f64
    if share and not np.any(shift):
        shared = _shared(host, tack.vec(3, dtype))
        if shared is not None:
            return shared, dtype
    return host - shift, dtype


def _shared_ids(values):
    """Offsets or connectivity as VTK holds them, shared where they are stored i32 or
    i64 values (an implicit array stays as it is)."""
    import numpy as np

    if isinstance(values, np.ndarray) and values.dtype in (np.int32, np.int64):
        shared = _shared(values, tack.i32 if values.dtype == np.int32 else tack.i64)
        return values if shared is None else shared
    return values


def _ids(values):
    """Ids for a topology's constructor: a Tack array as it is, anything else as a
    NumPy array."""
    import numpy as np

    return values if isinstance(values, tack.Array) else np.asarray(values)


def _to_vtk_types(types, share=False):
    """Cell types for ``vtkUnstructuredGrid.cells``: a ``ConstantArray``'s value as
    a number (VTK keeps it as a vtkConstantArray), else a vtkUnsignedCharArray
    (with ``share``, of the types' own memory on the CPU)."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray

    from tack.arrays import ConstantArray

    if isinstance(types, ConstantArray):
        return int(types.value.numpy().reshape(-1)[0])
    if share and _stored_on_host(types) and types.dtype == tack.u8:
        return vtkUnsignedCharArray(_host_values(types))
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


def _amr_to_dataset(amr, options):
    """A ``vtkOverlappingAMR`` as an ``AMRDataSet``: each level's blocks, each placed
    in its level by its ``vtkAMRBox``."""
    from tack.dataset import AMRDataSet, DataSet, StructuredTopology

    levels, ratios = [], []
    for level in range(amr.GetNumberOfLevels()):
        boxes = []
        for index in range(amr.GetNumberOfBlocks(level)):
            block = amr.GetDataSetAsCartesianGrid(level, index)
            if block is None:               # another rank's
                continue
            lo, hi = [0, 0, 0], [0, 0, 0]
            amr.GetAMRBox(level, index).GetDimensions(lo, hi)   # cells, hi inclusive
            data = vtk_to_dataset(block, **options)
            topology = data.topology
            n = len(topology.point_dims)
            cells = [h - lo_ + 1 for lo_, h in zip(lo[:n], hi[:n])]
            if [d - 1 for d in topology.point_dims] != cells:
                raise ValueError(f"block {index} of level {level} has {topology.point_dims} "
                                 f"points for a box of {cells} cells: blocks with ghost "
                                 "layers are not taken yet")
            placed = StructuredTopology(topology.point_dims, id_dtype=topology.id_dtype,
                                        device=data.device, offset=tuple(lo[:n]))
            fields = {name: _rehomed(field, placed) for name, field in data.fields.items()}
            boxes.append(DataSet(placed, data.geometry.values, fields=fields,
                                 dtype=data.dtype, ghosts=data.ghosts,
                                 global_ids=data.global_ids, **data._frame()))
        levels.append(boxes)
        if level + 1 < amr.GetNumberOfLevels():
            ratios.append(int(amr.GetRefinementRatio(level)))
    return AMRDataSet(levels, ratios)


def _bits(array):
    """A ``vtkBitArray`` as a NumPy array of 0 and 1."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray

    out = vtkUnsignedCharArray()
    out.DeepCopy(array)
    return np.asarray(out).reshape(-1).astype(np.uint8)


def _htg_to_octree(htg, dtype):
    """A ``vtkHyperTreeGrid`` of branching factor 2 as an ``OctreeDataSet``: each
    tree's breadth-first descriptor read in one call and turned into Morton
    keys, level by level (a refined node's children are its key shifted by the
    dimension, the child's index below: HyperTreeGrid numbers children x
    fastest, as a Morton key's bits run)."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkBitArray, vtkIdList, vtkTypeInt64Array

    from tack.dataset import OctreeDataSet, rectilinear_grid

    if htg.GetBranchFactor() != 2:
        raise ValueError("an octree has a branching factor of 2; this HyperTreeGrid's is "
                         f"{htg.GetBranchFactor()}")
    dims = list(htg.GetDimensions())
    d = htg.GetDimension()
    if [n > 1 for n in dims] != [True] * d + [False] * (3 - d):
        raise ValueError(f"an octree's root grid spans x and y (and z), not points {dims}")
    axes = [np.asarray(c, np.float64) for c in (htg.GetXCoordinates(), htg.GetYCoordinates(),
                                                htg.GetZCoordinates())][:d]
    roots = rectilinear_grid(*axes, dtype=dtype, unit=_units_label(htg.GetXCoordinates()))
    cells = [n - 1 for n in dims[:d]] + [1] * (3 - d)
    children = np.arange(2 ** d, dtype=np.uint64)
    tree, level, key, order = [], [], [], []
    for t in range(htg.GetMaxNumberOfTrees()):
        ht = htg.GetTree(t)
        if ht is None:
            continue
        descriptor, per_depth, ids = vtkBitArray(), vtkTypeInt64Array(), vtkIdList()
        ht.ComputeBreadthFirstOrderDescriptor(htg.GetDepthLimiter(), None, per_depth,
                                              descriptor, ids)
        refined = _bits(descriptor)
        counts = np.asarray(per_depth, np.int64).reshape(-1)
        keys = [np.zeros(1, np.uint64)]
        start = 0
        for depth, count in enumerate(counts[:-1]):
            split = refined[start:start + count].astype(bool)
            start += count
            keys.append(((keys[-1][split] << np.uint64(d))[:, None] | children).reshape(-1))
        if htg.GetTransposedRootIndexing():     # tree index k fastest: ours is x fastest
            k, rest = t % cells[2], t // cells[2]
            j, i = rest % cells[1], rest // cells[1]
            t = i + cells[0] * (j + cells[1] * k)
        n = int(counts.sum())
        tree.append(np.full(n, t, np.int64))
        level.append(np.repeat(np.arange(len(counts)), counts))
        key.append(np.concatenate(keys))
        order.append(np.array([ids.GetId(i) for i in range(ids.GetNumberOfIds())], np.int64))
    tree, level, key, order = (np.concatenate(a) if a else np.zeros(0, np.int64)
                               for a in (tree, level, key, order))
    data = htg.GetCellData()
    fields, units = {}, {}
    for i in range(data.GetNumberOfArrays()):
        array = data.GetArray(i)
        if array is None or array.GetName() in (None, "", "vtkGhostType"):
            continue
        # A bit array reads as packed bytes: unpack it, one 0 or 1 per cell.
        values = _bits(array) if isinstance(array, vtkBitArray) else np.asarray(array)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.dtype.kind == "f":
            values = values.astype(dtype.numpy_dtype)
        fields[array.GetName()] = values[order]
        unit = _units_label(array)
        if unit is not None:
            units[array.GetName()] = unit
    mask = None
    if htg.HasMask():
        mask = _bits(htg.GetMask())[order]
    return OctreeDataSet(roots, tree, level, key.astype(np.uint64), fields=fields, mask=mask,
                         units=units)


def _octree_to_htg(octree):
    """An ``OctreeDataSet`` as a ``vtkHyperTreeGrid``: each tree built from its
    breadth-first descriptor (its nodes by level, then key), its cell data in
    that order, the trees one after another; ``HIDDENCELL`` as the mask."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkDoubleArray
    from vtkmodules.vtkCommonDataModel import (
        vtkHyperTreeGrid,
        vtkHyperTreeGridNonOrientedCursor,
    )

    from tack.dataset import Ghosts
    from tack.dataset.octree import _root_axes

    d = octree.dimension
    axes = _root_axes(octree.roots)
    origin = np.asarray(octree.origin, np.float64)
    out = vtkHyperTreeGrid()
    out.SetBranchFactor(2)
    out.SetDimensions(*([len(a) for a in axes] + [1] * (3 - d)))
    coordinates = [a + origin[i] for i, a in enumerate(axes)] + [np.zeros(1)] * (3 - d)
    out.SetXCoordinates(_label(vtkDoubleArray(coordinates[0]), octree.unit))
    out.SetYCoordinates(_label(vtkDoubleArray(coordinates[1]), octree.unit))
    out.SetZCoordinates(_label(vtkDoubleArray(coordinates[2]), octree.unit))
    tree, level, key = octree._tree, octree._level, octree._key
    marks = octree.marks.numpy()
    # Breadth-first within each tree: by tree, level, then key.
    order = np.lexsort((key, level, tree))
    refined = (marks & Ghosts.REFINEDCELL) != 0
    cursor = vtkHyperTreeGridNonOrientedCursor()
    start = 0
    for t in np.unique(tree):
        rows = order[tree[order] == t]
        last = level[rows].max()
        bits = refined[rows][level[rows] < last]
        descriptor = _to_bits(bits)
        out.InitializeNonOrientedCursor(cursor, int(t), True)
        ht = cursor.GetTree()
        ht.SetGlobalIndexStart(start)
        ht.BuildFromBreadthFirstOrderDescriptor(descriptor, len(bits), 0)
        start += len(rows)
    cell_data = out.GetCellData()
    for name, values in octree.fields.items():
        cell_data[name] = np.ascontiguousarray(np.asarray(values.numpy())[order])
        _label(cell_data[name], octree.units.get(name))
    hidden = (marks[order] & Ghosts.HIDDENCELL) != 0
    if hidden.any():
        out.SetMask(_to_bits(hidden))
    return out


def _to_bits(values):
    """A ``vtkBitArray`` of ``values`` (0 and 1), copied in one call."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkBitArray, vtkUnsignedCharArray

    out = vtkBitArray()
    out.DeepCopy(vtkUnsignedCharArray(np.ascontiguousarray(values, np.uint8)))
    return out


def _rehomed(field, topology):
    """``field``'s values in the same kind of space on ``topology``, one of the same
    cells and points."""
    from tack.dataset import H1, Constant, Field

    space = H1(topology) if isinstance(field.space, H1) else Constant(topology)
    return Field(space, field.values)


def _amr_to_vtk(amr):
    """An ``AMRDataSet`` of images and rectilinear grids as a ``vtkOverlappingAMR``, or
    None if VTK's model cannot hold it: a curvilinear box, ratios that differ
    between axes, or image levels off one lattice."""
    import numpy as np
    from vtkmodules.vtkCommonDataModel import vtkAMRBox, vtkOverlappingAMR

    from tack.arrays import CartesianProduct, UniformCoordinates

    d = amr.dimension
    kinds = {type(box.geometry.values) for _, box in amr}
    if not kinds <= {UniformCoordinates, CartesianProduct}:
        return None
    if any(len(set(r[:d])) > 1 for r in amr.ratios):
        return None
    # VTK's origin and spacing per level: from a level-0 box, its lattice for an
    # image, its first cell for a rectilinear grid (VTK reads blocks' own
    # coordinates for those).
    level, first = next((lv, box) for lv, box in amr)
    if level != 0:
        return None
    coordinates = first.geometry.values
    world = first.world_positions()
    if isinstance(coordinates, UniformCoordinates):
        spacing0 = np.asarray(coordinates._spacing, np.float64)
    else:
        axes = [a.numpy().astype(np.float64) for a in (coordinates.x, coordinates.y,
                                                         coordinates.z)]
        spacing0 = np.array([a[1] - a[0] if len(a) > 1 else 1.0 for a in axes])
    offset = np.zeros(3)
    offset[:d] = first.topology.offset
    origin = world[0] - offset * spacing0
    out = vtkOverlappingAMR()
    out.Initialize([len(amr.boxes(level)) for level in range(amr.num_levels)])
    out.SetOrigin(list(map(float, origin)))
    spacing = spacing0.copy()
    for level in range(amr.num_levels):
        out.SetSpacing(level, list(map(float, spacing)))
        ratio = amr.ratios[level][0] if level < amr.num_levels - 1 else 2
        out.SetRefinementRatio(level, int(ratio))
        for index, box in enumerate(amr.boxes(level)):
            # Cells, hi inclusive; an axis the boxes do not have is empty (hi -1),
            # as VTK writes a 2D box.
            hi = [int(v) - 1 if a < d else -1 for a, v in enumerate(box[1])]
            out.SetAMRBox(level, index, vtkAMRBox([int(v) for v in box[0]], hi))
        spacing = spacing / np.asarray(amr.ratios[level] if level < amr.num_levels - 1
                                       else (1, 1, 1))
    for level, pieces in enumerate(amr.levels):
        for k, box in enumerate(pieces):
            out.SetDataSet(level, amr._first[level] + k, dataset_to_vtk(box))
    return out


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


def _vtk_points_of(data, share=False):
    """``data``'s positions as vtkPoints of the geometry's own precision:
    ``vtkPoints(data=...)`` given a NumPy array stores doubles, so give it a
    typed array (f32 positions as a vtkFloatArray). A dataset with an origin
    gives its world positions, in double precision. With ``share``, stored
    positions on the CPU and no origin, the geometry's own memory."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkFloatArray, vtkPoints

    values = data.geometry.values
    if share and not any(data.origin) and _stored_on_host(values):
        positions = _host_values(values)
    else:
        positions = data.world_positions() if any(data.origin) else data.positions()
    typed = vtkFloatArray if positions.dtype == np.float32 else vtkDoubleArray
    points = vtkPoints(data=typed(np.ascontiguousarray(positions)))
    _label(points.data, data.unit)
    return points


def _stored_on_host(values):
    """Whether ``values`` is a stored (not implicit) interleaved array in host memory
    VTK can share: on the CPU, or on Metal (unified memory)."""
    return (isinstance(values, tack.Array) and values.device.type == "cpu"
            and getattr(values, "layout", "aos") == "aos") or _on_metal(values)


def _vtk_cells(offsets, connectivity, share=False):
    """A ``vtkCellArray`` of ``offsets`` and ``connectivity``: implicit ones as VTK's
    affine arrays, stored ones as int64 copies -- or, with ``share``, both stored
    on the CPU in one width, their own memory."""
    from vtkmodules.vtkCommonCore import vtkTypeInt32Array, vtkTypeInt64Array
    from vtkmodules.vtkCommonDataModel import vtkCellArray

    if (share and _stored_on_host(offsets) and _stored_on_host(connectivity)
            and offsets.dtype == connectivity.dtype and offsets.dtype in (tack.i32, tack.i64)):
        # As typed VTK arrays: given NumPy arrays, the constructor converts them to
        # int64; given VTK arrays, it keeps them, 32-bit ones too.
        typed = vtkTypeInt32Array if offsets.dtype == tack.i32 else vtkTypeInt64Array
        return vtkCellArray(offsets=typed(_host_values(offsets)),
                            connectivity=typed(_host_values(connectivity)))
    return vtkCellArray(offsets=_to_vtk_ids(offsets), connectivity=_to_vtk_ids(connectivity))


def _to_vtk_values(values, share=False):
    """A field's values for ``point_data``/``cell_data``: a scalar ``ConstantArray``
    as a vtkConstantArray and a ``CountingArray`` as a vtkAffineArray (nothing
    stored), anything else copied to the host -- or, with ``share``, a stored
    array on the CPU as a view of its memory."""
    import numpy as np
    from vtkmodules.vtkCommonCore import vtkAffineArray, vtkConstantArray

    from tack.arrays import ConstantArray, CountingArray, to_host

    if isinstance(values, ConstantArray) and values.width is None:
        dtype = values.dtype.numpy_dtype
        return vtkConstantArray[dtype](values.size, values.value.numpy().reshape(-1)[0])
    if isinstance(values, CountingArray):
        return vtkAffineArray[values.dtype.numpy_dtype](values.size, values.step, values.start)
    if share and _stored_on_host(values):
        return _host_values(values)
    return np.array(to_host(values))            # a copy: the grid owns its values


def _once(written, kind, name, data):
    """Refuse a second VTK array of ``kind`` called ``name``: VTK keeps one point and
    one cell array of a name, so two point fields of one name (an ``H1`` field and
    its ``Values`` twin) cannot both go."""
    if (kind, name) in written:
        spaces = " and ".join(repr(f.space) for f in data.fields.named(name))
        raise ValueError(f"fields {name!r} in {spaces} would be two VTK {kind} arrays of "
                         "one name: rename one")
    written.add((kind, name))


def _write_fields(grid, data, share):
    """``data``'s point and cell fields into ``grid``'s point and cell data, with
    its normals as VTK's active normals (``dataset_to_vtk``)."""
    import numpy as np

    from tack.dataset import H1, Constant, Values
    from tack.dataset.carry import _point_values

    written = set()
    for name, field in data.fields.entries():
        space = field.space
        if isinstance(space, H1) or (isinstance(space, Values) and space.on == "points"):
            _once(written, "point", name, data)
            # An order-2 field's values at the points, its first ones: the grid
            # is linear, as the filters read order-2 data.
            grid.point_data[name] = _to_vtk_values(_point_values(field.values, data.num_points),
                                                   share)
            _label(grid.point_data[name], field.unit)
        elif isinstance(space, Constant) or (isinstance(space, Values) and space.on == "cells"):
            _once(written, "cell", name, data)
            grid.cell_data[name] = _to_vtk_values(field.values, share)
            _label(grid.cell_data[name], field.unit)
    for attributes, names in ((grid.point_data, _POINT_NORMALS),
                              (grid.cell_data, _CELL_NORMALS)):
        for name in names:
            if name in attributes.keys():
                values = np.asarray(attributes[name])
                if values.ndim == 2 and values.shape[1] == 3 and values.dtype.kind == "f":
                    attributes.SetActiveNormals(name)   # not writable as a property
                    break


#: The fields ``dataset_to_vtk`` makes VTK's active normals, first found first.
_POINT_NORMALS = ("Normals", "normals")
_CELL_NORMALS = ("Normals", "Normals (cells)", "cell_normals")


def dataset_to_vtk(data, copy=True, higher_order=False, cell_grid=False):
    """Copy a ``tack.dataset.DataSet`` into a new VTK dataset; an ``AMRDataSet`` of
    images and rectilinear grids, its ratios the same along every axis, into a
    ``vtkOverlappingAMR`` (its marks as ``vtkGhostType`` arrays), any other into
    a ``vtkPartitionedDataSetCollection`` of a partitioned dataset per level
    (``"level<l>"``), VTK's AMR holding no curvilinear grids; a
    ``PartitionedDataSet`` into a ``vtkPartitionedDataSet`` of its pieces, and a
    ``dict`` of names to
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
    arrays. Ghost marks become ``vtkGhostType`` arrays.

    The step's time (``data.time``) becomes VTK's ``DATA_TIME_STEP``; VTK has
    no unit for it.

    Normals become VTK's active normals, which its renderers and filters read
    and a plain array is not: a 3-vector point field named ``Normals`` (VTK's
    readers' and filters' name) or ``normals`` (``surface_normals``' and
    ``point_normals``'), and a 3-vector cell field named ``Normals``,
    ``Normals (cells)`` or ``cell_normals``. Tack keeps no roles of its own.

    With ``copy=False`` the VTK dataset shares Tack's memory where the arrays
    are stored on the CPU: the points (with no origin), the field values, the
    cell types, offsets and connectivity (ids of one width, as VTK keeps 32-bit
    ids too), ghost marks and global ids. The VTK arrays hold Tack's, and
    writes on either side reach the other. Implicit arrays go across as VTK's
    implicit ones either way; anything else -- arrays on a GPU, a polyhedral
    topology's faces -- is copied as before. Metal arrays are shared too, with
    any VTK: on Apple silicon their buffers are host memory, so VTK holds views
    of them, after one wait for the device (``MetalBuffer.share``); launches
    that use a shared buffer complete before returning, so VTK reads what Tack
    wrote, and the buffers come back as themselves through ``vtk_to_dataset``.

    With ``higher_order``, a dataset of segments, quads or hexahedra (pixels and
    voxels too), or of triangles, tetrahedra or wedges, whose geometry or an ``H1``
    field is above order 1 becomes VTK Lagrange cells (``VTK_LAGRANGE_CURVE``,
    ``_QUADRILATERAL``, ``_HEXAHEDRON``, ``_TRIANGLE``, ``_TETRAHEDRON``,
    ``_WEDGE``) of the largest order: points at the equispaced nodes of
    ``H1`` of that order, each shared node once, the geometry and ``H1`` fields
    evaluated there (exact for fields of that order or below), cell fields as
    cell data. ``vtk_to_dataset`` reads such a grid back into those spaces.
    ``higher_order="bezier"`` writes VTK Bezier cells (``VTK_BEZIER_CURVE``,
    ``_QUADRILATERAL``, ``_HEXAHEDRON``, ``_TRIANGLE``, ``_TETRAHEDRON``,
    ``_WEDGE``) instead: points and point data the
    coefficients of ``H1(order, basis="bernstein")``, which ``vtk_to_dataset``
    reads back as such; ``True`` chooses Bezier cells when the geometry or an
    ``H1`` field is in a Bernstein basis, ``"lagrange"`` Lagrange cells always.
    ``higher_order="quadratic"`` writes VTK's classic quadratic cells (order 2):
    ``VTK_QUADRATIC_QUAD``, ``_HEXAHEDRON`` and ``_WEDGE`` from serendipity (quad8,
    hex20, wedge15), ``VTK_QUADRATIC_TRIANGLE`` and ``_TETRA`` from the full order-2
    space; ``True`` chooses them when a field is serendipity. ``vtk_to_dataset``
    reads such cells into those spaces, the geometry too.

    With ``cell_grid``, the dataset becomes a ``vtkCellGrid`` (VTK's
    discontinuous-Galerkin cells) instead: each shape's cells a DG cell type, the
    geometry its shape attribute and each field an attribute, order 1 in VTK's
    complete basis ``C`` (shared through the points for ``H1``, per cell for
    ``L2``), higher orders in its arbitrary-order basis ``A`` (equispaced,
    lexicographic, as Tack's equispaced nodes; others converted there exactly),
    ``Constant`` fields and values on cells ``constant``; fields of other spaces
    are left out with a warning. The cells come type by type, in Tack's order
    within each. It needs ``vtkDGCell.SetCellSpec``, newer than VTK 9.7.1."""
    from tack.interop.vtk import _lagrange

    if cell_grid:
        from tack.interop.vtk import _cellgrid

        return _cellgrid.to_cell_grid(data)

    if higher_order not in (False, True, "lagrange", "bezier", "quadratic"):
        raise ValueError(f"higher_order is False, True, \"lagrange\", \"bezier\" or "
                         f"\"quadratic\", not {higher_order!r}")
    order = _lagrange.tensor_order(data) if higher_order else 1
    bezier = higher_order == "bezier" or (higher_order is True and _lagrange.bernstein(data))
    quadratic = higher_order == "quadratic" or (higher_order is True and order == 2
                                                and _lagrange.serendipity(data))
    if quadratic and order != 2:
        raise ValueError(f"VTK's quadratic cells are of order 2; this dataset's highest "
                         f"order is {order}")
    out = (_lagrange.to_vtk(data, order, bezier, quadratic) if order > 1
           else _dataset_to_vtk(data, copy))
    time = _time_of(data)
    if time is not None:
        from vtkmodules.vtkCommonDataModel import vtkDataObject

        out.GetInformation().Set(vtkDataObject.DATA_TIME_STEP(), time)
    return out


def _dataset_to_vtk(data, copy=True):
    """``dataset_to_vtk`` before the time is stamped."""
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
        AMRDataSet,
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
    from tack.dataset import OctreeDataSet

    if isinstance(data, OctreeDataSet):
        return _octree_to_htg(data)
    if isinstance(data, AMRDataSet):
        out = _amr_to_vtk(data)
        if out is not None:
            return out
        # VTK's AMR holds images and rectilinear grids: a level per partitioned
        # dataset of a collection, "level<l>".
        return _datasets_to_collection({f"level{level}": pieces
                                        for level, pieces in enumerate(data.levels)})
    if isinstance(data, PartitionedDataSet):
        from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet

        out = vtkPartitionedDataSet()
        # Assigned: the keyword constructor ignores it (VTK 9.7.1).
        out.number_of_partitions = len(data)
        for i, piece in enumerate(data):
            out.SetPartition(i, dataset_to_vtk(piece, copy=copy))
        return out
    share = not copy
    topology = data.topology
    coordinates = data.geometry.values
    # A grid of one point or none has a topology without cells (grid_topology),
    # but is still the image or rectilinear grid its coordinates describe.
    grid = isinstance(topology, StructuredTopology) or (
        topology.num_cells == 0 and isinstance(coordinates, (UniformCoordinates,
                                                             CartesianProduct)))
    if grid and isinstance(coordinates, UniformCoordinates):
        grid = vtkImageData(dimensions=coordinates.dims,
                            origin=tuple(np.asarray(coordinates._origin, np.float64)
                                         + np.asarray(data.origin)),
                            spacing=coordinates._spacing,
                            direction_matrix=tuple(coordinates._matrix.reshape(-1)))
    elif grid and isinstance(coordinates, CartesianProduct):
        grid = vtkRectilinearGrid(dimensions=coordinates.dims)
        axes = (coordinates.x, coordinates.y, coordinates.z)
        if any(data.origin):        # world lines, in double precision
            axes = [a.numpy().astype(np.float64) + o for a, o in zip(axes, data.origin)]
        else:
            axes = [np.array(a.numpy()) for a in axes]
        grid.x_coordinates, grid.y_coordinates, grid.z_coordinates = axes
        for axis in (grid.x_coordinates, grid.y_coordinates, grid.z_coordinates):
            _label(axis, data.unit)
    elif isinstance(topology, StructuredTopology):
        # Explicit coordinates, as a warp or transform leaves; both number the
        # points x fastest.
        grid = vtkStructuredGrid(dimensions=(*topology.point_dims, 1, 1)[:3])
        grid.points = _vtk_points_of(data, share)
    elif isinstance(topology, UnstructuredTopology):
        # Implicit types, offsets and connectivity go across as VTK's constant
        # and affine arrays: a cell type given as a number is a vtkConstantArray.
        grid = vtkUnstructuredGrid()
        grid.points = _vtk_points_of(data, share)
        grid.cells = (_to_vtk_types(topology.types, share),
                      _vtk_cells(topology.offsets, topology.connectivity, share))
    elif isinstance(topology, (PolygonalTopology, PolylineTopology)):
        polygons = isinstance(topology, PolygonalTopology)
        offsets, loops = topology.loops() if polygons else topology.lines()
        grid = vtkUnstructuredGrid()
        grid.points = _vtk_points_of(data, share)
        grid.cells = (7 if polygons else 4, _vtk_cells(offsets, loops, share))
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

    _write_fields(grid, data, share)
    if data.ghosts is not None:
        from vtkmodules.vtkCommonCore import vtkUnsignedCharArray

        for attributes, marks in ((grid.cell_data, data.ghosts.cells),
                                  (grid.point_data, data.ghosts.points)):
            if marks is not None:
                attributes[_GHOST_ARRAY] = vtkUnsignedCharArray(
                    _host_values(marks) if share and _stored_on_host(marks)
                    else np.ascontiguousarray(marks.numpy(), np.uint8))
    if data.global_ids is not None:
        from vtkmodules.vtkCommonCore import vtkIdTypeArray

        for attributes, ids, name in ((grid.cell_data, data.global_ids.cells, "GlobalCellIds"),
                                      (grid.point_data, data.global_ids.points,
                                       "GlobalPointIds")):
            if ids is not None:
                array = vtkIdTypeArray(_host_values(ids) if share and _stored_on_host(ids)
                                       else np.ascontiguousarray(ids.numpy(), np.int64))
                array.SetName(name)                 # vtkGenerateGlobalIds' names
                attributes.SetGlobalIds(array)      # no pythonic form in 9.7.1
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
    out.number_of_partitioned_data_sets = len(datasets)
    for i, (name, pieces) in enumerate(datasets.items()):
        if not isinstance(pieces, PartitionedDataSet):
            raise TypeError(f"collection entry {name!r} is a {type(pieces).__name__}, not "
                            "a PartitionedDataSet")
        out.SetPartitionedDataSet(i, dataset_to_vtk(pieces))
        out.GetMetaData(i).Set(vtkCompositeDataSet.NAME(), str(name))
    return out


def _time_key():
    from vtkmodules.vtkCommonDataModel import vtkDataObject

    return vtkDataObject.DATA_TIME_STEP()


def _time_of(data):
    """The time a Tack dataset (or pieces, a hierarchy, an octree) is at, or
    ``None``."""
    from tack.dataset import AMRDataSet, PartitionedDataSet

    if isinstance(data, AMRDataSet):
        data = data._as_pieces()
    if isinstance(data, PartitionedDataSet):
        times = [piece.time for piece in data if piece.time is not None]
        return times[0] if times else None
    return getattr(data, "time", None)


def _stamp(out, time):
    """Give what ``vtk_to_dataset`` made the time VTK says, where it has none."""
    from tack.dataset import AMRDataSet, OctreeDataSet, PartitionedDataSet

    if isinstance(out, OctreeDataSet):
        out = out.roots
    if isinstance(out, AMRDataSet):
        for _, box in out:
            _stamp(box, time)
    elif isinstance(out, PartitionedDataSet):
        for piece in out:
            _stamp(piece, time)
    elif isinstance(out, dict):
        for value in out.values():
            _stamp(value, time)
    elif out.time is None:
        out.time = time

