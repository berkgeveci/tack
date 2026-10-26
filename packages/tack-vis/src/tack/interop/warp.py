"""Arrays and datasets between Tack and NVIDIA Warp, without copies where they agree.

With the Warp device (``tack.device("warp")``) a Tack array's memory is a
``wp.array`` on Warp's device, and Tack launches on that device's current
Warp stream, the stream a caller's ``wp.launch`` uses. So a program can run
its own Warp kernels and Tack's filters on the same arrays:

    tack.set_default_device("warp")
    wp.launch(my_kernel, n, inputs=[f])
    data.fields["f"] = td.Field(td.H1(data), from_warp(f))
    iso = alg.contour(data, "f", 0.5)
    points = to_warp(iso.geometry.values)          # a wp.array of wp.vec3

`from_warp` and `to_warp` map Warp's vector and matrix types to Tack's
(``wp.vec3`` and ``tack.vec(3, tack.f32)``, ``wp.mat33`` and
``tack.mat(3, 3, tack.f32)``), keep the other side's memory alive for as
long as the result lives, and do not mark the memory as handed outside
Tack: on one device and stream, Tack's launches need not complete before
returning. Under ``wp.ScopedStream`` Tack follows the scope's stream too.

Warp arrays have no read-only flag: a read-only Tack array given to Warp
can be written there.

`to_simdata` and `from_simdata` exchange datasets with warp-simdata
(NVIDIA's simulation-data library on Warp, an optional import): image
data, structured grids, polydata and unstructured grids (polyhedra
included) against Tack's topologies, NODE and ELEMENT fields against point
and cell fields, collections against partitioned datasets. Any of
warp-simdata's other models comes through its ``to_vtk_data_model``.
"""

import math

import numpy as np

import tack
from tack.arrays import (
    ConstantArray,
    CountingArray,
    Permutation,
    Transform,
    View,
    _Implicit,
    _metadata,
    _operand,
    _operands,
    _store,
    dtype_of,
    materialize,
    size_of,
    width_of,
)
from tack.lang.array import Array, SoaArray, _array_view
from tack.lang.types import ElementType, mat, vec


def _warp():
    import warp as wp
    return wp


def _scalar(warp_scalar):
    """The Tack scalar type of a Warp scalar type."""
    from tack.codegen.warp_gen import _WARP_TYPES
    for dtype, name in _WARP_TYPES.items():
        if name == f"wp.{warp_scalar.__name__}":
            return dtype
    raise TypeError(f"no Tack type for Warp's {warp_scalar.__name__}")


def _tack_dtype(warp_dtype):
    """``wp.vec3`` as ``tack.vec(3, tack.f32)``, ``wp.mat33`` as ``tack.mat(3, 3,
    tack.f32)``, a scalar as itself."""
    wp = _warp()
    if wp.types.type_is_vector(warp_dtype):
        return vec(warp_dtype._length_, _scalar(warp_dtype._wp_scalar_type_))
    if wp.types.type_is_matrix(warp_dtype):
        rows, columns = warp_dtype._shape_
        return mat(rows, columns, _scalar(warp_dtype._wp_scalar_type_))
    return _scalar(warp_dtype)


_named = {}


def _warp_dtype(dtype):
    """The Warp type of a Tack scalar, vector or matrix type: Warp's named type
    where it has one (``wp.vec3f``, ``wp.mat33d``), which Warp code compares by
    identity, else the type ``wp.types.vector``/``matrix`` builds."""
    from tack.runtime.warp_backend import _warp_dtype as scalar
    wp = _warp()
    if not isinstance(dtype, ElementType):
        return scalar(dtype)
    if dtype in _named:
        return _named[dtype]
    if len(dtype.shape) == 1:
        built = wp.types.vector(length=dtype.shape[0], dtype=scalar(dtype.scalar_dtype))
    else:
        built = wp.types.matrix(shape=dtype.shape, dtype=scalar(dtype.scalar_dtype))
    prefix = "vec" if len(dtype.shape) == 1 else "mat"
    named = (getattr(wp, n) for n in dir(wp) if n.startswith(prefix))
    _named[dtype] = next((t for t in named if isinstance(t, type)
                          and wp.types.types_equal(t, built)), built)
    return _named[dtype]


def _warp_backend():
    return tack.device("warp")._backend


def from_warp(array, *, writable=True):
    """``array``, a contiguous ``wp.array``, as a Tack array on the Warp device,
    sharing its memory: a write through either is seen by the other.

    The element type follows Warp's (``wp.vec3`` gives ``tack.vec(3,
    tack.f32)``) and the shape is the array's. The Tack array keeps
    ``array`` alive. ``array`` must be on the device Tack's Warp device
    drives (Warp's default device).
    """
    from tack.runtime.warp_backend import WarpBuffer
    wp = _warp()
    if not isinstance(array, wp.array):
        raise TypeError(f"from_warp takes a wp.array, not {type(array).__name__}")
    if not array.is_contiguous:
        raise ValueError("from_warp wraps a contiguous wp.array; copy it with "
                         "wp.clone first")
    backend = _warp_backend()
    if str(array.device) != str(backend.wp_device):
        raise ValueError(f"the array is on Warp's {array.device}, but Tack's Warp device "
                         f"drives {backend.wp_device} (Warp's default device)")
    dtype = _tack_dtype(array.dtype)
    shape = tuple(array.shape)
    scalar = dtype.scalar_dtype if isinstance(dtype, ElementType) else dtype
    count = math.prod(shape) * (dtype.size if isinstance(dtype, ElementType) else 1)
    storage = (count,) if isinstance(dtype, ElementType) else shape
    if count:
        flat = wp.array(ptr=array.ptr, dtype=_warp_dtype(scalar), shape=(count,),
                        capacity=count * np.dtype(scalar.numpy_dtype).itemsize,
                        device=array.device)
    else:
        flat = None                     # nothing to share; WarpBuffer makes its own
    buffer = WarpBuffer(backend, scalar, storage, flat)
    buffer._owner = array               # the memory lives as long as the Tack array
    return _array_view(dtype, shape, buffer, writable=writable, device=backend)


def to_warp(array):
    """``array`` as a ``wp.array`` sharing its memory, with Warp's element type
    (``tack.vec(3, tack.f32)`` gives ``wp.vec3``) at its logical shape.

    An array on the Warp device is its own memory, kept alive by the result.
    An array on another Tack device goes through DLPack, which completes its
    queued launches. An implicit array (``tack.arrays``: ``ConstantArray``,
    ``UniformCoordinates``, ...) has no memory, and is materialized on its
    device first. Structure-of-arrays storage has no single ``wp.array``;
    give its components one at a time.
    """
    wp = _warp()
    if isinstance(array, _Implicit):
        array = materialize(array)
    if isinstance(array, SoaArray):
        raise TypeError("a structure-of-arrays array has no single wp.array; "
                        "give its components (array.components) one at a time")
    if not isinstance(array, Array):
        raise TypeError(f"to_warp takes a Tack array, not {type(array).__name__}")
    shape, dtype, nbytes = tuple(array.shape), array.dtype, array._buffer.nbytes
    backend = array._device
    if backend.name == "warp":
        address, device = array._buffer.address, backend.wp_device
    else:
        source = wp.from_dlpack(array)          # Tack's export completes its launches
        address, device = source.ptr, source.device
        array = (array, source)                 # keep both alive
    if not math.prod(shape):
        return wp.empty(shape, dtype=_warp_dtype(dtype), device=device)

    def keep(_ptr, _size, _held=array):
        """Warp calls this when the result is freed; it holds the Tack array till then."""

    return wp.array(ptr=address, dtype=_warp_dtype(dtype), shape=shape, capacity=nbytes,
                    device=device, deleter=keep)


# ── Datasets with warp-simdata ──────────────────────────────────────────
#
# warp-simdata (NVIDIA's simulation-data library on Warp) reduces any of its
# models -- CGNS, EnSight, OpenFOAM, FLASH, point clouds, ... -- to one of four
# VTK-shaped ones with ``to_vtk_data_model``: image data, structured grid,
# polydata and unstructured grid. Those four map to Tack's topologies as the
# VTK interop maps VTK's, so the exchange is between them. Its points are
# f32, its ids i32 and its cell types u8 in VTK's numbering; where Tack's
# arrays are those and on the Warp device they are shared, otherwise copied.


def _simdata():
    import warp_simdata
    return warp_simdata


def _wp_device(data):
    """Warp's device for ``data``'s arrays: the Warp device's own, else Warp's
    default device (the arrays are then copied through the host)."""
    backend = data.device._backend
    return backend.wp_device if backend.name == "warp" else _warp().get_device()


def _shared_or_copied(values, dtype, device):
    """``values`` (a Tack or implicit array) as a ``wp.array`` of ``dtype`` on
    ``device``: Tack's own memory when it is that already, else a copy."""
    wp = _warp()
    values = materialize(values)
    if (not isinstance(values, SoaArray) and values._device.name == "warp"
            and values.dtype == _tack_dtype(dtype)
            and str(values._device.wp_device) == str(device)):
        return to_warp(values)
    host = np.ascontiguousarray(values.numpy(), dtype=_tack_dtype(dtype).numpy_dtype
                                if not wp.types.type_is_vector(dtype)
                                else _tack_dtype(dtype).scalar_dtype.numpy_dtype)
    if np.issubdtype(host.dtype, np.integer) and host.size:
        wide = np.asarray(values.numpy())
        if wide.max() > np.iinfo(host.dtype).max:
            raise ValueError(f"ids up to {wide.max()} do not fit warp-simdata's int32")
    return wp.array(host, dtype=dtype, device=device)


def _points(data, device):
    """``data``'s positions as warp-simdata takes them: f32 world positions.
    Shared when the geometry is explicit f32 with its origin at zero."""
    wp = _warp()
    values = data.geometry.values
    if not any(data.origin) and isinstance(values, Array) and \
            values.dtype == vec(3, tack.f32) and values.shape == (data.num_points,):
        return _shared_or_copied(values, wp.vec3f, device)
    return wp.array(data.world_positions().astype(np.float32), dtype=wp.vec3f, device=device)


def _add_fields(data, out, device):
    """``data``'s point fields as NODE fields and cell fields as ELEMENT ones;
    an order-2 field gives its values at the points, as the VTK interop does.
    Fields on faces or edges, and L2 ones, have no warp-simdata counterpart."""
    sd = _simdata()
    from tack.dataset import H1, Constant, Values
    from tack.dataset.carry import _point_values

    for name, field in data.fields.items():
        space = field.space
        if isinstance(space, H1) or (isinstance(space, Values) and space.on == "points"):
            values, association = _point_values(field.values, data.num_points), sd.AssociationType.NODE
        elif isinstance(space, Constant) or (isinstance(space, Values) and space.on == "cells"):
            values, association = field.values, sd.AssociationType.ELEMENT
        else:
            continue
        values = materialize(values)
        if isinstance(values, SoaArray):
            # warp-simdata has structure-of-arrays fields too: each component shared.
            scalar = _warp_dtype(values.components[0].dtype)
            out.add_field(name, sd.Field.from_arrays(
                [_shared_or_copied(c, scalar, device) for c in values.components], association))
            continue
        target = _warp_dtype(values.dtype)
        out.add_field(name, sd.Field.from_array(_shared_or_copied(values, target, device),
                                                association))


def to_simdata(data):
    """A Tack ``DataSet`` (or ``PartitionedDataSet``) as a warp-simdata dataset (or
    ``DatasetCollection``) on Warp's device.

    A uniform grid becomes image data (an origin, a spacing and an extent; a
    rotated one becomes a structured grid), any other structured grid a
    structured grid, polygons and polylines polydata, and every other topology
    an unstructured grid, polyhedra with their faces. Point fields become NODE
    fields and cell fields ELEMENT ones. Arrays already f32 positions, i32 ids
    and u8 types on the Warp device are shared; others are copied, positions
    as f32 world positions (Tack's origin added, so far coordinates lose the
    precision the origin kept). Ghost marks, global ids, sets, units and time
    have no counterpart there and are left behind.
    """
    sd = _simdata()
    from warp_simdata.data_models.vtk import (
        image_data,
        polydata,
        structured_grid,
        unstructured_grid,
    )

    from tack.arrays import UniformCoordinates
    from tack.dataset import (
        PartitionedDataSet,
        PolygonalTopology,
        PolyhedralTopology,
        PolylineTopology,
        StructuredTopology,
    )

    wp = _warp()
    if isinstance(data, PartitionedDataSet):
        from warp_simdata.data_models.vtk.conversion import to_common_vtk_data_model
        return sd.DatasetCollection.from_datasets(
            to_common_vtk_data_model([to_simdata(piece) for piece in data]))
    device = _wp_device(data)
    topology, coordinates = data.topology, data.geometry.values
    if isinstance(topology, StructuredTopology):
        dims = list(topology.point_dims) + [1] * (3 - len(topology.point_dims))
        lo = list(topology.offset) + [0] * (3 - len(topology.offset))
        hi = [o + d - 1 for o, d in zip(lo, dims)]
        if isinstance(coordinates, UniformCoordinates) and np.allclose(
                coordinates._matrix, np.eye(3)):
            spacing = np.asarray(coordinates._spacing, np.float64)
            start = np.asarray(coordinates._origin, np.float64) + np.asarray(data.origin)
            out = image_data.create_dataset(
                origin=wp.vec3f(*(start - np.asarray(lo) * spacing)),   # node ijk at 0, as VTK
                spacing=wp.vec3f(*spacing), extent_min=wp.vec3i(*lo),
                extent_max=wp.vec3i(*hi), device=str(device))
        else:
            out = structured_grid.create_dataset(_points(data, device), wp.vec3i(*lo),
                                                 wp.vec3i(*hi))
    elif isinstance(topology, (PolygonalTopology, PolylineTopology)):
        polygons = isinstance(topology, PolygonalTopology)
        offsets, points = topology.loops() if polygons else topology.lines()
        rings = {"polys_offsets" if polygons else "lines_offsets":
                 _shared_or_copied(offsets, wp.int32, device),
                 "polys_connectivity" if polygons else "lines_connectivity":
                 _shared_or_copied(points, wp.int32, device)}
        out = polydata.create_dataset(_points(data, device), **rings)
    elif isinstance(topology, PolyhedralTopology):
        # Faces shared between cells: warp-simdata finds each cell's outward
        # winding itself (its face orientations).
        point_offsets, point_ids = topology.cell_points()
        out = unstructured_grid.create_dataset(
            _points(data, device),
            wp.full(topology.num_cells, 42, dtype=wp.uint8, device=device),
            _shared_or_copied(point_offsets, wp.int32, device),
            _shared_or_copied(point_ids, wp.int32, device),
            faces_offsets=_shared_or_copied(topology.face_offsets, wp.int32, device),
            faces_connectivity=_shared_or_copied(topology.face_points, wp.int32, device),
            face_locations_offsets=_shared_or_copied(topology.cell_offsets, wp.int32, device),
            face_locations_connectivity=_shared_or_copied(topology.cell_faces, wp.int32,
                                                          device))
    else:
        out = unstructured_grid.create_dataset(
            _points(data, device),
            _shared_or_copied(topology.types, wp.uint8, device),
            _shared_or_copied(topology.offsets, wp.int32, device),
            _shared_or_copied(topology.connectivity, wp.int32, device))
    _add_fields(data, out, device)
    return out


def _field_values(field):
    """A warp-simdata field's values as a Tack array on the Warp device: one array
    shared, several (structure of arrays) as ``tack.soa`` of shared components,
    anything else (a NanoVDB volume) through ``to_array``."""
    wp = _warp()
    values = field.get_data()
    if isinstance(values, wp.array):
        return from_warp(values)
    if isinstance(values, (list, tuple)) and values and all(isinstance(v, wp.array)
                                                             for v in values):
        return tack.soa(*(from_warp(v) for v in values))
    return from_warp(field.to_array())


def from_simdata(ds):
    """A warp-simdata dataset (or ``DatasetCollection``) as a Tack ``DataSet`` (or
    ``PartitionedDataSet``) on the Warp device, sharing its arrays.

    Point clouds, surface meshes, OpenFOAM polyMesh and CGNS sections of linear
    elements (one shape or MIXED) are taken as they are: their arrays become
    Tack's, CGNS's 1-based and MIXED connectivities read lazily (see "models
    imported as they are"). Every other model goes through warp-simdata's
    ``to_vtk_data_model`` first. Image data becomes a uniform grid (``offset`` from its extent), a structured
    grid a structured topology over its points, polydata of polygons or
    polylines a polygonal or polyline topology (vertices alone become vertex
    cells; a mix of kinds is refused: split it first), and an unstructured grid
    an unstructured topology, or a polyhedral one when it has polyhedra. NODE
    fields become H1 fields and ELEMENT fields Constant ones. The dataset must
    be on the device Tack's Warp device drives.
    """
    sd = _simdata()
    from warp_simdata.data_models.vtk import (
        image_data,
        polydata,
        structured_grid,
        unstructured_grid,
    )
    from warp_simdata.data_models.vtk.conversion import to_vtk_data_model

    from tack.arrays import ConstantArray, UniformCoordinates
    from tack.dataset import (
        PartitionedDataSet,
        PolygonalTopology,
        PolyhedralTopology,
        PolylineTopology,
        UnstructuredTopology,
    )
    from tack.dataset.dataset import grid_topology

    if isinstance(ds, sd.DatasetCollection):
        pieces = ds.datasets
        if any(not piece.has_field(name) for name in ds.get_field_names() for piece in pieces):
            # Fields of the collection itself: its VTK conversion hands them out.
            pieces = to_vtk_data_model(ds).datasets
        return PartitionedDataSet(from_simdata(piece) for piece in pieces)
    device = tack.device("warp")
    from warp_simdata.data_models.flash import axisymmetric as flash_model
    if ds.data_model is flash_model.DataModel:
        return _flash_amr(ds, device)
    with device:
        direct = _direct_topology(ds, device)
    if direct is not None:
        return _with_fields(ds, *direct, device)
    ds = to_vtk_data_model(ds)
    handle, kind = ds.handle, ds.data_model.DatasetHandle
    if kind is image_data.DatasetHandle:
        lo = np.asarray(handle.extent_min, np.int64)
        dims = np.asarray(handle.extent_max, np.int64) - lo + 1
        spacing = np.asarray(handle.spacing, np.float64)
        start = np.asarray(handle.origin, np.float64) + lo * spacing
        topology = grid_topology(dims.tolist(), offset=lo.tolist(), device=device)
        geometry = UniformCoordinates(dims.tolist(), start, spacing, np.eye(3), device=device)
    elif kind is structured_grid.DatasetHandle:
        lo = np.asarray(handle.extent_min, np.int64)
        dims = np.asarray(handle.extent_max, np.int64) - lo + 1
        topology = grid_topology(dims.tolist(), offset=lo.tolist(), device=device)
        geometry = from_warp(handle.points)
    elif kind is polydata.DatasetHandle:
        def present(offsets):
            return offsets is not None and offsets.shape[0] > 1
        kinds = [k for k in ("verts", "lines", "polys") if present(getattr(handle, f"{k}_offsets"))]
        if len(kinds) > 1:
            raise ValueError(f"polydata of {' and '.join(kinds)} together has no one Tack "
                             "topology: split it by kind first")
        n = ds.get_num_nodes()
        kind = kinds[0] if kinds else "verts"
        offsets = getattr(handle, f"{kind}_offsets")
        points = getattr(handle, f"{kind}_connectivity")
        offsets = from_warp(offsets) if present(offsets) else np.zeros(1, np.int32)
        points = from_warp(points) if present(getattr(handle, f"{kind}_offsets")) \
            else np.zeros(0, np.int32)
        with device:                        # host arrays among them go to the Warp device
            if kind == "polys":
                topology = PolygonalTopology(offsets, points, num_points=n)
            elif kind == "lines":
                topology = PolylineTopology(offsets, points, num_points=n)
            else:
                count = (len(offsets) if isinstance(offsets, np.ndarray) else offsets.shape[0]) - 1
                if points.shape[0] == count:
                    # One point each: every cell a VTK_VERTEX, an implicit array stored
                    # nowhere, as Tack's own point clouds' (uniform_cells).
                    types = ConstantArray(1, count, dtype=tack.u8)
                else:
                    # Poly-vertices among them (VTK_POLY_VERTEX, 2): a type per cell.
                    counts = np.diff(offsets.numpy() if isinstance(offsets, Array) else offsets)
                    types = np.where(counts == 1, 1, 2).astype(np.uint8)
                topology = UnstructuredTopology(types, offsets, points, num_points=n)
        geometry = from_warp(handle.points)
    elif kind is unstructured_grid.DatasetHandle:
        n = ds.get_num_nodes()
        types = from_warp(handle.cell_types)
        present = _present_types(types)
        if 42 in present:
            with device:                    # its new arrays on the Warp device
                topology = _polyhedra_from_simdata(handle, PolyhedralTopology, n)
        else:
            topology = UnstructuredTopology(types, from_warp(handle.cell_offsets),
                                            from_warp(handle.cell_connectivity), num_points=n)
        geometry = from_warp(handle.points)
    else:
        raise TypeError(f"no Tack topology for warp-simdata's {kind.__name__}")
    return _with_fields(ds, topology, geometry, device)


def _with_fields(ds, topology, geometry, device):
    """The dataset of ``topology`` and ``geometry`` with ``ds``'s fields: NODE as H1,
    ELEMENT as Constant, each sharing its storage."""
    sd = _simdata()
    from tack.dataset import H1, Constant, DataSet, Field

    with device:
        data = DataSet(topology, geometry)
        for name in ds.get_field_names():
            field = ds.get_field(name)
            space = H1(data) if field.association == sd.AssociationType.NODE else Constant(data)
            data.fields[name] = Field(space, _field_values(field))
    return data


# Polyhedra on the device. warp-simdata keeps a table of faces and, per cell,
# the faces it has with an orientation each (+1 the loop is outward for that
# cell, -1 inward; 0 if never computed, taken as +1). Tack keeps each face
# once, its loop outward for its side-0 cell, and a side per cell entry.

# The summary's entries: some face has two cells; some face is irregular (no
# cell, more than two, or seen outward or inward twice); some face is seen
# only from inside (its loop must be reversed).
_SHARED, _IRREGULAR, _FLIPS = range(3)


@tack.kernel
def _count_face_slots(cell_faces, orientation, outward, inward):
    for s in range(cell_faces.shape[0]):
        if orientation[s] < 0:
            tack.atomic_add(inward, cell_faces[s], 1)
        else:
            tack.atomic_add(outward, cell_faces[s], 1)


@tack.kernel
def _face_summary(outward, inward, summary):
    for f in range(outward.shape[0]):
        seen = outward[f] + inward[f]
        if seen == 2:
            tack.atomic_max(summary, 0, 1)
        if seen == 0 or seen > 2 or outward[f] > 1 or inward[f] > 1:
            tack.atomic_max(summary, 1, 1)
        if outward[f] == 0 and inward[f] == 1:
            tack.atomic_max(summary, 2, 1)


@tack.kernel
def _slot_sides(cell_faces, orientation, outward, sides):
    # Side 0 where the loop is outward for the cell, or will be once reversed
    # (a face no cell sees outward).
    for s in range(cell_faces.shape[0]):
        if orientation[s] >= 0 or outward[cell_faces[s]] == 0:
            sides[s] = 0
        else:
            sides[s] = 1


@tack.kernel
def _reverse_unseen(face_offsets, face_points, outward, out):
    # Each face's loop, walked backwards from the same first point where no cell
    # sees it outward (as dataset_to_vtk writes a side-1 copy), else as it is.
    for f in range(face_offsets.shape[0] - 1):
        a = face_offsets[f]
        n = face_offsets[f + 1] - a
        for i in range(n):
            if outward[f] == 0:
                out[a + i] = face_points[a + (n - i) % n]
            else:
                out[a + i] = face_points[a + i]


def _polyhedra_from_simdata(handle, PolyhedralTopology, num_points):
    """An unstructured grid of polyhedra as a ``PolyhedralTopology``.

    On the device when warp-simdata's faces are shared between cells (CGNS
    ``NFACE_n``, OpenFOAM): its face and cell-face arrays are Tack's, shared,
    the sides come from its face orientations, and only faces no cell sees
    outward are reversed, into a new face-points array (none, nothing copied).
    A grid whose faces are each cell's own copies (as VTK files have them), or
    irregular, is matched by point sets on the host (``from_cell_faces``)."""
    face_offsets, face_points = from_warp(handle.faces_offsets), from_warp(handle.faces_connectivity)
    cell_offsets = from_warp(handle.face_locations_offsets)
    cell_faces = from_warp(handle.face_locations_connectivity)
    orientation = from_warp(handle.face_orientations)
    num_faces, slots = face_offsets.shape[0] - 1, cell_faces.shape[0]
    outward, inward = tack.zeros((num_faces,), tack.i32), tack.zeros((num_faces,), tack.i32)
    summary = tack.zeros((3,), tack.i32)
    if slots:
        _count_face_slots(cell_faces, orientation, outward, inward)
    if num_faces:
        _face_summary(outward, inward, summary)
    found = summary.numpy()
    if found[_IRREGULAR] or not found[_SHARED]:
        return _polyhedra_on_host(handle, PolyhedralTopology, num_points)
    sides = tack.empty((slots,), tack.u8)
    _slot_sides(cell_faces, orientation, outward, sides)
    if found[_FLIPS]:
        loops = tack.empty_like(face_points)
        _reverse_unseen(face_offsets, face_points, outward, loops)
        face_points = loops
    return PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, sides,
                              num_points=num_points)


def _polyhedra_on_host(handle, PolyhedralTopology, num_points):
    """Each cell's faces walked outward by warp-simdata's face orientations (+1
    keeps a face loop, -1 reverses it), as VTK gives them, so the copies match
    by point set as ``from_cell_faces`` matches VTK's. Read on the host."""
    face_offsets = handle.faces_offsets.numpy()
    face_points = handle.faces_connectivity.numpy()
    where = handle.face_locations_offsets.numpy()
    faces = handle.face_locations_connectivity.numpy()
    orientation = handle.face_orientations.numpy()
    cells = []
    for c in range(len(where) - 1):
        loops = []
        for slot in range(where[c], where[c + 1]):
            ring = face_points[face_offsets[faces[slot]]:face_offsets[faces[slot] + 1]]
            loops.append(ring if orientation[slot] >= 0 else np.roll(ring[::-1], 1))
        cells.append(loops)
    return PolyhedralTopology.from_cell_faces(cells, num_points)


# ── warp-simdata models imported as they are ────────────────────────────
#
# Some of warp-simdata's models are already arrays Tack's topologies take, so
# they come across without its to_vtk_data_model (which copies them, element
# by element for most): point clouds, surface meshes, OpenFOAM polyMesh, and
# CGNS sections of linear elements, one shape or MIXED. CGNS ids are 1-based,
# and a MIXED section's type codes are inline in its connectivity; implicit
# arrays (Transform, Permutation and two below) read them as Tack's, lazily.

#: VTK's cell type for each linear CGNS element type (index: the CGNS type), 0
#: for the rest (higher-order types, which Tack's shapes do not have).
_VTK_BY_CGNS = np.zeros(32, np.uint8)
_VTK_BY_CGNS[[2, 3, 5, 7, 10, 12, 14, 17]] = [1, 3, 5, 9, 10, 14, 13, 12]
#: Corners of each of those CGNS types.
_CORNERS_BY_CGNS = {2: 1, 3: 2, 5: 3, 7: 4, 10: 4, 12: 5, 14: 6, 17: 8}
_CGNS_MIXED = 20


@tack.func
def _zero_based(i):
    return tack.i32(i - 1)


@tack.device_class
class _LessIndex(_Implicit):
    """``base[k] - k``: a CGNS MIXED section's element start offsets as offsets into
    its connectivity without the type codes (one inline code per earlier element)."""

    def __init__(self, base):
        self.base = _operand(base)
        _metadata(self, size_of(self.base), None, tack.i32)

    @tack.func
    def get(self, k):
        return tack.i32(self.base.get(k) - k)


@tack.device_class
class _MixedPoints(_Implicit):
    """A CGNS MIXED section's point ids, 0-based and without its inline type codes:
    entry ``j`` is element ``c``'s, the last whose ``start[c] - c`` is at most
    ``j``, found by a binary search (readers of a connectivity go by entry, not by
    element), at ``connectivity[j + c + 1]``."""

    def __init__(self, connectivity, start, elements):
        self.connectivity, self.start = _operands(connectivity, start)
        self.elements = int(elements)
        _metadata(self, size_of(self.connectivity) - int(elements), None, tack.i32)

    @tack.func
    def get(self, j):
        lo = 0
        hi = self.elements
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.start.get(mid) - mid <= j:
                lo = mid
            else:
                hi = mid
        return tack.i32(self.connectivity.get(j + lo + 1) - 1)


@tack.kernel
def _owner_sides(cell_face_offsets, cell_faces, owner, sides):
    # OpenFOAM winds each face out of its owner: side 0 there, 1 in its neighbour.
    for c in range(cell_face_offsets.shape[0] - 1):
        for s in range(cell_face_offsets[c], cell_face_offsets[c + 1]):
            if owner[cell_faces[s]] == c:
                sides[s] = 0
            else:
                sides[s] = 1


@tack.kernel
def _types_present(types, present):
    for c in range(types.shape[0]):
        tack.atomic_max(present, tack.i32(types[c]), 1)


def _present_types(types):
    """The cell types present, from flags a kernel sets on the device: a kilobyte
    comes to the host, not every cell's type."""
    present = tack.zeros((256,), tack.i32, device=types.device)
    if types.shape[0]:
        _types_present(types, present)
    return {int(t) for t in np.flatnonzero(present.numpy())}


def _direct_topology(ds, device):
    """``(topology, geometry)`` of a model Tack takes as it is, or None."""
    from warp_simdata.data_models.custom import (
        gaussian_point_cloud,
        point_cloud,
        surface_mesh,
        voronoi_point_cloud,
    )
    from warp_simdata.data_models.openfoam import polymesh
    from warp_simdata.data_models.sids import unstructured as sids

    from tack.dataset import PolygonalTopology, PolyhedralTopology, UnstructuredTopology

    model, handle = ds.data_model, ds.handle
    if model in (point_cloud.DataModel, gaussian_point_cloud.DataModel,
                 voronoi_point_cloud.DataModel):
        n = ds.get_num_nodes()
        topology = UnstructuredTopology(ConstantArray(1, n, dtype=tack.u8, device=device),
                                        CountingArray(n + 1, dtype=tack.i32),
                                        CountingArray(n, dtype=tack.i32), num_points=n)
        return topology, from_warp(handle.points)
    if model is surface_mesh.DataModel:
        topology = PolygonalTopology(from_warp(handle.face_offsets),
                                     from_warp(handle.face_vertex_indices),
                                     num_points=ds.get_num_nodes())
        return topology, from_warp(handle.points)
    if model is polymesh.DataModel:
        cell_face_offsets = from_warp(handle.cell_face_offsets)
        cell_faces = from_warp(handle.cell_faces)
        sides = tack.empty((cell_faces.shape[0],), tack.u8)
        if cell_faces.shape[0]:
            _owner_sides(cell_face_offsets, cell_faces, from_warp(handle.owner), sides)
        topology = PolyhedralTopology(from_warp(handle.face_offsets), from_warp(handle.faces),
                                      cell_face_offsets, cell_faces, sides,
                                      num_points=ds.get_num_nodes())
        return topology, from_warp(handle.points)
    if getattr(model, "DatasetHandle", None) is sids.DatasetHandle:
        kind = int(handle.element_type)
        n = int(handle.element_range[1]) - int(handle.element_range[0]) + 1
        connectivity = from_warp(handle.element_connectivity)
        if kind in _CORNERS_BY_CGNS:
            k = _CORNERS_BY_CGNS[kind]
            topology = UnstructuredTopology(
                ConstantArray(int(_VTK_BY_CGNS[kind]), n, dtype=tack.u8, device=device),
                CountingArray(n + 1, step=k, dtype=tack.i32),
                Transform(_zero_based, connectivity, dtype=tack.i32),
                num_points=ds.get_num_nodes())
        elif kind == _CGNS_MIXED:
            start = from_warp(handle.element_start_offset)
            codes = Permutation(View(start, 0, n), connectivity)
            table = tack.array(_VTK_BY_CGNS, device=device)
            topology = UnstructuredTopology(Permutation(codes, table), _LessIndex(start),
                                            _MixedPoints(connectivity, start, n),
                                            num_points=ds.get_num_nodes())
        else:
            return None             # NGON_n, NFACE_n, higher-order: through VTK's models
        return topology, from_warp(handle.grid_coords)
    return None


# ── FLASH: block-structured AMR in (r, z), revolved ─────────────────────
#
# warp-simdata's axisymmetric FLASH model is a 2D block hierarchy in meridional
# coordinates (radius, axial) revolved about the Y axis, nothing revolved stored.
# Tack takes it as an AMRDataSet of 3D structured boxes, one per block, axes (r,
# z, angle): the points revolved where read (_Revolved), each cell's value its
# meridional cell's for every angular sector (_Repeated), the levels refining r
# and z by 2 and keeping the sectors (ratios (2, 2, 1)). Unlike its revolved
# model, whose blocks share no points, the hierarchy knows its siblings and
# levels. A box touching the axis has degenerate hexahedra there (an edge on the
# axis), where its model has wedges; a full revolution repeats its seam's points.


@tack.device_class
class _Revolved(_Implicit):
    """The points of a meridional block revolved about the Y axis: point (i, j, l)
    of a ``(nr + 1, nz + 1, sectors + 1)`` box at ``(r cos a, z, r sin a)``, ``r =
    r0 + i * width / nr``, ``z`` likewise, ``a = a0 + l * step`` (warp-simdata's
    arithmetic): computed where read, stored nowhere."""

    def __init__(self, dims, radial, axial, angular, device):
        self.dims = tuple(int(d) for d in dims)
        self.px, self.py = self.dims[0], self.dims[1]
        # (r0, r width, z0, z width, a0, a step) on the device: the box's device
        # and its scalars' type (f32, as warp-simdata's points).
        self.frame = _store(np.array([*radial, *axial, *angular], np.float64), tack.f32,
                            device)
        _metadata(self, math.prod(self.dims), 3, tack.f32)

    @tack.func
    def get(self, p):
        rest = p // self.px
        i = p - rest * self.px
        j = rest % self.py
        f = self.frame
        r = f[0] + tack.f32(i) * f[1] / tack.f32(self.px - 1)
        z = f[2] + tack.f32(j) * f[3] / tack.f32(self.py - 1)
        a = f[4] + tack.f32(rest // self.py) * f[5]
        return [r * cos(a), z, r * sin(a)]


@tack.device_class
class _Repeated(_Implicit):
    """``base[k % period]`` for ``k < size``: a meridional block's cell values for
    every angular sector (a box's cells run r fastest, the angle slowest)."""

    def __init__(self, base, period, size):
        self.base = _operand(base)
        self.period = int(period)
        _metadata(self, size, width_of(self.base), dtype_of(self.base))

    @tack.func
    def get(self, k):
        return self.base.get(k % self.period)


def _flash_levels(gid):
    """Each block's refinement level, from the GID table's parents (entry 4,
    one-based; none below 1)."""
    parents = gid[:, 4]
    levels = np.zeros(len(gid), np.int64)
    for b in range(len(gid)):
        p = parents[b]
        while p > 0:
            levels[b] += 1
            p = parents[p - 1]
    return levels


def _flash_amr(ds, device):
    """warp-simdata's axisymmetric FLASH dataset as an ``AMRDataSet`` (see above).
    Its element fields (``create_element_field``'s, over the packed source
    blocks) become every box's cell fields, a ``View`` of its block's values
    repeated per sector: nothing copied."""
    sd = _simdata()
    from tack.dataset import AMRDataSet, Constant, DataSet, Field, StructuredTopology

    flash = ds.handle.flash
    nr, nz = int(flash.cell_dims[0]), int(flash.cell_dims[1])
    sectors = int(flash.angular_cells)
    angular = (float(flash.angle_min), float(flash.angle_step))
    bounds = flash.block_bounds.numpy()
    levels = _flash_levels(flash.gid.numpy())
    roots = levels == 0
    r_lo, z_lo = bounds[roots, 0].min(), bounds[roots, 2].min()
    dr0 = (bounds[roots][0, 1] - bounds[roots][0, 0]) / nr
    dz0 = (bounds[roots][0, 3] - bounds[roots][0, 2]) / nz
    packed = {}
    for name in ds.get_field_names():
        data = ds.get_field(name).get_data()
        source = data[0] if isinstance(data, tuple) and data and isinstance(data[0], sd.Field) \
            else None
        if source is None or source.association != sd.AssociationType.ELEMENT:
            continue                    # only the packed source-block element fields
        packed[name] = _field_values(source)
    cells, dims = nr * nz, (nr + 1, nz + 1, sectors + 1)
    boxes = [[] for _ in range(int(levels.max()) + 1)]
    with device:
        for b, (r0, r1, z0, z1) in enumerate(bounds):
            scale = 2.0 ** levels[b]
            offset = (int(np.rint((r0 - r_lo) / dr0 * scale)),
                      int(np.rint((z0 - z_lo) / dz0 * scale)), 0)
            box = DataSet(StructuredTopology(dims, offset=offset),
                          _Revolved(dims, (r0, r1 - r0), (z0, z1 - z0), angular, device))
            for name, values in packed.items():
                box.fields[name] = Field(Constant(box), _Repeated(View(values, b * cells, cells),
                                                                  cells, cells * sectors))
            boxes[levels[b]].append(box)
        return AMRDataSet(boxes, ratios=[(2, 2, 1)] * (len(boxes) - 1))
