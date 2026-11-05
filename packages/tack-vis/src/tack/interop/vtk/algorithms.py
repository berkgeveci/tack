# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tack's filters as VTK algorithms: ``VTKPythonAlgorithmBase`` subclasses that sit in
a VTK pipeline beside VTK's own.

    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
    from tack.interop.vtk.algorithms import Contour

    reader = vtkXMLUnstructuredGridReader(file_name="mesh.vtu")
    surface = (reader >> Contour(field="pressure", value=0.5))()

Each takes a ``vtkDataSet``, a ``vtkPartitionedDataSet``, or a tree of them: a
``vtkMultiBlockDataSet`` or a ``vtkPartitionedDataSetCollection``, as readers of
Exodus, IOSS, CGNS, EnSight and OpenFOAM files give. Its input becomes a Tack
dataset sharing VTK's memory (``vtk_to_dataset(..., copy=False)``), the filter runs
where that memory is -- on the CPU, or on the GPU holding an input of VTK's device
arrays -- and its result goes back to VTK sharing Tack's (``dataset_to_vtk(...,
copy=False)``): no array is copied either way where its values are stored as Tack
keeps them, so a chain of these filters on a GPU stays there. One VTK dataset,
unchanged, is one Tack dataset to every filter that reads it, so what Tack derives
from its topology (locators, faces, point links) is built once: by several filters
on one input, or by one run again after a parameter changes. A field filter's
output is its input's Tack dataset with the new fields, on the same topology.

A partitioned dataset is one Tack ``PartitionedDataSet``: the filter runs on each
piece and gives a ``vtkPartitionedDataSet``, and the filters whose answer needs
every piece (``GhostCells``, ``GenerateGlobalIds``, ``MergePieces``) take them
together. A tree keeps its structure: the filter runs on each partitioned dataset
in it (a ``vtkMultiPieceDataSet`` block too, whole) and on each dataset that is a
block of its own, and the output is a tree of the same shape, with the blocks'
names and a collection's data assembly.

An AMR hierarchy (``vtkOverlappingAMR``) is one Tack ``AMRDataSet``, its boxes marked
with what a finer box covers (``REFINEDCELL``) and what a sibling holds. The filters
whose Tack algorithm takes one take it whole -- ``Probe`` (as its source),
``Statistics``, ``Entropy``, ``Integrate``, ``Voxelize``, ``GhostCells`` and
``AverageDown`` -- so each region counts once; a hierarchy out (``GhostCells``,
``AverageDown``) has the input's boxes. The others run box by box, each level's
boxes as the pieces of a partitioned dataset, leaving out the cells finer boxes
cover, as Tack's filters do on AMR: a filter that computes fields gives the
hierarchy with them, one that makes a dataset a ``vtkPartitionedDataSetCollection``
with a partitioned dataset per level (``level0``, ...), whose contours and slices do
not join where the levels meet. ``GenerateGlobalIds``, ``Streamlines`` and
``Advect``, whose answer box by box would mislead, refuse one.

Filters of points, lines and surfaces (contours, slices, surfaces, edges, tubes,
glyphs, streamlines, point clouds) give ``vtkPolyData``, as VTK's do, so they feed
``vtkPolyDataMapper`` and VTK's polygonal filters; ``Clip`` and ``MergePoints`` give
it for a ``vtkPolyData`` input, as ``vtkClipPolyData`` and ``vtkCleanPolyData``.
Other filters that make a dataset give a ``vtkUnstructuredGrid`` (or keep a
structured grid's type, or a polygonal input's, where the Tack filter does);
filters that compute a field (``Gradients``, ``Elevation``, ``CellQuality``) give
their input's type, shallow-copied, with the new arrays added.

Parameters are properties, given to the constructor or set later; setting one
marks the algorithm modified. A wrapper has a parameter for every argument of its
Tack function but the data and what the wrapper supplies itself (the device, a
locator, the communicator), with the function's defaults. A field is given by
name (``field="p"``), or by name and where VTK keeps it, ``("p", "points")`` or
``("p", "cells")``, for a point and a cell array of one name; or, as VTK filters
take it, with ``SetInputArrayToProcess(index, 0, 0, association, name)``, the
wrapper's field parameters numbered in order. Where an implicit function goes
(``Clip``'s ``by``), VTK's own (``vtkPlane``, ``vtkSphere``, ``vtkBox``,
``vtkCylinder``, ``vtkPlanes``, ``vtkImplicitBoolean``) work as Tack's: the same
function, so the same zero set and values; a ``vtkPlane`` is normalized, its
values those of the unit normal.

The temporal filters (``RunningStatistics``, ``TimeDerivative``, ``Advection``,
``Pathlines``, ``Streaklines``) run in situ, taking the step their input holds at
each execution and giving the output at it, or drive the pipeline through their
input's time steps, one per pass and none twice: ``_Temporal``.

An exception in a filter is VTK's error: printed, the output left empty and
``Update()`` false. It is also kept as the algorithm's ``error`` (``None`` after a
run that succeeded), for Python callers.
"""

import tack

__all__ = [
    "Advect",
    "Advection",
    "AsPolygons",
    "AsPolyhedra",
    "AverageDown",
    "BoundaryFaces",
    "CellCenters",
    "CellGeometry",
    "CellIds",
    "CellQuality",
    "CheckWinding",
    "Clip",
    "Composite",
    "ConnectedComponents",
    "Contour",
    "Convert",
    "Cross",
    "Cylindrical",
    "Deposit",
    "Dot",
    "EdgeLengths",
    "Elevation",
    "Entropy",
    "ExtractCells",
    "ExtractEdges",
    "ExtractExtent",
    "ExtractGeometry",
    "ExtractPoints",
    "ExtractSurface",
    "FaceCenters",
    "FaceGeometry",
    "FeatureEdges",
    "FlowQuantities",
    "GenerateGlobalIds",
    "GhostCells",
    "GhostPoints",
    "Glyphs",
    "Gradients",
    "HaloCatalog",
    "ImplicitValues",
    "Integrate",
    "Interpolate",
    "Jump",
    "LogValues",
    "Magnitude",
    "Mask",
    "MaskPoints",
    "MergePieces",
    "MergePoints",
    "Outliers",
    "Pathlines",
    "PointIds",
    "PointNormals",
    "Probe",
    "Project",
    "RemoveGhosts",
    "RunningStatistics",
    "Shrink",
    "Slice",
    "SlicePlane",
    "SphDensity",
    "SphInterpolate",
    "Spherical",
    "Split",
    "Statistics",
    "Streaklines",
    "Streamlines",
    "Subdivide",
    "SurfaceNormals",
    "TackAlgorithm",
    "Tetrahedralize",
    "Threshold",
    "ThresholdPoints",
    "TimeDerivative",
    "Transform",
    "Triangulate",
    "Tubes",
    "UpwindFlux",
    "ValuesAtCenters",
    "VoxelDownsample",
    "Voxelize",
    "Warp",
]

import inspect

from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkMultiBlockDataSet,
    vtkMultiPieceDataSet,
    vtkOverlappingAMR,
    vtkPartitionedDataSet,
    vtkPartitionedDataSetCollection,
    vtkUniformGridAMR,
)
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm

#: What every input port takes: datasets, partitioned datasets, and trees of them.
_INPUT_TYPES = ("vtkDataSet", "vtkPartitionedDataSet", "vtkMultiBlockDataSet",
                "vtkPartitionedDataSetCollection")
_TREES = (vtkMultiBlockDataSet, vtkPartitionedDataSetCollection)

# ── Tack datasets kept per VTK dataset ──────────────────────────────
#
# A Tack dataset keeps what it derives from its topology -- locators, faces, point
# links, groups -- on the topology, so filters that read one VTK dataset (several
# on one input, or one run again after a parameter changed) build that once when
# they share one Tack dataset. The entry is an attribute of the VTK dataset's
# Python wrapper (``_CACHE``): VTK keeps a wrapper's attributes while its C++ object
# lives, on every later wrapper of it (the one GetInputData gives in the next
# filter), and frees them with the object, so an entry lives exactly as long as its
# dataset, with no observer to call (a DeleteEvent observer crashed Python's
# collector, VTK bugs 19 in the vault). On VTK master a wrapper costs the same however
# many objects keep attributes (0.6 us); VTK 9.7.1 sweeps them at every wrapper (50 us
# with 10,000), as it does the attributes the pythonic API gives every wrapper.
#
# An entry is stale when the dataset's modification time or time step differs from
# what it was cached at (``_key``). The modification time follows VTK's contract: it
# sees a change made with ``Modified()`` (an array's or the points'), arrays added or
# removed, not values written in place without it, nor ``InsertNextCell``. Tack shares
# VTK's memory, so such values reach its fields, but what it derived from the old
# state (a locator, the faces) is kept: call ``Modified()`` after changing a dataset
# in place, as VTK's own filters need. Beside the dataset sharing VTK's memory, an
# entry keeps its copy on each device a filter asked for (``device=``), so the data
# moves there once.
#
# An AMR hierarchy's entry is kept on its metadata (``_holder``), which VTK shares
# between a hierarchy and its shallow copies -- what calling a filter gives is one,
# made of new blocks over the same arrays -- and is valid for its blocks' arrays,
# each by address and modification time (``_signature``): the hierarchy's own
# modification time does not follow its blocks'. A hierarchy ``GhostCells`` grew
# reaches the next filter so, which could not convert it again.

#: The attribute of a VTK dataset's wrapper that holds its entry: ``((modification
#: time, time step), {device: DataSet})``.
_CACHE = "_tack_cache"


def _key(vtk_data):
    """What an entry is valid for: the dataset's modification time and time step; an
    AMR hierarchy's blocks' arrays (``_signature``) and time step."""
    from tack.interop.vtk import _time_key

    info = vtk_data.GetInformation()
    time = info.Get(_time_key()) if info.Has(_time_key()) else None
    if isinstance(vtk_data, vtkUniformGridAMR):
        return _signature(vtk_data), time
    return vtk_data.GetMTime(), time


def _holder(vtk_data):
    """What keeps ``vtk_data``'s entry: its wrapper; an AMR hierarchy's metadata, which
    its shallow copies share."""
    if isinstance(vtk_data, vtkOverlappingAMR):
        meta = vtk_data.GetOverlappingAMRMetaData()
        if meta is not None:
            return meta
    return vtk_data


def _signature(amr):
    """An AMR hierarchy as its entry knows it: each block here, its place and lattice
    (an image's origin and spacing) and its arrays (a rectilinear grid's coordinates
    too) by address and modification time. A shallow copy has the same; a block
    replaced, an array changed with ``Modified()``, added or removed, does not. The
    modification time is VTK's global counter, so a new array at a freed one's
    address differs."""
    from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid

    def stamp(array):
        return array.GetName(), array.GetAddressAsString("vtkObject"), array.GetMTime()

    out = []
    for level, index, block in _blocks(amr):
        if isinstance(block, vtkRectilinearGrid):
            lattice = tuple(stamp(a) for a in (block.GetXCoordinates(),
                                                block.GetYCoordinates(),
                                                block.GetZCoordinates()))
        else:
            lattice = block.GetOrigin(), block.GetSpacing()
        arrays = tuple(stamp(data.GetAbstractArray(i))
                       for data in (block.GetPointData(), block.GetCellData())
                       for i in range(data.GetNumberOfArrays()))
        out.append((level, index, block.GetDimensions(), lattice, arrays))
    return tuple(out)


def _blocks(amr):
    """``(level, index, block)`` for each block an AMR hierarchy holds here (another
    rank's are empty)."""
    for level in range(amr.GetNumberOfLevels()):
        for index in range(amr.GetNumberOfBlocks(level)):
            block = amr.GetDataSetAsCartesianGrid(level, index)
            if block is not None:
                yield level, index, block


def _remember(vtk_data, data):
    """Keep ``data`` as the Tack dataset of ``vtk_data`` as it is now."""
    setattr(_holder(vtk_data), _CACHE, (_key(vtk_data), {None: data}))


def _entry(vtk_data):
    """``vtk_data``'s entry, or ``None`` (none, or a stale one)."""
    entry = getattr(_holder(vtk_data), _CACHE, None)
    return entry if entry is not None and entry[0] == _key(vtk_data) else None


def _as_dataset(vtk_data, device=None, world=None):
    """``vtk_to_dataset(vtk_data, dtype=None, copy=False)``, the same Tack dataset for
    the same VTK dataset unchanged (each piece of a ``vtkPartitionedDataSet``);
    with ``device``, that dataset moved there (the same moved one each time). An
    AMR hierarchy is converted in ``world`` (``_as_hierarchy``)."""
    import tack
    from tack.dataset import PartitionedDataSet
    from tack.interop.vtk import _time_key, vtk_to_dataset

    if isinstance(device, str):
        device = tack.device(device)
    if isinstance(vtk_data, vtkOverlappingAMR):
        return _as_hierarchy(vtk_data, device, world)
    if isinstance(vtk_data, vtkPartitionedDataSet):
        pieces = [vtk_data.GetPartition(i) for i in range(vtk_data.number_of_partitions)]
        data = PartitionedDataSet(_as_dataset(piece, device) for piece in pieces
                                  if piece is not None)
        if vtk_data.GetInformation().Has(_time_key()) and any(
                piece.time is None for piece in data):
            # Stamping the pieces with the whole's time would change kept ones.
            data = vtk_to_dataset(vtk_data, dtype=None, copy=False)
            return data if device is None else data.to_device(device)
        return data
    entry = _entry(vtk_data)
    if entry is None:
        _remember(vtk_data, vtk_to_dataset(vtk_data, dtype=None, copy=False))
        entry = getattr(vtk_data, _CACHE)
    copies = entry[1]
    data = copies[None]
    if device is None or data.device == device:
        return data
    if device not in copies:
        copies[device] = data.to_device(device)
    return copies[device]


def _as_hierarchy(vtk_amr, device, world):
    """A ``vtkOverlappingAMR`` as its ``AMRDataSet``, this rank's boxes sharing VTK's
    memory, the hierarchy in ``world`` (by default VTK's): the same one while the
    hierarchy and its blocks are unchanged, in the same world; with ``device``, its
    boxes moved there."""
    from tack.interop.vtk import vtk_to_dataset

    entry = _entry(vtk_amr)
    kept = entry is not None and (world is None or _same_world(entry[1][None].world, world))
    if world is not None and world.size > 1:
        # Converting is collective: every rank converts again if one does.
        kept = all(world._allgather(kept))
    if not kept:
        _remember(vtk_amr, vtk_to_dataset(vtk_amr, dtype=None, copy=False, world=world))
        entry = _entry(vtk_amr)
    copies = entry[1]
    if device is None:
        return copies[None]
    if device not in copies:
        copies[device] = _hierarchy_on(copies[None], device)
    return copies[device]


def _same_world(a, b):
    """Whether two Worlds are one: the same, or each this process alone (``world()``
    makes one each call without an MPI controller)."""
    return a is b or (a.size == 1 and b.size == 1)


def _hierarchy_on(amr, device):
    """``amr`` with its boxes on ``device``: itself where they are."""
    from tack.dataset import AMRDataSet

    levels = [[box.to_device(device) for box in pieces] for pieces in amr.levels]
    if all(moved is box for moved, (_, box) in zip((b for lv in levels for b in lv), amr)):
        return amr                          # memory the device addresses already
    if amr.layers:
        raise ValueError("an AMR hierarchy with ghost layers stays on its device: give "
                         "GhostCells the device instead")
    return AMRDataSet._grown(amr, levels, amr._extents, amr.layers, amr._fills)


def _device_of(data):
    """The device a Tack dataset, or an AMR hierarchy's first box, is on."""
    from tack.dataset import AMRDataSet

    if isinstance(data, AMRDataSet):
        return next((box.device for _, box in data), None)
    return data.device


def _amr_like(source):
    """An empty ``vtkOverlappingAMR`` of ``source``'s hierarchy, with metadata of its own:
    ``CopyStructure`` shares the input's, which holds the input's entry."""
    from vtkmodules.vtkCommonDataModel import vtkOverlappingAMRMetaData

    out = vtkOverlappingAMR()
    out.CopyStructure(source)
    meta = vtkOverlappingAMRMetaData()
    meta.DeepCopy(source.GetOverlappingAMRMetaData())
    out.SetAMRMetaData(meta)
    return out


def _amr_output(source, amr):
    """A ``vtkOverlappingAMR`` of ``source``'s hierarchy (its boxes, origin, spacings and
    ratios, on every rank) holding ``amr``'s boxes, sharing Tack's memory: this
    rank's blocks of ``source`` in order, as they were converted. ``amr`` is
    remembered as its Tack hierarchy: one ``GhostCells`` grew could not be converted
    again, VTK's blocks with ghost layers being refused."""
    out = _amr_like(source)
    held = {}
    for level, index, _ in _blocks(source):
        held.setdefault(level, []).append(index)
    from tack.interop.vtk import dataset_to_vtk

    for level, pieces in enumerate(amr.levels):
        indices = held.get(level, [])
        if len(indices) != len(pieces):
            raise ValueError(f"level {level} has {len(pieces)} boxes here, the input "
                             f"{len(indices)}")
        for index, box in zip(indices, pieces):
            out.SetDataSet(level, index, dataset_to_vtk(box, copy=False))
    _remember(out, amr)
    return out


class Parameter:
    """A property of an algorithm: read as the value set (or the default), set
    with ``Modified()`` so the pipeline runs again."""

    def __init__(self, default=None, doc=None):
        self.default = default
        self.__doc__ = doc

    def __set_name__(self, owner, name):
        self.name = name
        self.slot = f"_parameter_{name}"

    def __get__(self, algorithm, owner=None):
        if algorithm is None:
            return self
        return algorithm.__dict__.get(self.slot, self.default)

    def __set__(self, algorithm, value):
        algorithm.__dict__[self.slot] = value
        # Counted apart from the modification time, which the pipeline also moves (a
        # new input given with SetInputDataObject): a temporal filter starts over
        # when a parameter changes.
        algorithm.__dict__["_parameter_changes"] = algorithm.__dict__.get(
            "_parameter_changes", 0) + 1
        algorithm.Modified()


# ── VTK's conventions ───────────────────────────────────────────────

#: Where VTK keeps a field, as Tack's kind of space: VTK's point data arrive as
#: ``H1`` fields and its cell data as ``Constant`` ones (``vtk_to_dataset``).
_WHERE = {"points": "H1", "cells": "Constant"}

#: vtkDataObject's field associations, by the names ``_WHERE`` takes.
_ASSOCIATIONS = {0: "points", 1: "cells"}

#: Arguments a wrapper supplies itself, never parameters: the data, the device it
#: runs on (``device``), a Tack locator, the communicator.
_SUPPLIED = ("data", "device", "locator", "world")


def _field_argument(value):
    """A field parameter's value as a Tack field argument: a name as it is, ``(name,
    "points")`` and ``(name, "cells")`` as ``(name, H1)`` and ``(name, Constant)``;
    anything else (a ``Field``, ``(name, space)``) as it is."""
    import tack.dataset as td

    if (isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str)
            and isinstance(value[1], str)):
        if value[1] not in _WHERE:
            raise ValueError(f"a field is at 'points' or 'cells', not {value[1]!r}")
        return value[0], getattr(td, _WHERE[value[1]])
    return value


def _implicit(function):
    """``function`` as a Tack implicit function where it is one of VTK's: a ``vtkPlane``,
    ``vtkSphere``, ``vtkBox``, ``vtkCylinder``, ``vtkPlanes`` or ``vtkImplicitBoolean``
    (union, intersection, difference) of them; anything else as it is. Tack's are
    VTK's functions, so a clip or slice at any value is the same; a ``vtkPlane``'s
    normal is normalized."""
    from vtkmodules.vtkCommonDataModel import vtkImplicitBoolean, vtkImplicitFunction

    import tack.dataset as td

    if not isinstance(function, vtkImplicitFunction):
        return function
    # By VTK's class name: the pythonic API makes some of these Python subclasses (VTK
    # master's vtkPlane), and a C++ subclass may evaluate otherwise.
    name = function.GetClassName()
    if function.GetTransform() is not None:
        raise ValueError(f"a {name} with a transform has no Tack implicit function")
    if name == "vtkPlane":
        return td.Plane(function.GetOrigin(), function.GetNormal())
    if name == "vtkSphere":
        return td.Sphere(function.GetCenter(), function.GetRadius())
    if name == "vtkBox":
        bounds = function.GetBounds()
        return td.Box(bounds[0::2], bounds[1::2])
    if name == "vtkCylinder":
        return td.Cylinder(function.GetCenter(), function.GetAxis(), function.GetRadius())
    if name == "vtkPlanes":
        import numpy as np

        points, normals = function.GetPoints(), function.GetNormals()
        if points is None or normals is None:
            raise ValueError("a vtkPlanes without points and normals has no planes")
        return td.Planes(np.asarray(points.GetData()), np.asarray(normals))
    if name == "vtkImplicitBoolean":
        combine = {vtkImplicitBoolean.VTK_UNION: td.Union,
                   vtkImplicitBoolean.VTK_INTERSECTION: td.Intersection,
                   vtkImplicitBoolean.VTK_DIFFERENCE: td.Difference}.get(
                       function.GetOperationType())
        if combine is None:
            raise ValueError("a vtkImplicitBoolean's union of magnitudes has no Tack "
                             "implicit function")
        functions = function.GetFunction()
        operands = [_implicit(functions.GetItemAsObject(i))
                    for i in range(functions.GetNumberOfItems())]
        if len(operands) < 2:
            raise ValueError(f"a vtkImplicitBoolean of {len(operands)} function(s) has no "
                             "Tack implicit function; give it two or more")
        return combine(*operands)
    raise ValueError(f"a {name} has no Tack implicit function; use vtkPlane, vtkSphere, "
                     "vtkBox, vtkCylinder, vtkPlanes or vtkImplicitBoolean, or a Tack "
                     "implicit function")


#: Worlds made of communicators given as ``world=``, by the communicator's id, each held
#: with its communicator.
_given_worlds = {}


def _in_world(data, world):
    """``data``; a ``PartitionedDataSet`` in ``world``, its pieces this process's."""
    from tack.dataset import PartitionedDataSet

    if isinstance(data, PartitionedDataSet) and data.world is not world:
        return PartitionedDataSet(list(data), world=world)
    return data


def _block_names(source, out, count):
    """Copy each block's metadata (its name) from tree ``source`` to ``out``."""
    for i in range(count):
        if source.HasMetaData(i):
            out.GetMetaData(i).Copy(source.GetMetaData(i))


#: Where each VTK cell type of points, lines and surfaces goes in a vtkPolyData:
#: 0 its vertices, 1 its lines, 2 its polygons.
_POLYDATA_PART = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 7: 2, 8: 2, 9: 2}


def _int64_array(values):
    from vtkmodules.vtkCommonCore import vtkTypeInt64Array

    array = vtkTypeInt64Array()
    array.SetNumberOfTuples(len(values))
    import numpy as np

    np.asarray(array)[:] = values
    return array


def _as_polydata(data):
    """``data`` -- an unstructured grid of vertices, lines and surface cells, or a
    partitioned dataset of them -- as ``vtkPolyData``, as VTK's filters of points,
    lines and surfaces give it: the same points, point data and field data; the
    cells as vertices, lines and polygons. Where they are all of one kind (as nearly
    always) the cells and cell data are the grid's own; otherwise they are taken in
    vertex, line, polygon order, as a polydata numbers them. A pixel becomes a quad
    (its corners reordered). Anything else as it is."""
    import numpy as np
    import vtkmodules.vtkCommonDataModel as dm

    if isinstance(data, vtkPartitionedDataSet):
        out = vtkPartitionedDataSet()
        out.SetNumberOfPartitions(data.GetNumberOfPartitions())
        for i in range(data.GetNumberOfPartitions()):
            part = data.GetPartition(i)
            out.SetPartition(i, _as_polydata(part) if part is not None else None)
        return out
    if not isinstance(data, dm.vtkUnstructuredGrid):
        return data
    out = dm.vtkPolyData()
    if data.GetPoints() is not None:
        out.SetPoints(data.GetPoints())
    out.GetPointData().ShallowCopy(data.GetPointData())
    out.GetFieldData().ShallowCopy(data.GetFieldData())
    n = data.GetNumberOfCells()
    if n == 0:
        return out
    types = np.asarray(data.cell_types, np.int64).reshape(-1)
    part = np.full(max(int(types.max()) + 1, 10), -1, np.int64)
    for t, k in _POLYDATA_PART.items():
        part[t] = k
    parts = part[types]
    if (parts < 0).any():
        bad = sorted(set(types[parts < 0].tolist()))
        raise TypeError(f"cell types {bad} have no place in a vtkPolyData")
    setters = (out.SetVerts, out.SetLines, out.SetPolys)
    present = np.unique(parts)
    if len(present) == 1 and not (types == 8).any():
        setters[int(present[0])](data.GetCells())
        out.GetCellData().ShallowCopy(data.GetCellData())
        return out
    offsets = np.asarray(data.cells.offsets_array, np.int64)
    connectivity = np.asarray(data.cells.connectivity_array, np.int64).copy()
    for c in np.flatnonzero(types == 8):            # a pixel's corners as a quad's
        a = offsets[c]
        connectivity[a + 2], connectivity[a + 3] = connectivity[a + 3], connectivity[a + 2]
    order = np.argsort(parts, kind="stable")
    for k in present:
        cells = order[parts[order] == k]
        sizes = offsets[cells + 1] - offsets[cells]
        rows = np.concatenate([connectivity[offsets[c]:offsets[c + 1]] for c in cells])
        array = dm.vtkCellArray()
        array.SetData(_int64_array(np.concatenate([[0], np.cumsum(sizes)])),
                      _int64_array(rows))
        setters[int(k)](array)
    cell_data = data.GetCellData()
    for i in range(cell_data.GetNumberOfArrays()):
        array = cell_data.GetAbstractArray(i)
        taken = array.NewInstance()
        taken.SetName(array.GetName())
        taken.SetNumberOfComponents(array.GetNumberOfComponents())
        taken.SetNumberOfTuples(n)
        for j, c in enumerate(order):
            taken.SetTuple(j, int(c), array)
        out.GetCellData().AddArray(taken)
    for role in _ROLES:
        active = getattr(cell_data, f"Get{role}")()
        if active is not None and active.GetName():
            getattr(out.GetCellData(), f"SetActive{role}")(active.GetName())
    return out


def _empty_everywhere(vtk_data, world):
    """Whether a partitioned dataset has no partitions on any rank of ``world``."""
    count = sum(vtk_data.GetPartition(i) is not None
                for i in range(vtk_data.GetNumberOfPartitions()))
    if world is not None and world.size > 1:
        return not any(world._allgather(count))
    return count == 0


def _as_partitioned(result):
    """A filter's result for a place that holds a partitioned dataset: as it is if it is
    one, else one partition."""
    if isinstance(result, vtkPartitionedDataSet):
        return result
    out = vtkPartitionedDataSet()
    out.SetNumberOfPartitions(1)
    out.SetPartition(0, result)
    return out


def _as_multipiece(result):
    """A partitioned result for a multiblock's ``vtkMultiPieceDataSet`` block, kept that
    type (a multiblock refuses a ``vtkPartitionedDataSet`` block); anything else as it
    is. By ``isinstance``: the pythonic API's partitioned datasets are subclasses."""
    if not isinstance(result, vtkPartitionedDataSet) or isinstance(result, vtkMultiPieceDataSet):
        return result
    out = vtkMultiPieceDataSet()
    out.SetNumberOfPieces(result.GetNumberOfPartitions())
    for i in range(result.GetNumberOfPartitions()):
        out.SetPiece(i, result.GetPartition(i))
    return out


class TackAlgorithm(VTKPythonAlgorithmBase):
    """A Tack filter as a VTK algorithm. ``FUNCTION`` names the ``tack.dataset`` function
    it runs; the class gets a parameter for each of its arguments but the data and
    what the wrapper supplies (``_SUPPLIED``, ``EXCLUDE``), with the function's
    defaults, unless the class gives one itself. ``FIELDS`` names the parameters
    that are fields, in ``SetInputArrayToProcess``'s order, ``IMPLICIT`` those
    that may be implicit functions, and ``DATASETS`` those that may be VTK datasets
    (an image a filter samples onto), converted as the input is.

    ``compute(piece)`` gives a Tack dataset from a Tack dataset, by default the
    function called with the parameters; ``execute(data)`` runs it on a dataset or
    on each piece of a ``PartitionedDataSet``, and a subclass whose answer needs every
    piece overrides it. ``OUTPUT`` names the output's VTK type for a dataset input:
    a VTK class name (``"vtkPolyData"`` for a filter of points, lines and surfaces,
    as VTK's give them); ``None``, the input's own; ``"structure"``, an image,
    rectilinear or structured grid's own and otherwise an unstructured grid (a
    filter that keeps the cells' structure); ``"points"``, a structured grid for any
    of those three and otherwise an unstructured grid (a filter that moves the
    points); ``"surfaces"``, an unstructured grid. With ``None``, ``"structure"``,
    ``"points"`` and ``"surfaces"`` a ``vtkPolyData`` input gives a ``vtkPolyData``
    (``_gives_polydata``), as VTK's polygonal filters keep it. ``INPUTS`` counts the input ports, the first ``REQUIRED_INPUTS`` of
    them required. ``AMR_PORTS`` are the ports that take an AMR hierarchy
    (``vtkOverlappingAMR``), the Tack function given the ``AMRDataSet`` whole; on
    port 0 of any other filter it runs box by box (``_by_box``) unless ``BY_BOX``
    is false, and anywhere else one is refused (``_check_amr``)."""

    OUTPUT = "vtkUnstructuredGrid"
    INPUTS = 1
    REQUIRED_INPUTS = None
    AMR_PORTS = ()
    BY_BOX = True
    FUNCTION = None
    FIELDS = ()
    IMPLICIT = ()
    DATASETS = ()
    EXCLUDE = ()

    #: The exception the last run raised, or ``None``.
    error = None

    device = Parameter(None, "the device the filter runs on, a name or a tack.Device; "
                             "by default where its input's memory is. The input moves "
                             "there once (VTK's arrays stay as they are)")
    world = Parameter(None, "the processes the pieces are spread over: a "
                            "tack.communication.World or an mpi4py communicator; by "
                            "default VTK's (tack.interop.vtk.world)")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.FUNCTION is None:
            return
        function = cls._function()
        arguments, required = [], set()
        for argument in list(inspect.signature(function).parameters.values())[1:]:
            if (argument.kind in (argument.VAR_POSITIONAL, argument.VAR_KEYWORD)
                    or argument.name in _SUPPLIED or argument.name in cls.EXCLUDE):
                continue
            arguments.append(argument.name)
            if argument.default is argument.empty:
                required.add(argument.name)
            if any(isinstance(vars(klass).get(argument.name), Parameter)
                   for klass in cls.__mro__):
                continue                    # the class's own, or an inherited one
            default = None if argument.default is argument.empty else argument.default
            doc = f"``td.{cls.FUNCTION}``'s ``{argument.name}``" + (
                " (required)" if argument.default is argument.empty else "")
            parameter = Parameter(default, doc)
            parameter.__set_name__(cls, argument.name)
            setattr(cls, argument.name, parameter)
        cls._ARGUMENTS = tuple(arguments)
        cls._REQUIRED = frozenset(required)
        cls._TAKES_WORLD = "world" in inspect.signature(function).parameters

    @classmethod
    def _function(cls):
        import tack.dataset as td

        return getattr(td, cls.FUNCTION)

    def __init__(self, **parameters):
        # The output port says vtkDataObject: RequestDataObject makes the output's
        # type from the input's (a partitioned dataset for pieces), which the
        # executive would replace if the port named a narrower one.
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=self.INPUTS, inputType="vtkDataSet",
                                        nOutputPorts=1, outputType="vtkDataObject")
        known = {name for klass in type(self).__mro__ for name, value in vars(klass).items()
                 if isinstance(value, Parameter)}
        for name, value in parameters.items():
            if name not in known:
                raise TypeError(f"{type(self).__name__} has no parameter {name!r}; "
                                f"it has {sorted(known)}")
            setattr(self, name, value)

    # ── The arguments ───────────────────────────────────────────────

    def _input_array(self, index):
        """The field ``SetInputArrayToProcess`` chose for field parameter ``index``, as a
        field parameter's value, or ``None``."""
        arrays = self.GetInformation().Get(vtkAlgorithm.INPUT_ARRAYS_TO_PROCESS())
        if arrays is None or index >= arrays.GetNumberOfInformationObjects():
            return None
        info = self.GetInputArrayInformation(index)
        if not info.Has(vtkDataObject.FIELD_NAME()):
            return None
        name = info.Get(vtkDataObject.FIELD_NAME())
        where = _ASSOCIATIONS.get(info.Get(vtkDataObject.FIELD_ASSOCIATION()))
        return name if where is None else (name, where)

    def arguments(self):
        """The Tack function's keyword arguments from the parameters: fields resolved
        (``_field_argument``, ``SetInputArrayToProcess``) and VTK's implicit functions
        made Tack's. A required one unset is an error naming it."""
        values = {}
        supplied = self._supplied()
        for name in getattr(self, "_ARGUMENTS", ()):
            value = getattr(self, name)
            if value is None and name in supplied:
                value = supplied[name]
            if name in self.FIELDS:
                if value is None:
                    value = self._input_array(self.FIELDS.index(name))
                value = _field_argument(value)
            elif name == "fields" and isinstance(value, (list, tuple)):
                value = [_field_argument(v) for v in value]
            if name in self.IMPLICIT:
                value = _implicit(value)
            if name in self.DATASETS and isinstance(value, vtkDataObject):
                value = self._dataset(value, self.device)
            if value is None and name in self._REQUIRED:
                raise ValueError(f"{type(self).__name__} needs {name!r}")
            values[name] = value
        if getattr(self, "_TAKES_WORLD", False):
            values["world"] = self._run_world
        return values

    def _world(self):
        """The World this run's pieces are spread over: ``world`` (a World, or an mpi4py
        communicator, kept as one World), else VTK's (``tack.interop.vtk.world``)."""
        from tack.communication import World
        from tack.interop.vtk import world

        given = self.world
        if given is None:
            return world()
        if isinstance(given, World):
            return given
        kept = _given_worlds.get(id(given))
        if kept is None or kept[0] is not given:
            # Held with its communicator: World(comm) duplicates it, so one per comm.
            kept = _given_worlds[id(given)] = (given, World(given))
        return kept[1]

    def _dataset(self, vtk_data, device):
        """``_as_dataset``, a partitioned one or a hierarchy in this run's World."""
        return _in_world(_as_dataset(vtk_data, device, self._run_world), self._run_world)

    def _supplied(self):
        """Arguments this run supplies where their parameters are unset: seeds read from
        a source port."""
        return {}

    # ── The pipeline ────────────────────────────────────────────────

    # The pipeline's information objects are read and set with Get/Set/Append:
    # vtkInformation takes no item assignment in VTK 9.7.1.

    def FillInputPortInformation(self, port, info):
        info.Set(vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), _INPUT_TYPES[0])
        for kind in _INPUT_TYPES[1:]:
            info.Append(vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), kind)
        if port in self.AMR_PORTS:
            # A partitioned collection already, in VTK's model: named for what it is.
            info.Append(vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkOverlappingAMR")
        required = self.INPUTS if self.REQUIRED_INPUTS is None else self.REQUIRED_INPUTS
        if port >= required:
            info.Set(vtkAlgorithm.INPUT_IS_OPTIONAL(), 1)
        return 1

    def _check_amr(self, data, port):
        """Refuse an AMR hierarchy where the filter takes none, whole or box by box.
        VTK makes one a partitioned collection, which went through ``_tree`` level
        by level into an output of the hierarchy's type that could not hold it: an
        empty hierarchy, and no error."""
        if not isinstance(data, vtkUniformGridAMR):
            return
        if not isinstance(data, vtkOverlappingAMR):
            raise TypeError(f"{type(self).__name__}: Tack's AMR is overlapping "
                            f"(vtkOverlappingAMR), not a {data.GetClassName()}")
        if port not in self.AMR_PORTS and not (port == 0 and self.BY_BOX):
            raise TypeError(f"{type(self).__name__} takes no AMR hierarchy on port {port}"
                            + ("" if port else ": box by box, its answer would mislead"))

    def _amr_output_type(self):
        """The output's VTK class for a hierarchy filtered box by box."""
        return vtkPartitionedDataSetCollection

    def _check_inputs(self, inInfo):
        for port in range(self.GetNumberOfInputPorts()):
            for i in range(inInfo[port].GetNumberOfInformationObjects()):
                self._check_amr(self.GetInputData(inInfo, port, i), port)

    def _output_type(self, input):
        """The VTK class of the output for ``input``, a dataset or a partitioned
        dataset (trees keep their type)."""
        import vtkmodules.vtkCommonDataModel as dm

        if isinstance(input, vtkOverlappingAMR):
            # A hierarchy, unless the filter makes a type of its own.
            return vtkOverlappingAMR if self.OUTPUT is None else getattr(dm, self.OUTPUT)
        if isinstance(input, vtkPartitionedDataSet):
            return vtkPartitionedDataSet
        if self._gives_polydata(input):
            return dm.vtkPolyData
        if self.OUTPUT == "surfaces":
            return dm.vtkUnstructuredGrid
        if self.OUTPUT is None:
            return type(input)
        structured = (dm.vtkImageData, dm.vtkRectilinearGrid, dm.vtkStructuredGrid)
        if self.OUTPUT == "structure":
            # By isinstance: a vtkUniformGrid is an image, and the pythonic API's
            # classes are subclasses.
            return next((kind for kind in structured if isinstance(input, kind)),
                        dm.vtkUnstructuredGrid)
        if self.OUTPUT == "points":
            return (dm.vtkStructuredGrid if isinstance(input, structured)
                    else dm.vtkUnstructuredGrid)
        return getattr(dm, self.OUTPUT)

    def _gives_polydata(self, input):
        """Whether the output for ``input`` (a dataset, or a partitioned dataset's first
        partition) is ``vtkPolyData``: a filter of points, lines and surfaces, or one
        that keeps its input's kind given polydata."""
        import vtkmodules.vtkCommonDataModel as dm

        if isinstance(input, vtkPartitionedDataSet):
            # By index: a multiblock's vtkMultiPieceDataSet block does not iterate.
            input = input.GetPartition(0) if input.GetNumberOfPartitions() else None
        if self.OUTPUT == "vtkPolyData":
            return True
        return (self.OUTPUT in (None, "structure", "points", "surfaces")
                and isinstance(input, dm.vtkPolyData))

    def RequestDataObject(self, request, inInfo, outInfo):
        input = self.GetInputData(inInfo, 0, 0)
        if input is None:
            return 0
        if isinstance(input, vtkOverlappingAMR) and 0 in self.AMR_PORTS:
            kind = self._output_type(input)
        elif isinstance(input, vtkOverlappingAMR) and self.BY_BOX:
            kind = self._amr_output_type()
        else:
            # A tree's type; a hierarchy this filter refuses (RequestData) its own.
            kind = type(input) if isinstance(input, _TREES) else self._output_type(input)
        output = outInfo.GetInformationObject(0)
        current = output.Get(vtkDataObject.DATA_OBJECT())
        if current is None or type(current) is not kind:
            output.Set(vtkDataObject.DATA_OBJECT(), kind())
        return 1

    def ProcessRequest(self, request, inInfo, outInfo):
        from vtkmodules.vtkCommonExecutionModel import vtkDemandDrivenPipeline

        if not request.Has(vtkDemandDrivenPipeline.REQUEST_DATA()):
            return VTKPythonAlgorithmBase.ProcessRequest(self, request, inInfo, outInfo)
        self.error = None
        try:
            return VTKPythonAlgorithmBase.ProcessRequest(self, request, inInfo, outInfo)
        except Exception as error:
            import traceback

            # Printed here, as VTK prints an exception that reaches it, and kept without
            # its traceback: that holds this run's frames, and with them this algorithm
            # and the pipeline's information objects, a cycle through VTK objects that
            # crashed Python's collector when it freed them.
            traceback.print_exception(error)
            self.error = _without_tracebacks(error)
            return 0

    def RequestData(self, request, inInfo, outInfo):
        self._run_world = self._world()
        self._check_inputs(inInfo)
        self._prepare(inInfo)
        input = self.GetInputData(inInfo, 0, 0)
        output = self.GetOutputData(outInfo, 0)
        if isinstance(input, vtkOverlappingAMR) and 0 not in self.AMR_PORTS:
            self._set_unit_output(output, self._by_box(input))
        elif isinstance(input, _TREES) and not isinstance(input, vtkOverlappingAMR):
            # Shares the blocks: what was made, and remembered, is the output's.
            output.CompositeShallowCopy(self._tree(input))
        else:
            self._set_unit_output(output, self._unit_of(input))
        _keep_active(self.GetInputData(inInfo, self.ACTIVE_FROM, 0), output,
                     self._made_active())
        return 1

    #: The input port whose active attributes the output keeps (Probe's source's).
    ACTIVE_FROM = 0

    def _made_active(self):
        """``{(where, role): name}``: the arrays this filter makes that VTK's makes
        active (``Scalars``, ``Normals``, ...), set over what the input's give."""
        return {}

    def _by_box(self, vtk_amr):
        """The output for an AMR hierarchy this filter takes box by box: each level's
        boxes, this rank's, as the pieces of a partitioned dataset (``_level``), the
        cells finer boxes cover left out by the filter as their marks say; a
        ``vtkPartitionedDataSetCollection`` of one partitioned dataset per level,
        named ``level<l>``."""
        from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet

        from tack.interop.vtk import dataset_to_vtk

        amr = self._dataset(vtk_amr, self.device)
        out = vtkPartitionedDataSetCollection()
        out.SetNumberOfPartitionedDataSets(amr.num_levels)
        # A box is an image, so only a filter that always gives polydata gives it here.
        polydata = self._gives_polydata(None)
        for level, pieces in enumerate(amr.levels):
            result = self._level(pieces)
            if result is not None:
                result = dataset_to_vtk(result, copy=False)
                result = _as_polydata(result) if polydata else result
            out.SetPartitionedDataSet(level, vtkPartitionedDataSet() if result is None else
                                      _as_partitioned(result))
            out.GetMetaData(level).Set(vtkCompositeDataSet.NAME(), f"level{level}")
        return out

    def _level(self, pieces):
        """The result for one level's boxes, a ``PartitionedDataSet``."""
        return self.execute(pieces)

    def _prepare(self, inInfo):
        """Read what every unit of a run needs from the other ports (a source)."""

    def _set_unit_output(self, output, result):
        # A hierarchy's entry goes with it: on its metadata, which the copy shares.
        _set_output(self, output, result)

    def _tree(self, node):
        """The output for a ``vtkMultiBlockDataSet`` or ``vtkPartitionedDataSetCollection``:
        the same tree, each unit in it (a partitioned dataset, or a block that is one
        dataset) replaced by the filter's result for it, the blocks' names and a
        collection's data assembly kept."""
        if isinstance(node, vtkPartitionedDataSetCollection):
            out = vtkPartitionedDataSetCollection()
            count = node.GetNumberOfPartitionedDataSets()
            out.SetNumberOfPartitionedDataSets(count)
            for i in range(count):
                child = node.GetPartitionedDataSet(i)
                if child is not None:
                    out.SetPartitionedDataSet(i, _as_partitioned(self._unit_of(child)))
            _block_names(node, out, count)
            if node.GetDataAssembly() is not None:
                out.SetDataAssembly(node.GetDataAssembly())
            return out
        out = vtkMultiBlockDataSet()
        count = node.GetNumberOfBlocks()
        out.SetNumberOfBlocks(count)
        for i in range(count):
            child = node.GetBlock(i)
            if child is None:
                continue
            if isinstance(child, _TREES):
                out.SetBlock(i, self._tree(child))
            elif isinstance(child, vtkMultiPieceDataSet):
                out.SetBlock(i, _as_multipiece(self._unit_of(child)))
            else:
                out.SetBlock(i, self._unit(child))
        _block_names(node, out, count)
        return out

    def _unit_of(self, vtk_data):
        """``_unit``, but for a partitioned dataset with no partitions on any rank (the
        node and side sets of an Exodus file, unread) an empty one, without running the
        filter: a filter over pieces has none to start from. Asked of every rank for
        each partitioned dataset, in the tree's order, so collective."""
        if isinstance(vtk_data, vtkPartitionedDataSet) and _empty_everywhere(
                vtk_data, self._run_world):
            return type(vtk_data)()
        return self._unit(vtk_data)

    def _unit(self, vtk_data):
        """The filter's result for one unit: a dataset, a partitioned dataset or an AMR
        hierarchy."""
        from tack.dataset import AMRDataSet
        from tack.interop.vtk import dataset_to_vtk

        result = self.execute(self._dataset(vtk_data, self.device))
        if isinstance(result, AMRDataSet):
            return _amr_output(vtk_data, result)
        out = dataset_to_vtk(result, copy=False)
        return _as_polydata(out) if self._gives_polydata(vtk_data) else out

    # ── The filter ──────────────────────────────────────────────────

    def execute(self, data):
        """The result for a Tack dataset or ``PartitionedDataSet``: ``compute`` on it, or
        on each piece."""
        from tack.dataset import PartitionedDataSet

        if isinstance(data, PartitionedDataSet):
            return PartitionedDataSet([self.compute(piece) for piece in data],
                                      world=data.world)
        return self.compute(data)

    def compute(self, data):
        if self.FUNCTION is None:
            raise NotImplementedError(f"{type(self).__name__} gives no compute()")
        return self._function()(data, **self.arguments())


#: VTK's attribute roles, as vtkDataSetAttributes names them.
_ROLES = ("Scalars", "Vectors", "Normals", "TCoords", "Tensors")


def _datasets_in(data):
    """The datasets in ``data``: itself, or a composite's leaves."""
    if data is None:
        return []
    if isinstance(data, vtkDataObject) and data.IsA("vtkCompositeDataSet"):
        out, iterator = [], data.NewIterator()
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            out.append(iterator.GetCurrentDataObject())
            iterator.GoToNextItem()
        return [d for d in out if d.IsA("vtkDataSet")]
    return [data] if data.IsA("vtkDataSet") else []


def _keep_active(source, out, made):
    """Give every dataset in ``out`` the active attributes VTK's filter would: those of
    ``source`` (its first dataset's) where an array of the name is there and the
    role has none yet -- the scalars a contour carried, the vectors a clip kept --
    then ``made``, the filter's own (``_made_active``). Tack keeps no roles, so the
    arrays it carries come back plain, and mappers, glyphs and lighting read the
    roles."""
    names = {}
    for leaf in _datasets_in(source)[:1]:
        for where, attributes in (("points", leaf.GetPointData()), ("cells", leaf.GetCellData())):
            for role in _ROLES:
                array = getattr(attributes, f"Get{role}")()
                if array is not None and array.GetName():
                    names[where, role] = array.GetName()
    if not names and not made:
        return
    for leaf in _datasets_in(out):
        # Setting a role modifies the dataset: an entry valid before stays valid,
        # the arrays being the same.
        entry = _entry(leaf)
        attributes = {"points": leaf.GetPointData(), "cells": leaf.GetCellData()}
        for (where, role), name in names.items():
            target = attributes[where]
            if getattr(target, f"Get{role}")() is None and target.GetAbstractArray(name):
                getattr(target, f"SetActive{role}")(name)
        for (where, role), name in made.items():
            if attributes[where].GetAbstractArray(name):
                getattr(attributes[where], f"SetActive{role}")(name)
        if entry is not None:
            setattr(_holder(leaf), _CACHE, (_key(leaf), entry[1]))


def _without_tracebacks(error, seen=None):
    """``error`` with its traceback, and those of the exceptions it was raised from or
    during, dropped."""
    seen = set() if seen is None else seen
    if error is not None and id(error) not in seen:
        seen.add(id(error))
        error.__traceback__ = None
        _without_tracebacks(error.__cause__, seen)
        _without_tracebacks(error.__context__, seen)
    return error


def _set_output(algorithm, output, result):
    """Make ``output`` ``result``, a VTK dataset of its type."""
    if not result.IsA(output.GetClassName()):
        # ShallowCopy between types copies nothing, silently.
        raise TypeError(f"{type(algorithm).__name__} made a {result.GetClassName()}, where "
                        f"its output is a {output.GetClassName()}")
    output.ShallowCopy(result)


class _FieldAlgorithm(TackAlgorithm):
    """A filter that computes fields on its input's points or cells: the output is
    its input, shallow-copied, with the new arrays added. ``new_fields(piece)`` gives
    ``{name: Field}``: by default the function's result, a dict of fields as it is
    or one field named by the ``name`` parameter."""

    OUTPUT = None

    def _amr_output_type(self):
        return vtkOverlappingAMR

    def _by_box(self, vtk_amr):
        """The hierarchy with the fields computed on each box, its blocks shallow copies
        of the input's with the new arrays; its Tack hierarchy the input's with them,
        on the same boxes and marks."""
        from tack.dataset import AMRDataSet

        amr = self._dataset(vtk_amr, self.device)
        out = _amr_like(vtk_amr)
        levels = [[] for _ in range(amr.num_levels)]
        whole = True
        for (level, index, block), (_, box) in zip(_blocks(vtk_amr), amr):
            one, kept = self._with_fields(block, box)
            out.SetDataSet(level, index, one)
            levels[level].append(kept)
            whole &= kept is not None
        if whole:
            _remember(out, AMRDataSet._grown(amr, levels, amr._extents, amr.layers,
                                             amr._fills))
        return out

    def _unit(self, vtk_data):
        data = self._dataset(vtk_data, self.device)
        if isinstance(vtk_data, vtkPartitionedDataSet):
            # Iterating a partitioned dataset gives its non-empty partitions, the
            # pieces vtk_to_dataset made, in order.
            out = vtk_data.NewInstance()
            for block, piece in zip(vtk_data, data):
                one, kept = self._with_fields(block, piece)
                out.append(one)
                # Appended as it is: the output's partition is this object.
                if kept is not None:
                    _remember(one, kept)
            return out
        out, kept = self._with_fields(vtk_data, data)
        if kept is not None:
            _remember(out, kept)
        return out

    def _set_unit_output(self, output, result):
        if isinstance(result, vtkPartitionedDataSet):
            # Shares the partitions, which were remembered.
            output.CompositeShallowCopy(result)
            return
        output.ShallowCopy(result)
        kept = _entry(result)
        if kept is not None:
            # The output is a new object over the same arrays: the same Tack dataset.
            _remember(output, kept[1][None])

    def _with_fields(self, block, piece):
        """A new VTK dataset of ``block``'s structure and arrays, and the fields
        computed on ``piece``, its Tack dataset, added to it; and the Tack dataset
        of that, ``piece`` with the fields on the same topology (and so what was
        derived from it), or ``None`` where a fresh conversion would differ: a new
        name a field of ``piece`` already has."""
        from tack.dataset import H1, Constant, DataSet, Values
        from tack.interop.vtk import _label, _set_values, _to_vtk_values

        out = block.NewInstance()
        out.ShallowCopy(block)
        fields = self.new_fields(piece)
        for name, field in fields.items():
            space = field.space
            on_points = isinstance(space, H1) or (isinstance(space, Values)
                                                  and space.on == "points")
            if not on_points and not (isinstance(space, Constant)
                                      or (isinstance(space, Values) and space.on == "cells")):
                raise TypeError(f"{name!r} is on {space!r}, which VTK keeps nowhere")
            attributes = out.point_data if on_points else out.cell_data
            values, share = _vtk_values_of(block, piece, field, name) if on_points else \
                (field.values, True)
            _set_values(attributes, name, _to_vtk_values(values, share=share, name=name))
            _label(attributes[name], field.unit)
        if any(name in piece.fields for name in fields):
            return out, None
        kept = DataSet(piece.topology, piece.geometry, fields={**piece.fields, **fields},
                       sets=piece.sets, ghosts=piece.ghosts, global_ids=piece.global_ids,
                       **piece._frame())
        return out, kept

    def new_fields(self, data):
        if self.FUNCTION is None:
            raise NotImplementedError(f"{type(self).__name__} gives no new_fields()")
        from tack.dataset import DataSet

        result = self._function()(data, **self.arguments())
        if hasattr(result, "_asdict"):              # CellGeometry, FlowQuantities, ...
            result = result._asdict()
        if isinstance(result, DataSet):             # the input with fields added
            return {name: result.fields[name] for name in self.RESULT_FIELDS}
        return result if isinstance(result, dict) else {self._output_name(): result}

    #: The fields a function that gives its input back with fields added adds.
    RESULT_FIELDS = ()

    def _output_name(self):
        """The new field's name: the ``name`` parameter, or by default the field the
        filter reads (``field``)."""
        name = getattr(self, "name", None)
        if name is not None:
            return name
        field = getattr(self, "field", None)
        if isinstance(field, str):
            return field
        if isinstance(field, tuple) and field and isinstance(field[0], str):
            return field[0]
        raise ValueError(f"{type(self).__name__} needs a name for its new field")


def _vtk_values_of(block, piece, field, name):
    """``(values, share)``: a point field of ``piece``, the Tack dataset of the VTK
    dataset ``block``, as values at ``block``'s points. On linear cells those are
    the field's own values, shared. On VTK's higher-order cells, whose points are
    the cells' nodes, the field goes into the space the import made (``project``:
    exactly, for an ``H1`` field of that order or below and for values on Tack's
    points, the corners) and each VTK point takes its node's value, a copy; the
    import recorded which node each point is (``_lagrange.VTK_NODES``, on the
    topology, or on the topology of the dataset kept for ``block`` where ``piece``
    is a copy on another device)."""
    import numpy as np

    from tack import arrays
    from tack.dataset import H1, project
    from tack.interop.vtk._lagrange import VTK_NODES

    nodes = getattr(piece.topology, VTK_NODES, None)
    if nodes is None:
        entry = _entry(block)
        if entry is not None:
            nodes = getattr(entry[1][None].topology, VTK_NODES, None)
    count = block.GetNumberOfPoints()
    if nodes is None:
        if count != piece.num_points:
            raise ValueError(f"{name!r} has values at {piece.num_points} points, and the VTK "
                             f"dataset {count}")
        return field.values, True
    order, basis, node_of = nodes
    if len(node_of) != count:
        raise ValueError(f"{name!r}: the VTK dataset has {count} points, its cells' nodes "
                         f"{len(node_of)}")
    space = H1(piece, order=order, basis=basis) if order > 1 else H1(piece)
    if not (isinstance(field.space, H1) and field.space.order == space.order
            and field.space.basis == space.basis):
        kind = arrays.dtype_of(field.values)
        if getattr(kind, "scalar_dtype", kind).numpy_dtype.kind != "f":
            raise TypeError(f"{name!r} holds integers at the cells' corners, and VTK's "
                            "higher-order cells have nodes between them, which an integer "
                            "field has no value at")
        field = project(piece, field, space)
    values = np.asarray(arrays.to_host(field.values))
    out = values[np.maximum(node_of, 0)]
    out[node_of < 0] = 0                    # a point no cell uses
    import tack

    return tack.array(np.ascontiguousarray(out)), False


class _Whole(TackAlgorithm):
    """A filter whose Tack function takes a ``PartitionedDataSet`` itself: a partitioned
    input goes to it whole, not piece by piece."""

    def execute(self, data):
        return self.compute(data)


# ── Filters that make a dataset ─────────────────────────────────────

def _component_of(data, key, component):
    """The field ``key`` of ``data``, or of a vector field one component, as VTK's
    filters take one: ``component`` (by default the first), or ``"magnitude"``. A
    component is a view of the vector's values (``ExtractComponent``), nothing
    copied."""
    from tack.arrays import ExtractComponent, width_of
    from tack.dataset import Field, magnitude

    field = data.fields[key]
    width = width_of(field.values)
    if not width:
        if component not in (None, 0):
            raise ValueError(f"{key!r} is a scalar field: it has no component {component!r}")
        return key
    if component == "magnitude":
        return magnitude(data, key)
    c = 0 if component is None else int(component)
    if not 0 <= c < width:
        raise ValueError(f"{key!r} has {width} components, not a component {c}")
    return Field(field.space, ExtractComponent(field.values, c), field.unit)


class _OfAComponent(TackAlgorithm):
    """A filter of a scalar ``field`` that takes a vector field's component, as VTK's
    do (a vector field's first by default)."""

    component = Parameter(None, "a vector field's component (by default the first), or "
                                "'magnitude'")

    def compute(self, data):
        arguments = self.arguments()
        arguments["field"] = _component_of(data, arguments["field"], self.component)
        return self._function()(data, **arguments)


class Contour(_OfAComponent):
    """``td.contour``: the isosurface (isolines in 2D) of a point field at ``value`` (a
    number or a sequence of them); of a vector field, its ``component`` (the first
    by default, as VTK's ``ArrayComponent``). VTK's ``vtkContourFilter``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "contour"
    FIELDS = ("field",)

    field = Parameter(None, "the point field's name")
    value = Parameter(0.0, "the isovalue, or a sequence of them")


class SlicePlane(TackAlgorithm):
    """``td.slice_plane``: the cut by the plane through ``origin`` normal to ``normal``.
    VTK's ``vtkCutter`` with a ``vtkPlane``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "slice_plane"

    origin = Parameter((0.0, 0.0, 0.0), "a point of the plane")
    normal = Parameter((1.0, 0.0, 0.0), "the plane's normal")


class Clip(TackAlgorithm):
    """``td.clip``: the part of the cells where ``by`` (a point field, or an implicit
    function: Tack's, such as ``td.Plane``, or VTK's, such as ``vtkPlane``) is above
    ``value`` (below with ``inside``). VTK's ``vtkTableBasedClipDataSet``."""

    OUTPUT = "surfaces"
    FUNCTION = "clip"
    FIELDS = ("by",)
    IMPLICIT = ("by",)

    by = Parameter(None, "a point field, or an implicit function (Tack's or VTK's)")
    value = Parameter(0.0, "the value the clip is at")
    inside = Parameter(False, "keep the part below the value instead")


class Threshold(_OfAComponent):
    """``td.threshold``: the cells whose ``field`` lies in ``[lower, upper]`` (a point
    field at all of a cell's points, or any with ``all_points=False``); of a vector
    field, its ``component`` (the first by default) or its magnitude
    (``component="magnitude"``), as VTK's selected-component mode. VTK's
    ``vtkThreshold``."""

    FUNCTION = "threshold"
    FIELDS = ("field",)

    field = Parameter(None, "the field")
    lower = Parameter(float("-inf"), "the lowest value kept")
    upper = Parameter(float("inf"), "the highest value kept")
    all_points = Parameter(True, "a point field: every point of a cell in range")


class ExtractSurface(_Whole):
    """``td.extract_surface``: the boundary faces of the cells, as polygons. Of pieces,
    the whole's surface: faces the pieces share are left out. VTK's
    ``vtkDataSetSurfaceFilter``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "extract_surface"
    EXCLUDE = ("faces",)


class Tetrahedralize(TackAlgorithm):
    """``td.tetrahedralize``: every 3D cell as tetrahedra, conforming across faces.
    VTK's ``vtkDataSetTriangleFilter``."""

    FUNCTION = "tetrahedralize"


class SurfaceNormals(TackAlgorithm):
    """``td.surface_normals``: a surface with ``normals`` on its points and
    ``cell_normals`` on its cells, its polygons wound consistently. VTK's
    ``vtkPolyDataNormals``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "surface_normals"

    consistent = Parameter(True, "wind the polygons consistently first")

    def _made_active(self):
        return {("points", "Normals"): "normals"}


class ConnectedComponents(_Whole):
    """``td.connected_components``: ``region`` on the cells and ``point_region`` on the
    points, cells connected through shared points (or, with ``radius``, points
    within it). Of pieces, regions are joined across them and numbered over the
    whole. VTK's ``vtkConnectivityFilter``."""

    OUTPUT = None
    FUNCTION = "connected_components"

    def _unit(self, vtk_data):
        # The input with the two arrays: its own type, as VTK's filter keeps it
        # (dataset_to_vtk would make a polygonal input an unstructured grid).
        result = self.execute(self._dataset(vtk_data, self.device))
        return _arrays_onto(vtk_data, result, ["region", "point_region"], keep=True)


class _Seeded(_Whole):
    """A filter of particles from seeds: ``seeds``, or where that is unset, the points
    of the data on input port 1 (``SetSourceConnection``, ``SetSourceData``), as
    ``vtkStreamTracer`` takes its seeds. Of pieces, the particles go from piece to
    piece (and rank to rank), each path traced over the whole."""

    INPUTS = 2
    REQUIRED_INPUTS = 1
    # Box by box, a path would end where a finer box covers the coarse one.
    BY_BOX = False

    def SetSourceConnection(self, port):
        self.SetInputConnection(1, port)

    def SetSourceData(self, data):
        self.SetInputDataObject(1, data)

    def _prepare(self, inInfo):
        self._seeds = None
        if self.seeds is None and inInfo[1].GetNumberOfInformationObjects():
            self._seeds = _points_of(self.GetInputData(inInfo, 1, 0))

    def _supplied(self):
        return {"seeds": getattr(self, "_seeds", None)}


def _points_of(vtk_data):
    """The world positions of the points of ``vtk_data`` -- a dataset, a partitioned
    dataset or a tree, every dataset in it -- as one ``(n, 3)`` double array."""
    import numpy as np

    if isinstance(vtk_data, (*_TREES, vtkPartitionedDataSet)):
        leaves = []
        iterator = vtk_data.NewIterator()
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            leaves.append(iterator.GetCurrentDataObject())
            iterator.GoToNextItem()
    else:
        leaves = [vtk_data]
    positions = [np.asarray(_as_dataset(leaf).world_positions(), np.float64).reshape(-1, 3)
                 for leaf in leaves if leaf is not None and leaf.GetNumberOfPoints()]
    return np.concatenate(positions) if positions else np.zeros((0, 3))


class Streamlines(_Seeded):
    """``td.streamlines``: the paths of ``seeds`` (an ``(n, 3)`` array, or the points of
    the data on port 1) through the velocity ``field``, up to ``steps`` steps of
    ``step_size``. VTK's ``vtkStreamTracer``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "streamlines"
    FIELDS = ("field",)

    field = Parameter(None, "the velocity field")
    seeds = Parameter(None, "the seed points, an (n, 3) array; by default the points "
                            "of the data on port 1")
    step_size = Parameter(0.01, "the step, in time (or in length with step_unit)")
    steps = Parameter(1000, "the most steps a path takes")
    integrator = Parameter("rk4", "rk4, rk45 or euler")
    step_unit = Parameter("time", "time, or length for steps of a fixed arc length")


class Probe(TackAlgorithm):
    """``td.probe``, as ``vtkProbeFilter``: the fields of the source -- the data on
    input port 1 (``SetSourceConnection``, ``SetSourceData``) -- at the points of the
    input (port 0), as point data on the input's structure, with ``valid`` (0 where
    no source cell holds the point). A partitioned source is probed as a whole, and a
    tree as the partitioned dataset of its blocks; an AMR hierarchy at each point's
    finest box (as ``vtkAMRResampleFilter``, onto any points); a tree on port 0 keeps
    its structure. It runs on the source's device unless ``device`` says otherwise,
    the input moved there."""

    OUTPUT = "structure"
    INPUTS = 2
    AMR_PORTS = (1,)
    ACTIVE_FROM = 1                     # the probed fields are the source's
    FUNCTION = "probe"
    EXCLUDE = ("where",)

    fields = Parameter(None, "the source fields to probe (by default every one)")

    def SetSourceConnection(self, port):
        self.SetInputConnection(1, port)

    def SetSourceData(self, data):
        self.SetInputDataObject(1, data)

    def _prepare(self, inInfo):
        import tack.dataset as td

        source = self.GetInputData(inInfo, 1, 0)
        if isinstance(source, _TREES) and not isinstance(source, vtkOverlappingAMR):
            # Every dataset in the tree, as the pieces of one partitioned dataset.
            leaves = vtkPartitionedDataSet()
            iterator = source.NewIterator()
            iterator.InitTraversal()
            while not iterator.IsDoneWithTraversal():
                leaves.SetPartition(leaves.GetNumberOfPartitions(),
                                    iterator.GetCurrentDataObject())
                iterator.GoToNextItem()
            source = leaves
        self._source = self._dataset(source, self.device)
        self._where_device = self.device
        if self._where_device is None and not isinstance(self._source, td.PartitionedDataSet):
            self._where_device = _device_of(self._source)

    def _level(self, pieces):
        import tack.dataset as td

        arguments = self.arguments()
        return td.PartitionedDataSet(
            [td.probe(self._source, piece if self._where_device is None
                      else piece.to_device(self._where_device), **arguments)
             for piece in pieces], world=pieces.world)

    def _unit(self, vtk_data):
        import tack.dataset as td
        from tack.interop.vtk import dataset_to_vtk

        where = self._dataset(vtk_data, self._where_device)
        out = dataset_to_vtk(td.probe(self._source, where, **self.arguments()), copy=False)
        return _as_polydata(out) if self._gives_polydata(vtk_data) else out


# ── Filters that compute fields ─────────────────────────────────────

class Gradients(_FieldAlgorithm):
    """``td.gradients``: the gradient of ``field`` as a new array ``name``, on the
    cells (or the points, ``at="points"``). VTK's ``vtkGradientFilter``."""

    FUNCTION = "gradients"
    FIELDS = ("field",)

    field = Parameter(None, "the field")
    at = Parameter("cells", "cells or points")
    name = Parameter("gradient", "the new array's name")


class Elevation(_FieldAlgorithm):
    """``td.point_elevation``: each point's place along ``low_point`` to ``high_point``,
    scaled to ``range``, as a point array ``name``. VTK's ``vtkElevationFilter``."""

    FUNCTION = "point_elevation"

    low_point = Parameter((0.0, 0.0, 0.0), "where the elevation is range[0]")
    high_point = Parameter((0.0, 0.0, 1.0), "where it is range[1]")
    range = Parameter((0.0, 1.0), "the elevation's range")
    name = Parameter("elevation", "the new array's name")

    def _made_active(self):
        return {("points", "Scalars"): self._output_name()}


class CellQuality(_FieldAlgorithm):
    """``td.cell_quality``: each measure (Verdict's, generalized) as a cell array of its
    name. VTK's ``vtkMeshQuality``."""

    FUNCTION = "cell_quality"

    measures = Parameter(None, "the measures' names (by default each shape's own)")


# ── More base classes ───────────────────────────────────────────────

class _OnFaces(TackAlgorithm):
    """A filter that computes fields on faces (``ON = "faces"``) or edges (``"edges"``),
    which VTK keeps nowhere on the cells: the output is every face as a polygon (a 2D
    mesh's edges as lines), or every edge as a line, numbered as the faces or edges,
    the fields its cell data; the input's point fields stay on the points and its
    cell fields go to the faces, each from its side 0 cell. ``name`` names a single
    new field."""

    ON = "faces"

    def compute(self, data):
        import numpy as np

        import tack.dataset as td

        result = self._function()(data, **self.arguments())
        new = result._asdict() if hasattr(result, "_asdict") else {self.name: result}
        fields = {**data.fields}
        for name, field in new.items():
            fields[name if name not in fields else (name, field.space)] = field
        with_fields = td.DataSet(data.topology, data.geometry, fields=fields, sets=data.sets,
                                 ghosts=data.ghosts, global_ids=data.global_ids,
                                 **data._frame())
        if self.ON == "edges":
            return td.extract_edges(with_fields)
        faces = np.arange(data.topology.faces().num_faces)
        return td.extract_surface(with_fields, faces=faces)


# ── Cells kept whole ────────────────────────────────────────────────

class ExtractCells(TackAlgorithm):
    """``td.extract_cells``: the cells given by ``cells`` -- ids (an array), an integer
    cell field's name, or a cell set's name -- with only the points they use. VTK's
    ``vtkExtractCells``."""

    FUNCTION = "extract_cells"


class ExtractGeometry(TackAlgorithm):
    """``td.extract_geometry``: the whole cells inside an implicit ``function`` (Tack's
    or VTK's), or outside it, or cut by it. VTK's ``vtkExtractGeometry``."""

    FUNCTION = "extract_geometry"
    IMPLICIT = ("function",)


class ExtractExtent(TackAlgorithm):
    """``td.extract_extent``: a box of a structured grid's points (``extent``, per axis a
    (low, high) pair or VTK's flat six), every ``rate``-th. VTK's ``vtkExtractGrid``."""

    OUTPUT = "structure"
    FUNCTION = "extract_extent"


class Mask(TackAlgorithm):
    """``td.mask``: every ``stride``-th cell, with only the points they use (Viskores'
    ``Mask``)."""

    FUNCTION = "mask"


# ── Cutting ─────────────────────────────────────────────────────────

class Slice(TackAlgorithm):
    """``td.slice``: where an implicit ``function`` (Tack's or VTK's) is ``value``.
    VTK's ``vtkCutter``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "slice"
    IMPLICIT = ("function",)


# ── Surfaces and edges ──────────────────────────────────────────────

class BoundaryFaces(TackAlgorithm):
    """``td.boundary_faces``: the boundary, a face set, given as its faces' polygons (a 2D
    mesh's boundary edges as lines): ``extract_surface`` of that set, which a 2D mesh's
    ``ExtractSurface`` passes through."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "boundary_faces"

    def compute(self, data):
        import tack.dataset as td

        return td.extract_surface(data, faces=td.boundary_faces(data))


class ExtractEdges(TackAlgorithm):
    """``td.extract_edges``: the edges (every one, or ``edges``' ids) as line cells.
    VTK's ``vtkExtractEdges``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "extract_edges"


class FeatureEdges(TackAlgorithm):
    """``td.feature_edges``: a surface's boundary, feature, non-manifold and manifold
    edges as line cells, with ``kind``. VTK's ``vtkFeatureEdges``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "feature_edges"


# ── Shapes for drawing ──────────────────────────────────────────────

class Tubes(TackAlgorithm):
    """``td.tubes``: lines as tubes of ``radius`` (scaled by the ``radius_by`` field),
    with outward normals. VTK's ``vtkTubeFilter``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "tubes"
    FIELDS = ("radius_by",)

    def _made_active(self):
        return {("points", "Normals"): "normals"}


class Glyphs(TackAlgorithm):
    """``td.glyphs``: an arrow at each point along the ``vectors`` field, scaled by
    ``scale`` (and the ``scale_by`` field). VTK's ``vtkGlyph3D``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "glyphs"
    FIELDS = ("vectors", "scale_by")


# ── Connectivity and winding ────────────────────────────────────────

class CheckWinding(_FieldAlgorithm):
    """``td.check_winding``: the cells whose faces do not wind consistently, a cell set
    of polyhedra or polygons, given as a cell array ``name``, 1 for those cells."""

    FUNCTION = "check_winding"

    name = Parameter("inconsistent", "the new cell array's name")

    def new_fields(self, data):
        import tack
        from tack.dataset import Constant, Field

        flags = self._function()(data).flags()
        return {self.name: Field(Constant(data.topology), flags.astype(tack.u8))}


class MergePoints(TackAlgorithm):
    """``td.merge_points``: coincident points (or points within ``tolerance``) made one.
    VTK's ``vtkStaticCleanUnstructuredGrid``."""

    OUTPUT = "surfaces"
    FUNCTION = "merge_points"


# ── Changing cell types ─────────────────────────────────────────────

class Triangulate(TackAlgorithm):
    """``td.triangulate``: 2D cells as triangles. VTK's ``vtkTriangleFilter``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "triangulate"


class AsPolyhedra(TackAlgorithm):
    """``td.as_polyhedra``: a shape-based mesh as a polyhedral one, every cell a
    polyhedron."""

    FUNCTION = "as_polyhedra"


class AsPolygons(TackAlgorithm):
    """``td.as_polygons``: 2D cells as a polygonal mesh, each cell's points its loop."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "as_polygons"


class Shrink(TackAlgorithm):
    """``td.shrink``: every cell shrunk toward its centroid by ``factor``, each on points
    of its own (``td.subdivide(..., 1)`` of the shrunk cells, which have no shared
    points for VTK). VTK's ``vtkShrinkFilter``."""

    FUNCTION = "shrink"

    def compute(self, data):
        import tack.dataset as td

        return td.subdivide(td.shrink(data, **self.arguments()), 1)


class Subdivide(TackAlgorithm):
    """``td.subdivide``: curved or higher-order cells as linear sub-cells, ``level`` a
    cell along each axis. VTK's ``vtkCellGridToUnstructuredGrid``."""

    FUNCTION = "subdivide"


# ── Moving the geometry ─────────────────────────────────────────────

def _matrix(value):
    """An affine map as a NumPy matrix: VTK's ``vtkMatrix4x4`` or a linear transform's
    matrix (``vtkTransform``) read out; anything else as it is."""
    import numpy as np
    from vtkmodules.vtkCommonMath import vtkMatrix4x4

    if hasattr(value, "GetMatrix") and not isinstance(value, vtkMatrix4x4):
        value = value.GetMatrix()
    if isinstance(value, vtkMatrix4x4):
        return np.array([[value.GetElement(i, j) for j in range(4)] for i in range(4)])
    return value


class Transform(TackAlgorithm):
    """``td.transform``: the geometry under an affine ``matrix`` (4x4 or 3x4, a
    ``vtkMatrix4x4`` or a ``vtkTransform``); an image stays an image. The input's
    active vectors turn with it and its active normals stay normal, on the points
    and on the cells, as VTK's ``vtkTransformFilter`` transforms them, unless
    ``vectors`` or ``normals`` name others (``[]``: none)."""

    OUTPUT = "structure"
    FUNCTION = "transform"

    vectors = Parameter(None, "the 3-vector fields that turn with the map; by default the "
                              "input's active vectors")
    normals = Parameter(None, "the 3-vector fields that stay normal to the moved surface; "
                              "by default the input's active normals")

    def _unit(self, vtk_data):
        self._active = _active_vectors(vtk_data)
        return super()._unit(vtk_data)

    def arguments(self):
        values = super().arguments()
        values["matrix"] = _matrix(values["matrix"])
        active = getattr(self, "_active", {"vectors": [], "normals": []})
        for name in ("vectors", "normals"):
            given = getattr(self, name)
            values[name] = active[name] if given is None else [
                _field_argument(v) for v in ([given] if isinstance(given, (str, tuple))
                                             else given)]
        return values


def _active_vectors(vtk_data):
    """The active 3-component vectors and normals of a VTK dataset (a partitioned one's
    first partition), on its points and cells, as Tack's field keys."""
    block = vtk_data
    if isinstance(vtk_data, vtkPartitionedDataSet):
        block = next(iter(vtk_data), None)
    out = {"vectors": [], "normals": []}
    if block is None:
        return out
    for attributes, where in ((block.GetPointData(), "points"), (block.GetCellData(), "cells")):
        for role, array in (("vectors", attributes.GetVectors()),
                            ("normals", attributes.GetNormals())):
            if array is not None and array.GetName() and array.GetNumberOfComponents() == 3:
                out[role].append(_field_argument((array.GetName(), where)))
    return out


class Warp(TackAlgorithm):
    """``td.warp``: the points moved along ``direction`` (a 3-vector field, or one vector)
    times ``scale`` (and the ``scale_by`` field). VTK's ``vtkWarpVector`` and
    ``vtkWarpScalar``."""

    OUTPUT = "points"
    FUNCTION = "warp"
    FIELDS = ("direction", "scale_by")


class Cylindrical(TackAlgorithm):
    """``td.cylindrical``: Cartesian positions as cylindrical ``(r, theta, z)``, or back
    with ``inverse`` (Viskores' ``CylindricalCoordinateTransform``)."""

    OUTPUT = "points"
    FUNCTION = "cylindrical"


class Spherical(TackAlgorithm):
    """``td.spherical``: Cartesian positions as spherical ``(r, theta, phi)``, or back with
    ``inverse`` (Viskores' ``SphericalCoordinateTransform``)."""

    OUTPUT = "points"
    FUNCTION = "spherical"


# ── Measures and quality ────────────────────────────────────────────

class CellGeometry(_FieldAlgorithm):
    """``td.cell_geometry``: each cell's ``measure`` (volume, area or length) and
    ``centroid``, as cell arrays. VTK's ``vtkCellSizeFilter``."""

    FUNCTION = "cell_geometry"


class FaceGeometry(_OnFaces):
    """``td.face_geometry``: each face's ``normal`` and ``area``, on the faces as
    polygons."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "face_geometry"


class CellCenters(_FieldAlgorithm):
    """``td.cell_centers``: each cell's parametric center, as a cell array ``name`` (VTK's
    ``vtkCellCenters`` makes them points instead)."""

    FUNCTION = "cell_centers"

    name = Parameter("center", "the new cell array's name")


class FaceCenters(_OnFaces):
    """``td.face_centers``: each face's area centroid, as ``name`` on the faces as
    polygons."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "face_centers"

    name = Parameter("center", "the new array's name")


class EdgeLengths(_OnFaces):
    """``td.edge_lengths``: each edge's length, as ``name`` on the edges as lines."""

    OUTPUT = "vtkPolyData"
    ON = "edges"
    FUNCTION = "edge_lengths"

    name = Parameter("length", "the new array's name")


class ImplicitValues(_FieldAlgorithm):
    """``td.implicit_values``: an implicit ``function`` (Tack's or VTK's) at the points, as
    a point array ``name``. VTK's ``vtkSampleImplicitFunctionFilter``."""

    FUNCTION = "implicit_values"
    IMPLICIT = ("function",)

    name = Parameter("implicit", "the new point array's name")

    def _made_active(self):
        return {("points", "Scalars"): self._output_name()}


class CellIds(_FieldAlgorithm):
    """``td.cell_ids``: each cell's id, as a cell array ``name``. VTK's
    ``vtkGenerateIds``."""

    FUNCTION = "cell_ids"

    name = Parameter("cell_id", "the new cell array's name")


class PointIds(_FieldAlgorithm):
    """``td.point_ids``: each point's id, as a point array ``name``. VTK's
    ``vtkGenerateIds``."""

    FUNCTION = "point_ids"

    name = Parameter("point_id", "the new point array's name")


# ── Fields from fields ──────────────────────────────────────────────

class FlowQuantities(_FieldAlgorithm):
    """``td.flow_quantities``: ``divergence``, ``vorticity`` and ``q_criterion`` from a
    velocity's ``gradient`` field. VTK's ``vtkGradientFilter``'s."""

    FUNCTION = "flow_quantities"
    FIELDS = ("gradient",)


_SPACES = {"points": "H1", "cells": "Constant"}


def _space(data, to):
    """The space ``to`` names on ``data``'s topology: ``"points"`` (``H1``) or ``"cells"``
    (``Constant``); a Tack space as it is."""
    import tack.dataset as td

    if isinstance(to, str):
        if to not in _SPACES:
            raise ValueError(f"to is 'points' or 'cells', not {to!r}")
        return getattr(td, _SPACES[to])(data.topology)
    return to


class Project(_FieldAlgorithm):
    """``td.project``: ``field`` moved ``to`` the points or the cells, as an array ``name``
    (by default the field's own name). VTK's ``vtkCellDataToPointData`` and
    ``vtkPointDataToCellData``."""

    FUNCTION = "project"
    FIELDS = ("field",)
    EXCLUDE = ("space",)

    to = Parameter("points", "where the field goes: points or cells")
    name = Parameter(None, "the new array's name; by default the field's")

    def new_fields(self, data):
        values = self.arguments()
        result = self._function()(data, values.pop("field"), _space(data, self.to), **values)
        return {self._output_name(): result}


class ValuesAtCenters(_FieldAlgorithm):
    """``td.values_at_centers``: ``field`` at each cell's parametric center, as a cell
    array ``name`` (by default the field's own name)."""

    FUNCTION = "values_at_centers"
    FIELDS = ("field",)

    name = Parameter(None, "the new cell array's name; by default the field's")


class Interpolate(_FieldAlgorithm):
    """``td.interpolate``: a ``@tack.func`` of world position, ``function``, at the points
    (or ``to="cells"``, at the cells), as an array ``name``."""

    FUNCTION = "interpolate"
    EXCLUDE = ("space",)

    to = Parameter("points", "where the values go: points or cells")
    name = Parameter("interpolated", "the new array's name")

    def new_fields(self, data):
        values = self.arguments()
        result = self._function()(data, values.pop("function"), _space(data, self.to),
                                  **values)
        return {self.name: result}


class Magnitude(_FieldAlgorithm):
    """``td.magnitude``: each vector's length, as an array ``name`` (Viskores'
    ``VectorMagnitude``)."""

    FUNCTION = "magnitude"
    FIELDS = ("field",)

    name = Parameter("magnitude", "the new array's name")


class Dot(_FieldAlgorithm):
    """``td.dot``: the dot product of the vector fields ``a`` and ``b``, as an array
    ``name`` (Viskores' ``DotProduct``)."""

    FUNCTION = "dot"
    FIELDS = ("a", "b")

    name = Parameter("dot", "the new array's name")


class Cross(_FieldAlgorithm):
    """``td.cross``: the cross product of the 3-vector fields ``a`` and ``b``, as an array
    ``name`` (Viskores' ``CrossProduct``)."""

    FUNCTION = "cross"
    FIELDS = ("a", "b")

    name = Parameter("cross", "the new array's name")


class Composite(_FieldAlgorithm):
    """``td.composite``: the scalar ``fields`` as the components of one vector field
    ``name`` (Viskores' ``CompositeVectors``)."""

    FUNCTION = "composite"

    fields = Parameter(None, "the scalar fields, in component order (required)")
    name = Parameter("composite", "the new array's name")

    def new_fields(self, data):
        if not self.fields:
            raise ValueError("Composite needs 'fields'")
        return {self.name: self._function()(data, *[_field_argument(f) for f in self.fields])}


class LogValues(_FieldAlgorithm):
    """``td.log_values``: the logarithm of ``field`` in ``base``, as an array ``name``
    (Viskores' ``LogValues``)."""

    FUNCTION = "log_values"
    FIELDS = ("field",)

    name = Parameter("log", "the new array's name")


class Convert(TackAlgorithm):
    """``td.convert``: the geometry (or the field ``field``) in ``unit``, its known unit
    converted; an image stays an image."""

    OUTPUT = "structure"
    FUNCTION = "convert"


# ── Finite volumes ──────────────────────────────────────────────────

class Jump(_OnFaces):
    """``td.jump``: ``field``'s difference across each face, as ``name`` on the faces as
    polygons."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "jump"
    FIELDS = ("field",)

    name = Parameter("jump", "the new array's name")


class UpwindFlux(_OnFaces):
    """``td.upwind_flux``: the upwind flux of ``field`` by a constant ``velocity`` through
    each face, as ``name`` on the faces as polygons."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "upwind_flux"
    FIELDS = ("field",)

    name = Parameter("flux", "the new array's name")


# ── Probing, sampling and particles ─────────────────────────────────

class Voxelize(_Whole):
    """``td.voxelize``: the input on the voxels of ``grid`` (a ``vtkImageData`` or a Tack
    image), with ``fraction``, the share of each voxel the input covers, and the
    fields' means. An AMR hierarchy is sampled at each sample's finest box."""

    OUTPUT = "vtkImageData"
    AMR_PORTS = (0,)
    FUNCTION = "voxelize"
    DATASETS = ("grid",)


class Advect(_Seeded):
    """``td.advect``: the particles from ``seeds`` (an ``(n, 3)`` array, or the points of the
    data on port 1) moved through the velocity ``field`` by up to ``steps`` steps,
    where they stopped, as vertex cells with ``steps``, ``time`` and ``status``
    (Viskores' ``ParticleAdvection``)."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "advect"
    FIELDS = ("field",)

    seeds = Parameter(None, "the seed points, an (n, 3) array; by default the points "
                            "of the data on port 1")


# ── Point clouds ────────────────────────────────────────────────────

class ExtractPoints(TackAlgorithm):
    """``td.extract_points``: the points inside an implicit ``function`` (Tack's or VTK's),
    or outside it, as vertex cells. VTK's ``vtkExtractPoints``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "extract_points"
    IMPLICIT = ("function",)


class ThresholdPoints(TackAlgorithm):
    """``td.threshold_points``: the points whose ``field`` lies in ``[lower, upper]``, as
    vertex cells. VTK's ``vtkThresholdPoints``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "threshold_points"
    FIELDS = ("field",)


class MaskPoints(TackAlgorithm):
    """``td.mask_points``: every ``stride``-th point, as vertex cells. VTK's
    ``vtkMaskPoints``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "mask_points"


class PointNormals(_FieldAlgorithm):
    """``td.point_normals``: each point's normal from its neighbourhood (``k`` nearest, or
    within ``radius``), as a point array ``normals``. VTK's
    ``vtkPCANormalEstimation``."""

    FUNCTION = "point_normals"
    RESULT_FIELDS = ("normals",)

    def _made_active(self):
        return {("points", "Normals"): "normals"}


class Outliers(_Whole):
    """``td.outliers``: the points with ``outlier`` (1 for an outlier) and
    ``mean_distance``, by a radius test or a statistical one, over every piece. VTK's
    ``vtkRadiusOutlierRemoval`` and ``vtkStatisticalOutlierRemoval`` (which remove
    them)."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "outliers"


class VoxelDownsample(_Whole):
    """``td.voxel_downsample``: one point per occupied voxel of side ``size``, over every
    piece. VTK's ``vtkVoxelGrid``."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "voxel_downsample"


class Deposit(_Whole):
    """``td.deposit``: the points (or their ``field``) counted, summed or averaged onto
    ``grid`` (a ``vtkImageData`` or a Tack image), over every piece: the grid with a
    cell array ``name`` (Viskores' ``ParticleDensityNearestGridPoint`` and
    ``ParticleDensityCloudInCell``)."""

    OUTPUT = "vtkImageData"
    FUNCTION = "deposit"
    FIELDS = ("field",)
    DATASETS = ("grid",)

    name = Parameter("deposit", "the new cell array's name")

    def compute(self, data):
        import tack.dataset as td

        values = self.arguments()
        grid = values["grid"]
        deposited = self._function()(data, **values)
        return td.DataSet(grid.topology, grid.geometry, fields={self.name: deposited},
                          **grid._frame())


class SphDensity(_FieldAlgorithm):
    """``td.sph_density``: each particle's SPH density from its ``mass`` and
    ``smoothing_length`` (fields' names, or numbers), as a point array ``density``."""

    FUNCTION = "sph_density"
    RESULT_FIELDS = ("density",)


class SphInterpolate(TackAlgorithm):
    """``td.sph_interpolate``: the particles' -- the data on input port 1
    (``SetSourceConnection``, ``SetSourceData``), with their density -- fields at the
    input's points by SPH kernel sums, on the input's structure. VTK's
    ``vtkSPHInterpolator``."""

    OUTPUT = "structure"
    INPUTS = 2
    FUNCTION = "sph_interpolate"
    EXCLUDE = ("target",)

    def SetSourceConnection(self, port):
        self.SetInputConnection(1, port)

    def SetSourceData(self, data):
        self.SetInputDataObject(1, data)

    def _prepare(self, inInfo):
        self._particles = self._dataset(self.GetInputData(inInfo, 1, 0), self.device)

    def compute(self, data):
        return self._function()(self._particles, data, **self.arguments())


class HaloCatalog(_Whole):
    """``td.halo_catalog``: the halos of particles labelled by ``region`` (as
    ``ConnectedComponents`` with a ``radius`` labels them), one point each with its
    count, mass, center and radius, over every piece."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "halo_catalog"


# ── Pieces and ghosts ───────────────────────────────────────────────

class Split(TackAlgorithm):
    """``td.split``: a structured grid cut into ``pieces``, with ``ghost_layers``, as a
    ``vtkPartitionedDataSet``."""

    FUNCTION = "split"

    def _output_type(self, input):
        return vtkPartitionedDataSet


class RemoveGhosts(TackAlgorithm):
    """``td.remove_ghosts``: a piece without its ghost and hidden cells; a structured
    piece stays structured. VTK's ``vtkRemoveGhosts``."""

    OUTPUT = "structure"
    FUNCTION = "remove_ghosts"


# ── Filters over every piece ────────────────────────────────────────

class _OverPieces(TackAlgorithm):
    """A filter whose answer needs every piece: it takes a ``vtkPartitionedDataSet``
    whole (a single dataset is one piece)."""

    def execute(self, data):
        from tack.dataset import AMRDataSet, PartitionedDataSet

        if not isinstance(data, (PartitionedDataSet, AMRDataSet)):
            # This process's one piece, beside the other processes' (``world``).
            data = PartitionedDataSet([data], world=self._run_world)
        return self.over_pieces(data)

    def over_pieces(self, pieces):
        return self._function()(pieces, **self.arguments())

    def _output_type(self, input):
        return vtkOverlappingAMR if isinstance(input, vtkOverlappingAMR) else vtkPartitionedDataSet


class GhostCells(_OverPieces):
    """``td.ghost_cells``: each piece grown by ``layers`` of its neighbours' cells,
    marked ``DUPLICATECELL``. VTK's ``vtkGhostCellsGenerator``. An AMR hierarchy's
    boxes each grow within their level, a ghost a sibling holds copied from it, else
    from the coarser level (AMReX's ``FillBoundary`` and ``FillPatchTwoLevels``): a
    hierarchy of the input's boxes, its blocks the grown ones, which the AMR filters
    after it take as they are."""

    AMR_PORTS = (0,)
    FUNCTION = "ghost_cells"

    layers = Parameter(1, "the layers of ghost cells")
    tolerance = Parameter(0.0, "how far apart two pieces' copies of a point may be")


class GenerateGlobalIds(_OverPieces):
    """``td.generate_global_ids``: ids of points and cells unique over the pieces, a
    shared point's copies one id (VTK's ``GlobalPointIds`` and ``GlobalCellIds``).
    VTK's ``vtkGenerateGlobalIds``."""

    FUNCTION = "generate_global_ids"
    BY_BOX = False                      # a level's ids would repeat another's

    tolerance = Parameter(0.0, "how far apart two pieces' copies of a point may be")


class MergePieces(_OverPieces):
    """``td.merge_pieces``: the pieces as one dataset, shared points once. Pieces of a
    structured grid that tile it give the grid, polygonal pieces ``vtkPolyData``, and
    anything else an unstructured grid. Over MPI ranks the dataset is the ``root``
    rank's; the others get an empty one of the same type. VTK's ``vtkMergeBlocks``."""

    FUNCTION = "merge_pieces"

    tolerance = Parameter(0.0, "how far apart two pieces' copies of a point may be")

    def _unit(self, vtk_data):
        import vtkmodules.vtkCommonDataModel as dm

        from tack.interop.vtk import dataset_to_vtk

        kind = self._output_type(vtk_data)
        merged = self.execute(self._dataset(vtk_data, self.device))
        if merged is None:
            # Not the root rank, which merge_pieces gives the whole.
            return kind()
        out = dataset_to_vtk(merged, copy=False)
        return _as_polydata(out) if kind is dm.vtkPolyData else out

    def _output_type(self, input):
        import vtkmodules.vtkCommonDataModel as dm

        if not isinstance(input, vtkPartitionedDataSet):
            return type(input)
        # By class name: the pythonic API's datasets are subclasses of VTK's.
        kinds = {block.GetClassName() for block in input if block is not None}
        if len(kinds) == 1:
            return getattr(dm, next(iter(kinds)))
        return dm.vtkUnstructuredGrid



class GhostPoints(_OverPieces):
    """``td.ghost_points``: point-cloud pieces, each with copies of the other pieces'
    points within ``radius`` (or its points' ``k`` nearest) appended as ghosts."""

    OUTPUT = "vtkPolyData"
    FUNCTION = "ghost_points"


# ── AMR hierarchies ─────────────────────────────────────────────────

class AverageDown(TackAlgorithm):
    """``td.average_down``: an AMR hierarchy with its finer levels' values of ``field`` in
    the coarse cells and points they cover, finest level first -- a refined cell the
    volume-weighted mean of the fine cells over it, a refined point the fine point's
    value at its location. AMReX's ``average_down``. The input stays as it was: the
    output's ``field`` is a copy, its other arrays the input's."""

    OUTPUT = None
    AMR_PORTS = (0,)
    FUNCTION = "average_down"
    FIELDS = ("field",)

    def compute(self, data):
        from tack.dataset import AMRDataSet

        if not isinstance(data, AMRDataSet):
            raise TypeError("AverageDown takes an AMR hierarchy (vtkOverlappingAMR), not a "
                            f"{type(data).__name__}")
        field = self.arguments()["field"]
        out = _with_copy_of(data, field)
        self._function()(out, field)            # in place
        return out


def _with_copy_of(amr, key):
    """``amr`` with each box's field ``key`` a copy of it, the rest shared."""
    import tack
    from tack.dataset import AMRDataSet, DataSet, Field

    levels = []
    for pieces in amr.levels:
        boxes = []
        for box in pieces:
            field = box.fields[key]
            values = field.values
            # An implicit array stays: average_down refuses it, naming it.
            copied = Field(field.space, values.copy() if isinstance(values, tack.Array)
                           else values, field.unit)
            fields = {name: copied if f is field else f for name, f in box.fields.items()}
            boxes.append(DataSet(box.topology, box.geometry, fields=fields, sets=box.sets,
                                 ghosts=box.ghosts, global_ids=box.global_ids,
                                 **box._frame()))
        levels.append(boxes)
    return AMRDataSet._grown(amr, levels, amr._extents, amr.layers, amr._fills)


# ── Reductions: one answer for all the input ────────────────────────

class _Reduction(TackAlgorithm):
    """A filter whose answer is one result for all its input: every dataset in it (a
    partitioned dataset's pieces, a tree's datasets) on every rank, reduced together,
    as VTK's ``vtkDescriptiveStatistics`` and ``vtkIntegrateAttributes`` reduce
    composite data; an AMR hierarchy's boxes with their marks, so what a finer box
    covers and what a sibling holds count once. Every rank gets the result. ``reduce(pieces)`` gives the VTK
    output from a ``PartitionedDataSet`` of every dataset in the input."""

    OUTPUT = "vtkTable"
    AMR_PORTS = (0,)

    fields = Parameter(None, "the fields: names, or (name, 'points' / 'cells'); by "
                             "default every field. A vector field is reduced a "
                             "component at a time")

    def RequestDataObject(self, request, inInfo, outInfo):
        import vtkmodules.vtkCommonDataModel as dm

        kind = getattr(dm, self.OUTPUT)
        output = outInfo.GetInformationObject(0)
        current = output.Get(vtkDataObject.DATA_OBJECT())
        if current is None or not current.IsA(self.OUTPUT):
            output.Set(vtkDataObject.DATA_OBJECT(), kind())
        return 1

    def RequestData(self, request, inInfo, outInfo):
        from tack.dataset import PartitionedDataSet

        self._run_world = self._world()
        self._check_inputs(inInfo)
        input = self.GetInputData(inInfo, 0, 0)
        if isinstance(input, vtkOverlappingAMR):
            hierarchy = _as_dataset(input, self.device, self._run_world)
            self.GetOutputData(outInfo, 0).ShallowCopy(self.reduce(hierarchy._as_pieces()))
            return 1
        if isinstance(input, (*_TREES, vtkPartitionedDataSet)):
            leaves = []
            iterator = input.NewIterator()
            iterator.InitTraversal()
            while not iterator.IsDoneWithTraversal():
                leaves.append(iterator.GetCurrentDataObject())
                iterator.GoToNextItem()
        else:
            leaves = [input]
        pieces = PartitionedDataSet([_as_dataset(leaf, self.device) for leaf in leaves],
                                    world=self._run_world)
        self.GetOutputData(outInfo, 0).ShallowCopy(self.reduce(pieces))
        return 1

    def _selected(self, pieces):
        """``(label, where, key, component)`` for each value to reduce: the chosen fields
        (``fields``, or ``SetInputArrayToProcess``'s, or every field), a vector field's
        components one by one (``component`` None for a scalar). As a rank with pieces
        finds them: a rank without any reduces the same values, with nothing of its
        own (collective)."""
        mine = self._selected_here(pieces)
        return next((s for s in pieces.world._allgather(mine) if s is not None), [])

    def _selected_here(self, pieces):
        """``_selected`` from this rank's first piece; ``None`` without one."""
        import tack.dataset as td
        from tack.arrays import width_of

        first = next(iter(pieces), None)
        if first is None:
            return None
        chosen = self.fields
        if chosen is None:
            chosen = [a for a in map(self._input_array, range(self._arrays_set()))
                      if a is not None] or None
        if chosen is None:
            chosen = [(name, "points" if _on_points(field.space) else "cells")
                      for name, field in first.fields.entries()]
        elif isinstance(chosen, (str, tuple)):
            chosen = [chosen]
        selected = []
        for value in chosen:
            key = _field_argument(value)
            field = first.fields[key]
            name = key if isinstance(key, str) else key[0]
            where = "points" if _on_points(field.space) else "cells"
            width = width_of(field.values)
            components = [None] if not width else list(range(width))
            selected += [(name, where, (name, td.H1 if where == "points" else td.Constant), c)
                         for c in components]
        return selected

    def _arrays_set(self):
        arrays = self.GetInformation().Get(vtkAlgorithm.INPUT_ARRAYS_TO_PROCESS())
        return 0 if arrays is None else arrays.GetNumberOfInformationObjects()


def _on_points(space):
    from tack.dataset import H1, Values

    return isinstance(space, H1) or (isinstance(space, Values) and space.on == "points")


def _component_pieces(pieces, key, component):
    """``pieces`` with the field ``key``'s ``component`` (a scalar field's self for None)
    as the field ``__value__``."""
    from tack.arrays import ExtractComponent
    from tack.dataset import DataSet, Field, PartitionedDataSet

    out = []
    for piece in pieces:
        field = piece.fields[key]
        values = field.values if component is None else ExtractComponent(field.values,
                                                                          component)
        out.append(DataSet(piece.topology, piece.geometry,
                           fields={"__value__": Field(field.space, values, field.unit)},
                           ghosts=piece.ghosts, global_ids=piece.global_ids,
                           **piece._frame()))
    return PartitionedDataSet(out, world=pieces.world)


def _table(columns):
    """A ``vtkTable`` of ``columns``, ``{name: list}``: strings as a string column,
    integers as ``i64``, the rest as doubles (``None`` as NaN)."""
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkStringArray, vtkTypeInt64Array
    from vtkmodules.vtkCommonDataModel import vtkTable

    table = vtkTable()
    for name, values in columns.items():
        if all(isinstance(v, str) for v in values) and values:
            array = vtkStringArray()
            for v in values:
                array.InsertNextValue(v)
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in values) and values:
            array = vtkTypeInt64Array()
            for v in values:
                array.InsertNextValue(v)
        else:
            array = vtkDoubleArray()
            for v in values:
                array.InsertNextValue(float("nan") if v is None else float(v))
        array.SetName(name)
        table.AddColumn(array)
    return table


class Statistics(_Reduction):
    """``td.statistics`` of each field (each component of a vector field) over all the
    input: a ``vtkTable``, a row per field and component, with columns ``field``,
    ``where``, ``component`` (-1 for a scalar), the statistics as Tack names them --
    ``n``, ``min``, ``max``, ``sum``, ``mean``, ``m2``, ``m3``, ``m4``,
    ``sample_variance``, ``population_variance``, ``sample_stddev``,
    ``population_stddev``, ``skewness``, ``kurtosis`` (not the excess) and
    ``weight`` -- and ``unit``, the field's (``""`` unknown). VTK's
    ``vtkDescriptiveStatistics``."""

    NAMES = ("n", "min", "max", "sum", "mean", "m2", "m3", "m4", "sample_variance",
             "population_variance", "sample_stddev", "population_stddev", "skewness",
             "kurtosis", "weight")

    weights = Parameter(None, "weights: 'volume' (each cell's measure, a point's share "
                              "of its cells'), or a field; by default none")

    def reduce(self, pieces):
        import tack.dataset as td

        columns = {"field": [], "where": [], "component": [],
                   **{name: [] for name in self.NAMES}, "unit": []}
        for name, where, key, component in self._selected(pieces):
            stats = td.statistics(_component_pieces(pieces, key, component), "__value__",
                                  weights=self.weights)
            columns["field"].append(name)
            columns["where"].append(where)
            columns["component"].append(-1 if component is None else component)
            for statistic in self.NAMES:
                columns[statistic].append(getattr(stats, statistic))
            columns["unit"].append(stats.unit or "")
        return _table(columns)


class Entropy(_Reduction):
    """``td.entropy`` of each field (each component of a vector field) over all the input,
    in bits, its values in ``bins`` equal bins over their range: a ``vtkTable`` with
    columns ``field``, ``where``, ``component`` (-1 for a scalar) and ``entropy``.
    Viskores' ``Entropy``."""

    bins = Parameter(10, "the number of equal bins over the values' range")
    weights = Parameter(None, "weights: 'volume', or a field; by default none")

    def reduce(self, pieces):
        import tack.dataset as td

        columns = {"field": [], "where": [], "component": [], "entropy": []}
        for name, where, key, component in self._selected(pieces):
            columns["field"].append(name)
            columns["where"].append(where)
            columns["component"].append(-1 if component is None else component)
            columns["entropy"].append(td.entropy(_component_pieces(pieces, key, component),
                                                 "__value__", bins=self.bins,
                                                 weights=self.weights))
        return _table(columns)


class Integrate(_Reduction):
    """``td.integrate`` over all the input, as VTK's ``vtkIntegrateAttributes`` gives it:
    a ``vtkUnstructuredGrid`` of one vertex at the input's centroid (the integral of
    position over the measure), each point field's integral as point data and each
    cell field's as cell data, and the measure as the cell array ``Length``, ``Area``
    or ``Volume``, by the cells' dimension. Integrals are by quadrature through each
    cell's own map, so curved and higher-order cells exactly."""

    OUTPUT = "vtkUnstructuredGrid"

    fields = Parameter(None, "the fields to integrate: names, or (name, 'points' / "
                             "'cells'); by default every field")

    def reduce(self, pieces):
        import numpy as np
        from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

        import tack.dataset as td
        from tack.interop.vtk import _label

        selected = {}
        for name, where, key, _ in self._selected(pieces):
            selected.setdefault((name, where), key)
        # The first piece's origin over every rank, and the widest cells: alike on a
        # rank without pieces, which takes part with nothing.
        first = next(iter(pieces), None)
        found = pieces.world._allgather((
            None if first is None else tuple(first.origin),
            max((piece.topology.dimension for piece in pieces), default=0)))
        origin = np.asarray(next((o for o, _ in found if o is not None), (0.0, 0.0, 0.0)),
                            float)
        dimension = max(d for _, d in found) or 3
        names = [f"{where}:{name}" for name, where in selected] + ["__position__"]
        out = []
        for piece in pieces:
            fields = {f"{where}:{name}": piece.fields[key]
                      for (name, where), key in selected.items()}
            position = piece.geometry
            if np.any(np.asarray(piece.origin, float) != origin):
                # Pieces on other origins: their positions relative to the first's.
                values = piece.world_positions() - origin
                position = td.Field(piece.geometry.space,
                                    tack.array(values.astype(np.float32 if values.dtype
                                                             == np.float32 else
                                                             np.float64)))
            fields["__position__"] = position
            out.append(td.DataSet(piece.topology, piece.geometry, fields=fields,
                                  ghosts=piece.ghosts, global_ids=piece.global_ids,
                                  **piece._frame()))
        integrals = td.integrate(td.PartitionedDataSet(out, world=pieces.world),
                                 fields=names)
        measure = integrals.measure
        centre = origin + (np.asarray(integrals.values["__position__"], float) / measure
                           if measure else 0.0)
        grid = vtkUnstructuredGrid()
        points = vtkPoints()
        points.InsertNextPoint(*centre)
        grid.SetPoints(points)
        grid.InsertNextCell(1, 1, [0])                      # VTK_VERTEX
        for (name, where), _ in selected.items():
            value = np.atleast_1d(np.asarray(integrals.values[f"{where}:{name}"], float))
            array = vtkDoubleArray()
            array.SetNumberOfComponents(value.size)
            array.InsertNextTuple(value.tolist())
            array.SetName(name)
            (grid.GetPointData() if where == "points" else grid.GetCellData()).AddArray(array)
            _label(array, integrals.units.get(f"{where}:{name}"))
        total = vtkDoubleArray()
        total.InsertNextValue(measure)
        total.SetName({1: "Length", 2: "Area"}.get(dimension, "Volume"))
        _label(total, integrals.measure_unit)
        grid.GetCellData().AddArray(total)
        return grid



# ── Temporal filters ────────────────────────────────────────────────

class _Temporal(TackAlgorithm):
    """A temporal filter of ``tack.dataset.temporal`` -- an object fed one step at a
    time -- in a VTK pipeline, in one of two modes (``mode``):

    - **In situ** the filter cannot drive the pipeline: each execution takes the step
      its input holds and gives the output at that step, stamped with its time. An
      execution at the time of the last step gives that output again, without
      taking the step twice; a step back in time, a parameter changed or VTK's
      ``NO_PRIOR_TEMPORAL_ACCESS`` saying reset start over.
    - **Driving** the filter asks its input for the steps it needs, one per pass
      (``CONTINUE_EXECUTING``), never one it has taken since it last started over.
      A filter that accumulates (``ACCUMULATES``: statistics, particles, paths)
      takes every step (or those in ``time_range``) and gives one output after the
      last, with no time: time is in its data, and it reports no time steps
      downstream. One whose answer is at a time (streaklines, a derivative) gives
      it at the time requested of it, going on from the steps it has taken; a
      request before them starts over.

    By default a filter drives where its input has time steps and VTK does not say
    the pipeline cannot go back in time (``NO_PRIOR_TEMPORAL_ACCESS``, which an in
    situ adaptor sets). Driving to a step gives the output the same steps give in
    situ. A tree (``vtkMultiBlockDataSet``, ``vtkPartitionedDataSetCollection``) gives
    a tree of the same shape: each unit in it (a partitioned dataset, or a block that
    is one dataset) has a Tack filter of its own, fed its part of each step. A
    partitioned dataset's pieces each keep their state, as Tack's filters do; an AMR
    hierarchy is refused. Subclasses give ``_make()``, the Tack filter, and
    ``_output(result, input, data, final)``, the VTK output for its result."""

    INPUTS = 1
    BY_BOX = False
    #: Whether the filter accumulates over the steps (one output, after the last,
    #: when driving), rather than answering at a time.
    ACCUMULATES = True
    #: Whether an answer at step k needs the steps before it (streaklines), or only
    #: step k - 1 (a derivative).
    NEEDS_EVERY_STEP = True

    mode = Parameter(None, "'in situ' or 'driving'; by default driving where the input "
                           "has time steps and the pipeline can go back in time")

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self._driving = False
        self._steps = []
        self._queue = []
        self._looping = False
        self._start_over()

    def _start_over(self):
        self._units = {}              # per unit of the input, in order: its _UnitState
        self._filter = None           # the unit's Tack filter, while it is fed
        self._fed = None              # driving: the index of the last step taken
        self._time = None             # in situ: the time of the last step taken
        self._made_at = None          # the parameter changes it was made at
        self._kept = None             # (output, index or time) of the last output
        self._stale = False

    def _changes(self):
        return self.__dict__.get("_parameter_changes", 0)

    def _drives(self, info):
        """Whether this run drives the pipeline (``mode``, else as the input says)."""
        s = _pipeline()
        mode = (self.mode or "").replace("_", " ")
        if mode in ("in situ", "driving"):
            return mode == "driving"
        if info.Has(s.NO_PRIOR_TEMPORAL_ACCESS()):
            return False
        return info.Has(s.TIME_STEPS()) and info.Length(s.TIME_STEPS()) > 0

    # ── The pipeline ────────────────────────────────────────────────

    def RequestInformation(self, request, inInfo, outInfo):
        s = _pipeline()
        info = inInfo[0].GetInformationObject(0)
        self._driving = self._drives(info)
        self._steps = list(info.Get(s.TIME_STEPS())) if info.Has(s.TIME_STEPS()) else []
        # The pipeline before this filter, or the filter, changed: what was taken
        # of the steps is stale (in situ, every new step is such a change, and
        # the steps' times say what is new instead).
        self._stale = True
        if self._driving and self.ACCUMULATES:
            out = outInfo.GetInformationObject(0)
            out.Remove(s.TIME_STEPS())
            out.Remove(s.TIME_RANGE())
        return 1

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        if not self._driving:
            return 1                  # the step the input holds
        s = _pipeline()
        if not self._looping:
            self._plan(outInfo.GetInformationObject(0))
        info = inInfo[0].GetInformationObject(0)
        if self._queue:
            info.Set(s.UPDATE_TIME_STEP(), self._steps[self._queue[0]])
        elif self._fed is not None:
            # Nothing to take: the input keeps the step it gave last, unexecuted.
            info.Set(s.UPDATE_TIME_STEP(), self._steps[self._fed])
        return 1

    def _plan(self, out):
        """The steps this run takes, in order (``_queue``): none where the output kept
        is the answer."""
        s = _pipeline()
        stale = self._stale or self._made_at != self._changes()
        self._stale = False
        if self.ACCUMULATES:
            window = self.time_range
            steps = [i for i, t in enumerate(self._steps)
                     if window is None or window[0] <= t <= window[1]]
            if not steps:
                raise ValueError(f"{type(self).__name__}: no time step in {window}")
            if stale or self._kept is None:
                self._start_over()
                self._queue = steps
            else:
                self._queue = []
            return
        wanted = out.Get(s.UPDATE_TIME_STEP()) if out.Has(s.UPDATE_TIME_STEP()) else None
        k = 0 if wanted is None else max(
            [i for i, t in enumerate(self._steps) if t <= wanted] or [0])
        fed = None if stale else self._fed
        if fed is not None and self._kept is not None and self._kept[1] == k:
            self._queue = []                            # the answer is kept
        elif fed is not None and (fed < k if self.NEEDS_EVERY_STEP else fed == k - 1):
            self._queue = list(range(fed + 1, k + 1))   # on from the steps taken
        else:
            self._start_over()
            first = 0 if self.NEEDS_EVERY_STEP else max(k - 1, 0)
            self._queue = list(range(first, k + 1))

    def RequestData(self, request, inInfo, outInfo):
        s = _pipeline()
        self._run_world = self._world()
        self._check_inputs(inInfo)
        input = self.GetInputData(inInfo, 0, 0)
        if isinstance(input, vtkOverlappingAMR):
            raise TypeError(f"{type(self).__name__} takes no AMR hierarchy")
        output = self.GetOutputData(outInfo, 0)
        try:
            if self._driving:
                self._drive(request, input, output)
            else:
                self._in_situ(inInfo[0].GetInformationObject(0), input, output)
        except BaseException:
            request.Remove(s.CONTINUE_EXECUTING())
            self._looping = False
            self._start_over()
            raise
        return 1

    def _drive(self, request, input, output):
        s = _pipeline()
        if not self._queue:
            self._set(output, *self._kept)
            return
        k = self._queue.pop(0)
        made = not self._units
        last = not self._queue
        result = self._feed(input, k, last)
        if made:
            self._made_at = self._changes()
        self._fed = k
        if not last:
            request.Set(s.CONTINUE_EXECUTING(), 1)
            self._looping = True
            return
        request.Remove(s.CONTINUE_EXECUTING())
        self._looping = False
        self._kept = (result, None if self.ACCUMULATES else k)
        self._set(output, *self._kept)

    def _in_situ(self, info, input, output):
        from tack.interop.vtk import _time_of

        s = _pipeline()
        if isinstance(input, _TREES):
            t = _tree_time(input)
        else:
            t = _time_of(self._dataset(input, self.device))
        if t is None:
            raise ValueError(f"{type(self).__name__} needs each step's time "
                             "(DATA_TIME_STEP)")
        reset = (not self._units or self._made_at != self._changes()
                 or (self._time is not None and t < self._time)
                 or (info.Has(s.NO_PRIOR_TEMPORAL_ACCESS())
                     and info.Get(s.NO_PRIOR_TEMPORAL_ACCESS())
                     == s.NO_PRIOR_TEMPORAL_ACCESS_RESET))
        if not reset and t == self._time:
            self._set(output, *self._kept)              # no new step
            return
        if reset:
            self._start_over()
        result = self._feed(input, None, True)
        self._made_at = self._changes()
        self._time = t
        self._kept = (result, t)
        self._set(output, *self._kept)

    def _feed(self, input, k, last):
        """Feed each unit of ``input`` its part of the step (driving, the step's index
        ``k``): the output, a tree of the units' outputs for a tree, where this step
        gives one (``last``; in situ, every step), else ``None``."""
        if isinstance(input, _TREES):
            self._feeding = (0, k, last, _tree_time(input))
            out = self._tree(input)
        else:
            self._feeding = (0, k, last, None)
            out = self._unit_of(input)
        return out if last else None

    def _unit(self, vtk_data):
        # One unit's part of the step (``_feed``), with its own Tack filter.
        from tack.interop.vtk import _time_of

        index, k, last, time = self._feeding
        self._feeding = (index + 1, k, last, time)
        data = self._dataset(vtk_data, self.device)
        if time is not None and _time_of(data) is None:
            data = _at_time(data, time)     # a reader stamps the tree, not its blocks
        unit = self._units.get(index)
        if unit is None:
            self._filter = None
            unit = self._units[index] = _UnitState(self._make(data),
                                                   getattr(self, "_names", None))
        self._filter, self._names = unit.filter, unit.names
        result = unit.filter.update(data)
        if not self._driving:
            return self._output(result, vtk_data, data, final=False)
        if self.ACCUMULATES:
            unit.results.append((k, result))
            if not last:
                return None
            results, unit.results = unit.results, []
            return self._output(results, vtk_data, data, final=True)
        return self._output(result, vtk_data, data, final=False) if last else None

    def _set(self, output, result, at):
        """Make ``output`` the VTK ``result``: at its step's time, or at none."""
        _set_output(self, output, result)
        info = output.GetInformation()
        if at is None:
            info.Remove(vtkDataObject.DATA_TIME_STEP())
        else:
            info.Set(vtkDataObject.DATA_TIME_STEP(),
                     self._steps[at] if self._driving else at)

    # ── What a subclass gives ───────────────────────────────────────

    def _make(self, data):
        raise NotImplementedError

    def _output(self, result, input, data, final):
        """The VTK output: for ``result``, the Tack filter's for the step ``data`` (the
        VTK ``input``); or when driving a filter that accumulates (``final``), for
        ``result``, every step's ``(index, result)``."""
        raise NotImplementedError

    def _fields(self, data):
        """``fields`` as Tack's keys; by default every field of the step."""
        from tack.dataset import PartitionedDataSet

        first = next(iter(data)) if isinstance(data, PartitionedDataSet) else data
        if self.fields is None:
            return list(first.fields)
        chosen = [self.fields] if isinstance(self.fields, (str, tuple)) else self.fields
        return [_field_argument(value) for value in chosen]


def _tree_time(tree):
    """The time of a tree's step: VTK's stamp on the tree (``DATA_TIME_STEP``), as a
    reader gives it, else on its first dataset that has one, else ``None``."""
    key = vtkDataObject.DATA_TIME_STEP()
    if tree.GetInformation().Has(key):
        return tree.GetInformation().Get(key)
    iterator = tree.NewIterator()
    iterator.InitTraversal()
    while not iterator.IsDoneWithTraversal():
        info = iterator.GetCurrentDataObject().GetInformation()
        if info.Has(key):
            return info.Get(key)
        iterator.GoToNextItem()
    return None


def _at_time(data, time):
    """A Tack dataset or ``PartitionedDataSet`` at ``time``: shallow copies, so that the
    dataset kept for the VTK one (``_as_dataset``) is left as it was."""
    import copy

    from tack.dataset import PartitionedDataSet

    if isinstance(data, PartitionedDataSet):
        return PartitionedDataSet([_at_time(piece, time) for piece in data], world=data.world)
    out = copy.copy(data)
    out.time = time
    return out


class _UnitState:
    """A temporal filter's state for one unit of its input: the Tack ``filter``, the
    field ``names`` it was made with, and, driving a filter that accumulates, each
    step's ``(index, result)`` so far."""

    def __init__(self, filter, names):
        self.filter = filter
        self.names = names
        self.results = []


def _pipeline():
    from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline

    return vtkStreamingDemandDrivenPipeline


def _arrays_onto(vtk_data, pieces, names, keep):
    """A shallow copy of the VTK dataset or partitioned dataset ``vtk_data`` with the
    fields ``names`` of the matching Tack ``pieces`` (a dataset, or a partitioned one)
    as arrays; its own arrays kept if ``keep``, else only its ghost marks."""
    from tack.interop.vtk import _label, _set_values, _to_vtk_values

    def onto(block, piece):
        out = block.NewInstance()
        out.ShallowCopy(block)
        if not keep:
            for attributes in (out.GetPointData(), out.GetCellData()):
                ghosts = attributes.GetArray("vtkGhostType")
                attributes.Initialize()
                if ghosts is not None:
                    attributes.AddArray(ghosts)
        for name in names:
            field = piece.fields[name]
            on_points = _on_points(field.space)
            attributes = out.point_data if on_points else out.cell_data
            values, share = _vtk_values_of(block, piece, field, name) if on_points else \
                (field.values, True)
            _set_values(attributes, name, _to_vtk_values(values, share=share, name=name))
            _label(attributes[name], field.unit)
        return out

    if isinstance(vtk_data, vtkPartitionedDataSet):
        out = vtk_data.NewInstance()
        for block, piece in zip(vtk_data, pieces):
            out.append(onto(block, piece))
        return out
    return onto(vtk_data, pieces)


def _count(vtk_data, name, value):
    """``vtk_data`` (a dataset, or each partition) with the field data array ``name``,
    one integer."""
    from vtkmodules.vtkCommonCore import vtkTypeInt64Array

    blocks = list(vtk_data) if isinstance(vtk_data, vtkPartitionedDataSet) else [vtk_data]
    for block in blocks:
        array = vtkTypeInt64Array()
        array.SetName(name)
        array.InsertNextValue(int(value))
        block.GetFieldData().AddArray(array)
    return vtk_data


class RunningStatistics(_Temporal):
    """``td.RunningStatistics``: per point or cell, ``<name>_mean``, ``<name>_variance``
    (the population variance), ``<name>_min`` and ``<name>_max`` of ``fields`` (every
    field by default) over the steps, on the input's structure, with field data
    ``count``, the steps taken. In situ, the statistics up to the step, at its time;
    driving, over every step (or those in ``time_range``), at no time. The steps must
    be on one mesh. VTK's ``vtkTemporalStatistics``."""

    OUTPUT = None

    fields = Parameter(None, "the fields: names, or (name, 'points' / 'cells'); by "
                             "default every field")
    time_range = Parameter(None, "driving: (first, last), the times of the steps to "
                                 "take; by default every step")

    def _make(self, data):
        import tack.dataset as td

        self._names = self._fields(data)
        return td.RunningStatistics(self._names)

    def _output(self, result, input, data, final):
        from tack.dataset.dataset import field_name

        if final:
            count, result = len(result), result[-1][1]
        else:
            count = self._filter.count
        names = [f"{field_name(name)}_{suffix}" for name in self._names
                 for suffix in ("mean", "variance", "min", "max")]
        return _count(_arrays_onto(input, result, names, keep=False), "count", count)


class TimeDerivative(_Temporal):
    """``td.TimeDerivative``: the step with ``<name>_dt`` beside its arrays, each of
    ``fields``' (every field by default) rate of change since the step before, a
    backward difference. The first step has no step before: it goes out without
    them. Driving, at the time requested, from that step and the one before it
    (only the requested one where the one before is kept). VTK has none."""

    OUTPUT = None
    ACCUMULATES = False
    NEEDS_EVERY_STEP = False

    fields = Parameter(None, "the fields: names, or (name, 'points' / 'cells'); by "
                             "default every field")
    copy = Parameter(True, "keep a copy of the step before (False: references, where "
                           "the step's arrays are not overwritten by the next)")

    def _make(self, data):
        import tack.dataset as td

        self._names = self._fields(data)
        return td.TimeDerivative(self._names, copy=self.copy)

    def _output(self, result, input, data, final):
        from tack.dataset.dataset import field_name

        if result is None:                              # the first step
            return _arrays_onto(input, data, [], keep=True)
        return _arrays_onto(input, result, [f"{field_name(name)}_dt" for name in self._names],
                            keep=True)


class _Particles(_Temporal):
    """A temporal filter of particles from seeds: ``seeds`` (an ``(n, 3)`` array of
    world positions) or the points of the data on port 1, read when it starts."""

    INPUTS = 2
    REQUIRED_INPUTS = 1
    OUTPUT = "vtkPolyData"

    field = Parameter(None, "the velocity field (required): a name, or (name, 'points' "
                            "/ 'cells')")
    seeds = Parameter(None, "the seeds, an (n, 3) array of world positions; by default "
                            "the points of the data on port 1")
    integrator = Parameter("rk4", "rk4 or euler")
    step_size = Parameter(None, "the longest substep, in the steps' time unit; by default "
                                "one substep from step to step")
    copy = Parameter(True, "keep a copy of the step before (False: references)")

    def SetSourceConnection(self, port):
        self.SetInputConnection(1, port)

    def SetSourceData(self, data):
        self.SetInputDataObject(1, data)

    def RequestData(self, request, inInfo, outInfo):
        self._source = (self.GetInputData(inInfo, 1, 0)
                        if inInfo[1].GetNumberOfInformationObjects() else None)
        return super().RequestData(request, inInfo, outInfo)

    def _arguments(self):
        field = self.field if self.field is not None else self._input_array(0)
        if field is None:
            raise ValueError(f"{type(self).__name__} needs 'field', the velocity")
        seeds = self.seeds
        if seeds is None and self._source is not None:
            seeds = _points_of(self._source)
        if seeds is None:
            raise ValueError(f"{type(self).__name__} needs seeds: 'seeds', or data on "
                             "port 1")
        return {"field": _field_argument(field), "seeds": seeds,
                "integrator": self.integrator, "step_size": self.step_size,
                "copy": self.copy}

    def _output(self, result, input, data, final):
        from tack.interop.vtk import dataset_to_vtk

        out = _as_polydata(dataset_to_vtk(result, copy=False))
        # Of pieces, this rank's paths or lines are one dataset: its one partition.
        return _as_partitioned(out) if isinstance(input, vtkPartitionedDataSet) else out


class Advection(_Particles):
    """``td.Advection``: particles from ``seeds`` carried by the velocity ``field``
    through the steps, the velocity linear in time between two steps. In situ, the
    particles at the step's time: a point cloud with ``seed``, ``source``, ``release``,
    ``time`` and ``status``. Driving, the particles at every step taken (or those in
    ``time_range``) in one point cloud, with ``step`` (its index among the input's
    time steps) beside those, at no time. Over MPI ranks each rank releases the seeds
    it is given."""

    OUTPUT = "vtkPolyData"
    time_range = Parameter(None, "driving: (first, last), the times of the steps to "
                                 "take; by default every step")

    def _make(self, data):
        import tack.dataset as td

        return td.Advection(**self._arguments())

    def _output(self, result, input, data, final):
        if not final:
            return super()._output(result, input, data, final)
        return super()._output(_every_step(result), input, data, final)


def _every_step(results):
    """One point cloud of every step's particles (``(index, cloud)``, each a cloud or a
    partitioned dataset of this rank's), with ``step``, at no time."""
    import numpy as np

    import tack.dataset as td

    clouds = []
    for k, cloud in results:
        for piece in (cloud if isinstance(cloud, td.PartitionedDataSet) else [cloud]):
            clouds.append((k, piece))
    first = clouds[0][1]
    names = ("seed", "source", "release", "time", "status")
    fields = {name: np.concatenate([np.asarray(c.fields[name].values.numpy())
                                    for _, c in clouds]) for name in names}
    fields["step"] = np.concatenate([np.full(c.num_points, k, np.int64) for k, c in clouds])
    unit = first.fields["time"].unit
    out = td.point_cloud(np.concatenate([c.world_positions() for _, c in clouds]),
                         fields=fields, origin=first.origin, device=first.device,
                         unit=first.unit,
                         units={"release": unit, "time": unit} if unit else None)
    last = results[-1][1]
    if isinstance(last, td.PartitionedDataSet):
        return td.PartitionedDataSet([out], world=last.world)
    return out


class Pathlines(_Particles):
    """``td.Pathlines``: each particle's path from ``seeds`` through the velocity
    ``field``: a polyline per particle, ``seed`` and ``time`` on its points. In situ,
    the paths up to the step, at its time; driving, through every step taken (or
    those in ``time_range``), at no time. VTK's ``vtkParticlePathFilter``."""

    OUTPUT = "vtkPolyData"
    time_range = Parameter(None, "driving: (first, last), the times of the steps to "
                                 "take; by default every step")

    def _make(self, data):
        import tack.dataset as td

        return td.Pathlines(**self._arguments())

    def _output(self, result, input, data, final):
        return super()._output(result[-1][1] if final else result, input, data, final)


class Streaklines(_Particles):
    """``td.Streaklines``: the streaklines at a time of particles released at ``seeds``
    through the velocity ``field``, ``releases`` at each seed from step to step: a
    polyline per seed through its particles still in the mesh, newest first, with
    ``source``, ``release`` and ``time`` on the points, at the step's time. Driving,
    at the time requested, going on from the steps taken. VTK's
    ``vtkStreaklineFilter``."""

    OUTPUT = "vtkPolyData"
    ACCUMULATES = False

    releases = Parameter(1, "the particles released at each seed from step to step")

    def _make(self, data):
        import tack.dataset as td

        return td.Streaklines(**self._arguments(), releases=self.releases)


del tack
