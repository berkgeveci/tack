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

Filters that make a dataset give a ``vtkUnstructuredGrid`` (or keep a structured
grid's type where the Tack filter does); filters that compute a field (``Gradients``,
``Elevation``, ``CellQuality``) give their input's type, shallow-copied, with the
new arrays added.

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

An exception in a filter is VTK's error: printed, the output left empty and
``Update()`` false. It is also kept as the algorithm's ``error`` (``None`` after a
run that succeeded), for Python callers.
"""

import tack

__all__ = [
    "CellQuality",
    "Clip",
    "ConnectedComponents",
    "Contour",
    "Elevation",
    "ExtractSurface",
    "GenerateGlobalIds",
    "GhostCells",
    "Gradients",
    "MergePieces",
    "Probe",
    "SlicePlane",
    "Streamlines",
    "SurfaceNormals",
    "TackAlgorithm",
    "Tetrahedralize",
    "Threshold",
]

import inspect

from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkMultiBlockDataSet,
    vtkMultiPieceDataSet,
    vtkPartitionedDataSet,
    vtkPartitionedDataSetCollection,
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
# they share one Tack dataset. Kept by the VTK object's address, modification time
# and time step: the modification time is a global counter, so the same object
# modified, or a later one at the same address, does not match. An entry goes when
# its VTK object is deleted, and holds nothing that keeps that object alive. Beside
# the dataset sharing VTK's memory, an entry keeps its copy on each device a filter
# asked for (``device=``), so the data moves there once.

_datasets = {}          # address -> ((modification time, time step), {device: DataSet})


def _key(vtk_data):
    from tack.interop.vtk import _time_key

    info = vtk_data.GetInformation()
    return vtk_data.GetMTime(), (info.Get(_time_key()) if info.Has(_time_key()) else None)


def _remember(vtk_data, data):
    """Keep ``data`` as the Tack dataset of ``vtk_data`` as it is now."""
    address = vtk_data.GetAddressAsString("vtkObject")
    if address not in _datasets:
        # The address, not the caller: a DeleteEvent's caller is None.
        vtk_data.AddObserver("DeleteEvent", lambda caller, event: _datasets.pop(address, None))
    _datasets[address] = (_key(vtk_data), {None: data})


def _as_dataset(vtk_data, device=None):
    """``vtk_to_dataset(vtk_data, dtype=None, copy=False)``, the same Tack dataset for
    the same VTK dataset unchanged (each piece of a ``vtkPartitionedDataSet``);
    with ``device``, that dataset moved there (the same moved one each time)."""
    import tack
    from tack.dataset import PartitionedDataSet
    from tack.interop.vtk import _time_key, vtk_to_dataset

    if isinstance(device, str):
        device = tack.device(device)
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
    entry = _datasets.get(vtk_data.GetAddressAsString("vtkObject"))
    if entry is None or entry[0] != _key(vtk_data):
        _remember(vtk_data, vtk_to_dataset(vtk_data, dtype=None, copy=False))
        entry = _datasets[vtk_data.GetAddressAsString("vtkObject")]
    copies = entry[1]
    data = copies[None]
    if device is None or data.device == device:
        return data
    if device not in copies:
        copies[device] = data.to_device(device)
    return copies[device]


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
    that are fields, in ``SetInputArrayToProcess``'s order, and ``IMPLICIT`` those
    that may be implicit functions.

    ``compute(piece)`` gives a Tack dataset from a Tack dataset, by default the
    function called with the parameters; ``execute(data)`` runs it on a dataset or
    on each piece of a ``PartitionedDataSet``, and a subclass whose answer needs every
    piece overrides it. ``OUTPUT`` names the output's VTK type for a dataset input
    (``None``: the input's own)."""

    OUTPUT = "vtkUnstructuredGrid"
    INPUTS = 1
    FUNCTION = None
    FIELDS = ()
    IMPLICIT = ()
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
        for name in getattr(self, "_ARGUMENTS", ()):
            value = getattr(self, name)
            if name in self.FIELDS:
                if value is None:
                    value = self._input_array(self.FIELDS.index(name))
                value = _field_argument(value)
            elif name == "fields" and isinstance(value, (list, tuple)):
                value = [_field_argument(v) for v in value]
            if name in self.IMPLICIT:
                value = _implicit(value)
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
        """``_as_dataset``, a partitioned one in this run's World."""
        return _in_world(_as_dataset(vtk_data, device), self._run_world)

    # ── The pipeline ────────────────────────────────────────────────

    # The pipeline's information objects are read and set with Get/Set/Append:
    # vtkInformation takes no item assignment in VTK 9.7.1.

    def FillInputPortInformation(self, port, info):
        info.Set(vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), _INPUT_TYPES[0])
        for kind in _INPUT_TYPES[1:]:
            info.Append(vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), kind)
        return 1

    def _output_type(self, input):
        """The VTK class of the output for ``input``, a dataset or a partitioned
        dataset (trees keep their type)."""
        if isinstance(input, vtkPartitionedDataSet):
            return vtkPartitionedDataSet
        if self.OUTPUT is None:
            return type(input)
        import vtkmodules.vtkCommonDataModel as dm

        return getattr(dm, self.OUTPUT)

    def RequestDataObject(self, request, inInfo, outInfo):
        input = self.GetInputData(inInfo, 0, 0)
        if input is None:
            return 0
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
        self._prepare(inInfo)
        input = self.GetInputData(inInfo, 0, 0)
        output = self.GetOutputData(outInfo, 0)
        if isinstance(input, _TREES):
            # Shares the blocks: what was made, and remembered, is the output's.
            output.CompositeShallowCopy(self._tree(input))
        else:
            self._set_unit_output(output, self._unit(input))
        return 1

    def _prepare(self, inInfo):
        """Read what every unit of a run needs from the other ports (a source)."""

    def _set_unit_output(self, output, result):
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
                    out.SetPartitionedDataSet(i, _as_partitioned(self._unit(child)))
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
                out.SetBlock(i, _as_multipiece(self._unit(child)))
            else:
                out.SetBlock(i, self._unit(child))
        _block_names(node, out, count)
        return out

    def _unit(self, vtk_data):
        """The filter's result for one unit: a dataset or a partitioned dataset."""
        from tack.interop.vtk import dataset_to_vtk

        return dataset_to_vtk(self.execute(self._dataset(vtk_data, self.device)), copy=False)

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
        kept = _datasets.get(result.GetAddressAsString("vtkObject"))
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
            _set_values(attributes, name, _to_vtk_values(field.values, share=True, name=name))
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
        result = self._function()(data, **self.arguments())
        return result if isinstance(result, dict) else {self.name: result}


# ── Filters that make a dataset ─────────────────────────────────────

class Contour(TackAlgorithm):
    """``td.contour``: the isosurface (isolines in 2D) of a point field at ``value`` (a
    number or a sequence of them). VTK's ``vtkContourFilter``."""

    FUNCTION = "contour"
    FIELDS = ("field",)

    field = Parameter(None, "the point field's name")
    value = Parameter(0.0, "the isovalue, or a sequence of them")


class SlicePlane(TackAlgorithm):
    """``td.slice_plane``: the cut by the plane through ``origin`` normal to ``normal``.
    VTK's ``vtkCutter`` with a ``vtkPlane``."""

    FUNCTION = "slice_plane"

    origin = Parameter((0.0, 0.0, 0.0), "a point of the plane")
    normal = Parameter((1.0, 0.0, 0.0), "the plane's normal")


class Clip(TackAlgorithm):
    """``td.clip``: the part of the cells where ``by`` (a point field, or an implicit
    function: Tack's, such as ``td.Plane``, or VTK's, such as ``vtkPlane``) is above
    ``value`` (below with ``inside``). VTK's ``vtkTableBasedClipDataSet``."""

    FUNCTION = "clip"
    FIELDS = ("by",)
    IMPLICIT = ("by",)

    by = Parameter(None, "a point field, or an implicit function (Tack's or VTK's)")
    value = Parameter(0.0, "the value the clip is at")
    inside = Parameter(False, "keep the part below the value instead")


class Threshold(TackAlgorithm):
    """``td.threshold``: the cells whose ``field`` lies in ``[lower, upper]`` (a point
    field at all of a cell's points, or any with ``all_points=False``). VTK's
    ``vtkThreshold``."""

    FUNCTION = "threshold"
    FIELDS = ("field",)

    field = Parameter(None, "the field")
    lower = Parameter(float("-inf"), "the lowest value kept")
    upper = Parameter(float("inf"), "the highest value kept")
    all_points = Parameter(True, "a point field: every point of a cell in range")


class ExtractSurface(TackAlgorithm):
    """``td.extract_surface``: the boundary faces of the cells, as polygons. VTK's
    ``vtkDataSetSurfaceFilter``."""

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

    FUNCTION = "surface_normals"

    consistent = Parameter(True, "wind the polygons consistently first")


class ConnectedComponents(TackAlgorithm):
    """``td.connected_components``: ``region`` on the cells and ``point_region`` on the
    points, cells connected through shared points (or, with ``radius``, points
    within it). VTK's ``vtkConnectivityFilter``."""

    OUTPUT = None
    FUNCTION = "connected_components"


class Streamlines(TackAlgorithm):
    """``td.streamlines``: the paths of ``seeds`` (an ``(n, 3)`` array) through the
    velocity ``field``, up to ``steps`` steps of ``step_size``. VTK's
    ``vtkStreamTracer``."""

    FUNCTION = "streamlines"
    FIELDS = ("field",)

    field = Parameter(None, "the velocity field")
    seeds = Parameter(None, "the seed points, an (n, 3) array")
    step_size = Parameter(0.01, "the step, in time (or in length with step_unit)")
    steps = Parameter(1000, "the most steps a path takes")
    integrator = Parameter("rk4", "rk4, rk45 or euler")
    step_unit = Parameter("time", "time, or length for steps of a fixed arc length")


class Probe(TackAlgorithm):
    """``td.probe``, as ``vtkProbeFilter``: the fields of the source -- the data on
    input port 1 (``SetSourceConnection``, ``SetSourceData``) -- at the points of the
    input (port 0), as point data on the input's structure, with ``valid`` (0 where
    no source cell holds the point). A partitioned source is probed as a whole, and a
    tree as the partitioned dataset of its blocks; a tree on port 0 keeps its
    structure. It runs on the source's device unless ``device`` says otherwise, the
    input moved there."""

    OUTPUT = None
    INPUTS = 2
    FUNCTION = "probe"
    EXCLUDE = ("where",)

    fields = Parameter(None, "the source fields to probe (by default every one)")

    def SetSourceConnection(self, port):
        self.SetInputConnection(1, port)

    def SetSourceData(self, data):
        self.SetInputDataObject(1, data)

    def _output_type(self, input):
        import vtkmodules.vtkCommonDataModel as dm

        if isinstance(input, vtkPartitionedDataSet):
            return vtkPartitionedDataSet
        # What dataset_to_vtk makes of the input's structure.
        for kind in (dm.vtkImageData, dm.vtkRectilinearGrid, dm.vtkStructuredGrid):
            if isinstance(input, kind):
                return kind
        return dm.vtkUnstructuredGrid

    def _prepare(self, inInfo):
        import tack.dataset as td

        source = self.GetInputData(inInfo, 1, 0)
        if isinstance(source, _TREES):
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
            self._where_device = self._source.device

    def _unit(self, vtk_data):
        import tack.dataset as td
        from tack.interop.vtk import dataset_to_vtk

        where = self._dataset(vtk_data, self._where_device)
        return dataset_to_vtk(td.probe(self._source, where, **self.arguments()), copy=False)


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


class CellQuality(_FieldAlgorithm):
    """``td.cell_quality``: each measure (Verdict's, generalized) as a cell array of its
    name. VTK's ``vtkMeshQuality``."""

    FUNCTION = "cell_quality"

    measures = Parameter(None, "the measures' names (by default each shape's own)")


# ── Filters over every piece ────────────────────────────────────────

class _OverPieces(TackAlgorithm):
    """A filter whose answer needs every piece: it takes a ``vtkPartitionedDataSet``
    whole (a single dataset is one piece)."""

    def execute(self, data):
        from tack.dataset import PartitionedDataSet

        if not isinstance(data, PartitionedDataSet):
            # This process's one piece, beside the other processes' (``world``).
            data = PartitionedDataSet([data], world=self._run_world)
        return self.over_pieces(data)

    def over_pieces(self, pieces):
        return self._function()(pieces, **self.arguments())

    def _output_type(self, input):
        return vtkPartitionedDataSet


class GhostCells(_OverPieces):
    """``td.ghost_cells``: each piece grown by ``layers`` of its neighbours' cells,
    marked ``DUPLICATECELL``. VTK's ``vtkGhostCellsGenerator``."""

    FUNCTION = "ghost_cells"

    layers = Parameter(1, "the layers of ghost cells")
    tolerance = Parameter(0.0, "how far apart two pieces' copies of a point may be")


class GenerateGlobalIds(_OverPieces):
    """``td.generate_global_ids``: ids of points and cells unique over the pieces, a
    shared point's copies one id (VTK's ``GlobalPointIds`` and ``GlobalCellIds``).
    VTK's ``vtkGenerateGlobalIds``."""

    FUNCTION = "generate_global_ids"

    tolerance = Parameter(0.0, "how far apart two pieces' copies of a point may be")


class MergePieces(_OverPieces):
    """``td.merge_pieces``: the pieces as one dataset, shared points once. Pieces of a
    structured grid that tile it give the grid; anything else an unstructured
    grid. Over MPI ranks the dataset is the ``root`` rank's; the others get an empty
    one of the same type. VTK's ``vtkMergeBlocks``."""

    FUNCTION = "merge_pieces"

    tolerance = Parameter(0.0, "how far apart two pieces' copies of a point may be")

    def _unit(self, vtk_data):
        from tack.interop.vtk import dataset_to_vtk

        merged = self.execute(self._dataset(vtk_data, self.device))
        if merged is None:
            # Not the root rank, which merge_pieces gives the whole.
            return self._output_type(vtk_data)()
        return dataset_to_vtk(merged, copy=False)

    def _output_type(self, input):
        import vtkmodules.vtkCommonDataModel as dm

        if not isinstance(input, vtkPartitionedDataSet):
            return type(input)
        kinds = {type(block) for block in input}
        if len(kinds) == 1 and next(iter(kinds)) is not dm.vtkPolyData:
            return next(iter(kinds))
        return dm.vtkUnstructuredGrid


del tack
