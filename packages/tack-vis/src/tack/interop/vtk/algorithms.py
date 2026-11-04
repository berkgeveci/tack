"""Tack's filters as VTK algorithms: ``VTKPythonAlgorithmBase`` subclasses that sit in
a VTK pipeline beside VTK's own.

    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
    from tack.interop.vtk.algorithms import Contour

    reader = vtkXMLUnstructuredGridReader(file_name="mesh.vtu")
    surface = (reader >> Contour(field="pressure", value=0.5))()

Each takes a ``vtkDataSet`` or a ``vtkPartitionedDataSet``. Its input becomes a
Tack dataset sharing VTK's memory (``vtk_to_dataset(..., copy=False)``), the
filter runs where that memory is -- on the CPU, or on the GPU holding an input of
Viskores arrays -- and its result goes back to VTK sharing Tack's
(``dataset_to_vtk(..., copy=False)``, a GPU's as Viskores arrays): no array is
copied either way where its values are stored as Tack keeps them, so a chain of
these filters on a GPU stays there. One VTK dataset, unchanged, is one Tack
dataset to every filter that reads it, so what Tack derives from its topology
(locators, faces, point links) is built once: by several filters on one input, or
by one run again after a parameter changes. A field filter's output is its
input's Tack dataset with the new fields, on the same topology.

A partitioned input runs the filter on each piece and gives a
``vtkPartitionedDataSet``; the filters whose answer needs every piece
(``GhostCells``, ``GenerateGlobalIds``, ``MergePieces``) take them together.

Filters that make a dataset give a ``vtkUnstructuredGrid`` (or keep a structured
grid's type where the Tack filter does); filters that compute a field (``Gradients``,
``Elevation``, ``CellQuality``) give their input's type, shallow-copied, with the
new arrays added.

Parameters are properties, given to the constructor or set later; setting one
marks the algorithm modified. Field names are given by name (``field=``).

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

from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkPartitionedDataSet
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm

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


class TackAlgorithm(VTKPythonAlgorithmBase):
    """A Tack filter as a VTK algorithm of one input and one output. Subclasses give
    ``compute(piece)``, a Tack dataset from a Tack dataset; ``execute(data)`` runs
    it on a dataset or on each piece of a ``PartitionedDataSet``, and a subclass
    whose answer needs every piece overrides it. ``OUTPUT`` names the output's
    VTK type for a dataset input (``None``: the input's own)."""

    OUTPUT = "vtkUnstructuredGrid"
    INPUTS = 1

    #: The exception the last run raised, or ``None``.
    error = None

    device = Parameter(None, "the device the filter runs on, a name or a tack.Device; "
                             "by default where its input's memory is. The input moves "
                             "there once (VTK's arrays stay as they are)")

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

    # ── The pipeline ────────────────────────────────────────────────

    # The pipeline's information objects are read and set with Get/Set/Append:
    # vtkInformation takes no item assignment in VTK 9.7.1.

    def FillInputPortInformation(self, port, info):
        info.Set(vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        info.Append(vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkPartitionedDataSet")
        return 1

    def _output_type(self, input):
        """The VTK class of the output for ``input``."""
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
        kind = self._output_type(input)
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
            self.error = error
            raise

    def RequestData(self, request, inInfo, outInfo):
        from tack.interop.vtk import dataset_to_vtk

        input = self.GetInputData(inInfo, 0, 0)
        output = self.GetOutputData(outInfo, 0)
        result = dataset_to_vtk(self.execute(_as_dataset(input, self.device)), copy=False)
        _set_output(self, output, result)
        return 1

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
        raise NotImplementedError(f"{type(self).__name__} gives no compute()")


def _set_output(algorithm, output, result):
    """Make ``output`` ``result``, a VTK dataset of its type."""
    if not result.IsA(output.GetClassName()):
        # ShallowCopy between types copies nothing, silently.
        raise TypeError(f"{type(algorithm).__name__} made a {result.GetClassName()}, where "
                        f"its output is a {output.GetClassName()}")
    output.ShallowCopy(result)


class _FieldAlgorithm(TackAlgorithm):
    """A filter that computes fields on its input's points or cells: the output is
    its input, shallow-copied, with the new arrays added. ``fields(piece)`` gives
    ``{name: Field}``."""

    OUTPUT = None

    def RequestData(self, request, inInfo, outInfo):
        input = self.GetInputData(inInfo, 0, 0)
        output = self.GetOutputData(outInfo, 0)
        data = _as_dataset(input, self.device)
        if isinstance(input, vtkPartitionedDataSet):
            # Iterating a partitioned dataset gives its non-empty partitions, the
            # pieces vtk_to_dataset made, in order.
            output.Initialize()
            for block, piece in zip(input, data):
                out, kept = self._with_fields(block, piece)
                output.append(out)
                # Appended as it is: the output's partition is this object.
                if kept is not None:
                    _remember(out, kept)
        else:
            out, kept = self._with_fields(input, data)
            output.ShallowCopy(out)
            if kept is not None:
                _remember(output, kept)
        return 1

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
        fields = self.fields(piece)
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

    def fields(self, data):
        raise NotImplementedError(f"{type(self).__name__} gives no fields()")


# ── Filters that make a dataset ─────────────────────────────────────

class Contour(TackAlgorithm):
    """``td.contour``: the isosurface (isolines in 2D) of a point field at ``value`` (a
    number or a sequence of them)."""

    field = Parameter(None, "the point field's name")
    value = Parameter(0.0, "the isovalue, or a sequence of them")

    def compute(self, data):
        import tack.dataset as td

        return td.contour(data, self.field, self.value)


class SlicePlane(TackAlgorithm):
    """``td.slice_plane``: the cut by the plane through ``origin`` normal to ``normal``."""

    origin = Parameter((0.0, 0.0, 0.0), "a point of the plane")
    normal = Parameter((1.0, 0.0, 0.0), "the plane's normal")

    def compute(self, data):
        import tack.dataset as td

        return td.slice_plane(data, self.origin, self.normal)


class Clip(TackAlgorithm):
    """``td.clip``: the part of the cells where ``by`` (a point field's name, or a Tack
    implicit function such as ``td.Plane``) is above ``value`` (below with
    ``inside``)."""

    by = Parameter(None, "a point field's name, or an implicit function")
    value = Parameter(0.0, "the value the clip is at")
    inside = Parameter(False, "keep the part below the value instead")

    def compute(self, data):
        import tack.dataset as td

        return td.clip(data, self.by, self.value, inside=self.inside)


class Threshold(TackAlgorithm):
    """``td.threshold``: the cells whose ``field`` lies in ``[lower, upper]`` (a point
    field at all of a cell's points, or any with ``all_points=False``)."""

    field = Parameter(None, "the field's name")
    lower = Parameter(float("-inf"), "the lowest value kept")
    upper = Parameter(float("inf"), "the highest value kept")
    all_points = Parameter(True, "a point field: every point of a cell in range")

    def compute(self, data):
        import tack.dataset as td

        return td.threshold(data, self.field, self.lower, self.upper,
                            all_points=self.all_points)


class ExtractSurface(TackAlgorithm):
    """``td.extract_surface``: the boundary faces of the cells, as polygons."""

    def compute(self, data):
        import tack.dataset as td

        return td.extract_surface(data)


class Tetrahedralize(TackAlgorithm):
    """``td.tetrahedralize``: every 3D cell as tetrahedra, conforming across faces."""

    def compute(self, data):
        import tack.dataset as td

        return td.tetrahedralize(data)


class SurfaceNormals(TackAlgorithm):
    """``td.surface_normals``: a surface with ``normals`` on its points and
    ``cell_normals`` on its cells, its polygons wound consistently."""

    consistent = Parameter(True, "wind the polygons consistently first")

    def compute(self, data):
        import tack.dataset as td

        return td.surface_normals(data, consistent=self.consistent)


class ConnectedComponents(TackAlgorithm):
    """``td.connected_components``: ``region`` on the cells and ``point_region`` on the
    points, cells connected through shared points."""

    OUTPUT = None

    def compute(self, data):
        import tack.dataset as td

        return td.connected_components(data)


class Streamlines(TackAlgorithm):
    """``td.streamlines``: the paths of ``seeds`` (an ``(n, 3)`` array) through the
    velocity ``field``, up to ``steps`` fixed steps of ``step_size``."""

    field = Parameter(None, "the velocity field's name")
    seeds = Parameter(None, "the seed points, an (n, 3) array")
    step_size = Parameter(0.01, "the step, in time (or in length with step_unit)")
    steps = Parameter(1000, "the most steps a path takes")
    integrator = Parameter("rk4", "rk4 or euler")
    step_unit = Parameter("time", "time, or length for steps of a fixed arc length")

    def compute(self, data):
        import tack.dataset as td

        return td.streamlines(data, self.field, self.seeds, self.step_size, self.steps,
                              integrator=self.integrator, step_unit=self.step_unit)


class Probe(TackAlgorithm):
    """``td.probe``, as ``vtkProbeFilter``: the fields of the source -- the dataset on
    input port 1 (``SetSourceConnection``, ``SetSourceData``) -- at the points of the
    input (port 0), as point data on the input's structure, with ``valid`` (0 where
    no source cell holds the point). It runs on the source's device unless
    ``device`` says otherwise, the input moved there."""

    OUTPUT = None
    INPUTS = 2

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

    def RequestData(self, request, inInfo, outInfo):
        import tack.dataset as td
        from tack.interop.vtk import dataset_to_vtk

        source = _as_dataset(self.GetInputData(inInfo, 1, 0), self.device)
        device = self.device
        if device is None and not isinstance(source, td.PartitionedDataSet):
            device = source.device
        where = _as_dataset(self.GetInputData(inInfo, 0, 0), device)
        result = dataset_to_vtk(td.probe(source, where, self.fields), copy=False)
        _set_output(self, self.GetOutputData(outInfo, 0), result)
        return 1


# ── Filters that compute fields ─────────────────────────────────────

class Gradients(_FieldAlgorithm):
    """``td.gradients``: the gradient of ``field`` as a new array ``name``, on the
    cells (or the points, ``at="points"``)."""

    field = Parameter(None, "the field's name")
    at = Parameter("cells", "cells or points")
    name = Parameter("gradient", "the new array's name")

    def fields(self, data):
        import tack.dataset as td

        return {self.name: td.gradients(data, self.field, at=self.at)}


class Elevation(_FieldAlgorithm):
    """``td.point_elevation``: each point's place along ``low_point`` to ``high_point``,
    scaled to ``range``, as a point array ``name``."""

    low_point = Parameter((0.0, 0.0, 0.0), "where the elevation is range[0]")
    high_point = Parameter((0.0, 0.0, 1.0), "where it is range[1]")
    range = Parameter((0.0, 1.0), "the elevation's range")
    name = Parameter("elevation", "the new array's name")

    def fields(self, data):
        import tack.dataset as td

        return {self.name: td.point_elevation(data, self.low_point, self.high_point,
                                              self.range)}


class CellQuality(_FieldAlgorithm):
    """``td.cell_quality``: each measure (Verdict's, generalized) as a cell array of its
    name."""

    measures = Parameter(None, "the measures' names (by default each shape's own)")

    def fields(self, data):
        import tack.dataset as td

        return td.cell_quality(data, self.measures)


# ── Filters over every piece ────────────────────────────────────────

class _OverPieces(TackAlgorithm):
    """A filter whose answer needs every piece: it takes a ``vtkPartitionedDataSet``
    whole (a single dataset is one piece)."""

    def execute(self, data):
        from tack.dataset import PartitionedDataSet

        if not isinstance(data, PartitionedDataSet):
            data = PartitionedDataSet([data])
        return self.over_pieces(data)

    def _output_type(self, input):
        if isinstance(input, vtkPartitionedDataSet):
            return vtkPartitionedDataSet
        return vtkPartitionedDataSet


class GhostCells(_OverPieces):
    """``td.ghost_cells``: each piece grown by ``layers`` of its neighbours' cells,
    marked ``DUPLICATECELL``."""

    layers = Parameter(1, "the layers of ghost cells")
    tolerance = Parameter(0.0, "how far apart two pieces' copies of a point may be")

    def over_pieces(self, pieces):
        import tack.dataset as td

        return td.ghost_cells(pieces, layers=self.layers, tolerance=self.tolerance)


class GenerateGlobalIds(_OverPieces):
    """``td.generate_global_ids``: ids of points and cells unique over the pieces, a
    shared point's copies one id (VTK's ``GlobalPointIds`` and ``GlobalCellIds``)."""

    tolerance = Parameter(0.0, "how far apart two pieces' copies of a point may be")

    def over_pieces(self, pieces):
        import tack.dataset as td

        return td.generate_global_ids(pieces, tolerance=self.tolerance)


class MergePieces(_OverPieces):
    """``td.merge_pieces``: the pieces as one dataset, shared points once. Pieces of a
    structured grid that tile it give the grid; anything else an unstructured
    grid."""

    tolerance = Parameter(0.0, "how far apart two pieces' copies of a point may be")

    def _output_type(self, input):
        import vtkmodules.vtkCommonDataModel as dm

        if not isinstance(input, vtkPartitionedDataSet):
            return type(input)
        kinds = {type(block) for block in input}
        if len(kinds) == 1 and next(iter(kinds)) is not dm.vtkPolyData:
            return next(iter(kinds))
        return dm.vtkUnstructuredGrid

    def over_pieces(self, pieces):
        import tack.dataset as td

        return td.merge_pieces(pieces, tolerance=self.tolerance)


del tack
