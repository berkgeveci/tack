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
these filters on a GPU stays there. A partitioned input runs the filter on each
piece and gives a ``vtkPartitionedDataSet``; the filters whose answer needs every
piece (``GhostCells``, ``GenerateGlobalIds``, ``MergePieces``) take them together.

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

    #: The exception the last run raised, or ``None``.
    error = None

    def __init__(self, **parameters):
        # The output port says vtkDataObject: RequestDataObject makes the output's
        # type from the input's (a partitioned dataset for pieces), which the
        # executive would replace if the port named a narrower one.
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, inputType="vtkDataSet",
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
        from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

        input = self.GetInputData(inInfo, 0, 0)
        output = self.GetOutputData(outInfo, 0)
        result = dataset_to_vtk(self.execute(vtk_to_dataset(input, dtype=None, copy=False)),
                                copy=False)
        if not result.IsA(output.GetClassName()):
            # ShallowCopy between types copies nothing, silently.
            raise TypeError(f"{type(self).__name__} made a {result.GetClassName()}, where its "
                            f"output is a {output.GetClassName()}")
        output.ShallowCopy(result)
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


class _FieldAlgorithm(TackAlgorithm):
    """A filter that computes fields on its input's points or cells: the output is
    its input, shallow-copied, with the new arrays added. ``fields(piece)`` gives
    ``{name: Field}``."""

    OUTPUT = None

    def RequestData(self, request, inInfo, outInfo):
        from tack.interop.vtk import vtk_to_dataset

        input = self.GetInputData(inInfo, 0, 0)
        output = self.GetOutputData(outInfo, 0)
        data = vtk_to_dataset(input, dtype=None, copy=False)
        if isinstance(input, vtkPartitionedDataSet):
            # Iterating a partitioned dataset gives its non-empty partitions, the
            # pieces vtk_to_dataset made, in order.
            output.Initialize()
            for block, piece in zip(input, data):
                output.append(self._with_fields(block, piece))
        else:
            output.ShallowCopy(self._with_fields(input, data))
        return 1

    def _with_fields(self, block, piece):
        """A new VTK dataset of ``block``'s structure and arrays, and the fields
        computed on ``piece``, its Tack dataset, added to it."""
        from tack.dataset import H1, Constant, Values
        from tack.interop.vtk import _label, _to_vtk_values

        out = block.NewInstance()
        out.ShallowCopy(block)
        for name, field in self.fields(piece).items():
            space = field.space
            on_points = isinstance(space, H1) or (isinstance(space, Values)
                                                  and space.on == "points")
            if not on_points and not (isinstance(space, Constant)
                                      or (isinstance(space, Values) and space.on == "cells")):
                raise TypeError(f"{name!r} is on {space!r}, which VTK keeps nowhere")
            attributes = out.point_data if on_points else out.cell_data
            attributes[name] = _to_vtk_values(field.values, share=True)
            _label(attributes[name], field.unit)
        return out

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
