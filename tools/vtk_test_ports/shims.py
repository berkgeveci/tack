# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""VTK's filters replaced by Tack's, for running VTK's own Python tests on Tack.

``install()`` puts a stand-in in place of each VTK class below, in its
``vtkmodules`` module (and ``vtkmodules.all``, ``vtk``), so a test script built on
VTK's API runs unchanged. A stand-in keeps a real VTK filter of the class, its
twin: every call the stand-in does not answer (``SetValue``, ``SetCutFunction``,
``GenerateTrianglesOff``, ...) goes to the twin, so the twin holds the test's
settings. When the pipeline runs the stand-in, it reads the twin's settings,
runs the Tack wrapper they translate to (``tack.interop.vtk.algorithms``), and
gives its output, converted to the class VTK's filter gives (an unstructured grid
to ``vtkPolyData``); it also runs the twin on the same input and records the two
outputs compared (``LOG``). A setting with no Tack counterpart raises
``Unsupported``: the stand-in then gives the twin's output, so the test still
runs, and records why.
"""

import math

import numpy as np
import vtkmodules.vtkCommonDataModel as dm
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

#: One record per stand-in execution.
LOG = []
#: The real VTK classes the stand-ins replaced, which the harness's own work uses.
REAL = {}


def _real(name, module):
    """VTK's own class ``name``, never a stand-in."""
    if name in REAL:
        return REAL[name]
    import importlib

    return getattr(importlib.import_module(f"vtkmodules.{module}"), name)
#: The test being run, for the records.
CURRENT = {"test": None}


class Unsupported(Exception):
    """A setting of the VTK filter that Tack has no counterpart for."""


def _ta():
    from tack.interop.vtk import algorithms as ta

    return ta


# ── Reading the twin's settings ──────────────────────────────────────

def _array(twin, data, index=0, kind="scalars"):
    """The array the twin would process: SetInputArrayToProcess's, else the
    input's active scalars (or vectors), as ``(name, 'points' | 'cells')``."""
    info = twin.GetInputArrayInformation(index)
    if info is not None and info.Has(dm.vtkDataObject.FIELD_NAME()):
        name = info.Get(dm.vtkDataObject.FIELD_NAME())
        association = info.Get(dm.vtkDataObject.FIELD_ASSOCIATION()) \
            if info.Has(dm.vtkDataObject.FIELD_ASSOCIATION()) else 0
        return name, ("cells" if association == 1 else "points")
    leaf = _first_leaf(data)
    if leaf is None:
        raise Unsupported("no input dataset")
    for attributes, where in ((leaf.GetPointData(), "points"), (leaf.GetCellData(), "cells")):
        array = attributes.GetScalars() if kind == "scalars" else attributes.GetVectors()
        if array is not None:
            if not array.GetName():
                raise Unsupported(f"the active {kind} have no name")
            return array.GetName(), where
    raise Unsupported(f"no active {kind}")


def _first_leaf(data):
    if isinstance(data, dm.vtkCompositeDataSet):
        it = data.NewIterator()
        it.InitTraversal()
        return None if it.IsDoneWithTraversal() else it.GetCurrentDataObject()
    return data


def _values(twin):
    return [twin.GetValue(i) for i in range(twin.GetNumberOfContours())]


# ── Translations: the twin's settings as a Tack wrapper ──────────────

def _no_polygons(twin):
    if hasattr(twin, "GetGenerateTriangles") and not twin.GetGenerateTriangles():
        raise Unsupported("GenerateTriangles off: Tack gives triangles (issue #351)")


def _contour(twin, inputs):
    _no_polygons(twin)
    ta = _ta()
    name, where = _array(twin, inputs[0])
    if where != "points":
        raise Unsupported("contour of cell data")
    component = twin.GetArrayComponent() if hasattr(twin, "GetArrayComponent") else 0
    values = _values(twin)
    if not values:
        raise Unsupported("no contour values")
    return ta.Contour(field=(name, "points"), value=values if len(values) > 1 else values[0],
                      component=component)


def _cutter(twin, inputs):
    _no_polygons(twin)
    ta = _ta()
    function = twin.GetCutFunction()
    if function is None:
        raise Unsupported("no cut function")
    values = _values(twin) or [0.0]
    return ta.Slice(function=function, value=values if len(values) > 1 else values[0])


def _plane_cutter(twin, inputs):
    ta = _ta()
    plane = twin.GetPlane()
    if plane is None:
        raise Unsupported("no plane")
    return ta.SlicePlane(origin=tuple(plane.GetOrigin()), normal=tuple(plane.GetNormal()))


def _clip(twin, inputs):
    ta = _ta()
    function = twin.GetClipFunction()
    if getattr(twin, "GetUseValueAsOffset", lambda: True)() is False and function is not None:
        raise Unsupported("UseValueAsOffset off")
    by = function if function is not None else _array(twin, inputs[0])
    if function is None and by[1] != "points":
        raise Unsupported("clip by cell data")
    return ta.Clip(by=by, value=twin.GetValue(), inside=bool(twin.GetInsideOut()))


def _threshold(twin, inputs):
    ta = _ta()
    if twin.GetUseContinuousCellRange():
        raise Unsupported("UseContinuousCellRange")
    if twin.GetInvert():
        raise Unsupported("Invert")
    lo, hi = twin.GetLowerThreshold(), twin.GetUpperThreshold()
    mode = twin.GetThresholdFunction()
    lower, upper = {0: (lo, hi), 1: (-math.inf, lo), 2: (hi, math.inf)}[mode]
    name, where = _array(twin, inputs[0])
    if twin.GetComponentMode() != 0:                    # USE_ALL, USE_ANY
        raise Unsupported("a component mode other than the selected component")
    component = twin.GetSelectedComponent()
    leaf = _first_leaf(inputs[0])
    attributes = leaf.GetPointData() if where == "points" else leaf.GetCellData()
    array = attributes.GetArray(name)
    if array is not None and array.GetNumberOfComponents() > 1 and \
            component >= array.GetNumberOfComponents():
        component = "magnitude"                         # VTK's: the count of components
    return ta.Threshold(field=(name, where), lower=lower, upper=upper,
                        all_points=bool(twin.GetAllScalars()), component=component)


def _elevation(twin, inputs):
    ta = _ta()
    return ta.Elevation(low_point=tuple(twin.GetLowPoint()), high_point=tuple(twin.GetHighPoint()),
                        range=tuple(twin.GetScalarRange()), name="Elevation")


def _shrink(twin, inputs):
    return _ta().Shrink(factor=twin.GetShrinkFactor())


def _probe(twin, inputs):
    if twin.GetCategoricalData() or twin.GetComputeTolerance() is False:
        raise Unsupported("categorical data or a set tolerance")
    return _ta().Probe()


def _warp_vector(twin, inputs):
    name, where = _array(twin, inputs[0], kind="vectors")
    if where != "points":
        raise Unsupported("warp by cell vectors")
    return _ta().Warp(direction=name, scale=twin.GetScaleFactor())


def _warp_scalar(twin, inputs):
    name, where = _array(twin, inputs[0])
    if where != "points":
        raise Unsupported("warp by cell scalars")
    leaf = _first_leaf(inputs[0])
    normals = leaf.GetPointData().GetNormals() if leaf is not None else None
    if twin.GetUseNormal() or normals is None:
        direction = tuple(twin.GetNormal())
    else:
        if not normals.GetName():
            raise Unsupported("unnamed normals")
        direction = normals.GetName()
    if getattr(twin, "GetXYPlane", lambda: 0)():
        raise Unsupported("XYPlane")
    return _ta().Warp(direction=direction, scale=twin.GetScaleFactor(), scale_by=name)


def _extract_geometry(twin, inputs):
    if twin.GetExtractOnlyBoundaryCells():
        raise Unsupported("ExtractOnlyBoundaryCells")
    return _ta().ExtractGeometry(function=twin.GetImplicitFunction(),
                                 inside=bool(twin.GetExtractInside()),
                                 boundary=bool(twin.GetExtractBoundaryCells()))


def _extract_grid(twin, inputs):
    if getattr(twin, "GetIncludeBoundary", lambda: 0)():
        raise Unsupported("IncludeBoundary")
    rate = tuple(twin.GetSampleRate())
    return _ta().ExtractExtent(extent=tuple(twin.GetVOI()), rate=rate)


def _surface(twin, inputs):
    if hasattr(twin, "GetRemoveGhostInterfaces") and not twin.GetRemoveGhostInterfaces():
        raise Unsupported("RemoveGhostInterfaces off: Tack leaves out faces shared with "
                          "ghost cells")
    for option in ("CellClipping", "PointClipping", "ExtentClipping"):
        if getattr(twin, f"Get{option}", lambda: 0)():
            raise Unsupported(option)
    leaf = _first_leaf(inputs[0])
    if leaf is None or leaf.GetNumberOfCells() == 0:
        raise Unsupported("no cells")
    dims = {leaf.GetCell(i).GetCellDimension() for i in range(min(leaf.GetNumberOfCells(), 64))}
    if dims != {3}:
        raise Unsupported("the surface of cells that are not 3D (VTK passes them through)")
    return _ta().ExtractSurface()


def _triangle(twin, inputs):
    return _ta().Triangulate()


def _tetrahedralize(twin, inputs):
    return _ta().Tetrahedralize()


def _normals(twin, inputs):
    if twin.GetSplitting():
        raise Unsupported("Splitting (sharp edges)")
    if twin.GetFlipNormals():
        raise Unsupported("FlipNormals")
    return _ta().SurfaceNormals(consistent=bool(twin.GetConsistency()))


def _feature_edges(twin, inputs):
    return _ta().FeatureEdges(angle=twin.GetFeatureAngle(), boundary=bool(twin.GetBoundaryEdges()),
                              feature=bool(twin.GetFeatureEdges()),
                              non_manifold=bool(twin.GetNonManifoldEdges()),
                              manifold=bool(twin.GetManifoldEdges()))


def _extract_edges(twin, inputs):
    return _ta().ExtractEdges()


def _tubes(twin, inputs):
    if twin.GetVaryRadius() != 0:
        raise Unsupported("VaryRadius")
    return _ta().Tubes(radius=twin.GetRadius(), sides=twin.GetNumberOfSides())


def _transform(twin, inputs):
    transform = twin.GetTransform()
    if transform is None or not transform.IsA("vtkLinearTransform"):
        raise Unsupported("a transform that is not linear")
    m = transform.GetMatrix()
    return _ta().Transform(matrix=[[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def _gradient(twin, inputs):
    if twin.GetComputeVorticity() or twin.GetComputeQCriterion() or twin.GetComputeDivergence():
        raise Unsupported("vorticity, Q or divergence")
    name, where = _array(twin, inputs[0])
    return _ta().Gradients(field=(name, where), at="points" if where == "points" else "cells",
                           name=twin.GetResultArrayName() or "Gradients")


def _integrate(twin, inputs):
    return _ta().Integrate()


def _mask_points(twin, inputs):
    if twin.GetRandomMode():
        raise Unsupported("RandomMode")
    if twin.GetOffset():
        raise Unsupported("Offset")
    leaf = _first_leaf(inputs[0])
    if leaf is not None and twin.GetMaximumNumberOfPoints() < leaf.GetNumberOfPoints() / max(
            twin.GetOnRatio(), 1):
        raise Unsupported("MaximumNumberOfPoints")
    return _ta().MaskPoints(stride=twin.GetOnRatio())


def _threshold_points(twin, inputs):
    name, where = _array(twin, inputs[0])
    if where != "points":
        raise Unsupported("cell scalars")
    mode = twin.GetThresholdFunction()
    lo, hi = twin.GetLowerThreshold(), twin.GetUpperThreshold()
    lower, upper = {0: (lo, hi), 1: (-math.inf, lo), 2: (hi, math.inf)}.get(mode, (lo, hi))
    return _ta().ThresholdPoints(field=name, lower=lower, upper=upper)


class _Sequence:
    """Tack wrappers run one after another on the stand-in's inputs (the first port's
    data through them all, the others to each), with a last step ``finish(output,
    inputs)`` that may build the VTK output; as the stand-in drives one wrapper."""

    def __init__(self, label, steps, finish=None):
        self.label, self.steps, self.finish = label, steps, finish
        self.error, self._inputs, self._out = None, {}, None

    def SetInputDataObject(self, port, data):
        self._inputs[port] = data

    def Update(self):
        data = self._inputs[0]
        for step in self.steps:
            step.SetInputDataObject(0, data)
            for port, other in self._inputs.items():
                if port:
                    step.SetInputDataObject(port, other)
            step.Update()
            if step.error is not None:
                self.error = step.error
                return
            data = step.GetOutputDataObject(0)
        self._out = self.finish(data, self._inputs) if self.finish else data

    def GetOutputDataObject(self, port):
        return self._out


def _cell_length(leaf):
    """A typical cell's length: the bounds' volume (area, length) over the cells, to
    the dimension's root."""
    b = np.array(leaf.GetBounds()).reshape(3, 2)
    sides = [hi - lo for lo, hi in b if hi - lo > 0]
    if not sides or leaf.GetNumberOfCells() == 0:
        return 1.0
    return float(np.prod(sides) / leaf.GetNumberOfCells()) ** (1.0 / len(sides))


def _stream_tracer(twin, inputs):
    if twin.GetIntegrationDirection() != 0:
        raise Unsupported("integration BACKWARD or BOTH")
    integrator = {0: "rk4", 1: "rk4", 2: "rk45"}[twin.GetIntegratorType()]
    name, where = _array(twin, inputs[0], kind="vectors")
    leaf = _first_leaf(inputs[0])
    step = twin.GetInitialIntegrationStep()
    if twin.GetIntegrationStepUnit() == 2:                      # CELL_LENGTH_UNIT
        step *= _cell_length(leaf)
    seeds = None if len(inputs) > 1 and inputs[1] is not None else \
        np.array([twin.GetStartPosition()])
    return _ta().Streamlines(field=(name, where), seeds=seeds, step_size=step,
                             step_unit="length", steps=twin.GetMaximumNumberOfSteps(),
                             max_length=twin.GetMaximumPropagation(), integrator=integrator)


def _point_to_cell(twin, inputs):
    leaf = _first_leaf(inputs[0])
    names = [leaf.GetPointData().GetArrayName(i)
             for i in range(leaf.GetPointData().GetNumberOfArrays())]
    names = [n for n in names if n and not n.startswith("vtk")]
    if not names:
        raise Unsupported("no point arrays")
    return _Sequence("Project", [_ta().Project(field=(n, "points"), to="cells", name=n,
                                               average="nodes") for n in names])


def _cell_to_point(twin, inputs):
    leaf = _first_leaf(inputs[0])
    names = [leaf.GetCellData().GetArrayName(i)
             for i in range(leaf.GetCellData().GetNumberOfArrays())]
    names = [n for n in names if n and not n.startswith("vtk")]
    if not names:
        raise Unsupported("no cell arrays")
    return _Sequence("Project", [_ta().Project(field=(n, "cells"), to="points", name=n)
                                 for n in names])


def _sample_implicit(twin, inputs):
    return _ta().ImplicitValues(function=twin.GetImplicitFunction(),
                                name=twin.GetScalarArrayName() or "Implicit scalars")


def _connectivity(twin, inputs):
    if twin.GetExtractionMode() != 5:                           # VTK_EXTRACT_ALL_REGIONS
        raise Unsupported(f"extraction mode {twin.GetExtractionModeAsString()}")
    return _ta().ConnectedComponents()


def _cell_centers(twin, inputs):
    def finish(data, given):
        out = dm.vtkPolyData()
        leaf = data
        centers = np.asarray(leaf.GetCellData().GetArray("__center"), np.float64)
        from vtkmodules.vtkCommonCore import vtkPoints

        points = vtkPoints()
        points.SetDataTypeToDouble()
        for x in centers.reshape(-1, 3):
            points.InsertNextPoint(*x)
        out.SetPoints(points)
        if twin.GetVertexCells():
            verts = dm.vtkCellArray()
            for i in range(len(centers)):
                verts.InsertNextCell(1, [i])
            out.SetVerts(verts)
        if twin.GetCopyArrays():
            cells = given[0].GetCellData() if isinstance(given[0], dm.vtkDataSet) else None
            for i in range(cells.GetNumberOfArrays() if cells is not None else 0):
                out.GetPointData().AddArray(cells.GetAbstractArray(i))
        return out

    if not isinstance(inputs[0], dm.vtkDataSet):
        raise Unsupported("a composite input")
    return _Sequence("CellCenters", [_ta().CellCenters(name="__center")], finish)


def _generate_ids(twin, inputs):
    steps = []
    if twin.GetPointIds():
        steps.append(_ta().PointIds(name=twin.GetPointIdsArrayName()))
    if twin.GetCellIds():
        steps.append(_ta().CellIds(name=twin.GetCellIdsArrayName()))
    if not steps:
        raise Unsupported("neither point nor cell ids")
    return _Sequence("Ids", steps)


def _static_clean(twin, inputs):
    if twin.GetToleranceIsAbsolute():
        return _ta().MergePoints(tolerance=twin.GetAbsoluteTolerance())
    # A tolerance relative to the bounds' diagonal.
    leaf = _first_leaf(inputs[0])
    b = np.array(leaf.GetBounds()).reshape(3, 2)
    return _ta().MergePoints(tolerance=twin.GetTolerance() * float(np.linalg.norm(
        b[:, 1] - b[:, 0])))


def _remove_ghosts(twin, inputs):
    return _ta().RemoveGhosts()


def _glyph(twin, inputs):
    raise Unsupported("Tack's Glyphs are arrows; VTK's glyph source is any dataset")


#: VTK class name: (module, translation).
TRANSLATIONS = {
    "vtkStreamTracer": ("vtkFiltersFlowPaths", _stream_tracer),
    "vtkPointDataToCellData": ("vtkFiltersCore", _point_to_cell),
    "vtkCellDataToPointData": ("vtkFiltersCore", _cell_to_point),
    "vtkSampleImplicitFunctionFilter": ("vtkFiltersGeneral", _sample_implicit),
    "vtkConnectivityFilter": ("vtkFiltersCore", _connectivity),
    "vtkPolyDataConnectivityFilter": ("vtkFiltersCore", _connectivity),
    "vtkCellCenters": ("vtkFiltersCore", _cell_centers),
    "vtkClipVolume": ("vtkFiltersGeneral", _clip),
    "vtkGenerateIds": ("vtkFiltersCore", _generate_ids),
    "vtkStaticCleanUnstructuredGrid": ("vtkFiltersCore", _static_clean),
    "vtkRemoveGhosts": ("vtkFiltersParallel", _remove_ghosts),
    "vtkExtractRectilinearGrid": ("vtkFiltersExtraction", _extract_grid),
    "vtkGlyph3D": ("vtkFiltersCore", _glyph),
    "vtkContourFilter": ("vtkFiltersCore", _contour),
    "vtkFlyingEdges3D": ("vtkFiltersCore", _contour),
    "vtkFlyingEdges2D": ("vtkFiltersCore", _contour),
    "vtkMarchingCubes": ("vtkFiltersCore", _contour),
    "vtkMarchingSquares": ("vtkFiltersCore", _contour),
    "vtkSynchronizedTemplates3D": ("vtkFiltersCore", _contour),
    "vtkContourGrid": ("vtkFiltersCore", _contour),
    "vtkContour3DLinearGrid": ("vtkFiltersCore", _contour),
    "vtkCutter": ("vtkFiltersCore", _cutter),
    "vtkPlaneCutter": ("vtkFiltersCore", _plane_cutter),
    "vtk3DLinearGridPlaneCutter": ("vtkFiltersCore", _plane_cutter),
    "vtkClipDataSet": ("vtkFiltersGeneral", _clip),
    "vtkTableBasedClipDataSet": ("vtkFiltersGeneral", _clip),
    "vtkClipPolyData": ("vtkFiltersCore", _clip),
    "vtkThreshold": ("vtkFiltersCore", _threshold),
    "vtkElevationFilter": ("vtkFiltersCore", _elevation),
    "vtkShrinkFilter": ("vtkFiltersGeneral", _shrink),
    "vtkProbeFilter": ("vtkFiltersCore", _probe),
    "vtkWarpVector": ("vtkFiltersGeneral", _warp_vector),
    "vtkWarpScalar": ("vtkFiltersGeneral", _warp_scalar),
    "vtkExtractGeometry": ("vtkFiltersExtraction", _extract_geometry),
    "vtkExtractGrid": ("vtkFiltersExtraction", _extract_grid),
    "vtkExtractVOI": ("vtkImagingCore", _extract_grid),
    "vtkDataSetSurfaceFilter": ("vtkFiltersGeometry", _surface),
    "vtkGeometryFilter": ("vtkFiltersGeometry", _surface),
    "vtkTriangleFilter": ("vtkFiltersCore", _triangle),
    "vtkDataSetTriangleFilter": ("vtkFiltersGeneral", _tetrahedralize),
    "vtkPolyDataNormals": ("vtkFiltersCore", _normals),
    "vtkFeatureEdges": ("vtkFiltersCore", _feature_edges),
    "vtkExtractEdges": ("vtkFiltersCore", _extract_edges),
    "vtkTubeFilter": ("vtkFiltersCore", _tubes),
    "vtkTransformFilter": ("vtkFiltersGeneral", _transform),
    "vtkTransformPolyDataFilter": ("vtkFiltersGeneral", _transform),
    "vtkGradientFilter": ("vtkFiltersGeneral", _gradient),
    "vtkIntegrateAttributes": ("vtkFiltersParallel", _integrate),
    "vtkMaskPoints": ("vtkFiltersCore", _mask_points),
    "vtkThresholdPoints": ("vtkFiltersCore", _threshold_points),
}


# ── The stand-in ─────────────────────────────────────────────────────

class _Shim(VTKPythonAlgorithmBase):
    """A VTK filter run by Tack; see the module."""

    REAL = None              # the VTK class
    TRANSLATE = None

    def __init__(self, *args, **kwargs):
        twin = self.REAL()
        self.__dict__["_twin"] = twin
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=twin.GetNumberOfInputPorts(),
                                        nOutputPorts=twin.GetNumberOfOutputPorts(),
                                        outputType="vtkDataObject")
        for name, value in kwargs.items():          # the pythonic constructor's properties
            setattr(self, name, value)

    # Calls the stand-in does not answer go to the twin; a setting marks the
    # stand-in modified, so the pipeline runs it again.
    def __getattr__(self, name):
        attribute = getattr(self.__dict__["_twin"], name)
        if callable(attribute) and (name.startswith(("Set", "Add", "Remove", "Initialize",
                                                     "Generate")) or name.endswith(("On", "Off"))):
            def call(*args):
                out = attribute(*args)
                self.Modified()
                return out
            return call
        return attribute

    def __setattr__(self, name, value):
        if name.startswith("_") or name in type(self).__dict__:
            self.__dict__[name] = value
            return
        # A pythonic property (``value=...``) is the twin's.
        setattr(self.__dict__["_twin"], name, value)
        self.Modified()

    def SetInputArrayToProcess(self, *args):
        self._twin.SetInputArrayToProcess(*args)
        VTKPythonAlgorithmBase.SetInputArrayToProcess(self, *args)

    def SetInputData(self, *args):
        if len(args) == 1:
            self.SetInputDataObject(0, args[0])
        else:
            self.SetInputDataObject(*args)

    def AddInputData(self, *args):
        self.AddInputDataObject(*args)

    def SetSourceConnection(self, port):
        self.SetInputConnection(1, port)

    def SetSourceData(self, data):
        self.SetInputDataObject(1, data)

    def GetOutput(self, port=0):
        return self.GetOutputDataObject(port)

    def GetPolyDataOutput(self):
        return self.GetOutputDataObject(0)

    def FillInputPortInformation(self, port, info):
        info.Copy(self._twin.GetInputPortInformation(port))
        return 1

    def _inputs(self, inInfo):
        return [self.GetInputData(inInfo, p, 0) if inInfo[p].GetNumberOfInformationObjects()
                else None for p in range(self.GetNumberOfInputPorts())]

    def _run_twin(self, inputs):
        for p, data in enumerate(inputs):
            if data is not None:
                self._twin.SetInputDataObject(p, data)
        self._twin.Update()
        return [self._twin.GetOutputDataObject(p) for p in range(self.GetNumberOfOutputPorts())]

    def RequestDataObject(self, request, inInfo, outInfo):
        inputs = self._inputs(inInfo)
        for p, data in enumerate(inputs):
            if data is not None:
                self._twin.SetInputDataObject(p, data)
        self._twin.UpdateDataObject()
        for p in range(self.GetNumberOfOutputPorts()):
            kind = type(self._twin.GetOutputDataObject(p))
            info = outInfo.GetInformationObject(p)
            current = info.Get(dm.vtkDataObject.DATA_OBJECT())
            if current is None or type(current) is not kind:
                info.Set(dm.vtkDataObject.DATA_OBJECT(), kind())
        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        # The twin's extent and times only: its whole information holds its own
        # output object, which would stand for this one's.
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as s

        self._twin.UpdateInformation()
        for p in range(self.GetNumberOfOutputPorts()):
            twin, out = self._twin.GetOutputInformation(p), outInfo.GetInformationObject(p)
            for key in (s.WHOLE_EXTENT(), s.TIME_STEPS(), s.TIME_RANGE()):
                if twin.Has(key):
                    out.CopyEntry(twin, key)
                else:
                    out.Remove(key)
        return 1

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        # The whole of each input: the twin and the Tack filter both read it whole.
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as s

        for p in range(self.GetNumberOfInputPorts()):
            for i in range(inInfo[p].GetNumberOfInformationObjects()):
                info = inInfo[p].GetInformationObject(i)
                if info.Has(s.WHOLE_EXTENT()):
                    info.Set(s.UPDATE_EXTENT(), info.Get(s.WHOLE_EXTENT()), 6)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        given = self._inputs(inInfo)
        references = self._run_twin(given)
        inputs = [_named(data) for data in given]
        record = {"test": CURRENT["test"], "filter": self.REAL.__name__,
                  "input": inputs[0].GetClassName() if inputs[0] is not None else None}
        result = None
        try:
            tack_filter = type(self).TRANSLATE(self._twin, inputs)
            record["tack"] = getattr(tack_filter, "label", type(tack_filter).__name__)
            for p, data in enumerate(inputs):
                if data is not None:
                    tack_filter.SetInputDataObject(p, data)
            tack_filter.Update()
            if tack_filter.error is not None:
                raise tack_filter.error
            result = _as_class(tack_filter.GetOutputDataObject(0), references[0])
            record.update(compare(references[0], result))
            record["active"] = _match_active(references[0], result)
            record["used"] = "tack"
        except Unsupported as reason:
            record["used"], record["why"] = "vtk", str(reason)
        except Exception as error:
            import traceback

            record["used"], record["error"] = "vtk", f"{type(error).__name__}: {error}"[:500]
            record["trace"] = "".join(traceback.format_exception(error))[-2500:]
            result = None
        LOG.append(record)
        for p in range(self.GetNumberOfOutputPorts()):
            out = self.GetOutputData(outInfo, p)
            out.ShallowCopy(result if (p == 0 and result is not None) else references[p])
        return 1


_ATTRIBUTES = (("Scalars", "scalars"), ("Vectors", "vectors"), ("Normals", "normals"),
               ("TCoords", "tcoords"))


def _named(data):
    """``data``, or a shallow copy whose unnamed active arrays (scalars, vectors,
    normals) have names, for Tack, which takes fields by name; older VTK tests leave
    them unnamed."""
    if not isinstance(data, dm.vtkDataSet):
        return data
    unnamed = [(attributes, kind) for attributes in (data.GetPointData(), data.GetCellData())
               for kind, _ in _ATTRIBUTES[:3]
               if getattr(attributes, f"Get{kind}")() is not None
               and not getattr(attributes, f"Get{kind}")().GetName()]
    if not unnamed:
        return data
    copy = data.NewInstance()
    copy.ShallowCopy(data)
    for attributes in (copy.GetPointData(), copy.GetCellData()):
        for kind, name in _ATTRIBUTES[:3]:
            array = getattr(attributes, f"Get{kind}")()
            if array is not None and not array.GetName():
                named = array.NewInstance()
                named.ShallowCopy(array)
                named.SetName(name)
                getattr(attributes, f"Set{kind}")(named)
    return copy


def _match_active(reference, result):
    """Make ``result``'s active arrays (scalars, vectors, normals, texture
    coordinates) those of ``reference`` where it has an array of the name, and say
    which VTK makes active and Tack did not."""
    notes = []
    if not (isinstance(reference, dm.vtkDataSet) and isinstance(result, dm.vtkDataSet)):
        return notes
    for where, ref, out in (("points", reference.GetPointData(), result.GetPointData()),
                            ("cells", reference.GetCellData(), result.GetCellData())):
        for kind, given in _ATTRIBUTES:
            active = getattr(ref, f"Get{kind}")()
            if active is None:
                continue
            # An unnamed array went to Tack by the name _named gave it.
            name = active.GetName() or given
            mine = getattr(out, f"Get{kind}")()
            if mine is not None and mine.GetName() == name:
                continue
            if out.GetAbstractArray(name) is not None:
                getattr(out, f"SetActive{kind}")(name)
                notes.append(f"{name} active {kind.lower()} on {where} in VTK only")
    return notes


def _as_class(result, reference):
    """``result`` as an object of ``reference``'s class: an unstructured grid of
    surface cells as ``vtkPolyData`` (through ``vtkGeometryFilter``), a tree block by
    block."""
    if result.IsA(reference.GetClassName()):
        return result
    if isinstance(reference, dm.vtkPolyData):
        geometry = _real("vtkGeometryFilter", "vtkFiltersGeometry")()
        geometry.SetInputData(result)
        geometry.Update()
        return geometry.GetOutput()
    if isinstance(reference, dm.vtkUnstructuredGrid) and isinstance(result, dm.vtkPolyData):
        # VTK's dataset clip gives a grid where Tack keeps a polygonal input polygonal.
        append = _real("vtkAppendFilter", "vtkFiltersCore")()
        append.SetInputData(result)
        append.Update()
        return append.GetOutput()
    if isinstance(reference, dm.vtkCompositeDataSet) and isinstance(result, dm.vtkCompositeDataSet):
        return _leaves_as(result, reference)
    raise TypeError(f"Tack gave a {result.GetClassName()}, VTK a {reference.GetClassName()}")


def _leaves_as(result, reference):
    """A tree ``result`` with each dataset leaf made ``vtkPolyData`` where ``reference``'s
    leaves are (a composite polydata mapper takes nothing else)."""
    want_polydata = any(isinstance(leaf, dm.vtkPolyData) for leaf in _leaves(reference))
    if not want_polydata:
        return result
    out = result.NewInstance()
    out.CopyStructure(result)
    it = result.NewIterator()
    it.InitTraversal()
    while not it.IsDoneWithTraversal():
        leaf = it.GetCurrentDataObject()
        if isinstance(leaf, dm.vtkDataSet) and not isinstance(leaf, dm.vtkPolyData):
            geometry = _real("vtkGeometryFilter", "vtkFiltersGeometry")()
            geometry.SetInputData(leaf)
            geometry.Update()
            leaf = geometry.GetOutput()
        out.SetDataSet(it, leaf)
        it.GoToNextItem()
    return out


# ── Comparing outputs ────────────────────────────────────────────────

def _leaves(data):
    if data is None:
        return []
    if isinstance(data, dm.vtkCompositeDataSet):
        out, it = [], data.NewIterator()
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            out.append(it.GetCurrentDataObject())
            it.GoToNextItem()
        return out
    return [data] if isinstance(data, dm.vtkDataSet) else []


def _triangulated(leaf):
    """``leaf`` with its polygons as triangles by VTK's vtkTriangleFilter (ear cutting):
    vtkIntegrateAttributes fans a polygon from its first corner, wrong where it is
    concave."""
    types = {leaf.GetCellType(i) for i in range(leaf.GetNumberOfCells())} \
        if not isinstance(leaf, (dm.vtkImageData, dm.vtkRectilinearGrid,
                                 dm.vtkStructuredGrid)) else set()
    if not types & {7}:                               # VTK_POLYGON
        return leaf
    surface = leaf
    if not isinstance(leaf, dm.vtkPolyData):
        geometry = _real("vtkGeometryFilter", "vtkFiltersGeometry")()
        geometry.SetInputData(leaf)
        geometry.Update()
        surface = geometry.GetOutput()
    triangles = _real("vtkTriangleFilter", "vtkFiltersCore")()
    triangles.SetInputData(surface)
    triangles.Update()
    return triangles.GetOutput()


def _measure(data):
    """Total length, area and volume, by VTK's vtkIntegrateAttributes."""
    vtkIntegrateAttributes = _real("vtkIntegrateAttributes", "vtkFiltersParallel")
    sums = {"Length": 0.0, "Area": 0.0, "Volume": 0.0}
    for leaf in _leaves(data):
        if leaf.GetNumberOfCells() == 0:
            continue
        integrate = vtkIntegrateAttributes()
        integrate.SetInputData(_triangulated(leaf))
        integrate.Update()
        out = integrate.GetOutput()
        for name in sums:
            array = out.GetCellData().GetArray(name)
            if array is not None:
                sums[name] += array.GetValue(0)
    return sums


def _used_bounds(leaf):
    """The bounds of the points ``leaf``'s cells use (all its points if it has no cells,
    or is structured): VTK's clip keeps unused points, Tack's feature edges too."""
    if leaf.GetNumberOfCells() == 0 or isinstance(
            leaf, (dm.vtkImageData, dm.vtkRectilinearGrid, dm.vtkStructuredGrid)):
        return leaf.GetBounds()
    if isinstance(leaf, dm.vtkPolyData):
        arrays = [cells.GetConnectivityArray() for cells in
                  (leaf.GetVerts(), leaf.GetLines(), leaf.GetPolys(), leaf.GetStrips())]
    elif isinstance(leaf, dm.vtkUnstructuredGrid):
        arrays = [leaf.GetCells().GetConnectivityArray()]
    else:
        return leaf.GetBounds()
    ids = np.unique(np.concatenate([np.asarray(a, np.int64) for a in arrays
                                    if a is not None and a.GetNumberOfTuples()] or
                                   [np.zeros(0, np.int64)]))
    ids = ids[ids >= 0]
    if not ids.size:
        return leaf.GetBounds()
    x = np.asarray(leaf.GetPoints().GetData(), np.float64)[ids]
    lo, hi = x.min(axis=0), x.max(axis=0)
    return lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]


def _bounds(data):
    boxes = [_used_bounds(leaf) for leaf in _leaves(data) if leaf.GetNumberOfPoints()]
    if not boxes:
        return None
    b = np.array(boxes)
    return [float(b[:, 0].min()), float(b[:, 1].max()), float(b[:, 2].min()),
            float(b[:, 3].max()), float(b[:, 4].min()), float(b[:, 5].max())]


def _array_summary(data):
    out = {}
    for leaf in _leaves(data):
        for attributes, where in ((leaf.GetPointData(), "points"), (leaf.GetCellData(), "cells")):
            for i in range(attributes.GetNumberOfArrays()):
                array = attributes.GetArray(i)
                if array is None or not array.GetName() or array.GetName().startswith("vtk") \
                        or array.GetNumberOfTuples() == 0:
                    continue
                values = np.asarray(array, dtype=np.float64).reshape(array.GetNumberOfTuples(), -1)
                key = (array.GetName(), where)
                if values.size:
                    lo, hi = values.min(axis=0), values.max(axis=0)
                    old = out.get(key)
                    out[key] = (np.minimum(old[0], lo), np.maximum(old[1], hi)) if old else (lo, hi)
    return out


def compare(reference, result):
    """VTK's output against Tack's: counts, bounds, measures and each common array's
    range, with a verdict: ``same`` (all agree), ``same geometry`` (measures and bounds
    agree, counts differ: points merged or cells split otherwise), or ``differ``."""
    points = [sum(x.GetNumberOfPoints() for x in _leaves(d)) for d in (reference, result)]
    cells = [sum(x.GetNumberOfCells() for x in _leaves(d)) for d in (reference, result)]
    measures = [_measure(d) for d in (reference, result)]
    bounds = [_bounds(d) for d in (reference, result)]
    size = 1.0
    if bounds[0]:
        b = np.array(bounds[0])
        size = max(float(np.linalg.norm(b[1::2] - b[::2])), 1e-12)
    notes = []
    bounds_ok = (bounds[0] is None) == (bounds[1] is None) and (
        bounds[0] is None or np.allclose(bounds[0], bounds[1], atol=1e-4 * size))
    if not bounds_ok:
        notes.append(f"bounds {bounds[0]} vs {bounds[1]}")
    measure_ok = True
    for name in ("Length", "Area", "Volume"):
        a, b = measures[0][name], measures[1][name]
        if abs(a - b) > 1e-4 * max(abs(a), abs(b), 1e-30) and max(abs(a), abs(b)) > 1e-12:
            measure_ok = False
            notes.append(f"{name} {a:.6g} vs {b:.6g}")
    arrays = [_array_summary(d) for d in (reference, result)]
    arrays_ok = True
    for key in sorted(set(arrays[0]) & set(arrays[1])):
        (lo0, hi0), (lo1, hi1) = arrays[0][key], arrays[1][key]
        if lo0.shape != lo1.shape:
            arrays_ok = False
            notes.append(f"{key[0]} ({key[1]}): {lo0.size} vs {lo1.size} components")
            continue
        scale = max(float(np.abs(hi0 - lo0).max()), 1e-6 * float(np.abs(
            np.concatenate([lo0, hi0])).max()), 1e-12)
        if not (np.allclose(lo0, lo1, atol=1e-3 * scale) and
                np.allclose(hi0, hi1, atol=1e-3 * scale)):
            arrays_ok = False
            notes.append(f"{key[0]} ({key[1]}) range {lo0.tolist()}..{hi0.tolist()} vs "
                         f"{lo1.tolist()}..{hi1.tolist()}")
    missing = sorted(f"{n} ({w})" for n, w in set(arrays[0]) - set(arrays[1]))
    if missing:
        notes.append("arrays only VTK gives: " + ", ".join(missing))
    counts_same = points[0] == points[1] and cells[0] == cells[1]
    if bounds_ok and measure_ok and arrays_ok:
        verdict = "same" if counts_same else "same geometry"
    else:
        verdict = "differ"
    return {"verdict": verdict, "points": points, "cells": cells, "measures": measures,
            "notes": notes}


# ── Installing ───────────────────────────────────────────────────────

def install():
    """Put the stand-ins in place of VTK's classes, in their modules and in
    ``vtkmodules.all`` and ``vtk``; returns the names replaced."""
    import importlib
    import sys

    replaced = []
    for name, (module_name, translate) in TRANSLATIONS.items():
        module = importlib.import_module(f"vtkmodules.{module_name}")
        real = REAL.get(name) or getattr(module, name, None)
        if real is None:
            continue
        REAL[name] = real
        shim = type(name, (_Shim,), {"REAL": real, "TRANSLATE": staticmethod(translate)})
        for attribute in dir(real):                  # VTK's constants (THRESHOLD_UPPER)
            if not attribute.startswith("_") and attribute not in _Shim.__dict__:
                value = getattr(real, attribute, None)
                if isinstance(value, (int, float, str)):
                    setattr(shim, attribute, value)
        setattr(module, name, shim)
        for other in ("vtkmodules.all", "vtk"):
            if other in sys.modules:
                setattr(sys.modules[other], name, shim)
        replaced.append(name)
    return replaced
