"""The filter pairs: a VTK filter and the Tack call that does the same work.

Each pair names the fields its input carries -- on both sides, because
filters carry every field they are given -- builds a VTK filter with its
input set (timed as ``Modified(); Update()``), and calls Tack. Options are
set so the outputs match: points merged, attributes interpolated, no
normals, no cached search structures.

``check`` compares the two outputs before any timing is believed: counts
and integrated area or volume for filters that make a dataset, values for
those that make an array.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

ISOVALUE = 0.5                    # p spans about [-0.25, 1.25]
THRESHOLD = (0.25, 0.75)          # c is each cell center's x across the box
NORMAL = np.array([1.0, 0.3, 0.2]) / np.linalg.norm([1.0, 0.3, 0.2])


@dataclass
class Pair:
    name: str
    vtk: Callable                     # (grid, mesh, form) -> configured VTK algorithm
    tack: Callable                    # (data, mesh) -> result
    point_fields: tuple = ()
    cell_fields: tuple = ()
    forms: tuple = ("shape", "polyhedral")
    output: str = "array"             # "array", "surface" or "volume"
    vtk_values: Callable = None       # VTK output -> host array
    tack_values: Callable = None      # Tack result -> host array
    others: dict = field(default_factory=dict)    # other VTK implementations


def _center(mesh):
    """The slice plane's origin: the box center, nudged off it. On the generated
    grids the center is a mesh point, which the plane would pass through exactly;
    VTK and Tack resolve such a point differently (both validly), and the
    outputs then differ by a few degenerate triangles."""
    if getattr(mesh, "slice_origin", None) is None:     # once, outside any timing
        lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
        mesh.slice_origin = 0.5 * (lo + hi) + np.array([0.0123457, 0.0071, 0.0041]) * (hi - lo)
    return mesh.slice_origin


# ── VTK filters ─────────────────────────────────────────────────────

def _contour_3d_linear_grid(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkContour3DLinearGrid

    f = vtkContour3DLinearGrid()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetValue(0, ISOVALUE)
    f.SetMergePoints(True)
    f.SetInterpolateAttributes(True)
    f.SetComputeNormals(False)
    f.SetComputeScalars(False)
    f.SetUseScalarTree(False)
    f.SetGenerateTriangles(form == "shape")     # Tack's polyhedral contour gives polygons
    return f


def _contour_filter(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkContourFilter

    f = vtkContourFilter()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetValue(0, ISOVALUE)
    f.SetComputeNormals(False)
    f.SetComputeGradients(False)
    f.SetComputeScalars(False)
    f.SetUseScalarTree(False)
    f.SetGenerateTriangles(form == "shape")
    return f


def _plane_cutter(grid, mesh, form):
    from vtkmodules.vtkCommonDataModel import vtkPlane
    from vtkmodules.vtkFiltersCore import vtkPlaneCutter

    plane = vtkPlane()
    plane.SetOrigin(*_center(mesh))
    plane.SetNormal(*NORMAL)
    f = vtkPlaneCutter()
    f.SetInputData(grid)
    f.SetPlane(plane)
    f.SetMergePoints(True)
    f.SetInterpolateAttributes(True)
    f.SetComputeNormals(False)
    f.SetGeneratePolygons(form != "shape")
    f.SetBuildTree(False)                          # a cached sphere tree would persist
    f.SetBuildHierarchy(False)
    return f


def _threshold(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkThreshold

    f = vtkThreshold()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 1, "c")
    f.SetLowerThreshold(THRESHOLD[0])
    f.SetUpperThreshold(THRESHOLD[1])
    f.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
    return f


def _geometry_filter(grid, mesh, form):
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    f = vtkGeometryFilter()
    f.SetInputData(grid)
    f.SetMerging(False)
    f.SetPassThroughCellIds(False)
    f.SetPassThroughPointIds(False)
    return f


def _cell_size(grid, mesh, form):
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    f = vtkCellSizeFilter()
    f.SetInputData(grid)
    f.SetComputeVertexCount(False)
    f.SetComputeLength(False)
    f.SetComputeArea(False)
    f.SetComputeVolume(True)
    f.SetComputeSum(False)
    return f


def _point_to_cell(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkPointDataToCellData

    f = vtkPointDataToCellData()
    f.SetInputData(grid)
    f.SetProcessAllArrays(True)                    # the input carries p alone
    f.SetPassPointData(False)
    return f


def _cell_to_point(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkCellDataToPointData

    f = vtkCellDataToPointData()
    f.SetInputData(grid)
    f.SetProcessAllArrays(True)                    # the input carries c alone
    f.SetPassCellData(False)
    return f


def _cell_derivatives(grid, mesh, form):
    from vtkmodules.vtkFiltersGeneral import vtkCellDerivatives

    f = vtkCellDerivatives()
    f.SetInputData(grid)
    f.SetVectorModeToComputeGradient()
    f.SetTensorModeToPassTensors()
    return f


def _cell_centers(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkCellCenters

    f = vtkCellCenters()
    f.SetInputData(grid)
    f.SetVertexCells(False)
    f.SetCopyArrays(False)
    return f


# ── Reading results ─────────────────────────────────────────────────

def _cell_array(name):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return lambda out: vtk_to_numpy(out.GetCellData().GetArray(name))


def _point_array(name):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return lambda out: vtk_to_numpy(out.GetPointData().GetArray(name))


def _cell_vectors(out):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return vtk_to_numpy(out.GetCellData().GetVectors())


def _points_of(out):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return vtk_to_numpy(out.GetPoints().GetData())


def _host(field_result, vectors=False):
    return field_result.values.to_numpy(vectors=True) if vectors else \
        field_result.values.to_numpy()


def _tack():
    import tack.data as td
    from tack.data import algorithms as alg

    return td, alg


PAIRS = [
    Pair("contour",
         vtk=_contour_3d_linear_grid, others={"vtkContourFilter": _contour_filter},
         tack=lambda d, m: _tack()[0].contour(d, "p", ISOVALUE),
         point_fields=("p",), output="surface"),
    Pair("slice",
         vtk=_plane_cutter,
         tack=lambda d, m: _tack()[0].slice_plane(d, tuple(_center(m)), tuple(NORMAL)),
         point_fields=("p",), output="surface"),
    Pair("threshold",
         vtk=_threshold,
         tack=lambda d, m: _tack()[0].threshold(d, "c", *THRESHOLD),
         cell_fields=("c",), output="volume"),
    Pair("external faces",
         vtk=_geometry_filter,
         tack=lambda d, m: _tack()[0].external_faces(d),
         point_fields=("p",), output="surface"),
    Pair("cell volume",
         vtk=_cell_size, vtk_values=_cell_array("Volume"),
         tack=lambda d, m: _tack()[1].cell_geometry(d)[0], tack_values=_host),
    Pair("point to cell",
         vtk=_point_to_cell, vtk_values=_cell_array("p"),
         tack=lambda d, m: _tack()[1].to_cells(d, d.fields["p"]), tack_values=_host,
         point_fields=("p",)),
    Pair("cell to point",
         vtk=_cell_to_point, vtk_values=_point_array("c"),
         tack=lambda d, m: _tack()[1].to_points(d, d.fields["c"]), tack_values=_host,
         cell_fields=("c",)),
    Pair("cell gradient",
         vtk=_cell_derivatives, vtk_values=_cell_vectors,
         tack=lambda d, m: _tack()[1].gradients(d, d.fields["p"]),
         tack_values=lambda r: _host(r, vectors=True),
         point_fields=("p",), forms=("shape",)),
    Pair("cell centers",
         vtk=_cell_centers, vtk_values=_points_of,
         tack=lambda d, m: _tack()[1].cell_centers(d),
         tack_values=lambda r: _host(r, vectors=True), forms=("shape",)),
]


# ── Checking ────────────────────────────────────────────────────────

def _integrated(dataset, what):
    from vtkmodules.vtkFiltersParallel import vtkIntegrateAttributes

    integrate = vtkIntegrateAttributes()
    integrate.SetInputData(dataset)
    integrate.Update()
    array = integrate.GetOutput().GetCellData().GetArray(what)
    return array.GetValue(0) if array is not None else 0.0


def summary(pair, vtk_output=None, tack_result=None):
    """What the check compares, from one side's output."""
    if pair.output == "array":
        return None
    if tack_result is not None:
        from tack.interop.vtk import dataset_to_vtk

        vtk_output = dataset_to_vtk(tack_result)
    what = "Area" if pair.output == "surface" else "Volume"
    return {"cells": vtk_output.GetNumberOfCells(), "points": vtk_output.GetNumberOfPoints(),
            what.lower(): _integrated(vtk_output, what)}


def check(pair, vtk_output, tack_result):
    """``(ok, details)``: whether the two outputs agree."""
    if pair.output == "array":
        ours = np.asarray(pair.tack_values(tack_result), float)
        theirs = np.asarray(pair.vtk_values(vtk_output), float)
        if ours.shape != theirs.shape:
            return False, {"shape": [list(ours.shape), list(theirs.shape)]}
        scale = max(np.abs(theirs).max(), 1e-30) if theirs.size else 1.0
        error = float(np.abs(ours - theirs).max() / scale) if ours.size else 0.0
        return error < 1e-4, {"relative error": error}
    ours, theirs = summary(pair, tack_result=tack_result), summary(pair, vtk_output=vtk_output)
    measure = "area" if pair.output == "surface" else "volume"
    error = abs(ours[measure] - theirs[measure]) / max(abs(theirs[measure]), 1e-30)
    ok = ours["cells"] == theirs["cells"] and ours["points"] == theirs["points"] and \
        error < 1e-4
    return ok, {"tack": ours, "vtk": theirs, f"{measure} relative error": error}
