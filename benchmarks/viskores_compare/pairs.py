"""The filter pairs: a Viskores filter, reached through VTK, and the Tack call that
does the same work.

Each pair names the fields its input carries, builds a ``vtkm...`` filter with its
input set (timed as ``Modified(); Update()``, which converts VTK's arrays to
Viskores and the result back on every call, as any VTK user reaching Viskores
pays), and calls Tack. Options are set so the outputs match.

``check`` compares Tack's output with Viskores' before any timing is believed:
counts and integrated area or volume for filters that make a dataset, values for
those that make an array or move the points. Where VTK has a CPU filter for the
same work, Viskores is also checked against it, for information.

Viskores supports no polyhedra, so every pair runs on the shape-based meshes.
"""

# The constants and the slice plane are the VTK comparison's, so the two
# benchmarks measure the same work.
import importlib.util
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "vtk_compare_pairs", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                      "vtk_compare", "pairs.py"))
_vtk_pairs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vtk_pairs)
ISOVALUE, NORMAL, THRESHOLD = _vtk_pairs.ISOVALUE, _vtk_pairs.NORMAL, _vtk_pairs.THRESHOLD
_center = _vtk_pairs._center

# On the generated grids p is exactly 0.5 at mesh points, where VTK, Tack and
# Viskores resolve the degenerate crossings differently (all validly); the
# contour is taken just off them.
CONTOUR_ISOVALUE = 0.5123
ELEVATION_RANGE = (0.0, 1.0)
WARP_SCALE = 0.1
SPHERE_FRACTION = 0.3517          # the clip sphere's radius, of the box's extent: off the grid points


@dataclass
class Pair:
    name: str
    viskores: Callable                # (grid, mesh) -> configured vtkm... algorithm
    tack: Callable                    # (data, mesh) -> result
    point_fields: tuple = ()
    cell_fields: tuple = ()
    output: str = "array"             # "array", "points", "surface" or "volume"
    viskores_values: Callable = None  # Viskores output -> host array
    tack_values: Callable = None      # Tack result -> host array
    vtk: Callable = None              # (grid, mesh) -> the CPU VTK filter, for reference


def _box(mesh):
    """The mesh's bounds, computed once: the Tack calls read them inside the timing."""
    if getattr(mesh, "box", None) is None:
        mesh.box = mesh.points.min(axis=0), mesh.points.max(axis=0)
    return mesh.box


def _sphere(mesh):
    lo, hi = _box(mesh)
    return 0.5 * (lo + hi), SPHERE_FRACTION * float(np.max(hi - lo))


def _elevation_line(mesh):
    lo, hi = _box(mesh)
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def _matrix():
    """A rotation about z, a scale and a translation, as a 4x4 row-major matrix."""
    c, s = np.cos(0.3), np.sin(0.3)
    return np.array([[1.5 * c, -1.5 * s, 0.0, 0.25],
                     [1.5 * s, 1.5 * c, 0.0, -0.5],
                     [0.0, 0.0, 0.75, 1.0],
                     [0.0, 0.0, 0.0, 1.0]])


# ── Viskores filters (vtkAcceleratorsVTKmFilters) ────────────────────

def _vf():
    import vtkmodules.vtkAcceleratorsVTKmFilters as vf

    return vf


def _contour(grid, mesh):
    f = _vf().vtkmContour()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetValue(0, CONTOUR_ISOVALUE)
    f.SetComputeNormals(False)
    f.SetComputeGradients(False)
    f.SetComputeScalars(False)
    return f


def _plane(mesh):
    from vtkmodules.vtkCommonDataModel import vtkPlane

    plane = vtkPlane()
    plane.SetOrigin(*_center(mesh))
    plane.SetNormal(*NORMAL)
    return plane


def _slice(grid, mesh):
    f = _vf().vtkmSlice()
    f.SetInputData(grid)
    f.SetCutFunction(_plane(mesh))
    return f


def _threshold(grid, mesh):
    from vtkmodules.vtkFiltersCore import vtkThreshold

    f = _vf().vtkmThreshold()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 1, "c")
    f.SetLowerThreshold(THRESHOLD[0])
    f.SetUpperThreshold(THRESHOLD[1])
    f.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
    return f


def _clip_field(grid, mesh):
    f = _vf().vtkmClip()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetValue(ISOVALUE)
    f.SetInsideOut(False)                         # keep where p >= value, as Tack's clip
    f.SetComputeScalars(False)
    return f


def _clip_sphere(grid, mesh):
    from vtkmodules.vtkCommonDataModel import vtkSphere

    center, radius = _sphere(mesh)
    sphere = vtkSphere()
    sphere.SetCenter(*center)
    sphere.SetRadius(radius)
    f = _vf().vtkmClip()
    f.SetInputData(grid)
    f.SetClipFunction(sphere)
    f.SetValue(0.0)
    f.SetInsideOut(False)                         # keep outside the sphere, as Tack's clip
    f.SetComputeScalars(False)
    return f


def _external_faces(grid, mesh):
    f = _vf().vtkmExternalFaces()
    f.SetInputData(grid)
    f.SetCompactPoints(False)                     # Tack and VTK keep every input point
    return f


def _average_to_cells(grid, mesh):
    f = _vf().vtkmAverageToCells()
    f.SetInputData(grid)
    f.SetProcessAllArrays(True)                   # the input carries p alone
    f.SetPassPointData(False)
    return f


def _average_to_points(grid, mesh):
    f = _vf().vtkmAverageToPoints()
    f.SetInputData(grid)
    f.SetProcessAllArrays(True)                   # the input carries c alone
    f.SetPassCellData(False)
    return f


def _point_gradient(grid, mesh):
    f = _vf().vtkmGradient()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetResultArrayName("Gradients")
    f.SetComputeGradient(True)
    f.SetComputeDivergence(False)
    f.SetComputeVorticity(False)
    f.SetComputeQCriterion(False)
    return f


def _elevation(grid, mesh):
    low, high = _elevation_line(mesh)
    f = _vf().vtkmPointElevation()
    f.SetInputData(grid)
    f.SetLowPoint(*low)
    f.SetHighPoint(*high)
    f.SetScalarRange(*ELEVATION_RANGE)
    return f


def _warp(grid, mesh):
    f = _vf().vtkmWarpScalar()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetUseNormal(True)                          # one direction for every point
    f.SetNormal(0.0, 0.0, 1.0)
    f.SetScaleFactor(WARP_SCALE)
    return f


def _coordinates(kind):
    def build(grid, mesh):
        f = _vf().vtkmCoordinateSystemTransform()
        f.SetInputData(grid)
        getattr(f, f"SetCartesianTo{kind}")()
        return f
    return build


def _point_transform(grid, mesh):
    from vtkmodules.vtkCommonTransforms import vtkTransform

    transform = vtkTransform()
    transform.SetMatrix(_matrix().ravel())
    f = _vf().vtkmPointTransform()
    f.SetInputData(grid)
    f.SetTransform(transform)
    return f


# ── CPU VTK filters, to check Viskores against ───────────────────────

def _vtk_contour(grid, mesh):
    from vtkmodules.vtkFiltersCore import vtkContour3DLinearGrid

    f = vtkContour3DLinearGrid()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetValue(0, CONTOUR_ISOVALUE)
    f.SetMergePoints(True)
    f.SetInterpolateAttributes(True)
    f.SetComputeNormals(False)
    f.SetComputeScalars(False)
    f.SetUseScalarTree(False)
    f.SetGenerateTriangles(True)
    return f


def _vtk_slice(grid, mesh):
    from vtkmodules.vtkFiltersCore import vtkPlaneCutter

    f = vtkPlaneCutter()
    f.SetInputData(grid)
    f.SetPlane(_plane(mesh))
    f.SetMergePoints(True)
    f.SetInterpolateAttributes(True)
    f.SetComputeNormals(False)
    f.SetGeneratePolygons(False)
    f.SetBuildTree(False)
    f.SetBuildHierarchy(False)
    return f


def _vtk_threshold(grid, mesh):
    from vtkmodules.vtkFiltersCore import vtkThreshold

    f = vtkThreshold()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 1, "c")
    f.SetLowerThreshold(THRESHOLD[0])
    f.SetUpperThreshold(THRESHOLD[1])
    f.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
    return f


def _vtk_clip_field(grid, mesh):
    from vtkmodules.vtkFiltersGeneral import vtkTableBasedClipDataSet

    f = vtkTableBasedClipDataSet()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetValue(ISOVALUE)
    f.SetInsideOut(False)
    return f


def _vtk_geometry(grid, mesh):
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    f = vtkGeometryFilter()
    f.SetInputData(grid)
    f.SetMerging(False)
    f.SetPassThroughCellIds(False)
    f.SetPassThroughPointIds(False)
    return f


def _vtk_point_to_cell(grid, mesh):
    from vtkmodules.vtkFiltersCore import vtkPointDataToCellData

    f = vtkPointDataToCellData()
    f.SetInputData(grid)
    f.SetProcessAllArrays(True)
    f.SetPassPointData(False)
    return f


def _vtk_cell_to_point(grid, mesh):
    from vtkmodules.vtkFiltersCore import vtkCellDataToPointData

    f = vtkCellDataToPointData()
    f.SetInputData(grid)
    f.SetProcessAllArrays(True)
    f.SetPassCellData(False)
    return f


def _vtk_gradient(grid, mesh):
    from vtkmodules.vtkFiltersGeneral import vtkGradientFilter

    f = vtkGradientFilter()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetResultArrayName("Gradients")
    return f


def _vtk_elevation(grid, mesh):
    """VTK names its array "Elevation"; renamed to Viskores' "elevation"."""
    from vtkmodules.vtkFiltersCore import vtkElevationFilter

    low, high = _elevation_line(mesh)
    f = vtkElevationFilter()
    f.SetInputData(grid)
    f.SetLowPoint(*low)
    f.SetHighPoint(*high)
    f.SetScalarRange(*ELEVATION_RANGE)
    return _Renamed(f, "Elevation", "elevation")


class _Renamed:
    """A VTK filter whose output point array is read under another name."""

    def __init__(self, algorithm, name, as_name):
        self.algorithm, self.name, self.as_name = algorithm, name, as_name

    def Update(self):
        self.algorithm.Update()

    def GetOutput(self):
        out = self.algorithm.GetOutput()
        array = out.GetPointData().GetArray(self.name)
        if array is not None:
            array.SetName(self.as_name)
        return out


def _vtk_warp(grid, mesh):
    from vtkmodules.vtkFiltersGeneral import vtkWarpScalar

    f = vtkWarpScalar()
    f.SetInputData(grid)
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetUseNormal(True)
    f.SetNormal(0.0, 0.0, 1.0)
    f.SetScaleFactor(WARP_SCALE)
    return f


def _vtk_point_transform(grid, mesh):
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersGeneral import vtkTransformFilter

    transform = vtkTransform()
    transform.SetMatrix(_matrix().ravel())
    f = vtkTransformFilter()
    f.SetInputData(grid)
    f.SetTransform(transform)
    return f


# ── Reading results ─────────────────────────────────────────────────

def _cell_array(name):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return lambda out: vtk_to_numpy(out.GetCellData().GetArray(name))


def _point_array(name):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return lambda out: vtk_to_numpy(out.GetPointData().GetArray(name))


def _points_of(out):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return vtk_to_numpy(out.GetPoints().GetData())


def _host(result, vectors=False):
    return result.values.to_numpy(vectors=True) if vectors else result.values.to_numpy()


def _geometry(result):
    return result.geometry.values.to_numpy(vectors=True)


def _tack():
    import tack.dataset as td
    from tack.dataset import algorithms as alg

    return td, alg


PAIRS = [
    Pair("contour", viskores=_contour, vtk=_vtk_contour,
         tack=lambda d, m: _tack()[0].contour(d, "p", CONTOUR_ISOVALUE),
         point_fields=("p",), output="surface"),
    Pair("slice", viskores=_slice, vtk=_vtk_slice,
         tack=lambda d, m: _tack()[0].slice_plane(d, tuple(_center(m)), tuple(NORMAL)),
         point_fields=("p",), output="surface"),
    Pair("threshold", viskores=_threshold, vtk=_vtk_threshold,
         tack=lambda d, m: _tack()[0].threshold(d, "c", *THRESHOLD),
         cell_fields=("c",), output="volume"),
    Pair("clip by field", viskores=_clip_field, vtk=_vtk_clip_field,
         tack=lambda d, m: _tack()[0].clip(d, "p", ISOVALUE),
         point_fields=("p",), output="volume"),
    Pair("clip by sphere", viskores=_clip_sphere,
         tack=lambda d, m: _tack()[0].clip(d, _tack()[0].implicit.Sphere(*_sphere(m))),
         point_fields=("p",), output="volume"),
    Pair("external faces", viskores=_external_faces, vtk=_vtk_geometry,
         tack=lambda d, m: _tack()[0].external_faces(d),
         point_fields=("p",), output="surface"),
    Pair("point to cell", viskores=_average_to_cells, vtk=_vtk_point_to_cell,
         viskores_values=_cell_array("p"),
         tack=lambda d, m: _tack()[1].to_cells(d, d.fields["p"]), tack_values=_host,
         point_fields=("p",)),
    Pair("cell to point", viskores=_average_to_points, vtk=_vtk_cell_to_point,
         viskores_values=_point_array("c"),
         tack=lambda d, m: _tack()[1].to_points(d, d.fields["c"]), tack_values=_host,
         cell_fields=("c",)),
    Pair("point gradient", viskores=_point_gradient, vtk=_vtk_gradient,
         viskores_values=_point_array("Gradients"),
         tack=lambda d, m: _tack()[1].gradients(d, d.fields["p"], at="points"),
         tack_values=lambda r: _host(r, vectors=True), point_fields=("p",)),
    Pair("point elevation", viskores=_elevation, vtk=_vtk_elevation,
         viskores_values=_point_array("elevation"),
         tack=lambda d, m: _tack()[0].point_elevation(
             d, *_elevation_line(m), ELEVATION_RANGE),
         tack_values=_host),
    Pair("warp", viskores=_warp, vtk=_vtk_warp, output="points",
         viskores_values=_points_of,
         tack=lambda d, m: _tack()[0].warp(d, (0.0, 0.0, 1.0), WARP_SCALE, "p"),
         tack_values=_geometry, point_fields=("p",)),
    Pair("cylindrical", viskores=_coordinates("Cylindrical"), output="points",
         viskores_values=_points_of,
         tack=lambda d, m: _tack()[0].cylindrical(d), tack_values=_geometry),
    Pair("spherical", viskores=_coordinates("Spherical"), output="points",
         viskores_values=_points_of,
         tack=lambda d, m: _tack()[0].spherical(d), tack_values=_geometry),
    Pair("point transform", viskores=_point_transform, vtk=_vtk_point_transform,
         output="points", viskores_values=_points_of,
         tack=lambda d, m: _tack()[0].transform(d, _matrix()),
         tack_values=_geometry),
]


# ── Checking ────────────────────────────────────────────────────────

def _measured(dataset, what):
    """The total area or volume of ``dataset``'s cells, each taken unsigned."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    size = vtkCellSizeFilter()
    size.SetInputData(dataset)
    for quantity in ("VertexCount", "Length", "Area", "Volume"):
        getattr(size, f"SetCompute{quantity}")(quantity == what)
    size.SetComputeSum(False)
    size.Update()
    array = size.GetOutput().GetCellData().GetArray(what)
    return float(np.abs(vtk_to_numpy(array)).sum()) if array is not None else 0.0


def summary(pair, vtk_output=None, tack_result=None):
    """What the check compares, from one side's output."""
    if tack_result is not None:
        from tack.interop.vtk import dataset_to_vtk

        vtk_output = dataset_to_vtk(tack_result)
    what = "Area" if pair.output == "surface" else "Volume"
    return {"cells": vtk_output.GetNumberOfCells(), "points": vtk_output.GetNumberOfPoints(),
            "distinct points": _distinct_points(vtk_output), what.lower(): _measured(vtk_output, what)}


def _distinct_points(dataset):
    """The number of point positions, merging points within a millionth of the
    bounds: Viskores leaves some coincident points of a tetrahedral contour or
    slice unmerged."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from vtkmodules.util.numpy_support import vtk_to_numpy

    n = dataset.GetNumberOfPoints()
    if n == 0:
        return 0
    points = vtk_to_numpy(dataset.GetPoints().GetData()).astype(np.float64)
    extent = max(float(np.max(points.max(axis=0) - points.min(axis=0))), 1e-30)
    near = cKDTree(points).query_pairs(1e-6 * extent, output_type="ndarray")
    if len(near) == 0:
        return n
    graph = coo_matrix((np.ones(len(near)), (near[:, 0], near[:, 1])), shape=(n, n))
    return int(connected_components(graph, directed=False)[0])


def _compare_arrays(ours, theirs):
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    if ours.shape != theirs.shape:
        return False, {"shape": [list(ours.shape), list(theirs.shape)]}
    scale = max(np.abs(theirs).max(), 1e-30) if theirs.size else 1.0
    error = float(np.abs(ours - theirs).max() / scale) if ours.size else 0.0
    return error < 1e-4, {"relative error": error}


def _compare_datasets(pair, ours, theirs):
    measure = "area" if pair.output == "surface" else "volume"
    error = abs(ours[measure] - theirs[measure]) / max(abs(theirs[measure]), 1e-30)
    ok = ours["cells"] == theirs["cells"] and \
        ours["distinct points"] == theirs["distinct points"] and error < 1e-4
    return ok, {"ours": ours, "theirs": theirs, f"{measure} relative error": error}


def check(pair, viskores_output, tack_result):
    """``(ok, details)``: whether Tack's output agrees with Viskores'."""
    if pair.output in ("array", "points"):
        return _compare_arrays(pair.tack_values(tack_result),
                               pair.viskores_values(viskores_output))
    return _compare_datasets(pair, summary(pair, tack_result=tack_result),
                             summary(pair, vtk_output=viskores_output))


def check_reference(pair, viskores_output, vtk_output):
    """``(ok, details)``: whether Viskores' output agrees with VTK's CPU filter."""
    if pair.output in ("array", "points"):
        values = pair.viskores_values
        return _compare_arrays(values(viskores_output), values(vtk_output))
    return _compare_datasets(pair, summary(pair, vtk_output=viskores_output),
                             summary(pair, vtk_output=vtk_output))

