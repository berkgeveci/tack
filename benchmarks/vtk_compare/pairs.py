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
# On the generated grids p is exactly 0.5 at mesh points, where Tack makes one
# point of every crossing there and VTK and Viskores keep the degenerate
# cells (all validly), so the counts would differ; the contour is taken just
# off them -- and off every point's f32 value: Tack compares an f32 field with
# the level in f32, VTK in f64, and at 0.5123 one point of the 1M tetrahedra
# was the level in f32 and not in f64.
CONTOUR_ISOVALUE = 0.51234567
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

    f = vtkContour3DLinearGrid(input_data=grid, merge_points=True, interpolate_attributes=True,
                               compute_normals=False, compute_scalars=False,
                               use_scalar_tree=False,
                               # Tack's polyhedral contour gives polygons
                               generate_triangles=form == "shape")
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetValue(0, CONTOUR_ISOVALUE)
    return f


def _contour_filter(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkContourFilter

    f = vtkContourFilter(input_data=grid, compute_normals=False, compute_gradients=False,
                         compute_scalars=False, use_scalar_tree=False,
                         generate_triangles=form == "shape")
    f.SetInputArrayToProcess(0, 0, 0, 0, "p")
    f.SetValue(0, CONTOUR_ISOVALUE)
    return f


def _plane_cutter(grid, mesh, form):
    from vtkmodules.vtkCommonDataModel import vtkPlane
    from vtkmodules.vtkFiltersCore import vtkPlaneCutter

    plane = vtkPlane(origin=tuple(_center(mesh)), normal=tuple(NORMAL))
    return vtkPlaneCutter(input_data=grid, plane=plane, merge_points=True,
                          interpolate_attributes=True, compute_normals=False,
                          generate_polygons=form != "shape",
                          build_tree=False,        # a cached sphere tree would persist
                          build_hierarchy=False)


def _threshold(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkThreshold

    f = vtkThreshold(input_data=grid, lower_threshold=THRESHOLD[0],
                     upper_threshold=THRESHOLD[1],
                     threshold_function=vtkThreshold.THRESHOLD_BETWEEN)
    f.SetInputArrayToProcess(0, 0, 0, 1, "c")
    return f


def _geometry_filter(grid, mesh, form):
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    return vtkGeometryFilter(input_data=grid, merging=False, pass_through_cell_ids=False,
                             pass_through_point_ids=False)


def _cell_size(grid, mesh, form):
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    return vtkCellSizeFilter(input_data=grid, compute_vertex_count=False, compute_length=False,
                             compute_area=False, compute_volume=True, compute_sum=False)


def _point_to_cell(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkPointDataToCellData

    return vtkPointDataToCellData(input_data=grid,
                                  process_all_arrays=True,     # the input carries p alone
                                  pass_point_data=False)


def _cell_to_point(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkCellDataToPointData

    return vtkCellDataToPointData(input_data=grid,
                                  process_all_arrays=True,     # the input carries c alone
                                  pass_cell_data=False)


def _cell_derivatives(grid, mesh, form):
    from vtkmodules.vtkFiltersGeneral import vtkCellDerivatives

    f = vtkCellDerivatives(input_data=grid)
    f.SetVectorModeToComputeGradient()
    f.SetTensorModeToPassTensors()
    return f


def _cell_centers(grid, mesh, form):
    from vtkmodules.vtkFiltersCore import vtkCellCenters

    return vtkCellCenters(input_data=grid, vertex_cells=False, copy_arrays=False)


# ── Reading results ─────────────────────────────────────────────────

def _named(data, name):
    """``data``'s array ``name`` as NumPy. A missing array raises: VTK reads it
    as an empty array, which would be checked as values."""
    if name not in data.keys():
        raise KeyError(f"the output has no array {name!r}")
    return np.asarray(data[name])


def _cell_array(name):
    return lambda out: _named(out.cell_data, name)


def _point_array(name):
    return lambda out: _named(out.point_data, name)


def _cell_vectors(out):
    vectors = out.cell_data.vectors
    if vectors is None:
        raise KeyError("the output has no cell vectors")
    return np.asarray(vectors)


def _points_of(out):
    return np.asarray(out.points.data)


def _host(field_result, vectors=False):
    values = field_result.values.numpy()
    return values if vectors else values.reshape(-1)


def _tack():
    import tack.dataset as td
    from tack.dataset import algorithms as alg

    return td, alg


PAIRS = [
    Pair("contour",
         vtk=_contour_3d_linear_grid, others={"vtkContourFilter": _contour_filter},
         tack=lambda d, m: _tack()[0].contour(d, "p", CONTOUR_ISOVALUE),
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
         tack=lambda d, m: _tack()[0].extract_surface(d),
         point_fields=("p",), output="surface"),
    Pair("cell volume",
         vtk=_cell_size, vtk_values=_cell_array("Volume"),
         tack=lambda d, m: _tack()[1].cell_geometry(d)[0], tack_values=_host),
    Pair("point to cell",
         vtk=_point_to_cell, vtk_values=_cell_array("p"),
         tack=lambda d, m: _tack()[1].project(d, d.fields["p"], _tack()[0].Constant(d)), tack_values=_host,
         point_fields=("p",)),
    Pair("cell to point",
         vtk=_cell_to_point, vtk_values=_point_array("c"),
         tack=lambda d, m: _tack()[1].project(d, d.fields["c"], _tack()[0].H1(d)), tack_values=_host,
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

    cell_data = vtkIntegrateAttributes()(dataset).cell_data
    return float(cell_data[what][0]) if what in cell_data.keys() else 0.0


def summary(pair, vtk_output=None, tack_result=None):
    """What the check compares, from one side's output."""
    if pair.output == "array":
        return None
    if tack_result is not None:
        from tack.interop.vtk import dataset_to_vtk

        vtk_output = dataset_to_vtk(tack_result)
    what = "Area" if pair.output == "surface" else "Volume"
    return {"cells": vtk_output.number_of_cells, "points": vtk_output.number_of_points,
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
