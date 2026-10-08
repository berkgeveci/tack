"""Tests for tack.data.slice_plane and tack.data.threshold.

Against NumPy on every backend: slice points lie on the plane, an
axis-aligned cut of a grid gives two triangles per cut cell, and the
rectilinear and explicit forms agree; threshold keeps exactly the cells
in range, in order and with their shapes, only the points they use, and
carries point and cell data. With VTK: vtkCutter and vtkThreshold, cell
for cell.
"""

import types

import numpy as np
import pytest
from test_contour import _grid_points, _hex_rows, _oriented, _scalars, _surface
from test_filters import _mixed, _wedge_columns

import tack
import tack.data as td
from tack.data import shapes as sh
from tack.runtime.dispatch import env_flag

try:
    from vtkmodules import vtkCommonDataModel, vtkFiltersCore
    from vtkmodules.util.numpy_support import vtk_to_numpy
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(vtkPlane=vtkCommonDataModel.vtkPlane,
                                vtkCutter=vtkFiltersCore.vtkCutter,
                                vtkThreshold=vtkFiltersCore.vtkThreshold)

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")


def _cells(data):
    """Each cell as (type, its points' coordinates in order), in cell order."""
    points = data.points.to_numpy(vectors=True)
    offsets = data.cells.offsets.to_numpy()
    connectivity = data.cells.connectivity.to_numpy()
    return [(int(t), tuple(tuple(np.round(points[p], 6))
                           for p in connectivity[offsets[i]:offsets[i + 1]]))
            for i, t in enumerate(data.cells.types.to_numpy())]


# ── Slice ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("form", ["structured", "rectilinear", "explicit"])
def test_an_axis_aligned_slice_of_a_grid(backend, form):
    n = 7
    points = _grid_points(n)
    if form == "structured":
        data = td.DataSet(points, td.StructuredCellSet((n, n, n)))
    elif form == "rectilinear":
        axis = np.linspace(-1, 1, n)
        data = td.rectilinear_grid(axis, axis, axis)
    else:
        data = td.DataSet(points, td.SingleTypeCellSet(sh.Hexahedron, _hex_rows(n)))
    data.point_data["y"] = _scalars(points[:, 1])
    cut = td.slice_plane(data, (0.1, 0.0, 0.0), (2.0, 0.0, 0.0))
    pts, triangles = _surface(cut)
    assert len(triangles) == 2 * (n - 1) ** 2
    np.testing.assert_allclose(pts[:, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(cut.point_data["y"].to_numpy(), pts[:, 1], atol=1e-6)


def test_an_oblique_slice_lies_on_its_plane(backend):
    data = _wedge_columns(6)
    data = td.DataSet(data.points.to_numpy(vectors=True), data.cells)
    origin, normal = np.array([2.3, 2.1, 1.7]), np.array([1.0, 0.4, 0.7])
    pts, triangles = _surface(td.slice_plane(data, origin, normal))
    assert len(triangles) > 50
    np.testing.assert_allclose((pts - origin) @ normal, 0.0, atol=1e-5)


def test_a_plane_missing_the_data_cuts_nothing(backend):
    n = 4
    data = td.DataSet(_grid_points(n), td.StructuredCellSet((n, n, n)))
    assert td.slice_plane(data, (5.0, 0.0, 0.0), (1.0, 0.0, 0.0)).num_cells == 0


# ── Threshold ───────────────────────────────────────────────────────

def _mixed_data(seed=1, num_cells=200, num_points=150):
    rng = np.random.default_rng(seed)
    types_, offsets, connectivity, rows = _mixed(rng, num_cells, num_points)
    points = rng.uniform(-1, 1, (num_points, 3))
    data = td.DataSet(points, td.ExplicitCellSet(types_, offsets, connectivity),
                      point_data={"p": _scalars(rng.uniform(size=num_points)),
                                  "xyz": tack.Vector.field(3, tack.f32, shape=(num_points,))},
                      cell_data={"c": _scalars(rng.uniform(size=num_cells)),
                                 "id": _scalars(np.arange(num_cells))})
    data.point_data["xyz"].from_numpy(points.astype(np.float32))
    return data, types_, rows, points


def _expected(types_, rows, points, keep):
    return [(int(types_[c]), tuple(tuple(np.round(points[p].astype(np.float32), 6))
                                   for p in rows[c])) for c in np.flatnonzero(keep)]


def test_threshold_by_cell_data(backend):
    data, types_, rows, points = _mixed_data()
    values = data.cell_data["c"].to_numpy()
    out = td.threshold(data, "c", 0.25, 0.6)
    keep = (values >= 0.25) & (values <= 0.6)
    assert out.num_cells == keep.sum()
    assert _cells(out) == _expected(types_, rows, points, keep)
    np.testing.assert_array_equal(out.cell_data["id"].to_numpy(), np.flatnonzero(keep))
    used = sorted({p for c in np.flatnonzero(keep) for p in rows[c]})
    assert out.num_points == len(used)
    np.testing.assert_allclose(out.points.to_numpy(vectors=True), points[used], atol=1e-6)
    np.testing.assert_allclose(out.point_data["xyz"].to_numpy(vectors=True), points[used],
                               atol=1e-6)


@pytest.mark.parametrize("all_points", [True, False])
def test_threshold_by_point_data(backend, all_points):
    data, types_, rows, points = _mixed_data(seed=2)
    values = data.point_data["p"].to_numpy()
    inside = (values >= 0.2) & (values <= 0.8)
    keep = np.array([(inside[r].all() if all_points else inside[r].any()) for r in rows])
    out = td.threshold(data, "p", 0.2, 0.8, all_points=all_points)
    assert _cells(out) == _expected(types_, rows, points, keep)


def test_threshold_of_everything_nothing_and_raw_fields(backend):
    data, types_, rows, points = _mixed_data(seed=3)
    assert td.threshold(data, "c", -1.0, 2.0).num_cells == data.num_cells
    empty = td.threshold(data, "c", 5.0, 6.0)
    assert empty.num_cells == empty.num_points == 0
    assert empty.cells.offsets.to_numpy().tolist() == [0]
    by_field = td.threshold(data, data.cell_data["c"], 0.3, 0.7)
    assert _cells(by_field) == _cells(td.threshold(data, "c", 0.3, 0.7))
    with pytest.raises(TypeError, match="not a vector field"):
        td.threshold(data, "xyz", 0.0, 1.0)
    with pytest.raises(ValueError, match="entries; the dataset has"):
        td.threshold(data, tack.zeros(tack.f32, (7,)), 0.0, 1.0)


def test_threshold_of_a_rectilinear_grid(backend):
    axis = np.linspace(-1, 1, 6)
    grid = td.rectilinear_grid(axis, axis, axis)
    explicit = td.DataSet(grid.points.to_numpy(), td.StructuredCellSet((6, 6, 6)))
    for data in (grid, explicit):
        data.cell_data["c"] = _scalars(np.arange(125) % 7)
    a, b = td.threshold(grid, "c", 2, 4), td.threshold(explicit, "c", 2, 4)
    assert a.num_cells == b.num_cells > 0 and _cells(a) == _cells(b)


# ── VTK ─────────────────────────────────────────────────────────────

def _data_with_values():
    data = _wedge_columns(6)
    pts = data.points.to_numpy(vectors=True)
    rng = np.random.default_rng(0)
    data.cell_data["c"] = _scalars(rng.uniform(size=data.num_cells), tack.f64)
    data.point_data["p"] = _scalars(pts[:, 0] + pts[:, 2], tack.f64)
    return data


@needs_vtk
def test_slice_is_vtks(f64_backend):
    from tack.interop.vtk import dataset_to_vtk
    data = _data_with_values()
    origin, normal = [2.3, 2.1, 1.7], [1.0, 0.4, 0.7]
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(dataset_to_vtk(data))
    cutter.Update()
    out = cutter.GetOutput()
    polys = out.GetPolys()
    offsets = vtk_to_numpy(polys.GetOffsetsArray())
    connectivity = vtk_to_numpy(polys.GetConnectivityArray())
    theirs = _oriented(vtk_to_numpy(out.GetPoints().GetData()),
                       np.array([connectivity[offsets[i]:offsets[i + 1]]
                                 for i in range(len(offsets) - 1)]))
    ours = _oriented(*_surface(td.slice_plane(data, origin, normal)))
    assert len(ours) == len(theirs) and ours == theirs


@needs_vtk
@pytest.mark.parametrize("name, lower, upper", [("c", 0.2, 0.6), ("p", 3.0, 6.5)])
def test_threshold_is_vtks(f64_backend, name, lower, upper):
    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
    data = _data_with_values()
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(dataset_to_vtk(data))
    threshold.SetInputArrayToProcess(0, 0, 0, 1 if name == "c" else 0, name)
    threshold.SetLowerThreshold(lower)
    threshold.SetUpperThreshold(upper)
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    threshold.Update()
    theirs = vtk_to_dataset(threshold.GetOutput(), dtype=tack.f64)
    ours = td.threshold(data, name, lower, upper)
    assert (ours.num_cells, ours.num_points) == (theirs.num_cells, theirs.num_points)
    assert _cells(ours) == _cells(theirs)
