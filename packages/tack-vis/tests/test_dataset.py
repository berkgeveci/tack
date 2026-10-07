"""Tests for tack.data's cell sets, DataSet and for_each_shape.

for_each_shape runs a kernel once per shape present, handing it a cell
view that is both the shape and that shape's cells. The checks below
record, per cell, what the kernel saw -- its shape, its point ids in
order, its id in the whole set, how often it was visited -- and compare
with the input arrays, so grouping, ordering and scattering back are all
covered without a reference library. Where VTK is installed, cell
centers are compared with vtkCellCenters, structured connectivity with
vtkStructuredGrid's, and datasets round-trip through VTK.
"""

import types

import numpy as np
import pytest
from test_shapes import EXPECTED

import tack
import tack.data as td
from tack.data import shapes as sh
from tack.runtime.dispatch import env_flag

try:
    from vtkmodules import vtkCommonCore, vtkCommonDataModel
    from vtkmodules.vtkFiltersCore import vtkCellCenters
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(**{name: getattr(module, name)
                                   for module in (vtkCommonCore, vtkCommonDataModel)
                                   for name in dir(module) if name.startswith(("vtk", "VTK"))})
    vtk.vtkCellCenters = vtkCellCenters

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

SHAPES = list(sh.SHAPES)


def _mixed_cells(rng, num_cells, num_points, shapes=SHAPES):
    """VTK-style arrays for ``num_cells`` cells of random shapes over random points."""
    kinds = rng.choice(shapes, num_cells)
    sizes = np.array([cls.NUM_POINTS for cls in kinds])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    connectivity = rng.integers(0, num_points, offsets[-1])
    return np.array([cls.ID for cls in kinds], np.uint8), offsets, connectivity


# ── What the kernel sees ────────────────────────────────────────────

@tack.kernel
def _record(cells, ids, signatures, visits):
    for c in range(cells.num_cells):
        cell = cells.cell_id(c)
        ids[cell] = cells.ID
        signature = 0
        for j in range(cells.NUM_POINTS):
            signature += (j + 1) * (cells.point_id(c, j) + 1)
        signatures[cell] = signature
        tack.atomic_add(visits, cell, 1)


def _recorded(data, num_cells):
    ids = tack.zeros(tack.i32, (num_cells,))
    signatures = tack.zeros(tack.i32, (num_cells,))
    visits = tack.zeros(tack.i32, (num_cells,))
    td.for_each_shape(_record, data, ids, signatures, visits)
    return ids.to_numpy(), signatures.to_numpy(), visits.to_numpy()


def _signatures(rows):
    """Each row's point ids weighted by position, as _record computes them."""
    return [sum((j + 1) * (p + 1) for j, p in enumerate(row)) for row in rows]


def test_explicit_cells_reach_the_kernel_by_shape_in_order(backend):
    rng = np.random.default_rng(1)
    types_, offsets, connectivity = _mixed_cells(rng, 500, 300)
    cells = td.ExplicitCellSet(types_, offsets, connectivity)
    ids, signatures, visits = _recorded(cells, 500)
    assert visits.tolist() == [1] * 500
    assert ids.tolist() == types_.tolist()
    rows = [connectivity[offsets[c]:offsets[c + 1]] for c in range(500)]
    assert signatures.tolist() == _signatures(rows)
    assert cells.shapes() == sorted({sh.shape_class(t) for t in types_}, key=lambda s: s.ID)


def test_single_type_cells_reach_the_kernel(backend):
    rng = np.random.default_rng(2)
    connectivity = rng.integers(0, 100, (64, 6))
    cells = td.SingleTypeCellSet(sh.Wedge, connectivity)
    ids, signatures, visits = _recorded(cells, 64)
    assert visits.tolist() == [1] * 64
    assert ids.tolist() == [sh.WEDGE] * 64
    assert signatures.tolist() == _signatures(connectivity)


def _structured_rows(point_dims):
    """vtkStructuredGrid's point ids for each cell, x fastest, in shape order."""
    dims = point_dims + (1,) * (3 - len(point_dims))
    shape = {1: sh.Line, 2: sh.Quad, 3: sh.Hexahedron}[len(point_dims)]
    corners = np.array(EXPECTED[shape]["points"])
    rows = []
    for k in range(max(dims[2] - 1, 1)):
        for j in range(max(dims[1] - 1, 1)):
            for i in range(dims[0] - 1):
                rows.append([(i + a) + (j + b) * dims[0] + (k + c) * dims[0] * dims[1]
                             for a, b, c in corners])
    return shape, rows


@pytest.mark.parametrize("point_dims", [(7,), (5, 4), (4, 3, 5)], ids=["1d", "2d", "3d"])
def test_structured_cells_reach_the_kernel(backend, point_dims):
    shape, rows = _structured_rows(point_dims)
    cells = td.StructuredCellSet(point_dims)
    assert cells.shapes() == [shape] and cells.num_cells == len(rows)
    ids, signatures, visits = _recorded(cells, len(rows))
    assert visits.tolist() == [1] * len(rows)
    assert ids.tolist() == [shape.ID] * len(rows)
    assert signatures.tolist() == _signatures(rows)


@needs_vtk
@pytest.mark.parametrize("point_dims", [(7,), (5, 4), (4, 3, 5)], ids=["1d", "2d", "3d"])
def test_structured_point_order_is_vtks(point_dims):
    shape, rows = _structured_rows(point_dims)
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(*(point_dims + (1,) * (3 - len(point_dims))))
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(int(np.prod(point_dims)))
    grid.SetPoints(points)
    for c, row in enumerate(rows):
        cell = grid.GetCell(c)
        assert cell.GetCellType() == shape.ID
        assert [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())] == row


# ── Points ──────────────────────────────────────────────────────────

@tack.kernel
def _centroids(cells, by_point, by_gather):
    for c in range(cells.num_cells):
        total = tack.Vector([0.0, 0.0, 0.0])
        for j in range(cells.NUM_POINTS):
            total += cells.point(c, j)
        by_point[cells.cell_id(c)] = total / cells.NUM_POINTS
        pts = tack.local_array(tack.f32, 3 * cells.NUM_POINTS)
        cells.gather_points(c, pts)
        mean = tack.Vector([0.0, 0.0, 0.0])
        for j in range(cells.NUM_POINTS):
            mean += tack.Vector([pts[3 * j], pts[3 * j + 1], pts[3 * j + 2]])
        by_gather[cells.cell_id(c)] = mean / cells.NUM_POINTS


def test_a_dataset_gives_the_kernel_its_points(backend):
    rng = np.random.default_rng(3)
    points = rng.uniform(-1, 1, (300, 3)).astype(np.float32)
    types_, offsets, connectivity = _mixed_cells(rng, 400, 300)
    data = td.DataSet(points, td.ExplicitCellSet(types_, offsets, connectivity))
    assert (data.num_points, data.num_cells) == (300, 400)
    by_point = tack.Vector.field(3, tack.f32, shape=(400,))
    by_gather = tack.Vector.field(3, tack.f32, shape=(400,))
    td.for_each_shape(_centroids, data, by_point, by_gather)
    want = [points[connectivity[offsets[c]:offsets[c + 1]]].mean(axis=0) for c in range(400)]
    np.testing.assert_allclose(by_point.to_numpy(vectors=True), want, atol=1e-6)
    np.testing.assert_allclose(by_gather.to_numpy(vectors=True), want, atol=1e-6)


# ── Grouping, caching, validation ───────────────────────────────────

def test_groups_are_computed_once_and_views_reuse_their_classes(backend):
    rng = np.random.default_rng(4)
    a = td.ExplicitCellSet(*_mixed_cells(rng, 50, 40, shapes=[sh.Tetra, sh.Hexahedron]))
    assert a.groups() is a.groups()
    b = td.ExplicitCellSet(*_mixed_cells(rng, 70, 40, shapes=[sh.Tetra, sh.Hexahedron]))
    assert [type(v) for v in a.views()] == [type(v) for v in b.views()]


def test_an_empty_cell_set_runs_nothing(backend):
    cells = td.ExplicitCellSet(np.zeros(0, np.uint8), np.zeros(1, np.int32),
                               np.zeros(0, np.int32))
    visits = tack.zeros(tack.i32, (1,))
    td.for_each_shape(_record, cells, visits, visits, visits)
    assert cells.shapes() == [] and visits.to_numpy().tolist() == [0]


def test_cell_types_that_are_not_linear_shapes_are_rejected(backend):
    cells = td.ExplicitCellSet(np.array([12, 7, 25], np.uint8), np.array([0, 8, 12, 32]),
                               np.zeros(32, np.int32))
    with pytest.raises(ValueError, match=r"cell types \[7, 25\] are not linear shapes"):
        cells.groups()


def test_a_cell_with_the_wrong_point_count_is_rejected(backend):
    cells = td.ExplicitCellSet(np.array([10, 12], np.uint8), np.array([0, 4, 11]),
                               np.zeros(11, np.int32))
    with pytest.raises(ValueError, match="1 cells have a point count"):
        cells.groups()


def test_malformed_inputs_are_rejected(backend):
    with pytest.raises(ValueError, match="offsets must have num_cells"):
        td.ExplicitCellSet(np.array([10], np.uint8), np.array([0]), np.zeros(4, np.int32))
    with pytest.raises(ValueError, match=r"connectivity must be \(num_cells, 8\)"):
        td.SingleTypeCellSet(sh.Hexahedron, np.zeros((3, 4), np.int32))
    for dims in [(), (1, 4), (2, 3, 4, 5)]:
        with pytest.raises(ValueError, match="point_dims"):
            td.StructuredCellSet(dims)
    with pytest.raises(ValueError, match=r"points must be \(num_points, 3\)"):
        td.DataSet(np.zeros((4, 2)), td.StructuredCellSet((4,)))


# ── VTK ─────────────────────────────────────────────────────────────

@tack.kernel
def _cell_centers(cells, out):
    for c in range(cells.num_cells):
        pc = cells.parametric_center()
        x = tack.Vector([0.0, 0.0, 0.0])
        for j in range(cells.NUM_POINTS):
            x += cells.shape_function(j, pc) * cells.point(c, j)
        out[cells.cell_id(c)] = x


def _vtk_mixed_grid(rng, num_cells):
    """A vtkUnstructuredGrid of separate, well-formed cells of every shape, shuffled.

    Each cell is its reference cell scaled per axis and moved, so pixels and
    voxels stay axis-aligned, as VTK defines them.
    """
    grid, points = vtk.vtkUnstructuredGrid(), vtk.vtkPoints()
    temperature = vtk.vtkFloatArray()
    temperature.SetName("temperature")
    for cls in rng.choice(SHAPES, num_cells):
        ref = np.array(EXPECTED[cls]["points"], float)
        corner, scale = rng.uniform(-10, 10, 3), rng.uniform(0.5, 2.0, 3)
        ids = [points.InsertNextPoint(*(corner + p * scale)) for p in ref]
        grid.InsertNextCell(int(cls.ID), len(ids), ids)
        temperature.InsertNextValue(rng.uniform())
    grid.SetPoints(points)
    grid.GetCellData().AddArray(temperature)
    return grid


@needs_vtk
def test_cell_centers_are_vtks(backend):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from tack.interop.vtk import vtk_to_dataset

    grid = _vtk_mixed_grid(np.random.default_rng(5), 300)
    data = vtk_to_dataset(grid)
    out = tack.Vector.field(3, tack.f32, shape=(data.num_cells,))
    td.for_each_shape(_cell_centers, data, out)
    centers = vtk.vtkCellCenters()
    centers.SetInputData(grid)
    centers.Update()
    want = vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
    np.testing.assert_allclose(out.to_numpy(vectors=True), want, atol=1e-4)


@needs_vtk
def test_datasets_round_trip_through_vtk(backend):
    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

    grid = _vtk_mixed_grid(np.random.default_rng(6), 100)
    data = vtk_to_dataset(grid)
    assert set(data.cell_data) == {"temperature"}
    back = dataset_to_vtk(data)
    assert back.GetNumberOfCells() == grid.GetNumberOfCells()
    for c in range(grid.GetNumberOfCells()):
        assert back.GetCellType(c) == grid.GetCellType(c)
        a, b = grid.GetCell(c), back.GetCell(c)
        assert [a.GetPointId(k) for k in range(a.GetNumberOfPoints())] == \
               [b.GetPointId(k) for k in range(b.GetNumberOfPoints())]
    for i in (0, grid.GetNumberOfPoints() - 1):
        np.testing.assert_allclose(back.GetPoint(i), grid.GetPoint(i), rtol=1e-6)
    np.testing.assert_allclose(
        [back.GetCellData().GetArray("temperature").GetValue(c) for c in range(100)],
        [grid.GetCellData().GetArray("temperature").GetValue(c) for c in range(100)])

    single = td.DataSet(np.zeros((8, 3)), td.SingleTypeCellSet(sh.Hexahedron, [list(range(8))]))
    assert dataset_to_vtk(single).GetCellType(0) == sh.HEXAHEDRON
    structured = td.DataSet(np.zeros((12, 3)), td.StructuredCellSet((4, 3)))
    round_trip = vtk_to_dataset(dataset_to_vtk(structured))
    assert round_trip.cells.point_dims == (4, 3) and round_trip.num_cells == 6
