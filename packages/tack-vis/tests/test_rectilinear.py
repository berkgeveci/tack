"""Rectilinear grids: structured cells over three coordinate arrays.

A ``RectilinearCoordinates`` stores x, y and z; point (i, j, k) is at
(x[i], y[j], z[k]), numbered x fastest as VTK numbers it. Structured cells
find their points' (i, j, k) from their own indices, with no division;
other cells over the same coordinates split the flat point id. Every test
compares with the same grid given as one position per point, on every
backend, and with VTK where it is installed.
"""

import types

import numpy as np
import pytest

import tack
import tack.data as td
from tack.data import shapes as sh
from tack.runtime.dispatch import env_flag

try:
    from vtkmodules import vtkCommonCore, vtkCommonDataModel, vtkFiltersCore
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(**{name: getattr(module, name)
                                   for module in (vtkCommonCore, vtkCommonDataModel,
                                                  vtkFiltersCore)
                                   for name in dir(module) if name.startswith("vtk")})
    vtk.vtkGeometryFilter = vtkGeometryFilter

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

GRIDS = [(7,), (5, 4), (4, 3, 5), (2, 2, 2), (9, 1, 1)]


def _axes(dims, seed=0):
    rng = np.random.default_rng(seed)
    axes = [np.cumsum(rng.uniform(0.5, 2.0, d)) - 3.0 for d in dims]
    return axes + [np.zeros(1)] * (3 - len(axes))


def _numpy_points(x, y, z):
    return np.array([(x[i], y[j], z[k]) for k in range(len(z)) for j in range(len(y))
                     for i in range(len(x))])


def _grid(dims, dtype=tack.f32, seed=0):
    x, y, z = _axes(dims, seed)
    return td.rectilinear_grid(x, y, z, dtype=dtype), _numpy_points(x, y, z)


def _point_dims(dims):
    dims = list(dims)
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return tuple(dims)


# ── Coordinates ─────────────────────────────────────────────────────

def test_points_are_numbered_x_fastest(backend):
    x, y, z = [0.0, 1.0, 3.0], [10.0, 20.0], [100.0, 200.0, 400.0, 800.0]
    coordinates = td.RectilinearCoordinates(x, y, z)
    assert coordinates.dims == (3, 2, 4) and coordinates.num_points == 24
    np.testing.assert_array_equal(coordinates.to_numpy(), _numpy_points(x, y, z))
    assert td.RectilinearCoordinates([0, 1, 2], [0, 1]).point_dims == (3, 2)
    assert td.RectilinearCoordinates([0, 1, 2]).point_dims == (3,)


def test_coordinate_arrays_are_checked(backend):
    with pytest.raises(ValueError, match="at least one value"):
        td.RectilinearCoordinates([])
    with pytest.raises(TypeError, match="one-dimensional f32 field"):
        td.RectilinearCoordinates(tack.zeros(tack.i32, (3,)))
    with pytest.raises(ValueError, match=r"coordinates are a \(3, 2\) grid, the cells a \(3, 3\)"):
        td.DataSet(td.RectilinearCoordinates([0, 1, 2], [0, 1]), td.StructuredCellSet((3, 3)))


# ── Views ───────────────────────────────────────────────────────────

@tack.kernel
def _positions(cells, out):
    for c in cells:
        for j in range(cells.NUM_POINTS):
            out[cells.cell_id(c) * cells.NUM_POINTS + j] = cells.point(c, j)


@tack.kernel
def _gathered(cells, out):
    for c in cells:
        pts = tack.local_array(tack.f32, 3 * cells.NUM_POINTS)
        cells.gather_points(c, pts)
        for j in range(cells.NUM_POINTS):
            out[cells.cell_id(c) * cells.NUM_POINTS + j] = [pts[3 * j], pts[3 * j + 1],
                                                            pts[3 * j + 2]]


def _ids(data):
    """Every cell's point ids, in cell order."""
    @tack.kernel
    def ids(cells, out):
        for c in cells:
            for j in range(cells.NUM_POINTS):
                out[cells.cell_id(c) * cells.NUM_POINTS + j] = cells.point_id(c, j)
    shape = data.cells.shapes()[0]
    out = tack.field(tack.i32, shape=(data.num_cells * shape.NUM_POINTS,))
    td.for_each_shape(ids, data.cells, out)
    return out.to_numpy(), shape


@pytest.mark.parametrize("dims", GRIDS, ids=str)
def test_structured_cells_read_rectilinear_points(backend, dims):
    grid, points = _grid(dims)
    ids, shape = _ids(grid)
    for kernel in (_positions, _gathered):
        out = tack.Vector.field(3, tack.f32, shape=(len(ids),))
        td.for_each_shape(kernel, grid, out)
        np.testing.assert_allclose(out.to_numpy(vectors=True), points[ids], atol=1e-5)


def test_explicit_cells_read_rectilinear_points(backend):
    _, points = _grid((4, 3, 5))
    coordinates = td.RectilinearCoordinates(*_axes((4, 3, 5)))
    rng = np.random.default_rng(1)
    rows = rng.integers(0, len(points), (40, 4))
    data = td.DataSet(coordinates, td.SingleTypeCellSet(sh.Tetra, rows))
    out = tack.Vector.field(3, tack.f32, shape=(160,))
    td.for_each_shape(_positions, data, out)
    np.testing.assert_allclose(out.to_numpy(vectors=True), points[rows.reshape(-1)], atol=1e-5)


@pytest.mark.parametrize("dims", [(5, 4), (4, 3, 5)], ids=str)
def test_structured_rectilinear_points_need_no_division(dims):
    tack.init(arch=tack.cpu)
    grid, _ = _grid(dims)
    view = grid.cells.views(grid.points)[0]
    out = tack.Vector.field(3, tack.f32, shape=(256,))
    text = tack.inspect(_positions, view, out, mode="ir")
    assert "//" not in text and "%" not in text


# ── Filters ─────────────────────────────────────────────────────────

def _pair(dims, dtype=tack.f32):
    """The same grid with rectilinear coordinates and with one position per point."""
    grid, points = _grid(dims, dtype)
    explicit = td.DataSet(points, td.StructuredCellSet(_point_dims(dims)), dtype=dtype)
    rng = np.random.default_rng(2)
    for name, n in (("p", grid.num_points), ("c", grid.num_cells)):
        values = rng.uniform(size=n)
        for data in (grid, explicit):
            f = tack.field(dtype, shape=(n,))
            f.from_numpy(values.astype(dtype.numpy_dtype))
            (data.point_data if name == "p" else data.cell_data)[name] = f
    return grid, explicit


@pytest.mark.parametrize("dims", GRIDS, ids=str)
def test_filters_match_the_explicit_grid(backend, dims):
    grid, explicit = _pair(dims)
    a, b = td.cell_centers(grid), td.cell_centers(explicit)
    np.testing.assert_allclose(a.points.to_numpy(vectors=True), b.points.to_numpy(vectors=True),
                               atol=1e-5)
    np.testing.assert_allclose(td.point_data_to_cell_data(grid).cell_data["p"].to_numpy(),
                               td.point_data_to_cell_data(explicit).cell_data["p"].to_numpy(),
                               atol=1e-6)
    np.testing.assert_array_equal(td.cell_data_to_point_data(grid).point_data["c"].to_numpy(),
                                  td.cell_data_to_point_data(explicit).point_data["c"].to_numpy())
    fa, fb = td.external_faces(grid), td.external_faces(explicit)
    np.testing.assert_array_equal(fa.cells.connectivity.to_numpy(),
                                  fb.cells.connectivity.to_numpy())
    assert isinstance(fa.points, td.RectilinearCoordinates)
    if fa.num_cells:
        # The faces are explicit cells over the rectilinear points.
        np.testing.assert_allclose(td.cell_centers(fa).points.to_numpy(vectors=True),
                                   td.cell_centers(fb).points.to_numpy(vectors=True), atol=1e-5)


# ── VTK ─────────────────────────────────────────────────────────────

def _vtk_grid(dims, seed=3):
    x, y, z = _axes(dims, seed)
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(len(x), len(y), len(z))
    for setter, axis in ((grid.SetXCoordinates, x), (grid.SetYCoordinates, y),
                         (grid.SetZCoordinates, z)):
        setter(numpy_to_vtk(np.asarray(axis, float), deep=1))
    values = numpy_to_vtk(np.random.default_rng(seed).uniform(size=grid.GetNumberOfCells()),
                          deep=1)
    values.SetName("c")
    grid.GetCellData().AddArray(values)
    return grid


def _run(filter_, grid):
    filter_.SetInputData(grid)
    filter_.Update()
    return filter_.GetOutput()


@needs_vtk
@pytest.mark.parametrize("dims", [(5, 4), (4, 3, 5)], ids=str)
def test_round_trip_through_vtk(dims):
    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
    tack.init(arch=tack.cpu)
    grid = _vtk_grid(dims)
    data = vtk_to_dataset(grid, dtype=tack.f64)
    assert isinstance(data.points, td.RectilinearCoordinates)
    assert data.cells.point_dims == _point_dims(dims)
    np.testing.assert_array_equal(data.points.to_numpy(),
                                  [grid.GetPoint(i) for i in range(grid.GetNumberOfPoints())])
    back = dataset_to_vtk(data)
    assert back.GetClassName() == "vtkRectilinearGrid"
    assert back.GetDimensions() == grid.GetDimensions()
    np.testing.assert_array_equal(vtk_to_numpy(back.GetCellData().GetArray("c")),
                                  vtk_to_numpy(grid.GetCellData().GetArray("c")))


@needs_vtk
@pytest.mark.parametrize("dims", [(5, 4), (4, 3, 5)], ids=str)
def test_filters_are_vtks(f64_backend, dims):
    from tack.interop.vtk import vtk_to_dataset
    grid = _vtk_grid(dims)
    data = vtk_to_dataset(grid, dtype=tack.f64)
    want = vtk_to_numpy(_run(vtk.vtkCellCenters(), grid).GetPoints().GetData())
    np.testing.assert_allclose(td.cell_centers(data).points.to_numpy(vectors=True), want,
                               atol=1e-12)
    want = vtk_to_numpy(_run(vtk.vtkCellDataToPointData(), grid).GetPointData().GetArray("c"))
    np.testing.assert_allclose(td.cell_data_to_point_data(data).point_data["c"].to_numpy(),
                               want, atol=1e-12)
    if len(dims) == 3:
        surface = td.external_faces(data)
        poly = _run(vtk.vtkGeometryFilter(), grid)
        assert surface.num_cells == poly.GetNumberOfCells()
        # vtkGeometryFilter writes a rectilinear grid's points as f32.
        pts = np.round(data.points.to_numpy(), 5)
        offsets = surface.cells.offsets.to_numpy()
        conn = surface.cells.connectivity.to_numpy()
        got = {tuple(sorted(map(tuple, pts[conn[offsets[i]:offsets[i + 1]]])))
               for i in range(surface.num_cells)}
        vtk_pts = np.round(vtk_to_numpy(poly.GetPoints().GetData()).astype(float), 5)
        want = set()
        for i in range(poly.GetNumberOfCells()):
            ids = poly.GetCell(i).GetPointIds()
            want.add(tuple(sorted(map(tuple, vtk_pts[[ids.GetId(k)
                                                      for k in range(ids.GetNumberOfIds())]]))))
        assert got == want
