"""VTK's tests of its higher-order unstructured cells, through Tack: the Lagrange,
quadratic, biquadratic, triquadratic and cubic cells of VTK's test data, and one
cell of each type as VTK's quadratic-cell tests build them, read by
``vtk_to_dataset`` and held to what VTK computes on them -- each cell's own
interpolation (``EvaluateLocation``) at random points, derivatives, and the
contours, slices, clips, boundaries and edges of ``LagrangeGeometricOperations``
and ``contour``/``clip``/``ExtractEdgesQuadraticCells``.

VTK linearizes a higher-order cell on its own nodes before cutting it, so VTK's
cuts are Tack's with ``subdivision`` the cell's order and ``exact=False``, to
rounding where the two triangulate alike; Tack's defaults (exact crossings) are
checked to approach the finer answers. The files need VTK's test data
(``TACK_VTK_DATA``, its ``ExternalData`` directory); without it the module skips.
"""

import os

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import implicit

vtk = pytest.importorskip("vtkmodules.vtkCommonDataModel")
_DATA = os.environ.get("TACK_VTK_DATA", "")
if not _DATA or not os.path.isdir(os.path.join(_DATA, "Testing", "Data")):
    pytest.skip("needs VTK's test data (TACK_VTK_DATA)", allow_module_level=True)

from vtkmodules.vtkCommonCore import reference

from tack.dataset.algorithms.gradients import _world_gradient
from tack.interop.vtk import vtk_to_dataset

#: VTK's higher-order test files: (path under the data directory).
_FILES = {
    "Elements.vtu": "Filters/Geometry/Testing/Data/Input/Elements.vtu",
    "quadraticTetra01.vtu": "Testing/Data/quadraticTetra01.vtu",
    "Disc_BiQuadraticQuads_0_0.vtu": "Testing/Data/Disc_BiQuadraticQuads_0_0.vtu",
    "Disc_QuadraticQuads_0_0.vtu": "Testing/Data/Disc_QuadraticQuads_0_0.vtu",
    "CylinderQuadratic.vtk": "Testing/Data/Quadratic/CylinderQuadratic.vtk",
    "quadratic_tets_with_ghost_cells_0.vtu":
        "Testing/Data/quadratic_tets_with_ghost_cells_0.vtu",
}


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Every comparison in f64."""


def _read(name):
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    path = os.path.join(_DATA, _FILES[name])
    reader = (vtkUnstructuredGridReader if path.endswith(".vtk")
              else vtkXMLUnstructuredGridReader)(file_name=path)
    reader.Update()
    return reader.GetOutput()


@tack.kernel
def _at(cells, u, pcs, values, positions):
    # The field and the geometry at each cell's parametric point.
    for c in cells:
        e = cells.entity_id(c)
        values[e] = u.value(c, pcs[e])
        positions[e] = cells.position(c, pcs[e])


def _vtk_at(grid, pcs, values):
    """VTK's positions and ``values`` (a point array, ``(points, width)``) at each
    cell's parametric point ``pcs[c]``, by the cell's own interpolation."""
    n = grid.number_of_cells
    positions = np.zeros((n, 3))
    out = np.zeros((n, values.shape[1]))
    for c in range(n):
        cell = grid.GetCell(c)
        x = [0.0, 0.0, 0.0]
        weights = [0.0] * cell.GetNumberOfPoints()
        at = list(pcs[c])
        if cell.GetCellType() == 35:                  # vtkCubicLine's r is on [-1, 1]
            at[0] = 2.0 * at[0] - 1.0
        cell.EvaluateLocation(reference(0), at, x, weights)
        positions[c] = x
        ids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
        out[c] = np.asarray(weights) @ values[ids]
    return positions, out


def _random_points(grid, rng):
    """A random parametric point inside each cell of ``grid``, ``(cells, 3)``."""
    pcs = np.zeros((grid.number_of_cells, 3))
    for c in range(grid.number_of_cells):
        cell = grid.GetCell(c)
        p = rng.uniform(0.05, 0.9, 3)
        if cell.GetCellType() in (22, 24, 34, 69, 71, 31, 32, 26, 73):   # simplex parts
            p[:2] *= 0.9 / max(p[:2].sum(), 1.0)
            if cell.GetCellType() in (24, 71):
                p *= 0.9 / max(p.sum(), 1.0)
        p[cell.GetCellDimension():] = 0.0
        pcs[c] = p
    return pcs


def _compare(grid, data, rng, rtol=1e-12):
    """Tack's geometry and every point field against VTK's cells' own interpolation."""
    pcs = _random_points(grid, rng)
    points = tack.empty((len(pcs),), tack.vec3(tack.f64))
    tack.copyto(points, pcs)
    for name in grid.point_data.keys():
        if name not in data.fields:
            continue
        values = np.asarray(grid.point_data[name], np.float64).reshape(grid.number_of_points, -1)
        width = values.shape[1]
        ours = tack.empty((len(pcs),), tack.f64 if width == 1 else tack.vec(width, tack.f64))
        positions = tack.empty((len(pcs),), tack.vec3(tack.f64))
        td.for_each(_at, data, "cells", data.fields[name], points, ours, positions)
        theirs_x, theirs = _vtk_at(grid, pcs, values)
        np.testing.assert_allclose(positions.numpy() + np.asarray(data.origin), theirs_x,
                                   atol=1e-12 * max(np.abs(theirs_x).max(), 1.0))
        np.testing.assert_allclose(ours.numpy().reshape(len(pcs), -1), theirs,
                                   atol=rtol * max(np.abs(theirs).max(), 1.0), err_msg=name)


@pytest.mark.parametrize("name", list(_FILES))
def test_files_read_and_evaluate_as_vtk(name):
    """Every file reads -- mixed shapes, dimensions and orders together (Elements.vtu's
    Lagrange hexahedra and quads; quadratic tetrahedra and triangles), the
    biquadratic quad at its order (it lost it, read as linear) -- and its geometry
    and point fields at random points of every cell are what VTK's cells give
    there. The serendipity quads' integer node numbers were truncated at the face
    centres."""
    grid = _read(name)
    data = vtk_to_dataset(grid, dtype=tack.f64)
    assert data.num_cells == grid.number_of_cells
    assert data.geometry.space.order == 2
    _compare(grid, data, np.random.default_rng(0))


# ── LagrangeGeometricOperations (Elements.vtu's Lagrange hexahedra) ──

def _measure(dataset, kind):
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    sizes = vtkCellSizeFilter(compute_sum=True)
    sizes.SetInputDataObject(dataset)
    sizes.Update()
    return float(np.asarray(sizes.GetOutput().field_data[kind])[0])


def _total(data):
    """A Tack dataset's (or parts') total measure."""
    parts = data.partitions if isinstance(data, td.PartitionedDataSet) else [data]
    return sum(float(td.cell_geometry(p).measure.values.numpy().sum()) for p in parts
               if p.num_cells)


@pytest.fixture(scope="module")
def hexahedra():
    """Elements.vtu's two Lagrange hexahedra of order 2, VTK's and Tack's."""
    from vtkmodules.vtkFiltersExtraction import vtkExtractCellsByType

    pick = vtkExtractCellsByType()
    pick.AddCellType(72)
    grid = pick(_read("Elements.vtu"))
    return grid, grid


def _plane():
    from vtkmodules.vtkCommonDataModel import vtkPlane

    return vtkPlane(origin=(4.0, 2.25, 2.25), normal=(1.0, 1.0, 1.0)), \
        implicit.Plane((4.0, 2.25, 2.25), (1.0, 1.0, 1.0))


@pytest.mark.parametrize("level", [2.5, 1.5, 0.5, 1.05])
def test_lagrange_contours_are_vtks(hexahedra, level):
    """``vtkContourFilter``'s isosurfaces of the hexahedra are Tack's contour on the
    cells' own order-2 nodes, linearly (``subdivision=2, exact=False``); Tack's
    default, with exact crossings, is nearer the fine answer than either."""
    from vtkmodules.vtkCommonDataModel import vtkDataSet
    from vtkmodules.vtkFiltersCore import vtkContourFilter

    grid, _ = hexahedra
    data = vtk_to_dataset(grid, dtype=tack.f64)
    contour = vtkContourFilter()
    contour.SetInputArrayToProcess(0, 0, 0, vtkDataSet.FIELD_ASSOCIATION_POINTS_THEN_CELLS,
                                   "Ellipsoid")
    contour.SetValue(0, level)
    theirs = contour(grid)
    ours = td.contour(data, "Ellipsoid", level, subdivision=2, exact=False)
    # VTK cuts a Lagrange cell's linear pieces in single precision: 1e-7.
    assert _total(ours) == pytest.approx(_measure(theirs, "Area"), rel=1e-7)
    fine = _total(td.contour(data, "Ellipsoid", level, subdivision=16))
    assert abs(_total(td.contour(data, "Ellipsoid", level)) - fine) <= \
        abs(_measure(theirs, "Area") - fine)


def test_lagrange_slices_and_clips_are_vtks(hexahedra):
    """``vtkCutter`` and ``vtkClipDataSet`` by a plane are Tack's slice and clip on the
    order-2 nodes (VTK keeps the plane's positive side: ``inside=False``)."""
    from vtkmodules.vtkFiltersCore import vtkCutter
    from vtkmodules.vtkFiltersGeneral import vtkClipDataSet

    grid, _ = hexahedra
    data = vtk_to_dataset(grid, dtype=tack.f64)
    plane, ours_plane = _plane()
    cut = vtkCutter(cut_function=plane)(grid)
    assert _total(td.slice(data, ours_plane, subdivision=2, exact=False)) == pytest.approx(
        _measure(cut, "Area"), rel=1e-12)
    kept = vtkClipDataSet(clip_function=plane)(grid)[0]
    ours = td.clip(data, ours_plane, inside=False, subdivision=2, exact=False)
    assert sum(td.integrate(p).measure for p in ours.partitions if p.num_cells) == \
        pytest.approx(_measure(kept, "Volume"), rel=1e-12)


def test_lagrange_boundaries_are_vtks(hexahedra):
    """``vtkDataSetSurfaceFilter``'s boundary is Tack's at level 0 (the corners), and
    finer, both approach the curved boundary's exact area (``integrate`` of Tack's
    curved ``extract_surface``), Tack's 2^n split nearer than VTK's level n."""
    from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter

    grid, _ = hexahedra
    data = vtk_to_dataset(grid, dtype=tack.f64)
    exact = td.integrate(td.extract_surface(data)).measure
    for level in range(4):
        theirs = _measure(vtkDataSetSurfaceFilter(nonlinear_subdivision_level=level)(grid),
                          "Area")
        ours = _total(td.extract_surface(td.subdivide(data, 2 ** level, fields=[])))
        if level == 0:
            assert ours == pytest.approx(theirs, rel=1e-12)
        else:
            assert abs(ours - exact) <= abs(theirs - exact)


# ── One cell of each type (contour/clip/ExtractEdgesQuadraticCells) ──

#: VTK's higher-order cell classes, by type, with the order of the space Tack reads
#: each into (Lagrange cells are made below at order 2 and 3).
_CELLS = {21: ("vtkQuadraticEdge", 2), 22: ("vtkQuadraticTriangle", 2),
          23: ("vtkQuadraticQuad", 2), 24: ("vtkQuadraticTetra", 2),
          25: ("vtkQuadraticHexahedron", 2), 26: ("vtkQuadraticWedge", 2),
          28: ("vtkBiQuadraticQuad", 2), 29: ("vtkTriQuadraticHexahedron", 2),
          30: ("vtkQuadraticLinearQuad", 2), 31: ("vtkQuadraticLinearWedge", 2),
          32: ("vtkBiQuadraticQuadraticWedge", 2), 33: ("vtkBiQuadraticQuadraticHexahedron", 2),
          34: ("vtkBiQuadraticTriangle", 3), 35: ("vtkCubicLine", 3)}


def _one_cell(vtk_type):
    """A grid of one cell of ``vtk_type`` on its own parametric nodes, bent so it is
    curved, with a point field ``u`` smooth in position."""
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

    cell = getattr(vtk, _CELLS[vtk_type][0])()
    n = cell.GetNumberOfPoints()
    p = np.asarray(cell.GetParametricCoords(), np.float64)[:3 * n].reshape(n, 3).copy()
    if vtk_type == 35:
        p[:, 0] = (p[:, 0] + 1.0) / 2.0
    x = np.c_[p[:, 0] + 0.1 * p[:, 1] ** 2, p[:, 1] + 0.15 * p[:, 0] ** 2,
              p[:, 2] + 0.1 * p[:, 0] * p[:, 1]]
    grid = vtkUnstructuredGrid(points=vtkPoints(data=x))
    grid.InsertNextCell(vtk_type, n, list(range(n)))
    u = x[:, 0] + 0.6 * x[:, 1] + 0.3 * x[:, 2] + 0.4 * x[:, 0] * x[:, 1]
    grid.point_data["u"] = vtkDoubleArray(u)
    return grid


@pytest.mark.parametrize("vtk_type", list(_CELLS))
def test_each_higher_order_cell_evaluates_as_vtk(vtk_type):
    """One cell of each of VTK's quadratic, biquadratic, triquadratic and cubic types
    reads into Tack's space of its order and evaluates as VTK's cell does: those
    whose nodes are Tack's copied, the others (a hex20 or a quadratic-linear quad
    in order 2, the 7-node triangle in order 3) through VTK's interpolation
    functions, exactly. Every one but the five Tack had (21-26) was refused or,
    the biquadratic quad, read as linear."""
    grid = _one_cell(vtk_type)
    data = vtk_to_dataset(grid, dtype=tack.f64)
    assert data.geometry.space.order == _CELLS[vtk_type][1]
    _compare(grid, data, np.random.default_rng(vtk_type))


@pytest.mark.parametrize("vtk_type", list(_CELLS))
def test_each_higher_order_cell_cuts_as_vtk(vtk_type):
    """``vtkContourFilter`` and ``vtkClipDataSet`` at 0.5 on one cell are Tack's on
    the cell's lattice of its order, linearly: to rounding where VTK cuts it into
    the same pieces, to 2% where it cuts it into others (the quadratic-linear quad
    into two quads, the 7-node triangle into six triangles, where Tack's lattice
    is finer). ``vtkExtractEdges`` splits each quadratic edge at its middle node:
    the length of Tack's edges split in two (``subdivision=2``)."""
    from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm
    from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkExtractEdges
    from vtkmodules.vtkFiltersGeneral import vtkClipDataSet

    grid = _one_cell(vtk_type)
    grid.point_data.SetActiveScalars("u")
    data = vtk_to_dataset(grid, dtype=tack.f64)
    order = _CELLS[vtk_type][1]
    rel = 2e-2 if vtk_type in (30, 34) else 1e-9
    dimension = grid.GetCell(0).GetCellDimension()
    # Points in double precision (VTK's default output is single).
    contour = vtkContourFilter(output_points_precision=vtkAlgorithm.DOUBLE_PRECISION)
    contour.SetValue(0, 0.5)
    theirs = contour(grid)
    ours = td.contour(data, "u", 0.5, subdivision=order, exact=False)
    if dimension == 1:
        assert ours.num_points == theirs.number_of_points
    else:
        kind = "Length" if dimension == 2 else "Area"
        assert _total(ours) == pytest.approx(_measure(theirs, kind), rel=rel)
    clip = vtkClipDataSet(value=0.5, output_points_precision=vtkAlgorithm.DOUBLE_PRECISION)
    kept = clip(grid)[0]
    ours_clip = td.clip(data, "u", 0.5, subdivision=order, exact=False)
    kind = {1: "Length", 2: "Area", 3: "Volume"}[dimension]
    total = sum(float(td.cell_geometry(p).measure.values.numpy().sum())
                for p in ours_clip.partitions if p.num_cells)
    # vtkClipDataSet places its crossings in single precision whatever its output's.
    assert total == pytest.approx(_measure(kept, kind), rel=max(rel, 1e-7))
    edges = vtkExtractEdges()(grid)
    if dimension > 1:
        ours_edges = td.extract_edges(data, subdivision=2)
        # vtkExtractEdges writes single-precision points.
        assert _total(ours_edges) == pytest.approx(_measure(edges, "Length"), rel=1e-7)


def test_quadratic_quad_derivatives_are_vtks():
    """VTK's ``quadraticQuadDeriv``: the derivatives of a field on the quadratic and
    biquadratic quads of Disc_*Quads are VTK's ``Derivatives`` at random points."""
    for name in ("Disc_QuadraticQuads_0_0.vtu", "Disc_BiQuadraticQuads_0_0.vtu"):
        grid = _read(name)
        data = vtk_to_dataset(grid, dtype=tack.f64)
        rng = np.random.default_rng(5)
        pcs = _random_points(grid, rng)
        points = tack.empty((len(pcs),), tack.vec3(tack.f64))
        tack.copyto(points, pcs)
        values = np.asarray(grid.points.data, np.float64)[:, 0] ** 2
        theirs = np.zeros((len(pcs), 3))
        for c in range(grid.number_of_cells):
            cell = grid.GetCell(c)
            ids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
            derivs = [0.0, 0.0, 0.0]
            cell.Derivatives(0, list(pcs[c]), list(values[ids]), 1, derivs)
            theirs[c] = derivs
        data.fields["x2"] = td.interpolate(data, _x_squared, data.geometry.space)
        ours = tack.empty((len(pcs),), tack.vec3(tack.f64))
        td.for_each(_gradient_at, data, "cells", data.fields["x2"], points, ours)
        np.testing.assert_allclose(ours.numpy()[:, :2], theirs[:, :2], atol=1e-9, err_msg=name)


@tack.func
def _x_squared(x):
    return x[0] * x[0]


@tack.kernel
def _gradient_at(cells, u, pcs, out):
    for c in cells:
        e = cells.entity_id(c)
        out[e] = _world_gradient(cells, u, c, pcs[e])   # in a surface's plane on 2D cells
