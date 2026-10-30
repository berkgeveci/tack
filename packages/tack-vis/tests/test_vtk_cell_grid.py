"""VTK's cell grid tests, through Tack: every ``.dg`` file of VTK's test data read
by ``vtk_to_dataset``, held to what VTK computes on it -- its evaluator at random
points of every cell (the geometry and each field), its cell centres, point
probe, attribute ranges, conversion to linear cells and transform.

VTK's own cell grid tests are mostly pictures; these compare the numbers. They
need a VTK with ``vtkCellAttribute``'s cell-type accessors and the evaluator's
interpolate-only mode (newer than 9.7.1) and VTK's test data
(``TACK_VTK_DATA``, its ``ExternalData`` directory): see the local build in
``~/Library/Modules/vtk-cell-grid``. Without them the module skips.
"""

import glob
import os
import warnings

import numpy as np
import pytest

import tack
import tack.dataset as td

vtk = pytest.importorskip("vtkmodules.vtkCommonDataModel")
_DATA = os.environ.get("TACK_VTK_DATA", "")
_FILES = sorted(glob.glob(os.path.join(_DATA, "Testing", "Data", "*.dg"))) if _DATA else []
if not _FILES or not hasattr(vtk.vtkCellAttribute, "GetCellTypes"):
    pytest.skip("needs VTK's cell grid test data (TACK_VTK_DATA) and a VTK with "
                "vtkCellAttribute's cell-type accessors", allow_module_level=True)

from tack.interop import vtk as interop
from tack.interop.vtk import _cellgrid, vtk_to_dataset

_NAMES = [os.path.basename(f) for f in _FILES]
#: What Tack refuses, and why.
_REFUSED = {"dgPyramid19.dg": "curved pyramid"}
#: The attributes left out (no Tack space): H(curl), H(div), pyramids' rational bases.
_LEFT_OUT = ("HCURL", "HDIV", "HGRAD G")


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Every comparison in f64."""


def _read(name):
    from vtkmodules.vtkIOCellGrid import vtkCellGridReader

    reader = vtkCellGridReader(file_name=os.path.join(_DATA, "Testing", "Data", name))
    reader.Update()
    return reader.GetOutput()


def _imported(name):
    grid = _read(name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = vtk_to_dataset(grid, dtype=tack.f64)
    return grid, data, [str(w.message) for w in caught]


@pytest.mark.parametrize("name", _NAMES)
def test_every_cell_grid_reads_or_is_refused(name):
    """Each file reads, its shape and every attribute Tack has a space for; what is
    left out is named (H(curl), H(div), pyramids' rational bases), and a curved
    pyramid is refused."""
    if name in _REFUSED:
        with pytest.raises(NotImplementedError, match=_REFUSED[name]):
            vtk_to_dataset(_read(name))
        return
    grid, data, messages = _imported(name)
    assert data.num_cells == grid.GetNumberOfCells()
    for message in messages:
        for part in message.split(": ", 1)[1].split(", "):
            assert any(kind in part for kind in _LEFT_OUT), part
    names = {_cellgrid._text(a.GetName()) for a in grid.GetCellAttributeList()}
    names.discard(_cellgrid._text(grid.GetShapeAttribute().GetName()))
    left = {part.split(" (")[0] for message in messages
            for part in message.split(": ", 1)[1].split(", ")}
    assert set(data.fields) == names - left


@tack.kernel
def _at(cells, u, pcs, per_cell, values, positions):
    # The field and the geometry at each cell's per_cell parametric points.
    for c in cells:
        e = cells.entity_id(c)
        for k in range(per_cell):
            pc = pcs[e * per_cell + k]
            values[e * per_cell + k] = u.value(c, pc)
            positions[e * per_cell + k] = cells.position(c, pc)


def _inside(shape, count, rng):
    """``count`` random parametric points inside a Tack ``shape``, ``(count, 3)``."""
    from tack.dataset import shapes

    p = rng.uniform(0.05, 0.95, (count, 3))
    if shape in (shapes.Triangle, shapes.Wedge):
        p[:, :2] *= 0.95 / np.maximum(p[:, :2].sum(axis=1, keepdims=True), 1.0)
    if shape is shapes.Tetra:
        p *= 0.95 / np.maximum(p.sum(axis=1, keepdims=True), 1.0)
    p[:, shape.DIMENSION:] = 0.0
    return p


def _cells_by_type(grid):
    """``(cell type, Tack shape, first cell, count)`` in Tack's cell order."""
    out, first = [], 0
    for cell_type in map(_cellgrid._text, grid.GetCellTypes()):
        count = grid.GetCellType(cell_type).GetNumberOfCells()
        out.append((cell_type, _cellgrid.SHAPES[cell_type], first, count))
        first += count
    return out


@pytest.mark.parametrize("name", [n for n in _NAMES if n not in _REFUSED])
def test_values_are_vtks_at_random_points(name):
    """The geometry and every field Tack read, at random points of every cell, are
    what VTK's evaluator gives there: copied attributes to the bit's neighbourhood,
    resampled ones (a basis Tack does not have, on the same polynomials) to
    rounding."""
    grid, data, _ = _imported(name)
    rng = np.random.default_rng(7)
    per_cell = 5
    pcs = np.zeros((data.num_cells * per_cell, 3))
    for _, shape, first, count in _cells_by_type(grid):
        pcs[first * per_cell:(first + count) * per_cell] = _inside(shape, count * per_cell, rng)
    attributes = {_cellgrid._text(a.GetName()): a for a in grid.GetCellAttributeList()}
    shape_name = _cellgrid._text(grid.GetShapeAttribute().GetName())
    for field_name in [None, *data.fields]:
        field = data.geometry if field_name is None else data.fields[field_name]
        attribute = attributes[shape_name if field_name is None else field_name]
        width = attribute.GetNumberOfComponents()
        n = data.num_cells * per_cell
        ours = tack.empty((n,), tack.f64 if width == 1 else tack.vec(width, tack.f64))
        positions = tack.empty((n,), tack.vec3(tack.f64))
        points = tack.empty((n,), tack.vec3(tack.f64))
        tack.copyto(points, pcs)
        td.for_each(_at, data, "cells", field, points, per_cell, ours, positions)
        ours = ours.numpy().reshape(n, -1)
        theirs = np.zeros_like(ours)
        for cell_type, shape, first, count in _cells_by_type(grid):
            rows = slice(first * per_cell, (first + count) * per_cell)
            theirs[rows] = _cellgrid.evaluate(
                grid, attribute, cell_type, np.repeat(np.arange(count), per_cell),
                _cellgrid._to_vtk_parameters(shape, pcs[rows]))
        if field_name is None:
            theirs = theirs - np.asarray(data.origin)
        scale = max(np.abs(theirs).max(), 1.0)
        np.testing.assert_allclose(ours, theirs, atol=1e-11 * scale,
                                   err_msg=f"{name}: {field_name or 'the geometry'}")


def _points(dataset):
    """A VTK point set's points as a plain array."""
    return np.array(dataset.GetPoints().GetData(), np.float64).reshape(-1, 3)


def _linear(output):
    """A cell grid filter's output through ``vtkCellGridToUnstructuredGrid``."""
    from vtkmodules.vtkFiltersCellGrid import vtkCellGridToUnstructuredGrid

    return vtkCellGridToUnstructuredGrid()(output)


#: VTK bugs these tests found (the vault's VTK bugs note), expected to fail until fixed.
_VTK_MIXED_CENTERS = pytest.mark.xfail(
    strict=True, reason="VTK: vtkCellGridCellCenters of several cell types gives one point at "
    "the origin ('Invalid cell ID 1 at index 2')")
_VTK_2D_PROBE = pytest.mark.xfail(
    strict=True, reason="VTK: vtkCellGridPointProbe finds no point in a 2D cell (its "
    "classifier's 3x3 Jacobian is singular there)")


@pytest.mark.parametrize("name", ["dgHexahedra.dg", "dgTetrahedra.dg", "dgWedges.dg",
                                  "dgQuadraticQuadrilaterals.dg",
                                  pytest.param("dgMixed.dg", marks=_VTK_MIXED_CENTERS),
                                  "dgTriangle.dg", "dgEdges.dg"])
def test_cell_centers_are_vtks(name):
    """``td.cell_centers`` are ``vtkCellGridCellCenters``' points (each cell's
    parametric centre mapped), and every field's value there is VTK's."""
    from vtkmodules.vtkFiltersCellGrid import vtkCellGridCellCenters

    grid, data, _ = _imported(name)
    centers = _linear(vtkCellGridCellCenters()(grid))
    theirs = _points(centers) - np.asarray(data.origin)
    ours = td.cell_centers(data).values.numpy()
    probe = td.probe(data, ours)
    order = [np.argmin(np.abs(theirs - p).sum(axis=1)) for p in ours]
    np.testing.assert_allclose(theirs[order], ours, atol=1e-12)
    for field_name in data.fields:
        if field_name in centers.point_data.keys() and not isinstance(
                data.fields[field_name].space, td.L2):
            np.testing.assert_allclose(
                np.asarray(centers.point_data[field_name])[order].reshape(len(ours), -1),
                probe.fields[field_name].values.numpy().reshape(len(ours), -1),
                atol=1e-12, err_msg=field_name)


@pytest.mark.parametrize("name", ["dgHexahedra.dg", "dgTetrahedra.dg", "dgWedges.dg",
                                  "fandisk.dg",
                                  pytest.param("dgQuadraticQuadrilaterals.dg",
                                               marks=_VTK_2D_PROBE),
                                  pytest.param("dgTriangle.dg", marks=_VTK_2D_PROBE)])
def test_probes_are_vtks(name):
    """``td.probe`` at points inside the cells (each cell's random interior points,
    mapped by its geometry) is ``vtkCellGridPointProbe``'s, for every continuous
    field. VTK's output has a point per (point, cell holding it), copied from the
    input in its own order, so its points are matched to the input's by position."""
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCellGrid import vtkCellGridPointProbe

    grid, data, _ = _imported(name)
    rng = np.random.default_rng(3)
    pcs = np.concatenate([_inside(shape, count * 2, rng)
                          for _, shape, _, count in _cells_by_type(grid)])
    points = tack.empty((len(pcs),), tack.vec3(tack.f64))
    tack.copyto(points, pcs)
    world = tack.empty((len(pcs),), tack.vec3(tack.f64))
    td.for_each(_at, data, "cells", data.geometry, points, 2,
                tack.empty((len(pcs),), tack.vec3(tack.f64)), world)
    where = world.numpy() + np.asarray(data.origin)
    cloud = vtkPolyData(points=vtkPoints(data=where))
    shared = [n for n, f in data.fields.items() if isinstance(f.space, td.H1)]
    ours = td.probe(data, world.numpy())
    assert ours.fields["valid"].values.numpy().all()
    for field_name in shared:
        probe = vtkCellGridPointProbe(attribute_name=field_name)
        probe.SetInputDataObject(0, cloud)
        probe.SetInputDataObject(1, grid)
        probe.Update()
        theirs = probe.GetOutputDataObject(0)
        at = _points(theirs)
        which = [int(np.flatnonzero((where == p).all(axis=1))[0]) for p in at]
        assert sorted(set(which)) == list(range(len(where)))     # every point found
        np.testing.assert_allclose(
            np.asarray(theirs.point_data[field_name]).reshape(len(at), -1),
            ours.fields[field_name].values.numpy().reshape(len(where), -1)[which],
            atol=1e-10, err_msg=field_name)


@pytest.mark.parametrize("name", ["dgTetrahedra.dg", "dgHexahedra.dg", "dgMixed.dg"])
def test_order_1_ranges_are_vtks(name):
    """A ``C1`` field's range is its values' (VTK's ``GetCellAttributeRange``, the
    coefficients' extremes, which at corners is right; a Gauss-point basis's
    coefficients are not its extremes)."""
    grid, data, _ = _imported(name)
    for attribute in grid.GetCellAttributeList():
        field_name = _cellgrid._text(attribute.GetName())
        field = data.fields.get(field_name)
        cell_type = attribute.GetCellTypes()[0]
        if field is None or _cellgrid._text(attribute.GetBasis(cell_type)) != "C" or \
                attribute.GetOrder(cell_type) != 1 or attribute.GetNumberOfComponents() != 1:
            continue
        theirs = [1.0, -1.0]
        grid.GetCellAttributeRange(attribute, 0, theirs, True)
        values = field.values.numpy()
        assert (values.min(), values.max()) == pytest.approx(tuple(theirs), abs=1e-12)


@pytest.mark.parametrize("name", ["dgHexahedra.dg", "dgTetrahedra.dg", "dgWedges.dg",
                                  "dgQuadraticQuadrilaterals.dg"])
def test_linear_cells_measure_as_vtks(name):
    """``td.subdivide(data, 2**n)``'s linear cells measure what
    ``vtkCellGridToUnstructuredGrid`` at ``SubdivisionLevel`` n gives (2^n steps an
    axis), to rounding, curved quads included (8.0, 10.91, 11.58, 11.75 toward
    ``td.integrate``'s exact 11.81)."""
    from vtkmodules.vtkFiltersCellGrid import vtkCellGridToUnstructuredGrid
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    grid, data, _ = _imported(name)
    kind = {3: "Volume", 2: "Area", 1: "Length"}[data.topology.dimension]
    for level in range(4):
        linear = vtkCellGridToUnstructuredGrid(subdivision_level=level)(grid)
        theirs = vtkCellSizeFilter(compute_sum=True)(linear)
        total = float(np.asarray(theirs.field_data[kind])[0])
        ours = td.cell_geometry(td.subdivide(data, 2 ** level, fields=[]))
        assert float(ours.measure.values.numpy().sum()) == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize("name", ["dgHexahedra.dg", "dgTetrahedra.dg"])
def test_transforms_are_vtks(name):
    """``td.transform`` moves the geometry where ``vtkCellGridTransform`` does (a
    rotation about z by -60 degrees, as VTK's test)."""
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCellGrid import vtkCellGridTransform

    grid, data, _ = _imported(name)
    rotation = vtkTransform()
    rotation.RotateZ(-60)
    theirs = vtkCellGridTransform(transform=rotation)(grid)
    matrix = np.array([[rotation.GetMatrix().GetElement(i, j) for j in range(4)]
                       for i in range(4)])
    moved = td.transform(data, matrix)
    shape = theirs.GetShapeAttribute()
    cell_type = shape.GetCellTypes()[0]
    np.testing.assert_allclose(
        moved.world_positions(),
        np.asarray(shape.GetArrayForCellTypeAndRole(cell_type, "values")).reshape(-1, 3),
        atol=1e-12)


# ── Writing ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [n for n in _NAMES if n not in _REFUSED])
def test_cell_grids_round_trip(name):
    """Read, written back (``dataset_to_vtk(..., cell_grid=True)``) and read again,
    every file gives the same geometry and fields."""
    _, data, _ = _imported(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = vtk_to_dataset(interop.dataset_to_vtk(data, cell_grid=True), dtype=tack.f64)
    assert again.num_cells == data.num_cells
    np.testing.assert_allclose(again.world_positions(), data.world_positions(), atol=1e-13)
    assert set(again.fields) == set(data.fields)
    for field_name, field in data.fields.items():
        assert again.fields[field_name].space == type(field.space)(again, **field.space.params)
        np.testing.assert_allclose(again.fields[field_name].values.numpy(),
                                   field.values.numpy(), atol=1e-12, err_msg=field_name)


@tack.func
def _bent(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.15 * x[0] * x[2], x[2] + 0.1 * x[0] * x[1]]


@tack.func
def _cubic(x):
    return x[0] * x[0] * x[1] - 2.0 * x[2] * x[2] * x[2] + x[0] * x[1] * x[2] + 1.0


def _tack_cases():
    from tack.dataset import shapes
    from tack.dataset.topology import uniform_cells

    grid = td.uniform_grid((3, 3, 3), spacing=(0.5, 0.5, 0.5), dtype=tack.f64)
    hexahedra = td.extract_cells(grid, np.arange(8))
    tetrahedra = td.tetrahedralize(grid)
    hexes = hexahedra.topology.connectivity.numpy().reshape(-1, 8)
    conn = [[h[0], h[1], h[3], h[4], h[5], h[7]] for h in hexes] + \
        [[h[1], h[2], h[3], h[5], h[6], h[7]] for h in hexes]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    wedges = td.DataSet(td.UnstructuredTopology(
        types, offsets, tack.array(np.array(conn, np.int32).reshape(-1)),
        num_points=hexahedra.num_points), hexahedra.positions())
    quads = td.uniform_grid((4, 3), spacing=(0.5, 0.5), dtype=tack.f64)
    return {"hexahedra": hexahedra, "tetrahedra": tetrahedra, "wedges": wedges,
            "quads": quads}


@pytest.mark.parametrize("kind", ["hexahedra", "tetrahedra", "wedges", "quads"])
@pytest.mark.parametrize("order", [2, 3])
def test_written_cell_grids_evaluate_as_tack(kind, order):
    """A curved Tack dataset written as a cell grid -- the geometry and an ``H1``
    field of ``order`` in VTK's ``A`` basis, an ``L2`` field of ``order`` per cell,
    a cell field -- evaluates in VTK's own evaluator, at random points of every
    cell, as Tack evaluates it: VTK's ``A`` nodes are Tack's equispaced nodes in
    Tack's order."""
    data = _tack_cases()[kind]
    vec = tack.vec3(tack.f64)
    data.geometry = td.interpolate(data, _bent, td.H1(data, order=order), dtype=vec)
    data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=order))
    data.fields["w"] = td.interpolate(data, _cubic, td.L2(data, order=order))
    data.fields["c"] = td.Field(td.Constant(data), tack.array(np.arange(data.num_cells, dtype=float)))
    grid = interop.dataset_to_vtk(data, cell_grid=True)
    rng = np.random.default_rng(1)
    per_cell = 4
    shape = _cellgrid.SHAPES[_cellgrid._text(grid.GetCellTypes()[0])]
    pcs = _inside(shape, data.num_cells * per_cell, rng)
    points = tack.empty((len(pcs),), vec)
    tack.copyto(points, pcs)
    attributes = {_cellgrid._text(a.GetName()): a for a in grid.GetCellAttributeList()}
    cells = np.repeat(np.arange(data.num_cells), per_cell)
    for field_name, field in [("shape", data.geometry), *data.fields.items()]:
        width = 3 if field_name == "shape" else 1
        ours = tack.empty((len(pcs),), vec if width == 3 else tack.f64)
        td.for_each(_at, data, "cells", field, points, per_cell, ours,
                    tack.empty((len(pcs),), vec))
        theirs = _cellgrid.evaluate(grid, attributes[field_name],
                                    _cellgrid._text(grid.GetCellTypes()[0]), cells,
                                    _cellgrid._to_vtk_parameters(shape, pcs))
        np.testing.assert_allclose(theirs, ours.numpy().reshape(len(pcs), -1), atol=1e-12,
                                   err_msg=field_name)
