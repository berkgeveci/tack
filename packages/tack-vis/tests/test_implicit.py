"""Implicit functions and the filters that take them, against VTK: the functions'
values, slice (vtkCutter), extract_geometry (vtkExtractGeometry), and the
point and cell extractions (vtkExtractPoints, vtkThresholdPoints,
vtkMaskPoints, vtkExtractCells). On polyhedra the same as on shape cells."""

import numpy as np
import pytest
from test_dataset_api import _two_hexes_and_a_pyramid

import tack.data as td

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

KINDS = {"hexahedron": 12, "tetra": 10, "wedge": 13}

# Away from the grid's integer points, so no point sits on (or within 1e-7 of)
# a zero: there vtkCutter triangulates differently from VTK's own contour
# filters, which agree with Tack. test_no_point_is_near_a_zero checks it.
FUNCTIONS = {
    "plane": (lambda: td.Plane((3.03, 2.97, 3.1), (1, 0.4, 0.2)),
              lambda: _vtk_plane((3.03, 2.97, 3.1), (1, 0.4, 0.2))),
    "sphere": (lambda: td.Sphere((3.1, 2.9, 3.05), 2.3),
               lambda: _vtk(vtk.vtkSphere, SetCenter=(3.1, 2.9, 3.05), SetRadius=2.3)),
    "box": (lambda: td.Box((1.1, 0.9, 1.3), (4.7, 5.1, 3.9)),
            lambda: _vtk(vtk.vtkBox, SetBounds=(1.1, 4.7, 0.9, 5.1, 1.3, 3.9))),
    "cylinder": (lambda: td.Cylinder((3.05, 3.1, 0.0), (0.2, 0.1, 1.0), 1.9),
                 lambda: _vtk(vtk.vtkCylinder, SetCenter=(3.05, 3.1, 0.0),
                              SetAxis=tuple(np.array([0.2, 0.1, 1.0]) / np.linalg.norm(
                                  [0.2, 0.1, 1.0])), SetRadius=1.9)),
    "planes": (lambda: td.Planes([(2.07, 0, 0), (4.13, 0, 0)], [(-1, 0.1, 0), (1, 0, 0.2)]),
               lambda: _vtk_planes([(2.07, 0, 0), (4.13, 0, 0)], [(-1, 0.1, 0), (1, 0, 0.2)])),
}


def _vtk(cls, **settings):
    f = cls()
    for name, value in settings.items():
        getattr(f, name)(*value) if isinstance(value, tuple) else getattr(f, name)(value)
    return f


def _unit(v):
    return tuple(np.asarray(v, float) / np.linalg.norm(v))


def _vtk_plane(origin, normal):
    return _vtk(vtk.vtkPlane, SetOrigin=origin, SetNormal=_unit(normal))


def _vtk_planes(origins, normals):
    points = vtk.vtkPoints()
    vectors = vtk.vtkDoubleArray()
    vectors.SetNumberOfComponents(3)
    for o, n in zip(origins, normals):
        points.InsertNextPoint(*o)
        vectors.InsertNextTuple3(*_unit(n))
    planes = vtk.vtkPlanes()
    planes.SetPoints(points)
    planes.SetNormals(vectors)
    return planes


def _grid(kind):
    source = vtk.vtkCellTypeSource()
    source.SetCellType(KINDS[kind])
    source.SetBlocksDimensions(6, 6, 6)
    source.Update()
    grid = source.GetOutput()
    x = vtk_to_numpy(grid.GetPoints().GetData())
    height = numpy_to_vtk(
        (x[:, 2] + 0.3 * np.sin(x[:, 0])).astype(np.float32), deep=True)
    height.SetName("h")
    grid.GetPointData().AddArray(height)
    return grid


def _measure(grid, what):
    integrate = vtk.vtkIntegrateAttributes()
    integrate.SetInputData(grid)
    integrate.Update()
    array = integrate.GetOutput().GetCellData().GetArray(what)
    return array.GetValue(0) if array is not None else 0.0


def _run(algorithm, grid):
    algorithm.SetInputData(grid)
    algorithm.Update()
    return algorithm.GetOutput()


@needs_vtk
@pytest.mark.parametrize("name", FUNCTIONS)
def test_values_are_vtks(backend, name):
    ours, theirs = FUNCTIONS[name]
    data = vtk_to_dataset(_grid("hexahedron"))
    values = td.algorithms.implicit_values(data, ours()).values.to_numpy()
    reference = theirs()
    expected = [reference.EvaluateFunction(*p) for p in data.positions()]
    np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-5)


@needs_vtk
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", FUNCTIONS)
def test_slice_is_vtks_cutter(backend, kind, name):
    ours, theirs = FUNCTIONS[name]
    grid = _grid(kind)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(theirs())
    cutter.GenerateTrianglesOn()
    reference = _run(cutter, grid)
    out = td.slice(vtk_to_dataset(grid), ours())
    assert (out.num_cells, out.num_points) == (reference.GetNumberOfCells(),
                                               reference.GetNumberOfPoints())
    np.testing.assert_allclose(_measure(dataset_to_vtk(out), "Area"),
                               _measure(reference, "Area"), rtol=1e-4)
    # The point data comes along, interpolated.
    assert "h" in out.fields


@needs_vtk
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("inside, boundary", [(True, False), (False, False), (True, True)])
def test_extract_geometry_is_vtks(backend, kind, name, inside, boundary):
    ours, theirs = FUNCTIONS[name]
    grid = _grid(kind)
    extract = vtk.vtkExtractGeometry()
    extract.SetImplicitFunction(theirs())
    extract.SetExtractInside(inside)
    extract.SetExtractBoundaryCells(boundary)
    reference = _run(extract, grid)
    out = td.extract_geometry(vtk_to_dataset(grid), ours(), inside=inside, boundary=boundary)
    assert (out.num_cells, out.num_points) == (reference.GetNumberOfCells(),
                                               reference.GetNumberOfPoints())
    if out.num_cells:
        np.testing.assert_allclose(_measure(dataset_to_vtk(out), "Volume"),
                                   _measure(reference, "Volume"), rtol=1e-5)


@needs_vtk
@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("inside", [True, False])
def test_extract_points_is_vtks(backend, name, inside):
    ours, theirs = FUNCTIONS[name]
    grid = _grid("hexahedron")
    extract = vtk.vtkExtractPoints()
    extract.SetImplicitFunction(theirs())
    extract.SetExtractInside(inside)
    extract.GenerateVerticesOn()
    reference = _run(extract, grid)
    out = td.extract_points(vtk_to_dataset(grid), ours(), inside=inside)
    assert out.num_points == out.num_cells == reference.GetNumberOfPoints()
    np.testing.assert_allclose(
        np.sort(out.positions(), axis=0),
        np.sort(vtk_to_numpy(reference.GetPoints().GetData()), axis=0))
    assert "h" in out.fields


@needs_vtk
def test_threshold_points_is_vtks(backend):
    grid = _grid("tetra")
    threshold = vtk.vtkThresholdPoints()
    threshold.SetInputArrayToProcess(0, 0, 0, 0, "h")
    threshold.SetThresholdFunction(vtk.vtkThresholdPoints.THRESHOLD_BETWEEN)
    threshold.SetLowerThreshold(2.05)
    threshold.SetUpperThreshold(3.95)
    reference = _run(threshold, grid)
    out = td.threshold_points(vtk_to_dataset(grid), "h", 2.05, 3.95)
    assert out.num_points == reference.GetNumberOfPoints() > 0
    values = out.fields["h"].values.to_numpy()
    assert (values >= 2.05).all() and (values <= 3.95).all()


@needs_vtk
@pytest.mark.parametrize("stride", [1, 3, 7])
def test_mask_points_is_vtks(backend, stride):
    grid = _grid("wedge")
    masker = vtk.vtkMaskPoints()
    masker.SetOnRatio(stride)
    masker.RandomModeOff()
    masker.SetOffset(0)
    masker.GenerateVerticesOn()
    masker.SetMaximumNumberOfPoints(10 ** 9)
    reference = _run(masker, grid)
    data = vtk_to_dataset(grid)
    out = td.mask_points(data, stride)
    np.testing.assert_allclose(out.positions(), data.positions()[::stride])
    np.testing.assert_allclose(
        out.positions(), vtk_to_numpy(reference.GetPoints().GetData()))


@needs_vtk
def test_extract_cells_and_mask_are_vtks(backend):
    grid = _grid("tetra")
    data = vtk_to_dataset(grid)
    ids = np.arange(0, data.num_cells, 5)
    cell_list = vtk.vtkIdList()
    for i in ids:
        cell_list.InsertNextId(int(i))
    extract = vtk.vtkExtractCells()
    extract.SetCellList(cell_list)
    reference = _run(extract, grid)
    for out in (td.extract_cells(data, ids), td.extract_cells(data, ids[::-1].copy()),
                td.mask(data, 5)):
        assert (out.num_cells, out.num_points) == (reference.GetNumberOfCells(),
                                                   reference.GetNumberOfPoints())
        np.testing.assert_allclose(_measure(dataset_to_vtk(out), "Volume"),
                                   _measure(reference, "Volume"), rtol=1e-5)
    with pytest.raises(IndexError):
        td.extract_cells(data, [data.num_cells])


@needs_vtk
@pytest.mark.parametrize("name", FUNCTIONS)
def test_on_polyhedra_as_on_shape_cells(backend, name):
    ours, _ = FUNCTIONS[name]
    data = vtk_to_dataset(_grid("wedge"))
    poly = td.as_polyhedra(data)
    f = ours()
    for run in (lambda d: td.slice(d, f), lambda d: td.extract_geometry(d, f),
                lambda d: td.extract_geometry(d, f, boundary=True),
                lambda d: td.extract_points(d, f)):
        theirs, mine = run(data), run(poly)
        assert mine.num_points == theirs.num_points
        np.testing.assert_allclose(np.unique(mine.positions().round(5), axis=0),
                                   np.unique(theirs.positions().round(5), axis=0), atol=1e-5)


def test_bad_functions_are_refused():
    with pytest.raises(ValueError):
        td.Plane((0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        td.Sphere((0, 0, 0), -1)
    with pytest.raises(ValueError):
        td.Box((1, 0, 0), (0, 1, 1))
    with pytest.raises(ValueError):
        td.Planes([], [])


def test_moving_a_function_compiles_nothing_new(backend):
    """Its parameters are instance values: runtime arguments, not constants."""
    from tack.runtime.dispatch import get_backend

    data = _two_hexes_and_a_pyramid()
    kernel = td.algorithms._implicit_values
    td.algorithms.implicit_values(data, td.Sphere((0, 0, 0), 1.0))
    variants = len(get_backend()._cache[kernel])
    for r in (0.5, 2.0, 3.0):
        values = td.algorithms.implicit_values(data, td.Sphere((1, 0, 0), r)).values.to_numpy()
        np.testing.assert_allclose(values, ((data.positions() - [1, 0, 0]) ** 2).sum(1) - r * r,
                                   rtol=1e-5, atol=1e-5)
    assert len(get_backend()._cache[kernel]) == variants


@needs_vtk
@pytest.mark.parametrize("kind", KINDS)
def test_no_point_is_near_a_zero(kind):
    """The comparisons above need every grid point clearly on one side."""
    points = vtk_to_numpy(_grid(kind).GetPoints().GetData())
    for name, (_, theirs) in FUNCTIONS.items():
        f = theirs()
        nearest = min(abs(f.EvaluateFunction(*p)) for p in points)
        assert nearest > 1e-3, f"{name} is {nearest} from a point of the {kind} grid"
