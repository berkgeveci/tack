"""Implicit functions and the filters that take them, against VTK: the functions'
values, slice (vtkCutter), extract_geometry (vtkExtractGeometry), and the
point and cell extractions (vtkExtractPoints, vtkThresholdPoints,
vtkMaskPoints, vtkExtractCells). On polyhedra the same as on shape cells."""

import numpy as np
import pytest
from test_dataset_api import _two_hexes_and_a_pyramid

import tack
import tack.dataset as td
from tack.dataset.algorithms.implicit_values import _implicit_values

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
    values = td.algorithms.implicit_values(data, ours()).values.numpy()
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
    values = out.fields["h"].values.numpy()
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
    kernel = _implicit_values
    td.algorithms.implicit_values(data, td.Sphere((0, 0, 0), 1.0))
    variants = len(get_backend()._cache[kernel])
    for r in (0.5, 2.0, 3.0):
        values = td.algorithms.implicit_values(data, td.Sphere((1, 0, 0), r)).values.numpy()
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


def test_outside_is_strictly_above_zero(backend):
    """Outside a region is above zero, strictly: points on the plane are not
    outside, and a cell with corners on it is not wholly outside. (The bound was
    the f64 number after zero, which is zero in an f32 kernel.)"""
    data = td.rectilinear_grid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    plane = td.Plane((0, 0, 0), (1, 0, 0))
    outside = td.extract_points(data, plane, inside=False)
    np.testing.assert_array_equal(outside.positions()[:, 0], 1.0)
    assert td.extract_points(data, plane).num_points == 4
    assert td.extract_geometry(data, plane, inside=False).num_cells == 0
    assert td.extract_geometry(data, plane, inside=False, boundary=True).num_cells == 1
    moved = td.Plane((-0.5, 0, 0), (1, 0, 0))
    assert td.extract_geometry(data, moved, inside=False).num_cells == 1


@tack.kernel
def _boundary_signs(values, out):
    for i in range(values.shape[0]):
        sign = 0
        if values[i] < 0.0:
            sign = -1
        if values[i] > 0.0:
            sign = 1
        out[i] = sign


def _check_zero_neighbors(dtype, record_property):
    fp = dtype.numpy_dtype.type
    neighbor = np.nextafter(fp(0), fp(1))
    normal = np.finfo(fp).tiny
    x = np.array([-1, -normal, -neighbor, -0., 0., neighbor, normal, 1], dtype=fp)
    pairs = np.array([[0, 1], [2, 3], [3, 4], [4, 5], [5, 6],
                      [6, 7], [2, 5], [0, 7], [1, 6]])
    top = td.UnstructuredTopology(np.full(len(pairs), td.shapes.LINE),
                                   np.arange(len(pairs) + 1) * 2, pairs.ravel(),
                                   num_points=len(x))
    data = td.DataSet(top, np.c_[x, np.zeros((len(x), 2), dtype=fp)], dtype=dtype)
    data.fields["point"] = td.point_ids(data)
    data.fields["cell"] = td.cell_ids(data)
    plane = td.Plane((0, 0, 0), (1, 0, 0))
    field = td.algorithms.implicit_values(data, plane)
    assert field.values.dtype is dtype

    # The plane is x. Only its subnormal neighbors may be flushed by device
    # arithmetic; normal values and both signed zeros have exact expectations.
    evaluated = field.values.numpy()
    ordinary = [0, 1, 3, 4, 6, 7]
    np.testing.assert_array_equal(evaluated[ordinary], x[ordinary])
    assert ((evaluated[2] == -neighbor and evaluated[5] == neighbor)
            or (evaluated[2] == 0 and evaluated[5] == 0))

    # Probe comparisons separately from the filters. Some devices preserve
    # subnormal storage but compare it as zero. Permit precisely that behavior,
    # and retain strict checks on zero and the smallest normal numbers.
    signs = tack.empty((len(x),), tack.i32)
    _boundary_signs(field.values, signs)
    compared = signs.numpy()
    np.testing.assert_array_equal(compared[ordinary], [-1, -1, 0, 0, 1, 1])
    neighbors = compared[[2, 5]].tolist()
    assert neighbors in ([-1, 1], [0, 0])
    flushed = neighbors == [0, 0]
    record_property("zero_neighbors", "flushed" if flushed else "preserved")

    # Literal entity ids are independent of the extraction/threshold kernels.
    inside_points = [0, 1, 2, 3, 4, 5] if flushed else [0, 1, 2, 3, 4]
    outside_points = [6, 7] if flushed else [5, 6, 7]
    zero_points = [2, 3, 4, 5] if flushed else [3, 4]
    cases = [(True, False, [0, 1, 2, 3, 6] if flushed else [0, 1, 2]),
             (True, True, [0, 1, 2, 3, 4, 6, 7, 8] if flushed else [0, 1, 2, 3, 6, 7, 8]),
             (False, False, [5] if flushed else [4, 5]),
             (False, True, [4, 5, 7, 8] if flushed else [3, 4, 5, 6, 7, 8])]
    for inside, expected in [(True, inside_points), (False, outside_points)]:
        out = td.extract_points(data, plane, inside=inside)
        np.testing.assert_array_equal(out.fields["point"].values.numpy(), expected)
    for inside, boundary, expected in cases:
        out = td.extract_geometry(data, plane, inside=inside, boundary=boundary)
        np.testing.assert_array_equal(out.fields["cell"].values.numpy(), expected)

    data.fields["x"] = field
    points = td.threshold_points(data, "x", 0., 0.)
    np.testing.assert_array_equal(points.fields["point"].values.numpy(), zero_points)
    zeros = td.threshold(data, "x", 0., 0.)
    np.testing.assert_array_equal(zeros.fields["cell"].values.numpy(),
                                  [1, 2, 3, 6] if flushed else [2])
    interval = td.threshold(data, "x", -float(normal), float(normal))
    np.testing.assert_array_equal(interval.fields["cell"].values.numpy(), [1, 2, 3, 4, 6, 8])


def test_zero_neighbors_f32(backend, record_property):
    _check_zero_neighbors(tack.f32, record_property)


def test_zero_neighbors_f64(f64_backend, record_property):
    _check_zero_neighbors(tack.f64, record_property)
