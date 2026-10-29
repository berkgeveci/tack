"""Viskores' test datasets (``_viskores_datasets``): small meshes with what real ones
carry and generated ones do not -- inside-out cells, cells of every dimension in
one mesh, pyramids and wedges beside hexahedra, polygons. Measures held to
Viskores' own expected values (``UnitTestCellMeasuresFilter``), filters to VTK's,
and every algorithm run on every dataset, to an answer or a stated refusal."""

import pathlib
import re

import _viskores_datasets as vd
import numpy as np
import pytest
import test_dataset_devices

import tack
import tack.dataset as td

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401
except ImportError:
    have_vtk = False
else:
    have_vtk = True

needs_vtk = pytest.mark.skipif(not have_vtk, reason="needs VTK, the reference")


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    """Compared in f64, on every backend that has f64."""
    yield f64_backend
    tack.set_default_device("cpu")


def _fields(m):
    """``m`` as a Tack dataset with ``s`` (the squared distance from the origin) on
    points and ``c`` (the cell's index) on cells; and ``s`` and ``c``."""
    data = vd.to_tack(m)
    x = data.positions().astype(np.float64)
    s = (x ** 2).sum(1)
    c = np.arange(data.num_cells, dtype=np.float64)
    data.fields["s"] = td.Field(td.H1(data), tack.array(s))
    data.fields["c"] = td.Field(td.Constant(data), tack.array(c))
    return data, s, c


def _with_fields(m):
    """``_fields(m)``, and ``m`` as a VTK dataset with the same fields."""
    data, s, c = _fields(m)
    vtk_data = vd.to_vtk(m)
    vtk_data.point_data["s"] = s.copy()
    vtk_data.cell_data["c"] = c.copy()
    return data, vtk_data, s


def _measure(data):
    return float(td.cell_geometry(data).measure.values.numpy().sum()) if data.num_cells else 0.0


def _vtk_measure(vtk_data):
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    sizes = vtkCellSizeFilter()(vtk_data).cell_data
    return sum(float(np.asarray(sizes[n]).sum()) for n in ("Length", "Area", "Volume")
               if n in sizes.keys())


@pytest.mark.quick
@pytest.mark.parametrize("name, expected", [
    ("3d_explicit_2", [-1.0]),
    ("3d_explicit_3", [-1.0 / 6.0]),
    ("3d_explicit_4", [-1.0, -1.0]),
    ("3d_explicit_5", [1.0, 1.0 / 3.0, 1.0 / 6.0, -0.5]),
    # Lines, vertices, a triangle, a quad and tetrahedra in one mesh.
    ("3d_explicit_6", [0.999924, 0.999924, 0.0, 0.0, 3.85516, 1.00119, 0.083426, 0.25028]),
])
def test_cell_measures_are_viskores(name, expected):
    """Inside-out cells measure negative, as Viskores' and VTK's do; and a mesh of
    every dimension measures each cell in its own (2D cells among tetrahedra
    once had no sides to walk)."""
    measure = td.cell_geometry(vd.to_tack(vd.ALL[name]())).measure.values.numpy()
    np.testing.assert_allclose(measure, expected, rtol=1e-5, atol=1e-12)


@pytest.mark.quick
def test_gradients_of_2d_cells_are_in_their_plane():
    """``UnitTestGradientExplicit``: a triangle and a quad's gradient of
    ``pointvar``, in their plane."""
    data = vd.to_tack(vd.make_3d_explicit_0())
    np.testing.assert_allclose(td.gradients(data, "pointvar").values.numpy(),
                               [[10.0, 10.1, 0.0]] * 2, rtol=1e-12)


@pytest.mark.quick
def test_fluxes_skip_cells_of_a_lower_dimension():
    """A surface's triangle among tetrahedra has none of the mesh's faces: no net
    flux and no reconstruction, zero, as gradients give it."""
    data = vd.to_tack(vd.make_3d_explicit_6())
    faces = data.topology.faces().num_faces
    flux = td.Field(td.Values(data, "faces"), tack.ones((faces,), tack.f64))
    lower = np.array(vd.make_3d_explicit_6().types) != vd.TETRA
    assert (td.net_flux(data, flux).values.numpy()[lower] == 0.0).all()
    assert (td.perot(data, flux).values.numpy()[lower] == 0.0).all()


@pytest.mark.quick
def test_as_polyhedra_refuses_inside_out_neighbours():
    """An inside-out cell beside an upright one goes round their face the same way,
    which no polyhedral face can record; refused as that, not later as faces
    that do not close. Cells all inside out are consistent and convert."""
    with pytest.raises(ValueError, match="wound the same way by both their cells"):
        td.as_polyhedra(vd.to_tack(vd.make_3d_explicit_5()))
    data, s, _ = _fields(vd.make_3d_explicit_4())
    polyhedra = td.as_polyhedra(data)
    np.testing.assert_allclose(td.cell_geometry(polyhedra).measure.values.numpy(), [-1, -1])
    level = float(np.median(s))
    assert _measure(td.clip(polyhedra, "s", level)) == pytest.approx(
        _measure(td.clip(data, "s", level)), rel=1e-12)


_DATA = pathlib.Path(__file__).parent / "data" / "viskores"


def _read(name):
    from vtkmodules.vtkIOLegacy import vtkDataSetReader

    return vtkDataSetReader(file_name=str(_DATA / name))()


@pytest.mark.quick
@needs_vtk
def test_contours_of_viskores_files():
    """``UnitTestContourFilter``'s files. Seven values over the 3D cells of one of
    every shape (Viskores contours a mesh's highest dimension, as Tack does): 10
    triangles, VTK's count and points; wedges at the gyroid's zero: 52."""
    from vtkmodules.vtkFiltersCore import vtkContourFilter
    from vtkmodules.vtkFiltersExtraction import vtkExtractCellsByType

    from tack.interop.vtk import vtk_to_dataset

    solids = vtkExtractCellsByType()
    for shape in (vd.TETRA, 11, vd.HEXAHEDRON, vd.WEDGE, vd.PYRAMID):
        solids.AddCellType(shape)
    solids = solids(_read("mixed-cell-shapes.vtk"))
    values = [5.5, 9.5, 11.5, 14.5, 17.5, 20.5, 25.5]
    with pytest.warns(UserWarning, match="point array and a cell array named 'scalars'"):
        data = vtk_to_dataset(solids)
    ours = td.contour(data, "scalars", values)
    f = vtkContourFilter()
    for i, value in enumerate(values):
        f.SetValue(i, value)
    f.SetInputArrayToProcess(0, 0, 0, 0, "scalars")
    theirs = f(solids)
    assert (ours.num_cells, ours.num_points) == (10, theirs.number_of_points) == \
        (theirs.number_of_cells, theirs.number_of_points)
    assert _measure(ours) == pytest.approx(_vtk_measure(theirs), rel=1e-6)
    assert td.contour(vtk_to_dataset(_read("wedge_cells.vtk")), "gyroid", 0.0).num_cells == 52


@pytest.mark.quick
@needs_vtk
def test_contours_by_cell_dimension():
    """``TestMixedShapes``' input cell dimensions, on the shapes of
    ``mixed-cell-shapes.vtk`` Tack holds in one topology (its polygon, strip and
    polyline left out): each dimension's cells contoured alone, as VTK contours
    those cells -- triangles, lines, vertices -- never mixed; the default is
    the highest."""
    from vtkmodules.vtkFiltersCore import vtkContourFilter
    from vtkmodules.vtkFiltersExtraction import vtkExtractCellsByType

    from tack.interop.vtk import vtk_to_dataset

    by_dimension = {3: (vd.TETRA, 11, vd.HEXAHEDRON, vd.WEDGE, vd.PYRAMID),
                    2: (vd.TRIANGLE, 8, vd.QUAD), 1: (vd.LINE,), 0: (vd.VERTEX,)}
    grid = _read("mixed-cell-shapes.vtk")

    def cells(*dimensions):
        e = vtkExtractCellsByType()
        for d in dimensions:
            for shape in by_dimension[d]:
                e.AddCellType(shape)
        return e(grid)

    with pytest.warns(UserWarning, match="named 'scalars'"):
        data = vtk_to_dataset(cells(3, 2, 1, 0))
    values = [5.5, 9.5, 11.5, 14.5, 17.5, 20.5, 25.5]
    f = vtkContourFilter()
    for i, value in enumerate(values):
        f.SetValue(i, value)
    f.SetInputArrayToProcess(0, 0, 0, 0, "scalars")
    shapes = {3: {vd.TRIANGLE}, 2: {vd.LINE}, 1: {vd.VERTEX}}
    for dimension in (3, 2, 1):
        ours = td.contour(data, "scalars", values, dimension=dimension)
        theirs = f(cells(dimension))
        assert ours.num_cells == theirs.number_of_cells > 0, dimension
        assert set(np.unique(ours.topology.types.numpy())) == shapes[dimension]
        assert _measure(ours) == pytest.approx(_vtk_measure(theirs), rel=1e-6)
    assert td.contour(data, "scalars", values).num_cells == f(cells(3)).number_of_cells
    with pytest.raises(ValueError, match="3, 2 or 1"):
        td.contour(data, "scalars", values, dimension=0)


@pytest.mark.quick
@needs_vtk
def test_a_point_and_a_cell_array_of_one_name_both_arrive():
    """VTK keeps point and cell arrays apart; a dataset's fields share one name
    space. The cell array of ``mixed-cell-shapes.vtk``'s ``scalars`` once
    replaced the point array, and a contour of it found nothing."""
    from tack.interop.vtk import vtk_to_dataset

    grid = vd.to_vtk(vd.make_3d_explicit_5())
    grid.point_data["v"] = np.arange(grid.number_of_points, dtype=np.float64)
    grid.cell_data["v"] = -np.arange(grid.number_of_cells, dtype=np.float64)
    with pytest.warns(UserWarning, match="the cell array is the field 'v \\(cells\\)'"):
        data = vtk_to_dataset(grid)
    assert isinstance(data.fields["v"].space, td.H1)
    np.testing.assert_array_equal(data.fields["v"].values.numpy(), np.arange(11))
    np.testing.assert_array_equal(data.fields["v (cells)"].values.numpy(), [0, -1, -2, -3])


#: Where Tack differs from VTK by design: in a mesh with 3D cells,
#: lower-dimensional cells add nothing to a contour or a surface (VTK's add
#: theirs); and clip takes no polygons yet.
_DESIGN = {
    "surface": {"3d_explicit_6"},
    "contour": {"3d_explicit_6"},
    "clip": {"2d_explicit_0", "3d_explicit_polygonal"},
}


@pytest.mark.quick
@needs_vtk
@pytest.mark.parametrize("name", vd.ALL)
def test_filters_are_vtks(name):
    """Contour and clip measures, threshold counts, cell centers and averages to and
    from points, against VTK's filters."""
    from vtkmodules.vtkFiltersCore import (
        vtkCellCenters,
        vtkCellDataToPointData,
        vtkContourFilter,
        vtkPointDataToCellData,
        vtkThreshold,
    )
    from vtkmodules.vtkFiltersGeneral import vtkTableBasedClipDataSet
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    data, vtk_data, s = _with_fields(vd.ALL[name]())
    level = float(np.quantile(s, 0.37)) + 1e-4          # off the points' values
    if name not in _DESIGN["contour"]:
        f = vtkContourFilter(compute_scalars=False)
        f.SetInputArrayToProcess(0, 0, 0, 0, "s")
        f.SetValue(0, level)
        assert _measure(td.contour(data, "s", level)) == pytest.approx(_vtk_measure(f(vtk_data)),
                                                                       rel=1e-6, abs=1e-12)
    if name not in _DESIGN["clip"]:
        f = vtkTableBasedClipDataSet(value=level)
        f.SetInputArrayToProcess(0, 0, 0, 0, "s")
        kept, _ = f(vtk_data)                     # the clipped output too: two outputs
        assert _measure(td.clip(data, "s", level)) == pytest.approx(_vtk_measure(kept),
                                                                    rel=1e-6, abs=1e-12)
    if name not in _DESIGN["surface"]:
        theirs = vtkGeometryFilter()(vtk_data)
        ours = td.extract_surface(data)
        assert ours.num_cells == theirs.number_of_cells
        assert _measure(ours) == pytest.approx(_vtk_measure(theirs), rel=1e-12)
    high = data.num_cells / 2
    f = vtkThreshold(lower_threshold=1.0, upper_threshold=high,
                     threshold_function=vtkThreshold.THRESHOLD_BETWEEN)
    f.SetInputArrayToProcess(0, 0, 0, 1, "c")
    assert td.threshold(data, "c", 1.0, high).num_cells == f(vtk_data).number_of_cells

    centers = vtkCellCenters(vertex_cells=False)(vtk_data)
    np.testing.assert_allclose(td.cell_centers(data).values.numpy(),
                               [centers.GetPoint(i) for i in range(centers.number_of_points)],
                               rtol=1e-6, atol=1e-12)                # VTK's are f32
    np.testing.assert_allclose(td.project(data, "s", td.Constant(data)).values.numpy(),
                               np.asarray(vtkPointDataToCellData()(vtk_data).cell_data["s"]),
                               rtol=1e-12)
    np.testing.assert_allclose(td.project(data, "c", td.H1(data)).values.numpy(),
                               np.asarray(vtkCellDataToPointData()(vtk_data).point_data["c"]),
                               rtol=1e-12)


#: The refusals an algorithm may give on these datasets: each names what it does
#: not take. Anything else -- an error from inside a kernel's resolution, say --
#: is a failure.
_REFUSALS = re.compile("|".join([
    "needs a reference element",                 # polygons: no traces, no order 2
    "are not 3D",                                # as_polyhedra of 2D or 0D cells
    "is not supported yet",                      # order 2 on pyramids and vertices
    "takes cells of the linear shapes",          # tetrahedralize, clip of polygons
    "wound the same way by both their cells",    # inside-out neighbours as polyhedra
    "cell ids lie outside",                      # extract_cells([0, 1]) of one cell
    "tetrahedra and wedges; not yet on",         # subdivide of other shapes
    "has the equispaced basis only",             # simplices above order 2, another basis
    "or all tensor shapes",                      # subdivide of a mix of families
]))


@pytest.mark.parametrize("name", vd.ALL)
def test_every_algorithm_answers_or_refuses(name):
    """Every algorithm of the device tests on each dataset: an answer, or one of the
    stated refusals. Mixed dimensions once failed inside cell_geometry, net_flux
    and perot ("no @tack.func method 'side_position'")."""
    data, _, _ = _fields(vd.ALL[name]())
    x = data.positions().astype(np.float64)
    data.fields["v"] = td.Field(td.H1(data), tack.array(x, tack.vec3(tack.f64)))
    data.sets["corner"] = td.Set(data, "cells", [0])
    failures = []
    for call, run in test_dataset_devices.CALLS.items():
        try:
            run(data)
        except Exception as e:                   # sorted below
            if not _REFUSALS.search(str(e)):
                failures.append(f"{call}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)
