"""Viskores' test datasets (``_viskores_datasets``): small meshes with what real ones
carry and generated ones do not -- inside-out cells, cells of every dimension in
one mesh, pyramids and wedges beside hexahedra, polygons. Measures held to
Viskores' own expected values (``UnitTestCellMeasuresFilter``), filters to VTK's,
and every algorithm run on every dataset, to an answer or a stated refusal."""

import re

import _viskores_datasets as vd
import numpy as np
import pytest
import test_dataset_devices

import tack
import tack.dataset as td


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    """Compared in f64, on every backend that has f64."""
    yield f64_backend
    tack.set_default_device("cpu")


def _with_fields(m):
    """``m`` as a Tack dataset and a VTK one, each with ``s`` (the squared distance
    from the origin) on points and ``c`` (the cell's index) on cells."""
    data, vtk_data = vd.to_tack(m), vd.to_vtk(m)
    x = data.positions().astype(np.float64)
    s = (x ** 2).sum(1)
    c = np.arange(data.num_cells, dtype=np.float64)
    data.fields["s"] = td.Field(td.H1(data), tack.array(s))
    data.fields["c"] = td.Field(td.Constant(data), tack.array(c))
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
    data, _, s = _with_fields(vd.make_3d_explicit_4())
    polyhedra = td.as_polyhedra(data)
    np.testing.assert_allclose(td.cell_geometry(polyhedra).measure.values.numpy(), [-1, -1])
    level = float(np.median(s))
    assert _measure(td.clip(polyhedra, "s", level)) == pytest.approx(
        _measure(td.clip(data, "s", level)), rel=1e-12)


#: Where Tack differs from VTK by design: a 2D mesh's surface is its boundary
#: edges (VTK's geometry filter passes the polygons); in a mesh with 3D cells,
#: lower-dimensional cells add nothing to a contour or a surface; and clip
#: takes no polygons yet.
_DESIGN = {
    "surface": {"2d_uniform_0", "2d_uniform_1", "2d_rectilinear_0", "2d_rectilinear_1",
                "2d_explicit_0", "3d_explicit_0", "3d_explicit_1", "3d_explicit_6",
                "3d_explicit_polygonal"},
    "contour": {"3d_explicit_6"},
    "clip": {"2d_explicit_0", "3d_explicit_polygonal"},
}


@pytest.mark.quick
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
]))


@pytest.mark.parametrize("name", vd.ALL)
def test_every_algorithm_answers_or_refuses(name):
    """Every algorithm of the device tests on each dataset: an answer, or one of the
    stated refusals. Mixed dimensions once failed inside cell_geometry, net_flux
    and perot ("no @tack.func method 'side_position'")."""
    data, _, _ = _with_fields(vd.ALL[name]())
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
