"""Implicit topology arrays: an ``UnstructuredTopology`` whose types, offsets or
connectivity are implicit arrays (``ConstantArray``, ``CountingArray``) keeps them
implicit, behaves exactly as one that stores the same values, is what filters
whose cells are all one shape give, and crosses to and from VTK as VTK's
constant and affine arrays.

Each dataset is checked against its stored twin -- the same topology with its
three arrays materialized -- through the derived entities and the algorithms
that read the cells: they must agree value for value.
"""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.arrays import ConstantArray, CountingArray, materialize
from tack.dataset.topology import uniform_cells
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import env_flag

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

try:
    from vtkmodules.vtkCommonCore import vtkAffineArray, vtkConstantArray, vtkPoints
    from vtkmodules.vtkCommonDataModel import VTK_TETRA, vtkCellArray, vtkUnstructuredGrid

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    HAVE_VTK = False
else:
    HAVE_VTK = True

needs_vtk = pytest.mark.skipif(not HAVE_VTK, reason="needs VTK, the reference")


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")


def _kinds(topology):
    return tuple(type(a).__name__ for a in
                 (topology.types, topology.offsets, topology.connectivity))


def _stored(data):
    """``data`` on a topology storing the same three arrays, with every field and
    set rebuilt there."""
    t = data.topology
    twin = td.UnstructuredTopology(materialize(t.types), materialize(t.offsets),
                                   materialize(t.connectivity), num_points=t.num_points,
                                   id_dtype=t.id_dtype)
    assert _kinds(twin) == ("Array", "Array", "Array")
    out = td.DataSet(twin, data.geometry.values)
    for name, field in data.fields.items():
        out.fields[name] = td.Field(field.space._rebuilt_on(twin), field.values)
    return out


def _host(values):
    return np.asarray(tack.arrays.to_host(values))


# ── The topology ────────────────────────────────────────────────────

def test_a_topology_keeps_its_implicit_arrays():
    types, offsets = uniform_cells(td.shapes.TRIANGLE, 2, tack.i32)
    t = td.UnstructuredTopology(types, offsets, tack.array([0, 1, 2, 1, 3, 2], dtype=tack.i32))
    assert _kinds(t) == ("ConstantArray", "CountingArray", "Array")
    # i32 ids, or i64 under the suite's --ids=i64; the offsets follow, still counting.
    assert (t.num_cells, t.num_points, t.id_dtype) == (2, 4, td.ids.choose())
    assert t.offsets.dtype == t.id_dtype
    assert [(g.shape.__name__, g.count) for g in t.groups()] == [("Triangle", 2)]
    assert (t.faces().num_faces, t.edges().num_edges) == (5, 5)
    np.testing.assert_array_equal(np.asarray(t.offsets), [0, 3, 6])

    vertices = td.UnstructuredTopology(ConstantArray(1, 3, dtype=tack.u8), CountingArray(4),
                                       CountingArray(3))
    assert _kinds(vertices) == ("ConstantArray", "CountingArray", "CountingArray")
    assert vertices.num_points == 3                     # one past the highest id
    points = td.DataSet(vertices, np.eye(3, dtype=np.float32))
    assert td.L2(points).size == 3                       # sized by the implicit connectivity
    # Types that are implicit but not constant go through the type histogram.
    cast = td.UnstructuredTopology(tack.arrays.Cast(CountingArray(2, start=5, step=0), tack.u8),
                                   CountingArray(3, step=3), CountingArray(6))
    assert [(g.shape.__name__, g.count) for g in cast.groups()] == [("Triangle", 2)]

    wide = td.UnstructuredTopology(ConstantArray(1, 3, dtype=tack.u8),
                                   CountingArray(4, dtype=tack.i64),
                                   CountingArray(3, dtype=tack.i64), num_points=3)
    assert wide.id_dtype == tack.i64
    narrow = td.UnstructuredTopology(ConstantArray(1, 3, dtype=tack.u8),
                                     CountingArray(4, dtype=tack.i64), CountingArray(3),
                                     num_points=3, id_dtype=tack.i32)
    assert narrow.offsets.dtype == tack.i32             # remade, still counting
    assert _kinds(narrow)[1:] == ("CountingArray", "CountingArray")
    np.testing.assert_array_equal(np.asarray(narrow.offsets), [0, 1, 2, 3])


def test_a_topology_refuses_implicit_arrays_that_do_not_fit():
    with pytest.raises(TypeError, match="expected a u8 field"):
        td.UnstructuredTopology(ConstantArray(5, 2), CountingArray(3, step=3),
                                CountingArray(6))
    with pytest.raises(ValueError, match="num_cells \\+ 1 = 3"):
        td.UnstructuredTopology(ConstantArray(5, 2, dtype=tack.u8), CountingArray(4, step=3),
                                CountingArray(6))
    with pytest.raises(ValueError, match="not linear shapes"):
        td.UnstructuredTopology(ConstantArray(99, 2, dtype=tack.u8),
                                CountingArray(3, step=3), CountingArray(6)).groups()


# ── Filters give them ───────────────────────────────────────────────

def test_filters_whose_cells_are_one_shape_give_implicit_arrays():
    w = td.wavelet(5)
    surface = td.extract_surface(w)
    implicit_types_and_offsets = {
        "contour": td.contour(w, "RTData", 150.0),
        "extract_edges": td.extract_edges(w),
        "tetrahedralize": td.tetrahedralize(w),
        "triangulate": td.triangulate(surface),
        "threshold of one shape": td.threshold(w, "RTData", 100.0, 200.0),
    }
    for name, out in implicit_types_and_offsets.items():
        assert _kinds(out.topology)[:2] == ("ConstantArray", "CountingArray"), name
        assert out.num_cells > 0, name
    every_one_its_own = {
        "mask_points": td.mask_points(w, 7),
        "threshold_points": td.threshold_points(w, "RTData", 100.0, 150.0),
        "extract_points": td.extract_points(w, td.Sphere((0.0, 0.0, 0.0), 3.0)),
        "probe": td.probe(w, np.zeros((4, 3))),
    }
    for name, out in every_one_its_own.items():
        assert _kinds(out.topology) == ("ConstantArray", "CountingArray", "CountingArray"), name
    # A 2D mesh's contour: lines, implicit types and offsets.
    assert _kinds(_lines_2d().topology)[:2] == ("ConstantArray", "CountingArray")
    # Mixed shapes stay stored.
    mixed = td.threshold(_mixed(), "c", 0, 100)
    assert mixed.num_cells == 2
    assert _kinds(mixed.topology)[:2] == ("Array", "Array")


def _lines_2d():
    grid = td.uniform_grid((9, 7), origin=(-2.0, -1.5, 0.0), spacing=(0.5, 0.5, 1.0))
    x = grid.positions()
    grid.fields["r2"] = td.Field(td.H1(grid), tack.array((x * x).sum(axis=1), dtype=tack.f32))
    return td.contour(grid, "r2", 1.0)


def _mixed():
    """A hexahedron and a tetrahedron: two shapes."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
                    [1, 1, 1], [0, 1, 1], [2, 0, 0]], np.float32)
    t = td.UnstructuredTopology([12, 10], [0, 8, 12], [0, 1, 2, 3, 4, 5, 6, 7, 1, 8, 2, 5])
    data = td.DataSet(t, pts)
    data.fields["c"] = td.Field(td.Constant(data), tack.array(np.arange(2, dtype=np.int32)))
    return data


# ── They behave as stored arrays do ─────────────────────────────────

def _cases():
    w = td.wavelet(4)
    tets = td.tetrahedralize(w)
    surface = td.contour(w, "RTData", 150.0)
    return {"tetrahedra": tets, "contour": surface}


@pytest.mark.parametrize("name", ["tetrahedra", "contour"])
def test_an_implicit_topology_computes_as_its_stored_twin_does(name):
    data = _cases()[name]
    assert _kinds(data.topology)[:2] == ("ConstantArray", "CountingArray")
    twin = _stored(data)
    a, b = data.topology, twin.topology
    assert [(g.shape, g.count) for g in a.groups()] == [(g.shape, g.count) for g in b.groups()]
    assert a.faces().num_faces == b.faces().num_faces
    assert a.edges().num_edges == b.edges().num_edges
    for compute in (lambda d: td.cell_centers(d).values,
                    lambda d: td.gradients(d, "RTData").values,
                    lambda d: td.project(d, "RTData", td.L2(d)).values,
                    lambda d: td.project(d, "RTData", td.Constant(d)).values,
                    lambda d: td.face_geometry(d)[1].values):          # face areas
        np.testing.assert_allclose(_host(compute(data)), _host(compute(twin)), rtol=1e-6)
    for cut in (lambda d: td.threshold(d, "RTData", 140.0, 220.0),
                lambda d: td.extract_surface(d),
                lambda d: td.slice_plane(d, (0.3, 0.2, 0.1), (1.0, 0.5, 0.2))):
        x, y = cut(data), cut(twin)
        assert x.num_cells == y.num_cells
        np.testing.assert_allclose(x.positions(), y.positions(), rtol=1e-6)
    if name == "tetrahedra":
        data.fields["grad"] = td.gradients(data, "RTData", at="points")
        twin.fields["grad"] = td.gradients(twin, "RTData", at="points")
        seeds = np.array([[0.5, 0.2, 0.1], [1.0, -1.0, 2.0], [-2.0, 1.5, -0.5]])
        x = td.advect(data, "grad", seeds, 0.002, 30)
        y = td.advect(twin, "grad", seeds, 0.002, 30)
        np.testing.assert_allclose(x.positions(), y.positions(), rtol=1e-6)


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's, as ``test_dataset_devices.py`` makes."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


def test_an_implicit_topology_moves_to_another_device():
    second = dispatch.as_device(_SecondCPU())
    data = td.tetrahedralize(td.wavelet(3))
    moved = data.to_device(second)
    assert moved.device == second
    assert _kinds(moved.topology)[:2] == ("ConstantArray", "CountingArray")
    np.testing.assert_allclose(_host(td.cell_centers(moved).values),
                               _host(td.cell_centers(data).values), rtol=1e-6)


# ── VTK ─────────────────────────────────────────────────────────────

@needs_vtk
def test_implicit_arrays_cross_to_vtk_and_back():
    w = td.wavelet(4)
    for out, vtk_kinds, back_kinds in (
            (td.contour(w, "RTData", 150.0), ("Constant", "Affine", None),
             ("ConstantArray", "CountingArray", "Array")),
            (td.mask_points(w, 5), ("Constant", "Affine", "Affine"),
             ("ConstantArray", "CountingArray", "CountingArray")),
            (_mixed(), (None, None, None), ("Array", "Array", "Array"))):
        grid = dataset_to_vtk(out)
        arrays = (grid.cell_types, grid.cells.offsets_array, grid.cells.connectivity_array)
        for array, kind in zip(arrays, vtk_kinds):
            name = array.GetClassName()
            assert (kind in name) if kind else ("Constant" not in name and "Affine" not in name)
        back = vtk_to_dataset(grid)
        assert _kinds(back.topology) == back_kinds
        for x, y in zip((back.topology.types, back.topology.offsets, back.topology.connectivity),
                        (out.topology.types, out.topology.offsets, out.topology.connectivity)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        np.testing.assert_allclose(back.positions(), out.positions())


@needs_vtk
def test_a_vtk_grid_built_with_implicit_arrays_stays_implicit():
    """Built the pythonic way: a cell type given as a number, affine offsets."""
    grid = vtkUnstructuredGrid()
    grid.points = vtkPoints(data=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    grid.cells = (VTK_TETRA, vtkCellArray(offsets=vtkAffineArray[np.int64](3, 4, 0),
                                          connectivity=[0, 1, 2, 3, 1, 2, 3, 4]))
    data = vtk_to_dataset(grid)
    assert _kinds(data.topology) == ("ConstantArray", "CountingArray", "Array")
    assert [(g.shape.__name__, g.count) for g in data.topology.groups()] == [("Tetra", 2)]
    explicit = vtkConstantArray[np.uint8](2, VTK_TETRA)
    grid.cells = (explicit, grid.cells)
    assert _kinds(vtk_to_dataset(grid).topology)[0] == "ConstantArray"
