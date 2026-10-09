"""Clip by Viskores' case tables, against VTK's vtkTableBasedClipDataSet: cells,
points, measure and orientation on every linear shape, by implicit functions
and by fields, both ways; fields carried onto the new points and pieces."""

import numpy as np
import pytest
from test_dataset_api import _scalars, _two_hexes_and_a_pyramid

import tack.data as td

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

SHAPES = {"hexahedron": 12, "tetra": 10, "wedge": 13, "pyramid": 14, "voxel": 11,
          "quad": 9, "pixel": 8, "triangle": 5}
FUNCTIONS = {"sphere": lambda: td.Sphere((1.4, 1.6, 0.9), 1.3),
             "plane": lambda: td.Plane((1.3, 1.1, 0.7), (1, 0.4, -0.3))}


def _grid(kind):
    source = vtk.vtkCellTypeSource()
    source.SetCellType(SHAPES[kind])
    source.SetBlocksDimensions(3, 3, 2 if SHAPES[kind] >= 10 else 0)
    source.Update()
    return source.GetOutput()


def _sizes(grid, what):
    sizes = vtk.vtkCellSizeFilter()
    sizes.SetInputData(grid)
    sizes.Update()
    return vtk_to_numpy(sizes.GetOutput().GetCellData().GetArray(what))


def _vtk_clip(grid, values, value, invert):
    copy = vtk.vtkUnstructuredGrid()
    copy.DeepCopy(grid)
    array = numpy_to_vtk(np.asarray(values, np.float32), deep=True)
    array.SetName("clip")
    copy.GetPointData().SetScalars(array)
    clip = vtk.vtkTableBasedClipDataSet()
    clip.SetInputData(copy)
    clip.SetValue(value)
    clip.SetInsideOut(invert)
    clip.Update()
    return clip.GetOutput()


def _check_like_vtk(out, reference, what):
    assert (out.num_cells, out.num_points) == (reference.GetNumberOfCells(),
                                               reference.GetNumberOfPoints())
    ours = _sizes(dataset_to_vtk(out), what)
    assert (ours > 0).all(), "every piece the right way out"
    assert ours.sum() == pytest.approx(_sizes(reference, what).sum(), rel=1e-5)
    np.testing.assert_array_equal(
        np.bincount(vtk_to_numpy(dataset_to_vtk(out).GetCellTypes()), minlength=15),
        np.bincount(vtk_to_numpy(reference.GetCellTypes()), minlength=15))


@needs_vtk
@pytest.mark.parametrize("kind", SHAPES)
@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("invert", [False, True])
def test_clip_by_a_function_is_vtks(backend, kind, name, invert):
    grid = _grid(kind)
    data = vtk_to_dataset(grid)
    f = FUNCTIONS[name]()
    values = td.algorithms.implicit_values(data, f).values.to_numpy()
    out = td.clip(data, f, invert=invert)
    what = "Volume" if SHAPES[kind] >= 10 else "Area"
    _check_like_vtk(out, _vtk_clip(grid, values, 0.0, invert), what)


@needs_vtk
@pytest.mark.parametrize("kind", ["hexahedron", "wedge", "tetra"])
def test_clip_by_a_field_carries_fields(backend, kind):
    """A linear point field is exact at every new point -- edge crossings and the
    centroid points some cases add -- and each piece has its cell's value."""
    grid = _grid(kind)
    data = vtk_to_dataset(grid)
    x = data.positions()
    data.fields["s"] = td.Field(td.H1(data), _scalars(np.sin(x[:, 0]) + x[:, 1] * x[:, 2]))
    linear = lambda p: 2.0 * p[:, 0] - 0.5 * p[:, 1] + 0.25 * p[:, 2]
    data.fields["lin"] = td.Field(td.H1(data), _scalars(linear(x)))
    data.fields["id"] = td.transforms.cell_ids(data)
    out = td.clip(data, "s", 1.1)
    reference = _vtk_clip(grid, data.fields["s"].values.to_numpy(), 1.1, False)
    _check_like_vtk(out, reference, "Volume")
    np.testing.assert_allclose(out.fields["lin"].values.to_numpy(), linear(out.positions()),
                               atol=1e-5)
    values = out.fields["s"].values.to_numpy()
    assert values.min() >= 1.1 - 1e-5            # the kept side, edge points at the value
    # Every piece lies inside the cell it came from.
    centers = td.algorithms.cell_centers(data).values.to_numpy(vectors=True)
    ids = np.asarray(td.arrays.to_host(out.fields["id"].values)).astype(int)
    pieces = out.positions()[out.topology.connectivity.to_numpy()]
    offsets = out.topology.offsets.to_numpy()
    for k in range(0, out.num_cells, 7):
        corners = pieces[offsets[k]:offsets[k + 1]].mean(axis=0)
        assert np.linalg.norm(corners - centers[ids[k]]) < 1.0


def test_clip_of_a_mixed_mesh_keeps_its_volume(backend):
    data = _two_hexes_and_a_pyramid()
    f = td.Plane((1.0, 0.5, 0.5), (1, 0.2, 0.1))
    kept, other = td.clip(data, f), td.clip(data, f, invert=True)
    total = td.algorithms.cell_geometry(data)[0].values.to_numpy().sum()
    both = sum(td.algorithms.cell_geometry(td.as_polyhedra(part))[0].values.to_numpy().sum()
               for part in (kept, other))
    assert both == pytest.approx(total, rel=1e-5)


def test_clip_refuses_what_it_cannot_cut(backend):
    data = _two_hexes_and_a_pyramid()
    with pytest.raises(NotImplementedError):
        td.clip(td.as_polyhedra(data), td.Plane((1, 0, 0), (1, 0, 0)))
    data.fields["c"] = td.Field(td.Constant(data), _scalars([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match="point data"):
        td.clip(data, "c", 1.5)
