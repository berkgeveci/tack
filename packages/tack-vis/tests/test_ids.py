"""Ids: a topology's id_dtype, i32 unless asked for i64, given i64 fields, or too
large; never wrapped. The whole suite also runs with every topology in i64
(``pytest --ids=i64``); these are the rules themselves."""

import numpy as np
import pytest
from test_dataset_api import _scalars, _two_hexes_and_a_pyramid

import tack
import tack.data as td
from tack.data import ids

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None


def _tetra(points, id_dtype=None):
    return td.UnstructuredTopology([10], [0, 4], points, id_dtype=id_dtype)


def test_small_topologies_are_i32_unless_asked(backend, request):
    if request.config.getoption("--ids") == "i64":
        pytest.skip("every topology is i64 in this run")
    assert _tetra([0, 1, 2, 3]).id_dtype == tack.i32
    t = _tetra([0, 1, 2, 3], id_dtype=tack.i64)
    assert t.id_dtype == t.offsets.dtype == t.connectivity.dtype == tack.i64
    assert td.StructuredTopology((3, 3, 3)).id_dtype == tack.i32


def test_i64_fields_keep_their_type(backend):
    connectivity = tack.field(tack.i64, shape=(4,))
    connectivity.from_numpy(np.arange(4, dtype=np.int64))
    assert _tetra(connectivity).id_dtype == tack.i64


def test_ids_past_32_bits_are_kept_not_wrapped(backend):
    big = 2**31 + 5
    t = _tetra(np.array([0, 1, 2, big], np.int64))
    assert t.id_dtype == tack.i64 and t.num_points == big + 1
    assert t.connectivity.to_numpy()[3] == big
    with pytest.raises(ValueError, match="i32"):
        _tetra(np.array([0, 1, 2, big], np.int64), id_dtype=tack.i32)
    with pytest.raises(ValueError, match="do not fit i32"):
        ids.as_ids(np.array([big]), tack.i32)
    with pytest.raises(TypeError):
        _tetra([0, 1, 2, 3], id_dtype=tack.u32)


def test_derived_entities_and_filters_follow_the_topology(backend):
    data = _two_hexes_and_a_pyramid()
    wide = td.DataSet(td.UnstructuredTopology(data.topology.types, data.topology.offsets,
                                              data.topology.connectivity,
                                              id_dtype=tack.i64), data.geometry.values)
    faces, edges = wide.topology.faces(), wide.topology.edges()
    assert faces.rows.dtype == faces.side_face.dtype == edges.rows.dtype == tack.i64
    narrow_faces = data.topology.faces()
    np.testing.assert_array_equal(faces.rows.to_numpy(), narrow_faces.rows.to_numpy())
    np.testing.assert_array_equal(edges.rows.to_numpy(), data.topology.edges().rows.to_numpy())
    x = wide.positions()
    wide.fields["s"] = td.Field(td.H1(wide), _scalars(x[:, 0]))
    cut = td.contour(wide, "s", 0.7)
    assert cut.id_dtype == tack.i64 and cut.num_cells
    assert td.as_polyhedra(wide).id_dtype == tack.i64
    cells, _ = td.CellLocator(wide).find([[0.5, 0.5, 0.5], [9.0, 9.0, 9.0]])
    assert cells.dtype == tack.i64
    np.testing.assert_array_equal(cells.to_numpy(), [0, -1])


def test_outputs_widen_only_when_they_must():
    class Data:
        id_dtype = tack.i32

    assert ids.for_output(Data, 10) == tack.i32
    assert ids.for_output(Data, 2**31) == tack.i64
    Data.id_dtype = tack.i64
    assert ids.for_output(Data, 10) == tack.i64
    assert td.CountingArray(10).dtype == tack.i32
    assert td.CountingArray(10, start=2**31).dtype == tack.i64


@pytest.mark.skipif(vtk is None, reason="needs VTK")
def test_vtk_ids_take_the_type_asked_for(backend, request):
    source = vtk.vtkCellTypeSource()
    source.SetCellType(12)
    source.SetBlocksDimensions(2, 2, 2)
    source.Update()
    default = tack.i64 if request.config.getoption("--ids") == "i64" else tack.i32
    assert vtk_to_dataset(source.GetOutput()).id_dtype == default
    assert vtk_to_dataset(source.GetOutput(), id_dtype=tack.i64).id_dtype == tack.i64
