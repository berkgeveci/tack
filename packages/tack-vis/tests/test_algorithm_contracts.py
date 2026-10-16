"""Shared contracts at algorithm boundaries and through empty pipelines."""

import numpy as np
import pytest

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

FIELD_OPERATIONS = {
    "threshold": lambda d, f: td.threshold(d, f, 0, 1),
    "threshold_points": lambda d, f: td.threshold_points(d, f, 0, 1),
    "contour": lambda d, f: td.contour(d, f, 0.5),
    "clip": lambda d, f: td.clip(d, f, 0.5),
    "project": lambda d, f: td.project(d, f, td.Constant(d)),
    "statistics": td.statistics,
    "log_values": td.log_values,
    "gradients": td.gradients,
    "values_at_centers": td.values_at_centers,
}


@pytest.mark.parametrize("name", FIELD_OPERATIONS)
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_algorithms_refuse_fields_of_another_topology(backend, name, empty, named):
    def mesh():
        return (td.DataSet(td.UnstructuredTopology([], [0], [], num_points=0),
                           np.empty((0, 3))) if empty else td.uniform_grid((2, 2, 2)))

    data, other = mesh(), mesh()
    foreign = td.Field(td.H1(other), tack.arrays.ConstantArray(0.5, other.num_points))
    # A direct dict assignment must not bypass the check at the algorithm boundary.
    data.fields["foreign"] = foreign
    with pytest.raises(ValueError, match="another topology"):
        FIELD_OPERATIONS[name](data, "foreign" if named else foreign)


def test_a_raw_device_array_is_not_a_dataset_field(backend):
    data = td.uniform_grid((2, 2, 2))
    with pytest.raises(TypeError, match="tack.dataset.Field"):
        td.threshold(data, tack.zeros((data.num_points,)), 0, 1)


@pytest.mark.parametrize("kind", ["shape", "polyhedron", "polygon", "polyline"])
@pytest.mark.parametrize("unused_points", [0, 2])
def test_empty_filter_pipelines_preserve_maps_and_ids(backend, kind, unused_points):
    if kind == "shape":
        top = td.UnstructuredTopology([], [0], [], num_points=unused_points,
                                       id_dtype=tack.i64)
    elif kind == "polyhedron":
        top = td.PolyhedralTopology([0], [], [0], [], [], num_points=unused_points,
                                     id_dtype=tack.i64)
    else:
        cls = td.PolygonalTopology if kind == "polygon" else td.PolylineTopology
        top = cls([0], [], num_points=unused_points, id_dtype=tack.i64)
    data = td.DataSet(top, np.zeros((unused_points, 3), np.float32))
    data.fields["p"] = td.Field(td.H1(data), tack.arrays.ConstantArray(0., unused_points))
    data.fields["c"] = td.Field(td.Constant(data), tack.arrays.ConstantArray(0., 0))
    data.sets["empty"] = td.Set(data, "cells", [])
    for out in (td.threshold(data, "p", 0, 1), td.extract_cells(data, []), td.mask(data, 2)):
        assert out.id_dtype is tack.i64
        assert out.num_cells == out.num_points == 0
        assert set(out.fields) == {"p", "c"}
        assert "empty" in out.sets and len(out.sets["empty"]) == 0
        assert out.geometry.values.size == 0
    points = td.to_points(data, "c")
    np.testing.assert_array_equal(points.values.numpy(), np.zeros(unused_points))
    assert td.to_cells(data, "p").values.size == 0
    if kind == "polyline":
        with pytest.raises(NotImplementedError, match="polylines"):
            td.contour(data, "p", 0.5)
    else:
        assert td.contour(data, "p", 0.5).num_cells == 0
        with pytest.raises(KeyError, match="missing"):
            td.contour(data, "p", 0.5, fields="missing")
    if kind == "shape":
        assert td.clip(data, "p", 0.5).num_cells == 0
        for algorithm in (td.clip, td.extract_cells):
            argument = "p" if algorithm is td.clip else []
            with pytest.raises(KeyError, match="missing"):
                algorithm(data, argument, fields="missing")


def test_extracting_an_empty_face_set_keeps_point_fields(backend):
    data = td.uniform_grid((2, 2, 2))
    data.fields["p"] = td.point_ids(data)
    data.sets["none"] = td.Set(data, "faces", [])
    out = td.extract_surface(data, "none")
    assert out.num_cells == 0 and out.num_points == data.num_points
    assert out.topology.offsets.numpy().tolist() == [0]
    np.testing.assert_array_equal(tack.arrays.to_host(out.fields["p"].values), np.arange(8))
    assert out.sets["none"].on == "cells" and len(out.sets["none"]) == 0


def test_empty_particles_keep_ids_and_field_layouts(backend):
    top = td.UnstructuredTopology([], [0], [], num_points=0, id_dtype=tack.i64)
    data = td.DataSet(top, np.empty((0, 3)))
    data.fields["v"] = td.Field(td.H1(data), tack.arrays.ConstantArray([1., 0., 0.], 0))
    for algorithm in (td.advect, td.streamlines):
        out = algorithm(data, "v", [], 0.1, 1)
        assert out.id_dtype is tack.i64
        assert out.num_cells == out.num_points == 0
    out = td.advect(data, "v", [[1., 2., 3.]], 0.1, 1)
    np.testing.assert_array_equal(out.positions(), [[1., 2., 3.]])
    np.testing.assert_array_equal(out.fields["steps"].values.numpy(), [0])
    np.testing.assert_array_equal(out.fields["status"].values.numpy(),
                                  [int(td.ParticleStatus.SPATIAL_BOUNDS)])


@pytest.mark.parametrize("polyhedral", [False, True])
@pytest.mark.parametrize("bad", ["past", "negative"])
def test_extract_surface_refuses_face_ids_outside_the_topology(backend, polyhedral, bad):
    """A face id the topology does not have was read out of bounds, silently
    giving wrong cells (only debug mode noticed)."""
    data = td.uniform_grid((3, 3, 3))
    if polyhedral:
        data = td.as_polyhedra(data)
    nf = data.topology.faces().num_faces
    with pytest.raises(IndexError, match=rf"1 face ids lie outside \[0, {nf}\)"):
        td.extract_surface(data, [0, nf if bad == "past" else -1])
    assert td.extract_surface(data, [0, nf - 1]).num_cells == 2


@pytest.mark.parametrize("polyhedral", [False, True])
def test_extract_surface_refuses_a_face_set_of_another_topology(backend, polyhedral):
    data, other = td.uniform_grid((3, 3, 3)), td.uniform_grid((6, 6, 6))
    if polyhedral:
        data = td.as_polyhedra(data)
    with pytest.raises(ValueError, match="another topology"):
        td.extract_surface(data, td.Set(other, "faces", [0]))


def test_extract_cells_refuses_cell_ids_outside_the_topology(backend):
    data = td.uniform_grid((3, 3, 3))
    with pytest.raises(IndexError, match=r"2 cell ids lie outside \[0, 8\)"):
        td.extract_cells(data, [8, -1, 3])
    assert td.extract_cells(data, [0, 7]).num_cells == 2
