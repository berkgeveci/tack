"""Sets (``tack.dataset.Set``): named selections of points, cells or faces. Made and
checked, iterated over (cell and face sets), and carried through filters by the
same maps as fields."""

import numpy as np
import pytest
from test_dataset_api import _scalars

import tack
import tack.dataset as td


def _grid():
    # 3 x 2 hexahedra.
    return td.rectilinear_grid([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0], [0.0, 1.0])


@tack.kernel
def _mark_cells(cells, u, out):
    for c in cells:
        out[cells.entity_id(c)] = 1 + u.value(c, cells.parametric_center())


def test_a_set_is_sorted_unique_and_checked(backend):
    data = _grid()
    s = td.Set(data, "cells", [4, 1, 4, 0])
    assert (s.on, len(s)) == ("cells", 3)
    np.testing.assert_array_equal(s.numpy(), [0, 1, 4])
    np.testing.assert_array_equal(s.flags().numpy(), [1, 1, 0, 0, 1, 0])
    assert s.ids.dtype == data.id_dtype
    with pytest.raises(ValueError, match="outside the topology's 6 cells"):
        td.Set(data, "cells", [6])
    with pytest.raises(ValueError, match="points, cells or faces"):
        td.Set(data, "edges", [0])
    assert len(td.Set(data, "points", [])) == 0


def test_data_sets_are_checked(backend):
    data = _grid()
    data.sets["left"] = td.Set(data, "cells", [0, 3])
    with pytest.raises(TypeError, match="not a tack.dataset.Set"):
        data.sets["raw"] = tack.arange(3)
    with pytest.raises(ValueError, match="iteration domain's name"):
        data.sets["cells"] = td.Set(data, "cells", [0])
    with pytest.raises(ValueError, match="another topology"):
        data.sets["other"] = td.Set(_grid(), "cells", [0])
    with pytest.raises(TypeError, match="not a tack.dataset.Set"):
        td.DataSet(data.topology, data.geometry, sets={"raw": [0, 1]})
    # Every way in is checked, and a refused one leaves the sets as they were.
    sets = data.sets
    for put in (lambda: sets.update(raw=[0]), lambda: sets.setdefault("raw", [0]),
                lambda: sets.__ior__({"good": td.Set(data, "cells", [1]), "raw": [0]})):
        with pytest.raises(TypeError, match="not a tack.dataset.Set"):
            put()
    with pytest.raises(TypeError, match="not a tack.dataset.Set"):
        data.sets |= {"raw": [0]}
    assert list(data.sets) == ["left"]


def test_for_each_over_a_cell_set(backend):
    data = _grid()
    data.fields["x"] = td.Field(td.H1(data), _scalars(data.positions()[:, 0]))
    data.sets["middle"] = td.Set(data, "cells", [1, 4])
    out = tack.zeros((data.num_cells,))
    td.for_each(_mark_cells, data, "middle", data.fields["x"], out)
    np.testing.assert_allclose(out.numpy(), [0, 2.5, 0, 0, 2.5, 0])
    data.sets["corners"] = td.Set(data, "points", [0])
    with pytest.raises(ValueError, match="not an iteration domain"):
        td.for_each(_mark_cells, data, "corners", data.fields["x"], out)


def test_a_cell_set_in_a_mixed_mesh(backend):
    data = td.as_polyhedra(_grid())
    data.sets["ends"] = td.Set(data, "cells", [0, 5])
    groups = data.launch_groups("ends")
    assert sum(g.count for g in groups) == 2
    volumes = td.cell_geometry(data).measure.values.numpy()
    np.testing.assert_allclose(volumes, 1)


def test_side_sets_from_cells_and_local_faces(backend):
    data = _grid()
    faces = data.topology.faces()
    sides = faces.sides.numpy()
    # Each boundary face named by its cell and its local face.
    boundary = td.boundary_faces(data)
    named = sides[boundary.numpy()]
    s = td.Set.from_sides(data, named[:, 0], named[:, 1])
    np.testing.assert_array_equal(s.numpy(), boundary.numpy())
    # Either side of an interior face names that face.
    shared = np.flatnonzero(sides[:, 2] >= 0)[0]
    both = td.Set.from_sides(data, sides[shared, [0, 2]], sides[shared, [1, 3]])
    np.testing.assert_array_equal(both.numpy(), [shared])
    with pytest.raises(ValueError, match="a local face they do not have"):
        td.Set.from_sides(data, [0], [6])


def _prism(m):
    """One polyhedron: a prism over an m-gon, so m + 2 faces (caps first)."""
    angle = np.linspace(0, 2 * np.pi, m, endpoint=False)
    ring = np.c_[np.cos(angle), np.sin(angle)]
    points = np.r_[np.c_[ring, np.zeros(m)], np.c_[ring, np.ones(m)]].astype(np.float32)
    faces = [list(range(m))[::-1], list(range(m, 2 * m))] + [
        [i, (i + 1) % m, m + (i + 1) % m, m + i] for i in range(m)]
    topology = td.PolyhedralTopology(np.cumsum([0] + [len(f) for f in faces]),
                                     np.concatenate(faces), [0, len(faces)],
                                     np.arange(len(faces)), np.zeros(len(faces), np.uint8))
    return td.DataSet(topology, points)


def test_side_sets_of_cells_with_many_faces(backend):
    """A polyhedron may have more faces than a 64-bit mask has bits: each (cell,
    local face) pair names its own face, and only that one. A mask selected face
    64 with face 0, and refused local faces past 62."""
    data = _prism(64)
    for local in (0, 1, 63, 65):
        np.testing.assert_array_equal(td.Set.from_sides(data, [0], [local]).numpy(), [local])
    with pytest.raises(ValueError, match="a local face they do not have"):
        td.Set.from_sides(data, [0], [66])


@pytest.mark.parametrize("make", [td.as_polyhedra, lambda d: td.as_polygons(td.external_faces(d))])
def test_side_sets_where_cells_list_their_faces(backend, make):
    data = make(_grid())
    sides = data.topology.faces().sides.numpy()
    boundary = td.boundary_faces(data)
    named = sides[boundary.numpy()]
    s = td.Set.from_sides(data, named[:, 0], named[:, 1])
    np.testing.assert_array_equal(s.numpy(), boundary.numpy())


def test_sets_are_made_from_pairs_too(backend):
    data = _grid()
    left = td.Set(data, "cells", [0, 3])
    built = td.DataSet(data.topology, data.geometry, sets=[("left", left)])
    assert built.sets["left"] is left
    with pytest.raises(TypeError, match="not a tack.dataset.Set"):
        td.DataSet(data.topology, data.geometry, sets=[("left", left), ("raw", [0])])


def test_the_boundary_set_is_taken_as_the_faces_hold_it(backend, monkeypatch):
    """boundary_faces wraps the boundary the faces already hold, sorted and
    unique: it is not checked again over every face, which made external
    faces three times slower on 5M cells. Its flags are made when asked for."""
    from tack.dataset import sets

    data = _grid()
    faces = data.topology.faces()
    checked = []
    adopt = sets.Set._adopt
    monkeypatch.setattr(sets.Set, "_adopt", lambda s, flags: (checked.append(s), adopt(s, flags)))
    boundary = td.boundary_faces(data)
    assert not checked
    assert boundary.ids is faces.boundary()
    expected = np.zeros(faces.num_faces, np.int32)
    expected[faces.boundary().numpy()] = 1
    np.testing.assert_array_equal(boundary.flags().numpy(), expected)
    data.sets["boundary"] = boundary
    surface = td.external_faces(data)
    assert surface.num_cells == len(boundary)


def test_filters_carry_sets(backend):
    data = _grid()
    data.fields["c"] = td.Field(td.Constant(data), _scalars(np.arange(6.0)))
    data.sets["odd"] = td.Set(data, "cells", [1, 3, 5])
    data.sets["origin"] = td.Set(data, "points", [0])
    data.sets["boundary"] = td.boundary_faces(data)
    # threshold keeps cells 2..5: their cells and points renumbered.
    kept = td.threshold(data, "c", 2, 5)
    np.testing.assert_array_equal(kept.sets["odd"].numpy(), [1, 3])
    assert len(kept.sets["origin"]) == 0
    assert "boundary" not in kept.sets                  # faces are derived afresh
    # The boundary surface's cells are the boundary faces: the face set is all of them.
    surface = td.external_faces(data)
    assert len(surface.sets["boundary"]) == surface.num_cells
    assert surface.sets["boundary"].on == "cells"
    # Same entities, same sets.
    poly = td.as_polyhedra(data)
    for name in data.sets:
        np.testing.assert_array_equal(poly.sets[name].numpy(), data.sets[name].numpy())
        assert poly.sets[name].topology is poly.topology
    moved = td.warp(data, (0.0, 0.0, 1.0))
    np.testing.assert_array_equal(moved.sets["odd"].numpy(), [1, 3, 5])
    # A contour's points are new: point sets are left behind, cell sets go by pieces.
    data.fields["x"] = td.Field(td.H1(data), _scalars(data.positions()[:, 0]))
    cut = td.contour(data, "x", 1.5)
    assert "origin" not in cut.sets
    pieces = cut.fields["c"].values.numpy()
    np.testing.assert_array_equal(np.sort(cut.sets["odd"].numpy()),
                                  np.flatnonzero(np.isin(pieces, [1, 3, 5])))
