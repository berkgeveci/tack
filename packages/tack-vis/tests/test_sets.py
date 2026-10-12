"""Sets (``tack.data.Set``): named selections of points, cells or faces. Made and
checked, iterated over (cell and face sets), and carried through filters by the
same maps as fields."""

import numpy as np
import pytest
from test_dataset_api import _scalars

import tack
import tack.data as td


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
    np.testing.assert_array_equal(s.to_numpy(), [0, 1, 4])
    np.testing.assert_array_equal(s.flags().to_numpy(), [1, 1, 0, 0, 1, 0])
    assert s.ids.dtype == data.id_dtype
    with pytest.raises(ValueError, match="outside the topology's 6 cells"):
        td.Set(data, "cells", [6])
    with pytest.raises(ValueError, match="points, cells or faces"):
        td.Set(data, "edges", [0])
    assert len(td.Set(data, "points", [])) == 0


def test_data_sets_are_checked(backend):
    data = _grid()
    data.sets["left"] = td.Set(data, "cells", [0, 3])
    with pytest.raises(TypeError, match="not a tack.data.Set"):
        data.sets["raw"] = tack.arange(3, tack.i32)
    with pytest.raises(ValueError, match="iteration domain's name"):
        data.sets["cells"] = td.Set(data, "cells", [0])
    with pytest.raises(ValueError, match="another topology"):
        data.sets["other"] = td.Set(_grid(), "cells", [0])
    with pytest.raises(TypeError, match="not a tack.data.Set"):
        td.DataSet(data.topology, data.geometry, sets={"raw": [0, 1]})


def test_for_each_over_a_cell_set(backend):
    data = _grid()
    data.fields["x"] = td.Field(td.H1(data), _scalars(data.positions()[:, 0]))
    data.sets["middle"] = td.Set(data, "cells", [1, 4])
    out = tack.zeros(tack.f32, (data.num_cells,))
    td.for_each(_mark_cells, data, "middle", data.fields["x"], out)
    np.testing.assert_allclose(out.to_numpy(), [0, 2.5, 0, 0, 2.5, 0])
    data.sets["corners"] = td.Set(data, "points", [0])
    with pytest.raises(ValueError, match="not an iteration domain"):
        td.for_each(_mark_cells, data, "corners", data.fields["x"], out)


def test_a_cell_set_in_a_mixed_mesh(backend):
    data = td.as_polyhedra(_grid())
    data.sets["ends"] = td.Set(data, "cells", [0, 5])
    groups = data.launch_groups("ends")
    assert sum(g.count for g in groups) == 2
    volumes = td.cell_geometry(data).measure.values.to_numpy()
    np.testing.assert_allclose(volumes, 1)


def test_side_sets_from_cells_and_local_faces(backend):
    data = _grid()
    faces = data.topology.faces()
    sides = faces.sides.to_numpy(vectors=True)
    # Each boundary face named by its cell and its local face.
    boundary = td.boundary_faces(data)
    named = sides[boundary.to_numpy()]
    s = td.Set.from_sides(data, named[:, 0], named[:, 1])
    np.testing.assert_array_equal(s.to_numpy(), boundary.to_numpy())
    # Either side of an interior face names that face.
    shared = np.flatnonzero(sides[:, 2] >= 0)[0]
    both = td.Set.from_sides(data, sides[shared, [0, 2]], sides[shared, [1, 3]])
    np.testing.assert_array_equal(both.to_numpy(), [shared])
    with pytest.raises(ValueError, match="a local face they do not have"):
        td.Set.from_sides(data, [0], [6])


def test_filters_carry_sets(backend):
    data = _grid()
    data.fields["c"] = td.Field(td.Constant(data), _scalars(np.arange(6.0)))
    data.sets["odd"] = td.Set(data, "cells", [1, 3, 5])
    data.sets["origin"] = td.Set(data, "points", [0])
    data.sets["boundary"] = td.boundary_faces(data)
    # threshold keeps cells 2..5: their cells and points renumbered.
    kept = td.threshold(data, "c", 2, 5)
    np.testing.assert_array_equal(kept.sets["odd"].to_numpy(), [1, 3])
    assert len(kept.sets["origin"]) == 0
    assert "boundary" not in kept.sets                  # faces are derived afresh
    # The boundary surface's cells are the boundary faces: the face set is all of them.
    surface = td.external_faces(data)
    assert len(surface.sets["boundary"]) == surface.num_cells
    assert surface.sets["boundary"].on == "cells"
    # Same entities, same sets.
    poly = td.as_polyhedra(data)
    for name in data.sets:
        np.testing.assert_array_equal(poly.sets[name].to_numpy(), data.sets[name].to_numpy())
        assert poly.sets[name].topology is poly.topology
    moved = td.warp(data, (0.0, 0.0, 1.0))
    np.testing.assert_array_equal(moved.sets["odd"].to_numpy(), [1, 3, 5])
    # A contour's points are new: point sets are left behind, cell sets go by pieces.
    data.fields["x"] = td.Field(td.H1(data), _scalars(data.positions()[:, 0]))
    cut = td.contour(data, "x", 1.5)
    assert "origin" not in cut.sets
    pieces = cut.fields["c"].values.to_numpy()
    np.testing.assert_array_equal(np.sort(cut.sets["odd"].to_numpy()),
                                  np.flatnonzero(np.isin(pieces, [1, 3, 5])))
