"""Checking a topology's arrays: ``validate()`` on every kind, and debug mode,
which checks the ids of every topology when it is made, before anything is
derived from them (a bad id would be read or written out of bounds)."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.arrays import CountingArray
from tack.dataset.topology import uniform_cells

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

TETRA = 10


@pytest.fixture
def debug():
    was = tack.runtime.dispatch.debug_enabled()
    tack.set_debug(True)
    yield
    tack.set_debug(was)


def _tets(offsets=(0, 4, 8, 12), connectivity=(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5),
          num_points=6, types=TETRA):
    n = len(offsets) - 1
    return td.UnstructuredTopology(np.full(n, types, np.uint8), np.asarray(offsets),
                                   np.asarray(connectivity), num_points=num_points)


def test_a_valid_topology_of_every_kind_passes(backend):
    grid = td.uniform_grid((3, 3, 3))
    tets = td.tetrahedralize(grid)
    for data in (grid, tets, td.as_polyhedra(tets), td.as_polygons(td.uniform_grid((3, 3))),
                 td.streamlines(grid, td.Field(td.H1(grid), tack.arrays.ConstantArray(
                     [1.0, 0.0, 0.0], grid.num_points)), [[0.1, 0.5, 0.5]], 0.1, 5)):
        assert data.topology.validate() is data.topology
    # Implicit offsets and connectivity are checked as stored ones are.
    types, offsets = uniform_cells(TETRA, 2, tack.i32)
    implicit = td.UnstructuredTopology(types, offsets, CountingArray(8), num_points=8)
    assert implicit.validate() is implicit


def test_validate_says_what_is_wrong_and_how_often(backend):
    with pytest.raises(ValueError, match=r"2 of the 12 entries of connectivity are outside "
                                         r"\[0, 6\) \(num_points\)"):
        _tets(connectivity=(0, 1, 2, 3, 1, 2, 3, 6, 2, 3, 4, -1)).validate()
    with pytest.raises(ValueError, match=r"1 of the 3 ranges in offsets run backwards"):
        _tets(offsets=(0, 6, 4, 12)).validate()
    with pytest.raises(ValueError, match="does not start at 0 and end at 12"):
        _tets(offsets=(0, 4, 8, 11)).validate()
    with pytest.raises(ValueError, match="point count"):            # tetras of 3 points
        _tets(offsets=(0, 3, 8, 12)).validate()
    with pytest.raises(ValueError, match="not linear shapes"):
        _tets(types=99).validate()
    poly = td.as_polyhedra(td.tetrahedralize(td.uniform_grid((2, 2, 2)))).topology
    face_offsets, face_points, cell_offsets, cell_faces, sides = poly.arrays()
    wrong = cell_faces.copy()
    wrong[0] = poly.num_faces
    with pytest.raises(ValueError, match=r"1 of the \d+ entries of cell_faces are outside"):
        td.PolyhedralTopology(face_offsets, face_points, cell_offsets, wrong, sides).validate()
    two = sides.copy()
    two[0] = 2
    with pytest.raises(ValueError, match="neither 0 nor 1"):
        td.PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces,
                              two).validate()
    lines = td.PolylineTopology([0, 2, 4], [0, 1, 1, 2], num_points=3)
    lines.line_points = tack.array(np.array([0, 1, 1, 5], np.int32))
    with pytest.raises(ValueError, match="entries of line_points are outside"):
        lines.validate()


def test_without_debug_mode_nothing_is_checked_when_made(backend, no_debug):
    bad = _tets(connectivity=(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 9))
    with pytest.raises(ValueError, match="connectivity are outside"):
        bad.validate()


def test_debug_mode_checks_ids_when_a_topology_is_made(backend, debug):
    with pytest.raises(ValueError, match="connectivity are outside"):
        _tets(connectivity=(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 9))
    # Polygons and polylines are checked before the facets and ends are derived,
    # which index by point.
    with pytest.raises(ValueError, match="loop_points are outside"):
        td.PolygonalTopology([0, 3, 6], [0, 1, 2, 2, 1, 7], num_points=4)
    with pytest.raises(ValueError, match="line_points are outside"):
        td.PolylineTopology([0, 2, 5], [0, 1, 1, 2, -1], num_points=3)
    assert _tets().num_cells == 3                          # a valid one is made as before
