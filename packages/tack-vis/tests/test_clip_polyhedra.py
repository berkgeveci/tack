"""``clip`` of polyhedral topologies: each cut cell becomes the polyhedron of its kept
part (its faces clipped, its iso-polygons as caps), faces shared stay shared. Held
to the shape-based clip of the same cells where both are exact (planes), to its
volume elsewhere, and to the output's own consistency."""

import itertools

import numpy as np
import pytest

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    """Volumes compared in f64, on every backend that has f64."""
    yield f64_backend
    tack.set_default_device("cpu")


def _mesh(kind):
    grid = td.uniform_grid((5, 4, 4), spacing=(0.5, 0.5, 0.5), dtype=tack.f64)
    if kind == "tetrahedra":
        grid = td.tetrahedralize(grid)
    x = grid.positions().astype(np.float64)
    grid.fields["plane"] = td.Field(td.H1(grid), tack.array(x @ [0.8, 0.3, -0.5]))
    grid.fields["r"] = td.Field(td.H1(grid), tack.array(np.linalg.norm(x - [1.0, 0.8, 0.7],
                                                                       axis=1)))
    grid.fields["id"] = td.Field(td.Constant(grid), tack.array(
        np.arange(grid.num_cells, dtype=np.float64)))
    return grid


def _volume(data):
    return float(td.cell_geometry(data).measure.values.numpy().sum())


@pytest.mark.parametrize("kind", ["hexahedra", "tetrahedra"])
@pytest.mark.parametrize("inside", [False, True])
def test_a_plane_cuts_polyhedra_as_it_cuts_their_shapes(kind, inside):
    mesh = _mesh(kind)
    poly = td.as_polyhedra(mesh)
    got = td.clip(poly, "plane", 0.6, inside=inside)
    want = td.clip(mesh, "plane", 0.6, inside=inside)
    assert isinstance(got.topology, td.PolyhedralTopology)
    assert _volume(got) == pytest.approx(_volume(want), rel=1e-12)
    other = td.clip(poly, "plane", 0.6, inside=not inside)
    assert _volume(got) + _volume(other) == pytest.approx(_volume(mesh), rel=1e-12)
    # A linear field's crossings are on the plane: point data interpolated exactly.
    values = got.fields["plane"].values.numpy()
    np.testing.assert_allclose(values, got.positions() @ [0.8, 0.3, -0.5], atol=1e-12)
    assert (values >= 0.6 - 1e-12).all() if not inside else (values <= 0.6 + 1e-12).all()


def test_a_sphere_cut_and_the_outputs_consistency():
    mesh = _mesh("hexahedra")
    poly = td.as_polyhedra(mesh)
    got = td.clip(poly, "r", 0.6, inside=True)
    want = td.clip(mesh, "r", 0.6, inside=True)
    # Both linear along edges, capped differently (planar polygons, triangles).
    assert _volume(got) == pytest.approx(_volume(want), rel=2e-2)
    assert (td.cell_geometry(got).measure.values.numpy() > 0).all()
    td.check_winding(got)                      # every face wound out of its side 0
    surface = td.extract_surface(got, td.boundary_faces(got))
    # A closed surface: every edge of its polygons on exactly two of them.
    offsets, loops = (a.numpy() for a in surface.topology.loops())
    edges = [tuple(sorted((loops[i], loops[i + 1 if i + 1 < b else a])))
             for a, b in itertools.pairwise(offsets) for i in range(a, b)]
    _, counts = np.unique(np.array(edges), axis=0, return_counts=True)
    assert (counts == 2).all()


def test_cell_data_follows_each_cut_cell_and_clips_compose():
    mesh = _mesh("hexahedra")
    poly = td.as_polyhedra(mesh)
    once = td.clip(poly, "plane", 0.6)
    ids = once.fields["id"].values.numpy()
    assert (np.diff(ids) > 0).all()            # each kept cell once, in order
    # Clipped again by a plane across the first: the polyhedra of the first clip
    # are not convex boxes any more, and the volume is still the shapes'.
    x = once.positions()
    once.fields["y"] = td.Field(td.H1(once), tack.array(x[:, 1].astype(np.float64)))
    twice = td.clip(once, "y", 0.9)
    shape_once = td.clip(mesh, "plane", 0.6)
    shape_once.fields["y"] = td.Field(td.H1(shape_once), tack.array(
        shape_once.positions()[:, 1].astype(np.float64)))
    assert _volume(twice) == pytest.approx(_volume(td.clip(shape_once, "y", 0.9)), rel=1e-12)


def test_cells_wholly_kept_or_dropped():
    mesh = _mesh("hexahedra")
    poly = td.as_polyhedra(mesh)
    everything = td.clip(poly, "plane", -100.0)
    assert everything.num_cells == poly.num_cells
    assert _volume(everything) == pytest.approx(_volume(poly), rel=1e-12)
    nothing = td.clip(poly, "plane", 100.0)
    assert nothing.num_cells == 0 and nothing.num_points == 0


def test_polygons_and_polylines_are_still_refused():
    flat = td.as_polygons(td.uniform_grid((3, 3)))
    flat.fields["x"] = td.Field(td.H1(flat), tack.array(flat.positions()[:, 0]))
    with pytest.raises(NotImplementedError, match="not polygons or polylines"):
        td.clip(flat, "x", 0.5)


@pytest.mark.parametrize("inside", [False, True])
def test_a_cut_through_faces_keeps_them_wound_out_of_their_cell(inside):
    """Cut exactly through grid faces: a face whose points are all at the value
    stays, on the kept cell, though the cell it was wound out of is gone -- it is
    turned around to be wound out of the cell that is left."""
    grid = td.uniform_grid((4, 3, 3), dtype=tack.f64)
    grid.fields["x"] = td.Field(td.H1(grid), tack.array(grid.positions()[:, 0]))
    out = td.clip(td.as_polyhedra(grid), "x", 1.0, inside=inside)
    td.check_winding(out)
    assert _volume(out) == pytest.approx(4.0 if inside else 8.0, rel=1e-12)
    surface = td.extract_surface(out, td.boundary_faces(out))
    assert _volume_of_surface(surface) == pytest.approx(_volume(out), rel=1e-12)


def _volume_of_surface(surface):
    """The volume a closed, outward-wound polygon surface bounds (divergence theorem)."""
    x = surface.positions().astype(np.float64)
    offsets, loops = (a.numpy() for a in surface.topology.loops())
    total = 0.0
    for a, b in itertools.pairwise(offsets):
        p = x[loops[a:b]]
        for i in range(1, len(p) - 1):
            total += np.dot(p[0], np.cross(p[i], p[i + 1]))
    return total / 6.0
