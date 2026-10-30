"""``tubes`` and ``glyphs``: geometry made to be drawn, VTK's vtkTubeFilter and
vtkGlyph3D with its arrow source, with the input's fields carried."""

import numpy as np
import pytest

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    yield f64_backend


def _vectors(a):
    out = tack.empty((len(a),), tack.vec3(tack.f64))
    tack.copyto(out, np.ascontiguousarray(a, np.float64))
    return out


def _helix(n=40, turns=2.0):
    t = np.linspace(0.0, 2 * np.pi * turns, n)
    points = np.stack([np.cos(t), np.sin(t), 0.3 * t], axis=1)
    topology = td.PolylineTopology(tack.array(np.array([0, n], np.int32)),
                                   tack.array(np.arange(n, dtype=np.int32)), num_points=n)
    data = td.DataSet(topology, points, dtype=tack.f64)
    data.fields["s"] = td.Field(td.Values(data, "points"), tack.array(t))
    data.fields["c"] = td.Field(td.Values(data, "cells"), tack.array(np.array([7.0])))
    return data


def _quad_normals(data):
    x = data.positions()
    q = data.topology.connectivity.numpy().reshape(-1, 4)
    return np.cross(x[q[:, 1]] - x[q[:, 0]], x[q[:, 3]] - x[q[:, 0]]), x[q].mean(axis=1)


def test_a_tube_rounds_its_line():
    """Each ring is ``sides`` points at ``radius`` from its line point, in the plane
    across the line, with the outward direction as ``normals``; quads wind outward;
    point fields come from the ring's point, cell fields from the line."""
    line = _helix()
    sides = 10
    tube = td.tubes(line, 0.1, sides=sides)
    assert tube.num_points == line.num_points * sides
    assert tube.num_cells == (line.num_points - 1) * sides
    x = tube.positions().reshape(line.num_points, sides, 3)
    centres = line.positions()
    np.testing.assert_allclose(np.linalg.norm(x - centres[:, None], axis=2), 0.1, rtol=1e-12)
    tangent = np.gradient(centres, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    across = np.einsum("pk,psk->ps", tangent, x - centres[:, None])
    np.testing.assert_allclose(across, 0.0, atol=1e-12)
    normals = tube.fields["normals"].values.numpy().reshape(line.num_points, sides, 3)
    np.testing.assert_allclose(normals, (x - centres[:, None]) / 0.1, atol=1e-12)
    np.testing.assert_array_equal(tube.fields["s"].values.numpy(),
                                  np.repeat(line.fields["s"].values.numpy(), sides))
    np.testing.assert_array_equal(tube.fields["c"].values.numpy(), 7.0)
    n, middle = _quad_normals(tube)
    centre_of = np.repeat(centres[:-1] + centres[1:], sides, axis=0) / 2
    assert ((n * (middle - centre_of)).sum(axis=1) > 0).all()


def test_tubes_do_not_twist():
    """The ring's frame is carried along the line: consecutive rings' first points
    stay close, so the quads do not shear round the tube."""
    tube = td.tubes(_helix(n=200), 0.05, sides=8)
    first = tube.positions().reshape(200, 8, 3)[:, 0]
    steps = np.linalg.norm(np.diff(first, axis=0), axis=1)
    path = np.linalg.norm(np.diff(_helix(n=200).positions(), axis=0), axis=1)
    assert (steps < 1.5 * path).all()


def test_tubes_of_line_cells_and_radii_by_a_field():
    """Line cells (``extract_edges``') are tubes too, each its own; ``radius_by`` scales
    each ring."""
    grid = td.uniform_grid((3, 3), spacing=(1.0, 1.0), dtype=tack.f64)
    edges = td.extract_edges(grid)
    edges.fields["w"] = td.Field(td.H1(edges), tack.array(np.arange(edges.num_points) + 1.0))
    tube = td.tubes(edges, 0.05, sides=6, radius_by="w")
    assert tube.num_cells == edges.num_cells * 6
    ends = edges.topology.connectivity.numpy()
    x = tube.positions().reshape(-1, 6, 3)
    r = np.linalg.norm(x - edges.positions()[ends][:, None], axis=2)
    np.testing.assert_allclose(r, np.broadcast_to(0.05 * (ends + 1.0)[:, None], r.shape),
                               rtol=1e-12)


def test_streamlines_as_tubes():
    """``streamlines``' polylines go straight in, with their ``time`` and ``seed``."""
    grid = td.uniform_grid((6, 6, 6), spacing=(0.2, 0.2, 0.2), dtype=tack.f64)
    x = grid.positions()
    velocity = np.stack([-(x[:, 1] - 0.5), x[:, 0] - 0.5, np.full(len(x), 0.1)], axis=1)
    grid.fields["v"] = td.Field(td.H1(grid), _vectors(velocity))
    seeds = np.array([[0.7, 0.5, 0.1], [0.5, 0.8, 0.2]])
    paths = td.streamlines(grid, "v", seeds, 0.05, 50)
    tube = td.tubes(paths, 0.01)
    assert {"time", "seed", "normals"} <= set(tube.fields)
    assert tube.num_cells == (paths.topology.line_points.shape[0] - paths.num_cells) * 12


def test_an_arrow_points_along_its_vector():
    """Each point's arrow starts at it and ends at the point plus ``scale`` times its
    vector, its shaft and head round that line; ``scale_by`` sets the length."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.3, -0.4, 1.2]])
    data = td.point_cloud(points, dtype=tack.f64)
    data.fields["v"] = td.Field(td.Values(data, "points"), _vectors(vectors))
    data.fields["k"] = td.Field(td.Values(data, "points"), tack.array(np.array([1.0, 2.0, 3.0])))
    res = 8
    arrows = td.glyphs(data, "v", scale=0.5, resolution=res)
    per = 3 * res + 2
    assert arrows.num_points == 3 * per and arrows.num_cells == 3 * 6 * res
    x = arrows.world_positions().reshape(3, per, 3)       # point_cloud picks an origin
    np.testing.assert_allclose(x[:, 3 * res], points + 0.5 * vectors, atol=1e-12)   # tips
    np.testing.assert_allclose(x[:, 3 * res + 1], points, atol=1e-12)               # bases
    length = 0.5 * np.linalg.norm(vectors, axis=1)
    axis = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    for ring, radius, along in ((0, 0.03, 0.0), (1, 0.03, 0.65), (2, 0.1, 0.65)):
        r = x[:, ring * res:(ring + 1) * res] - points[:, None]
        on_axis = np.einsum("pk,psk->ps", axis, r)
        np.testing.assert_allclose(on_axis, np.broadcast_to((along * length)[:, None],
                                                             on_axis.shape), atol=1e-12)
        off = np.linalg.norm(r - on_axis[..., None] * axis[:, None], axis=2)
        np.testing.assert_allclose(off, np.broadcast_to((radius * length)[:, None], off.shape),
                                   atol=1e-12)
    np.testing.assert_array_equal(arrows.fields["k"].values.numpy(),
                                  np.repeat([1.0, 2.0, 3.0], per))
    by = td.glyphs(data, "v", scale=2.0, scale_by="k", resolution=res)
    np.testing.assert_allclose(by.world_positions().reshape(3, per, 3)[:, 3 * res],
                               points + 2.0 * np.array([1.0, 2.0, 3.0])[:, None] * axis,
                               atol=1e-12)


def test_arrows_are_closed_and_wound_outward():
    """An arrow is a closed surface wound outward: its signed volume, by the
    divergence theorem, is the shaft's and head's."""
    data = td.point_cloud(np.zeros((1, 3)), dtype=tack.f64)
    data.fields["v"] = td.Field(td.Values(data, "points"), _vectors([[0.0, 0.0, 1.0]]))
    res = 64
    arrow = td.glyphs(data, "v", resolution=res)
    x = arrow.world_positions()
    tri = arrow.topology.connectivity.numpy().reshape(-1, 3)
    volume = np.einsum("ij,ij->i", x[tri[:, 0]], np.cross(x[tri[:, 1]], x[tri[:, 2]])).sum() / 6
    polygon = res / 2 * np.sin(2 * np.pi / res)              # a unit circle's inscribed area
    expected = polygon * (0.03 ** 2 * 0.65 + 0.1 ** 2 * 0.35 / 3)
    np.testing.assert_allclose(volume, expected, rtol=1e-10)


def test_refusals():
    with pytest.raises(TypeError, match="takes lines"):
        td.tubes(td.uniform_grid((3, 3, 3), dtype=tack.f64), 0.1)
    with pytest.raises(ValueError, match="at least 3 sides"):
        td.tubes(_helix(), 0.1, sides=2)
    data = td.point_cloud(np.zeros((2, 3)), dtype=tack.f64)
    data.fields["s"] = td.Field(td.Values(data, "points"), tack.array(np.zeros(2)))
    with pytest.raises(TypeError, match="3-vector"):
        td.glyphs(data, "s")
