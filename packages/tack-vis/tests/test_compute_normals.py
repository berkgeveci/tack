"""Tests for tack.algorithms.compute_normals.

Vertex normals are accumulated with atomics from area-weighted face
normals, so the checks that matter are: unit length, correct direction,
and area weighting actually applied.
"""

import numpy as np
from vis_helpers import make_grid, sphere_field, upload

import tack
from tack.algorithms.compute_normals import compute_normals
from tack.algorithms.flying_edges import flying_edges


def _normals(points_np, conn_np):
    """Run compute_normals on host arrays, return (n_pts, 3) numpy."""
    n_pts = points_np.shape[0]
    n_tris = conn_np.shape[0]
    points = upload(points_np.ravel())
    conn = upload(conn_np.ravel(), dtype=tack.i32, np_dtype=np.int32)
    out = compute_normals(points, conn, n_pts, n_tris)
    return out.to_numpy().reshape(-1, 3)


def test_single_triangle_in_xy_plane(backend):
    """A triangle wound counter-clockwise in z=0 has normal +z."""
    points = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])
    conn = np.array([[0, 1, 2]])
    n = _normals(points, conn)
    np.testing.assert_allclose(n, np.tile([0.0, 0.0, 1.0], (3, 1)), atol=1e-6)


def test_winding_flips_the_normal(backend):
    """Reversing the winding reverses the normal."""
    points = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])
    n = _normals(points, np.array([[0, 2, 1]]))
    np.testing.assert_allclose(n, np.tile([0.0, 0.0, -1.0], (3, 1)), atol=1e-6)


def test_normals_are_unit_length(backend):
    """A flat quad: both triangles agree, every normal has length 1."""
    points = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0]])
    conn = np.array([[0, 1, 2], [0, 2, 3]])
    n = _normals(points, conn)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)


def test_shared_vertex_averages_adjacent_faces(backend):
    """A vertex on a 90° fold gets the bisector of the two face normals."""
    # Two unit squares meeting along the y axis, folded 90° apart:
    # one lies in z=0 (normal +z), one in x=0 (normal -x).
    points = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 1.0, 1.0]])
    conn = np.array([[0, 1, 2], [0, 2, 3],     # z=0 plane, normal +z
                     [0, 5, 3], [0, 4, 5]])    # x=0 plane, normal -x
    n = _normals(points, conn)
    bisector = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    # Vertices 0 and 3 are on the fold and touch both planes.
    np.testing.assert_allclose(n[0], bisector, atol=1e-5)
    np.testing.assert_allclose(n[3], bisector, atol=1e-5)
    # Vertex 2 only touches the z=0 plane.
    np.testing.assert_allclose(n[2], [0.0, 0.0, 1.0], atol=1e-6)


def test_face_normals_are_area_weighted(backend):
    """A large face dominates a small one at their shared vertex."""
    # Vertex 0 is shared by a big +z triangle and a tiny -x triangle.
    big = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    small = np.array([[0.0, 0.0, 0.01], [0.0, 0.01, 0.0]])
    points = np.vstack([big, small])
    conn = np.array([[0, 1, 2], [0, 3, 4]])
    n = _normals(points, conn)
    # The +z contribution is ~10⁶ times larger, so vertex 0 reads as +z.
    assert n[0][2] > 0.999


def test_unreferenced_vertex_stays_zero(backend):
    """Normalization leaves a zero-length accumulator alone, not NaN."""
    points = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [5.0, 5.0, 5.0]])   # touched by no triangle
    n = _normals(points, np.array([[0, 1, 2]]))
    np.testing.assert_array_equal(n[3], [0.0, 0.0, 0.0])
    assert np.isfinite(n).all()


def test_degenerate_triangle_contributes_nothing(backend):
    """A zero-area triangle has a zero cross product, so it adds nothing."""
    points = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0]])   # collinear
    n = _normals(points, np.array([[0, 1, 2]]))
    np.testing.assert_array_equal(n, np.zeros((3, 3)))


def test_sphere_normals_point_outward(backend):
    """End to end: flying edges → normals on an analytic sphere."""
    radius = 1.0
    grid = make_grid(24)
    result = flying_edges(sphere_field(grid, radius), grid, 0.0)

    normals = compute_normals(result["points_field"], result["conn_field"],
                              result["total_points"],
                              result["total_tris"]).to_numpy().reshape(-1, 3)
    points = result["points"].astype(np.float64)
    radial = points / np.linalg.norm(points, axis=1, keepdims=True)

    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    # The exact normal of a sphere is its radial direction.
    alignment = np.einsum("ij,ij->i", normals, radial)
    assert alignment.min() > 0.99
    assert alignment.mean() > 0.998


def test_normals_refine_toward_the_analytic_sphere(backend):
    """Finer grids track the true sphere normal more closely."""
    worst = []
    for n in (12, 24):
        grid = make_grid(n)
        result = flying_edges(sphere_field(grid, 1.0), grid, 0.0)
        normals = compute_normals(
            result["points_field"], result["conn_field"],
            result["total_points"], result["total_tris"]
        ).to_numpy().reshape(-1, 3)
        points = result["points"].astype(np.float64)
        radial = points / np.linalg.norm(points, axis=1, keepdims=True)
        worst.append(1.0 - np.einsum("ij,ij->i", normals, radial).min())
    assert worst[1] < worst[0]
