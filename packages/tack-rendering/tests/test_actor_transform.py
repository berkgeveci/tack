"""Tests for Actor 4x4 transform support."""

import numpy as np
import pytest
import tack
from tack.rendering import Scene, Actor, PointLight

_backends = []
for _arch in ["cpu", "metal"]:
    try:
        tack.init(arch=getattr(tack, _arch))
        _backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=_backends)
def backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


def _triangle():
    """Create a simple triangle: (0,0,0), (1,0,0), (0.5,1,0)."""
    pts = tack.field(dtype=tack.f32, shape=(9,))
    pts.from_numpy(np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float32))
    conn = tack.field(dtype=tack.i32, shape=(3,))
    conn.from_numpy(np.array([0, 1, 2], dtype=np.int32))
    return pts, conn


# --- No transform (identity) ---

def test_no_transform(backend):
    """Actor without transform leaves points unchanged."""
    pts, conn = _triangle()
    actor = Actor(pts, conn)
    assert actor.transform is None

    scene = Scene()
    scene.add(actor)
    scene.add(PointLight((5, 5, 5)))
    prepared = scene._prepare()

    result = prepared['points'].to_numpy()
    np.testing.assert_allclose(result, [0, 0, 0, 1, 0, 0, 0.5, 1, 0], atol=1e-6)


# --- Translation ---

def test_translation(backend):
    """Translation by (2, 3, 4) shifts all vertices."""
    pts, conn = _triangle()
    xform = np.eye(4, dtype=np.float32)
    xform[0, 3] = 2.0
    xform[1, 3] = 3.0
    xform[2, 3] = 4.0

    actor = Actor(pts, conn, transform=xform)
    scene = Scene()
    scene.add(actor)
    scene.add(PointLight((5, 5, 5)))
    prepared = scene._prepare()

    result = prepared['points'].to_numpy()
    expected = np.array([2, 3, 4, 3, 3, 4, 2.5, 4, 4], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-5)


# --- Rotation ---

def test_rotation_90z(backend):
    """90-degree rotation around Z: (1,0,0) → (0,1,0)."""
    pts, conn = _triangle()
    xform = np.array([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=np.float32)

    actor = Actor(pts, conn, transform=xform)
    scene = Scene()
    scene.add(actor)
    scene.add(PointLight((5, 5, 5)))
    prepared = scene._prepare()

    result = prepared['points'].to_numpy()
    # (0,0,0)→(0,0,0), (1,0,0)→(0,1,0), (0.5,1,0)→(-1,0.5,0)
    np.testing.assert_allclose(result[0:3], [0, 0, 0], atol=1e-5)
    np.testing.assert_allclose(result[3:6], [0, 1, 0], atol=1e-5)
    np.testing.assert_allclose(result[6:9], [-1, 0.5, 0], atol=1e-5)


# --- Uniform scale ---

def test_scale(backend):
    """Uniform 2x scale doubles all coordinates."""
    pts, conn = _triangle()
    xform = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32)

    actor = Actor(pts, conn, transform=xform)
    scene = Scene()
    scene.add(actor)
    scene.add(PointLight((5, 5, 5)))
    prepared = scene._prepare()

    result = prepared['points'].to_numpy()
    expected = np.array([0, 0, 0, 2, 0, 0, 1, 2, 0], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-5)


# --- Multi-actor: one with transform, one without ---

def test_multi_actor_mixed_transform(backend):
    """Two actors: one transformed, one not."""
    pts1, conn1 = _triangle()
    pts2, conn2 = _triangle()

    xform = np.eye(4, dtype=np.float32)
    xform[0, 3] = 10.0  # translate actor1 by x=10

    actor1 = Actor(pts1, conn1, color=(1, 0, 0), transform=xform)
    actor2 = Actor(pts2, conn2, color=(0, 1, 0))  # no transform

    scene = Scene()
    scene.add(actor1)
    scene.add(actor2)
    scene.add(PointLight((5, 5, 5)))
    prepared = scene._prepare()

    result = prepared['points'].to_numpy()
    # actor1 (translated): x+10
    np.testing.assert_allclose(result[0], 10.0, atol=1e-5)
    np.testing.assert_allclose(result[3], 11.0, atol=1e-5)
    # actor2 (untransformed): original x
    np.testing.assert_allclose(result[9], 0.0, atol=1e-5)
    np.testing.assert_allclose(result[12], 1.0, atol=1e-5)


# --- Original field is not mutated ---

def test_transform_does_not_mutate_original(backend):
    """Applying a transform should not modify the original points field."""
    pts, conn = _triangle()
    original = pts.to_numpy().copy()

    xform = np.eye(4, dtype=np.float32)
    xform[0, 3] = 100.0

    actor = Actor(pts, conn, transform=xform)
    scene = Scene()
    scene.add(actor)
    scene.add(PointLight((5, 5, 5)))
    scene._prepare()

    # Original field should be unchanged
    np.testing.assert_array_equal(pts.to_numpy(), original)


# --- Transform accepts list input ---

def test_transform_from_list(backend):
    """Transform can be a nested list, not just numpy array."""
    pts, conn = _triangle()
    xform = [[1, 0, 0, 5],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]

    actor = Actor(pts, conn, transform=xform)
    scene = Scene()
    scene.add(actor)
    scene.add(PointLight((5, 5, 5)))
    prepared = scene._prepare()

    result = prepared['points'].to_numpy()
    assert result[0] == pytest.approx(5.0)


# --- Normals are correctly transformed ---

def test_normals_transformed(backend):
    """Pre-computed normals should be transformed by inverse-transpose."""
    pts, conn = _triangle()
    # Normal pointing in +Z for a flat triangle in XY plane
    normals = tack.field(dtype=tack.f32, shape=(9,))
    normals.from_numpy(np.array([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32))

    # 90-degree rotation around X: Z→Y, Y→-Z
    xform = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1],
    ], dtype=np.float32)

    actor = Actor(pts, conn, normals=normals, transform=xform)
    scene = Scene()
    scene.add(actor)
    scene.add(PointLight((5, 5, 5)))
    prepared = scene._prepare()

    result_n = prepared['normals'].to_numpy()
    # Normal (0,0,1) rotated 90° around X → (0,-1,0)... wait, inverse-transpose
    # For a rotation, inverse-transpose = the rotation itself
    # So (0,0,1) → (0,-1,0)... let's check: R*n = [0, -1, 0]
    # Actually R = [[1,0,0],[0,0,-1],[0,1,0]], so R*(0,0,1) = (0,-1,0)...
    # But wait, inv(R)^T = R for orthogonal matrices, so same result
    np.testing.assert_allclose(result_n[0:3], [0, -1, 0], atol=1e-4)
