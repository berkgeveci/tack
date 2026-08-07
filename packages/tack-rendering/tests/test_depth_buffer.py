"""Tests for depth buffer output from the path tracer."""

import numpy as np
import pytest
import tack
from tack.rendering import PerspectiveCamera, Canvas, Scene, Actor, PointLight, render


def _make_scene(z=0.0):
    """Triangle at z=z, camera at z=3."""
    pts = tack.field(dtype=tack.f32, shape=(9,))
    pts.from_numpy(np.array([0, 0, z, 1, 0, z, 0.5, 1, z], dtype=np.float32))
    conn = tack.field(dtype=tack.i32, shape=(3,))
    conn.from_numpy(np.array([0, 1, 2], dtype=np.int32))
    scene = Scene()
    scene.add(Actor(pts, conn, color=(1, 0, 0)))
    scene.add(PointLight((2, 3, 5)))
    camera = PerspectiveCamera((0.5, 0.5, 3), (0.5, 0.5, 0),
                                width=32, height=32)
    return scene, camera


def test_depth_buffer_exists(backend):
    """Depth buffer is populated after render."""
    scene, camera = _make_scene()
    canvas = Canvas(32, 32)
    render(canvas, scene, camera, samples=1, max_bounces=1)
    depth = canvas.depth_to_numpy()
    assert depth.shape == (32, 32)
    assert (depth >= 0).sum() > 0  # some hits
    assert (depth < 0).sum() > 0   # some background


def test_depth_values_correct(backend):
    """Hit depth matches expected camera-to-surface distance."""
    scene, camera = _make_scene(z=0.0)
    canvas = Canvas(32, 32)
    render(canvas, scene, camera, samples=1, max_bounces=1)
    depth = canvas.depth_to_numpy()
    hit_depths = depth[depth >= 0]
    # Camera at z=3, triangle at z=0 → depth ≈ 3
    assert hit_depths.min() == pytest.approx(3.0, abs=0.2)


def test_depth_background_is_negative(backend):
    """Background pixels have depth = -1."""
    scene, camera = _make_scene()
    canvas = Canvas(32, 32)
    render(canvas, scene, camera, samples=1, max_bounces=1)
    depth = canvas.depth_to_numpy()
    bg = depth[depth < 0]
    np.testing.assert_array_equal(bg, -1.0)


def test_depth_closer_surface(backend):
    """A closer surface produces smaller depth values."""
    scene_far, camera = _make_scene(z=0.0)   # distance ≈ 3
    scene_near, _ = _make_scene(z=2.0)        # distance ≈ 1

    canvas_far = Canvas(32, 32)
    render(canvas_far, scene_far, camera, samples=1, max_bounces=1)
    far_depth = canvas_far.depth_to_numpy()

    canvas_near = Canvas(32, 32)
    render(canvas_near, scene_near, camera, samples=1, max_bounces=1)
    near_depth = canvas_near.depth_to_numpy()

    far_hits = far_depth[far_depth >= 0]
    near_hits = near_depth[near_depth >= 0]
    assert near_hits.mean() < far_hits.mean()
