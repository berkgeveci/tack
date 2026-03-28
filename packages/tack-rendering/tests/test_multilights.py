"""Tests for multiple lights support."""

import numpy as np
import pytest
import tack
from tack.rendering import (
    Scene, Actor, PointLight, Canvas, PerspectiveCamera, render,
)


@pytest.fixture(autouse=True)
def init_cpu():
    tack.init(arch=tack.cpu)


def _make_plane_scene(lights):
    """Create a simple plane scene with given lights for brightness testing."""
    pts_np = np.array([
        -2, 0, -2,
         2, 0, -2,
         2, 0,  2,
        -2, 0,  2,
    ], dtype=np.float32)
    conn_np = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

    pts = tack.field(dtype=tack.f32, shape=(pts_np.size,))
    pts.from_numpy(pts_np)
    conn = tack.field(dtype=tack.i32, shape=(conn_np.size,))
    conn.from_numpy(conn_np)

    scene = Scene()
    scene.add(Actor(pts, conn, color=(1.0, 1.0, 1.0)))
    for lt in lights:
        scene.add(lt)
    return scene


def _render_mean_brightness(scene, samples=1):
    """Render a small image and return the mean pixel brightness."""
    camera = PerspectiveCamera(
        position=(0, 3, 0.01), look_at=(0, 0, 0), fov=60,
        width=32, height=32)
    canvas = Canvas(32, 32)
    render(canvas, scene, camera, samples=samples, max_bounces=0,
           background=(0.0, 0.0, 0.0))
    img = canvas.to_numpy()
    return float(img[:, :, :3].mean())


class TestMultipleLights:
    def test_two_lights_brighter_than_one(self):
        """Two identical lights should produce a brighter image than one."""
        light = PointLight(position=(0, 5, 0), intensity=10.0)
        scene1 = _make_plane_scene([light])
        scene2 = _make_plane_scene([light, PointLight(position=(3, 5, 0),
                                                      intensity=10.0)])

        b1 = _render_mean_brightness(scene1)
        b2 = _render_mean_brightness(scene2)
        assert b2 > b1

    def test_single_light_backward_compat(self):
        """Single light should still work (backward compatibility)."""
        scene = _make_plane_scene([
            PointLight(position=(0, 5, 0), intensity=50.0)])
        b = _render_mean_brightness(scene)
        assert b > 0

    def test_no_scene_lights_uses_default(self):
        """No lights in scene → default fallback light via light_position."""
        scene = _make_plane_scene([])
        camera = PerspectiveCamera(
            position=(0, 3, 0.01), look_at=(0, 0, 0), fov=60,
            width=32, height=32)
        canvas = Canvas(32, 32)
        render(canvas, scene, camera, samples=1, max_bounces=0,
               background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy()
        # Default light should illuminate the plane
        assert img[:, :, :3].max() > 0

    def test_light_position_override(self):
        """light_position kwarg should override scene lights."""
        scene = _make_plane_scene([
            PointLight(position=(100, 100, 100), intensity=1.0)])
        camera = PerspectiveCamera(
            position=(0, 3, 0.01), look_at=(0, 0, 0), fov=60,
            width=32, height=32)
        canvas = Canvas(32, 32)
        # Override with a close, bright light
        render(canvas, scene, camera, samples=1, max_bounces=0,
               light_position=(0, 5, 0), light_intensity=50.0,
               background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy()
        assert img[:, :, :3].mean() > 10

    def test_three_lights(self):
        """Three lights should all contribute."""
        one_light = _make_plane_scene([
            PointLight(position=(0, 5, 0), intensity=30.0)])
        three_lights = _make_plane_scene([
            PointLight(position=(0, 5, 0), intensity=30.0),
            PointLight(position=(-3, 5, 0), intensity=30.0),
            PointLight(position=(3, 5, 0), intensity=30.0),
        ])
        b1 = _render_mean_brightness(one_light)
        b3 = _render_mean_brightness(three_lights)
        assert b3 > b1


class TestLightColor:
    def test_red_light_tints_red(self):
        """A red light should produce a red-tinted image."""
        scene = _make_plane_scene([
            PointLight(position=(0, 5, 0), intensity=80.0,
                       color=(1.0, 0.0, 0.0))])
        camera = PerspectiveCamera(
            position=(0, 3, 0.01), look_at=(0, 0, 0), fov=60,
            width=32, height=32)
        canvas = Canvas(32, 32)
        render(canvas, scene, camera, samples=1, max_bounces=0,
               background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy().astype(float)
        r_mean = img[:, :, 0].mean()
        g_mean = img[:, :, 1].mean()
        b_mean = img[:, :, 2].mean()
        # Red channel should dominate
        assert r_mean > g_mean * 2
        assert r_mean > b_mean * 2

    def test_colored_lights_mix(self):
        """Red + blue lights should produce both red and blue pixels."""
        scene = _make_plane_scene([
            PointLight(position=(-2, 5, 0), intensity=80.0,
                       color=(1.0, 0.0, 0.0)),
            PointLight(position=(2, 5, 0), intensity=80.0,
                       color=(0.0, 0.0, 1.0)),
        ])
        camera = PerspectiveCamera(
            position=(0, 3, 0.01), look_at=(0, 0, 0), fov=60,
            width=32, height=32)
        canvas = Canvas(32, 32)
        render(canvas, scene, camera, samples=1, max_bounces=0,
               background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy().astype(float)
        # Both red and blue channels should have significant values
        assert img[:, :, 0].max() > 20  # red from red light
        assert img[:, :, 2].max() > 20  # blue from blue light
