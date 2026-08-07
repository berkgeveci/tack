"""Tests for the unified render() dispatcher."""

import numpy as np
import pytest

import tack
from tack.rendering import (
    Actor,
    Canvas,
    PerspectiveCamera,
    PointLight,
    Scene,
    TransferFunction,
    Volume,
    render,
)


@pytest.fixture(autouse=True)
def init_cpu():
    tack.init(arch=tack.cpu)


def _make_triangle_scene():
    pts_np = np.array([-1, -1, 0, 1, -1, 0, 0, 1, 0], dtype=np.float32)
    conn_np = np.array([0, 1, 2], dtype=np.int32)
    pts = tack.field(dtype=tack.f32, shape=(pts_np.size,))
    pts.from_numpy(pts_np)
    conn = tack.field(dtype=tack.i32, shape=(conn_np.size,))
    conn.from_numpy(conn_np)
    scene = Scene()
    scene.add(Actor(pts, conn, color=(0.8, 0.2, 0.2)))
    scene.add(PointLight(position=(0, 5, 5), intensity=50.0))
    return scene


def _make_volume_scene():
    N = 8
    data = np.ones(N**3, dtype=np.float32) * 0.5
    tf = TransferFunction('grayscale', opacity_func=lambda t: 0.5,
                          range=(0.0, 1.0))
    vol = Volume(data, dims=(N, N, N), origin=(-1, -1, -1),
                 spacing=(2.0/(N-1), 2.0/(N-1), 2.0/(N-1)),
                 transfer_function=tf, opacity_scale=10.0)
    scene = Scene()
    scene.add(vol)
    return scene


def _camera_and_canvas(size=32):
    camera = PerspectiveCamera(
        position=(0, 0, 4), look_at=(0, 0, 0), fov=60,
        width=size, height=size)
    canvas = Canvas(size, size)
    return camera, canvas


class TestUnifiedRender:
    def test_surface_only(self):
        scene = _make_triangle_scene()
        camera, canvas = _camera_and_canvas()
        render(canvas, scene, camera, samples=1, max_bounces=0,
               background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy()
        assert img[:, :, :3].max() > 0

    def test_volume_only(self):
        scene = _make_volume_scene()
        camera, canvas = _camera_and_canvas()
        render(canvas, scene, camera, background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy()
        assert img[:, :, :3].max() > 0

    def test_empty_scene(self):
        """Empty scene should not crash."""
        scene = Scene()
        camera, canvas = _camera_and_canvas()
        render(canvas, scene, camera)

    def test_mixed_scene(self):
        """Scene with both surfaces and volumes should render without error."""
        scene = _make_triangle_scene()
        N = 8
        data = np.ones(N**3, dtype=np.float32) * 0.5
        tf = TransferFunction('viridis', opacity_func=lambda t: 0.3,
                              range=(0.0, 1.0))
        vol = Volume(data, dims=(N, N, N), origin=(-1, -1, -1),
                     spacing=(2.0/(N-1), 2.0/(N-1), 2.0/(N-1)),
                     transfer_function=tf, opacity_scale=5.0)
        scene.add(vol)
        camera, canvas = _camera_and_canvas()
        render(canvas, scene, camera, samples=1, max_bounces=0,
               background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy()
        assert img[:, :, :3].max() > 0

    def test_backward_compat_surface_args(self):
        """Surface-specific args (samples, max_bounces) should work."""
        scene = _make_triangle_scene()
        camera, canvas = _camera_and_canvas()
        render(canvas, scene, camera, samples=2, max_bounces=1,
               light_position=(0, 5, 5), light_intensity=80.0)
        img = canvas.to_numpy()
        assert img[:, :, :3].max() > 0
