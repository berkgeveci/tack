"""Tests for OrthographicCamera."""

import numpy as np
import pytest
import pgc
from pgc.rendering import (
    OrthographicCamera, PerspectiveCamera, Canvas, Scene, Actor,
    PointLight, render, Volume, TransferFunction, render_volume,
)


@pytest.fixture(autouse=True)
def init_cpu():
    pgc.init(arch=pgc.cpu)


def _make_triangle_scene():
    pts_np = np.array([-1, -1, 0, 1, -1, 0, 0, 1, 0], dtype=np.float32)
    conn_np = np.array([0, 1, 2], dtype=np.int32)
    pts = pgc.field(dtype=pgc.f32, shape=(pts_np.size,))
    pts.from_numpy(pts_np)
    conn = pgc.field(dtype=pgc.i32, shape=(conn_np.size,))
    conn.from_numpy(conn_np)
    scene = Scene()
    scene.add(Actor(pts, conn, color=(0.8, 0.2, 0.2)))
    scene.add(PointLight(position=(0, 5, 5), intensity=50.0))
    return scene


class TestOrthographicCamera:
    def test_construction(self):
        cam = OrthographicCamera(
            position=(0, 0, 5), look_at=(0, 0, 0),
            view_height=4.0, width=64, height=64)
        assert cam.width == 64
        assert cam.height == 64

    def test_parallel_rays(self):
        """All rays should have the same direction."""
        cam = OrthographicCamera(
            position=(0, 0, 5), look_at=(0, 0, 0),
            view_height=4.0, width=64, height=64)
        # corner encodes the direction, dx/dy are zero
        assert cam.dx_x == 0.0
        assert cam.dx_y == 0.0
        assert cam.dx_z == 0.0
        assert cam.dy_x == 0.0
        assert cam.dy_y == 0.0
        assert cam.dy_z == 0.0

    def test_varying_origin(self):
        """Ray origins should vary per pixel (odx/ody non-zero)."""
        cam = OrthographicCamera(
            position=(0, 0, 5), look_at=(0, 0, 0),
            view_height=4.0, width=64, height=64)
        # odx should be along right axis (x for this camera)
        assert abs(cam.odx_x) > 0

    def test_render_surface(self):
        """Orthographic render should produce a visible triangle."""
        scene = _make_triangle_scene()
        cam = OrthographicCamera(
            position=(0, 0, 5), look_at=(0, 0, 0),
            view_height=4.0, width=64, height=64)
        canvas = Canvas(64, 64)
        render(canvas, scene, cam, samples=1, max_bounces=0,
               background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy()
        assert img[:, :, :3].max() > 0

    def test_no_perspective_distortion(self):
        """Object should have same size regardless of distance (ortho property).

        Move camera far away — with ortho, the rendered object should
        be the same size. With perspective, it would shrink.
        """
        scene = _make_triangle_scene()

        cam_near = OrthographicCamera(
            position=(0, 0, 5), look_at=(0, 0, 0),
            view_height=4.0, width=64, height=64)
        cam_far = OrthographicCamera(
            position=(0, 0, 50), look_at=(0, 0, 0),
            view_height=4.0, width=64, height=64)

        c1 = Canvas(64, 64)
        render(c1, scene, cam_near, samples=1, max_bounces=0,
               background=(0.0, 0.0, 0.0))
        c2 = Canvas(64, 64)
        render(c2, scene, cam_far, samples=1, max_bounces=0,
               background=(0.0, 0.0, 0.0))

        img1 = c1.to_numpy()
        img2 = c2.to_numpy()
        # Count non-black pixels (the triangle)
        lit1 = (img1[:, :, :3].max(axis=2) > 10).sum()
        lit2 = (img2[:, :, :3].max(axis=2) > 10).sum()
        # Should be roughly the same size (within 10%)
        assert abs(lit1 - lit2) < 0.1 * max(lit1, lit2)

    def test_render_volume(self):
        """Orthographic camera should work with standalone volume renderer."""
        N = 8
        data = np.full(N**3, 0.5, dtype=np.float32)
        tf = TransferFunction('grayscale', opacity_func=lambda t: 0.5,
                              range=(0.0, 1.0))
        vol = Volume(data, dims=(N, N, N), origin=(-1, -1, -1),
                     spacing=(2.0/(N-1), 2.0/(N-1), 2.0/(N-1)),
                     transfer_function=tf, opacity_scale=10.0)
        cam = OrthographicCamera(
            position=(0, 0, 5), look_at=(0, 0, 0),
            view_height=4.0, width=32, height=32)
        canvas = Canvas(32, 32)
        render_volume(canvas, vol, cam, background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy()
        assert img[:, :, :3].max() > 0

    def test_basis_vectors(self):
        """Camera should store basis vectors for annotations."""
        cam = OrthographicCamera(
            position=(0, 0, 5), look_at=(0, 0, 0),
            view_height=4.0, width=64, height=64)
        assert hasattr(cam, '_right')
        assert hasattr(cam, '_up')
        assert hasattr(cam, '_forward')
        # Forward should point toward -Z (from pos=(0,0,5) to look_at=(0,0,0))
        np.testing.assert_allclose(cam._forward, [0, 0, -1], atol=0.01)
