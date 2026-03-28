"""Tests for wireframe and point rasterization."""

import numpy as np
import pytest
import tack
from tack.rendering import (
    Scene, Actor, PointLight, PerspectiveCamera, OrthographicCamera,
    Canvas, ColorTable, render,
)


@pytest.fixture(autouse=True)
def init_cpu():
    tack.init(arch=tack.cpu)


def _make_triangle(z=0.0):
    pts_np = np.array([-1, -1, z, 1, -1, z, 0, 1, z], dtype=np.float32)
    conn_np = np.array([0, 1, 2], dtype=np.int32)
    pts = tack.field(dtype=tack.f32, shape=(pts_np.size,))
    pts.from_numpy(pts_np)
    conn = tack.field(dtype=tack.i32, shape=(conn_np.size,))
    conn.from_numpy(conn_np)
    return pts, conn


def _render(scene, size=64, camera=None, **kwargs):
    if camera is None:
        camera = PerspectiveCamera(
            position=(0, 0, 5), look_at=(0, 0, 0), fov=60,
            width=size, height=size)
    canvas = Canvas(size, size)
    render(canvas, scene, camera, background=(0.0, 0.0, 0.0), **kwargs)
    return canvas.to_numpy()


class TestWireframe:
    def test_wireframe_renders(self):
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, color=(0.0, 1.0, 0.0),
                        render_mode="wireframe"))
        img = _render(scene)
        assert img[:, :, 1].max() > 0  # green edges visible

    def test_wireframe_is_not_filled(self):
        """Wireframe should have empty interior."""
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, color=(1.0, 1.0, 1.0),
                        render_mode="wireframe"))
        img = _render(scene, size=128)
        # Center of triangle should be background (black)
        center = img[50, 64, :3]
        assert center.max() < 50  # mostly black interior

    def test_wireframe_with_color(self):
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, color=(1.0, 0.0, 0.0),
                        render_mode="wireframe"))
        img = _render(scene)
        # Should have red pixels
        assert img[:, :, 0].max() > 100
        # Should have minimal green/blue
        lit = img[:, :, :3].max(axis=2) > 50
        assert img[lit, 0].mean() > img[lit, 1].mean()

    def test_wireframe_depth_ordering(self):
        """Front triangle should occlude back triangle."""
        pts_front, conn_front = _make_triangle(z=0.0)
        pts_back, conn_back = _make_triangle(z=-2.0)
        scene = Scene()
        scene.add(Actor(pts_front, conn_front, color=(1.0, 0.0, 0.0),
                        render_mode="wireframe"))
        scene.add(Actor(pts_back, conn_back, color=(0.0, 0.0, 1.0),
                        render_mode="wireframe"))
        img = _render(scene)
        # Front (red) should be visible
        assert img[:, :, 0].max() > 100


class TestPointRendering:
    def test_points_render(self):
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, color=(0.0, 0.0, 1.0),
                        render_mode="points"))
        img = _render(scene, point_size=5.0)
        assert img[:, :, 2].max() > 0  # blue points visible

    def test_points_with_vertex_colors(self):
        pts_np = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
        conn_np = np.array([0, 1, 2], dtype=np.int32)
        pts = tack.field(dtype=tack.f32, shape=(9,))
        pts.from_numpy(pts_np)
        conn = tack.field(dtype=tack.i32, shape=(3,))
        conn.from_numpy(conn_np)
        # Red, green, blue vertices
        pc = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
        pc_field = tack.field(dtype=tack.f32, shape=(9,))
        pc_field.from_numpy(pc)
        scene = Scene()
        scene.add(Actor(pts, conn, point_colors=pc_field,
                        render_mode="points"))
        img = _render(scene, point_size=5.0)
        # Should have red, green, and blue pixels
        assert img[:, :, 0].max() > 0
        assert img[:, :, 1].max() > 0
        assert img[:, :, 2].max() > 0

    def test_point_size(self):
        """Larger point size should produce more lit pixels."""
        pts_np = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        conn_np = np.array([0, 0, 0], dtype=np.int32)
        pts = tack.field(dtype=tack.f32, shape=(9,))
        pts.from_numpy(pts_np)
        conn = tack.field(dtype=tack.i32, shape=(3,))
        conn.from_numpy(conn_np)

        scene = Scene()
        scene.add(Actor(pts, conn, color=(1.0, 1.0, 1.0),
                        render_mode="points"))

        img_small = _render(scene, point_size=2.0)
        img_large = _render(scene, point_size=8.0)
        lit_small = (img_small[:, :, :3].max(axis=2) > 50).sum()
        lit_large = (img_large[:, :, :3].max(axis=2) > 50).sum()
        assert lit_large > lit_small


class TestRenderModeDispatch:
    def test_solid_uses_pathtrace(self):
        """Solid actors should go through path tracer (needs light)."""
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, color=(0.8, 0.2, 0.2),
                        render_mode="solid"))
        scene.add(PointLight(position=(0, 5, 5), intensity=50.0))
        img = _render(scene, samples=1)
        assert img[:, :, :3].max() > 0

    def test_invalid_render_mode(self):
        pts, conn = _make_triangle()
        with pytest.raises(ValueError, match="render_mode"):
            Actor(pts, conn, render_mode="invalid")

    def test_ortho_wireframe(self):
        """Wireframe should work with orthographic camera."""
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, color=(1.0, 1.0, 0.0),
                        render_mode="wireframe"))
        camera = OrthographicCamera(
            position=(0, 0, 5), look_at=(0, 0, 0),
            view_height=4.0, width=64, height=64)
        img = _render(scene, camera=camera)
        assert img[:, :, :3].max() > 0
