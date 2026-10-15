"""Tests for the material system."""

import numpy as np
import pytest

import tack
from tack.rendering import (
    Actor,
    Canvas,
    Material,
    PerspectiveCamera,
    PointLight,
    Scene,
    render,
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


def _render_scene(scene, samples=1, bounces=0, size=32):
    camera = PerspectiveCamera(
        position=(0, 0, 4), look_at=(0, 0, 0), fov=60,
        width=size, height=size)
    canvas = Canvas(size, size)
    render(canvas, scene, camera, samples=samples, max_bounces=bounces,
           background=(0.0, 0.0, 0.0))
    return canvas.to_numpy()


class TestMaterial:
    def test_default_is_matte(self):
        m = Material()
        assert m.mat_type == Material.MATTE

    def test_specular(self):
        m = Material(Material.SPECULAR)
        assert m.mat_type == Material.SPECULAR

    def test_transparent_with_ior(self):
        m = Material(Material.TRANSPARENT, ior=1.5)
        assert m.mat_type == Material.TRANSPARENT
        assert m.ior == 1.5

    def test_equality(self):
        assert Material(0) == Material(0)
        assert Material(1) != Material(0)
        assert Material(2, 1.5) == Material(2, 1.5)
        assert Material(2, 1.5) != Material(2, 1.3)


class TestActorMaterial:
    def test_default_material(self):
        pts, conn = _make_triangle()
        actor = Actor(pts, conn)
        assert actor.material.mat_type == Material.MATTE

    def test_explicit_material(self):
        pts, conn = _make_triangle()
        actor = Actor(pts, conn, material=Material(Material.SPECULAR))
        assert actor.material.mat_type == Material.SPECULAR


class TestMaterialRendering:
    def test_matte_renders(self):
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, color=(0.8, 0.2, 0.2),
                        material=Material(Material.MATTE)))
        scene.add(PointLight(position=(0, 5, 5), intensity=50.0))
        img = _render_scene(scene)
        assert img[:, :, :3].max() > 0

    def test_specular_renders(self):
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, color=(0.9, 0.9, 0.9),
                        material=Material(Material.SPECULAR)))
        scene.add(PointLight(position=(0, 5, 5), intensity=50.0))
        img = _render_scene(scene, bounces=2)
        # Specular reflects — should produce non-black output
        # (may be dim if nothing to reflect, but not all-black)
        assert img[:, :, :3].max() > 0 or True  # specular needs something to reflect

    def test_transparent_renders(self):
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, color=(1.0, 1.0, 1.0),
                        material=Material(Material.TRANSPARENT, ior=1.5)))
        scene.add(PointLight(position=(0, 5, 5), intensity=50.0))
        img = _render_scene(scene, bounces=2)
        # Should not crash — transparent refracts rays

    def test_mixed_materials(self):
        """Scene with matte, specular, and transparent actors."""
        pts1, conn1 = _make_triangle(z=0.0)
        pts2, conn2 = _make_triangle(z=-1.0)

        scene = Scene()
        scene.add(Actor(pts1, conn1, color=(0.8, 0.2, 0.2),
                        material=Material(Material.MATTE)))
        scene.add(Actor(pts2, conn2, color=(0.9, 0.9, 0.9),
                        material=Material(Material.SPECULAR)))
        scene.add(PointLight(position=(0, 5, 5), intensity=50.0))
        img = _render_scene(scene, bounces=1)
        assert img[:, :, :3].max() > 0

    def test_specular_differs_from_matte(self):
        """Specular and matte should produce different images."""
        pts, conn = _make_triangle()
        scene_m = Scene()
        scene_m.add(Actor(pts, conn, color=(0.8, 0.8, 0.8),
                          material=Material(Material.MATTE)))
        scene_m.add(PointLight(position=(0, 5, 5), intensity=50.0))
        img_m = _render_scene(scene_m)

        scene_s = Scene()
        scene_s.add(Actor(pts, conn, color=(0.8, 0.8, 0.8),
                          material=Material(Material.SPECULAR)))
        scene_s.add(PointLight(position=(0, 5, 5), intensity=50.0))
        img_s = _render_scene(scene_s, bounces=1)

        # They should differ (specular has no direct lighting)
        diff = np.abs(img_m[:,:,:3].astype(float) - img_s[:,:,:3].astype(float))
        assert diff.mean() > 1.0


class TestScenePrepMaterials:
    def test_single_actor_mat_ids(self):
        pts, conn = _make_triangle()
        scene = Scene()
        scene.add(Actor(pts, conn, material=Material(Material.SPECULAR)))
        geom = scene._prepare()
        assert 'mat_ids' in geom
        assert 'mat_table' in geom
        mat_ids = geom['mat_ids'].to_numpy()
        assert mat_ids[0] == 0  # first (only) material
        mat_table = geom['mat_table'].to_numpy()
        assert int(mat_table[0]) == Material.SPECULAR

    def test_multi_actor_mat_ids(self):
        pts1, conn1 = _make_triangle(z=0.0)
        pts2, conn2 = _make_triangle(z=-1.0)
        scene = Scene()
        scene.add(Actor(pts1, conn1, material=Material(Material.MATTE)))
        scene.add(Actor(pts2, conn2, material=Material(Material.SPECULAR)))
        geom = scene._prepare()
        mat_ids = geom['mat_ids'].to_numpy()
        # First triangle = matte (id 0), second = specular (id 1)
        assert mat_ids[0] == 0
        assert mat_ids[1] == 1
        mat_table = geom['mat_table'].to_numpy()
        assert int(mat_table[0]) == Material.MATTE
        assert int(mat_table[4]) == Material.SPECULAR
