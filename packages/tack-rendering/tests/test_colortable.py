"""Tests for ColorTable and scalar field coloring."""

import numpy as np
import pytest

import tack
from tack.rendering import Actor, Canvas, ColorTable, PerspectiveCamera, PointLight, Scene, render


@pytest.fixture(autouse=True)
def init_cpu():
    tack.init(arch=tack.cpu)


# ================================================================
# ColorTable unit tests
# ================================================================

class TestColorTablePresets:
    def test_available_presets(self):
        presets = ColorTable.available_presets()
        assert 'viridis' in presets
        assert 'cool_to_warm' in presets
        assert 'inferno' in presets
        assert 'plasma' in presets
        assert 'grayscale' in presets
        assert 'rainbow' in presets

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            ColorTable('nonexistent')

    def test_default_is_viridis(self):
        ct = ColorTable()
        assert ct.preset == 'viridis'

    @pytest.mark.parametrize('preset', ColorTable.available_presets())
    def test_lut_shape(self, preset):
        ct = ColorTable(preset, n_samples=64)
        assert ct.lut_numpy.shape == (64, 3)
        assert ct.lut_numpy.dtype == np.float32

    @pytest.mark.parametrize('preset', ColorTable.available_presets())
    def test_lut_values_in_range(self, preset):
        ct = ColorTable(preset)
        assert ct.lut_numpy.min() >= 0.0
        assert ct.lut_numpy.max() <= 1.0

    def test_custom_n_samples(self):
        ct = ColorTable('viridis', n_samples=512)
        assert ct.lut_numpy.shape == (512, 3)


class TestColorTableMapping:
    def test_grayscale_endpoints(self):
        """Grayscale: 0 → black, 1 → white."""
        ct = ColorTable('grayscale', n_samples=256)
        lut = ct.lut_numpy
        np.testing.assert_allclose(lut[0], [0.0, 0.0, 0.0], atol=0.01)
        np.testing.assert_allclose(lut[-1], [1.0, 1.0, 1.0], atol=0.01)

    def test_map_scalars_uniform(self):
        """All scalars == 0.5 should map to the midpoint color."""
        ct = ColorTable('grayscale', n_samples=256, range=(0.0, 1.0))
        n = 10
        scalars = tack.field(dtype=tack.f32, shape=(n,))
        scalars.from_numpy(np.full(n, 0.5, dtype=np.float32))

        colors = ct.map_scalars(scalars, n)
        result = colors.to_numpy().reshape(-1, 3)
        # Midpoint of grayscale should be ~0.5
        np.testing.assert_allclose(result, 0.5, atol=0.02)

    def test_map_scalars_endpoints(self):
        """Min scalar → first color, max scalar → last color."""
        ct = ColorTable('grayscale', n_samples=256, range=(0.0, 1.0))
        scalars = tack.field(dtype=tack.f32, shape=(2,))
        scalars.from_numpy(np.array([0.0, 1.0], dtype=np.float32))

        colors = ct.map_scalars(scalars, 2)
        result = colors.to_numpy().reshape(-1, 3)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=0.01)
        np.testing.assert_allclose(result[1], [1.0, 1.0, 1.0], atol=0.01)

    def test_map_scalars_clamping(self):
        """Values outside range should clamp to endpoints."""
        ct = ColorTable('grayscale', n_samples=256, range=(0.0, 1.0))
        scalars = tack.field(dtype=tack.f32, shape=(2,))
        scalars.from_numpy(np.array([-10.0, 10.0], dtype=np.float32))

        colors = ct.map_scalars(scalars, 2)
        result = colors.to_numpy().reshape(-1, 3)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=0.01)
        np.testing.assert_allclose(result[1], [1.0, 1.0, 1.0], atol=0.01)

    def test_map_scalars_auto_range(self):
        """Without explicit range, auto-detect from data."""
        ct = ColorTable('grayscale', n_samples=256)
        scalars = tack.field(dtype=tack.f32, shape=(3,))
        scalars.from_numpy(np.array([10.0, 15.0, 20.0], dtype=np.float32))

        colors = ct.map_scalars(scalars, 3)
        result = colors.to_numpy().reshape(-1, 3)
        # 10 → min (black), 20 → max (white), 15 → mid (grey)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=0.02)
        np.testing.assert_allclose(result[1], [0.5, 0.5, 0.5], atol=0.02)
        np.testing.assert_allclose(result[2], [1.0, 1.0, 1.0], atol=0.02)

    def test_map_scalars_override_range(self):
        """Scalar range override via map_scalars argument."""
        ct = ColorTable('grayscale', n_samples=256, range=(0.0, 100.0))
        scalars = tack.field(dtype=tack.f32, shape=(2,))
        scalars.from_numpy(np.array([0.0, 10.0], dtype=np.float32))

        # Override to [0, 10] instead of the CT's [0, 100]
        colors = ct.map_scalars(scalars, 2, scalar_range=(0.0, 10.0))
        result = colors.to_numpy().reshape(-1, 3)
        np.testing.assert_allclose(result[1], [1.0, 1.0, 1.0], atol=0.01)

    def test_viridis_not_grayscale(self):
        """Viridis should produce different R, G, B channels."""
        ct = ColorTable('viridis', n_samples=256)
        lut = ct.lut_numpy
        # At the midpoint, viridis has distinct R, G, B
        mid = lut[128]
        assert not np.allclose(mid[0], mid[1], atol=0.05)


# ================================================================
# Actor scalar coloring integration
# ================================================================

def _make_triangle():
    """Simple single-triangle mesh for testing."""
    pts_np = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float32)
    conn_np = np.array([0, 1, 2], dtype=np.int32)
    pts = tack.field(dtype=tack.f32, shape=(pts_np.size,))
    pts.from_numpy(pts_np)
    conn = tack.field(dtype=tack.i32, shape=(conn_np.size,))
    conn.from_numpy(conn_np)
    return pts, conn


class TestActorScalarColoring:
    def test_scalars_without_colortable_raises(self):
        pts, conn = _make_triangle()
        scalars_np = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="scalars requires a color_table"):
            Actor(pts, conn, scalars=scalars_np)

    def test_actor_with_scalars_and_colortable(self):
        pts, conn = _make_triangle()
        scalars_np = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        ct = ColorTable('grayscale')
        actor = Actor(pts, conn, scalars=scalars_np, color_table=ct)
        assert actor.scalars is not None
        assert actor.color_table is ct

    def test_actor_scalars_as_field(self):
        pts, conn = _make_triangle()
        s = tack.field(dtype=tack.f32, shape=(3,))
        s.from_numpy(np.array([0.0, 0.5, 1.0], dtype=np.float32))
        ct = ColorTable('viridis')
        actor = Actor(pts, conn, scalars=s, color_table=ct)
        assert actor.scalars is s

    def test_point_colors_takes_precedence(self):
        """If both point_colors and scalars are given, point_colors wins."""
        pts, conn = _make_triangle()
        pc_np = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                         dtype=np.uint8)
        scalars_np = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        ct = ColorTable('grayscale')
        actor = Actor(pts, conn, point_colors=pc_np,
                      scalars=scalars_np, color_table=ct)
        # point_colors should be set (overrides scalar)
        assert actor.point_colors is not None


class TestScenePrepareScalars:
    def test_single_actor_scalar_coloring(self):
        """Scene._prepare() maps scalars → point_colors for single actor."""
        pts, conn = _make_triangle()
        scalars_np = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        ct = ColorTable('grayscale', range=(0.0, 1.0))
        actor = Actor(pts, conn, scalars=scalars_np, color_table=ct,
                      smooth=False)

        scene = Scene()
        scene.add(actor)
        geom = scene._prepare()

        assert geom['has_point_colors'] == 1
        colors = geom['point_colors'].to_numpy().reshape(-1, 3)
        # Vertex 0 (scalar=0) → black, vertex 2 (scalar=1) → white
        np.testing.assert_allclose(colors[0], [0.0, 0.0, 0.0], atol=0.02)
        np.testing.assert_allclose(colors[2], [1.0, 1.0, 1.0], atol=0.02)

    def test_multi_actor_mixed_coloring(self):
        """Multi-actor scene: one with scalars, one with uniform color."""
        pts1, conn1 = _make_triangle()
        pts2, conn2 = _make_triangle()
        scalars_np = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        ct = ColorTable('grayscale', range=(0.0, 1.0))

        actor1 = Actor(pts1, conn1, scalars=scalars_np, color_table=ct)
        actor2 = Actor(pts2, conn2, color=(1.0, 0.0, 0.0))

        scene = Scene()
        scene.add(actor1)
        scene.add(actor2)
        geom = scene._prepare()

        assert geom['has_point_colors'] == 1
        colors = geom['point_colors'].to_numpy().reshape(-1, 3)
        # First actor: scalar-mapped grayscale
        np.testing.assert_allclose(colors[0], [0.0, 0.0, 0.0], atol=0.02)
        np.testing.assert_allclose(colors[2], [1.0, 1.0, 1.0], atol=0.02)
        # Second actor: uniform red fill
        np.testing.assert_allclose(colors[3], [1.0, 0.0, 0.0], atol=0.02)
        np.testing.assert_allclose(colors[5], [1.0, 0.0, 0.0], atol=0.02)

    def test_point_colors_precedence_in_prepare(self):
        """point_colors should take precedence over scalars in _prepare()."""
        pts, conn = _make_triangle()
        pc_np = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                         dtype=np.uint8)
        scalars_np = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        ct = ColorTable('grayscale', range=(0.0, 1.0))
        actor = Actor(pts, conn, point_colors=pc_np,
                      scalars=scalars_np, color_table=ct)

        scene = Scene()
        scene.add(actor)
        geom = scene._prepare()

        assert geom['has_point_colors'] == 1
        colors = geom['point_colors'].to_numpy().reshape(-1, 3)
        # Should be the explicit red/green/blue, not grayscale
        np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=0.02)
        np.testing.assert_allclose(colors[1], [0.0, 1.0, 0.0], atol=0.02)


# ================================================================
# Full render integration test
# ================================================================

class TestRenderWithScalars:
    def test_render_produces_image(self):
        """Render a scalar-colored triangle and verify non-black output."""
        pts_np = np.array([
            -1, -1, 0,
             1, -1, 0,
             0,  1, 0,
        ], dtype=np.float32)
        conn_np = np.array([0, 1, 2], dtype=np.int32)

        pts = tack.field(dtype=tack.f32, shape=(pts_np.size,))
        pts.from_numpy(pts_np)
        conn = tack.field(dtype=tack.i32, shape=(conn_np.size,))
        conn.from_numpy(conn_np)

        scalars_np = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        ct = ColorTable('viridis', range=(0.0, 1.0))

        scene = Scene()
        scene.add(Actor(pts, conn, scalars=scalars_np, color_table=ct))
        scene.add(PointLight(position=(0, 5, 5), intensity=50.0))

        camera = PerspectiveCamera(
            position=(0, 0, 3), look_at=(0, 0, 0), fov=60,
            width=64, height=64)
        canvas = Canvas(64, 64)
        render(canvas, scene, camera, samples=1, max_bounces=0)

        img = canvas.to_numpy()
        # At least some pixels should be non-black (the triangle is visible)
        assert img[:, :, :3].max() > 0
