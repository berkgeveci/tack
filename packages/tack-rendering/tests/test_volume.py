"""Tests for volume rendering: TransferFunction, Volume, render_volume."""

import numpy as np
import pytest

import tack
from tack.rendering import (
    Canvas,
    PerspectiveCamera,
    Scene,
    TransferFunction,
    Volume,
    render_volume,
)


@pytest.fixture(autouse=True)
def init_cpu():
    tack.init(arch=tack.cpu)


# ================================================================
# TransferFunction tests
# ================================================================

class TestTransferFunction:
    def test_default_preset(self):
        tf = TransferFunction()
        assert tf.preset == 'viridis'
        assert tf.lut_numpy.shape == (256, 4)
        assert tf.lut_numpy.dtype == np.float32

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            TransferFunction('nonexistent')

    def test_custom_opacity(self):
        tf = TransferFunction('grayscale', opacity_func=lambda t: 1.0)
        # All alpha values should be 1.0
        np.testing.assert_allclose(tf.lut_numpy[:, 3], 1.0, atol=0.01)

    def test_zero_opacity(self):
        tf = TransferFunction('grayscale', opacity_func=lambda t: 0.0)
        np.testing.assert_allclose(tf.lut_numpy[:, 3], 0.0, atol=0.01)

    def test_from_numpy(self):
        rgba = np.zeros((64, 4), dtype=np.float32)
        rgba[:, 0] = 1.0  # all red
        rgba[:, 3] = 0.5  # half opaque
        tf = TransferFunction.from_numpy(rgba, range=(0.0, 1.0))
        assert tf.n_samples == 64
        assert tf.range == (0.0, 1.0)
        np.testing.assert_allclose(tf.lut_numpy[:, 0], 1.0)
        np.testing.assert_allclose(tf.lut_numpy[:, 3], 0.5)

    def test_from_numpy_bad_shape(self):
        with pytest.raises(ValueError, match="Expected"):
            TransferFunction.from_numpy(np.zeros((10, 3), dtype=np.float32))

    def test_lut_values_in_range(self):
        tf = TransferFunction('viridis')
        assert tf.lut_numpy.min() >= 0.0
        assert tf.lut_numpy.max() <= 1.0

    def test_gpu_field_upload(self):
        tf = TransferFunction('grayscale', n_samples=16)
        field = tf._get_lut_field()
        assert field.shape == (16 * 4,)
        data = field.to_numpy()
        # First entry: grayscale=black, check R=0
        assert data[0] < 0.01
        # Last entry: grayscale=white, check R~1
        assert data[(16 - 1) * 4] > 0.9


# ================================================================
# Volume tests
# ================================================================

def _make_test_volume(n=8):
    """Create a small uniform-grid volume with a sphere-like scalar field."""
    nx, ny, nz = n, n, n
    # Scalar: distance from center
    x = np.linspace(-1, 1, nx, dtype=np.float32)
    y = np.linspace(-1, 1, ny, dtype=np.float32)
    z = np.linspace(-1, 1, nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    scalars = np.sqrt(xx**2 + yy**2 + zz**2).astype(np.float32).ravel()

    tf = TransferFunction('cool_to_warm',
                          opacity_func=lambda t: 0.3 * (1.0 - t),
                          range=(0.0, 1.7))
    return Volume(scalars, dims=(nx, ny, nz),
                 origin=(-1, -1, -1), spacing=(2.0/(nx-1), 2.0/(ny-1), 2.0/(nz-1)),
                 transfer_function=tf)


class TestVolume:
    def test_construction_from_numpy(self):
        vol = _make_test_volume()
        assert vol.dims == (8, 8, 8)
        assert vol.origin == (-1.0, -1.0, -1.0)

    def test_construction_from_field(self):
        data = np.ones(27, dtype=np.float32)
        f = tack.field(dtype=tack.f32, shape=(27,))
        f.from_numpy(data)
        vol = Volume(f, dims=(3, 3, 3))
        assert vol.dims == (3, 3, 3)

    def test_bounds(self):
        vol = Volume(np.zeros(27, dtype=np.float32),
                     dims=(3, 3, 3), origin=(0, 0, 0), spacing=(1, 1, 1))
        assert vol.bounds_min == (0.0, 0.0, 0.0)
        assert vol.bounds_max == (2.0, 2.0, 2.0)

    def test_scalar_range(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        dtype=np.float32)
        vol = Volume(data, dims=(2, 2, 2))
        assert vol.scalar_range == (1.0, 8.0)

    def test_default_transfer_function(self):
        vol = Volume(np.zeros(8, dtype=np.float32), dims=(2, 2, 2))
        assert vol.transfer_function is not None

    def test_scene_add_volume(self):
        vol = _make_test_volume()
        scene = Scene()
        scene.add(vol)
        assert len(scene.volumes) == 1


# ================================================================
# Render integration tests
# ================================================================

class TestRenderVolume:
    def test_render_produces_image(self):
        """Volume render should produce a non-black image."""
        vol = _make_test_volume(n=16)
        camera = PerspectiveCamera(
            position=(0, 0, 4), look_at=(0, 0, 0), fov=60,
            width=32, height=32)
        canvas = Canvas(32, 32)
        render_volume(canvas, vol, camera)
        img = canvas.to_numpy()
        assert img[:, :, :3].max() > 0

    def test_render_fully_transparent_is_background(self):
        """Fully transparent volume should show only background."""
        tf = TransferFunction('grayscale', opacity_func=lambda t: 0.0)
        data = np.ones(27, dtype=np.float32)
        vol = Volume(data, dims=(3, 3, 3), origin=(-1, -1, -1),
                     spacing=(1, 1, 1), transfer_function=tf)
        camera = PerspectiveCamera(
            position=(0, 0, 5), look_at=(0, 0, 0), fov=60,
            width=16, height=16)
        canvas = Canvas(16, 16)
        bg = (0.2, 0.3, 0.4)
        render_volume(canvas, vol, camera, background=bg)
        img = canvas.to_numpy()
        # Center pixel should be roughly the background color (gamma-corrected)
        cx, cy = 8, 8
        r, g, b = img[cy, cx, 0], img[cy, cx, 1], img[cy, cx, 2]
        # Gamma-corrected background: pow(0.2, 0.4545) * 255 ≈ 117
        assert r > 50  # should be non-zero (background)

    def test_render_opaque_volume(self):
        """Fully opaque volume should block background completely."""
        tf = TransferFunction('grayscale', opacity_func=lambda t: 1.0,
                              range=(0.0, 1.0))
        data = np.full(64, 0.5, dtype=np.float32)
        vol = Volume(data, dims=(4, 4, 4), origin=(-1, -1, -1),
                     spacing=(2.0/3, 2.0/3, 2.0/3), transfer_function=tf,
                     opacity_scale=50.0)
        camera = PerspectiveCamera(
            position=(0, 0, 4), look_at=(0, 0, 0), fov=60,
            width=16, height=16)
        canvas = Canvas(16, 16)
        render_volume(canvas, vol, camera, background=(0.0, 0.0, 0.0))
        img = canvas.to_numpy()
        # Center pixel: opaque volume should contribute color, not black bg
        cx, cy = 8, 8
        assert img[cy, cx, :3].max() > 0
