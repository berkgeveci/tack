"""Tests for annotations: ColorBar, AxisIndicator, TextOverlay."""

import numpy as np
import pytest
from pgc.rendering import (
    ColorTable, PerspectiveCamera,
    ColorBar, AxisIndicator, TextOverlay, annotate,
)
from pgc.rendering.annotate import _draw_line, _draw_rect


# ================================================================
# Drawing helpers
# ================================================================

class TestDrawLine:
    def test_horizontal_line(self):
        img = np.zeros((10, 20, 4), dtype=np.uint8)
        _draw_line(img, 2, 5, 18, 5, (255, 0, 0))
        assert img[5, 10, 0] == 255  # red pixel on the line
        assert img[0, 10, 0] == 0    # off the line

    def test_vertical_line(self):
        img = np.zeros((20, 10, 4), dtype=np.uint8)
        _draw_line(img, 5, 2, 5, 18, (0, 255, 0))
        assert img[10, 5, 1] == 255

    def test_diagonal_line(self):
        img = np.zeros((20, 20, 4), dtype=np.uint8)
        _draw_line(img, 0, 0, 19, 19, (0, 0, 255))
        assert img[10, 10, 2] == 255

    def test_clipping(self):
        """Line extending outside image should not crash."""
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        _draw_line(img, -5, 5, 15, 5, (255, 255, 255))
        assert img[5, 5, 0] == 255


class TestDrawRect:
    def test_outline(self):
        img = np.zeros((20, 20, 4), dtype=np.uint8)
        _draw_rect(img, 5, 5, 10, 10, (255, 0, 0))
        assert img[5, 5, 0] == 255   # top-left corner
        assert img[10, 10, 0] == 0   # interior should be empty

    def test_fill(self):
        img = np.zeros((20, 20, 4), dtype=np.uint8)
        _draw_rect(img, 5, 5, 10, 10, (0, 255, 0), fill=True)
        assert img[10, 10, 1] == 255  # interior should be filled


# ================================================================
# TextOverlay
# ================================================================

class TestTextOverlay:
    def test_draw_modifies_image(self):
        img = np.zeros((64, 128, 4), dtype=np.uint8)
        overlay = TextOverlay("Hello", position=(10, 10))
        overlay.draw(img)
        # If Pillow is available, pixels should be modified
        try:
            import PIL  # noqa: F401
            assert img[10:30, 10:80, :3].max() > 0
        except ImportError:
            pass  # no Pillow, text silently skipped


# ================================================================
# ColorBar
# ================================================================

class TestColorBar:
    def test_horizontal_gradient(self):
        ct = ColorTable('grayscale')
        img = np.zeros((100, 200, 4), dtype=np.uint8)
        bar = ColorBar(ct, scalar_range=(0.0, 1.0), position=(20, 40, 160, 16))
        bar.draw(img)
        # Left side should be dark, right side should be bright
        left_brightness = img[48, 25, :3].mean()
        right_brightness = img[48, 175, :3].mean()
        assert right_brightness > left_brightness

    def test_bottom_position(self):
        ct = ColorTable('viridis')
        img = np.zeros((200, 300, 4), dtype=np.uint8)
        bar = ColorBar(ct, scalar_range=(-1.0, 1.0), position="bottom")
        bar.draw(img)
        # Bottom portion should have non-zero pixels (the bar)
        assert img[150:, :, :3].max() > 0

    def test_vertical_bar(self):
        ct = ColorTable('inferno')
        img = np.zeros((200, 200, 4), dtype=np.uint8)
        bar = ColorBar(ct, scalar_range=(0.0, 100.0),
                       position=(170, 20, 12, 160))
        bar.draw(img)
        # Right side should have the bar
        assert img[100, 175, :3].max() > 0


# ================================================================
# AxisIndicator
# ================================================================

class TestAxisIndicator:
    def test_draws_lines(self):
        cam = PerspectiveCamera(position=(0, 0, 5), look_at=(0, 0, 0),
                                fov=45, width=128, height=128)
        img = np.zeros((128, 128, 4), dtype=np.uint8)
        axis = AxisIndicator(cam, position="bottom_left", size=40)
        axis.draw(img)
        # Should have colored pixels in the bottom-left quadrant
        assert img[40:, :80, :3].max() > 0

    def test_camera_orientation(self):
        """Looking down Z-axis: X should point right, Y should point up."""
        cam = PerspectiveCamera(position=(0, 0, 5), look_at=(0, 0, 0),
                                fov=45, width=200, height=200)
        img = np.zeros((200, 200, 4), dtype=np.uint8)
        axis = AxisIndicator(cam, position=(100, 100), size=40,
                             labels=False)
        axis.draw(img)
        # X axis (red) should extend to the right of center
        right_red = img[95:105, 120:145, 0].max()
        assert right_red > 100  # red pixels to the right

    def test_all_corners(self):
        cam = PerspectiveCamera(position=(0, 0, 5), look_at=(0, 0, 0),
                                fov=45, width=200, height=200)
        for pos in ["bottom_left", "bottom_right", "top_left", "top_right"]:
            img = np.zeros((200, 200, 4), dtype=np.uint8)
            axis = AxisIndicator(cam, position=pos)
            axis.draw(img)
            assert img[:, :, :3].max() > 0


# ================================================================
# annotate convenience function
# ================================================================

class TestAnnotate:
    def test_returns_copy(self):
        img = np.zeros((64, 64, 4), dtype=np.uint8)
        result = annotate(img, [])
        assert result is not img
        np.testing.assert_array_equal(result, img)

    def test_applies_multiple(self):
        ct = ColorTable('viridis')
        cam = PerspectiveCamera(position=(0, 0, 5), look_at=(0, 0, 0),
                                fov=45, width=128, height=128)
        img = np.zeros((128, 128, 4), dtype=np.uint8)
        result = annotate(img, [
            ColorBar(ct, scalar_range=(0.0, 1.0)),
            AxisIndicator(cam),
        ])
        assert result[:, :, :3].max() > 0
