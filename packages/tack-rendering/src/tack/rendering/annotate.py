"""Annotations for rendered images: color bar, axis indicator, text overlay.

All annotations operate on ``(H, W, 4) uint8`` numpy arrays (the output
of ``Canvas.to_numpy()``).  Text rendering uses Pillow when available;
lines and rectangles are drawn with pure numpy.
"""

import warnings
import numpy as np


# ================================================================
# Pillow helpers
# ================================================================

_pillow_warned = False


def _has_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        return True
    except ImportError:
        global _pillow_warned
        if not _pillow_warned:
            warnings.warn(
                "Pillow not installed — text annotations will be skipped. "
                "Install with: pip install Pillow",
                stacklevel=3)
            _pillow_warned = True
        return False


def _draw_text(image, text, x, y, color=(255, 255, 255), font_size=14,
               anchor="lt"):
    """Draw text onto an RGBA numpy image using Pillow.

    anchor uses PIL anchor codes: "lt" = left-top, "rt" = right-top,
    "lb" = left-bottom, "rb" = right-bottom, "mm" = middle-middle.
    """
    if not _has_pillow():
        return
    from PIL import Image as PILImage, ImageDraw, ImageFont
    pil_img = PILImage.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=(*color, 255), font=font, anchor=anchor)
    image[:] = np.array(pil_img)


# ================================================================
# Numpy drawing helpers
# ================================================================

def _draw_line(image, x0, y0, x1, y1, color, width=1):
    """Bresenham line drawing on an RGBA numpy image."""
    h, w = image.shape[:2]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    hw = width // 2

    while True:
        for wy in range(-hw, hw + 1):
            for wx in range(-hw, hw + 1):
                px, py = x0 + wx, y0 + wy
                if 0 <= px < w and 0 <= py < h:
                    image[py, px, :3] = color
                    image[py, px, 3] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_rect(image, x, y, w_rect, h_rect, color, fill=False):
    """Draw a rectangle on an RGBA numpy image."""
    h, w = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + w_rect)
    y1 = min(h, y + h_rect)
    if fill:
        image[y0:y1, x0:x1, :3] = color
        image[y0:y1, x0:x1, 3] = 255
    else:
        # Top and bottom edges
        image[y0, x0:x1, :3] = color
        image[y0, x0:x1, 3] = 255
        if y1 - 1 >= 0:
            image[y1 - 1, x0:x1, :3] = color
            image[y1 - 1, x0:x1, 3] = 255
        # Left and right edges
        image[y0:y1, x0, :3] = color
        image[y0:y1, x0, 3] = 255
        if x1 - 1 >= 0:
            image[y0:y1, x1 - 1, :3] = color
            image[y0:y1, x1 - 1, 3] = 255


# ================================================================
# TextOverlay
# ================================================================

class TextOverlay:
    """Draw text at a specified position.

    Requires Pillow for text rendering.

    Args:
        text: String to render.
        position: ``(x, y)`` pixel coordinates.
        color: ``(R, G, B)`` uint8 tuple.
        font_size: Font height in pixels.
        anchor: PIL anchor string — ``"lt"`` (left-top, default),
            ``"rt"`` (right-top), ``"lb"`` (left-bottom), ``"mm"`` (center).
    """

    def __init__(self, text, position=(10, 10), color=(255, 255, 255),
                 font_size=14, anchor="lt"):
        self.text = text
        self.position = position
        self.color = tuple(color)
        self.font_size = font_size
        self.anchor = anchor

    def draw(self, image):
        _draw_text(image, self.text, self.position[0], self.position[1],
                   self.color, self.font_size, self.anchor)


# ================================================================
# ColorBar
# ================================================================

class ColorBar:
    """Draw a color bar showing scalar-to-color mapping.

    Args:
        color_table: :class:`~tack.rendering.ColorTable` or
            :class:`~tack.rendering.TransferFunction` (uses ``.lut_numpy``).
        scalar_range: ``(min, max)`` for tick labels.
        position: ``"bottom"`` (default), ``"right"``, or explicit
            ``(x, y, width, height)`` in pixels.
        label: Optional title string.
        n_ticks: Number of tick labels.
        font_size: Font size for labels.
    """

    def __init__(self, color_table, scalar_range=(0.0, 1.0),
                 position="bottom", label=None, n_ticks=5, font_size=12):
        self.color_table = color_table
        self.scalar_range = scalar_range
        self.position = position
        self.label = label
        self.n_ticks = n_ticks
        self.font_size = font_size

    def draw(self, image):
        h, w = image.shape[:2]
        lut = self.color_table.lut_numpy[:, :3]  # (n, 3) float [0,1]
        n_lut = len(lut)

        # Determine bar rectangle
        if isinstance(self.position, str):
            if self.position == "bottom":
                margin = 20
                bar_h = 16
                bar_w = int(w * 0.6)
                bar_x = (w - bar_w) // 2
                bar_y = h - margin - bar_h - 20
            elif self.position == "right":
                margin = 20
                bar_w = 16
                bar_h = int(h * 0.6)
                bar_x = w - margin - bar_w - 40
                bar_y = (h - bar_h) // 2
            else:
                bar_x, bar_y, bar_w, bar_h = 20, h - 60, int(w * 0.6), 16
        else:
            bar_x, bar_y, bar_w, bar_h = self.position

        is_horizontal = bar_w >= bar_h

        # Draw gradient
        if is_horizontal:
            for c in range(bar_w):
                t = c / max(bar_w - 1, 1)
                idx = min(int(t * (n_lut - 1)), n_lut - 1)
                rgb = (lut[idx] * 255).astype(np.uint8)
                col = bar_x + c
                if 0 <= col < w:
                    r0 = max(0, bar_y)
                    r1 = min(h, bar_y + bar_h)
                    image[r0:r1, col, :3] = rgb
                    image[r0:r1, col, 3] = 255
        else:
            for r in range(bar_h):
                t = 1.0 - r / max(bar_h - 1, 1)  # bottom=min, top=max
                idx = min(int(t * (n_lut - 1)), n_lut - 1)
                rgb = (lut[idx] * 255).astype(np.uint8)
                row = bar_y + r
                if 0 <= row < h:
                    c0 = max(0, bar_x)
                    c1 = min(w, bar_x + bar_w)
                    image[row, c0:c1, :3] = rgb
                    image[row, c0:c1, 3] = 255

        # Border
        _draw_rect(image, bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2,
                   (200, 200, 200))

        # Tick marks and labels
        smin, smax = self.scalar_range
        for i in range(self.n_ticks):
            t = i / max(self.n_ticks - 1, 1)
            val = smin + t * (smax - smin)
            tick_label = f"{val:.2g}"

            if is_horizontal:
                tx = bar_x + int(t * (bar_w - 1))
                # Tick line
                _draw_line(image, tx, bar_y + bar_h, tx,
                           bar_y + bar_h + 4, (200, 200, 200))
                # Label
                _draw_text(image, tick_label, tx, bar_y + bar_h + 6,
                           (200, 200, 200), self.font_size, "mt")
            else:
                ty = bar_y + bar_h - int(t * (bar_h - 1))
                _draw_line(image, bar_x + bar_w, ty,
                           bar_x + bar_w + 4, ty, (200, 200, 200))
                _draw_text(image, tick_label, bar_x + bar_w + 6, ty,
                           (200, 200, 200), self.font_size, "lm")

        # Optional label
        if self.label:
            if is_horizontal:
                _draw_text(image, self.label, bar_x + bar_w // 2,
                           bar_y - 6, (220, 220, 220), self.font_size, "mb")
            else:
                _draw_text(image, self.label, bar_x + bar_w // 2,
                           bar_y - 6, (220, 220, 220), self.font_size, "mb")


# ================================================================
# AxisIndicator
# ================================================================

class AxisIndicator:
    """Draw an XYZ axis indicator showing camera orientation.

    Args:
        camera: :class:`~tack.rendering.PerspectiveCamera`.
        position: ``"bottom_left"`` (default), ``"bottom_right"``,
            ``"top_left"``, ``"top_right"``, or ``(cx, cy)`` center pixel.
        size: Radius in pixels.
        labels: Whether to draw X/Y/Z labels.
        font_size: Font size for labels.
    """

    def __init__(self, camera, position="bottom_left", size=50,
                 labels=True, font_size=11):
        self.camera = camera
        self.position = position
        self.size = size
        self.labels = labels
        self.font_size = font_size

    def draw(self, image):
        h, w = image.shape[:2]
        margin = self.size + 15

        # Determine center
        if isinstance(self.position, str):
            positions = {
                "bottom_left": (margin, h - margin),
                "bottom_right": (w - margin, h - margin),
                "top_left": (margin, margin),
                "top_right": (w - margin, margin),
            }
            cx, cy = positions.get(self.position, (margin, h - margin))
        else:
            cx, cy = self.position

        cam = self.camera
        right = cam._right
        up = cam._up

        axes = [
            (np.array([1, 0, 0], dtype=np.float64), (220, 60, 60), "X"),
            (np.array([0, 1, 0], dtype=np.float64), (60, 220, 60), "Y"),
            (np.array([0, 0, 1], dtype=np.float64), (60, 100, 220), "Z"),
        ]

        r = self.size * 0.7

        for axis, color, name in axes:
            # Project world axis onto screen
            sx = float(np.dot(axis, right))
            sy = -float(np.dot(axis, up))  # screen y is down
            length = max(abs(sx), abs(sy), 0.001)
            # Normalize but preserve relative lengths
            sx_pix = int(cx + sx * r)
            sy_pix = int(cy + sy * r)

            _draw_line(image, cx, cy, sx_pix, sy_pix, color, width=2)

            if self.labels:
                # Label slightly past the tip
                lx = int(cx + sx * (r + 12))
                ly = int(cy + sy * (r + 12))
                _draw_text(image, name, lx, ly, color,
                           self.font_size, "mm")


# ================================================================
# Convenience function
# ================================================================

def annotate(image, annotations):
    """Apply annotations to a numpy image.

    Args:
        image: ``(H, W, 4) uint8`` RGBA numpy array.
        annotations: List of annotation objects.

    Returns:
        Annotated ``(H, W, 4) uint8`` RGBA numpy array.
    """
    result = image.copy()
    for ann in annotations:
        ann.draw(result)
    return result
