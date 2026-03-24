"""Framebuffer for rendered images."""

import numpy as np
import pgc


class Canvas:
    """RGBA framebuffer backed by pgc fields.

    Stores color as three f32 fields (linear RGB, 0-1 range)
    and an optional depth buffer (f32) for rasterization.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        n = width * height
        self.color_r = pgc.field(dtype=pgc.f32, shape=(n,))
        self.color_g = pgc.field(dtype=pgc.f32, shape=(n,))
        self.color_b = pgc.field(dtype=pgc.f32, shape=(n,))
        self.depth = pgc.field(dtype=pgc.f32, shape=(n,))

    def to_numpy(self):
        """Return (H, W, 4) uint8 RGBA array."""
        r = self.color_r.to_numpy().reshape(self.height, self.width)
        g = self.color_g.to_numpy().reshape(self.height, self.width)
        b = self.color_b.to_numpy().reshape(self.height, self.width)
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        img[:, :, 0] = np.clip(r * 255, 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(g * 255, 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(b * 255, 0, 255).astype(np.uint8)
        img[:, :, 3] = 255
        return img
