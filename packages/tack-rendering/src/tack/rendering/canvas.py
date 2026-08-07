"""Framebuffer for rendered images."""

import numpy as np

import tack


class Canvas:
    """RGBA framebuffer backed by tack fields.

    Stores color as three f32 fields (linear RGB, 0-1 range)
    and a depth buffer (f32) for rasterization.  Also caches work
    buffers used by the renderers to avoid per-frame allocations.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        n = width * height
        self.color_r = tack.field(dtype=tack.f32, shape=(n,))
        self.color_g = tack.field(dtype=tack.f32, shape=(n,))
        self.color_b = tack.field(dtype=tack.f32, shape=(n,))
        self.depth = tack.field(dtype=tack.f32, shape=(n,))
        self._work = {}  # cached work buffers keyed by (name, dtype, shape)

    def get_work_buffer(self, name, dtype, shape):
        """Get or create a reusable work buffer.

        Buffers are cached by (name, dtype, shape) and reused across
        renders to avoid per-frame GPU memory allocations.
        """
        key = (name, dtype, shape)
        if key not in self._work:
            self._work[key] = tack.field(dtype=dtype, shape=shape)
        return self._work[key]

    def depth_to_numpy(self):
        """Return (H, W) float32 depth buffer.

        Values are ray hit distances. -1 means no hit (background).
        Available after rendering.
        """
        return self.depth.to_numpy().reshape(self.height, self.width)

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
