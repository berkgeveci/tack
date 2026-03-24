"""Perspective camera for ray generation.

The camera precomputes an orthonormal basis and per-pixel deltas so that
ray direction computation in the kernel is a simple linear combination.
"""

import math
import numpy as np
import pgc


@pgc.data_oriented
class PerspectiveCamera:
    """Perspective camera.

    Stores precomputed ray generation parameters as scalar attributes
    so they become compile-time constants in kernels.

    The image convention is: pixel (0, 0) is top-left, y increases downward.
    """

    def __init__(self, position, look_at, up=(0, 1, 0), fov=45.0,
                 width=512, height=512):
        pos = np.asarray(position, dtype=np.float64)
        target = np.asarray(look_at, dtype=np.float64)
        up_vec = np.asarray(up, dtype=np.float64)

        forward = target - pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up_vec)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)

        aspect = width / height
        half_h = math.tan(math.radians(fov) * 0.5)
        half_w = half_h * aspect

        # Per-pixel step in world space
        dx = right * (2.0 * half_w / width)
        dy = -true_up * (2.0 * half_h / height)

        # Top-left corner direction (before normalization)
        corner = forward - right * half_w + true_up * half_h

        self.pos_x = float(pos[0])
        self.pos_y = float(pos[1])
        self.pos_z = float(pos[2])
        self.corner_x = float(corner[0])
        self.corner_y = float(corner[1])
        self.corner_z = float(corner[2])
        self.dx_x = float(dx[0])
        self.dx_y = float(dx[1])
        self.dx_z = float(dx[2])
        self.dy_x = float(dy[0])
        self.dy_y = float(dy[1])
        self.dy_z = float(dy[2])
        self.width = width
        self.height = height

        # Store basis vectors for CPU-side use (annotations, etc.)
        # Prefixed with _ so @pgc.data_oriented ignores them.
        self._right = right.copy()
        self._up = true_up.copy()
        self._forward = forward.copy()

    @pgc.func
    def ray_dx(self, px, py):
        return self.corner_x + self.dx_x * (float(px) + 0.5) + self.dy_x * (float(py) + 0.5)

    @pgc.func
    def ray_dy(self, px, py):
        return self.corner_y + self.dx_y * (float(px) + 0.5) + self.dy_y * (float(py) + 0.5)

    @pgc.func
    def ray_dz(self, px, py):
        return self.corner_z + self.dx_z * (float(px) + 0.5) + self.dy_z * (float(py) + 0.5)
