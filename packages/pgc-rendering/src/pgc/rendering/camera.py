"""Cameras for ray generation: perspective and orthographic.

Both camera types store precomputed ray parameters as scalar attributes
so that ray generation in kernels is a simple linear combination.

For each pixel (px, py):
  direction = corner + dx * px + dy * py   (normalized in kernel)
  origin    = pos + odx * px + ody * py

Perspective: direction varies per pixel, origin is constant (odx=ody=0).
Orthographic: direction is constant, origin varies per pixel.
"""

import math
import numpy as np
import pgc


def _set_camera_attrs(obj, pos, corner, dx, dy, odx, ody, width, height,
                      right, up, forward):
    """Set the standard camera scalar attributes."""
    obj.pos_x = float(pos[0])
    obj.pos_y = float(pos[1])
    obj.pos_z = float(pos[2])
    obj.corner_x = float(corner[0])
    obj.corner_y = float(corner[1])
    obj.corner_z = float(corner[2])
    obj.dx_x = float(dx[0])
    obj.dx_y = float(dx[1])
    obj.dx_z = float(dx[2])
    obj.dy_x = float(dy[0])
    obj.dy_y = float(dy[1])
    obj.dy_z = float(dy[2])
    obj.odx_x = float(odx[0])
    obj.odx_y = float(odx[1])
    obj.odx_z = float(odx[2])
    obj.ody_x = float(ody[0])
    obj.ody_y = float(ody[1])
    obj.ody_z = float(ody[2])
    obj.width = width
    obj.height = height
    # CPU-side basis vectors (prefixed with _ to hide from @pgc.data_oriented)
    obj._right = right.copy()
    obj._up = up.copy()
    obj._forward = forward.copy()


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

        dx = right * (2.0 * half_w / width)
        dy = -true_up * (2.0 * half_h / height)
        corner = forward - right * half_w + true_up * half_h
        odx = np.zeros(3)
        ody = np.zeros(3)

        _set_camera_attrs(self, pos, corner, dx, dy, odx, ody,
                          width, height, right, true_up, forward)


@pgc.data_oriented
class OrthographicCamera:
    """Orthographic (parallel projection) camera.

    All rays have the same direction.  The ray origin varies per pixel,
    spanning a rectangle of the given ``height`` in world units (width
    is derived from the aspect ratio).

    Args:
        position: Camera position (center of the near plane).
        look_at: Point the camera faces.
        up: World up vector.
        view_height: Height of the view rectangle in world units.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    def __init__(self, position, look_at, up=(0, 1, 0), view_height=10.0,
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
        half_h = view_height * 0.5
        half_w = half_h * aspect

        # Direction is constant (forward), encoded as corner with zero dx/dy
        corner = forward
        dx = np.zeros(3)
        dy = np.zeros(3)

        # Origin varies per pixel
        odx = right * (2.0 * half_w / width)
        ody = -true_up * (2.0 * half_h / height)
        # Shift pos to top-left corner origin
        pos_corner = pos - right * half_w + true_up * half_h

        _set_camera_attrs(self, pos_corner, corner, dx, dy, odx, ody,
                          width, height, right, true_up, forward)
