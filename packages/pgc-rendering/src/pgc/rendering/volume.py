"""Volume rendering for uniform grids.

Provides a GPU ray-casting volume renderer with front-to-back compositing
and transfer-function mapping.  Works with the existing PerspectiveCamera
and Canvas from pgc.rendering.
"""

import numpy as np
import pgc

from pgc.rendering.colortable import _PRESETS


# ================================================================
# TRANSFER FUNCTION
# ================================================================

class TransferFunction:
    """Maps scalar values to RGBA via a sampled lookup table.

    The color portion reuses ColorTable preset data.  Opacity is defined
    by a callable ``opacity_func(t) -> alpha`` where *t* is in [0, 1].

    Args:
        preset: ColorTable preset name for the color channels.
        opacity_func: Callable ``(t) -> alpha`` where *t* is normalized
            [0, 1] across the scalar range.  Default: ramp ``t * 0.5``.
        n_samples: Number of entries in the lookup table.
        range: ``(min, max)`` scalar range.  If ``None``, auto-detected
            from volume data at render time.

    Examples::

        tf = TransferFunction('viridis')
        tf = TransferFunction('cool_to_warm',
                              opacity_func=lambda t: 0.1 * t**2,
                              range=(-1.0, 1.0))
        tf = TransferFunction.from_numpy(rgba_array, range=(0, 1))
    """

    def __init__(self, preset='viridis', opacity_func=None, n_samples=256,
                 range=None):
        if preset not in _PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {', '.join(sorted(_PRESETS))}")
        if opacity_func is None:
            opacity_func = lambda t: t * 0.5  # noqa: E731

        self.preset = preset
        self.n_samples = n_samples
        self.range = range
        self._lut_np = self._build_rgba_lut(
            _PRESETS[preset](), opacity_func, n_samples)
        self._lut_field = None

    @classmethod
    def from_numpy(cls, rgba_np, range=None):
        """Create a TransferFunction from an explicit (n, 4) float32 RGBA array.

        Args:
            rgba_np: (n, 4) float32 array with R, G, B, A in [0, 1].
            range: (min, max) scalar range.
        """
        rgba_np = np.asarray(rgba_np, dtype=np.float32)
        if rgba_np.ndim != 2 or rgba_np.shape[1] != 4:
            raise ValueError("Expected (n, 4) array")
        obj = cls.__new__(cls)
        obj.preset = None
        obj.n_samples = rgba_np.shape[0]
        obj.range = range
        obj._lut_np = rgba_np.copy()
        obj._lut_field = None
        return obj

    @staticmethod
    def _build_rgba_lut(color_control_points, opacity_func, n_samples):
        """Interpolate color control points + opacity function into RGBA LUT."""
        from pgc.rendering.colortable import ColorTable
        color_lut = ColorTable._build_lut(color_control_points, n_samples)
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
        alpha = np.array([float(opacity_func(ti)) for ti in t],
                         dtype=np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
        rgba = np.zeros((n_samples, 4), dtype=np.float32)
        rgba[:, :3] = color_lut
        rgba[:, 3] = alpha
        return rgba

    @property
    def lut_numpy(self):
        """(n_samples, 4) float32 numpy array of RGBA values."""
        return self._lut_np

    def _get_lut_field(self):
        """Get or create the GPU field for the interleaved RGBA LUT."""
        if self._lut_field is None:
            flat = self._lut_np.reshape(-1).astype(np.float32)
            self._lut_field = pgc.field(dtype=pgc.f32, shape=(flat.shape[0],))
            self._lut_field.from_numpy(flat)
        return self._lut_field


# ================================================================
# VOLUME CLASS
# ================================================================

class Volume:
    """Uniform-grid volume for ray-casting rendering.

    Args:
        scalar_field: pgc.field f32 or numpy array holding the 3D scalar
            data in row-major (x varies fastest) order.
        dims: ``(nx, ny, nz)`` — number of *points* per axis.
        origin: ``(ox, oy, oz)`` — world-space origin of the grid.
        spacing: ``(sx, sy, sz)`` — cell size per axis.
        transfer_function: :class:`TransferFunction` instance.
        opacity_scale: Multiplier for opacity (density knob).  Default 8.0.
        max_steps: Maximum ray march steps.  Default 800.
    """

    def __init__(self, scalar_field, dims, origin=(0, 0, 0),
                 spacing=(1, 1, 1), transfer_function=None,
                 opacity_scale=8.0, max_steps=800):
        # Handle numpy input
        if isinstance(scalar_field, np.ndarray):
            flat = scalar_field.reshape(-1).astype(np.float32)
            f = pgc.field(dtype=pgc.f32, shape=(flat.shape[0],))
            f.from_numpy(flat)
            self.scalar_field = f
        else:
            self.scalar_field = scalar_field

        self.dims = tuple(int(d) for d in dims)
        self.origin = tuple(float(v) for v in origin)
        self.spacing = tuple(float(v) for v in spacing)
        self.opacity_scale = float(opacity_scale)
        self.max_steps = int(max_steps)

        if transfer_function is None:
            transfer_function = TransferFunction()
        self.transfer_function = transfer_function

        # Bounding box
        nx, ny, nz = self.dims
        ox, oy, oz = self.origin
        sx, sy, sz = self.spacing
        self.bounds_min = (ox, oy, oz)
        self.bounds_max = (ox + (nx - 1) * sx, oy + (ny - 1) * sy,
                           oz + (nz - 1) * sz)

        # Texture3d for trilinear interpolation
        self._texture = pgc.texture3d(self.scalar_field,
                                      shape=self.dims, interp='linear')

        # Step size: half the smallest cell dimension
        self.step_size = 0.5 * min(sx, sy, sz)

    @property
    def scalar_range(self):
        """(min, max) of the scalar field.  Cached after first access."""
        if not hasattr(self, '_scalar_range'):
            data = self.scalar_field.to_numpy()
            self._scalar_range = (float(data.min()), float(data.max()))
        return self._scalar_range


# ================================================================
# VOLUME RENDER CONFIGURATION (template)
# ================================================================

@pgc.data_oriented
class _VolumeConfig:
    """Compile-time parameters for volume ray casting."""

    def __init__(self, volume, scalar_range, bg_color):
        # Bounds
        self.bmin_x = float(volume.bounds_min[0])
        self.bmin_y = float(volume.bounds_min[1])
        self.bmin_z = float(volume.bounds_min[2])
        self.bmax_x = float(volume.bounds_max[0])
        self.bmax_y = float(volume.bounds_max[1])
        self.bmax_z = float(volume.bounds_max[2])
        # Extent for normalized coords
        self.ext_x = float(volume.bounds_max[0] - volume.bounds_min[0])
        self.ext_y = float(volume.bounds_max[1] - volume.bounds_min[1])
        self.ext_z = float(volume.bounds_max[2] - volume.bounds_min[2])
        # Scalar range
        self.vmin = float(scalar_range[0])
        self.vrange = float(scalar_range[1] - scalar_range[0])
        # Rendering params
        self.step_size = float(volume.step_size)
        self.opacity_scale = float(volume.opacity_scale)
        self.max_steps = int(volume.max_steps)
        self.tf_size = int(volume.transfer_function.n_samples)
        # Background
        self.bg_r = float(bg_color[0])
        self.bg_g = float(bg_color[1])
        self.bg_b = float(bg_color[2])


# ================================================================
# GPU HELPERS
# ================================================================

@pgc.func
def _apply_tf(tf, val, vmin, vrange, tf_size):
    """Look up RGBA from transfer function for a scalar value."""
    t = (val - vmin) / vrange
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    idx = int(t * float(tf_size - 1))
    if idx < 0:
        idx = 0
    if idx >= tf_size:
        idx = tf_size - 1
    base = idx * 4
    return tf[base], tf[base + 1], tf[base + 2], tf[base + 3]


# ================================================================
# VOLUME RENDER KERNEL
# ================================================================

@pgc.kernel
def _volume_render(fb_r, fb_g, fb_b,
                   tex: pgc.template(),
                   tf,
                   camera: pgc.template(),
                   config: pgc.template(),
                   width, height, n_pixels):
    """Cast one ray per pixel with front-to-back compositing."""

    for pid in range(n_pixels):
        px = pid % width
        py = pid // width

        # Ray from camera (supports both perspective and orthographic)
        ppx = float(px) + 0.5
        ppy = float(py) + 0.5
        rdx = camera.corner_x + camera.dx_x * ppx + camera.dy_x * ppy
        rdy = camera.corner_y + camera.dx_y * ppx + camera.dy_y * ppy
        rdz = camera.corner_z + camera.dx_z * ppx + camera.dy_z * ppy
        rd_len = sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
        rdx = rdx / rd_len
        rdy = rdy / rd_len
        rdz = rdz / rd_len
        rox = camera.pos_x + camera.odx_x * ppx + camera.ody_x * ppy
        roy = camera.pos_y + camera.odx_y * ppx + camera.ody_y * ppy
        roz = camera.pos_z + camera.odx_z * ppx + camera.ody_z * ppy

        # Ray-AABB intersection
        t_near = -3.4e38
        t_far = 3.4e38

        # X slab
        if abs(rdx) < 1.0e-10:
            if rox < config.bmin_x or rox > config.bmax_x:
                t_near = 3.4e38
                t_far = -3.4e38
        else:
            t1 = (config.bmin_x - rox) / rdx
            t2 = (config.bmax_x - rox) / rdx
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

        # Y slab
        if abs(rdy) < 1.0e-10:
            if roy < config.bmin_y or roy > config.bmax_y:
                t_near = 3.4e38
                t_far = -3.4e38
        else:
            t1 = (config.bmin_y - roy) / rdy
            t2 = (config.bmax_y - roy) / rdy
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

        # Z slab
        if abs(rdz) < 1.0e-10:
            if roz < config.bmin_z or roz > config.bmax_z:
                t_near = 3.4e38
                t_far = -3.4e38
        else:
            t1 = (config.bmin_z - roz) / rdz
            t2 = (config.bmax_z - roz) / rdz
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

        if t_near < 0.0:
            t_near = 0.0

        # March and composite (front-to-back)
        cr = 0.0
        cg = 0.0
        cb = 0.0
        alpha = 0.0

        if t_near < t_far:
            t = t_near
            for _ in range(config.max_steps):
                if t >= t_far:
                    break
                if alpha > 0.99:
                    break

                sx = rox + t * rdx
                sy = roy + t * rdy
                sz = roz + t * rdz

                # Normalized coordinates [0, 1] for texture sampling
                tu = (sx - config.bmin_x) / (config.ext_x + 1.0e-20)
                tv = (sy - config.bmin_y) / (config.ext_y + 1.0e-20)
                tw = (sz - config.bmin_z) / (config.ext_z + 1.0e-20)

                val = tex.sample(tu, tv, tw)

                sr, sg, sb, sa = _apply_tf(tf, val, config.vmin,
                                           config.vrange, config.tf_size)

                # Opacity correction for step size
                sa = 1.0 - exp(0.0 - sa * config.opacity_scale * config.step_size)
                if sa > 1.0:
                    sa = 1.0

                # Front-to-back compositing
                cr = cr + (1.0 - alpha) * sa * sr
                cg = cg + (1.0 - alpha) * sa * sg
                cb = cb + (1.0 - alpha) * sa * sb
                alpha = alpha + (1.0 - alpha) * sa

                t = t + config.step_size

        # Blend with background
        fb_r[pid] = cr + (1.0 - alpha) * config.bg_r
        fb_g[pid] = cg + (1.0 - alpha) * config.bg_g
        fb_b[pid] = cb + (1.0 - alpha) * config.bg_b


# ================================================================
# RESOLVE KERNEL (gamma correction)
# ================================================================

@pgc.kernel
def _resolve_volume(canvas_r, canvas_g, canvas_b,
                    fb_r, fb_g, fb_b, n_pixels):
    """Apply gamma correction to volume-rendered output."""
    for i in range(n_pixels):
        r = fb_r[i]
        g = fb_g[i]
        b = fb_b[i]
        canvas_r[i] = pow(min(r, 1.0), 0.4545)
        canvas_g[i] = pow(min(g, 1.0), 0.4545)
        canvas_b[i] = pow(min(b, 1.0), 0.4545)


# ================================================================
# PUBLIC API
# ================================================================

def render_volume(canvas, volume, camera, background=(0.05, 0.05, 0.1)):
    """Ray-cast a volume into the canvas.

    Args:
        canvas: Canvas to render into.
        volume: :class:`Volume` instance.
        camera: PerspectiveCamera.
        background: RGB background color in [0, 1].
    """
    import time as _time

    # Scalar range
    tf = volume.transfer_function
    if tf.range is not None:
        scalar_range = tf.range
    else:
        scalar_range = volume.scalar_range

    config = _VolumeConfig(volume, scalar_range, background)
    tf_field = tf._get_lut_field()

    width = camera.width
    height = camera.height
    n_pixels = width * height

    fb_r = canvas.get_work_buffer('vol_fb_r', pgc.f32, (n_pixels,))
    fb_g = canvas.get_work_buffer('vol_fb_g', pgc.f32, (n_pixels,))
    fb_b = canvas.get_work_buffer('vol_fb_b', pgc.f32, (n_pixels,))

    _t0 = _time.perf_counter()
    _volume_render(fb_r, fb_g, fb_b,
                   volume._texture, tf_field,
                   camera, config,
                   width, height, n_pixels)
    _t_render = _time.perf_counter() - _t0

    _resolve_volume(canvas.color_r, canvas.color_g, canvas.color_b,
                    fb_r, fb_g, fb_b, n_pixels)

    print(f"  [volume] render={_t_render:.3f}s")
