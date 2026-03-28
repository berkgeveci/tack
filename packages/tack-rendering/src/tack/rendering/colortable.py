"""Color tables for mapping scalar fields to colors.

A ColorTable maps a scalar range to RGB colors using a sampled lookup
table.  Presets cover common scientific visualization needs (viridis,
cool-to-warm, inferno, etc.).  The lookup table is uploaded once as a
tack.field and reused across renders.
"""

import numpy as np
import tack


# ================================================================
# PRESET COLORMAP DATA
# ================================================================

def _viridis_data():
    """Viridis colormap — perceptually uniform, 9 control points."""
    return np.array([
        [0.267004, 0.004874, 0.329415],
        [0.282327, 0.140926, 0.457517],
        [0.253935, 0.265254, 0.529983],
        [0.206756, 0.371758, 0.553117],
        [0.163625, 0.471133, 0.558148],
        [0.127568, 0.566949, 0.550556],
        [0.134692, 0.658636, 0.517649],
        [0.477504, 0.821444, 0.318195],
        [0.993248, 0.906157, 0.143936],
    ], dtype=np.float32)


def _cool_to_warm_data():
    """Cool-to-warm diverging colormap (blue → white → red)."""
    return np.array([
        [0.230, 0.299, 0.754],
        [0.552, 0.600, 0.880],
        [0.866, 0.866, 0.950],
        [1.000, 1.000, 1.000],
        [0.950, 0.866, 0.866],
        [0.880, 0.600, 0.552],
        [0.706, 0.016, 0.150],
    ], dtype=np.float32)


def _inferno_data():
    """Inferno colormap — perceptually uniform, 9 control points."""
    return np.array([
        [0.001462, 0.000466, 0.013866],
        [0.087411, 0.044556, 0.224813],
        [0.258234, 0.038571, 0.406485],
        [0.416331, 0.090834, 0.432943],
        [0.578040, 0.148039, 0.404001],
        [0.735683, 0.215906, 0.329651],
        [0.882371, 0.319610, 0.212519],
        [0.978422, 0.557937, 0.034931],
        [0.988362, 0.998364, 0.644924],
    ], dtype=np.float32)


def _plasma_data():
    """Plasma colormap — perceptually uniform, 9 control points."""
    return np.array([
        [0.050383, 0.029803, 0.527975],
        [0.254627, 0.013882, 0.615419],
        [0.417642, 0.000564, 0.658390],
        [0.562738, 0.051545, 0.641509],
        [0.692840, 0.165141, 0.564522],
        [0.798216, 0.280197, 0.469538],
        [0.881443, 0.392529, 0.383229],
        [0.949217, 0.517763, 0.295662],
        [0.940015, 0.975158, 0.131326],
    ], dtype=np.float32)


def _grayscale_data():
    """Simple black-to-white grayscale."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)


def _rainbow_data():
    """Rainbow colormap (HSV-like)."""
    return np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)


_PRESETS = {
    'viridis': _viridis_data,
    'cool_to_warm': _cool_to_warm_data,
    'inferno': _inferno_data,
    'plasma': _plasma_data,
    'grayscale': _grayscale_data,
    'rainbow': _rainbow_data,
}


# ================================================================
# GPU KERNEL for scalar → color mapping
# ================================================================

@tack.kernel
def _map_scalars_to_colors(scalars, colors, lut, n_verts, n_lut,
                           scalar_min, scalar_inv_range):
    """Map per-vertex scalars to RGB colors via lookup table.

    Args:
        scalars: (n_verts,) f32 scalar values.
        colors: (n_verts * 3,) f32 output RGB in [0, 1].
        lut: (n_lut * 3,) f32 sampled color table.
        n_verts: number of vertices.
        n_lut: number of entries in lookup table.
        scalar_min: minimum of scalar range.
        scalar_inv_range: 1.0 / (scalar_max - scalar_min).
    """
    for i in range(n_verts):
        # Normalize scalar to [0, 1]
        t = (scalars[i] - scalar_min) * scalar_inv_range
        # Clamp
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        # Lookup index (linear interpolation between two entries)
        fidx = t * float(n_lut - 1)
        idx0 = int(fidx)
        if idx0 >= n_lut - 1:
            idx0 = n_lut - 2
        frac = fidx - float(idx0)
        idx1 = idx0 + 1

        r0 = lut[idx0 * 3]
        g0 = lut[idx0 * 3 + 1]
        b0 = lut[idx0 * 3 + 2]
        r1 = lut[idx1 * 3]
        g1 = lut[idx1 * 3 + 1]
        b1 = lut[idx1 * 3 + 2]

        colors[i * 3]     = r0 + frac * (r1 - r0)
        colors[i * 3 + 1] = g0 + frac * (g1 - g0)
        colors[i * 3 + 2] = b0 + frac * (b1 - b0)


# ================================================================
# COLORTABLE CLASS
# ================================================================

class ColorTable:
    """Maps a scalar range to RGB colors via a sampled lookup table.

    Args:
        preset: Name of a preset colormap.  One of: ``'viridis'``,
            ``'cool_to_warm'``, ``'inferno'``, ``'plasma'``,
            ``'grayscale'``, ``'rainbow'``.
        n_samples: Number of entries in the sampled lookup table.
            More samples give smoother gradients.  Default 256.
        range: ``(min, max)`` scalar range.  If ``None``, the range
            is auto-detected from the scalar data at render time.

    Examples::

        ct = ColorTable('viridis')
        ct = ColorTable('cool_to_warm', range=(-1.0, 1.0))
        actor = Actor(points, conn, scalars=pressure, color_table=ct)
    """

    def __init__(self, preset='viridis', n_samples=256, range=None):
        if preset not in _PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {', '.join(sorted(_PRESETS))}")
        self.preset = preset
        self.n_samples = n_samples
        self.range = range
        self._lut_np = self._build_lut(_PRESETS[preset](), n_samples)
        self._lut_field = None

    @staticmethod
    def _build_lut(control_points, n_samples):
        """Interpolate control points into a fixed-size lookup table."""
        n_cp = len(control_points)
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
        cp_t = np.linspace(0.0, 1.0, n_cp, dtype=np.float32)
        lut = np.zeros((n_samples, 3), dtype=np.float32)
        for c in range(3):
            lut[:, c] = np.interp(t, cp_t, control_points[:, c])
        return lut

    @staticmethod
    def available_presets():
        """Return list of available preset names."""
        return sorted(_PRESETS.keys())

    @property
    def lut_numpy(self):
        """(n_samples, 3) float32 numpy array of the sampled colors."""
        return self._lut_np

    def _get_lut_field(self):
        """Get or create the GPU field for the lookup table."""
        if self._lut_field is None:
            flat = self._lut_np.reshape(-1).astype(np.float32)
            self._lut_field = tack.field(dtype=tack.f32, shape=(flat.shape[0],))
            self._lut_field.from_numpy(flat)
        return self._lut_field

    def map_scalars(self, scalars_field, n_verts, scalar_range=None):
        """Map per-vertex scalars to RGB colors on GPU.

        Args:
            scalars_field: tack.field f32 (n_verts,) of scalar values.
            n_verts: number of vertices.
            scalar_range: (min, max) override.  If None, uses self.range,
                or auto-detects from data.

        Returns:
            tack.field f32 (n_verts * 3,) of interleaved RGB colors in [0, 1].
        """
        if scalar_range is not None:
            smin, smax = float(scalar_range[0]), float(scalar_range[1])
        elif self.range is not None:
            smin, smax = float(self.range[0]), float(self.range[1])
        else:
            # Auto-detect from data (requires host round-trip)
            data = scalars_field.to_numpy()
            smin, smax = float(data.min()), float(data.max())

        inv_range = 1.0 / (smax - smin) if smax > smin else 1.0

        colors = tack.field(dtype=tack.f32, shape=(n_verts * 3,))
        lut = self._get_lut_field()
        _map_scalars_to_colors(scalars_field, colors, lut, n_verts,
                               self.n_samples, smin, inv_range)
        return colors
