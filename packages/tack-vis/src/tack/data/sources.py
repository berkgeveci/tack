"""Datasets made from formulas, for tests and examples: Viskores' sources
(``viskores::source``), on rectilinear grids with uniform spacing.

``wavelet`` is VTK's and Viskores' wavelet (vtkRTAnalyticSource): a
Gaussian plus three periodic terms, the point field ``RTData``. ``tangle``
is the tangle cube, a quartic with a surface of several handles at 0.5, the
point field ``tangle``.
"""

import numpy as np

import tack
from tack.data.dataset import Field, rectilinear_grid
from tack.data.spaces import H1

__all__ = ["tangle", "wavelet"]


def _extent(extent):
    e = np.asarray(extent, int).reshape(-1)
    if e.size == 1:
        lo, hi = np.full(3, -abs(e[0])), np.full(3, abs(e[0]))
    elif e.size == 6:
        lo, hi = e[0::2], e[1::2]
    else:
        raise ValueError("an extent is one number n (-n to n) or six: x0, x1, y0, y1, z0, z1")
    if (hi < lo).any():
        raise ValueError(f"extent {extent} runs backwards")
    return lo, hi


@tack.kernel
def _wavelet(out, nx, ny, x0, y0, z0, sx, sy, sz, cx, cy, cz, kx, ky, kz, fx, fy, fz,
             mx, my, mz, maximum, temp2):
    for p in range(out.shape[0]):
        i = p % nx
        j = (p // nx) % ny
        k = p // (nx * ny)
        ax = (cx - (x0 + i) * sx) * kx
        ay = (cy - (y0 + j) * sy) * ky
        az = (cz - (z0 + k) * sz) * kz
        out[p] = (maximum * exp(-(ax * ax + ay * ay + az * az) * temp2)
                  + mx * sin(fx * ax) + my * sin(fy * ay) + mz * cos(fz * az))


def wavelet(extent=10, spacing=1.0, center=(0.0, 0.0, 0.0), maximum=255.0,
            frequency=(60.0, 30.0, 40.0), magnitude=(10.0, 18.0, 5.0),
            standard_deviation=0.5, dtype=tack.f32):
    """VTK's and Viskores' wavelet on the grid of point indices ``extent`` (``n``:
    -n to n on every axis, or ``x0, x1, y0, y1, z0, z1``), points at index times
    ``spacing``: ``maximum * exp(-|a|^2 / (2 sd^2))`` plus ``magnitude`` times
    sin, sin and cos of ``frequency * a``, where ``a`` is ``center`` minus the
    point, each axis divided by its extent's length."""
    lo, hi = _extent(extent)
    s = np.broadcast_to(np.asarray(spacing, float), (3,))
    axes = [(np.arange(lo[d], hi[d] + 1) * s[d]).astype(dtype.numpy_dtype) for d in range(3)]
    data = rectilinear_grid(*axes, dtype=dtype)
    n = data.num_points
    out = tack.field(dtype, shape=(n,))
    scale = [1.0 / (hi[d] - lo[d]) if hi[d] > lo[d] else 1.0 for d in range(3)]
    _wavelet(out, int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1), *map(float, lo),
             *map(float, s), *map(float, center), *scale, *map(float, frequency),
             *map(float, magnitude), float(maximum),
             1.0 / (2.0 * standard_deviation * standard_deviation))
    data.fields["RTData"] = Field(H1(data), out)
    return data


@tack.kernel
def _tangle(out, nx, ny, cx, cy, cz):
    for p in range(out.shape[0]):
        i = p % nx
        j = (p // nx) % ny
        k = p // (nx * ny)
        x = 3.0 * (-1.0 + 2.0 * i / cx)
        y = 3.0 * (-1.0 + 2.0 * j / cy)
        z = 3.0 * (-1.0 + 2.0 * k / cz)
        out[p] = (x * x * x * x - 5.0 * x * x + y * y * y * y - 5.0 * y * y
                  + z * z * z * z - 5.0 * z * z + 11.8) * 0.2 + 0.5


def tangle(dims=(16, 16, 16), dtype=tack.f32):
    """Viskores' tangle on ``dims`` points per axis over [0, 1]^3: the quartic
    ``(x^4 - 5x^2 + y^4 - 5y^2 + z^4 - 5z^2 + 11.8) / 5 + 1/2`` of the point
    mapped to [-3, 3]^3, the point field ``tangle``."""
    dims = np.broadcast_to(np.asarray(dims, int), (3,))
    if (dims < 2).any():
        raise ValueError("tangle needs at least two points per axis")
    cells = dims - 1
    axes = [(np.arange(dims[d]) / cells[d]).astype(dtype.numpy_dtype) for d in range(3)]
    data = rectilinear_grid(*axes, dtype=dtype)
    out = tack.field(dtype, shape=(data.num_points,))
    _tangle(out, int(dims[0]), int(dims[1]), *map(float, cells))
    data.fields["tangle"] = Field(H1(data), out)
    return data
