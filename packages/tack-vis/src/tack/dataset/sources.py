"""Datasets made from formulas, for tests and examples: Viskores' sources
(``viskores::source``), on rectilinear grids with uniform spacing.

``wavelet`` is VTK's and Viskores' wavelet (vtkRTAnalyticSource): a
Gaussian plus three periodic terms, the point field ``RTData``. ``tangle``
is the tangle cube, a quartic with a surface of several handles at 0.5, the
point field ``tangle``.
"""

import numpy as np

import tack
from tack.dataset.dataset import Field, uniform_grid
from tack.dataset.spaces import H1

__all__ = ["amr_gaussian_pulse", "tangle", "wavelet"]


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
            standard_deviation=0.5, dtype=None, device=None):
    """VTK's and Viskores' wavelet on the grid of point indices ``extent`` (``n``:
    -n to n on every axis, or ``x0, x1, y0, y1, z0, z1``), points at index times
    ``spacing``: ``maximum * exp(-|a|^2 / (2 sd^2))`` plus ``magnitude`` times
    sin, sin and cos of ``frequency * a``, where ``a`` is ``center`` minus the
    point, each axis divided by its extent's length. It is on ``device`` (by
    default the default device)."""
    dtype = tack.get_default_float() if dtype is None else dtype
    lo, hi = _extent(extent)
    s = np.broadcast_to(np.asarray(spacing, float), (3,))
    data = uniform_grid([int(hi[d] - lo[d] + 1) for d in range(3)],
                        [lo[d] * s[d] for d in range(3)], s, dtype=dtype, device=device)
    n = data.num_points
    out = tack.empty((n,), dtype, device=data.device)
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


def tangle(dims=(16, 16, 16), dtype=None, device=None):
    """Viskores' tangle on ``dims`` points per axis over [0, 1]^3: the quartic
    ``(x^4 - 5x^2 + y^4 - 5y^2 + z^4 - 5z^2 + 11.8) / 5 + 1/2`` of the point
    mapped to [-3, 3]^3, the point field ``tangle``. It is on ``device`` (by
    default the default device)."""
    dtype = tack.get_default_float() if dtype is None else dtype
    dims = np.broadcast_to(np.asarray(dims, int), (3,))
    if (dims < 2).any():
        raise ValueError("tangle needs at least two points per axis")
    cells = dims - 1
    data = uniform_grid(dims, (0.0, 0.0, 0.0), 1.0 / cells, dtype=dtype, device=device)
    out = tack.empty((data.num_points,), dtype, device=data.device)
    _tangle(out, int(dims[0]), int(dims[1]), *map(float, cells))
    data.fields["tangle"] = Field(H1(data), out)
    return data


def amr_gaussian_pulse(dimension=3, root_spacing=0.5, refinement_ratio=2,
                       pulse_origin=(0.0, 0.0, 0.0), pulse_width=(0.5, 0.5, 0.5),
                       amplitude=0.0001, dtype=None, device=None):
    """A two-level AMR hierarchy with a Gaussian pulse, as VTK's
    ``vtkAMRGaussianPulseSource`` makes it: a root image of 6 x 6 (x 6) points from
    ``(-2, -2, 0)``, and two boxes refining parts of it by ``refinement_ratio``.

    Each box has VTK's two cell fields: ``Gaussian-Pulse``, ``amplitude *
    exp(-sum(((c - pulse_origin) / pulse_width)^2))`` over the first
    ``dimension`` axes at each cell's centre ``c``, and ``Centroid``, the
    centres. An ``AMRDataSet``, its refined cells and points marked."""
    dtype = tack.get_default_float() if dtype is None else dtype
    from tack.dataset.amr import AMRDataSet
    from tack.dataset.spaces import Constant

    if dimension not in (2, 3):
        raise ValueError(f"dimension is 2 or 3, not {dimension!r}")
    r = int(refinement_ratio)
    h = float(root_spacing)
    origin = np.array([-2.0, -2.0, 0.0])
    # VTK's patches, as point ranges of the root: x, y (and z) inclusive.
    patches = ([(0, 2), (0, 3), (0, 5)], [(3, 5), (2, 5), (0, 5)])

    def box(points, spacing, corner, offset):
        dims = tuple(points)[:dimension]
        grid = uniform_grid(dims, origin=tuple(corner)[:dimension],
                            spacing=(spacing,) * dimension, dtype=dtype, device=device)
        topology = type(grid.topology)(dims, id_dtype=grid.topology.id_dtype,
                                       device=device, offset=tuple(offset)[:dimension])
        centres = _cell_centres(dims, corner[:dimension], spacing)
        d = (centres - np.asarray(pulse_origin, np.float64)[:dimension]) / \
            np.asarray(pulse_width, np.float64)[:dimension]
        pulse = float(amplitude) * np.exp(-(d * d).sum(axis=1))
        full = np.zeros((len(centres), 3))
        full[:, :dimension] = centres
        from tack.dataset.dataset import DataSet

        out = DataSet(topology, grid.geometry.values, dtype=dtype)
        with out.device:
            out.fields["Gaussian-Pulse"] = Field(Constant(out), tack.array(
                pulse.astype(dtype.numpy_dtype)))
            out.fields["Centroid"] = Field(Constant(out), tack.array(
                full.astype(dtype.numpy_dtype), tack.vec3(dtype)))
        return out

    root = box((6, 6, 6), h, origin, (0, 0, 0))
    fine = []
    for patch in patches:
        lo = np.array([a for a, _ in patch])
        points = [r * (b - a) + 1 for a, b in patch]
        fine.append(box(points, h / r, origin + lo * h, lo * r))
    return AMRDataSet([[root], fine], ratios=[r])


def _cell_centres(dims, corner, spacing):
    """The centres of an image's cells, x fastest, as an ``(n, dimension)`` array."""
    axes = [np.asarray(corner[a]) + (np.arange(n - 1) + 0.5) * spacing
            for a, n in enumerate(dims)]
    mesh = np.meshgrid(*axes[::-1], indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh[::-1]], axis=1)
