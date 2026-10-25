"""``sph_interpolate`` and ``sph_density``: values of smoothed-particle hydrodynamics (SPH) particles -- each with a smoothing length, a mass and a density -- at other points, by kernel sums, as VTK's vtkSPHInterpolator and SPLASH; and densities by the SPH sum.

The kernels are VTK's (vtkSPHCubicKernel, vtkSPHQuarticKernel,
vtkSPHQuinticKernel, vtkWendlandQuinticKernel), normalized in one, two or three
dimensions. A point's value is ``sum_j (m_j / rho_j) A_j W(|x - x_j|, h_j)``,
each particle reaching as far as its own kernel does (the scatter form, VTK's).
Each point gathers the particles near it through a ``PointLocator``'s bins, in
bin and then index order: nothing is atomic, and the sums have the same bits
from run to run. On another backend they agree to rounding: a GPU may fuse a
kernel's multiply-adds.
"""

import math

import numpy as np

import tack
from tack import arrays
from tack.dataset import units
from tack.dataset.algorithms.outliers import _locator
from tack.dataset.carry import _point_values
from tack.dataset.dataset import DataSet, Field
from tack.dataset.spaces import H1, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['sph_density', 'sph_interpolate']

#: Each kernel's support, in smoothing lengths, and its normalization in 1, 2
#: and 3 dimensions (VTK's vtkSPH*Kernel; the Wendland kernel has no 1D form).
KERNELS = {
    "cubic": (2.0, (2.0 / 3.0, 10.0 / (7.0 * math.pi), 1.0 / math.pi)),
    "quartic": (2.5, (1.0 / 24.0, 96.0 / (1199.0 * math.pi), 1.0 / (20.0 * math.pi))),
    "quintic": (3.0, (1.0 / 120.0, 7.0 / (478.0 * math.pi), 1.0 / (120.0 * math.pi))),
    "wendland": (2.0, (None, 7.0 / (4.0 * math.pi), 21.0 / (16.0 * math.pi))),
}
_KINDS = {name: k for k, name in enumerate(KERNELS)}


@tack.device_class
class _Kernel:
    """An SPH kernel: ``value(r, h)`` is ``W(r, h)``. Which kernel, and its
    dimension, are compiled in."""

    KIND = 0
    DIMENSION = 3

    def __init__(self, name, dimension):
        support, sigmas = KERNELS[name]
        self.KIND = _KINDS[name]
        self.DIMENSION = int(dimension)
        self.sigma = float(sigmas[dimension - 1])

    @tack.func
    def shape(self, q):
        f = q * 0.0
        if self.KIND == 0:                      # cubic (M4), support 2
            if q < 1.0:
                f = 1.0 - 1.5 * q * q + 0.75 * q * q * q
            elif q < 2.0:
                a = 2.0 - q
                f = 0.25 * a * a * a
        elif self.KIND == 1:                    # quartic (M5), support 2.5
            if q < 2.5:
                a = 2.5 - q
                f = a * a * a * a
                if q < 1.5:
                    b = 1.5 - q
                    f = f - 5.0 * b * b * b * b
                    if q < 0.5:
                        c = 0.5 - q
                        f = f + 10.0 * c * c * c * c
        elif self.KIND == 2:                    # quintic (M6), support 3
            if q < 3.0:
                a = 3.0 - q
                f = a * a * a * a * a
                if q < 2.0:
                    b = 2.0 - q
                    f = f - 6.0 * b * b * b * b * b
                    if q < 1.0:
                        c = 1.0 - q
                        f = f + 15.0 * c * c * c * c * c
        else:                                   # Wendland C2, support 2
            if q < 2.0:
                a = 1.0 - 0.5 * q
                f = a * a * a * a * (2.0 * q + 1.0)
        return f

    @tack.func
    def value(self, r, h):
        scale = h
        if self.DIMENSION == 2:
            scale = h * h
        elif self.DIMENSION == 3:
            scale = h * h * h
        return self.sigma / scale * self.shape(r / h)


@tack.kernel
def _interpolated(grid, kernel, support, reach, queries, smoothing, volumes, values, out,
                  shepard):
    # Each query gathers every particle whose kernel reaches it.
    for q in range(out.shape[0]):
        x = queries[q]
        total = values[0] * 0.0
        weights = smoothing[0] * 0.0
        i0, i1, j0, j1, k0, k1 = grid.span(x, reach)
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    b = grid.bin(i, j, k)
                    for s in range(grid.first(b), grid.end(b)):
                        p = grid.point(s)
                        h = smoothing[p]
                        d = grid.offset(p, x)
                        r = sqrt(d.dot(d))
                        if r < support * h:
                            w = volumes[p] * kernel.value(r, h)
                            total += values[p] * w
                            weights += w
        out[q] = total
        shepard[q] = weights


@tack.kernel
def _normalized(out, shepard):
    for q in range(out.shape[0]):
        if shepard[q] > 0.0:
            out[q] = out[q] / shepard[q]


@tack.kernel
def _volumes(mass, density, volumes):
    for p in range(volumes.shape[0]):
        volumes[p] = mass[p] / density[p]


@tack.kernel
def _densities(grid, kernel, support, smoothing, mass, density):
    # Each particle's density from its own smoothing length (the gather form,
    # SPLASH's), itself included.
    for q in range(density.shape[0]):
        x = grid.positions[q]
        h = smoothing[q]
        total = mass[0] * 0.0
        i0, i1, j0, j1, k0, k1 = grid.span(x, support * h)
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    b = grid.bin(i, j, k)
                    for s in range(grid.first(b), grid.end(b)):
                        p = grid.point(s)
                        d = grid.offset(p, x)
                        r = sqrt(d.dot(d))
                        if r < support * h:
                            total += mass[p] * kernel.value(r, h)
        density[q] = total


def _per_particle(data, value, what):
    """A point field of ``data`` (a name or a ``Field``), or one number for every
    particle, as an array of the geometry's type."""
    n = data.num_points
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not value > 0:
            raise ValueError(f"{what} must be positive, not {value!r}")
        out = tack.empty((n,), data.dtype)
        out.fill(float(value))
        return out
    field = data.fields[value] if isinstance(value, str) else value
    on_points = (isinstance(field.space, H1) and field.space.order == 1) or (
        isinstance(field.space, Values) and field.space.on == "points")
    if not on_points or field.space.topology is not data.topology:
        raise TypeError(f"{what} is a point field of the particles, or a number")
    values = arrays.materialize(_point_values(field.values, n))
    if values.dtype is not data.dtype:
        values = values.astype(data.dtype)
    return values


def _checked(kernel, dimension):
    if kernel not in KERNELS:
        raise ValueError(f"kernel is one of {', '.join(map(repr, KERNELS))}, not {kernel!r}")
    if dimension not in (1, 2, 3):
        raise ValueError(f"dimension is 1, 2 or 3, not {dimension!r}")
    if KERNELS[kernel][1][dimension - 1] is None:
        raise ValueError(f"the {kernel} kernel has no {dimension}D form")
    return _Kernel(kernel, dimension), KERNELS[kernel][0]


@on_inputs_device
def sph_interpolate(particles, target, fields=None, kernel="cubic", smoothing_length="h",
                    mass="mass", density="density", normalize=False, dimension=3,
                    periodic=None, locator=None):
    """``particles``' point fields at ``target``'s points by SPH kernel sums: ``A(x) =
    sum_j (m_j / rho_j) A_j W(|x - x_j|, h_j)``, each particle reaching to its
    kernel's support (2, 2.5 or 3 smoothing lengths), as VTK's
    vtkSPHInterpolator and SPLASH.

    ``target`` is a dataset -- an image to resample onto, a slice, a point cloud
    -- or an ``(n, 3)`` array of world positions, which becomes vertex cells;
    the result is ``target`` with each field (``fields`` chooses; all point
    fields but the smoothing length, mass and density by default) on its points,
    and ``shepard``, the sum of the particles' ``(m / rho) W`` there: where it
    is near 1 the particles cover the point, near 0 they do not. ``normalize``
    divides by it (VTK's ShepardProcessing), which makes a constant exact near
    the particles' edge.

    ``smoothing_length``, ``mass`` and ``density`` are the particles' point
    fields (names or ``Field``s) or numbers for all of them. ``kernel`` is
    ``"cubic"``, ``"quartic"``, ``"quintic"`` or ``"wendland"``, normalized in
    ``dimension`` dimensions. ``periodic`` (a box) reaches across its faces.
    ``locator`` reuses a ``PointLocator`` of the particles. The target's points are moved to the particles' origin by the
    difference of the two, in double precision."""
    from tack.dataset.algorithms._extract import _vertices
    from tack.dataset.dataset import relative_positions
    from tack.dataset.ghosts import keep_marks

    w, support = _checked(kernel, dimension)
    if not isinstance(particles.geometry.space, H1):
        raise TypeError("sph_interpolate needs the particles' positions: an H1 geometry")
    if not isinstance(target, DataSet):
        target = _vertices(relative_positions(target, particles.origin, particles.unit), particles)
    if not isinstance(target.geometry.space, H1):
        raise TypeError("sph_interpolate needs the target's points: an H1 geometry")
    smoothing = _per_particle(particles, smoothing_length, "smoothing_length")
    reserved = {v for v in (smoothing_length, mass, density) if isinstance(v, str)}
    names = [name for name in (fields if fields is not None else particles.fields)
             if name not in reserved or fields is not None]
    n = particles.num_points
    volumes = tack.empty((n,), particles.dtype)
    if n:
        _volumes(_per_particle(particles, mass, "mass"),
                 _per_particle(particles, density, "density"), volumes)
    units.check("sph_interpolate's target and particles' spatial unit", target.unit,
                particles.unit)
    # Each term is f times m/rho, a volume, times W, per volume: m / (rho L^d) is
    # a pure number for consistent units, unknown where the mass or density is a
    # number.
    weight_unit = units.derived(_unit_of(particles, mass), (_unit_of(particles, density), -1),
                                (particles.unit, -dimension))
    locator = _locator(particles, locator, periodic)
    m = target.num_points
    queries = arrays.materialize(_point_values(target.geometry.values, m))
    if target.origin != particles.origin:
        from tack.dataset.algorithms._moving import _shifted_values

        queries = _shifted_values(queries, target.dtype,
                                  np.asarray(target.origin) - np.asarray(particles.origin))
    if queries.dtype != particles.geometry.values.dtype:
        queries = queries.astype(tack.vec(3, particles.dtype))
    reach = support * (float(arrays.to_host(smoothing).max()) if n else 0.0)
    locator.check_radius(reach)
    out = DataSet(target.topology, target.geometry, fields=dict(target.fields),
                  sets=dict(target.sets), dtype=target.dtype, origin=target.origin, unit=target.unit)
    shepard = tack.zeros((m,), particles.dtype)
    grid = locator.grid()
    for name in names:
        field = particles.fields[name]
        on_points = (isinstance(field.space, H1) and field.space.order == 1) or (
            isinstance(field.space, Values) and field.space.on == "points")
        if not on_points:
            continue
        values = arrays.materialize(_point_values(field.values, n))
        scalar = getattr(values.dtype, "scalar_dtype", values.dtype)
        if scalar.numpy_dtype.kind != "f":
            continue                            # an integer field has no kernel average
        result = tack.zeros((m,), values.dtype)
        if n and m:
            _interpolated(grid, w, support, reach, queries, smoothing, volumes, values,
                          result, shepard)
            if normalize:
                _normalized(result, shepard)
        same = normalize or weight_unit == "dimensionless"
        out.fields[name] = Field(H1(out), result, field.unit if same else
                                 units.derived(field.unit, weight_unit))
    if not names and n and m:                      # no fields: the Shepard sum alone
        _interpolated(grid, w, support, reach, queries, smoothing, volumes, volumes,
                      tack.zeros((m,), particles.dtype), shepard)
    out.fields["shepard"] = Field(H1(out), shepard, weight_unit)
    return keep_marks(target, out)


@on_inputs_device
def sph_density(particles, kernel="cubic", smoothing_length="h", mass="mass", dimension=3,
                periodic=None, locator=None):
    """``particles`` with ``density``, each particle's SPH density ``rho_i = sum_j m_j
    W(|x_i - x_j|, h_i)`` -- itself included, from its own smoothing length (the
    gather form, as SPLASH computes it) -- for particles that come without one.
    The arguments are ``sph_interpolate``'s."""
    from tack.dataset.ghosts import keep_marks

    w, support = _checked(kernel, dimension)
    if not isinstance(particles.geometry.space, H1):
        raise TypeError("sph_density needs the particles' positions: an H1 geometry")
    n = particles.num_points
    density = tack.zeros((n,), particles.dtype)
    if n:
        found = _locator(particles, locator, periodic)
        smoothing = _per_particle(particles, smoothing_length, "smoothing_length")
        found.check_radius(support * float(arrays.to_host(smoothing).max()))
        _densities(found.grid(), w, support, smoothing,
                   _per_particle(particles, mass, "mass"), density)
    out = DataSet(particles.topology, particles.geometry, fields=dict(particles.fields),
                  sets=dict(particles.sets), dtype=particles.dtype, origin=particles.origin, unit=particles.unit)
    out.fields["density"] = Field(H1(out), density, units.derived(
        _unit_of(particles, mass), (particles.unit, -dimension)))
    return keep_marks(particles, out)


def _unit_of(particles, value):
    """The unit of a per-particle argument: a field's, by name; a number's is
    unknown."""
    return particles.fields[value].unit if isinstance(value, str) else None

