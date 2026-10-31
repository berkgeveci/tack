"""``halo_catalog``: the halos of dark matter particles -- groups such as ``connected_components(particles, radius=linking_length, min_size=..., periodic=box)`` makes (friends-of-friends) -- as a point cloud of one point per halo, with its mass, centres, most bound particle, velocities and radius. Ported from Viskores' cosmotools (``viskores/worklet/cosmotools``) and improved: per-particle masses, softening, periodic boxes, and the exact pruned search for the most bound particle on every large halo.

The most bound particle (MBP) is the one of least potential ``phi_i = -G sum_j m_j
/ sqrt(r_ij^2 + eps^2)`` over its halo. Halos up to ``_EXACT`` particles sum every
pair. Larger ones use Viskores' MxN estimate (``CosmoToolsCenterFinder``),
tightened: bins over the halo, each particle's potential from its 27 neighbouring
bins summed exactly, then bounded from below and above by every other bin's mass
at the nearest and farthest point of that bin from the particle (Viskores bounds
bin against bin), and only the particles whose lower bound beats the least upper
bound summed exactly. The answer is the exact MBP.
"""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.sort import _run_offsets, argsort, gather
from tack.dataset import units
from tack.dataset.algorithms._common import _field, _length
from tack.dataset.algorithms._extract import _vertices
from tack.dataset.algorithms.sph import _per_particle
from tack.dataset.buckets import bucket_order
from tack.dataset.carry import _point_values
from tack.dataset.dataset import Field, is_field_key
from tack.dataset.ghosts import Ghosts
from tack.dataset.partitioned import PartitionedDataSet, common_origin
from tack.dataset.point_locator import periodic_box
from tack.dataset.spaces import H1, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['halo_catalog']

#: Halos of at most this many particles sum every pair for their MBP.
_EXACT = 2048
#: Particles per bin of the pruned search.
_PER_BIN = 32
_CENTER = ("mbp", "com")


@tack.kernel
def _take(values, order, out):
    for s in range(out.shape[0]):
        out[s] = values[order[s]]


@tack.kernel
def _unwrapped(positions, order, offsets, start, periodic, lx, ly, lz, out, run_of):
    # Each halo's positions in sorted order, each to its nearest image of the
    # halo's first particle, so a halo across a periodic face is in one piece.
    for r in range(start, offsets.shape[0] - 1):
        first = offsets[r]
        x0 = positions[order[first]]
        for s in range(first, offsets[r + 1]):
            d = positions[order[s]] - x0
            if periodic == 1:
                d = [d[0] - lx * floor(d[0] / lx + 0.5), d[1] - ly * floor(d[1] / ly + 0.5),
                     d[2] - lz * floor(d[2] / lz + 0.5)]
            out[s] = x0 + d
            run_of[s] = r


@tack.kernel
def _exact_potentials(u, m, offsets, run_of, exact, g, eps2, phi):
    for s in range(phi.shape[0]):
        r = run_of[s]
        if r >= 0:
            if exact[r] == 1:
                x = u[s]
                total = m[s] * 0.0
                for t in range(offsets[r], offsets[r + 1]):
                    d = u[t] - x
                    r2 = d.dot(d) + eps2
                    if t != s and r2 > 0.0:
                        total -= m[t] / sqrt(r2)
                phi[s] = g * total


@tack.kernel
def _masses(u, m, offsets, start, count, mass, com):
    for r in range(start, offsets.shape[0] - 1):
        h = r - start
        first = offsets[r]
        total = m[first]
        moment = u[first] * m[first]
        for s in range(first + 1, offsets[r + 1]):
            total += m[s]
            moment += u[s] * m[s]
        count[h] = offsets[r + 1] - first
        mass[h] = total
        com[h] = moment / total


@tack.kernel
def _velocities(v, m, offsets, start, mass, mean, dispersion):
    # Mass-weighted mean velocity; the one-dimensional dispersion,
    # sqrt(sum m |v - mean|^2 / (3 M)).
    for r in range(start, offsets.shape[0] - 1):
        h = r - start
        first = offsets[r]
        moment = v[first] * m[first]
        for s in range(first + 1, offsets[r + 1]):
            moment += v[s] * m[s]
        vbar = moment / mass[h]
        spread = mass[h] * 0.0
        for s in range(first, offsets[r + 1]):
            d = v[s] - vbar
            spread += m[s] * d.dot(d)
        mean[h] = vbar
        dispersion[h] = sqrt(spread / (3.0 * mass[h]))


@tack.kernel
def _least(phi, order, offsets, start, entry, mbp, potential):
    # Each halo's least potential, the lowest particle id on a tie.
    for r in range(start, offsets.shape[0] - 1):
        h = r - start
        best = offsets[r]
        for s in range(best + 1, offsets[r + 1]):
            if phi[s] < phi[best] or (phi[s] == phi[best] and order[s] < order[best]):
                best = s
        entry[h] = best
        mbp[h] = order[best]
        potential[h] = phi[best]


@tack.kernel
def _centers(u, com, entry, use_mbp, offsets, start, center, radius):
    for r in range(start, offsets.shape[0] - 1):
        h = r - start
        c = com[h]
        if use_mbp == 1:
            c = u[entry[h]]
        far = c[0] * 0.0
        for s in range(offsets[r], offsets[r + 1]):
            d = u[s] - c
            far = max(far, d.dot(d))
        center[h] = c
        radius[h] = sqrt(far)


@tack.kernel
def _wrapped(points, periodic, ox, oy, oz, lx, ly, lz):
    # Back into the periodic box (local corner o, sides l).
    for h in range(points.shape[0]):
        x = points[h]
        if periodic == 1:
            points[h] = [ox + (x[0] - ox) - lx * floor((x[0] - ox) / lx),
                         oy + (x[1] - oy) - ly * floor((x[1] - oy) / ly),
                         oz + (x[2] - oz) - lz * floor((x[2] - oz) / lz)]


# ── The pruned search of one large halo (Viskores' MxN) ─────────────

@tack.kernel
def _halo_bins(u, first, lx, ly, lz, inverse, nx, ny, nz, keys):
    for i in range(keys.shape[0]):
        x = u[first + i]
        bx = min(max(tack.i64(floor((x[0] - lx) * inverse)), 0), nx - 1)
        by = min(max(tack.i64(floor((x[1] - ly) * inverse)), 0), ny - 1)
        bz = min(max(tack.i64(floor((x[2] - lz) * inverse)), 0), nz - 1)
        keys[i] = bx + nx * (by + ny * bz)


@tack.kernel
def _bin_masses(m, first, order, starts, mass):
    for b in range(mass.shape[0]):
        total = mass[b] * 0.0
        for s in range(starts[b], starts[b + 1]):
            total += m[first + order[s]]
        mass[b] = total


@tack.kernel
def _near(u, m, first, keys, order, starts, nx, ny, nz, g, eps2, near):
    # Each particle's potential from the particles of its 27 neighbouring bins.
    for i in range(near.shape[0]):
        x = u[first + i]
        b = keys[i]
        bx = b % nx
        by = (b // nx) % ny
        bz = b // (nx * ny)
        total = m[first] * 0.0
        for kz in range(max(bz - 1, 0), min(bz + 2, nz)):
            for ky in range(max(by - 1, 0), min(by + 2, ny)):
                for kx in range(max(bx - 1, 0), min(bx + 2, nx)):
                    c = kx + nx * (ky + ny * kz)
                    for s in range(starts[c], starts[c + 1]):
                        t = order[s]
                        d = u[first + t] - x
                        r2 = d.dot(d) + eps2
                        if t != i and r2 > 0.0:
                            total -= m[first + t] / sqrt(r2)
        near[i] = g * total


@tack.kernel
def _bounds(u, first, keys, mass, near, lx, ly, lz, side, nx, ny, nz, g, eps2, lower,
            upper):
    # Each particle's potential bounded: its near sum, plus every bin beyond its
    # neighbours with that bin's mass at the nearest and the farthest point of
    # the bin's box from the particle. (Viskores bounds a bin against a bin,
    # box to box; from the particle the bounds are much tighter.)
    for i in range(lower.shape[0]):
        x = u[first + i]
        b = keys[i]
        bx = b % nx
        by = (b // nx) % ny
        bz = b // (nx * ny)
        lo = near[i]
        hi = lo
        for c in range(mass.shape[0]):
            cx = c % nx
            cy = (c // nx) % ny
            cz = c // (nx * ny)
            if mass[c] > 0.0 and max(abs(cx - bx), max(abs(cy - by), abs(cz - bz))) > 1:
                ax = lx + cx * side
                ay = ly + cy * side
                az = lz + cz * side
                nxd = max(max(ax - x[0], x[0] - ax - side), 0.0)
                nyd = max(max(ay - x[1], x[1] - ay - side), 0.0)
                nzd = max(max(az - x[2], x[2] - az - side), 0.0)
                fxd = max(abs(x[0] - ax), abs(x[0] - ax - side))
                fyd = max(abs(x[1] - ay), abs(x[1] - ay - side))
                fzd = max(abs(x[2] - az), abs(x[2] - az - side))
                lo -= g * mass[c] / sqrt(nxd * nxd + nyd * nyd + nzd * nzd + eps2)
                hi -= g * mass[c] / sqrt(fxd * fxd + fyd * fyd + fzd * fzd + eps2)
        lower[i] = lo
        upper[i] = hi


@tack.kernel
def _candidates(u, m, first, n, lower, cutoff, g, eps2, phi):
    # The particles that may be the MBP, summed exactly; the others cannot be
    # least, and take no potential (+infinity is never least).
    for i in range(n):
        if lower[i] <= cutoff:
            x = u[first + i]
            total = m[first] * 0.0
            for t in range(n):
                d = u[first + t] - x
                r2 = d.dot(d) + eps2
                if t != i and r2 > 0.0:
                    total -= m[first + t] / sqrt(r2)
            phi[first + i] = g * total
        else:
            phi[first + i] = 3.0e38


def _pruned(u, m, first, n, g, eps2, phi, dtype):
    """The MBP search of the halo at ``[first, first + n)`` of the sorted arrays."""
    x = arrays.to_host(u)[first:first + n].astype(np.float64)
    low, high = x.min(axis=0), x.max(axis=0)
    per_axis = max(1, int(np.ceil((n / _PER_BIN) ** (1.0 / 3.0))))
    side = max(float((high - low).max()) / per_axis, 1e-30)
    dims = np.maximum(np.ceil((high - low) / side).astype(np.int64), 1)
    nbins = int(np.prod(dims))
    keys = tack.empty((n,), tack.i64)
    _halo_bins(u, first, *map(float, low), 1.0 / side, *map(int, dims), keys)
    order, starts = bucket_order(keys, nbins)
    mass = tack.zeros((nbins,), dtype)
    _bin_masses(m, first, order, starts, mass)
    near = tack.empty((n,), dtype)
    _near(u, m, first, keys, order, starts, *map(int, dims), g, eps2, near)
    lower = tack.empty((n,), dtype)
    upper = tack.empty((n,), dtype)
    _bounds(u, first, keys, mass, near, *map(float, low), side, *map(int, dims), g, eps2,
            lower, upper)
    cutoff = float(arrays.to_host(upper).min())
    _candidates(u, m, first, n, lower, cutoff, g, eps2, phi)


@on_inputs_device
def halo_catalog(particles, region="point_region", mass=1.0, velocity=None, center="mbp",
                 softening=0.0, G=1.0, periodic=None):
    """The halos of ``particles`` -- the groups of ``region``, a point field of
    halo labels (``connected_components``' ``point_region``: friends-of-
    friends with ``radius`` the linking length; -1 for no halo) -- as a point
    cloud of one point per halo, ported from Viskores' cosmotools and ParaView's
    LANL halo finder. Each halo's fields:

    - ``halo``, its label; ``count``; ``mass``, the sum of its particles'
      (``mass`` a point field or one number for all, as Viskores takes it);
    - ``com``, its centre of mass;
    - ``mbp``, its most bound particle (the index of the particle of least
      potential ``-G sum m_j / sqrt(r^2 + softening^2)`` over the halo), and
      ``potential``, that least potential;
    - with ``velocity`` (a 3-vector point field): ``velocity``, the mass-
      weighted mean, and ``dispersion``, the one-dimensional velocity
      dispersion ``sqrt(sum m |v - mean|^2 / 3M)``;
    - ``radius``, the farthest particle from the halo's point.

    The point is the MBP's position (``center="mbp"``) or the centre of mass
    (``"com"``). In a ``periodic`` box (a side, three, or corners, as
    ``PointLocator`` takes it) each halo is unwrapped about its first particle
    before anything is summed, so a halo across a face is whole, and its points
    are put back in the box. Positions are relative to the particles' origin,
    which the catalogue shares. Every sum runs in particle order, so the
    result has the same bits from run to run; counts and masses are the same
    on every backend, the other sums agree to rounding (a GPU may fuse a
    multiply-add), and so the most bound particle can differ between
    backends only where two potentials tie to rounding.

    Over a ``PartitionedDataSet`` (labels numbered across the pieces, as
    ``connected_components`` numbers them), each halo's owned particles are
    brought to one block by a sort over the blocks, in order of piece and then
    index, and catalogued there: a ``PartitionedDataSet`` of catalogues, the
    halos in order of label across them, each ``mbp`` an index in the piece
    ``mbp_piece`` (a global piece id)."""
    softening = _length(particles, softening, "the softening")
    periodic = _length(particles, periodic, "the periodic box")
    # One mass for every particle may bring its unit ("1e10 Msun", (1e10, "Msun")).
    given_mass_unit = None
    if units.quantity(mass) is not None:
        mass, given_mass_unit = units.number_and_unit(mass)
    if center not in _CENTER:
        raise ValueError(f"center is one of {', '.join(map(repr, _CENTER))}, not {center!r}")
    if isinstance(particles, PartitionedDataSet):
        return _catalog_pieces(particles, region, mass, velocity, center, softening, G,
                               periodic, given_mass_unit)
        raise ValueError(f"center is one of {', '.join(map(repr, _CENTER))}, not {center!r}")
    if not isinstance(particles.geometry.space, H1):
        raise TypeError("halo_catalog needs the particles' positions: an H1 geometry")
    n = particles.num_points
    labels_field = particles.fields[region] if is_field_key(region) else region
    labels = arrays.materialize(_point_values(labels_field.values, n))
    dtype = particles.dtype
    positions = arrays.materialize(_point_values(particles.geometry.values, n))
    box = periodic_box(periodic)
    if box is not None:
        lower = np.asarray(box[0]) - np.asarray(particles.origin)
        sides = np.asarray(box[1]) - np.asarray(box[0])
    else:
        lower, sides = np.zeros(3), np.zeros(3)
    order = argsort(labels)
    sorted_labels = gather(labels, order)
    offsets, nruns = _run_offsets(sorted_labels, n)
    start = 1 if nruns and int(sorted_labels[0]) < 0 else 0
    halos = max(nruns - start, 0)
    masses = _taken(_per_particle(particles, mass, "mass"), order, n)
    u = tack.empty((n,), tack.vec(3, dtype))
    run_of = tack.full((n,), -1, offsets.dtype)
    if halos:
        _unwrapped(positions, order, offsets, start, int(box is not None), *map(float, sides),
                   u, run_of)
    counts = tack.empty((halos,), particles.id_dtype)
    total = tack.empty((halos,), dtype)
    com = tack.empty((halos,), tack.vec(3, dtype))
    if halos:
        _masses(u, masses, offsets, start, counts, total, com)
    # Potentials: every pair on small halos, the pruned search on large ones.
    sizes = np.diff(arrays.to_host(offsets).astype(np.int64)) if nruns else np.zeros(0)
    large = [r for r in range(start, nruns) if sizes[r] > _EXACT]
    exact = tack.array((sizes <= _EXACT).astype(np.int32)) if nruns else tack.zeros((1,),
                                                                                    tack.i32)
    phi = tack.zeros((n,), dtype)
    g, eps2 = float(G), float(softening) ** 2
    if halos:
        _exact_potentials(u, masses, offsets, run_of, exact, g, eps2, phi)
        bounds = arrays.to_host(offsets).astype(np.int64)
        for r in large:
            _pruned(u, masses, int(bounds[r]), int(sizes[r]), g, eps2, phi, dtype)
    entry = tack.empty((halos,), offsets.dtype)
    mbp = tack.empty((halos,), tack.i64)
    potential = tack.empty((halos,), dtype)
    centers = tack.empty((halos,), tack.vec(3, dtype))
    radius = tack.empty((halos,), dtype)
    if halos:
        _least(phi, order, offsets, start, entry, mbp, potential)
        _centers(u, com, entry, int(center == "mbp"), offsets, start, centers, radius)
        _wrapped(centers, int(box is not None), *map(float, lower), *map(float, sides))
        _wrapped(com, int(box is not None), *map(float, lower), *map(float, sides))
    out = _vertices(centers, particles)
    points = Values(out, "points")
    halo_labels = tack.empty((halos,), labels.dtype)
    if halos:
        _halo_labels(sorted_labels, offsets, start, halo_labels)
    # The mass is the particles' mass field's; the potential's G is a number, so
    # its unit is unknown; labels, counts and indices are not quantities.
    mass_unit = _field(particles, mass).unit if isinstance(mass, (str, Field)) else \
        given_mass_unit
    for name, values, unit in (("halo", halo_labels, None), ("count", counts, None),
                               ("mass", total, mass_unit), ("mbp", mbp, None),
                               ("potential", potential, None),
                               ("radius", radius, particles.unit)):
        out.fields[name] = Field(points, values, unit)
    out.fields["com"] = Field(H1(out), com, particles.unit)
    if velocity is not None:
        field = particles.fields[velocity] if is_field_key(velocity) else velocity
        v = _taken(arrays.materialize(_point_values(field.values, n)), order, n)
        mean = tack.empty((halos,), v.dtype)
        dispersion = tack.empty((halos,), dtype)
        if halos:
            _velocities(v, masses, offsets, start, total, mean, dispersion)
        out.fields["velocity"] = Field(H1(out), mean, field.unit)
        out.fields["dispersion"] = Field(points, dispersion, field.unit)
    return out


@tack.kernel
def _halo_labels(sorted_labels, offsets, start, out):
    for r in range(start, offsets.shape[0] - 1):
        out[r - start] = sorted_labels[offsets[r]]


def _taken(values, order, n):
    out = tack.empty((n,), values.dtype)
    if n:
        _take(values, order, out)
    return out


def _catalog_pieces(data, region, mass, velocity, center, softening, G, periodic,
                    mass_unit=None):
    """``halo_catalog`` over pieces: each halo's owned particles sorted to one block
    (``tack.communication.sort_by_key`` by label), catalogued there."""
    from tack import communication as tc
    from tack.dataset.carry import _take

    for what, value in (("region", region), ("velocity", velocity)):
        if value is not None and not is_field_key(value):
            raise TypeError(f"halo_catalog of a PartitionedDataSet takes {what} by name: a "
                            "Field belongs to one partition's topology")
    by_name = is_field_key(mass)
    if not by_name and not isinstance(mass, (int, float, np.integer, np.floating)):
        raise TypeError("halo_catalog of a PartitionedDataSet takes mass by name or as a "
                        "number")
    common_origin(data, "halo_catalog of a PartitionedDataSet")
    pieces = list(data)
    blocks = tc.Blocks([piece.device for piece in pieces], data.world)
    keys, values = {}, {}
    for g, piece in zip(blocks.local, pieces):
        if not isinstance(piece.geometry.space, H1):
            raise TypeError("halo_catalog needs the particles' positions: an H1 geometry")
        n = piece.num_points
        labels = np.asarray(arrays.to_host(_point_values(piece.fields[region].values, n)),
                            np.int64).reshape(-1)
        owned = labels >= 0
        if piece.ghosts is not None and piece.ghosts.points is not None:
            owned &= (piece.ghosts.points.numpy() & Ghosts.DUPLICATEPOINT) == 0
        chosen = np.flatnonzero(owned).astype(np.int64)
        device = blocks.device(g)
        with device:
            at = tack.array(chosen, tack.i64)
            keys[g] = tack.array(labels[chosen], tack.i64)
            values[g] = {
                "x": _take(arrays.materialize(_point_values(piece.geometry.values, n)), at),
                "piece": tack.full((len(chosen),), g, tack.i64), "index": at}
            if by_name:
                values[g]["m"] = _take(arrays.materialize(
                    _point_values(piece.fields[mass].values, n)), at).astype(piece.dtype)
            if velocity is not None:
                values[g]["v"] = _take(arrays.materialize(
                    _point_values(piece.fields[velocity].values, n)), at)
    sorted_keys, sorted_values = tc.sort_by_key(blocks, keys, values)
    out = []
    for g, piece in zip(blocks.local, pieces):
        got = sorted_values[g]
        with blocks.device(g):
            cloud = _vertices(got["x"], piece)
            cloud.fields["region"] = Field(H1(cloud), sorted_keys[g])
            masses = Field(H1(cloud), got["m"], piece.fields[mass].unit) if by_name else \
                (mass if mass_unit is None else (mass, mass_unit))
            speeds = Field(H1(cloud), got["v"], piece.fields[velocity].unit) \
                if velocity is not None else None
            catalog = halo_catalog(cloud, "region", mass=masses, velocity=speeds,
                                   center=center, softening=softening, G=G, periodic=periodic)
            best = catalog.fields["mbp"].values
            points = Values(catalog, "points")
            catalog.fields["mbp"] = Field(points, _take(got["index"], best))
            catalog.fields["mbp_piece"] = Field(points, _take(got["piece"], best))
        out.append(catalog)
    return PartitionedDataSet(out, world=data.world)
