"""Finding points near points: uniform bins over a dataset's points.

``PointLocator`` puts each point in a bin of a uniform grid over the points'
box and sorts the points by bin (``buckets.bucket_order``: a count, a scan, a
scatter, each bin in index order), as VTK's ``vtkStaticPointLocator`` does.
Nothing is atomic, so every query visits the same points in the same order on
every backend and its answer has the same bits.

A kernel searches through ``locator.grid()``, a ``@tack.device_class`` holding
the sorted points and the grid: ``span(x, r)`` gives the bins a ball touches,
``first(b)``/``end(b)`` the slots of bin ``b`` and ``point(s)`` the point in
slot ``s``. The host queries -- ``neighbors`` (every point within a radius,
CSR) and ``nearest`` (the k nearest) -- are such kernels.

Positions are the geometry's values, relative to the dataset's origin; query
points given as a host array are world positions, taken relative to it in
double precision, as ``probe`` takes them.

A periodic box (``periodic=``, a cosmology simulation's) makes the bins tile the
box exactly and wrap: distances are to the nearest image of each point, and a
search reaches across the faces. A search radius must stay under half the box.
"""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset.buckets import bucket_order
from tack.runtime.dispatch import on_inputs_device

__all__ = ["PointLocator"]

#: The most neighbours ``nearest`` keeps per point: its private arrays' size.
MAX_K = 32
#: Bins along one axis at most, and bins in all at most per point.
_AXIS_BINS = 1 << 20
_BINS_PER_POINT = 4


@tack.device_class
class _PointGrid:
    """The sorted points and their grid, for kernels: a bin is
    ``i + nx * (j + ny * k)``, its points ``point(s)`` for ``s`` in
    ``range(first(b), end(b))``, in index order; ``offset(p, x)`` is point
    ``p`` less ``x``. In a periodic box (``PERIODIC``, compiled in) bin indices
    wrap and ``offset`` is to the nearest image."""

    PERIODIC = 0

    def __init__(self, positions, order, starts, origin, inverse, dims, side, box=None):
        self.positions = positions
        self.order = order
        self.starts = starts
        self.ox, self.oy, self.oz = (float(v) for v in origin)
        self.ix, self.iy, self.iz = (float(v) for v in inverse)
        self.nx, self.ny, self.nz = (int(v) for v in dims)
        #: The smallest bin side along an axis the points span: a ball of this
        #: radius times L about a point lies within L rings of its bin.
        self.side = float(side)
        self.rings = int(max(dims))
        self.lx = self.ly = self.lz = 0.0
        if box is not None:
            self.PERIODIC = 1
            self.lx, self.ly, self.lz = (float(v) for v in box)
            # Each wrapped bin once: offsets -(n - 1) // 2 to n // 2.
            self.rings = int(max((int(n) + 1) // 2 for n in dims))

    @tack.func
    def _axis(self, x, o, inverse, n):
        i = tack.i64(floor((x - o) * inverse))
        if self.PERIODIC == 0:
            i = min(max(i, 0), n - 1)
        return i

    @tack.func
    def _wrap(self, i, n):
        w = i
        if self.PERIODIC == 1:
            w = i % n
        return w

    @tack.func
    def cell(self, x):
        """The bin's indices along each axis for position ``x``: clamped to the grid,
        or wrapped into the box."""
        return (self._wrap(self._axis(x[0], self.ox, self.ix, self.nx), self.nx),
                self._wrap(self._axis(x[1], self.oy, self.iy, self.ny), self.ny),
                self._wrap(self._axis(x[2], self.oz, self.iz, self.nz), self.nz))

    @tack.func
    def _range(self, x, r, o, inverse, n):
        lo = self._axis(x - r, o, inverse, n)
        hi = self._axis(x + r, o, inverse, n)
        if self.PERIODIC == 1:
            hi = min(hi, lo + n - 1)          # one box width: no bin twice
        return lo, hi

    @tack.func
    def span(self, x, r):
        """The bins a ball of radius ``r`` about ``x`` touches: ``(i0, i1, j0, j1,
        k0, k1)``, inclusive; clamped to the grid, or in a periodic box unwrapped
        (``bin`` wraps them) and at most one box wide."""
        i0, i1 = self._range(x[0], r, self.ox, self.ix, self.nx)
        j0, j1 = self._range(x[1], r, self.oy, self.iy, self.ny)
        k0, k1 = self._range(x[2], r, self.oz, self.iz, self.nz)
        return i0, i1, j0, j1, k0, k1

    @tack.func
    def bin(self, i, j, k):
        return (self._wrap(i, self.nx)
                + self.nx * (self._wrap(j, self.ny) + self.ny * self._wrap(k, self.nz)))

    @tack.func
    def _image(self, d, length):
        e = d
        if self.PERIODIC == 1:
            e = d - length * floor(d / length + 0.5)
        return e

    @tack.func
    def offset(self, p, x):
        """Point ``p`` less ``x``: to its nearest image in a periodic box."""
        d = self.positions[p] - x
        if self.PERIODIC == 1:
            d = [self._image(d[0], self.lx), self._image(d[1], self.ly),
                 self._image(d[2], self.lz)]
        return d

    @tack.func
    def ring_bin(self, i, c, n):
        """Whether bin ``i`` along an axis of ``n`` bins is one to search about bin
        ``c``: within the grid, or in a periodic box one of the ``n`` wrapped
        offsets from ``c``."""
        ok = i >= 0 and i < n
        if self.PERIODIC == 1:
            ok = i - c >= -((n - 1) // 2) and i - c <= n // 2
        return ok

    @tack.func
    def first(self, b):
        return self.starts[b]

    @tack.func
    def end(self, b):
        return self.starts[b + 1]

    @tack.func
    def point(self, s):
        return self.order[s]


@tack.kernel
def _bins(grid, keys):
    for p in range(keys.shape[0]):
        i, j, k = grid.cell(grid.positions[p])
        keys[p] = grid.bin(i, j, k)


@tack.kernel
def _count_within(grid, queries, r, own, counts):
    for q in range(counts.shape[0]):
        x = queries[q]
        r2 = r * r
        i0, i1, j0, j1, k0, k1 = grid.span(x, r)
        c = 0
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    b = grid.bin(i, j, k)
                    for s in range(grid.first(b), grid.end(b)):
                        p = grid.point(s)
                        d = grid.offset(p, x)
                        if d.dot(d) <= r2 and (own == 0 or p != q):
                            c += 1
        counts[q] = c


@tack.kernel
def _fill_within(grid, queries, r, own, offsets, ids):
    for q in range(queries.shape[0]):
        x = queries[q]
        r2 = r * r
        i0, i1, j0, j1, k0, k1 = grid.span(x, r)
        at = offsets[q]
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    b = grid.bin(i, j, k)
                    for s in range(grid.first(b), grid.end(b)):
                        p = grid.point(s)
                        d = grid.offset(p, x)
                        if d.dot(d) <= r2 and (own == 0 or p != q):
                            ids[at] = p
                            at += 1


@tack.kernel
def _nearest(grid, queries, k, own, ids, d2s):
    # Rings of bins about the query's bin, outwards, until the k-th distance is
    # within the rings searched; the k best kept sorted by (distance, id).
    for q in range(queries.shape[0]):
        x = queries[q]
        best_d = tack.local_array_like(d2s, 32)
        best_i = tack.local_array_like(ids, 32)
        found = 0
        ci, cj, ck = grid.cell(x)
        ring = 0
        searching = 1
        while searching == 1:
            for kk in range(ck - ring, ck + ring + 1):
                for jj in range(cj - ring, cj + ring + 1):
                    for ii in range(ci - ring, ci + ring + 1):
                        inside = (grid.ring_bin(ii, ci, grid.nx) and grid.ring_bin(jj, cj, grid.ny)
                                  and grid.ring_bin(kk, ck, grid.nz))
                        far = max(abs(ii - ci), max(abs(jj - cj), abs(kk - ck)))
                        if inside and far == ring:
                            b = grid.bin(ii, jj, kk)
                            for s in range(grid.first(b), grid.end(b)):
                                p = grid.point(s)
                                if own == 0 or p != q:
                                    d = grid.offset(p, x)
                                    d2 = d.dot(d)
                                    take = 0
                                    if found < k:
                                        take = 1
                                    elif d2 < best_d[k - 1] or (d2 == best_d[k - 1]
                                                                and p < best_i[k - 1]):
                                        take = 1
                                    if take == 1:
                                        at = min(found, k - 1)
                                        moving = 1
                                        while moving == 1 and at > 0:
                                            if best_d[at - 1] > d2 or (best_d[at - 1] == d2
                                                                       and best_i[at - 1] > p):
                                                best_d[at] = best_d[at - 1]
                                                best_i[at] = best_i[at - 1]
                                                at -= 1
                                            else:
                                                moving = 0
                                        best_d[at] = d2
                                        best_i[at] = p
                                        found = min(found + 1, k)
            reach = ring * grid.side
            if found == k and best_d[k - 1] <= reach * reach:
                searching = 0
            if ring >= grid.rings:
                searching = 0
            ring += 1
        for t in range(k):
            if t < found:
                ids[q * k + t] = best_i[t]
                d2s[q * k + t] = best_d[t]
            else:
                ids[q * k + t] = -1
                d2s[q * k + t] = -1.0


class PointLocator:
    """Uniform bins over ``data``'s points, for finding points near points; build
    once, query many times. ``bin_size`` is a bin's side (in the geometry's
    units); by default bins hold about ``points_per_bin`` points each, cubes
    along the axes the points span.

    ``periodic`` is a periodic box, in world coordinates: its side (a number,
    the box from zero), its three sides, or ``(lower, upper)`` corners. The bins
    then tile the box exactly, points outside it are wrapped in, and every
    distance is to a point's nearest image.

    It records the geometry it was made from and that geometry's state
    (``serves``), as ``CellLocator`` does: filters take it for any dataset of
    the same topology and geometry, and refuse it once Tack has written the
    positions since."""

    @on_inputs_device
    def __init__(self, data, bin_size=None, points_per_bin=4, periodic=None):
        from tack.dataset.carry import _point_values
        from tack.dataset.dataset import DataSet
        from tack.dataset.spaces import H1

        if not isinstance(data.geometry.space, H1):
            raise TypeError("a PointLocator needs an H1 geometry: one position per point")
        if bin_size is not None and not bin_size > 0:
            raise ValueError(f"bin_size must be positive, not {bin_size!r}")
        #: The points, without the fields.
        self.data = DataSet(data.topology, data.geometry, origin=data.origin)
        self._geometry = data.geometry.values
        self._revision = arrays._revision(self._geometry)
        n = data.num_points
        self.positions = arrays.materialize(_point_values(data.geometry.values, n))
        #: The periodic box, ``(lower, upper)`` in world coordinates, or None.
        self.periodic = periodic_box(periodic)
        if self.periodic is not None:
            low = np.asarray(self.periodic[0]) - np.asarray(data.origin)
            high = np.asarray(self.periodic[1]) - np.asarray(data.origin)
        elif n:
            x = arrays.to_host(self.positions).reshape(-1, 3).astype(np.float64)
            low, high = x.min(axis=0), x.max(axis=0)
        else:
            low = high = np.zeros(3)
        extent = high - low
        size = max(float(extent.max()), 1e-30)
        live = extent > 1e-9 * size
        if bin_size is None:
            volume = float(np.prod(extent[live])) if live.any() else 0.0
            side = (volume * points_per_bin / max(n, 1)) ** (1.0 / max(int(live.sum()), 1))
        else:
            side = float(bin_size)
        side = max(side, 1e-30)
        # Bins that tile a periodic box exactly (at most the side asked), or
        # cover the points.
        rounding = np.floor if self.periodic is not None else np.ceil
        dims = np.where(live, np.clip(rounding(extent / side), 1, _AXIS_BINS), 1)
        while live.any() and np.prod(dims) > max(_BINS_PER_POINT * n, 1):
            side *= 2.0                  # too fine for the points: coarser bins
            dims = np.where(live, np.clip(rounding(extent / side), 1, _AXIS_BINS), 1)
        self.dims = dims.astype(np.int64)
        self.origin = low
        self.inverse = np.where(live, self.dims / np.where(live, extent, 1.0), 0.0)
        sides = np.where(live, extent / self.dims, np.inf)
        self.side = float(sides.min()) if live.any() else 1e30
        self._box = None if self.periodic is None else extent
        nbins = int(np.prod(self.dims))
        keys = tack.empty((n,), tack.i64)
        self.order = self.starts = tack.empty((1,), tack.i64)    # for the binning grid
        if n:
            _bins(self.grid(), keys)
        self.order, self.starts = bucket_order(keys, nbins)

    def serves(self, data, periodic=None):
        """Whether it finds ``data``'s points as they are -- the topology and the
        geometry's values it was made from, with nothing written to them by Tack
        since -- in the periodic box given (or none)."""
        return (data.topology is self.data.topology
                and data.geometry.values is self._geometry
                and arrays._revision(self._geometry) == self._revision
                and self.periodic == periodic_box(periodic))

    @property
    def _device(self):
        data = self.__dict__.get("data")
        return data._device if data is not None else None

    def grid(self):
        """The sorted points and the grid, as a template for kernels (``_PointGrid``)."""
        return _PointGrid(self.positions, self.order, self.starts, self.origin,
                          self.inverse, self.dims, self.side, self._box)

    def check_radius(self, radius):
        """Refuse a search radius of half a periodic box or more: a point's nearest
        image is then not the only one within it."""
        if self._box is not None and radius >= 0.5 * float(np.min(self._box)):
            raise ValueError(f"a radius of {radius} reaches half the periodic box "
                             f"({float(np.min(self._box))} across): search less far")

    def _queries(self, points):
        """``(queries, own)``: the locator's own points, or ``points`` -- a host array
        of world positions, or a Tack array of 3-vectors relative to the origin."""
        if points is None:
            return self.positions, 1
        if isinstance(points, tack.Array):
            return points, 0
        from tack.dataset.dataset import relative_positions

        local = relative_positions(points, self.data.origin)
        dtype = self.data.dtype
        queries = tack.empty((len(local),), tack.vec(3, dtype))
        if len(local):
            tack.copyto(queries, local.astype(dtype.numpy_dtype))
        return queries, 0

    @on_inputs_device
    def neighbors(self, radius, points=None):
        """Every point within ``radius`` (inclusive) of each query point, CSR:
        ``(offsets, ids)``, the neighbours of query ``q`` at ``ids[offsets[q]:
        offsets[q + 1]]``, by bin and then by index. The queries are the points
        themselves (each leaving itself out) unless ``points`` are given."""
        if not radius >= 0:
            raise ValueError(f"radius must be at least zero, not {radius!r}")
        self.check_radius(radius)
        queries, own = self._queries(points)
        m = int(queries.shape[0])
        id_dtype = self.data.id_dtype
        counts = tack.empty((m,), id_dtype)
        offsets = tack.empty((m + 1,), id_dtype)
        if m == 0:
            offsets.fill(0)
            return offsets, tack.empty((0,), id_dtype)
        grid = self.grid()
        _count_within(grid, queries, float(radius), own, counts)
        total = exclusive_scan(counts, offsets, m)
        from tack.dataset.algorithms._common import _close

        _close(offsets, m, total)
        ids = tack.empty((int(total),), id_dtype)
        if total:
            _fill_within(grid, queries, float(radius), own, offsets, ids)
        return offsets, ids

    @on_inputs_device
    def nearest(self, k, points=None):
        """The ``k`` nearest points (``1 <= k <= MAX_K``) to each query point:
        ``(ids, squared_distances)``, each ``(m, k)``, nearest first, ties by
        index; -1 (and a squared distance of -1) where there are fewer points
        than ``k``. The queries are the points themselves (each leaving itself
        out) unless ``points`` are given."""
        if not (isinstance(k, (int, np.integer)) and 1 <= k <= MAX_K):
            raise ValueError(f"k is a whole number from 1 to {MAX_K}, not {k!r}")
        queries, own = self._queries(points)
        m = int(queries.shape[0])
        ids = tack.empty((m * k,), self.data.id_dtype)
        d2s = tack.empty((m * k,), self.data.dtype)
        if m:
            _nearest(self.grid(), queries, int(k), own, ids, d2s)
        return ids.reshape((m, k)), d2s.reshape((m, k))


def periodic_box(periodic):
    """A periodic box as ``((lower), (upper))``, three floats each, or None: from a
    side (the box from zero), three sides, or the two corners."""
    if periodic is None:
        return None
    values = np.asarray(periodic, np.float64)
    if values.shape in ((), (3,)):
        lower, upper = np.zeros(3), np.broadcast_to(values, (3,))
    elif values.shape == (2, 3):
        lower, upper = values
    else:
        raise ValueError(f"a periodic box is a side, three sides or two corners, not "
                         f"{periodic!r}")
    if not (np.all(np.isfinite(values)) and np.all(upper > lower)):
        raise ValueError(f"a periodic box must have positive, finite sides, not {periodic!r}")
    return tuple(map(float, lower)), tuple(map(float, upper))


def cached_point_locator(data, periodic=None):
    """The point locator ``data``'s topology keeps (one, at the default bins), made
    now if it has none or the one it has does not serve ``data`` in that box."""
    topology = data.topology
    locator = topology._point_locator
    if locator is None or not locator.serves(data, periodic):
        locator = PointLocator(data, periodic=periodic)
        topology._point_locator = locator
    return locator
