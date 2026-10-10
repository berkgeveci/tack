"""Finding the cell a point lies in, and probing fields there.

``CellLocator(data)`` bins the cells' bounding boxes on a uniform grid over
the dataset's bounds, about one cell per bin (Viskores' CellLocatorUniformBins
and VTK's cell locators do the same in spirit). ``find(points)`` returns each
point's cell -- the smallest id among the cells that hold it, -1 for none --
and its parametric coordinates there, found by the shapes' own Newton
inversion (``world_to_parametric``, VTK's EvaluatePosition) and domain test.

The search runs from the cells, not the points: the points are bucketed into
the same bins, and every cell, through its own view, tests the points in the
bins its box overlaps. A view addresses its cells its own way -- a structured
grid's by (i, j, k) -- so a cell is reached through its group's launch, never
looked up by id. The smallest holder wins by an atomic minimum, and a second
pass lets it write the coordinates. ``probe`` evaluates fields the same way:
the points are bucketed by the cell found, and each cell evaluates its fields
at its own points through the fields' bases.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import gather
from tack.data import arrays, ids
from tack.data.buckets import bucket_order
from tack.data.dataset import DataSet, Field, for_each
from tack.data.spaces import H1, Constant, Values

__all__ = ["CellLocator", "probe"]

# A point just outside a cell counts as in it within this share of the cell's
# diagonal, as vtkProbeFilter's computed tolerance has it (0.1%).
_TOLERANCE = 1e-3


@tack.kernel
def _cell_boxes(cells, lo, hi):
    for c in cells:
        e = cells.entity_id(c)
        a = cells.point(c, 0)
        b = a
        for j in range(1, cells.NUM_POINTS):
            p = cells.point(c, j)
            a = [min(a[0], p[0]), min(a[1], p[1]), min(a[2], p[2])]
            b = [max(b[0], p[0]), max(b[1], p[1]), max(b[2], p[2])]
        lo[e] = a
        hi[e] = b


@tack.func
def _bin(x, origin, inverse, n, slack):
    """The bin along one axis of coordinate ``x``, the grid widened by ``slack``
    on both sides (its edge bins take the margin); -1 outside that."""
    f = (x - origin) * inverse
    reach = slack * inverse
    i = tack.i64(-1)
    if f >= -reach and f <= n + reach:
        i = min(max(tack.i64(floor(f)), 0), n - 1)
    return i


@tack.kernel
def _point_bins(points, ox, oy, oz, ix, iy, iz, nx, ny, nz, slack, keys):
    # Each point under its bin (past the last bin: outside the grid).
    for q in range(keys.shape[0]):
        x = points[q]
        i = _bin(x[0], ox, ix, nx, slack)
        j = _bin(x[1], oy, iy, ny, slack)
        k = _bin(x[2], oz, iz, nz, slack)
        b = tack.i64(nx * ny * nz)
        if i >= 0 and j >= 0 and k >= 0:
            b = i + nx * (j + ny * k)
        keys[q] = b


@tack.kernel
def _test_points(cells, points, lo, hi, bin_offsets, bin_points, ox, oy, oz, ix, iy, iz,
                 nx, ny, nz, tolerance, write, found, pcs):
    for c in cells:
        e = cells.entity_id(c)
        pts = tack.local_array_like(lo, 3 * cells.NUM_POINTS)
        for j in range(cells.NUM_POINTS):
            p = cells.point(c, j)
            pts[3 * j] = p[0]
            pts[3 * j + 1] = p[1]
            pts[3 * j + 2] = p[2]
        a = lo[e]
        b = hi[e]
        # The bins of the box widened by the cell's tolerance.
        slack = tolerance * (b - a).norm()
        i0 = max(tack.i64(floor((a[0] - slack - ox) * ix)), 0)
        j0 = max(tack.i64(floor((a[1] - slack - oy) * iy)), 0)
        k0 = max(tack.i64(floor((a[2] - slack - oz) * iz)), 0)
        i1 = min(tack.i64(floor((b[0] + slack - ox) * ix)), nx - 1)
        j1 = min(tack.i64(floor((b[1] + slack - oy) * iy)), ny - 1)
        k1 = min(tack.i64(floor((b[2] + slack - oz) * iz)), nz - 1)
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    bin_ = i + nx * (j + ny * k)
                    for n in range(bin_offsets[bin_], bin_offsets[bin_ + 1]):
                        q = bin_points[n]
                        x = points[q]
                        if (x[0] >= a[0] - slack and x[0] <= b[0] + slack
                                and x[1] >= a[1] - slack and x[1] <= b[1] + slack
                                and x[2] >= a[2] - slack and x[2] <= b[2] + slack):
                            pc, ok = cells.world_to_parametric(pts, x)
                            # VTK's probe: inside, or within 0.1% of the cell's
                            # diagonal of its nearest point (surface cells too).
                            near = 0
                            if ok == 1:
                                if cells.DIMENSION == 3 and cells.is_inside(pc, 0.0) == 1:
                                    near = 1
                                else:
                                    d = cells.nearest_point(pts, x, pc) - x
                                    near = 1 if d.dot(d) <= slack * slack else 0
                            if near == 1:
                                if write == 0:
                                    tack.atomic_min(found, q, e)
                                elif found[q] == e:
                                    pcs[q] = pc


@tack.kernel
def _bins_per_cell(lo, hi, ox, oy, oz, ix, iy, iz, nx, ny, nz, tolerance, counts):
    # The bins each cell's box, widened by its tolerance, overlaps.
    for e in range(counts.shape[0]):
        a = lo[e]
        b = hi[e]
        slack = tolerance * (b - a).norm()
        i0 = max(tack.i64(floor((a[0] - slack - ox) * ix)), 0)
        j0 = max(tack.i64(floor((a[1] - slack - oy) * iy)), 0)
        k0 = max(tack.i64(floor((a[2] - slack - oz) * iz)), 0)
        i1 = min(tack.i64(floor((b[0] + slack - ox) * ix)), nx - 1)
        j1 = min(tack.i64(floor((b[1] + slack - oy) * iy)), ny - 1)
        k1 = min(tack.i64(floor((b[2] + slack - oz) * iz)), nz - 1)
        counts[e] = max(i1 - i0 + 1, 0) * max(j1 - j0 + 1, 0) * max(k1 - k0 + 1, 0)


@tack.kernel
def _bin_entries(lo, hi, ox, oy, oz, ix, iy, iz, nx, ny, nz, tolerance, starts, keys, cells):
    for e in range(starts.shape[0]):
        a = lo[e]
        b = hi[e]
        slack = tolerance * (b - a).norm()
        i0 = max(tack.i64(floor((a[0] - slack - ox) * ix)), 0)
        j0 = max(tack.i64(floor((a[1] - slack - oy) * iy)), 0)
        k0 = max(tack.i64(floor((a[2] - slack - oz) * iz)), 0)
        i1 = min(tack.i64(floor((b[0] + slack - ox) * ix)), nx - 1)
        j1 = min(tack.i64(floor((b[1] + slack - oy) * iy)), ny - 1)
        k1 = min(tack.i64(floor((b[2] + slack - oz) * iz)), nz - 1)
        at = starts[e]
        for k in range(k0, k1 + 1):
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    keys[at] = i + nx * (j + ny * k)
                    cells[at] = e
                    at += 1


@tack.kernel
def _unfound(found, none):
    for q in range(found.shape[0]):
        if found[q] == none:
            found[q] = -1


class CellLocator:
    """Locates points in ``data``'s cells; build once, ``find`` many times. The bins
    follow the cells' bounding boxes at build time: rebuild after moving the
    geometry. ``density`` is the bins per cell."""

    def __init__(self, data, density=1.0):
        if not getattr(data.topology, "reference_cells", True):
            raise NotImplementedError("locating points in polyhedra is not built yet")
        self.data = data
        n = data.num_cells
        dtype = data.dtype
        self.lo = tack.Vector.field(3, dtype, shape=(n,))
        self.hi = tack.Vector.field(3, dtype, shape=(n,))
        if n:
            for_each(_cell_boxes, data, "cells", self.lo, self.hi)
            lo, hi = self.lo.to_numpy(vectors=True), self.hi.to_numpy(vectors=True)
            low, high = lo.min(axis=0).astype(float), hi.max(axis=0).astype(float)
        else:
            low = high = np.zeros(3)
        extent = high - low
        size = max(float(extent.max()), 1e-30)
        live = extent > 1e-9 * size
        # About density * n bins, cubes along the axes the cells span.
        side = (np.prod(extent[live]) / max(density * n, 1)) ** (1.0 / max(live.sum(), 1))
        dims = np.where(live, np.clip(np.ceil(extent / max(side, 1e-30)), 1, 1024), 1)
        self.dims = dims.astype(np.int64)
        self.origin = low
        self.inverse = np.where(live, self.dims / np.where(live, extent, 1.0), 0.0)
        # A point this close to the grid can be within a cell's tolerance.
        self.slack = _TOLERANCE * float(np.linalg.norm(extent))

    def _grid(self):
        return (*map(float, self.origin), *map(float, self.inverse), *map(int, self.dims))

    def cell_bins(self):
        """The cells each bin may hold, for queries that run from the points (a
        particle asking for its cell): ``(offsets, cells)``, CSR by bin, the
        cells whose boxes, widened by their tolerance, overlap the bin. Made on
        first use and kept."""
        if "_cell_bins" not in self.__dict__:
            data = self.data
            n = data.num_cells
            nbins = int(np.prod(self.dims))
            counts = tack.field(tack.i32, shape=(n,))
            starts = tack.field(tack.i64, shape=(n,))
            total = 0
            if n:
                _bins_per_cell(self.lo, self.hi, *self._grid(), _TOLERANCE, counts)
                total = exclusive_scan(counts, starts, n)
            index = ids.at_least(data.id_dtype, total)
            keys = tack.field(tack.i32, shape=(total,))
            cells = tack.field(data.id_dtype, shape=(total,))
            if total:
                _bin_entries(self.lo, self.hi, *self._grid(), _TOLERANCE, starts, keys, cells)
            order, offsets = bucket_order(keys, nbins, index)
            self._cell_bins = (offsets, gather(cells, order) if total else cells)
        return self._cell_bins

    def find(self, points):
        """``(cells, pcs)`` for ``points`` (an ``(n, 3)`` host array or a field of
        3-vectors): a field of the cell holding each point (-1 for none), in the
        dataset's ``id_dtype``, and a field of its parametric coordinates there."""
        data = self.data
        dtype = data.dtype
        if not hasattr(points, "to_numpy"):
            host = np.ascontiguousarray(np.asarray(points, dtype.numpy_dtype).reshape(-1, 3))
            points = tack.Vector.field(3, dtype, shape=(len(host),))
            if len(host):
                points.from_numpy(host)
        m = arrays.size_of(points)
        # The smallest cell holding a point wins by an atomic minimum: in i32 below
        # 2**31 cells, as every backend has those atomics; in i64 past that.
        scratch = ids.at_least(tack.i32, data.num_cells)
        none = ids.LIMIT if scratch == tack.i32 else 2**63 - 1
        if scratch == tack.i64:
            from tack.runtime.dispatch import get_backend

            if tack.i64 not in get_backend().supported_atomic_dtypes:
                raise NotImplementedError(
                    f"locating points among {data.num_cells} cells needs 64-bit atomics, "
                    f"which the {get_backend().name} backend does not have")
        found = tack.full(scratch, (m,), none) if m else tack.field(scratch, shape=(0,))
        pcs = tack.Vector.field(3, dtype, shape=(m,))
        if m:
            pcs.from_numpy(np.zeros((m, 3), dtype.numpy_dtype))
        if m and data.num_cells:
            nbins = int(np.prod(self.dims))
            keys = tack.field(tack.i32, shape=(m,))
            _point_bins(points, *self._grid(), float(self.slack), keys)
            # The extra bin: outside the grid.
            order, starts = bucket_order(keys, nbins + 1, ids.at_least(data.id_dtype, m))
            for write in (0, 1):
                for_each(_test_points, data, "cells", points, self.lo, self.hi, starts, order,
                         *self._grid(), _TOLERANCE, write, found, pcs)
            _unfound(found, none)
        elif m:
            found.fill(-1)
        return ids.as_ids(found, data.id_dtype), pcs


@tack.kernel
def _evaluate(cells, u, links, link_points, pcs, out):
    for c in cells:
        e = cells.entity_id(c)
        for n in range(links[e], links[e + 1]):
            q = link_points[n]
            out[q] = u.value(c, pcs[q])


@tack.kernel
def _found_keys(found, ncells, keys):
    for q in range(found.shape[0]):
        keys[q] = found[q] if found[q] >= 0 else ncells


@tack.kernel
def _valid(found, out):
    for q in range(found.shape[0]):
        out[q] = 1 if found[q] >= 0 else 0


def _vertices(points):
    from tack.data.filters import _vertex_cells
    from tack.data.topology import UnstructuredTopology

    n = len(points)
    idt = ids.choose(None, n)
    types = tack.field(tack.u8, shape=(n,))
    offsets = tack.zeros(idt, (n + 1,))
    connectivity = tack.field(idt, shape=(n,))
    if n:
        _vertex_cells(types, offsets, connectivity)
    return DataSet(UnstructuredTopology(types, offsets, connectivity, num_points=n), points)


def probe(data, where, fields=None, locator=None):
    """``data``'s fields at the points of ``where`` -- a dataset, or an ``(n, 3)``
    array of points, which becomes vertex cells -- as Viskores' Probe and VTK's
    vtkProbeFilter: the result is ``where`` with each field as point data, a
    field with a basis (``H1``, ``L2``, ``Constant``, values on points)
    evaluated through it in the cell holding the point, values on cells taken
    from that cell, and ``valid`` (1, or 0 where no cell holds the point and the
    values are 0). ``locator`` reuses a ``CellLocator`` of ``data``."""
    from tack.data.carry import selected

    if not isinstance(where, DataSet):
        where = _vertices(np.asarray(where, float).reshape(-1, 3))
    if not isinstance(where.geometry.space, H1):
        raise TypeError("probe needs the probed points: an H1 geometry")
    locator = locator or CellLocator(data)
    points = arrays.materialize(where.geometry.values)
    found, pcs = locator.find(points)
    m = found.shape[0]
    keys = tack.field(found.dtype, shape=(m,))
    if m:
        _found_keys(found, data.num_cells, keys)
    # The extra cell: not found.
    order, links = bucket_order(keys, data.num_cells + 1, ids.at_least(data.id_dtype, m))

    out = {}
    for name in selected(data, fields):
        f = data.fields[name]
        space = f.space
        if isinstance(space, Values) and space.on in ("points", "cells"):
            f = Field((H1 if space.on == "points" else Constant)(data), f.values)
            space = f.space
        if not getattr(space, "interpolated", False) or space.on not in ("points", "cells"):
            continue
        width = arrays.width_of(f.values)
        dtype = arrays.dtype_of(f.values)
        values = (tack.Vector.field(width, dtype, shape=(m,)) if width
                  else tack.field(dtype, shape=(m,)))
        if m:
            values.from_numpy(np.zeros((m, width) if width else (m,), dtype.numpy_dtype))
            for group in data.launch_groups("cells", [f]):
                if group.count:
                    _evaluate(data.domain_view("cells", group), f.view(group), links, order,
                              pcs, values)
        out[name] = Field(H1(where), values)
    valid = tack.field(tack.i32, shape=(m,))
    if m:
        _valid(found, valid)
    out["valid"] = Field(Values(where, "points"), valid)
    kept = {name: f for name, f in where.fields.items() if name != "shape"}
    kept.update(out)
    return DataSet(where.topology, where.geometry, fields=kept, sets=dict(where.sets))
