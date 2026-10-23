"""Finding the cell a point lies in, and probing fields there.

``CellLocator(data)`` bins the cells' bounding boxes on a uniform grid over
the dataset's bounds, about one cell per bin (Viskores' CellLocatorUniformBins
and VTK's cell locators do the same in spirit). ``find(points)`` returns each
point's cell -- of the cells that hold it the smallest id, or, if none does
but some come within the tolerance, the smallest of those; -1 for none --
and its parametric coordinates there, found by the shapes' own Newton
inversion (``world_to_parametric``, VTK's EvaluatePosition) and domain test.

The search runs from the points where ``by_id`` serves the dataset (an
order-1 geometry over the shapes it knows, structured grids, polyhedra): one
launch, each point searching the cells its bin may hold (``cell_bins``),
reached by id. Elsewhere it runs from the cells: the points are bucketed into
the same bins, and every cell, through its own view, tests the points in the
bins its box overlaps (a view addresses its cells its own way, a structured
grid's by (i, j, k)), the winner decided by an atomic minimum and a second pass
writing its coordinates. A fresh locator's first query of few points in many
cells also runs from the cells, sparing the cell bins (about eight entries a
cell) until it is used again. Both rank candidates alike: a cell that holds the
point ranks ahead of one that only comes within the tolerance, then the smaller
id wins. Without the first part a point just inside a cell went to a
lower-numbered neighbour within the tolerance, and was extrapolated there.

``probe`` reads point and cell data by cell id where the search ran from the
points (one launch per field); other fields, and every field where it ran
from the cells, are evaluated cell by cell through the fields' bases, the
points bucketed by the cell found.
"""

from typing import NamedTuple

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import gather
from tack.dataset import ids
from tack.dataset.buckets import bucket_groups
from tack.dataset.by_id import _bin
from tack.dataset.dataset import for_each
from tack.dataset.shapes import HOLD_TOLERANCE
from tack.runtime.dispatch import on_inputs_device

__all__ = ["CellLocator", "Locations"]


class Locations(NamedTuple):
    """Where ``CellLocator.find`` put each point: ``cells``, the cell holding it (-1
    for none), in the dataset's ``id_dtype``; ``pcs``, its coordinates there; and
    ``tets``, in polyhedra, its sub-tetrahedron in the cell (i32, -1 for none),
    ``None`` on cells with a reference element. On those ``pcs`` are parametric
    coordinates; in a polyhedron they are the weights of the sub-tetrahedron's
    face point and edge points (``views._PolyhedralLocation``), which a field's
    view evaluates with ``tet_value(c, t, pc)``."""

    cells: object
    pcs: object
    tets: object = None

# A point just outside a cell counts as in it within this share of the cell's
# diagonal, as vtkProbeFilter's computed tolerance has it (0.1%).
_TOLERANCE = 1e-3


@tack.kernel
def _cell_boxes(cells, lo, hi):
    for c in cells:
        e = cells.entity_id(c)
        a = cells.corner_position(c, 0)
        b = a
        for j in range(1, cells.NUM_POINTS):
            p = cells.corner_position(c, j)
            a = [min(a[0], p[0]), min(a[1], p[1]), min(a[2], p[2])]
            b = [max(b[0], p[0]), max(b[1], p[1]), max(b[2], p[2])]
        lo[e] = a
        hi[e] = b


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
                 nx, ny, nz, tolerance, ncells, write, found, pcs):
    for c in cells:
        e = cells.entity_id(c)
        pts = tack.local_array_like(lo, 3 * cells.NUM_POINTS)
        for j in range(cells.NUM_POINTS):
            p = cells.corner_position(c, j)
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
                            holds = 0
                            if ok == 1:
                                holds = cells.is_inside(pc, HOLD_TOLERANCE)
                                if cells.DIMENSION == 3 and holds == 1:
                                    near = 1
                                else:
                                    d = cells.nearest_point(pts, x, pc) - x
                                    near = 1 if d.dot(d) <= slack * slack else 0
                            if near == 1:
                                # Holding the point (within HOLD_TOLERANCE) ranks
                                # ahead of coming within the tolerance; then the
                                # smaller id.
                                rank = e if holds == 1 else e + ncells
                                if write == 0:
                                    tack.atomic_min(found, q, rank)
                                elif found[q] == rank:
                                    pcs[q] = pc


@tack.kernel
def _polyhedral_boxes(cells, lo, hi):
    for c in cells:
        e = cells.entity_id(c)
        a = cells.point_position(cells.point_id(c, 0))
        b = a
        for j in range(1, cells.num_points(c)):
            p = cells.point_position(cells.point_id(c, j))
            a = [min(a[0], p[0]), min(a[1], p[1]), min(a[2], p[2])]
            b = [max(b[0], p[0]), max(b[1], p[1]), max(b[2], p[2])]
        lo[e] = a
        hi[e] = b


@tack.kernel
def _not_star_shaped(cells, bad):
    # 1 where a sub-tetrahedron is inverted, or the cell has no volume: a cell
    # not star-shaped from its cell point, whose sub-tetrahedra overlap.
    for c in cells:
        p = cells.cell_point(c)
        total = 0.0
        least = 0.0
        for k in range(cells.num_faces(c)):
            n = cells.side_num_points(c, k)
            f = cells.face_point(c, k) - p
            for j in range(n):
                nxt = j + 1 if j + 1 < n else 0
                v = f.dot((cells.side_position(c, k, j) - p).cross(
                    cells.side_position(c, k, nxt) - p))
                total += v
                least = min(least, v)
        bad[cells.entity_id(c)] = 1 if total <= 0.0 or least < -1e-5 * total else 0


@tack.kernel
def _not_star_shaped_polygons(cells, bad):
    # 1 where a triangle of the fan from the cell point turns the other way from
    # the polygon (its Newell normal), or the polygon has no area: one not
    # star-shaped from its cell point, whose triangles overlap.
    for c in cells:
        p = cells.cell_point(c)
        normal = p - p
        for k in range(cells.num_faces(c)):
            normal += (cells.side_position(c, k, 0) - p).cross(cells.side_position(c, k, 1) - p)
        total = normal.dot(normal)
        least = total
        for k in range(cells.num_faces(c)):
            v = (cells.side_position(c, k, 0) - p).cross(
                cells.side_position(c, k, 1) - p).dot(normal)
            least = min(least, v)
        bad[cells.entity_id(c)] = 1 if total <= 0.0 or least < -1e-5 * total else 0


@tack.kernel
def _test_points_in_polyhedra(cells, points, lo, hi, bin_offsets, bin_points, ox, oy, oz,
                              ix, iy, iz, nx, ny, nz, tolerance, ncells, write, found, tets,
                              pcs):
    # As _test_points, locating in the cell's sub-tetrahedra, and ranking the
    # same way: holding the point ahead of coming within the tolerance.
    for c in cells:
        e = cells.entity_id(c)
        middle = cells.cell_point(c)
        a = lo[e]
        b = hi[e]
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
                            t, pc, inside = cells.locate_from(c, middle, x, slack)
                            if t >= 0:
                                rank = e if inside == 1 else e + ncells
                                if write == 0:
                                    tack.atomic_min(found, q, rank)
                                elif found[q] == rank:
                                    tets[q] = t
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
def _find_from_points(mesh, points, found, pcs, tets, write_tets):
    # One thread per point: its cell by the cell bins, ranked as the cell-side
    # search ranks them (by_id._MeshSearch.find); no atomics, no grouping.
    for q in range(points.shape[0]):
        e, kind, pc, t = mesh.find(points[q])
        found[q] = e if e < mesh.ncells else -1
        pcs[q] = pc
        if write_tets == 1:
            tets[q] = t


@tack.kernel
def _resolve(found, none, ncells):
    # A rank back to its cell; -1 for none.
    for q in range(found.shape[0]):
        r = found[q]
        if r == none:
            found[q] = -1
        elif r >= ncells:
            found[q] = r - ncells


class CellLocator:
    """Locates points in ``data``'s cells; build once, ``find`` many times. The bins
    follow the cells' bounding boxes at build time: rebuild after moving the
    geometry. ``density`` is the bins per cell."""

    @on_inputs_device
    def __init__(self, data, density=1.0):
        #: Cells with no reference element -- polyhedra, polygons, polylines --
        #: have no parametric coordinates: points are located in a piece of each
        #: cell's fan (a polyline's segment), which ``find`` says (``tets``).
        self.polyhedral = not getattr(data.topology, "reference_cells", True)
        self.data = data
        n = data.num_cells
        dtype = data.dtype
        self.lo = tack.empty((n,), tack.vec(3, dtype))
        self.hi = tack.empty((n,), tack.vec(3, dtype))
        if n and self.polyhedral:
            bad = tack.zeros((n,), tack.i32)
            dimension = data.topology.dimension
            if dimension >= 2:
                # A polyline needs no fan, so has no such failure.
                for_each(_not_star_shaped if dimension == 3 else _not_star_shaped_polygons,
                         data, "cells", bad)
            count = int(bad.numpy().sum())
            if count:
                raise NotImplementedError(
                    f"{count} of the {n} cells are not star-shaped from their cell point "
                    "(the mean of their points), or have no volume: locating points in "
                    "them needs another inside test, not built yet")
            for_each(_polyhedral_boxes, data, "cells", self.lo, self.hi)
        elif n:
            for_each(_cell_boxes, data, "cells", self.lo, self.hi)
        if n:
            lo, hi = self.lo.numpy(), self.hi.numpy()
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
            counts = tack.empty((n,), tack.i32)
            starts = tack.empty((n,), tack.i64)
            total = 0
            if n:
                _bins_per_cell(self.lo, self.hi, *self._grid(), _TOLERANCE, counts)
                total = exclusive_scan(counts, starts, n)
            index = ids.at_least(data.id_dtype, total)
            keys = tack.empty((total,), tack.i32)
            cells = tack.empty((total,), data.id_dtype)
            if total:
                _bin_entries(self.lo, self.hi, *self._grid(), _TOLERANCE, starts, keys, cells)
            # A bin's cells in any order: the search ranks its candidates itself.
            order, offsets = bucket_groups(keys, nbins, index)
            self._cell_bins = (offsets, gather(cells, order) if total else cells)
        return self._cell_bins

    @property
    def _device(self):
        data = self.__dict__.get("data")
        return data._device if data is not None else None

    def _search(self):
        """The search from the points (``by_id._MeshSearch``) over this locator's
        cell bins, or ``None`` where ``by_id`` cannot serve the dataset. Made on
        first use and kept, as the bins are."""
        if "_mesh_search" not in self.__dict__:
            from tack.dataset.by_id import mesh_search

            self._mesh_search = mesh_search(self.data, self)
        return self._mesh_search

    @on_inputs_device
    def find(self, points):
        """``Locations(cells, pcs, tets)`` for ``points`` (an ``(n, 3)`` host array or
        a field of 3-vectors): each point's cell (-1 for none) and its
        coordinates there, and in polyhedra its sub-tetrahedron (``tets`` is
        ``None`` otherwise). See ``Locations``."""
        data = self.data
        dtype = data.dtype
        if not hasattr(points, "numpy"):
            host = np.ascontiguousarray(np.asarray(points, dtype.numpy_dtype).reshape(-1, 3))
            points = tack.empty((len(host),), tack.vec(3, dtype))
            if len(host):
                tack.copyto(points, host)
        m = arrays.size_of(points)
        # From the points needs the cell bins (about 8 entries a cell); a first
        # query of few points in many cells is cheaper from the cells. Both give
        # the same answer.
        first = not self.__dict__.get("_found_before")
        self._found_before = True
        worth_bins = ("_cell_bins" in self.__dict__ or not first
                      or 4 * m >= data.num_cells)
        search = self._search() if worth_bins else None
        if search is not None:
            # From the points: one launch, each point searching its bin's cells.
            found = tack.empty((m,), data.id_dtype)
            pcs = tack.empty((m,), tack.vec(3, dtype))
            tets = tack.empty((m,), tack.i32) if self.polyhedral else None
            if m:
                _find_from_points(search, points, found, pcs,
                                  tets if tets is not None else found,
                                  1 if tets is not None else 0)
            return Locations(found, pcs, tets)
        # From the cells, where by_id cannot serve the dataset (an order-2 or L2
        # geometry, line or vertex cells). The best rank wins by an atomic minimum (a cell id, or past the ids for
        # a cell the point is only near): in i32 below 2**30 cells, as every
        # backend has those atomics; in i64 past that.
        scratch = ids.at_least(tack.i32, 2 * data.num_cells)
        none = ids.LIMIT if scratch == tack.i32 else 2**63 - 1
        if scratch == tack.i64:
            from tack.runtime.dispatch import get_backend

            if tack.i64 not in get_backend().supported_atomic_dtypes:
                raise NotImplementedError(
                    f"locating points among {data.num_cells} cells needs 64-bit atomics, "
                    f"which the {get_backend().name} backend does not have")
        found = tack.full((m,), none, scratch) if m else tack.empty((0,), scratch)
        pcs = tack.empty((m,), tack.vec(3, dtype))
        tets = tack.empty((m,), tack.i32) if self.polyhedral else None
        if m:
            pcs.fill(0)
            if tets is not None:
                tets.fill(-1)
        if m and data.num_cells:
            nbins = int(np.prod(self.dims))
            keys = tack.empty((m,), tack.i32)
            _point_bins(points, *self._grid(), float(self.slack), keys)
            # The extra bin: outside the grid, where no cell looks, so not filled.
            order, starts = bucket_groups(keys, nbins + 1, ids.at_least(data.id_dtype, m),
                                          skip=nbins)
            for write in (0, 1):
                if self.polyhedral:
                    for_each(_test_points_in_polyhedra, data, "cells", points, self.lo,
                             self.hi, starts, order, *self._grid(), _TOLERANCE,
                             data.num_cells, write, found, tets, pcs)
                else:
                    for_each(_test_points, data, "cells", points, self.lo, self.hi, starts,
                             order, *self._grid(), _TOLERANCE, data.num_cells, write, found,
                             pcs)
            _resolve(found, none, data.num_cells)
        elif m:
            found.fill(-1)
        if self.polyhedral:
            return Locations(ids.as_ids(found, data.id_dtype), pcs, tets)
        return Locations(ids.as_ids(found, data.id_dtype), pcs)



