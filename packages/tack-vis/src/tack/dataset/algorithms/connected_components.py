"""``connected_components``: which cells are connected through shared points, as VTK's
vtkConnectivityFilter and Viskores' CellSetConnectivity: the dataset with a
``region`` field on its cells and points."""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids
from tack.dataset.dataset import DataSet, Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.partitioned import PartitionedDataSet
from tack.dataset.spaces import Values
from tack.dataset.topology import StructuredTopology
from tack.runtime.dispatch import get_backend, on_inputs_device

__all__ = ["connected_components"]


@tack.kernel
def _structured_corners(members, cell_of, cx, cy, px, py, corners):
    # Each cell's corners of a structured grid, x fastest, in any order.
    for e in range(members.shape[0]):
        c = e // corners
        cell_of[e] = c
        t = e % corners
        i = c % cx
        j = (c // cx) % cy
        k = c // (cx * cy)
        members[e] = (i + (t & 1)) + px * ((j + ((t >> 1) & 1)) + py * (k + ((t >> 2) & 1)))


@tack.kernel
def _cells_of_entries(offsets, cell_of):
    for c in range(offsets.shape[0] - 1):
        for e in range(offsets[c], offsets[c + 1]):
            cell_of[e] = c


@tack.kernel
def _hook(cell_of, members, ncells, hidden, labels, changed):
    # Each cell-point entry joins the two trees: the larger root under the
    # smaller (an atomic minimum, so concurrent hooks agree).
    for e in range(members.shape[0]):
        c = cell_of[e]
        if hidden[c] == 0:
            a = labels[c]
            b = labels[ncells + members[e]]
            if a != b:
                tack.atomic_min(labels, max(a, b), min(a, b))
                tack.atomic_max(changed, 0, 1)


@tack.kernel
def _jump(labels):
    # Every node straight to its root.
    for v in range(labels.shape[0]):
        r = labels[v]
        while labels[r] != r:
            r = labels[r]
        labels[v] = r


@tack.kernel
def _roots(labels, hidden, flags):
    for c in range(flags.shape[0]):
        flags[c] = 1 if labels[c] == c and hidden[c] == 0 else 0


@tack.kernel
def _regions(labels, hidden, index, ncells, cell_region, point_region):
    for v in range(labels.shape[0]):
        r = labels[v]
        if v < ncells:
            cell_region[v] = index[r] if hidden[v] == 0 else -1
        else:
            point_region[v - ncells] = index[r] if r < ncells else -1


def _cell_points(data):
    """Each cell's points on the device, as ``(cell of each entry, point ids)``."""
    topology = data.topology
    if isinstance(topology, StructuredTopology):
        dims = [*topology.point_dims, 1, 1]
        corners = 2 ** len(topology.point_dims)
        members = tack.empty((data.num_cells * corners,), tack.i64)
        cell_of = tack.empty((data.num_cells * corners,), tack.i64)
        if members.size:
            _structured_corners(members, cell_of, dims[0] - 1, max(dims[1] - 1, 1), dims[0],
                                dims[1], corners)
        return cell_of, members
    from tack.dataset.ghosts import _rows

    offsets, members = _rows(topology)
    offsets, members = arrays.materialize(offsets), arrays.materialize(members)
    cell_of = tack.empty((members.shape[0],), offsets.dtype)
    if members.shape[0]:
        _cells_of_entries(offsets, cell_of)
    return cell_of, members


def _labels(data):
    """Each cell's and point's component on ``data``'s device: ``(cell regions,
    point regions, count)``, regions numbered from 0 in order of their lowest
    cell, -1 for hidden cells and points no visible cell uses."""
    ncells, npoints = data.num_cells, data.num_points
    dtype = ids.at_least(tack.i32, ncells + npoints + 1)
    if dtype not in get_backend().supported_atomic_dtypes:
        raise NotImplementedError(f"connected components of {ncells + npoints} cells and "
                                  f"points need {dtype} atomics, which the "
                                  f"{get_backend().name} backend does not have")
    marks = data.ghosts.cells if data.ghosts is not None else None
    hidden = (tack.array((marks.numpy() & Ghosts.HIDDENCELL).astype(np.uint8), tack.u8)
              if marks is not None else tack.zeros((max(ncells, 1),), tack.u8))
    cell_of, members = _cell_points(data)
    labels = tack.arange(ncells + npoints, dtype=dtype)
    changed = tack.zeros((1,), tack.i32)
    while members.shape[0]:
        changed.fill(0)
        _hook(cell_of, members, ncells, hidden, labels, changed)
        _jump(labels)
        if not int(changed.numpy()[0]):
            break
    flags = tack.empty((ncells,), dtype)
    index = tack.empty((ncells,), dtype)
    count = 0
    if ncells:
        _roots(labels, hidden, flags)
        count = int(exclusive_scan(flags, index, ncells))
    cell_region = tack.empty((ncells,), dtype)
    point_region = tack.empty((npoints,), dtype)
    if ncells + npoints:
        _regions(labels, hidden, index if ncells else tack.zeros((1,), dtype), ncells,
                 cell_region if ncells else tack.zeros((1,), dtype), point_region
                 if npoints else tack.zeros((1,), dtype))
    return cell_region, point_region, count


@on_inputs_device
def connected_components(data):
    """Which cells are connected, through shared points: ``data`` with ``region``, a
    field on its cells and one on its points, the regions numbered from 0 in
    order of their lowest cell id, as VTK's vtkConnectivityFilter (``RegionId``,
    all regions) and Viskores' CellSetConnectivity: ``region`` on the cells,
    ``point_region`` on the points. A point takes its cells' region; a point no
    cell uses, and a hidden cell, has -1. The largest region, or any, is a
    ``threshold`` of ``region`` away.

    Computed on the device: union-find over the cells and their points, each
    cell-point pair hooking the larger root under the smaller with an atomic
    minimum, then every node jumping to its root, until nothing changes.

    Over a ``PartitionedDataSet``, regions are numbered across every piece of
    every rank: each piece labels its own, and pieces join the regions that
    meet at the points they share (``_components_pieces``). With global ids on
    every piece, regions are numbered in order of their lowest global cell id
    -- the whole's numbering, for pieces carrying the whole's ids."""
    if isinstance(data, PartitionedDataSet):
        return _components_pieces(data)
    cell_region, point_region, _ = _labels(data)
    return _with_regions(data, cell_region, point_region)


def _with_regions(data, cell_region, point_region):
    from tack.dataset.ghosts import keep_marks

    out = DataSet(data.topology, data.geometry, fields=dict(data.fields),
                  sets=dict(data.sets), dtype=data.dtype)
    out.fields["region"] = Field(Values(out, "cells"), cell_region)
    out.fields["point_region"] = Field(Values(out, "points"), point_region)
    return keep_marks(data, out)


def _components_pieces(data):
    """``connected_components`` over the pieces of every rank.

    Each piece labels its own cells and points (``_labels``). Points several
    pieces hold -- the copies matched by global ids where every piece has them,
    else by exact position (``_matching.match``) -- join the local regions they
    lie in: each piece sends its partners, for each shared point, its region
    there, and the edges between (piece, region) pairs, a small graph, are
    gathered and joined on the host. A global region takes the lowest key of
    its pieces' regions: the lowest global id of an owned cell, else (piece,
    cell)."""
    from tack import communication as tc
    from tack.dataset import _matching
    from tack.dataset._matching import INDEX_BITS

    pieces = list(data)
    blocks = tc.Blocks([p.device for p in pieces], data.world)
    local = {}
    for g, piece in zip(blocks.local, pieces):
        with piece.device:
            local[g] = _labels(piece)
    counts = {g: local[g][2] for g in blocks.local}
    starts, total = blocks.exclusive_scan(counts)

    # Copies of points across pieces.
    numbered = bool(blocks.all_reduce({g: int(p.global_ids is not None and
                                              p.global_ids.points is not None)
                                       for g, p in zip(blocks.local, pieces)}, "min")) \
        if blocks.count else False
    duplicates = blocks.all_reduce({g: int(p.ghosts is not None and p.ghosts.points is not None
                                           and bool((p.ghosts.points.numpy() &
                                                     Ghosts.DUPLICATEPOINT).any()))
                                    for g, p in zip(blocks.local, pieces)}, "max")
    entries = {}
    for g, piece in zip(blocks.local, pieces):
        if numbered or duplicates:
            chosen = np.arange(piece.num_points, dtype=np.int64)
        else:
            faces = piece.topology.faces()
            rows = arrays.to_host(faces.rows).reshape(-1, 4)
            chosen = np.unique(rows[arrays.to_host(faces.boundary()).astype(np.int64)])
            chosen = chosen[chosen >= 0].astype(np.int64)
        if numbered:
            columns = [piece.global_ids.points.numpy()[chosen].astype(np.int64).view(np.uint64)]
        else:
            columns = _matching.position_bits(piece.positions()[chosen])
        entries[g] = (chosen, columns, np.zeros(len(chosen), np.uint8))
    partners = _matching.match(blocks, entries)

    # Each piece tells its partners its region at each shared point.
    outgoing = {}
    for g in blocks.local:
        index, other, other_index, _ = partners[g]
        point_region = local[g][1].numpy()
        mine = point_region[index] if len(index) else np.zeros(0, np.int64)
        keep = mine >= 0
        outgoing[g] = {}
        for d in np.unique(other[keep]).tolist():
            rows = keep & (other == d)
            pairs = np.unique(np.stack([other_index[rows], mine[rows]], axis=1), axis=0)
            outgoing[g][int(d)] = {
                "point": tack.array(pairs[:, 0].copy(), tack.i64, device=blocks.device(g)),
                "region": tack.array(pairs[:, 1].copy(), tack.i64, device=blocks.device(g))}
    incoming = blocks.exchange(outgoing)
    edges = {}
    for g in blocks.local:
        point_region = local[g][1].numpy()
        found = []
        for d, message in incoming[g].items():
            theirs = message["region"].numpy()
            ours = point_region[message["point"].numpy()]
            keep = ours >= 0
            found.append(np.stack([starts[g] + ours[keep], starts[d] + theirs[keep]], axis=1))
        edges[g] = np.unique(np.concatenate(found), axis=0) if found else np.zeros((0, 2), np.int64)

    # Each local region's key: its lowest owned cell, by global id or (piece, cell).
    keys = {}
    for g, piece in zip(blocks.local, pieces):
        cell_region, _, count = local[g]
        regions = cell_region.numpy()
        owned = regions >= 0
        if piece.ghosts is not None and piece.ghosts.cells is not None:
            owned &= (piece.ghosts.cells.numpy() & Ghosts.DUPLICATECELL) == 0
        if numbered and piece.global_ids.cells is not None:
            key = piece.global_ids.cells.numpy().astype(np.int64)
        else:
            key = (np.int64(g) << INDEX_BITS) | np.arange(piece.num_cells, dtype=np.int64)
        lowest = np.full(count, np.iinfo(np.int64).max, np.int64)
        np.minimum.at(lowest, regions[owned], key[owned])
        keys[g] = lowest
    every_edges = blocks.all_gather({g: edges[g] for g in blocks.local})
    every_keys = blocks.all_gather({g: keys[g] for g in blocks.local})

    # Join on the host: the graph has one node per piece's region.
    parent = np.arange(total, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in every_edges:
        for a, b in e.tolist():
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(x) for x in range(total)], np.int64)
    key = np.concatenate(every_keys) if every_keys else np.zeros(0, np.int64)
    root_key = np.full(total, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(root_key, roots, key)
    distinct = np.unique(roots)
    order = distinct[np.argsort(root_key[distinct], kind="stable")]
    number = np.empty(total, np.int64)
    number[order] = np.arange(len(order))
    region_of = number[roots]

    out = []
    for g, piece in zip(blocks.local, pieces):
        cell_region, point_region, _ = local[g]
        with piece.device:
            mapping = region_of[starts[g]:starts[g] + counts[g]]
            cells = cell_region.numpy()
            points = point_region.numpy()
            dtype = cell_region.dtype
            cells = np.where(cells >= 0, mapping[np.maximum(cells, 0)] if counts[g] else -1, -1)
            points = np.where(points >= 0, mapping[np.maximum(points, 0)] if counts[g] else -1,
                              -1)
            out.append(_with_regions(piece, tack.array(cells.astype(dtype.numpy_dtype), dtype),
                                     tack.array(points.astype(dtype.numpy_dtype), dtype)))
    return PartitionedDataSet(out, world=data.world)
