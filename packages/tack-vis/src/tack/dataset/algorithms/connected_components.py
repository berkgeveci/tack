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
from tack.dataset.partitioned import PartitionedDataSet, common_origin
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
def _used_points(cell_of, members, hidden, used):
    # The points a visible cell uses: the ones a radius may join.
    for e in range(members.shape[0]):
        if hidden[cell_of[e]] == 0:
            used[members[e]] = 1


@tack.kernel
def _hook_near(grid, ncells, used, r, labels, changed):
    # Points within ``r`` of each other join, each pair once (p > q), through the
    # locator's bins: nothing is stored per pair.
    for q in range(used.shape[0]):
        if used[q] == 1:
            x = grid.positions[q]
            r2 = r * r
            i0, i1, j0, j1, k0, k1 = grid.span(x, r)
            for k in range(k0, k1 + 1):
                for j in range(j0, j1 + 1):
                    for i in range(i0, i1 + 1):
                        b = grid.bin(i, j, k)
                        for s in range(grid.first(b), grid.end(b)):
                            p = grid.point(s)
                            if p > q and used[p] == 1:
                                if grid.distance2(p, x) <= r2:
                                    a = labels[ncells + q]
                                    c = labels[ncells + p]
                                    if a != c:
                                        tack.atomic_min(labels, max(a, c), min(a, c))
                                        tack.atomic_max(changed, 0, 1)


@tack.kernel
def _region_sizes(point_region, sizes):
    for p in range(point_region.shape[0]):
        r = point_region[p]
        if r >= 0:
            tack.atomic_add(sizes, r, 1)


@tack.kernel
def _big_enough(sizes, least, flags):
    for r in range(flags.shape[0]):
        flags[r] = 1 if sizes[r] >= least else 0


@tack.kernel
def _renumbered(region, flags, index):
    # Regions too small become -1; the others are numbered again, in order.
    for v in range(region.shape[0]):
        r = region[v]
        if r >= 0:
            region[v] = index[r] if flags[r] == 1 else -1


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


def _labels(data, radius=None, locator=None, periodic=None):
    """Each cell's and point's component on ``data``'s device: ``(cell regions,
    point regions, count)``, regions numbered from 0 in order of their lowest
    cell, -1 for hidden cells and points no visible cell uses. With ``radius``,
    points of visible cells within it of each other join too."""
    ncells, npoints = data.num_cells, data.num_points
    dtype = ids.at_least(tack.i32, ncells + npoints + 1)
    if dtype not in get_backend().supported_atomic_dtypes:
        raise NotImplementedError(f"connected components of {ncells + npoints} cells and "
                                  f"points need {dtype} atomics, which the "
                                  f"{get_backend().name} backend does not have")
    marks = data.ghosts.cells if data.ghosts is not None else None
    hidden = (tack.array((marks.numpy() & Ghosts.MASKED_CELLS).astype(np.uint8), tack.u8)
              if marks is not None else tack.zeros((max(ncells, 1),), tack.u8))
    cell_of, members = _cell_points(data)
    labels = tack.arange(ncells + npoints, dtype=dtype)
    changed = tack.zeros((1,), tack.i32)
    near = None
    if radius is not None and npoints and members.shape[0]:
        from tack.dataset.algorithms.outliers import _locator

        used = tack.zeros((npoints,), tack.u8)
        _used_points(cell_of, members, hidden, used)
        found = _locator(data, locator, periodic)
        found.check_radius(radius)
        near = (found.grid(), used)
    while members.shape[0]:
        changed.fill(0)
        _hook(cell_of, members, ncells, hidden, labels, changed)
        if near is not None:
            _hook_near(near[0], ncells, near[1], float(radius), labels, changed)
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
def connected_components(data, radius=None, min_size=None, periodic=None, locator=None):
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
    -- the whole's numbering, for pieces carrying the whole's ids -- else of
    their lowest (piece, cell). With ``radius``, the pieces are point clouds:
    each labels itself with copies of the other pieces' points within
    ``radius`` (``ghost_points``), and a copy joins its region to its owner's.
    ``min_size`` counts each point once, where it is owned.

    With ``radius``, points within that distance of each other (inclusive) are
    connected too: on a point cloud, whose cells are its points, that is
    Euclidean clustering (VTK's vtkEuclideanClusterExtraction, PCL's) and,
    with the linking length, friends-of-friends halo finding (yt's FoF).
    ``min_size`` gives regions of fewer points -1, the rest numbered again in
    the same order. ``periodic`` (a box, as ``PointLocator`` takes it) joins
    points across its faces, as a cosmology box needs. ``locator`` reuses a
    ``PointLocator`` of ``data``."""
    if radius is not None and not radius >= 0:
        raise ValueError(f"radius must be at least zero, not {radius!r}")
    if isinstance(data, PartitionedDataSet):
        if locator is not None:
            raise ValueError("a PointLocator finds one piece's points; pieces take none")
        if radius is not None:
            return _components_near(data, float(radius), min_size, periodic)
        if periodic is not None:
            raise ValueError("periodic joins points within a radius across the box's faces")
        return _components_pieces(data, min_size)
    cell_region, point_region, count = _labels(data, radius, locator, periodic)
    if min_size is not None and count:
        _smallest_dropped(cell_region, point_region, count, int(min_size))
    return _with_regions(data, cell_region, point_region)


def _smallest_dropped(cell_region, point_region, count, least):
    """Regions of fewer than ``least`` points made -1, the rest renumbered in order."""
    sizes = tack.zeros((count,), point_region.dtype)
    if point_region.shape[0]:
        _region_sizes(point_region, sizes)
    flags = tack.empty((count,), point_region.dtype)
    index = tack.empty((count,), point_region.dtype)
    _big_enough(sizes, least, flags)
    exclusive_scan(flags, index, count)
    for region in (cell_region, point_region):
        if region.shape[0]:
            _renumbered(region, flags, index)


def _with_regions(data, cell_region, point_region):
    from tack.dataset.ghosts import keep_marks

    out = DataSet(data.topology, data.geometry, fields=dict(data.fields),
                  sets=dict(data.sets), dtype=data.dtype)
    out.fields["region"] = Field(Values(out, "cells"), cell_region)
    out.fields["point_region"] = Field(Values(out, "points"), point_region)
    return keep_marks(data, out)


def _components_pieces(data, least=None):
    """``connected_components`` over the pieces of every rank.

    Each piece labels its own cells and points (``_labels``). Points several
    pieces hold -- the copies matched by global ids where every piece has them,
    else by exact position (``_matching.match``) -- join the local regions they
    lie in: each piece sends its partners, for each shared point, its region
    there, and the edges between (piece, region) pairs are joined
    (``_joined_regions``). A global region takes the lowest key of its pieces'
    regions: the lowest global id of an owned cell, else (piece, cell)."""
    from tack import communication as tc
    from tack.dataset import _matching

    pieces = list(data)
    common_origin(data, "connected_components of a PartitionedDataSet")
    blocks = tc.Blocks([p.device for p in pieces], data.world)
    local = {}
    for g, piece in zip(blocks.local, pieces):
        with piece.device:
            local[g] = _labels(piece)
    counts = {g: local[g][2] for g in blocks.local}
    starts = _starts(blocks, counts)

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
    edges, keys, sizes = {}, {}, {}
    for g, piece in zip(blocks.local, pieces):
        point_region = local[g][1].numpy()
        found = []
        for d, message in incoming[g].items():
            theirs = message["region"].numpy()
            ours = point_region[message["point"].numpy()]
            keep = ours >= 0
            found.append(np.stack([starts[g] + ours[keep], starts[d] + theirs[keep]], axis=1))
        edges[g] = np.unique(np.concatenate(found), axis=0) if found else np.zeros((0, 2), np.int64)
        keys[g] = _lowest_owned_cells(g, piece, local[g], numbered)
        if least is not None:
            # Each point counted once: where its lowest unmarked copy is.
            marks = piece.ghosts.points.numpy() if piece.ghosts is not None and \
                piece.ghosts.points is not None else np.zeros(piece.num_points, np.uint8)
            identity = _matching.owners(g, piece.num_points, partners[g],
                                        (marks & Ghosts.DUPLICATEPOINT) != 0)
            sizes[g] = _owned_sizes(point_region, (identity >> _matching.INDEX_BITS) == g,
                                    counts[g])
    number = _joined_regions(blocks, counts, edges, keys,
                             sizes if least is not None else None, least)
    return PartitionedDataSet([_renumbered_piece(piece, local[g], number[g]) for g, piece
                               in zip(blocks.local, pieces)], world=data.world)


def _components_near(data, radius, least, periodic):
    """``connected_components(radius=)`` over pieces of point clouds: each piece
    labelled with copies of the other pieces' points within ``radius``
    (``ghost_points``; images in a periodic box), each copy's region joined to
    its owner's region at that point, which the copies' plan brings."""
    from tack.dataset._matching import INDEX_BITS
    from tack.dataset.algorithms.ghost_points import _ghosted

    if not radius >= 0:
        raise ValueError(f"radius must be at least zero, not {radius!r}")
    ghosted = _ghosted(data, radius, None, periodic, None, [], None)
    blocks = ghosted.blocks
    local, owner_region = {}, {}
    for g, piece in zip(blocks.local, ghosted.pieces):
        with piece.device:
            local[g] = _labels(piece, radius)
            owner_region[g] = local[g][1].astype(tack.i64)
    ghosted.plan.run(owner_region)         # each copy: its owner's region there
    counts = {g: local[g][2] for g in blocks.local}
    starts = _starts(blocks, counts)
    numbered = bool(blocks.all_reduce({g: int(p.global_ids is not None and
                                              p.global_ids.cells is not None)
                                       for g, p in zip(blocks.local, data)}, "min")) \
        if blocks.count else False
    edges, keys, sizes = {}, {}, {}
    for g, piece, source in zip(blocks.local, ghosted.pieces, data):
        n = ghosted.owned[g]
        point_region = local[g][1].numpy().astype(np.int64)
        theirs = owner_region[g].numpy()[n:]
        ours = point_region[n:]
        owner = ghosted.identity[g][n:] >> INDEX_BITS
        keep = (ours >= 0) & (theirs >= 0)
        a, b = ours[keep], starts[owner[keep]] + theirs[keep]
        owned = np.bincount(point_region[:n][point_region[:n] >= 0], minlength=counts[g])
        # A region of copies alone that meets one region of their owners is an
        # alias of it, and joins nothing.
        alias = owned[a] == 0
        if alias.any():
            pairs = np.unique(np.stack([a[alias], b[alias]], axis=1), axis=0)
            meets = np.bincount(pairs[:, 0], minlength=counts[g])
            alias &= meets[a] < 2
        edges[g] = np.unique(np.stack([starts[g] + a[~alias], b[~alias]], axis=1), axis=0) \
            if len(a) else np.zeros((0, 2), np.int64)
        cells, _, count = _host_labels(local[g])
        keys[g] = _lowest_owned_cells(g, source, (cells[:n], None, count), numbered)
        sizes[g] = owned.astype(np.int64)
    number = _joined_regions(blocks, counts, edges, keys, sizes, least)
    out = []
    for g, source in zip(blocks.local, data):
        n = ghosted.owned[g]
        cells, points, count = _host_labels(local[g])
        with source.device:
            out.append(_renumbered_piece(source, (cells[:n], points[:n], count), number[g],
                                         local[g][0].dtype))
    return PartitionedDataSet(out, world=data.world)


def _host_labels(labels):
    cell_region, point_region, count = labels
    return cell_region.numpy().astype(np.int64), point_region.numpy().astype(np.int64), count


def _starts(blocks, counts):
    """Every block's first global node: the local regions' counts scanned."""
    every = blocks.all_gather(counts)
    return np.concatenate(([0], np.cumsum(every, dtype=np.int64))).astype(np.int64)


def _lowest_owned_cells(g, piece, labels, numbered):
    """Each local region's key: its lowest owned cell, by global id or (piece, cell);
    the largest i64 for a region with no owned cell."""
    from tack.dataset._matching import INDEX_BITS

    cell_region, _, count = labels
    regions = cell_region if isinstance(cell_region, np.ndarray) else cell_region.numpy()
    regions = regions.astype(np.int64)
    owned = regions >= 0
    if piece.ghosts is not None and piece.ghosts.cells is not None:
        owned &= (piece.ghosts.cells.numpy()[:len(regions)] & Ghosts.DUPLICATECELL) == 0
    if numbered and piece.global_ids.cells is not None:
        key = piece.global_ids.cells.numpy().astype(np.int64)[:len(regions)]
    else:
        key = (np.int64(g) << INDEX_BITS) | np.arange(len(regions), dtype=np.int64)
    lowest = np.full(count, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(lowest, regions[owned], key[owned])
    return lowest


def _owned_sizes(point_region, owned, count):
    regions = point_region[owned]
    return np.bincount(regions[regions >= 0], minlength=count).astype(np.int64)


def _joined_regions(blocks, counts, edges, keys, sizes=None, least=None):
    """Each local region's global number (-1 if dropped), ``{g: array}``.

    ``counts`` is each block's local region count, ``edges`` its pairs of global
    nodes (a block's first node is its regions' count scanned) that are one
    region, ``keys`` each local region's key, ``sizes`` its owned points. Regions
    no edge reaches stay as they are, on their block; the rest -- the edges and
    their nodes' keys and sizes, which only pieces' boundaries make -- are
    gathered and joined on every rank, each component keyed by its lowest key,
    sized by its sum. Regions of fewer than ``least`` points are dropped, and
    the rest numbered in order of key (ties by node) by a sort over the blocks,
    so no rank holds every region."""
    from tack import communication as tc

    starts = _starts(blocks, counts)
    every = blocks.all_gather({g: edges[g] for g in blocks.local})
    pairs = np.concatenate(every).reshape(-1, 2) if every else np.zeros((0, 2), np.int64)
    nodes = np.unique(pairs)
    mine = {}
    for g in blocks.local:
        here = nodes[(nodes >= starts[g]) & (nodes < starts[g + 1])] - starts[g]
        mine[g] = (keys[g][here], sizes[g][here] if sizes is not None
                   else np.zeros(len(here), np.int64))
    gathered = blocks.all_gather(mine)
    node_key = np.concatenate([k for k, _ in gathered]) if gathered else np.zeros(0, np.int64)
    node_size = np.concatenate([s for _, s in gathered]) if gathered else np.zeros(0, np.int64)

    # Join on the host: nodes are sorted, so the root of a component, its
    # smallest index, is its smallest node.
    parent = np.arange(len(nodes), dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in np.searchsorted(nodes, pairs).tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(x) for x in range(len(nodes))], np.int64)
    component_key = np.full(len(nodes), np.iinfo(np.int64).max, np.int64)
    np.minimum.at(component_key, roots, node_key)
    component_size = np.zeros(len(nodes), np.int64)
    np.add.at(component_size, roots, node_size)

    # Every surviving region once: a region no edge reaches on its block, a
    # component on the block of its root.
    entries, in_graph = {}, {}
    for g in blocks.local:
        node = starts[g] + np.arange(counts[g], dtype=np.int64)
        at = np.searchsorted(nodes, node)
        hit = np.zeros(counts[g], bool)
        valid = at < len(nodes)
        hit[valid] = nodes[at[valid]] == node[valid]
        in_graph[g] = (hit, at)
        key, size = keys[g].copy(), (sizes[g].copy() if sizes is not None else None)
        root = np.zeros(counts[g], bool)
        root[hit] = roots[at[hit]] == at[hit]
        key[hit] = component_key[at[hit]]
        if size is not None:
            size[hit] = component_size[at[hit]]
        alive = ~hit | root
        if least is not None:
            alive &= size >= least
        entries[g] = (key[alive], node[alive])
    device = blocks.device
    sorted_keys, sorted_values = tc.sort_by_key(
        blocks, {g: tack.array(entries[g][0], tack.i64, device=device(g)) for g in blocks.local},
        {g: {"node": tack.array(entries[g][1], tack.i64, device=device(g))}
         for g in blocks.local})
    first, _ = blocks.exclusive_scan({g: int(sorted_keys[g].size) for g in blocks.local})
    outgoing = {}
    for g in blocks.local:
        node = sorted_values[g]["node"].numpy()
        rank = first[g] + np.arange(len(node), dtype=np.int64)
        home = np.searchsorted(starts, node, side="right") - 1
        outgoing[g] = {int(b): {"node": tack.array(node[home == b], tack.i64, device=device(g)),
                                "number": tack.array(rank[home == b], tack.i64,
                                                     device=device(g))}
                       for b in np.unique(home).tolist()}
    incoming = blocks.exchange(outgoing)
    numbers, root_numbers = {}, {}
    for g in blocks.local:
        number = np.full(counts[g], -1, np.int64)
        for message in incoming[g].values():
            number[message["node"].numpy() - starts[g]] = message["number"].numpy()
        numbers[g] = number
        hit, at = in_graph[g]
        is_root = np.zeros(counts[g], bool)
        is_root[hit] = roots[at[hit]] == at[hit]
        root_numbers[g] = (at[is_root], number[is_root])
    every_root = blocks.all_gather(root_numbers)
    by_root = np.full(len(nodes), -1, np.int64)
    for at, number in every_root:
        by_root[at] = number
    for g in blocks.local:
        hit, at = in_graph[g]
        numbers[g][hit] = by_root[roots[at[hit]]]
    return numbers


def _renumbered_piece(piece, labels, mapping, dtype=None):
    """``piece`` with its local regions ``labels`` numbered by ``mapping``."""
    cell_region, point_region, _ = labels
    dtype = dtype or cell_region.dtype
    cells = cell_region if isinstance(cell_region, np.ndarray) else cell_region.numpy()
    points = point_region if isinstance(point_region, np.ndarray) else point_region.numpy()
    with piece.device:
        cells = np.where(cells >= 0, mapping[np.maximum(cells, 0)] if len(mapping) else -1, -1)
        points = np.where(points >= 0, mapping[np.maximum(points, 0)] if len(mapping) else -1,
                          -1)
        return _with_regions(piece, tack.array(cells.astype(dtype.numpy_dtype), dtype),
                             tack.array(points.astype(dtype.numpy_dtype), dtype))
