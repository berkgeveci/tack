"""``merge_pieces``: a partitioned dataset's pieces as one dataset, each cell and point
once, for writing one file or handing to code that knows no pieces."""

import numpy as np

import tack
from tack import arrays
from tack import communication as tc
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.dataset.dataset import DataSet, Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.global_ids import GlobalIds
from tack.dataset.partitioned import PartitionedDataSet, common_origin
from tack.dataset.spaces import H1, Constant, Values
from tack.dataset.topology import StructuredTopology, UnstructuredTopology

__all__ = ["merge_pieces"]

def merge_pieces(data, root=0, fields=None, device=None):
    """``data``'s pieces as one dataset: every cell a piece owns, once, and every point
    once, with the point and cell fields (``fields`` choosing them as filters
    do), as VTK's vtkMergeBlocks after vtkRemoveGhosts.

    Copies of a point several pieces hold become one point, matched as
    ``generate_global_ids`` matches them (the ghost marks' owners, else exact
    positions); ghost cells are left out, and hidden cells kept, marked. The
    points and cells are numbered as ``generate_global_ids`` numbers them:
    each piece's owned ones after the pieces before it. The pieces' own global
    ids, if they have them, come along as the result's.

    Structured pieces that tile a box of cells (each placed by its
    ``topology.offset``) give one structured grid of the box -- ``split``'s
    input back -- with its offset and its coordinates (uniform, rectilinear
    or explicit); other pieces give an unstructured dataset.

    Across MPI ranks every rank passes its pieces and the result is on the
    ``root`` rank (``None`` elsewhere), on ``device`` (by default the device
    of the root rank's first piece)."""
    if not isinstance(data, PartitionedDataSet):
        raise TypeError(f"merge_pieces takes a PartitionedDataSet, not {type(data).__name__}")
    from tack.dataset.algorithms.ghost_cells import _carried, _kind

    pieces = list(data)
    names = _carried(pieces, fields)
    common_origin(data, "merge_pieces")
    blocks = tc.Blocks([piece.device for piece in pieces], data.world)
    world = blocks.world
    if not 0 <= root < world.size:
        raise ValueError(f"root is a rank, 0 to {world.size - 1}, not {root}")
    firsts = blocks.all_gather({g: g == blocks.local.start for g in blocks.local})
    starts = [g for g, first in enumerate(firsts) if first]
    target = next((g for g in starts if blocks.rank_of(g) == root), None)
    if target is None:
        raise ValueError(f"merge_pieces gathers onto a piece of rank {root}, which has none")
    on_root = target in blocks.local
    out_device = (tack.device(device) if device is not None else
                  blocks.device(target)) if on_root else None
    structured = blocks.all_gather({g: isinstance(p.topology, StructuredTopology)
                                    for g, p in zip(blocks.local, pieces)})
    if all(structured):
        tiled = _box(blocks, pieces)
        if tiled is not None:
            return _merged_structured(blocks, pieces, names, _kind, target, out_device,
                                      tiled)
    return _merged_unstructured(blocks, pieces, names, _kind, target, out_device)


def _box(blocks, pieces):
    """The box structured pieces tile, ``(lower, upper, every)``, or ``None``."""
    every = blocks.all_gather({g: tuple(p.topology.offset)
                               for g, p in zip(blocks.local, pieces)})
    if len({len(o) for o in every}) != 1:
        return None
    lower = np.array(every, np.int64)
    # Ghost layers overlap; what tiles is each piece's owned cells, which for
    # a structured piece with layers is its block less them. Recover the
    # owned blocks from the marks where there are any.
    own = blocks.all_gather({g: _owned_block(p) for g, p in zip(blocks.local, pieces)})
    if any(o is None for o in own):
        return None
    own_lower = np.array([o[0] for o in own], np.int64)
    own_upper = np.array([o[1] for o in own], np.int64)
    box_lower, box_upper = own_lower.min(axis=0), own_upper.max(axis=0)
    total = int(np.prod(own_upper - own_lower, axis=1).sum())
    if total != int(np.prod(box_upper - box_lower)):
        return None
    for a in range(len(own_lower)):
        for b in range(a + 1, len(own_lower)):
            if np.all(np.maximum(own_lower[a], own_lower[b]) <
                      np.minimum(own_upper[a], own_upper[b])):
                return None
    return box_lower, box_upper, own_lower, own_upper, lower


def _owned_block(piece):
    """The global index box of a structured piece's owned cells, ``(lower, upper)``,
    or ``None`` if they are not a box."""
    topology = piece.topology
    offset = np.array(topology.offset, np.int64)
    cells = np.array([d - 1 for d in topology.point_dims], np.int64)
    marks = piece.ghosts
    if marks is None or marks.cells is None:
        return offset, offset + cells
    owned = (marks.cells.numpy() & Ghosts.DUPLICATECELL) == 0
    if not owned.any():
        return offset, offset          # an empty block
    index = np.flatnonzero(owned)
    local = []
    rest = index.copy()
    for extent in cells:
        local.append(rest % extent)
        rest //= extent
    local = np.stack(local, axis=1)
    lo, hi = local.min(axis=0), local.max(axis=0) + 1
    if int(np.prod(hi - lo)) != len(index):
        return None
    return offset + lo, offset + hi


def _flat(local, extent):
    index = np.zeros(len(local), np.int64)
    for axis in reversed(range(local.shape[1])):
        index = index * extent[axis] + local[:, axis]
    return index


def _grid(lo, hi):
    axes = [np.arange(a, b) for a, b in zip(lo, hi)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel(order="F") for m in mesh], axis=1).astype(np.int64)


def _merged_structured(blocks, pieces, names, kind_of, target, device, tiled):
    """Structured pieces tiling a box: the box's grid, every value at its index."""
    from tack.dataset.algorithms._structured_ghosts import _axes, _grown_geometry

    box_lower, box_upper, own_lower, own_upper, lower = tiled
    point_extent = box_upper - box_lower + 1
    cell_extent = box_upper - box_lower
    outgoing = {}
    for g, piece in zip(blocks.local, pieces):
        dims = np.array(piece.topology.point_dims, np.int64)
        offset = np.array(piece.topology.offset, np.int64)
        # The points it gives: its owned block's, the lowest piece's where two
        # share them, as the marks say.
        cells_here = _grid(own_lower[g], own_upper[g])
        points_here = _grid(own_lower[g], own_upper[g] + 1)
        cell_local = _flat(cells_here - offset, dims - 1)
        point_local = _flat(points_here - offset, dims)
        if piece.ghosts is not None and piece.ghosts.points is not None:
            keep = (piece.ghosts.points.numpy()[point_local] & Ghosts.DUPLICATEPOINT) == 0
            points_here, point_local = points_here[keep], point_local[keep]
        else:
            # Without marks, a point on a shared face goes to the lowest piece.
            mine = np.ones(len(points_here), bool)
            for d in range(g):
                inside = np.all((points_here >= own_lower[d]) & (points_here <= own_upper[d]),
                                axis=1)
                mine &= ~inside
            points_here, point_local = points_here[mine], point_local[mine]
        dev = blocks.device(g)
        message = {"cell_at": tack.array(_flat(cells_here - box_lower, cell_extent), tack.i64,
                                         device=dev),
                   "point_at": tack.array(_flat(points_here - box_lower, point_extent),
                                          tack.i64, device=dev)}
        _add_values(message, piece, names, kind_of, point_local, cell_local, dev,
                    explicit=not isinstance(piece.geometry.values,
                                            (UniformCoordinates, CartesianProduct)))
        outgoing[g] = {target: message}
    rectilinear = _axes(blocks, pieces, lower)
    incoming = blocks.exchange(outgoing)
    if target not in blocks.local:
        return None
    first = pieces[target - blocks.local.start]
    np_, nc = int(np.prod(point_extent)), int(np.prod(cell_extent))
    with device:
        topology = StructuredTopology(tuple(int(d) for d in point_extent),
                                      id_dtype=first.topology.id_dtype,
                                      offset=tuple(int(v) for v in box_lower))
        parts = _placed(incoming[target], np_, nc)
        if "\0positions" in parts["points"]:
            geometry = parts["points"]["\0positions"]
        else:
            geometry = _grown_geometry(first, rectilinear, box_lower, topology,
                                       np.array(first.topology.offset, np.int64), device)
        return _assembled(topology, geometry, first, names, kind_of, parts)


def _merged_unstructured(blocks, pieces, names, kind_of, target, device):
    """Any pieces: owned cells and points, numbered by ``generate_global_ids``."""
    from tack.dataset.algorithms.extract_cells import extract_cells
    from tack.dataset.algorithms.generate_global_ids import generate_global_ids

    converted = []
    for piece in pieces:
        if isinstance(piece.topology, StructuredTopology):
            # As an unstructured mesh of every cell: extract_cells leaves hidden
            # cells out, so the marks go aside and come back (every cell and
            # point is kept, in order).
            bare = DataSet(piece.topology, piece.geometry, fields=dict(piece.fields),
                           dtype=piece.dtype, global_ids=piece.global_ids, **piece._frame())
            with piece.device:
                converted_piece = extract_cells(bare, np.arange(piece.num_cells))
            converted_piece.ghosts = piece.ghosts
            piece = converted_piece
        converted.append(piece)
    numbered = list(generate_global_ids(PartitionedDataSet(converted, world=blocks.world)))
    outgoing = {}
    for g, piece, original in zip(blocks.local, numbered, converted):
        if not isinstance(piece.topology, UnstructuredTopology):
            raise TypeError(f"merge_pieces takes unstructured or structured pieces; "
                            f"partition {g - blocks.local.start} is "
                            f"{type(piece.topology).__name__}")
        point_ids = piece.global_ids.points.numpy()
        cell_ids = piece.global_ids.cells.numpy()
        cell_marks = (np.zeros(piece.num_cells, np.uint8) if piece.ghosts is None
                      or piece.ghosts.cells is None else piece.ghosts.cells.numpy())
        point_marks = (np.zeros(piece.num_points, np.uint8) if piece.ghosts is None
                       or piece.ghosts.points is None else piece.ghosts.points.numpy())
        owned_cells = np.flatnonzero((cell_marks & Ghosts.DUPLICATECELL) == 0)
        owned_points = np.flatnonzero((point_marks & Ghosts.DUPLICATEPOINT) == 0)
        offsets = np.asarray(arrays.to_host(piece.topology.offsets), np.int64).reshape(-1)
        conn = np.asarray(arrays.to_host(piece.topology.connectivity), np.int64).reshape(-1)
        types = np.asarray(arrays.to_host(piece.topology.types), np.uint8).reshape(-1)
        sizes = np.diff(offsets)[owned_cells]
        rows = (np.concatenate([conn[offsets[c]:offsets[c + 1]] for c in owned_cells])
                if len(owned_cells) else np.zeros(0, np.int64))
        dev = blocks.device(g)
        message = {"cell_at": tack.array(cell_ids[owned_cells], tack.i64, device=dev),
                   "point_at": tack.array(point_ids[owned_points], tack.i64, device=dev),
                   "types": tack.array(types[owned_cells], tack.u8, device=dev),
                   "sizes": tack.array(sizes.astype(np.int64), tack.i64, device=dev),
                   "connectivity": tack.array(point_ids[rows].astype(np.int64), tack.i64,
                                              device=dev)}
        _add_values(message, original, names, kind_of, owned_points, owned_cells, dev,
                    explicit=True)
        outgoing[g] = {target: message}
    incoming = blocks.exchange(outgoing)
    if target not in blocks.local:
        return None
    first = converted[target - blocks.local.start]
    messages = list(incoming[target].values())
    np_ = sum(m["point_at"].size for m in messages)
    nc = sum(m["cell_at"].size for m in messages)
    with device:
        order = np.argsort(np.concatenate([m["cell_at"].numpy() for m in messages]),
                           kind="stable") if nc else np.zeros(0, np.int64)
        types = np.concatenate([m["types"].numpy() for m in messages])[order] if nc \
            else np.zeros(0, np.uint8)
        sizes_in = [m["sizes"].numpy() for m in messages]
        conn_in = [m["connectivity"].numpy() for m in messages]
        sizes = np.concatenate(sizes_in) if nc else np.zeros(0, np.int64)
        starts = np.concatenate(([0], np.cumsum(sizes)))
        conn = np.concatenate(conn_in) if nc else np.zeros(0, np.int64)
        rows = [conn[starts[c]:starts[c + 1]] for c in order]
        offsets = np.concatenate(([0], np.cumsum(sizes[order])))
        connectivity = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        topology = UnstructuredTopology(types, offsets, connectivity, num_points=np_,
                                        id_dtype=first.topology.id_dtype)
        parts = _placed(messages, np_, nc)
        return _assembled(topology, parts["points"].pop("\0positions"), first, names,
                          kind_of, parts)


def _add_values(message, piece, names, kind_of, points, cells, device, explicit):
    """The values of a piece's chosen points and cells, into ``message``."""
    from tack.dataset.carry import _take

    point_ids = tack.array(np.asarray(points, np.int64), tack.i64, device=device)
    cell_ids = tack.array(np.asarray(cells, np.int64), tack.i64, device=device)
    for name in names:
        field = piece.fields[name]
        kind = kind_of(field)
        message[f"{kind}.{name}"] = _take(field.values, point_ids if kind == "points"
                                          else cell_ids)
    if explicit:
        message["points.\0positions"] = _take(piece.geometry.values, point_ids)
    marks = piece.ghosts
    message["cells.\0marks"] = _take(
        tack.zeros((piece.num_cells,), tack.u8, device=device)
        if marks is None or marks.cells is None else marks.cells, cell_ids)
    message["points.\0marks"] = _take(
        tack.zeros((piece.num_points,), tack.u8, device=device)
        if marks is None or marks.points is None else marks.points, point_ids)
    ids = piece.global_ids
    for kind, chosen in (("points", point_ids), ("cells", cell_ids)):
        if ids is not None and getattr(ids, kind) is not None:
            message[f"{kind}.\0ids"] = _take(getattr(ids, kind), chosen)


def _placed(messages, np_, nc):
    """The values every message brought, each at its index: ``{"points": {name:
    array}, "cells": {...}}`` on the current device."""
    from tack.communication._kernels import scatter

    messages = list(messages.values()) if isinstance(messages, dict) else list(messages)
    out = {"points": {}, "cells": {}}
    for kind, n, at in (("points", np_, "point_at"), ("cells", nc, "cell_at")):
        names = sorted({k[len(kind) + 1:] for m in messages for k in m
                        if k.startswith(kind + ".")})
        for name in names:
            sample = next(m[f"{kind}.{name}"] for m in messages if f"{kind}.{name}" in m)
            array = tack.zeros((n,), sample.dtype)
            for m in messages:
                if f"{kind}.{name}" in m and m[at].size:
                    scatter(m[f"{kind}.{name}"].to_device(array.device),
                            m[at].to_device(array.device), array)
            out[kind][name] = array
    return out


def _assembled(topology, geometry, like, names, kind_of, parts):
    marks_cells = parts["cells"].pop("\0marks").numpy() & ~np.uint8(Ghosts.DUPLICATECELL)
    marks_points = parts["points"].pop("\0marks").numpy() & ~np.uint8(Ghosts.DUPLICATEPOINT)
    ghosts = None
    if marks_cells.any() or marks_points.any():
        ghosts = Ghosts(cells=marks_cells if marks_cells.any() else None,
                        points=marks_points if marks_points.any() else None)
    out = DataSet(topology, geometry, dtype=like.dtype, ghosts=ghosts, **like._frame())
    for name in names:
        field = like.fields[name]
        space = field.space
        if kind_of(field) == "points":
            out.fields[name] = Field(H1(out) if isinstance(space, H1) else Values(out, "points"),
                                     parts["points"][name], field.unit)
        else:
            out.fields[name] = Field(Constant(out) if isinstance(space, Constant)
                                     else Values(out, "cells"), parts["cells"][name],
                                     field.unit)
    point_ids, cell_ids = parts["points"].get("\0ids"), parts["cells"].get("\0ids")
    if point_ids is not None or cell_ids is not None:
        out.global_ids = GlobalIds(cells=cell_ids, points=point_ids)
    return out
