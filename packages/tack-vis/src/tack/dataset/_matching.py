"""Finding the copies of points and cells across the pieces of a partitioned dataset.

Copies are found by exact value -- a point's coordinates, a cell's points' global
ids -- through ``tack.communication.sort_by_key``: each entry is keyed by a hash
of its value, the keys are sorted over every block, and entries of one value,
which land on one block, are matched there. Each piece then hears, for each of
its entries, which other pieces hold a copy, at which index, and whether that
copy is marked a duplicate. ``ghost_cells`` and ``generate_global_ids`` share
this; the matching runs on the host, the exchanges on the pieces' devices.
"""

import numpy as np

import tack
from tack import communication as tc

# A copy's identity, (owner piece, index there), as one integer.
INDEX_BITS = 40
INDEX_MASK = (1 << INDEX_BITS) - 1


def mix(x):
    """splitmix64's finalizer, on u64 arrays."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def hashed(columns, seed=0):
    """A 64-bit hash of each row of ``columns`` (u64 arrays of one length)."""
    key = np.full(len(columns[0]) if columns else 0, np.uint64(seed), np.uint64)
    with np.errstate(over="ignore"):
        for column in columns:
            key = mix(key ^ mix(column.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)))
    return key


def position_bits(positions):
    """Each position's coordinates as f64 bits, three u64 columns (-0 as 0)."""
    exact = np.ascontiguousarray(np.asarray(positions, np.float64).reshape(-1, 3) + 0.0)
    bits = exact.view(np.uint64)
    return [bits[:, axis].copy() for axis in range(3)]


def match(blocks, entries):
    """Copies by exact value. ``entries`` is ``{gid: (index, columns, flags)}`` for
    every block on this rank: entry indices (i64), their values as u64 columns of
    one count everywhere, and a u8 flag each. Returns ``{gid: (index, other piece,
    other index, other flag)}``, a row for every copy another piece holds of each
    entry, as host arrays."""
    keys, values = {}, {}
    widths = set()
    for g in blocks.local:
        index, columns, flags = entries[g]
        widths.add(len(columns))
        device = blocks.device(g)
        keys[g] = tack.array(hashed(columns), tack.u64, device=device)
        values[g] = {"piece": tack.array(np.full(len(index), g, np.int64), tack.i64,
                                         device=device),
                     "index": tack.array(np.asarray(index, np.int64), tack.i64, device=device),
                     "flag": tack.array(np.asarray(flags, np.uint8), tack.u8, device=device)}
        for c, column in enumerate(columns):
            values[g][f"c{c}"] = tack.array(np.asarray(column, np.uint64), tack.u64,
                                            device=device)
    width = max(widths) if widths else 0
    sorted_keys, sorted_values = tc.sort_by_key(blocks, keys, values)

    found = {g: _pairs(sorted_keys[g].numpy(),
                       [sorted_values[g][f"c{c}"].numpy() for c in range(width)],
                       *(sorted_values[g][n].numpy() for n in ("piece", "index", "flag")))
             for g in blocks.local}
    return _deliver(blocks, found)


def _deliver(blocks, found):
    """Each block's rows ``(piece, index, other piece, other index, other flag)``
    handed to their pieces: ``{gid: (index, other, other index, other flag)}``."""
    if _all_here(blocks):
        return _partners_here(blocks, found)
    outgoing = {}
    for g in blocks.local:
        piece, *columns = found[g]
        order, starts = _runs_by_piece(piece)
        device = blocks.device(g)
        outgoing[g] = {}
        for at in range(len(starts) - 1):
            rows = order[starts[at]:starts[at + 1]]
            message = {name: tack.array(column[rows], tack.i64, device=device)
                       for name, column in zip(("index", "other", "other_index"), columns)}
            message["other_flag"] = tack.array(columns[3][rows], tack.u8, device=device)
            outgoing[g][int(piece[rows[0]])] = message
    incoming = blocks.exchange(outgoing)
    partners = {}
    for g in blocks.local:
        parts = list(incoming[g].values())
        partners[g] = tuple(
            np.concatenate([part[name].numpy() for part in parts]) if parts
            else np.zeros(0, np.uint8 if name == "other_flag" else np.int64)
            for name in ("index", "other", "other_index", "other_flag"))
    return partners


def match_positions(blocks, entries, tolerance=0.0):
    """Copies of points by position: ``entries`` is ``{gid: (index, positions,
    flags)}`` (positions an ``(n, 3)`` host array relative to the pieces' common
    origin). At zero ``tolerance``, coordinates equal bit for bit (``match``);
    otherwise every two points of different pieces at most ``tolerance`` apart
    are copies, and the owners of chains of them need ``settled``. Returns
    ``match``'s rows."""
    if not tolerance:
        return match(blocks, {g: (index, position_bits(positions), flags)
                              for g, (index, positions, flags) in entries.items()})
    return _match_near(blocks, entries, float(tolerance))


def _match_near(blocks, entries, tolerance):
    """Points of different pieces at most ``tolerance`` apart, through cells of
    a lattice a little over two tolerances wide: each point goes to its own cell
    (its home) and to the others its box of plus or minus the tolerance
    reaches, at most eight, all by ``sort_by_key``. Two points that close meet
    in each one's home, where the other comes to it, so each is told of the
    other once."""
    width = 2.0 * tolerance * (1.0 + 1e-6)      # a box (2 tolerances, a hair more)
    reach = tolerance * (1.0 + 1e-9)            # spans at most two cells an axis
    keys, values = {}, {}
    for g in blocks.local:
        index, positions, flags = entries[g]
        index = np.asarray(index, np.int64)
        x = np.asarray(positions, np.float64).reshape(-1, 3) + 0.0
        home = np.floor(x / width).astype(np.int64)
        lo = np.floor((x - reach) / width).astype(np.int64)
        hi = np.floor((x + reach) / width).astype(np.int64)
        cells, rows = [], []
        for corner in np.ndindex(2, 2, 2):
            cell = lo + np.array(corner, np.int64)
            reached = np.flatnonzero((cell <= hi).all(axis=1))
            cells.append(cell[reached])
            rows.append(reached)
        cells, rows = np.concatenate(cells), np.concatenate(rows)
        columns = [cells[:, axis].view(np.uint64) for axis in range(3)]
        device = blocks.device(g)
        keys[g] = tack.array(hashed(columns), tack.u64, device=device)
        bits = x[rows].view(np.uint64)
        values[g] = {"piece": tack.array(np.full(len(rows), g, np.int64), tack.i64,
                                         device=device),
                     "index": tack.array(index[rows], tack.i64, device=device),
                     "flag": tack.array(np.asarray(flags, np.uint8)[rows], tack.u8,
                                        device=device),
                     "home": tack.array((cells == home[rows]).all(axis=1).astype(np.uint8),
                                        tack.u8, device=device)}
        for axis in range(3):
            values[g][f"c{axis}"] = tack.array(columns[axis].copy(), tack.u64, device=device)
            values[g][f"x{axis}"] = tack.array(bits[:, axis].copy(), tack.u64, device=device)
    sorted_keys, sorted_values = tc.sort_by_key(blocks, keys, values)
    found = {}
    for g in blocks.local:
        v = {name: column.numpy() for name, column in sorted_values[g].items()}
        found[g] = _near_pairs(sorted_keys[g].numpy(), [v[f"c{a}"] for a in range(3)],
                               np.stack([v[f"x{a}"] for a in range(3)], axis=1)
                               .view(np.float64), v["piece"], v["index"], v["flag"],
                               v["home"], tolerance)
    return _deliver(blocks, found)


def _near_pairs(keys, cells, x, piece, index, flag, home, tolerance):
    """``_pairs``' rows for every two entries of one cell on different pieces at
    most ``tolerance`` apart, the second at home there."""
    if not len(keys):
        return tuple(np.zeros(0, np.uint8 if k == 4 else np.int64) for k in range(5))
    change = np.r_[True, keys[1:] != keys[:-1]]
    if not all(np.array_equal(c[1:][~change[1:]], c[:-1][~change[1:]]) for c in cells):
        order = np.lexsort([*cells[::-1], keys])
        keys, cells, x = keys[order], [c[order] for c in cells], x[order]
        piece, index, flag, home = piece[order], index[order], flag[order], home[order]
        change = np.r_[True, keys[1:] != keys[:-1]]
        for c in cells:
            change[1:] |= c[1:] != c[:-1]
    group = np.cumsum(change) - 1
    sizes = np.bincount(group)
    starts = np.concatenate(([0], np.cumsum(sizes)))[group]
    size = sizes[group]
    first = np.repeat(np.arange(len(group)), size)
    within = np.arange(size.sum()) - np.repeat(np.cumsum(size) - size, size)
    second = np.repeat(starts, size) + within
    keep = (piece[first] != piece[second]) & (home[second] == 1)
    first, second = first[keep], second[keep]
    d = x[first] - x[second]
    near = np.einsum("ij,ij->i", d, d) <= tolerance * tolerance
    first, second = first[near], second[near]
    return piece[first], index[first], piece[second], index[second], flag[second]


def settled(blocks, identities, partners):
    """Each entry's identity, ``owners``' smallest copy, made the smallest over
    chains of copies (``a`` near ``b``, ``b`` near ``c``): partners swap
    identities until none changes, which near copies need and exact ones do not
    (every exact copy is every other's partner). ``identities`` is ``{gid:
    owners(...)}``, updated in place and returned."""
    while True:
        outgoing = {}
        for g in blocks.local:
            index, other, other_index, _ = partners[g]
            order, starts = _runs_by_piece(other)
            device = blocks.device(g)
            outgoing[g] = {}
            for at in range(len(starts) - 1):
                rows = order[starts[at]:starts[at + 1]]
                outgoing[g][int(other[rows[0]])] = {
                    "index": tack.array(other_index[rows], tack.i64, device=device),
                    "identity": tack.array(identities[g][index[rows]], tack.i64,
                                           device=device)}
        incoming = blocks.exchange(outgoing)
        changed = {}
        for g in blocks.local:
            before = identities[g].copy()
            for message in incoming[g].values():
                np.minimum.at(identities[g], message["index"].numpy(),
                              message["identity"].numpy())
            changed[g] = int(not np.array_equal(before, identities[g]))
        if not blocks.all_reduce(changed, "max"):
            return identities


def _runs_by_piece(piece):
    """Rows in order of piece (stable) and where each piece's run starts."""
    order = np.argsort(piece, kind="stable")
    ordered = piece[order]
    if not len(piece):
        return order, np.zeros(1, np.int64)
    return order, np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])


def _all_here(blocks):
    """Whether every block is on this rank."""
    return len(blocks.local) == blocks.count


def _partners_here(blocks, found):
    """``match``'s answer when every block is on this rank: the rows handed to their
    pieces on the host, in the order the messages would bring them (by sending
    block, then row), without a message from every block to every piece."""
    gids = list(blocks.local)
    rows = [np.concatenate([found[g][k] for g in gids]) for k in range(5)]
    order, starts = _runs_by_piece(rows[0])
    rows = [column[order] for column in rows]
    partners = {g: (np.zeros(0, np.int64),) * 3 + (np.zeros(0, np.uint8),) for g in gids}
    for at in range(len(starts) - 1):
        run = slice(starts[at], starts[at + 1])
        partners[int(rows[0][starts[at]])] = tuple(column[run] for column in rows[1:])
    return partners


def _pairs(keys, columns, piece, index, flag):
    """Rows ``(piece, index, other piece, other index, other flag)`` for every two
    entries of one value on different pieces, from sorted keys and their entries."""
    if not len(keys):
        return tuple(np.zeros(0, np.uint8 if k == 4 else np.int64) for k in range(5))
    # Rows grouped by value, groups in key order, rows of a group in their order.
    # The keys come sorted, so runs of a key are the groups unless two values
    # hash alike; only then are the rows sorted by value (np.lexsort is stable).
    # (np.unique(axis=0) sorted rows as void scalars: 7 of ghost cells' 18
    # seconds over 128 pieces.)
    change = np.r_[True, keys[1:] != keys[:-1]]
    if not all(np.array_equal(c[1:][~change[1:]], c[:-1][~change[1:]]) for c in columns):
        order = np.lexsort([*columns[::-1], keys])
        keys, columns = keys[order], [c[order] for c in columns]
        piece, index, flag = piece[order], index[order], flag[order]
        change = np.r_[True, keys[1:] != keys[:-1]]
        for c in columns:
            change[1:] |= c[1:] != c[:-1]
    group = np.cumsum(change) - 1
    sizes = np.bincount(group)
    starts = np.concatenate(([0], np.cumsum(sizes)))[group]
    size = sizes[group]
    first = np.repeat(np.arange(len(group)), size)
    within = np.arange(size.sum()) - np.repeat(np.cumsum(size) - size, size)
    second = np.repeat(starts, size) + within
    keep = piece[first] != piece[second]
    first, second = first[keep], second[keep]
    return piece[first], index[first], piece[second], index[second], flag[second]


def owners(g, count, partners, duplicate):
    """Each entry's identity, ``(owner << INDEX_BITS) | index there``: the lowest
    piece holding a copy not marked a duplicate (``duplicate``, a bool per entry
    here; partners' flags for theirs). An entry marked a duplicate with no
    unmarked copy anywhere is refused."""
    if count >= 1 << INDEX_BITS:
        raise ValueError(f"pieces of fewer than 2**{INDEX_BITS} entries are handled")
    none = np.iinfo(np.int64).max
    identity = np.where(duplicate, none, (np.int64(g) << INDEX_BITS)
                        | np.arange(count, dtype=np.int64))
    index, other, other_index, other_flag = partners
    owned = other_flag == 0
    if owned.any():
        theirs = (other[owned].astype(np.int64) << INDEX_BITS) | other_index[owned]
        np.minimum.at(identity, index[owned], theirs)
    lost = np.flatnonzero(identity == none)
    if len(lost):
        raise ValueError(f"piece {g}: {len(lost)} entries are marked duplicates but no "
                         f"piece holds them unmarked")
    return identity


def plan(blocks, receives):
    """The plan giving each piece the values at ``receives``: ``{gid: {source:
    (source's indices, local indices)}}``. Each source hears which of its
    indices to send, in order."""
    outgoing = {g: {source: {"wanted": tack.array(np.asarray(theirs, np.int64), tack.i64,
                                                  device=blocks.device(g))}
                    for source, (theirs, _) in receives[g].items() if len(theirs)}
                for g in blocks.local}
    asked = blocks.exchange(outgoing)
    sends = {g: {asker: message["wanted"] for asker, message in asked[g].items()}
             for g in blocks.local}
    local = {g: {source: np.asarray(mine, np.int64) for source, (theirs, mine)
                 in receives[g].items() if len(theirs)} for g in blocks.local}
    return tc.Plan(blocks, sends, local)


def from_owners(g, identity, which):
    """``receives`` entries for ``plan``: each of ``which`` (local indices) from its
    owner, ``{owner: (owner's indices, local indices)}``."""
    which = np.asarray(which, np.int64)
    owner = identity[which] >> INDEX_BITS
    out = {}
    for source in np.unique(owner).tolist():
        mine = which[owner == source]
        if source != g:
            out[int(source)] = (identity[mine] & INDEX_MASK, mine)
    return out
