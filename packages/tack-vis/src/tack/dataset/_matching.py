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

    outgoing = {}
    for g in blocks.local:
        found = _pairs(sorted_keys[g].numpy(),
                       [sorted_values[g][f"c{c}"].numpy() for c in range(width)],
                       *(sorted_values[g][n].numpy() for n in ("piece", "index", "flag")))
        device = blocks.device(g)
        outgoing[g] = {}
        for piece in np.unique(found[0]).tolist():
            rows = found[0] == piece
            message = {name: tack.array(column[rows], tack.i64, device=device)
                       for name, column in zip(("index", "other", "other_index"), found[1:4])}
            message["other_flag"] = tack.array(found[4][rows], tack.u8, device=device)
            outgoing[g][int(piece)] = message
    incoming = blocks.exchange(outgoing)
    partners = {}
    for g in blocks.local:
        parts = list(incoming[g].values())
        partners[g] = tuple(
            np.concatenate([part[name].numpy() for part in parts]) if parts
            else np.zeros(0, np.uint8 if name == "other_flag" else np.int64)
            for name in ("index", "other", "other_index", "other_flag"))
    return partners


def _pairs(keys, columns, piece, index, flag):
    """Rows ``(piece, index, other piece, other index, other flag)`` for every two
    entries of one value on different pieces, from sorted keys and their entries."""
    if not len(keys):
        return tuple(np.zeros(0, np.uint8 if k == 4 else np.int64) for k in range(5))
    table = np.stack([keys, *columns], axis=1).view(np.int64)
    _, group, sizes = np.unique(table, axis=0, return_inverse=True, return_counts=True)
    group = group.reshape(-1)
    order = np.argsort(group, kind="stable")
    group, piece, index, flag = group[order], piece[order], index[order], flag[order]
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
