"""A sort over every block's keys: the global order cut into one range per block."""

import numpy as np

import tack
from tack.algorithms.sort import argsort
from tack.lang.array import Array

from ._kernels import gather, search_right, take

_KEY_TYPES = (tack.i32, tack.u32, tack.i64, tack.u64)


def sort_by_key(blocks, keys, values=None, samples=16):
    """Sort the keys of every block, carrying ``values`` along: block 0 ends with the
    smallest keys, the last block with the largest, each block's ascending.

    ``keys`` is ``{gid: array}`` and ``values`` ``{gid: {name: array}}``, for every
    block on this rank: one-dimensional ``i32``, ``u32``, ``i64`` or ``u64`` keys of
    one type on every block, and values of the keys' length, each name of one
    element type everywhere, all on the block's device. Returns ``(keys,
    values)`` in the same form (``values`` None if not given), on the same
    devices.

    Every copy of a key ends on one block, so equal keys can be grouped where
    they land; heavily repeated keys can leave blocks uneven. The sort is
    stable: equal keys keep the order of their blocks, then their order within
    a block. A sample sort, as DIY's: each block sorts its own keys and sends
    ``samples`` of them, weighted by its count, from which every block finds the
    same ``count - 1`` splitters; each block sends each range to its block, and
    sorts what arrives.
    """
    blocks._every_local(keys, "sort_by_key()")
    given = values is not None
    values = {gid: {} for gid in blocks.local} if values is None else values
    blocks._every_local(values, "sort_by_key()'s values")
    for gid in blocks.local:
        _check(blocks, gid, keys[gid], values[gid])
    kinds = blocks.all_gather({gid: (keys[gid].dtype, {name: v.dtype for name, v in
                                                       values[gid].items()})
                               for gid in blocks.local})
    if len({repr(kind) for kind in kinds}) > 1:
        raise TypeError(f"sort_by_key() needs one key type and the same values, of one "
                        f"type each, on every block, not {sorted({repr(k) for k in kinds})}")

    ordered, carried, picks = {}, {}, {}
    for gid in blocks.local:
        perm = argsort(keys[gid])
        ordered[gid] = gather(keys[gid], perm)
        carried[gid] = {name: gather(v, perm) for name, v in values[gid].items()}
        n = ordered[gid].size
        m = min(samples, n)
        at = ((np.arange(m) + 0.5) * n / max(m, 1)).astype(np.int64)
        chosen = gather(ordered[gid], tack.array(at, tack.i64, device=blocks.device(gid)))
        picks[gid] = (chosen.numpy(), n)
    splitters = _splitters(blocks.all_gather(picks), blocks.count)

    outgoing = {}
    for gid in blocks.local:
        n = ordered[gid].size
        bounds = [0, *search_right(ordered[gid], splitters).tolist(), n]
        outgoing[gid] = {}
        for destination in range(blocks.count):
            start, count = bounds[destination], bounds[destination + 1] - bounds[destination]
            if count:
                message = {"keys": take(ordered[gid], start, count)}
                for name, v in carried[gid].items():
                    message["values." + name] = take(v, start, count)
                outgoing[gid][destination] = message
    incoming = blocks.exchange(outgoing)

    sorted_keys, sorted_values = {}, {}
    for gid in blocks.local:
        runs = list(incoming[gid].values())             # in order of source block
        device = blocks.device(gid)
        key_type, value_types = kinds[gid]
        joined = _joined([run["keys"] for run in runs], key_type, device)
        perm = argsort(joined)
        sorted_keys[gid] = gather(joined, perm)
        sorted_values[gid] = {
            name: gather(_joined([run["values." + name] for run in runs], dtype, device), perm)
            for name, dtype in value_types.items()}
    return sorted_keys, (sorted_values if given else None)


def _check(blocks, gid, keys, values):
    device = blocks._backend(gid)
    if not isinstance(keys, Array) or keys.dtype not in _KEY_TYPES or len(keys.shape) != 1:
        raise TypeError(f"sort_by_key() sorts one-dimensional i32, u32, i64 or u64 keys; "
                        f"block {gid}'s are {getattr(keys, 'dtype', type(keys).__name__)}")
    if keys._backend.memory_domain != device.memory_domain:
        raise ValueError(f"block {gid}'s keys are on {keys.device}, not on the block's "
                         f"device {blocks.device(gid)}")
    for name, v in values.items():
        if not isinstance(v, Array) or v.shape[:1] != keys.shape or len(v.shape) != 1:
            raise ValueError(f"block {gid}'s {name!r} needs one element per key "
                             f"({keys.size}), as a one-dimensional Tack array")
        if v._backend.memory_domain != device.memory_domain:
            raise ValueError(f"block {gid}'s {name!r} is on {v.device}, not on the "
                             f"block's device {blocks.device(gid)}")


def _splitters(picks, count):
    """``count - 1`` keys cutting the weighted samples into equal shares: every
    block's sample stands for its count over its samples."""
    total = sum(n for _, n in picks)
    values = np.concatenate([chosen for chosen, _ in picks]) if picks else np.zeros(0)
    if not total:                   # no keys anywhere: any splitters cut nothing
        return np.zeros(max(count - 1, 0), np.int64)
    weights = np.concatenate([np.full(len(chosen), n / len(chosen)) for chosen, n in picks
                              if len(chosen)])
    order = np.argsort(values, kind="stable")
    values, cumulative = values[order], np.cumsum(weights[order])
    targets = np.arange(1, count) * (total / count)
    at = np.minimum(np.searchsorted(cumulative, targets, side="left"), len(values) - 1)
    return values[at]


def _joined(arrays, dtype, device):
    if not arrays:
        return tack.empty(0, dtype, device=device)
    return arrays[0] if len(arrays) == 1 else tack.concat(arrays)
