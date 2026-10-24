"""A retained exchange: elements gathered on their blocks, sent, and scattered on others."""

import numpy as np

import tack
from tack.lang.array import Array

from ._kernels import gather, scatter


class Plan:
    """Which elements each block sends to which, and where the receiver puts them:
    an exchange kept for reuse, as ghost values are refreshed from their owners.

    ``sends`` is ``{gid: {destination: indices}}`` and ``receives``
    ``{gid: {source: indices}}``, for blocks on this rank of ``blocks``: the
    element indices a block reads for each destination, and those it writes from
    each source, in matching order. Indices are integer Tack arrays on the
    block's device, or anything ``numpy.asarray`` makes integers of. Every send
    must have a receive of its length on the other block, and the other way
    round; this is checked across ranks when the plan is made.
    """

    def __init__(self, blocks, sends, receives):
        self.blocks = blocks
        blocks._local_keys(sends, "Plan's sends")
        blocks._local_keys(receives, "Plan's receives")
        self._sends = _indices(blocks, sends, "sends", "destination")
        self._receives = _indices(blocks, receives, "receives", "source")
        counts = blocks.all_gather({gid: (
            {other: idx.size for other, idx in self._sends.get(gid, {}).items()},
            {other: idx.size for other, idx in self._receives.get(gid, {}).items()})
            for gid in blocks.local})
        for gid in blocks.local:
            sent, received = counts[gid]
            for other, count in sent.items():
                theirs = counts[other][1].get(gid, 0)
                if theirs != count:
                    raise ValueError(f"block {gid} sends {count} elements to block {other}, "
                                     f"which receives {theirs} from it")
            for other, count in received.items():
                theirs = counts[other][0].get(gid, 0)
                if theirs != count:
                    raise ValueError(f"block {gid} receives {count} elements from block "
                                     f"{other}, which sends {theirs} to it")

    def __repr__(self):
        pairs = sum(len(d) for d in self._sends.values())
        return f"Plan({pairs} sends from {len(self._sends)} blocks)"

    def run(self, values):
        """Copy each sent element of ``values`` into its place on the receiving block.

        ``values`` is ``{gid: array}`` -- or ``{gid: {name: array}}`` to move
        several arrays in one exchange -- with an entry for every block on this
        rank that sends or receives; each array is on its block's device, and an
        array and its counterparts on other blocks have one element type. Received
        elements are written in place; the others are left as they were."""
        blocks = self.blocks
        if len({isinstance(entry, dict) for entry in values.values()}) > 1:
            raise TypeError("Plan.run() takes an array for every block or a dict of "
                            "arrays for every block, not both")
        named = {gid: entry if isinstance(entry, dict) else {"values": entry}
                 for gid, entry in values.items()}
        blocks._local_keys(named, "Plan.run()")
        needed = sorted(set(self._sends) | set(self._receives))
        missing = [gid for gid in needed if gid not in named]
        if missing:
            raise ValueError(f"Plan.run() needs values for blocks {missing}, which send or "
                             f"receive")
        for gid in needed:
            for name, array in named[gid].items():
                if not isinstance(array, Array):
                    raise TypeError(f"Plan.run() moves Tack arrays; block {gid}'s {name!r} "
                                    f"is a {type(array).__name__}")
                if array._backend.memory_domain != blocks._backend(gid).memory_domain:
                    raise ValueError(f"block {gid}'s {name!r} is on {array.device}, not on "
                                     f"the block's device {blocks.device(gid)}")
        names = {gid: sorted(named[gid]) for gid in needed}
        if len({tuple(n) for n in names.values()}) > 1:
            raise ValueError(f"Plan.run() needs the same names on every block, not "
                             f"{sorted({tuple(n) for n in names.values()})}")

        outgoing = {}
        for gid, destinations in self._sends.items():
            outgoing[gid] = {}
            for destination, indices in destinations.items():
                message = {}
                for name, array in named[gid].items():
                    message[name] = gather(array, indices)
                outgoing[gid][destination] = message
        incoming = blocks.exchange(outgoing)
        for gid, sources in self._receives.items():
            for source, indices in sources.items():
                message = incoming[gid][source]
                for name, array in named[gid].items():
                    received = message[name]
                    if received.dtype != array.dtype:
                        raise TypeError(f"block {source} sent {name!r} as {received.dtype}, "
                                        f"and block {gid} holds {array.dtype}")
                    scatter(received, indices, array)


def _indices(blocks, table, what, other_name):
    out = {}
    for gid, entries in table.items():
        device = blocks._backend(gid)
        kept = {}
        for other in sorted(entries):
            blocks._check_gid(other, f"a {other_name} in Plan's {what}")
            if other == gid:
                raise ValueError(f"Plan's {what} has block {gid} exchanging with itself")
            indices = entries[other]
            if isinstance(indices, Array):
                if indices.dtype not in (tack.i32, tack.i64, tack.u32, tack.u64):
                    raise TypeError(f"Plan's {what} for blocks {gid} and {other} are "
                                    f"{indices.dtype}, not integers")
                if indices._backend.memory_domain != device.memory_domain:
                    raise ValueError(f"Plan's {what} for blocks {gid} and {other} are on "
                                     f"{indices.device}, not on block {gid}'s device")
                indices = indices.reshape(indices.size)
            else:
                host = np.asarray(indices)
                if host.size and host.dtype.kind not in "iu":
                    raise TypeError(f"Plan's {what} for blocks {gid} and {other} are "
                                    f"{host.dtype}, not integers")
                indices = tack.array(host.reshape(-1).astype(np.int64), tack.i64,
                                     device=device)
            if indices.size:
                kept[other] = indices
        if kept:
            out[gid] = kept
    return out
