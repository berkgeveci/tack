"""Blocks: units of work with global ids, each on a device, and what passes between them."""

import operator

import numpy as np

from tack.lang.array import Array
from tack.runtime.dispatch import as_device, backend_of

from ._world import World

# Each reduction's operation on Python numbers and on NumPy arrays.
_OPS = {
    "sum": (operator.add, np.add),
    "min": (min, np.minimum),
    "max": (max, np.maximum),
}


class Blocks:
    """This rank's blocks, one per entry of ``devices`` (a `tack.Device` or a spec
    such as ``"cuda:1"`` each), in a `World` (``World()``, this process alone, if
    not given).

    Global ids run over every rank's blocks, contiguous by rank: rank 0's first,
    then rank 1's, as VTK numbers a partitioned dataset's pieces across ranks.
    ``local`` is the range of this rank's.
    """

    def __init__(self, devices, world=None):
        self.world = World() if world is None else world
        self._devices = [backend_of(d) for d in devices]
        counts = self.world._allgather(len(self._devices))
        starts = np.concatenate(([0], np.cumsum(counts))).astype(int)
        self._starts = starts
        self.count = int(starts[-1])
        self.local = range(int(starts[self.world.rank]), int(starts[self.world.rank + 1]))

    def __repr__(self):
        return f"Blocks({self.count} blocks, {len(self.local)} on rank {self.world.rank})"

    def rank_of(self, gid):
        """The rank holding block ``gid``."""
        self._check_gid(gid, "block")
        return int(np.searchsorted(self._starts, gid, side="right")) - 1

    def device(self, gid):
        """The `tack.Device` of local block ``gid``."""
        return as_device(self._backend(gid))

    def _backend(self, gid):
        if gid not in self.local:
            raise ValueError(f"block {gid} is not on this rank (its blocks are "
                             f"{_span(self.local)})")
        return self._devices[gid - self.local.start]

    def _check_gid(self, gid, what):
        if not isinstance(gid, (int, np.integer)) or not 0 <= gid < self.count:
            raise ValueError(f"{what} {gid!r} is not a block id (0 to {self.count - 1})")

    def _local_keys(self, values, what):
        for gid in values:
            if gid not in self.local:
                raise ValueError(f"{what} names block {gid!r}, which is not on this rank "
                                 f"(its blocks are {_span(self.local)})")

    def _every_local(self, values, what):
        self._local_keys(values, what)
        missing = [gid for gid in self.local if gid not in values]
        if missing:
            raise ValueError(f"{what} needs a value for every block on this rank; "
                             f"blocks {missing} have none")

    def exchange(self, outgoing):
        """Send messages between blocks and return what each block received.

        ``outgoing`` is ``{source: {destination: message}}``: for blocks on this
        rank, a message, a dict of names to Tack arrays on the source block's
        device, for any block. The result is ``{destination: {source: message}}``
        for every block on this rank, in order of global id, with an empty dict
        where nothing came; each message's arrays are on the destination block's
        device.

        A message's arrays belong to the receiver afterwards: where the
        destination's device addresses the source's memory, they are the sent
        arrays themselves, so a sender must not write to what it sent.
        """
        self._local_keys(outgoing, "exchange()")
        messages = []
        for source in sorted(outgoing):
            device = self._backend(source)
            for destination in sorted(outgoing[source]):
                self._check_gid(destination, "a message's destination")
                message = outgoing[source][destination]
                _check_message(message, source, destination, device)
                messages.append((source, destination, message))
        incoming = {gid: {} for gid in self.local}
        for source, destination, message in self.world._exchange(messages, self.rank_of, self.device):
            incoming[destination][source] = message
        return {gid: dict(sorted(received.items())) for gid, received in incoming.items()}

    def all_gather(self, values):
        """Every block's value, a list indexed by global id, on every rank.

        ``values`` is ``{gid: value}`` for every block on this rank; the values
        are small Python objects (over MPI they are pickled)."""
        self._every_local(values, "all_gather()")
        gathered = {}
        for part in self.world._allgather({gid: values[gid] for gid in self.local}):
            gathered.update(part)
        return [gathered[gid] for gid in range(self.count)]

    def all_reduce(self, values, op="sum"):
        """``values`` (``{gid: number or NumPy array}``, every block on this rank)
        combined over every block by ``op`` (``"sum"``, ``"min"`` or ``"max"``).

        Values combine in order of global id, the same order whichever rank
        holds which block, so a floating-point sum has the same bits however
        the blocks are assigned."""
        if op not in _OPS:
            raise ValueError(f"all_reduce() takes op 'sum', 'min' or 'max', not {op!r}")
        every = self.all_gather(values)
        if not every:
            raise ValueError("all_reduce() over no blocks has no value")
        scalar, array = _OPS[op]
        result = every[0]
        for value in every[1:]:
            both_numbers = not isinstance(result, np.ndarray) and \
                not isinstance(value, np.ndarray)
            result = scalar(result, value) if both_numbers else array(result, value)
        return result

    def exclusive_scan(self, counts):
        """Each block's offset, the sum of the counts of the blocks before it:
        ``({gid: offset}, total)`` for the blocks on this rank, from ``counts``
        (``{gid: int}``, every block on this rank) -- where each block's share of
        a global numbering starts."""
        every = self.all_gather(counts)
        for gid, count in enumerate(every):
            if not isinstance(count, (int, np.integer)) or count < 0:
                raise ValueError(f"exclusive_scan() takes counts that are integers of at "
                                 f"least 0; block {gid}'s is {count!r}")
        starts = np.concatenate(([0], np.cumsum(every, dtype=np.int64)))
        return {gid: int(starts[gid]) for gid in self.local}, int(starts[-1])

    def link_by_bounds(self, boxes, margin=0.0):
        """Each block's neighbours: ``{gid: [other gids]}`` for the blocks on this
        rank, every block whose box overlaps or touches its box grown by
        ``margin`` on every side.

        ``boxes`` is ``{gid: (lower, upper)}`` for every block on this rank, two
        corners of equal length. Blocks that share a face touch, so a margin of
        0 finds the neighbours of a decomposition into abutting pieces; a margin
        of ``r`` finds every block within ``r``."""
        if margin < 0:
            raise ValueError(f"link_by_bounds() takes a margin of at least 0, not {margin}")
        for gid, box in boxes.items():
            lower, upper = (np.asarray(corner, float) for corner in box)
            if lower.shape != upper.shape or lower.ndim != 1:
                raise ValueError(f"block {gid}'s box needs two corners of equal length, "
                                 f"not {box!r}")
            if (lower > upper).any():
                raise ValueError(f"block {gid}'s box has a lower corner above its upper "
                                 f"one: {box!r}")
        every = self.all_gather({gid: tuple(np.asarray(c, float) for c in box)
                                 for gid, box in boxes.items()})
        dims = {len(lower) for lower, _ in every}
        if len(dims) > 1:
            raise ValueError(f"link_by_bounds() needs boxes of one dimension, not {sorted(dims)}")
        lower = np.array([box[0] for box in every]).reshape(self.count, -1)
        upper = np.array([box[1] for box in every]).reshape(self.count, -1)
        links = {}
        for gid in self.local:
            near = np.all((lower <= upper[gid] + margin) & (lower[gid] <= upper + margin), axis=1)
            near[gid] = False
            links[gid] = [int(g) for g in np.flatnonzero(near)]
        return links


def _check_message(message, source, destination, device):
    if not isinstance(message, dict):
        raise TypeError(f"the message from block {source} to block {destination} is a "
                        f"{type(message).__name__}, not a dict of names to arrays")
    for name, values in message.items():
        if not isinstance(name, str):
            raise TypeError(f"the message from block {source} to block {destination} "
                            f"names an array {name!r}, not a string")
        if not isinstance(values, Array):
            raise TypeError(f"{name!r} in the message from block {source} to block "
                            f"{destination} is a {type(values).__name__}, not a Tack array")
        if values.layout != "aos":
            raise TypeError(f"{name!r} in the message from block {source} to block "
                            f"{destination} is a structure of arrays; send its "
                            f"components")
        if values._backend.memory_domain != device.memory_domain:
            raise ValueError(f"{name!r} in the message from block {source} to block "
                             f"{destination} is on {values.device}, not on its source "
                             f"block's device {as_device(device)}")


def _span(local):
    return "none" if not local else f"{local.start} to {local.stop - 1}"
