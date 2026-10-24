"""Routing points to the blocks that hold them, and answers back."""

import numpy as np

import tack
from tack.lang.array import Array

from ._kernels import gather, scatter


class Routing:
    """Where each point goes: the lowest block that holds it.

    ``positions`` is ``{gid: points}`` for every block on this rank, an
    ``(n, d)`` host array of coordinates each (``n`` may be 0); ``boxes``
    ``{gid: (lower, upper)}`` for every block on this rank, ``None`` for a block
    that holds nothing. A point is asked of every block whose box holds it (with
    a relative slack of 1e-6), and ``holds(gid, points)`` -- called by the block
    asked, with the points it was asked about as an ``(m, d)`` float64 host
    array -- says how well it holds each: 0 not at all, or a rank (``True`` is
    1). The point goes to the block of the highest rank, the lowest such block
    on a tie: a locator that holds a point beats one only within its tolerance. Coordinates travel as their f64 bits in ``u64`` arrays, which
    every device stores, so a block decides on exactly the points it was given.

    ``destination`` is ``{gid: array}``, each point's block, or -1 where no block
    holds it. ``send`` moves values with the points to their destinations, and
    ``reply`` returns values from there to where the points came from.
    """

    def __init__(self, blocks, positions, boxes, holds):
        self.blocks = blocks
        blocks._every_local(positions, "Routing's positions")
        blocks._every_local(boxes, "Routing's boxes")
        self._positions = {g: _points(positions[g]) for g in blocks.local}
        every = blocks.all_gather({g: None if boxes[g] is None else
                                   tuple(np.asarray(c, np.float64) for c in boxes[g])
                                   for g in blocks.local})

        # 1. Each point to every block whose box holds it.
        asked = {}
        for g in blocks.local:
            x = self._positions[g]
            asked[g] = {}
            for d, box in enumerate(every):
                if box is None or not len(x):
                    continue
                lower, upper = box
                slack = 1e-6 * max(float(np.max(upper - lower)) if len(upper) else 0.0, 1.0)
                inside = np.all((x >= lower - slack) & (x <= upper + slack), axis=1)
                rows = np.flatnonzero(inside)
                if len(rows):
                    message = _bits(x[rows], blocks.device(g))
                    message["slot"] = tack.array(rows.astype(np.int64), tack.i64,
                                                 device=blocks.device(g))
                    asked[g][d] = message
        questions = blocks.exchange(asked)

        # 2. Each block says which it holds.
        answers = {}
        for g in blocks.local:
            answers[g] = {}
            for origin, message in questions[g].items():
                held = np.asarray(holds(g, _unbits(message))).astype(np.int32).reshape(-1)
                answers[g][origin] = {"slot": message["slot"],
                                      "held": tack.array(held, tack.i32,
                                                         device=blocks.device(g))}
        replies = blocks.exchange(answers)

        # 3. The highest rank takes it, the lowest block on a tie (replies come
        # in order of block, so a later one must rank strictly higher).
        self.destination = {}
        for g in blocks.local:
            n = len(self._positions[g])
            destination = np.full(n, -1, np.int64)
            best = np.zeros(n, np.int32)
            for d, reply in replies[g].items():
                slots = reply["slot"].numpy()
                rank = reply["held"].numpy()
                better = rank > best[slots]
                destination[slots[better]] = d
                best[slots[better]] = rank[better]
            self.destination[g] = destination
        self._arrivals = None

    def send(self, values=None):
        """Move ``values`` (``{gid: {name: array}}``, an element per point, on the
        block's device; may be omitted) with the points to their destinations.
        Returns ``{gid: (positions, {name: array})}`` for every block on this
        rank: what arrived there, the points as an ``(m, d)`` host array and the
        values on the block's device, in order of origin block and then of the
        points there. ``reply`` answers in that order."""
        blocks = self.blocks
        values = values or {g: {} for g in blocks.local}
        outgoing, self._order = {}, {}
        for g in blocks.local:
            outgoing[g] = {}
            for d in np.unique(self.destination[g][self.destination[g] >= 0]).tolist():
                rows = np.flatnonzero(self.destination[g] == d)
                device = blocks.device(g)
                message = _bits(self._positions[g][rows], device)
                message["slot"] = tack.array(rows.astype(np.int64), tack.i64, device=device)
                indices = tack.array(rows.astype(np.int64), tack.i64, device=device)
                for name, array in values.get(g, {}).items():
                    if not isinstance(array, Array):
                        raise TypeError(f"Routing.send() moves Tack arrays; block {g}'s "
                                        f"{name!r} is a {type(array).__name__}")
                    message["value." + name] = gather(array, indices)
                outgoing[g][int(d)] = message
        incoming = blocks.exchange(outgoing)
        arrived = {}
        self._arrivals = {}
        for g in blocks.local:
            parts = list(incoming[g].items())               # in order of origin
            self._arrivals[g] = [(origin, m["slot"], m["slot"].size) for origin, m in parts]
            x = (np.concatenate([_unbits(m) for _, m in parts]) if parts
                 else np.zeros((0, self._width()), np.float64))
            names = sorted({k for _, m in parts for k in m if k.startswith("value.")})
            joined = {}
            for name in names:
                pieces = [m[name] for _, m in parts]
                joined[name[len("value."):]] = pieces[0] if len(pieces) == 1 \
                    else tack.concat(pieces)
            arrived[g] = (x, joined)
        return arrived

    def reply(self, values, like):
        """Send ``values`` -- ``{gid: {name: array}}``, an element for each point
        that arrived at the block (``send``'s order) -- back to where the points
        came from. Returns ``{gid: {name: array}}`` for the blocks on this rank,
        an element per point of the block's ``positions``, zero where the point
        had no destination. ``like`` is ``{name: element type}``, for blocks
        none of whose points were taken."""
        if self._arrivals is None:
            raise ValueError("Routing.reply() answers what Routing.send() delivered: "
                             "send first")
        blocks = self.blocks
        outgoing = {}
        for g in blocks.local:
            outgoing[g] = {}
            start = 0
            for origin, slots, count in self._arrivals[g]:
                indices = tack.array(np.arange(start, start + count, dtype=np.int64),
                                     tack.i64, device=blocks.device(g))
                message = {"slot": slots.to_device(blocks.device(g))}
                for name in like:
                    message["value." + name] = gather(values[g][name], indices)
                outgoing[g][origin] = message
                start += count
        incoming = blocks.exchange(outgoing)
        out = {}
        for g in blocks.local:
            n = len(self._positions[g])
            out[g] = {}
            for name, dtype in like.items():
                array = tack.zeros((n,), dtype, device=blocks.device(g))
                for _, message in incoming[g].items():
                    scatter(message["value." + name], message["slot"], array)
                out[g][name] = array
        return out

    def _width(self):
        widths = [x.shape[1] for x in self._positions.values() if x.ndim == 2]
        return widths[0] if widths else 3


def _points(points):
    """Points as an ``(n, d)`` float64 host array."""
    x = np.asarray(points, np.float64)
    return x if x.ndim == 2 else x.reshape(len(x), -1)


def _bits(x, device):
    bits = np.ascontiguousarray(np.asarray(x, np.float64)).view(np.uint64)
    bits = bits.reshape(len(x), -1)
    return {f"x{axis}": tack.array(bits[:, axis].copy(), tack.u64, device=device)
            for axis in range(bits.shape[1])}


def _unbits(message):
    axes = sorted(int(k[1:]) for k in message if k.startswith("x") and k[1:].isdigit())
    if not axes:
        return np.zeros((0, 0))
    return np.stack([message[f"x{axis}"].numpy().view(np.float64) for axis in axes], axis=1)
