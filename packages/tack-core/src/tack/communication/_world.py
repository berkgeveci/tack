"""The transport between blocks: what moves messages and gathers values across ranks."""


class World:
    """The processes blocks live in, and how messages move between them.

    ``World()`` is this process alone (rank 0 of 1): its blocks may be on any
    of its devices, and a message reaches its destination as copies of its
    arrays on the destination block's device -- the arrays themselves when
    that device can already address them.
    """

    def __init__(self):
        self.rank = 0
        self.size = 1

    def __repr__(self):
        return f"World(rank={self.rank}, size={self.size})"

    def _allgather(self, value):
        """Every rank's ``value``, by rank."""
        return [value]

    def _exchange(self, messages, rank_of, device_of):
        """Deliver ``messages``, ``(source, destination, {name: Array})`` triples from
        this rank's blocks, to the ranks holding their destinations (``rank_of(gid)``),
        and return those addressed to this rank's blocks, each array on its
        destination's device (``device_of(gid)``, a `tack.Device`)."""
        return [(source, destination, {name: values.to_device(device_of(destination))
                                       for name, values in message.items()})
                for source, destination, message in messages]
