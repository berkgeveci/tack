"""Communication between blocks: the patterns algorithms over pieces are written on.

A block is a unit of work on one device -- in ``tack.dataset``, a piece of a
partitioned dataset -- with a global id. `Blocks` holds this process's blocks
and their devices; its methods move data between blocks and combine values
over them:

- `Blocks.exchange`: messages, each a dict of named Tack arrays, from any
  block to any block. Neighbour exchange and routing by destination are the
  same call.
- `Blocks.all_reduce`, `Blocks.all_gather`, `Blocks.exclusive_scan`: small
  host values combined over every block, in global-id order, so the result
  does not depend on how blocks are assigned.
- `Blocks.link_by_bounds`: each block's neighbours, by overlapping boxes.
- `Plan`: an exchange kept for reuse -- which elements each block sends to
  which, and where the receiver puts them -- as ghost values are refreshed.
- `Routing`: points to the lowest block that holds them (by the blocks'
  boxes and a test of the caller's), values sent with them, and answers
  sent back -- particles and probes across pieces.
- `sort_by_key`: every block's keys sorted together, the global order cut
  into one range per block, every copy of a key on one block -- where equal
  keys (matching points, global ids) are grouped.

A `World` is the transport: ``World()`` is one process, blocks on any of its
devices, a message moved by copying its arrays to the destination block's
device; ``World(comm)`` spans the ranks of an mpi4py communicator.
Algorithms written on `Blocks` run unchanged on either. It is DIY's model
(blocks, links, exchange, collectives) with Tack arrays as the messages
instead of bytes, and functions in place of callbacks: ``exchange`` takes
``{source: {destination: message}}`` and returns
``{destination: {source: message}}``.
"""

from ._blocks import Blocks
from ._plan import Plan
from ._route import Routing
from ._sort import sort_by_key
from ._world import World

__all__ = ["Blocks", "Plan", "Routing", "World", "sort_by_key"]
