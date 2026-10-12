"""``edge_lengths``: each edge's length: a field on edges."""

import tack
from tack.data.dataset import Field, for_each
from tack.data.spaces import Values

__all__ = ['edge_lengths']


@tack.kernel
def _edge_lengths(edges, out):
    for e in edges:
        out[edges.entity_id(e)] = (edges.point(e, 1) - edges.point(e, 0)).norm()


def edge_lengths(data):
    """Each edge's length: a field on edges."""
    out = tack.field(data.dtype, shape=(data.topology.edges().num_edges,))
    for_each(_edge_lengths, data, "edges", out)
    return Field(Values(data, "edges"), out)
