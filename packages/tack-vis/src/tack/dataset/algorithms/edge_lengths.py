"""``edge_lengths``: each edge's length: a field on edges."""

import tack
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['edge_lengths']


@tack.kernel
def _edge_lengths(edges, out):
    for e in edges:
        out[edges.entity_id(e)] = (edges.corner_position(e, 1) - edges.corner_position(e, 0)).norm()


@on_inputs_device
def edge_lengths(data):
    """Each edge's length: a field on edges."""
    out = tack.empty((data.topology.edges().num_edges,), data.dtype)
    for_each(_edge_lengths, data, "edges", out)
    return Field(Values(data, "edges"), out)
