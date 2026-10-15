"""``discontinuous``: ``field`` as an order-1 ``L2`` field -- ``project`` into ``L2``."""

from tack.dataset.algorithms.project import project
from tack.dataset.spaces import L2

__all__ = ['discontinuous']


def discontinuous(data, field):
    """``field`` as an order-1 ``L2`` field, ``project(data, field, L2(data))``: each
    cell's values at its corners, stored per cell. The same function, in the DG
    layout, for a field of order up to 1; edit the values and cells disagree
    where they meet."""
    return project(data, field, L2(data))
