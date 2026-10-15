"""Per-face computations shared by ``jump``, ``upwind_flux``, ``divergence``
and ``perot``."""

from tack.dataset.spaces import Constant, Values


def _on_faces(data, field):
    """Whether ``field`` is values on ``data``'s faces, oriented or not."""
    space = field.space
    return (isinstance(space, Values) and space.on == "faces"
            and space.topology is data.topology)


def _on_cells(data, field):
    space = field.space
    return isinstance(space, Constant) or space is Values(data, "cells")
