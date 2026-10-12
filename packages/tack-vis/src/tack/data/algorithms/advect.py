"""``advect``: seeds moved through the velocity ``field`` (a 3-vector field or its name: point data, or cell data) by up to ``steps`` fixed steps of ``step_size``, as Viskores' ParticleAdvection: a dataset of the particles where they stopped, one vertex cell each, with ``steps`` (taken), ``time`` and ``status`` (the bits ``SUCCESS``, ``TERMINATE``, ``SPATIAL_BOUNDS``, ``TOOK_ANY_STEPS``, ``ZERO_VELOCITY``). ``seeds`` is an ``(n, 3)`` array or a field of 3-vectors; ``integrator`` ``"rk4"`` or ``"euler"``; ``locator`` reuses a ``CellLocator`` of ``data``."""

import numpy as np

from tack.data.algorithms._advection import ParticleStatus, _run
from tack.data.algorithms._extract import _vertices
from tack.data.dataset import Field
from tack.data.spaces import Values

__all__ = ['ParticleStatus', 'advect']


def advect(data, field, seeds, step_size, steps, integrator="rk4", locator=None):
    """Seeds moved through the velocity ``field`` (a 3-vector field or its name: point
    data, or cell data) by up to ``steps`` fixed steps of ``step_size``, as
    Viskores' ParticleAdvection: a dataset of the particles where they stopped,
    one vertex cell each, with ``steps`` (taken), ``time`` and ``status`` (the
    bits ``SUCCESS``, ``TERMINATE``, ``SPATIAL_BOUNDS``, ``TOOK_ANY_STEPS``,
    ``ZERO_VELOCITY``). ``seeds`` is an ``(n, 3)`` array or a field of
    3-vectors; ``integrator`` ``"rk4"`` or ``"euler"``; ``locator`` reuses a
    ``CellLocator`` of ``data``."""
    xs, times, taken, status, _, _, _ = _run(data, field, seeds, step_size, steps,
                                             integrator, locator, record=False)
    out = _vertices(xs.to_numpy(vectors=True) if xs.size else np.zeros((0, 3)))
    points = Values(out, "points")
    for name, values in (("steps", taken), ("time", times), ("status", status)):
        out.fields[name] = Field(points, values)
    return out
