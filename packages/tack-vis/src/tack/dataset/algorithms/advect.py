"""``advect``: seeds moved through the velocity ``field`` (a 3-vector field or its name: point data, or cell data) by up to ``steps`` fixed steps of ``step_size``, as Viskores' ParticleAdvection: a dataset of the particles where they stopped, one vertex cell each, with ``steps`` (taken), ``time`` and ``status`` (the bits ``SUCCESS``, ``TERMINATE``, ``SPATIAL_BOUNDS``, ``TOOK_ANY_STEPS``, ``ZERO_VELOCITY``). ``seeds`` is an ``(n, 3)`` array or a field of 3-vectors; ``integrator`` ``"rk4"`` or ``"euler"``; ``locator`` reuses a ``CellLocator`` of ``data``."""

from tack.dataset.algorithms._advection import ParticleStatus, _run
from tack.dataset.algorithms._extract import _vertices
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['ParticleStatus', 'advect']


@on_inputs_device
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
    out = _vertices(xs, data)                  # the particles stay on the device
    points = Values(out, "points")
    for name, values in (("steps", taken), ("time", times), ("status", status)):
        out.fields[name] = Field(points, values)
    return out
