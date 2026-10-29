"""``advect``: seeds moved through the velocity ``field`` (a 3-vector field or its name: point data, or cell data) by up to ``steps`` fixed steps of ``step_size``, as Viskores' ParticleAdvection: a dataset of the particles where they stopped, one vertex cell each, with ``steps`` (taken), ``time`` and ``status`` (the bits ``SUCCESS``, ``TERMINATE``, ``SPATIAL_BOUNDS``, ``TOOK_ANY_STEPS``, ``ZERO_VELOCITY``). ``seeds`` is an ``(n, 3)`` array or a field of 3-vectors; ``integrator`` ``"rk4"`` or ``"euler"``; ``locator`` reuses a ``CellLocator`` of ``data``."""

import numpy as np

from tack.dataset.algorithms._advection import ParticleStatus, _parameters, _run
from tack.dataset.algorithms._common import _time_unit
from tack.dataset.algorithms._extract import _vertices
from tack.dataset.dataset import DataSet, Field
from tack.dataset.partitioned import PartitionedDataSet
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['ParticleStatus', 'advect']


@on_inputs_device
def advect(data, field, seeds, step_size, steps, integrator="rk4", locator=None,
           tolerance=None, min_step=None, max_step=None, max_length=None,
           neighbors=False):
    """Seeds moved through the velocity ``field`` (a 3-vector field or its name: point
    data, cell data, or a field of any order or ``L2``, on a straight or curved
    geometry) by up to ``steps`` fixed steps of ``step_size``, as
    Viskores' ParticleAdvection: a dataset of the particles where they stopped,
    one vertex cell each, with ``steps`` (taken), ``time`` and ``status`` (the
    bits ``SUCCESS``, ``TERMINATE``, ``SPATIAL_BOUNDS``, ``TOOK_ANY_STEPS``,
    ``ZERO_VELOCITY``). ``seeds`` is an ``(n, 3)`` array or a field of
    3-vectors; ``integrator`` ``"rk4"``, ``"euler"`` or ``"rk45"`` (adaptive steps,
    with ``tolerance``, ``min_step`` and ``max_step``: see ``streamlines``);
    ``max_length`` stops a particle whose path has gone that far; ``locator``
    reuses a ``CellLocator`` of ``data``.
    ``neighbors=True`` tries the face neighbours of the cell a particle was last
    found in before searching the locator's bins: the same particles, bit for
    bit, in about half the time on large meshes, for a table of six ids a cell
    kept with the locator (``CellLocator.cell_neighbors``).

    ``seeds`` may also be what ``advect`` returned: the particles go on from
    where they stopped, with their times and the steps they took (``steps``
    counts those too), and a particle that stopped -- its steps done, at zero
    velocity, or outside -- stays as it is. A ``seed`` point field on it comes
    along.

    Over a ``PartitionedDataSet`` (``field`` by name), particles cross from
    piece to piece, in rounds: each goes to the lowest piece one of whose own
    cells (not a ghost) holds it, steps until it leaves that piece, and is
    handed on, until none is in flight on any rank.

    Each particle is one thread that takes all its steps, so on a GPU a few
    seeds use a few threads of a device built for thousands, each slower than
    a CPU core: below a few thousand seeds a CPU copy of the mesh is faster
    (the Visualization Guide's "Integrate a steady velocity field")."""
    # A step as a time ("0.5 s") is in the spatial unit over the velocity's.
    step_size, tolerance, min_step, max_step, max_length = _parameters(
        data, field, "time", "advect", step_size, tolerance, min_step, max_step, max_length)
    adaptive = {"tolerance": tolerance, "min_step": min_step, "max_step": max_step,
                "max_length": max_length}
    if isinstance(data, PartitionedDataSet):
        from tack.dataset.algorithms._particles import advect_pieces
        return advect_pieces(data, field, seeds, step_size, steps, integrator, neighbors,
                             **adaptive)
    from tack.dataset.ghosts import refuse_hidden, visible

    if locator is None:
        data, field = visible(data, field)
    else:
        refuse_hidden(data, "advect with a locator")
    seeds, resume, seed_ids = _resumed(seeds, data)
    xs, times, taken, status, _, _, _, _ = _run(data, field, seeds, step_size, steps,
                                                integrator, locator, record=False,
                                                resume=resume, neighbors=neighbors,
                                                **adaptive)
    out = _vertices(xs, data)                  # the particles stay on the device
    points = Values(out, "points")
    for name, values, unit in (("steps", taken, None), ("time", times, _time_unit(data, field)),
                               ("status", status, None)):
        out.fields[name] = Field(points, values, unit)
    if seed_ids is not None:
        out.fields["seed"] = Field(points, seed_ids.to_device(out.device))
    return out


def _resumed(seeds, data):
    """``seeds`` as positions, and the state to go on from when they are what
    ``advect`` returned: ``(positions, (times, steps, status) or None, seed ids
    or None)``."""
    if not isinstance(seeds, DataSet):
        return seeds, None, None
    missing = [n for n in ("steps", "time", "status") if n not in seeds.fields]
    if missing:
        raise ValueError(f"seeds given as a dataset go on from what advect returned, "
                         f"which has steps, time and status; these have no "
                         f"{', '.join(missing)}")
    resume = tuple(seeds.fields[n].values for n in ("time", "steps", "status"))
    ids = seeds.fields["seed"].values if "seed" in seeds.fields else None
    values = seeds.geometry.values
    if seeds.origin != data.origin:              # positions relative to data's origin
        from tack.dataset.algorithms._moving import _shifted_values

        values = _shifted_values(values, seeds.dtype,
                                 np.asarray(seeds.origin) - np.asarray(data.origin))
    return values, resume, ids
