"""``divergence``: each cell's finite-volume divergence, its net outflow (``net_flux``) over its measure."""

import tack
from tack.dataset.algorithms.cell_geometry import cell_geometry
from tack.dataset.algorithms.net_flux import net_flux
from tack.dataset.carry import _like
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['divergence']


@tack.kernel
def _per_measure(net, measure, out):
    for c in range(out.shape[0]):
        out[c] = net[c] / measure[c]


@on_inputs_device
def divergence(data, flux):
    """Each cell's finite-volume divergence of ``flux``, the integrated flux through
    each face out of its side 0 (``upwind_flux``, or a normal component times the
    area): ``net_flux`` over the cell's measure (``cell_geometry``), so for a
    linear field it is the field's divergence. A cell of no measure gives no
    finite value, as ``perot`` and ``gradients`` give. ``flow_quantities`` gives
    the finite-element divergence instead, the trace of a field's gradient."""
    net = net_flux(data, flux)
    measure = cell_geometry(data).measure
    out = _like(net.values, data.num_cells)
    if data.num_cells:
        _per_measure(net.values, measure.values, out)
    return Field(Values(data, "cells"), out)
