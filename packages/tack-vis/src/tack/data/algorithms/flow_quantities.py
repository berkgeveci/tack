"""``flow_quantities``: divergence, vorticity and the Q-criterion from the gradient of a 3-vector field (``gradients`` of it: 9 values per entity), in the gradient's space, as VTK's vtkGradientFilter and Viskores' Gradient compute them: a dict of the three fields."""

from typing import NamedTuple

import tack
from tack.data import arrays
from tack.data.algorithms._common import _field
from tack.data.arrays import materialize, size_of, width_of
from tack.data.dataset import Field

__all__ = ['FlowQuantities', 'flow_quantities']


@tack.kernel
def _flow_quantities(g, divergence, vorticity, q):
    for i in range(divergence.shape[0]):
        J = g[i]
        divergence[i] = J[0] + J[4] + J[8]
        vorticity[i] = [J[7] - J[5], J[2] - J[6], J[3] - J[1]]
        # Q = (|rotation|^2 - |strain|^2) / 2 = -tr(J J) / 2, as VTK computes it.
        q[i] = (-0.5 * (J[0] * J[0] + J[4] * J[4] + J[8] * J[8])
                - (J[1] * J[3] + J[2] * J[6] + J[5] * J[7]))


class FlowQuantities(NamedTuple):
    """``flow_quantities``' fields, each in the gradient's space."""

    divergence: Field
    vorticity: Field
    q_criterion: Field


def flow_quantities(data, gradient):
    """Divergence, vorticity and the Q-criterion from the gradient of a 3-vector field
    (``gradients`` of it: 9 values per entity), in the gradient's space, as VTK's
    vtkGradientFilter and Viskores' Gradient compute them, as ``FlowQuantities``:
    ``divergence``, ``vorticity`` and ``q_criterion``."""
    gradient = _field(data, gradient)
    if width_of(gradient.values) != 9:
        raise TypeError("flow_quantities takes the gradient of a 3-vector field (9 values)")
    n = size_of(gradient.values)
    dtype = arrays.dtype_of(gradient.values)
    divergence = tack.field(dtype, shape=(n,))
    vorticity = tack.Vector.field(3, dtype, shape=(n,))
    q = tack.field(dtype, shape=(n,))
    if n:
        _flow_quantities(materialize(gradient.values), divergence, vorticity, q)
    return FlowQuantities(Field(gradient.space, divergence), Field(gradient.space, vorticity),
                          Field(gradient.space, q))
