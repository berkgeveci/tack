"""``flow_quantities``: divergence, vorticity and the Q-criterion from the gradient of a 3-vector field (``gradients`` of it: 9 values per entity), in the gradient's space, as VTK's vtkGradientFilter and Viskores' Gradient compute them: a dict of the three fields."""

import tack
from tack.data import arrays
from tack.data.arrays import materialize, size_of, width_of
from tack.data.dataset import Field

__all__ = ['flow_quantities']


@tack.kernel
def _flow_quantities(g, divergence, vorticity, q):
    for i in range(divergence.shape[0]):
        J = g[i]
        divergence[i] = J[0] + J[4] + J[8]
        vorticity[i] = [J[7] - J[5], J[2] - J[6], J[3] - J[1]]
        # Q = (|rotation|^2 - |strain|^2) / 2 = -tr(J J) / 2, as VTK computes it.
        q[i] = (-0.5 * (J[0] * J[0] + J[4] * J[4] + J[8] * J[8])
                - (J[1] * J[3] + J[2] * J[6] + J[5] * J[7]))


def flow_quantities(gradient):
    """Divergence, vorticity and the Q-criterion from the gradient of a 3-vector field
    (``gradients`` of it: 9 values per entity), in the gradient's space, as VTK's
    vtkGradientFilter and Viskores' Gradient compute them: a dict of the three
    fields."""
    if width_of(gradient.values) != 9:
        raise TypeError("flow_quantities takes the gradient of a 3-vector field (9 values)")
    n = size_of(gradient.values)
    dtype = arrays.dtype_of(gradient.values)
    divergence = tack.field(dtype, shape=(n,))
    vorticity = tack.Vector.field(3, dtype, shape=(n,))
    q = tack.field(dtype, shape=(n,))
    if n:
        _flow_quantities(materialize(gradient.values), divergence, vorticity, q)
    return {"divergence": Field(gradient.space, divergence),
            "vorticity": Field(gradient.space, vorticity),
            "q_criterion": Field(gradient.space, q)}
