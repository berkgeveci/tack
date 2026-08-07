"""FieldAccessor — per-element DOF access patterns.

Two variants:
- ContiguousDofs: DG fields where element i owns dofs[offset[i]..offset[i+1])
- GatheredDofs: H1 fields where element i's DOFs are at dof_values[indices[offset[i]+j]]

Both provide the same interface: get_dof(elem, j) → scalar value.
Use as tack.template() parameter to specialize kernels at compile time.
"""

import numpy as np

import tack


@tack.data_oriented
class ContiguousDofs:
    """DG-style contiguous DOF access.

    Element i's DOFs are at dof_values[offsets[i] .. offsets[i+1]).
    No indirection — direct array access.
    """

    def __init__(self, dof_values, elem_offsets):
        """Create from Tack fields.

        Args:
            dof_values: tack.field of DOF values (float).
            elem_offsets: tack.field(i32) of per-element offsets,
                         length num_elements + 1.
        """
        self.dof_values = dof_values
        self.elem_offsets = elem_offsets

    @tack.func
    def get_dof(self, elem, j):
        """Get j-th DOF value for element elem."""
        return self.dof_values[self.elem_offsets[elem] + j]

    @tack.func
    def ndof(self, elem):
        """Number of DOFs for element elem."""
        return self.elem_offsets[elem + 1] - self.elem_offsets[elem]


@tack.data_oriented
class GatheredDofs:
    """H1-style gathered DOF access.

    Element i's DOFs are at dof_values[dof_indices[offsets[i] + j]]
    for j = 0 .. ndof-1. The dof_indices array provides the indirection
    from per-element local DOF to global DOF index.

    Optionally includes a permutation from tensor-product order to
    MFEM's DOF ordering (corners-edges-interior for H1).
    """

    def __init__(self, dof_values, dof_indices, elem_offsets,
                 tp_to_mfem=None):
        """Create from Tack fields.

        Args:
            dof_values: tack.field of global DOF values (float).
            dof_indices: tack.field(i32) of gather indices.
            elem_offsets: tack.field(i32) of per-element offsets into
                         dof_indices, length num_elements + 1.
            tp_to_mfem: optional tack.field(i32) permutation from
                        tensor-product order to MFEM DOF order.
                        If provided, get_dof(elem, j) treats j as
                        tensor-product index.
        """
        self.dof_values = dof_values
        self.dof_indices = dof_indices
        self.elem_offsets = elem_offsets
        self.tp_to_mfem = tp_to_mfem

    @tack.func
    def get_dof(self, elem, j):
        """Get j-th DOF value for element elem.

        j is in tensor-product order if tp_to_mfem was provided.
        """
        local_j = j
        if self.tp_to_mfem is not None:
            local_j = self.tp_to_mfem[j]
        return self.dof_values[
            self.dof_indices[self.elem_offsets[elem] + local_j]]

    @tack.func
    def ndof(self, elem):
        return self.elem_offsets[elem + 1] - self.elem_offsets[elem]


# ── Construction helpers ────────────────────────────────────────────

def contiguous_from_numpy(dof_values_np, elem_offsets_np, np_fp=np.float64):
    """Create ContiguousDofs from numpy arrays.

    Args:
        dof_values_np: numpy array of DOF values.
        elem_offsets_np: numpy int32 array of per-element offsets.
        np_fp: float dtype for the DOF values.
    """
    return ContiguousDofs(
        dof_values=tack.field_like(dof_values_np.astype(np_fp)),
        elem_offsets=tack.field_like(elem_offsets_np.astype(np.int32)))


def gathered_from_numpy(dof_values_np, dof_indices_np, elem_offsets_np,
                        tp_perm_np=None, np_fp=np.float64):
    """Create GatheredDofs from numpy arrays.

    Args:
        dof_values_np: numpy array of global DOF values.
        dof_indices_np: numpy int32 array of gather indices.
        elem_offsets_np: numpy int32 array of per-element offsets.
        tp_perm_np: optional numpy int32 permutation array
                    (tensor-product → MFEM order).
        np_fp: float dtype for DOF values.
    """
    tp_field = None
    if tp_perm_np is not None:
        tp_field = tack.field_like(tp_perm_np.astype(np.int32))
    return GatheredDofs(
        dof_values=tack.field_like(dof_values_np.astype(np_fp)),
        dof_indices=tack.field_like(dof_indices_np.astype(np.int32)),
        elem_offsets=tack.field_like(elem_offsets_np.astype(np.int32)),
        tp_to_mfem=tp_field)
