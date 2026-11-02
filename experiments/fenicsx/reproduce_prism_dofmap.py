# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Reproduce the DOLFINx 0.11 prism H1 dofmap failure, without importing Tack.

Run in fenicsx-tack. Basix creates the 40-DOF element successfully, then
fem.functionspace raises 'Multiple index maps of this dimension'.
"""

import basix
import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI


def main():
    domain = mesh.create_box(MPI.COMM_SELF, [[0, 0, 0], [1, 1, 1]], [1, 1, 1],
                             cell_type=mesh.CellType.prism)
    element = basix.ufl.element('P', 'prism', 3,
                                lagrange_variant=basix.LagrangeVariant.equispaced)
    print('Basix element constructed:', element.dim, flush=True)
    fem.functionspace(domain, element)


if __name__ == '__main__':
    main()
