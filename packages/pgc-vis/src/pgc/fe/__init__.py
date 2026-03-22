"""PGC FE — Finite element field evaluation on GPU.

Provides portable abstractions for evaluating high-order finite element
fields from simulation codes like MFEM.

Three layers:
- ElementBasis: evaluate basis functions at parametric coordinates
- FieldAccessor: per-element DOF access (contiguous or gathered)
- GeometryMap: physical ↔ parametric coordinate mapping
"""

from pgc.fe.basis import QuadBasis, HexBasis, lagrange_1d, lagrange_1d_deriv
from pgc.fe.accessor import ContiguousDofs, GatheredDofs
from pgc.fe.geometry import LinearQuadMap, LinearHexMap

__all__ = [
    "QuadBasis", "HexBasis",
    "lagrange_1d", "lagrange_1d_deriv",
    "ContiguousDofs", "GatheredDofs",
    "LinearQuadMap", "LinearHexMap",
]
