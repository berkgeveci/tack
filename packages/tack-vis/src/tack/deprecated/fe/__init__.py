"""Tack FE — Finite element field evaluation on GPU.

Provides portable abstractions for evaluating high-order finite element
fields from simulation codes like MFEM.

Three layers:
- ElementBasis: evaluate basis functions at parametric coordinates
- FieldAccessor: per-element DOF access (contiguous or gathered)
- GeometryMap: physical ↔ parametric coordinate mapping
"""

from tack.deprecated.fe.accessor import ContiguousDofs, GatheredDofs
from tack.deprecated.fe.basis import HexBasis, QuadBasis, lagrange_1d, lagrange_1d_deriv
from tack.deprecated.fe.geometry import LinearHexMap, LinearQuadMap

__all__ = [
    "ContiguousDofs",
    "GatheredDofs",
    "HexBasis",
    "LinearHexMap",
    "LinearQuadMap",
    "QuadBasis",
    "lagrange_1d",
    "lagrange_1d_deriv",
]
