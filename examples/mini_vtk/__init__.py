"""mini_vtk — a minimal VTK/Viskores-like library built on PGC.

Provides array abstractions (AOS, SOA, constant), cell set types
(structured, explicit), cell type abstractions (hex, tet, wedge),
a Dataset container, and composable filters.
"""

from .dataset import Dataset, make_rectilinear_dataset, make_explicit_hex_dataset
from .arrays import AOSArray, AOSTupleArray, ConstantArray, ConstantTupleArray3, make_soa_type
from .cellsets import CellSetStructured3D, CellSetExplicit
from .cells import Hexahedron, Tetrahedron, Wedge
from . import filters
