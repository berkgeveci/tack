"""mini_vtk -- a minimal VTK/Viskores-like library built on Tack.

Provides array abstractions (AOS, SOA, constant), cell set types
(structured, explicit), cell type abstractions (hex, tet, wedge),
a Dataset container, and composable filters.
"""

from . import filters
from .arrays import AOSArray, AOSTupleArray, ConstantArray, ConstantTupleArray3, make_soa_type
from .cells import Hexahedron, Tetrahedron, Wedge
from .cellsets import CellSetExplicit, CellSetStructured3D
from .dataset import Dataset, make_explicit_hex_dataset, make_rectilinear_dataset
