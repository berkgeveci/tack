"""Flying Edges isosurface extraction — single-block and multi-block.

Implements the Flying Edges algorithm (merged-point variant) for uniform
grids.  The multi-block entry point runs per-block kernels with a two-pass
count/emit strategy so that all blocks write into a single pair of output
arrays (points + connectivity).

Public API
----------
flying_edges(scalar, grid, isovalue)
    Isosurface a single uniform grid block.

flying_edges_multiblock(blocks, isovalue)
    Isosurface a list of uniform grid blocks into unified output.

UniformGrid
    Data-oriented grid descriptor (origin + spacing).

MCTables
    Marching-cubes lookup tables as tack fields.
"""

import weakref

import numpy as np

import tack

# ================================================================
# MARCHING CUBES TABLES
# ================================================================

_EDGE_CORNERS = np.array([
    0, 1,  1, 2,  3, 2,  0, 3,
    4, 5,  5, 6,  7, 6,  4, 7,
    0, 4,  1, 5,  3, 7,  2, 6,
], dtype=np.int32)

# fmt: off
_TRI_TABLE = np.array([
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,3,8,1,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    9,11,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    2,3,8,2,8,11,11,8,9,-1,-1,-1,-1,-1,-1,-1,
    3,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,2,10,8,0,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,0,9,2,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,2,10,1,10,9,9,10,8,-1,-1,-1,-1,-1,-1,-1,
    3,1,11,10,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,1,11,0,11,8,8,11,10,-1,-1,-1,-1,-1,-1,-1,
    3,0,9,3,9,10,10,9,11,-1,-1,-1,-1,-1,-1,-1,
    9,11,8,11,10,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,9,1,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1,
    1,11,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    3,7,4,3,4,0,1,11,2,-1,-1,-1,-1,-1,-1,-1,
    9,11,2,9,2,0,8,7,4,-1,-1,-1,-1,-1,-1,-1,
    2,9,11,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1,
    8,7,4,3,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    10,7,4,10,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1,
    9,1,0,8,7,4,2,10,3,-1,-1,-1,-1,-1,-1,-1,
    4,10,7,9,10,4,9,2,10,9,1,2,-1,-1,-1,-1,
    3,1,11,3,11,10,7,4,8,-1,-1,-1,-1,-1,-1,-1,
    1,11,10,1,10,4,1,4,0,7,4,10,-1,-1,-1,-1,
    4,8,7,9,10,0,9,11,10,10,3,0,-1,-1,-1,-1,
    4,10,7,4,9,10,9,11,10,-1,-1,-1,-1,-1,-1,-1,
    9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    9,4,5,0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1,
    1,11,2,9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    3,8,0,1,11,2,4,5,9,-1,-1,-1,-1,-1,-1,-1,
    5,11,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1,
    2,5,11,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1,
    9,4,5,2,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,2,10,0,10,8,4,5,9,-1,-1,-1,-1,-1,-1,-1,
    0,4,5,0,5,1,2,10,3,-1,-1,-1,-1,-1,-1,-1,
    2,5,1,2,8,5,2,10,8,4,5,8,-1,-1,-1,-1,
    11,10,3,11,3,1,9,4,5,-1,-1,-1,-1,-1,-1,-1,
    4,5,9,0,1,8,8,1,11,8,11,10,-1,-1,-1,-1,
    5,0,4,5,10,0,5,11,10,10,3,0,-1,-1,-1,-1,
    5,8,4,5,11,8,11,10,8,-1,-1,-1,-1,-1,-1,-1,
    9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1,
    0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1,
    1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    9,8,7,9,7,5,11,2,1,-1,-1,-1,-1,-1,-1,-1,
    11,2,1,9,0,5,5,0,3,5,3,7,-1,-1,-1,-1,
    8,2,0,8,5,2,8,7,5,11,2,5,-1,-1,-1,-1,
    2,5,11,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,
    7,5,9,7,9,8,3,2,10,-1,-1,-1,-1,-1,-1,-1,
    9,7,5,9,2,7,9,0,2,2,10,7,-1,-1,-1,-1,
    2,10,3,0,8,1,1,8,7,1,7,5,-1,-1,-1,-1,
    10,1,2,10,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1,
    9,8,5,8,7,5,11,3,1,11,10,3,-1,-1,-1,-1,
    5,0,7,5,9,0,7,0,10,1,11,0,10,0,11,-1,
    10,0,11,10,3,0,11,0,5,8,7,0,5,0,7,-1,
    10,5,11,7,5,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    11,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,3,8,5,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    9,1,0,5,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,3,8,1,8,9,5,6,11,-1,-1,-1,-1,-1,-1,-1,
    1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,5,6,1,6,2,3,8,0,-1,-1,-1,-1,-1,-1,-1,
    9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1,
    5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1,
    2,10,3,11,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    10,8,0,10,0,2,11,5,6,-1,-1,-1,-1,-1,-1,-1,
    0,9,1,2,10,3,5,6,11,-1,-1,-1,-1,-1,-1,-1,
    5,6,11,1,2,9,9,2,10,9,10,8,-1,-1,-1,-1,
    6,10,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1,
    0,10,8,0,5,10,0,1,5,5,6,10,-1,-1,-1,-1,
    3,6,10,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1,
    6,9,5,6,10,9,10,8,9,-1,-1,-1,-1,-1,-1,-1,
    5,6,11,4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,0,3,4,3,7,6,11,5,-1,-1,-1,-1,-1,-1,-1,
    1,0,9,5,6,11,8,7,4,-1,-1,-1,-1,-1,-1,-1,
    11,5,6,1,7,9,1,3,7,7,4,9,-1,-1,-1,-1,
    6,2,1,6,1,5,4,8,7,-1,-1,-1,-1,-1,-1,-1,
    1,5,2,5,6,2,3,4,0,3,7,4,-1,-1,-1,-1,
    8,7,4,9,5,0,0,5,6,0,6,2,-1,-1,-1,-1,
    7,9,3,7,4,9,3,9,2,5,6,9,2,9,6,-1,
    3,2,10,7,4,8,11,5,6,-1,-1,-1,-1,-1,-1,-1,
    5,6,11,4,2,7,4,0,2,2,10,7,-1,-1,-1,-1,
    0,9,1,4,8,7,2,10,3,5,6,11,-1,-1,-1,-1,
    9,1,2,9,2,10,9,10,4,7,4,10,5,6,11,-1,
    8,7,4,3,5,10,3,1,5,5,6,10,-1,-1,-1,-1,
    5,10,1,5,6,10,1,10,0,7,4,10,0,10,4,-1,
    0,9,5,0,5,6,0,6,3,10,3,6,8,7,4,-1,
    6,9,5,6,10,9,4,9,7,7,9,10,-1,-1,-1,-1,
    11,9,4,6,11,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,6,11,4,11,9,0,3,8,-1,-1,-1,-1,-1,-1,-1,
    11,1,0,11,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1,
    8,1,3,8,6,1,8,4,6,6,11,1,-1,-1,-1,-1,
    1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1,
    3,8,0,1,9,2,2,9,4,2,4,6,-1,-1,-1,-1,
    0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,
    11,9,4,11,4,6,10,3,2,-1,-1,-1,-1,-1,-1,-1,
    0,2,8,2,10,8,4,11,9,4,6,11,-1,-1,-1,-1,
    3,2,10,0,6,1,0,4,6,6,11,1,-1,-1,-1,-1,
    6,1,4,6,11,1,4,1,8,2,10,1,8,1,10,-1,
    9,4,6,9,6,3,9,3,1,10,3,6,-1,-1,-1,-1,
    8,1,10,8,0,1,10,1,6,9,4,1,6,1,4,-1,
    3,6,10,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1,
    6,8,4,10,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    7,6,11,7,11,8,8,11,9,-1,-1,-1,-1,-1,-1,-1,
    0,3,7,0,7,11,0,11,9,6,11,7,-1,-1,-1,-1,
    11,7,6,1,7,11,1,8,7,1,0,8,-1,-1,-1,-1,
    11,7,6,11,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1,
    1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1,
    2,9,6,2,1,9,6,9,7,0,3,9,7,9,3,-1,
    7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1,
    7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    2,10,3,11,8,6,11,9,8,8,7,6,-1,-1,-1,-1,
    2,7,0,2,10,7,0,7,9,6,11,7,9,7,11,-1,
    1,0,8,1,8,7,1,7,11,6,11,7,2,10,3,-1,
    10,1,2,10,7,1,11,1,6,6,1,7,-1,-1,-1,-1,
    8,6,9,8,7,6,9,6,1,10,3,6,1,6,3,-1,
    0,1,9,10,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    7,0,8,7,6,0,3,0,10,10,0,6,-1,-1,-1,-1,
    7,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    7,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    3,8,0,10,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,9,1,10,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    8,9,1,8,1,3,10,6,7,-1,-1,-1,-1,-1,-1,-1,
    11,2,1,6,7,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,11,2,3,8,0,6,7,10,-1,-1,-1,-1,-1,-1,-1,
    2,0,9,2,9,11,6,7,10,-1,-1,-1,-1,-1,-1,-1,
    6,7,10,2,3,11,11,3,8,11,8,9,-1,-1,-1,-1,
    7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1,
    2,6,7,2,7,3,0,9,1,-1,-1,-1,-1,-1,-1,-1,
    1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1,
    11,6,7,11,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1,
    11,6,7,1,11,7,1,7,8,1,8,0,-1,-1,-1,-1,
    0,7,3,0,11,7,0,9,11,6,7,11,-1,-1,-1,-1,
    7,11,6,7,8,11,8,9,11,-1,-1,-1,-1,-1,-1,-1,
    6,4,8,10,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    3,10,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1,
    8,10,6,8,6,4,9,1,0,-1,-1,-1,-1,-1,-1,-1,
    9,6,4,9,3,6,9,1,3,10,6,3,-1,-1,-1,-1,
    6,4,8,6,8,10,2,1,11,-1,-1,-1,-1,-1,-1,-1,
    1,11,2,3,10,0,0,10,6,0,6,4,-1,-1,-1,-1,
    4,8,10,4,10,6,0,9,2,2,9,11,-1,-1,-1,-1,
    11,3,9,11,2,3,9,3,4,10,6,3,4,3,6,-1,
    8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,
    0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,0,9,2,4,3,2,6,4,4,8,3,-1,-1,-1,-1,
    1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1,
    8,3,1,8,1,6,8,6,4,6,1,11,-1,-1,-1,-1,
    11,0,1,11,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1,
    4,3,6,4,8,3,6,3,11,0,9,3,11,3,9,-1,
    11,4,9,6,4,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,5,9,7,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,3,8,4,5,9,10,6,7,-1,-1,-1,-1,-1,-1,-1,
    5,1,0,5,0,4,7,10,6,-1,-1,-1,-1,-1,-1,-1,
    10,6,7,8,4,3,3,4,5,3,5,1,-1,-1,-1,-1,
    9,4,5,11,2,1,7,10,6,-1,-1,-1,-1,-1,-1,-1,
    6,7,10,1,11,2,0,3,8,4,5,9,-1,-1,-1,-1,
    7,10,6,5,11,4,4,11,2,4,2,0,-1,-1,-1,-1,
    3,8,4,3,4,5,3,5,2,11,2,5,10,6,7,-1,
    7,3,2,7,2,6,5,9,4,-1,-1,-1,-1,-1,-1,-1,
    9,4,5,0,6,8,0,2,6,6,7,8,-1,-1,-1,-1,
    3,2,6,3,6,7,1,0,5,5,0,4,-1,-1,-1,-1,
    6,8,2,6,7,8,2,8,1,4,5,8,1,8,5,-1,
    9,4,5,11,6,1,1,6,7,1,7,3,-1,-1,-1,-1,
    1,11,6,1,6,7,1,7,0,8,0,7,9,4,5,-1,
    4,11,0,4,5,11,0,11,3,6,7,11,3,11,7,-1,
    7,11,6,7,8,11,5,11,4,4,11,8,-1,-1,-1,-1,
    6,5,9,6,9,10,10,9,8,-1,-1,-1,-1,-1,-1,-1,
    3,10,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1,
    0,8,10,0,10,5,0,5,1,5,10,6,-1,-1,-1,-1,
    6,3,10,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1,
    1,11,2,9,10,5,9,8,10,10,6,5,-1,-1,-1,-1,
    0,3,10,0,10,6,0,6,9,5,9,6,1,11,2,-1,
    10,5,8,10,6,5,8,5,0,11,2,5,0,5,2,-1,
    6,3,10,6,5,3,2,3,11,11,3,5,-1,-1,-1,-1,
    5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1,
    9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1,
    1,8,5,1,0,8,5,8,6,3,2,8,6,8,2,-1,
    1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,6,3,1,11,6,3,6,8,5,9,6,8,6,9,-1,
    11,0,1,11,6,0,9,0,5,5,0,6,-1,-1,-1,-1,
    0,8,3,5,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    11,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    10,11,5,7,10,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    10,11,5,10,5,7,8,0,3,-1,-1,-1,-1,-1,-1,-1,
    5,7,10,5,10,11,1,0,9,-1,-1,-1,-1,-1,-1,-1,
    11,5,7,11,7,10,9,1,8,8,1,3,-1,-1,-1,-1,
    10,2,1,10,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1,
    0,3,8,1,7,2,1,5,7,7,10,2,-1,-1,-1,-1,
    9,5,7,9,7,2,9,2,0,2,7,10,-1,-1,-1,-1,
    7,2,5,7,10,2,5,2,9,3,8,2,9,2,8,-1,
    2,11,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,
    8,0,2,8,2,5,8,5,7,11,5,2,-1,-1,-1,-1,
    9,1,0,5,3,11,5,7,3,3,2,11,-1,-1,-1,-1,
    9,2,8,9,1,2,8,2,7,11,5,2,7,2,5,-1,
    1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1,
    9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1,
    9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    5,4,8,5,8,11,11,8,10,-1,-1,-1,-1,-1,-1,-1,
    5,4,0,5,0,10,5,10,11,10,0,3,-1,-1,-1,-1,
    0,9,1,8,11,4,8,10,11,11,5,4,-1,-1,-1,-1,
    11,4,10,11,5,4,10,4,3,9,1,4,3,4,1,-1,
    2,1,5,2,5,8,2,8,10,4,8,5,-1,-1,-1,-1,
    0,10,4,0,3,10,4,10,5,2,1,10,5,10,1,-1,
    0,5,2,0,9,5,2,5,10,4,8,5,10,5,8,-1,
    9,5,4,2,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    2,11,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1,
    5,2,11,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1,
    3,2,11,3,11,5,3,5,8,4,8,5,0,9,1,-1,
    5,2,11,5,4,2,1,2,9,9,2,4,-1,-1,-1,-1,
    8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1,
    0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    8,5,4,8,3,5,9,5,0,0,5,3,-1,-1,-1,-1,
    9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,7,10,4,10,9,9,10,11,-1,-1,-1,-1,-1,-1,-1,
    0,3,8,4,7,9,9,7,10,9,10,11,-1,-1,-1,-1,
    1,10,11,1,4,10,1,0,4,7,10,4,-1,-1,-1,-1,
    3,4,1,3,8,4,1,4,11,7,10,4,11,4,10,-1,
    4,7,10,9,4,10,9,10,2,9,2,1,-1,-1,-1,-1,
    9,4,7,9,7,10,9,10,1,2,1,10,0,3,8,-1,
    10,4,7,10,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1,
    10,4,7,10,2,4,8,4,3,3,4,2,-1,-1,-1,-1,
    2,11,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1,
    9,7,11,9,4,7,11,7,2,8,0,7,2,7,0,-1,
    3,11,7,3,2,11,7,11,4,1,0,11,4,11,0,-1,
    1,2,11,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1,
    4,1,9,4,7,1,0,1,8,8,1,7,-1,-1,-1,-1,
    4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    9,8,11,11,8,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    3,9,0,3,10,9,10,11,9,-1,-1,-1,-1,-1,-1,-1,
    0,11,1,0,8,11,8,10,11,-1,-1,-1,-1,-1,-1,-1,
    3,11,1,10,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,10,2,1,9,10,9,8,10,-1,-1,-1,-1,-1,-1,-1,
    3,9,0,3,10,9,1,9,2,2,9,10,-1,-1,-1,-1,
    0,10,2,8,10,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    3,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    2,8,3,2,11,8,11,9,8,-1,-1,-1,-1,-1,-1,-1,
    9,2,11,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    2,8,3,2,11,8,0,8,1,1,8,11,-1,-1,-1,-1,
    1,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
], dtype=np.int32)
# fmt: on

_NUM_TRIS = np.zeros(256, dtype=np.int32)
for _case in range(256):
    _count = 0
    for _t in range(0, 16, 3):
        if _TRI_TABLE[_case * 16 + _t] >= 0:
            _count += 1
    _NUM_TRIS[_case] = _count


# ================================================================
# TEMPLATES
# ================================================================

@tack.data_oriented
class UniformGrid:
    """Uniform grid descriptor: origin + spacing."""

    def __init__(self, nx, ny, nz, x0, y0, z0, dx, dy, dz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nx_p1 = nx + 1
        self.ny_p1 = ny + 1
        self.nxy_p1 = (nx + 1) * (ny + 1)
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.dx = dx
        self.dy = dy
        self.dz = dz

    @tack.func
    def get_x(self, i, j, k):
        return self.x0 + float(i) * self.dx

    @tack.func
    def get_y(self, i, j, k):
        return self.y0 + float(j) * self.dy

    @tack.func
    def get_z(self, i, j, k):
        return self.z0 + float(k) * self.dz


@tack.data_oriented
class MCTables:
    """Marching cubes lookup tables as tack fields."""

    def __init__(self):
        self.tri_table = tack.field(dtype=tack.i32, shape=(4096,))
        self.tri_table.from_numpy(_TRI_TABLE)
        self.num_tris = tack.field(dtype=tack.i32, shape=(256,))
        self.num_tris.from_numpy(_NUM_TRIS)


# One table set per backend, created on first use.  Keying on the backend
# matters because tack.init() builds a fresh backend object and MCTables
# holds device fields — a single global would hand a CPU NumpyBuffer to
# Metal after a backend switch.  Weak keys let a retired backend's tables
# go with it.
_tables = weakref.WeakKeyDictionary()


def _get_tables():
    from tack.runtime.dispatch import get_backend

    backend = get_backend()
    tables = _tables.get(backend)
    if tables is None:
        tables = MCTables()
        _tables[backend] = tables
    return tables


# ================================================================
# HELPERS
# ================================================================

@tack.func
def _select12(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, idx):
    result = v0
    if idx == 1: result = v1
    if idx == 2: result = v2
    if idx == 3: result = v3
    if idx == 4: result = v4
    if idx == 5: result = v5
    if idx == 6: result = v6
    if idx == 7: result = v7
    if idx == 8: result = v8
    if idx == 9: result = v9
    if idx == 10: result = v10
    if idx == 11: result = v11
    return result


# ================================================================
# FLYING EDGES KERNELS
# ================================================================

@tack.kernel
def _fe_count_edges_xyz(scalar, row_xc, row_yc, row_zc,
                        grid: tack.template(), isovalue, n_node_rows):
    for row in range(n_node_rows):
        j = row % grid.ny_p1
        k = row // grid.ny_p1
        base = k * grid.nxy_p1 + j * grid.nx_p1
        xc = 0
        yc = 0
        zc = 0
        for i in range(grid.nx_p1):
            s = scalar[base + i]
            above = 0
            if s > isovalue:
                above = 1
            if i < grid.nx:
                s_nx = scalar[base + i + 1]
                a_nx = 0
                if s_nx > isovalue:
                    a_nx = 1
                if above != a_nx:
                    xc = xc + 1
            if j < grid.ny:
                s_ny = scalar[base + grid.nx_p1 + i]
                a_ny = 0
                if s_ny > isovalue:
                    a_ny = 1
                if above != a_ny:
                    yc = yc + 1
            if k < grid.nz:
                s_nz = scalar[base + grid.nxy_p1 + i]
                a_nz = 0
                if s_nz > isovalue:
                    a_nz = 1
                if above != a_nz:
                    zc = zc + 1
        row_xc[row] = xc
        row_yc[row] = yc
        row_zc[row] = zc


@tack.kernel
def _fe_count_rows(scalar, row_tri_count, mask, grid: tack.template(),
                   tables: tack.template(), isovalue, n_rows):
    """Count triangles per cell-row, skipping masked cells."""
    for row in range(n_rows):
        cj = row % grid.ny
        ck = row // grid.ny
        base00 = ck * grid.nxy_p1 + cj * grid.nx_p1
        base10 = base00 + grid.nx_p1
        base01 = base00 + grid.nxy_p1
        base11 = base10 + grid.nxy_p1
        count = 0
        s0 = scalar[base00]
        s3 = scalar[base10]
        s4 = scalar[base01]
        s7 = scalar[base11]
        for ci in range(grid.nx):
            s1 = scalar[base00 + ci + 1]
            s2 = scalar[base10 + ci + 1]
            s5 = scalar[base01 + ci + 1]
            s6 = scalar[base11 + ci + 1]
            cell_idx = ck * grid.ny * grid.nx + cj * grid.nx + ci
            if mask[cell_idx] == 0:
                case_idx = 0
                if s0 > isovalue: case_idx = case_idx + 1
                if s1 > isovalue: case_idx = case_idx + 2
                if s2 > isovalue: case_idx = case_idx + 4
                if s3 > isovalue: case_idx = case_idx + 8
                if s4 > isovalue: case_idx = case_idx + 16
                if s5 > isovalue: case_idx = case_idx + 32
                if s6 > isovalue: case_idx = case_idx + 64
                if s7 > isovalue: case_idx = case_idx + 128
                count = count + tables.num_tris[case_idx]
            s0 = s1
            s3 = s2
            s4 = s5
            s7 = s6
        row_tri_count[row] = count


@tack.kernel
def _fe_emit_points_xyz(scalar, row_xo, row_yo, row_zo,
                        points,
                        grid: tack.template(), isovalue, n_node_rows):
    """Emit unique interpolated points interleaved as [x,y,z,x,y,z,...]."""
    for row in range(n_node_rows):
        j = row % grid.ny_p1
        k = row // grid.ny_p1
        base = k * grid.nxy_p1 + j * grid.nx_p1
        xi = row_xo[row]
        yi = row_yo[row]
        zi = row_zo[row]
        for i in range(grid.nx_p1):
            node_idx = base + i
            s = scalar[node_idx]
            above = 0
            if s > isovalue:
                above = 1
            px = grid.get_x(i, j, k)
            py = grid.get_y(i, j, k)
            pz = grid.get_z(i, j, k)
            if i < grid.nx:
                s_nx = scalar[node_idx + 1]
                a_nx = 0
                if s_nx > isovalue:
                    a_nx = 1
                if above != a_nx:
                    t = (isovalue - s) / (s_nx - s + 1.0e-10)
                    px1 = grid.get_x(i + 1, j, k)
                    py1 = grid.get_y(i + 1, j, k)
                    pz1 = grid.get_z(i + 1, j, k)
                    points[xi * 3]     = px + t * (px1 - px)
                    points[xi * 3 + 1] = py + t * (py1 - py)
                    points[xi * 3 + 2] = pz + t * (pz1 - pz)
                    xi = xi + 1
            if j < grid.ny:
                s_ny = scalar[node_idx + grid.nx_p1]
                a_ny = 0
                if s_ny > isovalue:
                    a_ny = 1
                if above != a_ny:
                    t = (isovalue - s) / (s_ny - s + 1.0e-10)
                    px1 = grid.get_x(i, j + 1, k)
                    py1 = grid.get_y(i, j + 1, k)
                    pz1 = grid.get_z(i, j + 1, k)
                    points[yi * 3]     = px + t * (px1 - px)
                    points[yi * 3 + 1] = py + t * (py1 - py)
                    points[yi * 3 + 2] = pz + t * (pz1 - pz)
                    yi = yi + 1
            if k < grid.nz:
                s_nz = scalar[node_idx + grid.nxy_p1]
                a_nz = 0
                if s_nz > isovalue:
                    a_nz = 1
                if above != a_nz:
                    t = (isovalue - s) / (s_nz - s + 1.0e-10)
                    px1 = grid.get_x(i, j, k + 1)
                    py1 = grid.get_y(i, j, k + 1)
                    pz1 = grid.get_z(i, j, k + 1)
                    points[zi * 3]     = px + t * (px1 - px)
                    points[zi * 3 + 1] = py + t * (py1 - py)
                    points[zi * 3 + 2] = pz + t * (pz1 - pz)
                    zi = zi + 1


@tack.kernel
def _fe_emit_tris(scalar, row_tri_offset,
                  row_xo, row_yo, row_zo,
                  conn, mask,
                  grid: tack.template(), tables: tack.template(),
                  isovalue, n_cell_rows):
    """Emit triangle connectivity interleaved, skipping masked cells."""
    for row in range(n_cell_rows):
        cj = row % grid.ny
        ck = row // grid.ny
        r00 = ck * grid.ny_p1 + cj
        r10 = r00 + 1
        r01 = r00 + grid.ny_p1
        r11 = r01 + 1
        xo00 = row_xo[r00]
        yo00 = row_yo[r00]
        zo00 = row_zo[r00]
        xo10 = row_xo[r10]
        zo10 = row_zo[r10]
        xo01 = row_xo[r01]
        yo01 = row_yo[r01]
        xo11 = row_xo[r11]
        rx00 = 0
        ry00 = 0
        rz00 = 0
        rx10 = 0
        rz10 = 0
        rx01 = 0
        ry01 = 0
        rx11 = 0
        base00 = ck * grid.nxy_p1 + cj * grid.nx_p1
        base10 = base00 + grid.nx_p1
        base01 = base00 + grid.nxy_p1
        base11 = base10 + grid.nxy_p1
        tri_idx = row_tri_offset[row]
        s0 = scalar[base00]
        s3 = scalar[base10]
        s4 = scalar[base01]
        s7 = scalar[base11]
        for ci in range(grid.nx):
            s1 = scalar[base00 + ci + 1]
            s2 = scalar[base10 + ci + 1]
            s5 = scalar[base01 + ci + 1]
            s6 = scalar[base11 + ci + 1]
            a0 = 0
            a1 = 0
            a2 = 0
            a3 = 0
            a4 = 0
            a5 = 0
            a6 = 0
            a7 = 0
            if s0 > isovalue: a0 = 1
            if s1 > isovalue: a1 = 1
            if s2 > isovalue: a2 = 1
            if s3 > isovalue: a3 = 1
            if s4 > isovalue: a4 = 1
            if s5 > isovalue: a5 = 1
            if s6 > isovalue: a6 = 1
            if s7 > isovalue: a7 = 1
            case_idx = a0 + a1*2 + a2*4 + a3*8 + a4*16 + a5*32 + a6*64 + a7*128
            ix00 = 0
            iy00 = 0
            iz00 = 0
            if a0 != a1: ix00 = 1
            if a0 != a3: iy00 = 1
            if a0 != a4: iz00 = 1
            ix10 = 0
            iz10 = 0
            if a3 != a2: ix10 = 1
            if a3 != a7: iz10 = 1
            ix01 = 0
            iy01 = 0
            if a4 != a5: ix01 = 1
            if a4 != a7: iy01 = 1
            ix11 = 0
            if a7 != a6: ix11 = 1
            pid0  = xo00 + rx00
            pid1  = yo00 + ry00 + iy00
            pid2  = xo10 + rx10
            pid3  = yo00 + ry00
            pid4  = xo01 + rx01
            pid5  = yo01 + ry01 + iy01
            pid6  = xo11 + rx11
            pid7  = yo01 + ry01
            pid8  = zo00 + rz00
            pid9  = zo00 + rz00 + iz00
            pid10 = zo10 + rz10
            pid11 = zo10 + rz10 + iz10
            rx00 = rx00 + ix00
            ry00 = ry00 + iy00
            rz00 = rz00 + iz00
            rx10 = rx10 + ix10
            rz10 = rz10 + iz10
            rx01 = rx01 + ix01
            ry01 = ry01 + iy01
            rx11 = rx11 + ix11
            cell_idx = ck * grid.ny * grid.nx + cj * grid.nx + ci
            if mask[cell_idx] == 0:
                for t in range(0, 16, 3):
                    e0 = tables.tri_table[case_idx * 16 + t]
                    if e0 >= 0:
                        e1 = tables.tri_table[case_idx * 16 + t + 1]
                        e2 = tables.tri_table[case_idx * 16 + t + 2]
                        conn[tri_idx * 3]     = _select12(pid0, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10, pid11, e0)
                        conn[tri_idx * 3 + 1] = _select12(pid0, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10, pid11, e1)
                        conn[tri_idx * 3 + 2] = _select12(pid0, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10, pid11, e2)
                        tri_idx = tri_idx + 1
            s0 = s1
            s3 = s2
            s4 = s5
            s7 = s6


# ================================================================
# PUBLIC API
# ================================================================

def _count_block(scalar, grid, isovalue, mask):
    """Pass 1: count points and triangles for one block.

    Returns a dict with counts and intra-block row offsets (numpy).
    """
    tables = _get_tables()
    n_node_rows = (grid.ny + 1) * (grid.nz + 1)
    n_cell_rows = grid.ny * grid.nz

    row_xc = tack.field(dtype=tack.i32, shape=(n_node_rows,))
    row_yc = tack.field(dtype=tack.i32, shape=(n_node_rows,))
    row_zc = tack.field(dtype=tack.i32, shape=(n_node_rows,))
    _fe_count_edges_xyz(scalar, row_xc, row_yc, row_zc,
                        grid, isovalue, n_node_rows)

    row_tri_count = tack.field(dtype=tack.i32, shape=(n_cell_rows,))
    _fe_count_rows(scalar, row_tri_count, mask, grid, tables,
                   isovalue, n_cell_rows)

    xc_np = row_xc.to_numpy()
    yc_np = row_yc.to_numpy()
    zc_np = row_zc.to_numpy()
    total_x = int(np.sum(xc_np))
    total_y = int(np.sum(yc_np))
    total_z = int(np.sum(zc_np))
    total_points = total_x + total_y + total_z

    tri_np = row_tri_count.to_numpy()
    total_tris = int(np.sum(tri_np))

    # Intra-block prefix sums (local offsets starting at 0)
    xo_np = np.zeros(n_node_rows, dtype=np.int32)
    yo_np = np.zeros(n_node_rows, dtype=np.int32)
    zo_np = np.zeros(n_node_rows, dtype=np.int32)
    xo_np[1:] = np.cumsum(xc_np[:-1])
    yo_np[1:] = np.cumsum(yc_np[:-1])
    yo_np += total_x
    zo_np[1:] = np.cumsum(zc_np[:-1])
    zo_np += total_x + total_y

    tri_offsets_np = np.zeros(n_cell_rows, dtype=np.int32)
    tri_offsets_np[1:] = np.cumsum(tri_np[:-1])

    return {
        'n_node_rows': n_node_rows,
        'n_cell_rows': n_cell_rows,
        'total_points': total_points,
        'total_tris': total_tris,
        'xo_np': xo_np,
        'yo_np': yo_np,
        'zo_np': zo_np,
        'tri_offsets_np': tri_offsets_np,
    }


def _emit_block(scalar, grid, isovalue, mask, info, pt_offset, tri_offset,
                points, conn):
    """Pass 2: emit points and triangles for one block into shared output."""
    tables = _get_tables()

    xo_np = info['xo_np'] + pt_offset
    yo_np = info['yo_np'] + pt_offset
    zo_np = info['zo_np'] + pt_offset
    tri_off_np = info['tri_offsets_np'] + tri_offset

    row_xo = tack.field(dtype=tack.i32, shape=(info['n_node_rows'],))
    row_yo = tack.field(dtype=tack.i32, shape=(info['n_node_rows'],))
    row_zo = tack.field(dtype=tack.i32, shape=(info['n_node_rows'],))
    row_xo.from_numpy(xo_np)
    row_yo.from_numpy(yo_np)
    row_zo.from_numpy(zo_np)

    row_tri_offset = tack.field(dtype=tack.i32, shape=(info['n_cell_rows'],))
    row_tri_offset.from_numpy(tri_off_np)

    _fe_emit_points_xyz(scalar, row_xo, row_yo, row_zo,
                        points,
                        grid, isovalue, info['n_node_rows'])

    _fe_emit_tris(scalar, row_tri_offset,
                  row_xo, row_yo, row_zo,
                  conn, mask,
                  grid, tables,
                  isovalue, info['n_cell_rows'])


def flying_edges(scalar, grid, isovalue):
    """Isosurface a single uniform grid block.

    Args:
        scalar: tack.field of node scalars, shape ((nx+1)*(ny+1)*(nz+1),)
        grid: UniformGrid instance
        isovalue: scalar threshold

    Returns:
        dict with 'points' (numpy f32, shape (n,3) interleaved),
        'conn' (numpy i32, shape (m,3) interleaved),
        'total_points', 'total_tris', or None if empty.
    """
    return flying_edges_multiblock(
        [{'scalar': scalar, 'grid': grid}], isovalue)


def flying_edges_multiblock(blocks, isovalue):
    """Isosurface a list of uniform grid blocks into unified output.

    Args:
        blocks: list of dicts with 'scalar' (tack.field) and 'grid' (UniformGrid)
        isovalue: scalar threshold

    Returns:
        dict with 'points' (numpy f32, shape (n,3) interleaved x,y,z),
        'conn' (numpy i32, shape (m,3) interleaved v0,v1,v2),
        'total_points', 'total_tris', or None if empty.
    """
    n_blocks = len(blocks)
    if n_blocks == 0:
        return None

    # Ensure each block has a mask (all-zeros if not provided)
    for b in range(n_blocks):
        if 'mask' not in blocks[b]:
            n_cells = blocks[b]['grid'].nx * blocks[b]['grid'].ny * blocks[b]['grid'].nz
            mask = tack.field(dtype=tack.i32, shape=(n_cells,))
            mask.fill(0)
            blocks[b]['mask'] = mask

    # Pass 1: count per block
    infos = []
    for b in range(n_blocks):
        info = _count_block(blocks[b]['scalar'], blocks[b]['grid'], isovalue,
                            blocks[b]['mask'])
        infos.append(info)

    # Global prefix sum across blocks
    pt_offsets = np.zeros(n_blocks, dtype=np.int64)
    tri_offsets = np.zeros(n_blocks, dtype=np.int64)
    for b in range(1, n_blocks):
        pt_offsets[b] = pt_offsets[b-1] + infos[b-1]['total_points']
        tri_offsets[b] = tri_offsets[b-1] + infos[b-1]['total_tris']

    total_points = int(pt_offsets[-1] + infos[-1]['total_points'])
    total_tris = int(tri_offsets[-1] + infos[-1]['total_tris'])

    if total_points == 0 or total_tris == 0:
        return None

    # Allocate unified interleaved output
    points = tack.field(dtype=tack.f32, shape=(total_points * 3,))
    conn = tack.field(dtype=tack.i32, shape=(total_tris * 3,))

    # Pass 2: emit per block
    for b in range(n_blocks):
        if infos[b]['total_points'] == 0:
            continue
        _emit_block(blocks[b]['scalar'], blocks[b]['grid'], isovalue,
                    blocks[b]['mask'], infos[b],
                    int(pt_offsets[b]), int(tri_offsets[b]),
                    points, conn)

    return {
        'points': points.to_numpy().reshape(-1, 3),
        'conn': conn.to_numpy().reshape(-1, 3),
        'points_field': points,   # raw tack field for GPU interop
        'conn_field': conn,       # raw tack field for GPU interop
        'total_points': total_points,
        'total_tris': total_tris,
        'block_point_counts': [int(infos[b]['total_points']) for b in range(n_blocks)],
        'block_tri_counts': [int(infos[b]['total_tris']) for b in range(n_blocks)],
    }
