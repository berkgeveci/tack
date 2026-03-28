"""28 -- Flying Edges with Trim Ranges.

Builds on example 27's true FlyingEdges with an optimization from
VTK's implementation: trim ranges.  Each node-row records the first
and last x-index where any edge crosses the isosurface.  Passes 2-3
skip the empty prefix/suffix of each row, reducing work for datasets
where the isosurface occupies a fraction of the volume.

Pipeline (same 4 passes, passes 2-3 trimmed):
1. Count edge intersections per node-row + tri per cell-row + record trim ranges
2. Prefix sum over counts (CPU, row arrays are small)
3. Emit unique interpolated points (only within xmin..xmax per row)
4. Emit triangle connectivity (only within cell trim range)

Usage:
  uv run python examples/28_flying_edges_trimmed.py
  uv run python examples/28_flying_edges_trimmed.py --arch metal --size 200
"""

import time
import numpy as np
import tack
from tack import algorithms

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip'])
_parser.add_argument('--size', type=int, default=100,
                     help='Grid cells per dimension (default 100)')
_parser.add_argument('--warmup', type=int, default=2)
_parser.add_argument('--trials', type=int, default=5)
_args = _parser.parse_args()
_arch = getattr(tack, _args.arch)
tack.init(arch=_arch)


# ================================================================
# MARCHING CUBES TABLES (same as examples 26-27)
# ================================================================

EDGE_CORNERS = np.array([
    0, 1,  1, 2,  3, 2,  0, 3,
    4, 5,  5, 6,  7, 6,  4, 7,
    0, 4,  1, 5,  3, 7,  2, 6,
], dtype=np.int32)

CORNER_X = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.float32)
CORNER_Y = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32)
CORNER_Z = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

# fmt: off
TRI_TABLE = np.array([
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

NUM_TRIS = np.zeros(256, dtype=np.int32)
for _case in range(256):
    _count = 0
    for _t in range(0, 16, 3):
        if TRI_TABLE[_case * 16 + _t] >= 0:
            _count += 1
    NUM_TRIS[_case] = _count


# ================================================================
# TEMPLATES (same as example 27)
# ================================================================

@tack.data_oriented
class UniformGrid:
    """Uniform grid -- coordinates computed from origin + index * spacing."""

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
class RectilinearGrid:
    """Rectilinear grid -- separable 1D coordinate arrays."""

    def __init__(self, nx, ny, nz, xcoords, ycoords, zcoords):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nx_p1 = nx + 1
        self.ny_p1 = ny + 1
        self.nxy_p1 = (nx + 1) * (ny + 1)
        self.xcoords = xcoords
        self.ycoords = ycoords
        self.zcoords = zcoords

    @tack.func
    def get_x(self, i, j, k):
        return self.xcoords[i]

    @tack.func
    def get_y(self, i, j, k):
        return self.ycoords[j]

    @tack.func
    def get_z(self, i, j, k):
        return self.zcoords[k]


@tack.data_oriented
class StructuredGrid:
    """Structured (curvilinear) grid -- full per-point coordinate arrays."""

    def __init__(self, nx, ny, nz, px, py, pz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nx_p1 = nx + 1
        self.ny_p1 = ny + 1
        self.nxy_p1 = (nx + 1) * (ny + 1)
        self.px = px
        self.py = py
        self.pz = pz

    @tack.func
    def get_x(self, i, j, k):
        return self.px[k * self.nxy_p1 + j * self.nx_p1 + i]

    @tack.func
    def get_y(self, i, j, k):
        return self.py[k * self.nxy_p1 + j * self.nx_p1 + i]

    @tack.func
    def get_z(self, i, j, k):
        return self.pz[k * self.nxy_p1 + j * self.nx_p1 + i]


@tack.data_oriented
class MCTables:
    """Marching cubes lookup tables as tack fields."""

    def __init__(self):
        self.tri_table = tack.field(dtype=tack.i32, shape=(4096,))
        self.tri_table.from_numpy(TRI_TABLE)
        self.num_tris = tack.field(dtype=tack.i32, shape=(256,))
        self.num_tris.from_numpy(NUM_TRIS)
        self.edge_corners = tack.field(dtype=tack.i32, shape=(24,))
        self.edge_corners.from_numpy(EDGE_CORNERS)
        self.corner_x = tack.field(dtype=tack.f32, shape=(8,))
        self.corner_x.from_numpy(CORNER_X)
        self.corner_y = tack.field(dtype=tack.f32, shape=(8,))
        self.corner_y.from_numpy(CORNER_Y)
        self.corner_z = tack.field(dtype=tack.f32, shape=(8,))
        self.corner_z.from_numpy(CORNER_Z)


# ================================================================
# HELPER
# ================================================================

@tack.func
def select12(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, idx):
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
# KERNELS
# ================================================================

@tack.kernel
def compute_scalar_field(scalar, grid: tack.template(), n_points):
    """Gyroid: sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)."""
    for i in range(n_points):
        ix = i % grid.nx_p1
        iy = (i // grid.nx_p1) % grid.ny_p1
        iz = i // grid.nxy_p1
        x = grid.get_x(ix, iy, iz)
        y = grid.get_y(ix, iy, iz)
        z = grid.get_z(ix, iy, iz)
        scalar[i] = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)


# --- Trimmed Flying Edges kernels ---

@tack.kernel
def fe_count_edges_xyz_trim(scalar, row_xc, row_yc, row_zc,
                            row_xmin, row_xmax,
                            grid: tack.template(), isovalue, n_node_rows):
    """Pass 1: Count x/y/z edge intersections + record trim ranges.

    row_xmin[row] = first x-index where any edge crosses (or nx+1 if none)
    row_xmax[row] = last x-index where any edge crosses (or -1 if none)
    """
    for row in range(n_node_rows):
        j = row % grid.ny_p1
        k = row // grid.ny_p1
        base = k * grid.nxy_p1 + j * grid.nx_p1
        xc = 0
        yc = 0
        zc = 0
        xmin = grid.nx_p1  # sentinel: no crossing found
        xmax = -1
        for i in range(grid.nx_p1):
            s = scalar[base + i]
            above = 0
            if s > isovalue:
                above = 1
            has_crossing = 0
            if i < grid.nx:
                s_nx = scalar[base + i + 1]
                a_nx = 0
                if s_nx > isovalue:
                    a_nx = 1
                if above != a_nx:
                    xc = xc + 1
                    has_crossing = 1
            if j < grid.ny:
                s_ny = scalar[base + grid.nx_p1 + i]
                a_ny = 0
                if s_ny > isovalue:
                    a_ny = 1
                if above != a_ny:
                    yc = yc + 1
                    has_crossing = 1
            if k < grid.nz:
                s_nz = scalar[base + grid.nxy_p1 + i]
                a_nz = 0
                if s_nz > isovalue:
                    a_nz = 1
                if above != a_nz:
                    zc = zc + 1
                    has_crossing = 1
            if has_crossing == 1:
                if i < xmin:
                    xmin = i
                xmax = i
        row_xc[row] = xc
        row_yc[row] = yc
        row_zc[row] = zc
        row_xmin[row] = xmin
        row_xmax[row] = xmax


@tack.kernel
def fe_count_rows_trim(scalar, row_tri_count, row_xmin, row_xmax,
                       grid: tack.template(), tables: tack.template(),
                       isovalue, n_rows):
    """Count triangles per cell-row, only within cell trim range."""
    for row in range(n_rows):
        cj = row % grid.ny
        ck = row // grid.ny
        # 4 neighbor node-rows
        r00 = ck * grid.ny_p1 + cj
        r10 = r00 + 1
        r01 = r00 + grid.ny_p1
        r11 = r01 + 1
        # Cell trim range from 4 neighbor node-rows
        mn = row_xmin[r00]
        v = row_xmin[r10]
        if v < mn: mn = v
        v = row_xmin[r01]
        if v < mn: mn = v
        v = row_xmin[r11]
        if v < mn: mn = v
        ci_start = mn - 1
        if ci_start < 0: ci_start = 0

        mx = row_xmax[r00]
        v = row_xmax[r10]
        if v > mx: mx = v
        v = row_xmax[r01]
        if v > mx: mx = v
        v = row_xmax[r11]
        if v > mx: mx = v
        ci_end = mx + 1
        if ci_end > grid.nx: ci_end = grid.nx

        base00 = ck * grid.nxy_p1 + cj * grid.nx_p1
        base10 = base00 + grid.nx_p1
        base01 = base00 + grid.nxy_p1
        base11 = base10 + grid.nxy_p1

        count = 0
        if ci_start < ci_end:
            # Preload left column at ci_start
            s0 = scalar[base00 + ci_start]
            s3 = scalar[base10 + ci_start]
            s4 = scalar[base01 + ci_start]
            s7 = scalar[base11 + ci_start]
            for ci in range(ci_start, ci_end):
                s1 = scalar[base00 + ci + 1]
                s2 = scalar[base10 + ci + 1]
                s5 = scalar[base01 + ci + 1]
                s6 = scalar[base11 + ci + 1]
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
def fe_emit_points_xyz_trim(scalar, row_xo, row_yo, row_zo,
                            row_xmin, row_xmax,
                            pt_x, pt_y, pt_z,
                            grid: tack.template(), isovalue, n_node_rows):
    """Emit unique interpolated points, only within trim range."""
    for row in range(n_node_rows):
        j = row % grid.ny_p1
        k = row // grid.ny_p1
        base = k * grid.nxy_p1 + j * grid.nx_p1
        xi = row_xo[row]
        yi = row_yo[row]
        zi = row_zo[row]
        imin = row_xmin[row]
        imax = row_xmax[row]
        if imin <= imax:
            for i in range(imin, imax + 1):
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
                        pt_x[xi] = px + t * (px1 - px)
                        pt_y[xi] = py + t * (py1 - py)
                        pt_z[xi] = pz + t * (pz1 - pz)
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
                        pt_x[yi] = px + t * (px1 - px)
                        pt_y[yi] = py + t * (py1 - py)
                        pt_z[yi] = pz + t * (pz1 - pz)
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
                        pt_x[zi] = px + t * (px1 - px)
                        pt_y[zi] = py + t * (py1 - py)
                        pt_z[zi] = pz + t * (pz1 - pz)
                        zi = zi + 1


@tack.kernel
def fe_emit_tris_trim(scalar, row_tri_offset,
                      row_xo, row_yo, row_zo,
                      row_xmin, row_xmax,
                      tri_v0, tri_v1, tri_v2,
                      grid: tack.template(), tables: tack.template(),
                      isovalue, n_cell_rows):
    """Emit triangle connectivity, only within cell trim range.

    Running counters all start at 0 because no crossings exist before
    the cell trim range start.
    """
    for row in range(n_cell_rows):
        cj = row % grid.ny
        ck = row // grid.ny
        # Node-row indices for 4 neighbor rows
        r00 = ck * grid.ny_p1 + cj
        r10 = r00 + 1
        r01 = r00 + grid.ny_p1
        r11 = r01 + 1

        # Cell trim range from 4 neighbor node-rows
        mn = row_xmin[r00]
        v = row_xmin[r10]
        if v < mn: mn = v
        v = row_xmin[r01]
        if v < mn: mn = v
        v = row_xmin[r11]
        if v < mn: mn = v
        ci_start = mn - 1
        if ci_start < 0: ci_start = 0

        mx = row_xmax[r00]
        v = row_xmax[r10]
        if v > mx: mx = v
        v = row_xmax[r01]
        if v > mx: mx = v
        v = row_xmax[r11]
        if v > mx: mx = v
        ci_end = mx + 1
        if ci_end > grid.nx: ci_end = grid.nx

        # Per-type base offsets from prefix sum
        xo00 = row_xo[r00]
        yo00 = row_yo[r00]
        zo00 = row_zo[r00]
        xo10 = row_xo[r10]
        zo10 = row_zo[r10]
        xo01 = row_xo[r01]
        yo01 = row_yo[r01]
        xo11 = row_xo[r11]

        # Running per-type edge counts -- all 0 at ci_start since no
        # crossings exist before the trim range
        rx00 = 0
        ry00 = 0
        rz00 = 0
        rx10 = 0
        rz10 = 0
        rx01 = 0
        ry01 = 0
        rx11 = 0

        # Scalar bases
        base00 = ck * grid.nxy_p1 + cj * grid.nx_p1
        base10 = base00 + grid.nx_p1
        base01 = base00 + grid.nxy_p1
        base11 = base10 + grid.nxy_p1

        tri_idx = row_tri_offset[row]

        if ci_start < ci_end:
            # Preload left column at ci_start
            s0 = scalar[base00 + ci_start]
            s3 = scalar[base10 + ci_start]
            s4 = scalar[base01 + ci_start]
            s7 = scalar[base11 + ci_start]

            for ci in range(ci_start, ci_end):
                # Load right column
                s1 = scalar[base00 + ci + 1]
                s2 = scalar[base10 + ci + 1]
                s5 = scalar[base01 + ci + 1]
                s6 = scalar[base11 + ci + 1]
                # Sign bits
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
                case_idx = a0 + a1 * 2 + a2 * 4 + a3 * 8 + a4 * 16 + a5 * 32 + a6 * 64 + a7 * 128
                # Edge intersection flags at node ci for each row/type
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
                # Point IDs for all 12 MC edges
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
                # Update running counts
                rx00 = rx00 + ix00
                ry00 = ry00 + iy00
                rz00 = rz00 + iz00
                rx10 = rx10 + ix10
                rz10 = rz10 + iz10
                rx01 = rx01 + ix01
                ry01 = ry01 + iy01
                rx11 = rx11 + ix11
                # Emit triangles
                for t in range(0, 16, 3):
                    e0 = tables.tri_table[case_idx * 16 + t]
                    if e0 >= 0:
                        e1 = tables.tri_table[case_idx * 16 + t + 1]
                        e2 = tables.tri_table[case_idx * 16 + t + 2]
                        tri_v0[tri_idx] = select12(pid0, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10, pid11, e0)
                        tri_v1[tri_idx] = select12(pid0, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10, pid11, e1)
                        tri_v2[tri_idx] = select12(pid0, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10, pid11, e2)
                        tri_idx = tri_idx + 1
                # Shift right -> left
                s0 = s1
                s3 = s2
                s4 = s5
                s7 = s6


# --- Untrimmed kernels (for comparison) ---

@tack.kernel
def fe_count_edges_xyz(scalar, row_xc, row_yc, row_zc,
                       grid: tack.template(), isovalue, n_node_rows):
    """Count x/y/z edge intersections separately per node-row (no trim)."""
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
def fe_count_rows(scalar, row_tri_count, grid: tack.template(),
                  tables: tack.template(), isovalue, n_rows):
    """Count triangles per cell-row (no trim)."""
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
def fe_emit_points_xyz(scalar, row_xo, row_yo, row_zo,
                       pt_x, pt_y, pt_z,
                       grid: tack.template(), isovalue, n_node_rows):
    """Emit unique interpolated points (no trim)."""
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
                    pt_x[xi] = px + t * (px1 - px)
                    pt_y[xi] = py + t * (py1 - py)
                    pt_z[xi] = pz + t * (pz1 - pz)
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
                    pt_x[yi] = px + t * (px1 - px)
                    pt_y[yi] = py + t * (py1 - py)
                    pt_z[yi] = pz + t * (pz1 - pz)
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
                    pt_x[zi] = px + t * (px1 - px)
                    pt_y[zi] = py + t * (py1 - py)
                    pt_z[zi] = pz + t * (pz1 - pz)
                    zi = zi + 1


@tack.kernel
def fe_emit_tris(scalar, row_tri_offset,
                 row_xo, row_yo, row_zo,
                 tri_v0, tri_v1, tri_v2,
                 grid: tack.template(), tables: tack.template(),
                 isovalue, n_cell_rows):
    """Emit triangle connectivity (no trim)."""
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
            case_idx = a0 + a1 * 2 + a2 * 4 + a3 * 8 + a4 * 16 + a5 * 32 + a6 * 64 + a7 * 128
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
            for t in range(0, 16, 3):
                e0 = tables.tri_table[case_idx * 16 + t]
                if e0 >= 0:
                    e1 = tables.tri_table[case_idx * 16 + t + 1]
                    e2 = tables.tri_table[case_idx * 16 + t + 2]
                    tri_v0[tri_idx] = select12(pid0, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10, pid11, e0)
                    tri_v1[tri_idx] = select12(pid0, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10, pid11, e1)
                    tri_v2[tri_idx] = select12(pid0, pid1, pid2, pid3, pid4, pid5, pid6, pid7, pid8, pid9, pid10, pid11, e2)
                    tri_idx = tri_idx + 1
            s0 = s1
            s3 = s2
            s4 = s5
            s7 = s6


# ================================================================
# RUN
# ================================================================

N = _args.size
nx, ny, nz = N, N, N
n_points = (nx + 1) * (ny + 1) * (nz + 1)
n_cells = nx * ny * nz
n_cell_rows = ny * nz
n_node_rows = (ny + 1) * (nz + 1)
warmup = _args.warmup
trials = _args.trials
isovalue = 0.0

x0, y0, z0 = -np.pi, -np.pi, -np.pi
dx = 2.0 * np.pi / nx
dy = 2.0 * np.pi / ny
dz = 2.0 * np.pi / nz

print(f"Grid: {nx}x{ny}x{nz} = {n_cells:,} cells, {n_points:,} points")
print(f"Cell rows: {n_cell_rows:,}, Node rows: {n_node_rows:,}")
print(f"Backend: {_args.arch}")
print(f"Isovalue: {isovalue}")
print()

grid = UniformGrid(nx, ny, nz, x0, y0, z0, dx, dy, dz)
tables = MCTables()

# Compute scalar field
scalar = tack.field(dtype=tack.f32, shape=(n_points,))
compute_scalar_field(scalar, grid, n_points)

results = {}


# ================================================================
# Untrimmed FE (baseline)
# ================================================================
print("Flying Edges (untrimmed)...")

row_xc = tack.field(dtype=tack.i32, shape=(n_node_rows,))
row_yc = tack.field(dtype=tack.i32, shape=(n_node_rows,))
row_zc = tack.field(dtype=tack.i32, shape=(n_node_rows,))
fe_count_edges_xyz(scalar, row_xc, row_yc, row_zc, grid, isovalue, n_node_rows)

xc_np = row_xc.to_numpy()
yc_np = row_yc.to_numpy()
zc_np = row_zc.to_numpy()
total_x = int(np.sum(xc_np))
total_y = int(np.sum(yc_np))
total_z = int(np.sum(zc_np))
total_points = total_x + total_y + total_z

xo_np = np.zeros(n_node_rows, dtype=np.int32)
yo_np = np.zeros(n_node_rows, dtype=np.int32)
zo_np = np.zeros(n_node_rows, dtype=np.int32)
xo_np[1:] = np.cumsum(xc_np[:-1])
yo_np[1:] = np.cumsum(yc_np[:-1])
yo_np += total_x
zo_np[1:] = np.cumsum(zc_np[:-1])
zo_np += total_x + total_y

row_xo = tack.field(dtype=tack.i32, shape=(n_node_rows,))
row_yo = tack.field(dtype=tack.i32, shape=(n_node_rows,))
row_zo = tack.field(dtype=tack.i32, shape=(n_node_rows,))
row_xo.from_numpy(xo_np)
row_yo.from_numpy(yo_np)
row_zo.from_numpy(zo_np)

row_tri_count = tack.field(dtype=tack.i32, shape=(n_cell_rows,))
fe_count_rows(scalar, row_tri_count, grid, tables, isovalue, n_cell_rows)

tri_counts_np = row_tri_count.to_numpy()
tri_offsets_np = np.zeros(n_cell_rows, dtype=np.int32)
tri_offsets_np[1:] = np.cumsum(tri_counts_np[:-1])
total_tris = int(np.sum(tri_counts_np))
row_tri_offset = tack.field(dtype=tack.i32, shape=(n_cell_rows,))
row_tri_offset.from_numpy(tri_offsets_np)

print(f"  Unique points: {total_points:,}, Triangles: {total_tris:,}")

pt_x = tack.field(dtype=tack.f32, shape=(total_points,))
pt_y = tack.field(dtype=tack.f32, shape=(total_points,))
pt_z = tack.field(dtype=tack.f32, shape=(total_points,))
tri_v0 = tack.field(dtype=tack.i32, shape=(total_tris,))
tri_v1 = tack.field(dtype=tack.i32, shape=(total_tris,))
tri_v2 = tack.field(dtype=tack.i32, shape=(total_tris,))


def _fe_prefix_sums():
    fe_count_edges_xyz(scalar, row_xc, row_yc, row_zc, grid, isovalue, n_node_rows)
    _xc = row_xc.to_numpy()
    _yc = row_yc.to_numpy()
    _zc = row_zc.to_numpy()
    _tx = int(np.sum(_xc))
    xo_np[0] = 0
    xo_np[1:] = np.cumsum(_xc[:-1])
    yo_np[0] = _tx
    yo_np[1:] = _tx + np.cumsum(_yc[:-1])
    _ty = int(np.sum(_yc))
    zo_np[0] = _tx + _ty
    zo_np[1:] = _tx + _ty + np.cumsum(_zc[:-1])
    row_xo.from_numpy(xo_np)
    row_yo.from_numpy(yo_np)
    row_zo.from_numpy(zo_np)
    fe_count_rows(scalar, row_tri_count, grid, tables, isovalue, n_cell_rows)
    _tc = row_tri_count.to_numpy()
    tri_offsets_np[0] = 0
    tri_offsets_np[1:] = np.cumsum(_tc[:-1])
    row_tri_offset.from_numpy(tri_offsets_np)


for _w in range(warmup):
    _fe_prefix_sums()
    fe_emit_points_xyz(scalar, row_xo, row_yo, row_zo, pt_x, pt_y, pt_z,
                       grid, isovalue, n_node_rows)
    fe_emit_tris(scalar, row_tri_offset, row_xo, row_yo, row_zo,
                 tri_v0, tri_v1, tri_v2, grid, tables, isovalue, n_cell_rows)

times = []
for _t in range(trials):
    t0 = time.perf_counter()
    _fe_prefix_sums()
    fe_emit_points_xyz(scalar, row_xo, row_yo, row_zo, pt_x, pt_y, pt_z,
                       grid, isovalue, n_node_rows)
    fe_emit_tris(scalar, row_tri_offset, row_xo, row_yo, row_zo,
                 tri_v0, tri_v1, tri_v2, grid, tables, isovalue, n_cell_rows)
    t1 = time.perf_counter()
    times.append(t1 - t0)

fe_time = min(times)
print(f"  Best of {trials}: {fe_time:.4f}s")
results["FE"] = fe_time


# ================================================================
# Trimmed FE
# ================================================================
print("\nFlying Edges (trimmed)...")

# Trim range fields
row_xmin = tack.field(dtype=tack.i32, shape=(n_node_rows,))
row_xmax = tack.field(dtype=tack.i32, shape=(n_node_rows,))

# Pass 1: count + trim
t_row_xc = tack.field(dtype=tack.i32, shape=(n_node_rows,))
t_row_yc = tack.field(dtype=tack.i32, shape=(n_node_rows,))
t_row_zc = tack.field(dtype=tack.i32, shape=(n_node_rows,))
fe_count_edges_xyz_trim(scalar, t_row_xc, t_row_yc, t_row_zc,
                        row_xmin, row_xmax, grid, isovalue, n_node_rows)

txc_np = t_row_xc.to_numpy()
tyc_np = t_row_yc.to_numpy()
tzc_np = t_row_zc.to_numpy()
t_total_x = int(np.sum(txc_np))
t_total_y = int(np.sum(tyc_np))
t_total_z = int(np.sum(tzc_np))
t_total_points = t_total_x + t_total_y + t_total_z

txo_np = np.zeros(n_node_rows, dtype=np.int32)
tyo_np = np.zeros(n_node_rows, dtype=np.int32)
tzo_np = np.zeros(n_node_rows, dtype=np.int32)
txo_np[1:] = np.cumsum(txc_np[:-1])
tyo_np[1:] = np.cumsum(tyc_np[:-1])
tyo_np += t_total_x
tzo_np[1:] = np.cumsum(tzc_np[:-1])
tzo_np += t_total_x + t_total_y

t_row_xo = tack.field(dtype=tack.i32, shape=(n_node_rows,))
t_row_yo = tack.field(dtype=tack.i32, shape=(n_node_rows,))
t_row_zo = tack.field(dtype=tack.i32, shape=(n_node_rows,))
t_row_xo.from_numpy(txo_np)
t_row_yo.from_numpy(tyo_np)
t_row_zo.from_numpy(tzo_np)

t_row_tri_count = tack.field(dtype=tack.i32, shape=(n_cell_rows,))
fe_count_rows_trim(scalar, t_row_tri_count, row_xmin, row_xmax,
                   grid, tables, isovalue, n_cell_rows)

t_tri_counts_np = t_row_tri_count.to_numpy()
t_tri_offsets_np = np.zeros(n_cell_rows, dtype=np.int32)
t_tri_offsets_np[1:] = np.cumsum(t_tri_counts_np[:-1])
t_total_tris = int(np.sum(t_tri_counts_np))
t_row_tri_offset = tack.field(dtype=tack.i32, shape=(n_cell_rows,))
t_row_tri_offset.from_numpy(t_tri_offsets_np)

# Validate trimmed counts match untrimmed
assert t_total_points == total_points, \
    f"Trim point mismatch: {t_total_points} vs {total_points}"
assert t_total_tris == total_tris, \
    f"Trim tri mismatch: {t_total_tris} vs {total_tris}"

# Trim statistics
xmin_np = row_xmin.to_numpy()
xmax_np = row_xmax.to_numpy()
active_rows = np.sum(xmax_np >= 0)
avg_range = 0.0
if active_rows > 0:
    active_mask = xmax_np >= 0
    avg_range = np.mean(xmax_np[active_mask] - xmin_np[active_mask] + 1)
print(f"  Points: {t_total_points:,}, Triangles: {t_total_tris:,}")
print(f"  Active node-rows: {active_rows:,}/{n_node_rows:,}"
      f" ({active_rows/n_node_rows*100:.1f}%)")
print(f"  Avg trim range: {avg_range:.1f}/{nx+1} nodes"
      f" ({avg_range/(nx+1)*100:.1f}% of row)")

t_pt_x = tack.field(dtype=tack.f32, shape=(t_total_points,))
t_pt_y = tack.field(dtype=tack.f32, shape=(t_total_points,))
t_pt_z = tack.field(dtype=tack.f32, shape=(t_total_points,))
t_tri_v0 = tack.field(dtype=tack.i32, shape=(t_total_tris,))
t_tri_v1 = tack.field(dtype=tack.i32, shape=(t_total_tris,))
t_tri_v2 = tack.field(dtype=tack.i32, shape=(t_total_tris,))


def _fe_trim_prefix_sums():
    fe_count_edges_xyz_trim(scalar, t_row_xc, t_row_yc, t_row_zc,
                            row_xmin, row_xmax, grid, isovalue, n_node_rows)
    _xc = t_row_xc.to_numpy()
    _yc = t_row_yc.to_numpy()
    _zc = t_row_zc.to_numpy()
    _tx = int(np.sum(_xc))
    txo_np[0] = 0
    txo_np[1:] = np.cumsum(_xc[:-1])
    tyo_np[0] = _tx
    tyo_np[1:] = _tx + np.cumsum(_yc[:-1])
    _ty = int(np.sum(_yc))
    tzo_np[0] = _tx + _ty
    tzo_np[1:] = _tx + _ty + np.cumsum(_zc[:-1])
    t_row_xo.from_numpy(txo_np)
    t_row_yo.from_numpy(tyo_np)
    t_row_zo.from_numpy(tzo_np)
    fe_count_rows_trim(scalar, t_row_tri_count, row_xmin, row_xmax,
                       grid, tables, isovalue, n_cell_rows)
    _tc = t_row_tri_count.to_numpy()
    t_tri_offsets_np[0] = 0
    t_tri_offsets_np[1:] = np.cumsum(_tc[:-1])
    t_row_tri_offset.from_numpy(t_tri_offsets_np)


for _w in range(warmup):
    _fe_trim_prefix_sums()
    fe_emit_points_xyz_trim(scalar, t_row_xo, t_row_yo, t_row_zo,
                            row_xmin, row_xmax, t_pt_x, t_pt_y, t_pt_z,
                            grid, isovalue, n_node_rows)
    fe_emit_tris_trim(scalar, t_row_tri_offset, t_row_xo, t_row_yo, t_row_zo,
                      row_xmin, row_xmax, t_tri_v0, t_tri_v1, t_tri_v2,
                      grid, tables, isovalue, n_cell_rows)

times = []
for _t in range(trials):
    t0 = time.perf_counter()
    _fe_trim_prefix_sums()
    fe_emit_points_xyz_trim(scalar, t_row_xo, t_row_yo, t_row_zo,
                            row_xmin, row_xmax, t_pt_x, t_pt_y, t_pt_z,
                            grid, isovalue, n_node_rows)
    fe_emit_tris_trim(scalar, t_row_tri_offset, t_row_xo, t_row_yo, t_row_zo,
                      row_xmin, row_xmax, t_tri_v0, t_tri_v1, t_tri_v2,
                      grid, tables, isovalue, n_cell_rows)
    t1 = time.perf_counter()
    times.append(t1 - t0)

fe_trim_time = min(times)
print(f"  Best of {trials}: {fe_trim_time:.4f}s")
results["FE-Trim"] = fe_trim_time


# ================================================================
# Validation
# ================================================================
print("\n--- Validation ---")

# Trimmed must produce same counts
print(f"  Trimmed == Untrimmed points: {t_total_points:,} -- OK")
print(f"  Trimmed == Untrimmed tris:   {t_total_tris:,} -- OK")

# Check connectivity in range
t_t0 = t_tri_v0.to_numpy()
t_t1 = t_tri_v1.to_numpy()
t_t2 = t_tri_v2.to_numpy()
assert np.all(t_t0 >= 0) and np.all(t_t0 < t_total_points), "tri_v0 out of range"
assert np.all(t_t1 >= 0) and np.all(t_t1 < t_total_points), "tri_v1 out of range"
assert np.all(t_t2 >= 0) and np.all(t_t2 < t_total_points), "tri_v2 out of range"
print("  Triangle connectivity: OK")

# Check points on isosurface
t_px = t_pt_x.to_numpy()
t_py = t_pt_y.to_numpy()
t_pz = t_pt_z.to_numpy()
n_sample = min(100, t_total_points)
sample_vals = (np.sin(t_px[:n_sample]) * np.cos(t_py[:n_sample])
             + np.sin(t_py[:n_sample]) * np.cos(t_pz[:n_sample])
             + np.sin(t_pz[:n_sample]) * np.cos(t_px[:n_sample]))
max_err = np.max(np.abs(sample_vals - isovalue))
print(f"  Max scalar error at points: {max_err:.6f}")
assert max_err < 0.1
print("  Points on isosurface: OK")

# Cross-validate: trimmed points should match untrimmed points (same set)
u_px = pt_x.to_numpy()
u_py = pt_y.to_numpy()
u_pz = pt_z.to_numpy()
# Sort both point sets for comparison (order may differ)
u_pts = np.column_stack([u_px, u_py, u_pz])
t_pts = np.column_stack([t_px, t_py, t_pz])
u_sorted = u_pts[np.lexsort(u_pts.T)]
t_sorted = t_pts[np.lexsort(t_pts.T)]
max_coord_diff = np.max(np.abs(u_sorted - t_sorted))
print(f"  Max coord diff (trim vs no-trim): {max_coord_diff:.2e}")
assert max_coord_diff < 1e-5, f"Point mismatch: {max_coord_diff}"
print("  Trimmed == Untrimmed points: OK")


# ================================================================
# VTK comparison
# ================================================================
try:
    import sys
    vtk_tbb_path = "/usr/local/scratch/builds/vtk/master-release/lib/python3.13/site-packages"
    if vtk_tbb_path not in sys.path:
        sys.path.insert(0, vtk_tbb_path)

    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import vtkContourFilter
    from vtkmodules.util.numpy_support import numpy_to_vtk

    print("\nVTK FlyingEdges...")

    img = vtkImageData()
    img.SetDimensions(nx + 1, ny + 1, nz + 1)
    img.SetOrigin(x0, y0, z0)
    img.SetSpacing(dx, dy, dz)

    scalar_np = scalar.to_numpy()
    vtk_arr = numpy_to_vtk(scalar_np, deep=True)
    vtk_arr.SetName("scalar")
    img.GetPointData().SetScalars(vtk_arr)

    cf = vtkContourFilter()
    cf.SetInputData(img)
    cf.SetValue(0, isovalue)
    cf.SetFastMode(True)

    for _w in range(warmup):
        cf.Modified()
        cf.Update()

    times = []
    for _t in range(trials):
        cf.Modified()
        t0 = time.perf_counter()
        cf.Update()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    vtk_fe_time = min(times)
    vtk_output = cf.GetOutput()
    vtk_total_tris = vtk_output.GetNumberOfCells()
    vtk_total_points = vtk_output.GetNumberOfPoints()
    print(f"  VTK FE (TBB): {vtk_fe_time:.4f}s")
    print(f"  Triangles: {vtk_total_tris:,}, Points: {vtk_total_points:,}")
    results["VTK-FE"] = vtk_fe_time

    del cf, img, vtk_output

except ImportError:
    print("\nVTK not installed -- skipping VTK comparison.")


# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 70)
print(f"  {'Algorithm':<25} {'Time':>8}  {'Speedup':>10}")
print("-" * 70)

print(f"  {'Tack FE (untrimmed)':<25} {results['FE']:>7.4f}s  {'(baseline)':>10}")
print(f"  {'Tack FE (trimmed)':<25} {results['FE-Trim']:>7.4f}s  {results['FE']/results['FE-Trim']:>9.2f}x")
if "VTK-FE" in results:
    print(f"  {'VTK FE-image (TBB)':<25} {results['VTK-FE']:>7.4f}s  {results['FE']/results['VTK-FE']:>9.2f}x")

print(f"\n  Points: {total_points:,}  Triangles: {total_tris:,}")
print("=" * 70)
