"""27 — Flying Edges: merged-point isosurface extraction.

True FlyingEdges algorithm with edge ownership for merged (unique) points.
Each of the 12 MC edges is owned by exactly one voxel via its corner-0 node.
Each node owns at most 3 edges (x, y, z directions), so every edge
intersection is computed and stored exactly once.

Pipeline:
1. Count edge intersections per node-row + triangles per cell-row
2. Prefix sum over edge counts and triangle counts
3. Emit unique interpolated points, fill edge→point-ID mapping
4. Emit triangle connectivity using edge→point-ID lookups

Produces ~6x fewer output points than unmerged marching cubes.

Scalar field: gyroid — sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)

Usage:
  uv run python examples/27_flying_edges.py
  uv run python examples/27_flying_edges.py --arch metal
  uv run python examples/27_flying_edges.py --arch metal --size 200
"""

import time
import numpy as np
import pgc
from pgc import algorithms

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan', 'level_zero'])
_parser.add_argument('--size', type=int, default=100,
                     help='Grid cells per dimension (default 100)')
_parser.add_argument('--warmup', type=int, default=2)
_parser.add_argument('--trials', type=int, default=5)
_args = _parser.parse_args()
_arch = getattr(pgc, _args.arch)
pgc.init(arch=_arch)


# ================================================================
# MARCHING CUBES TABLES (same as example 26)
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
# TEMPLATES
# ================================================================

@pgc.data_oriented
class GridParams:
    """Grid constants — inlined as compile-time values via template."""

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


@pgc.data_oriented
class MCTables:
    """Marching cubes lookup tables as pgc fields."""

    def __init__(self):
        self.tri_table = pgc.field(dtype=pgc.i32, shape=(4096,))
        self.tri_table.from_numpy(TRI_TABLE)
        self.num_tris = pgc.field(dtype=pgc.i32, shape=(256,))
        self.num_tris.from_numpy(NUM_TRIS)
        self.edge_corners = pgc.field(dtype=pgc.i32, shape=(24,))
        self.edge_corners.from_numpy(EDGE_CORNERS)
        self.corner_x = pgc.field(dtype=pgc.f32, shape=(8,))
        self.corner_x.from_numpy(CORNER_X)
        self.corner_y = pgc.field(dtype=pgc.f32, shape=(8,))
        self.corner_y.from_numpy(CORNER_Y)
        self.corner_z = pgc.field(dtype=pgc.f32, shape=(8,))
        self.corner_z.from_numpy(CORNER_Z)


# ================================================================
# HELPER
# ================================================================

@pgc.func
def select8(v0, v1, v2, v3, v4, v5, v6, v7, idx):
    result = v0
    if idx == 1: result = v1
    if idx == 2: result = v2
    if idx == 3: result = v3
    if idx == 4: result = v4
    if idx == 5: result = v5
    if idx == 6: result = v6
    if idx == 7: result = v7
    return result


@pgc.func
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

@pgc.kernel
def _add_offset(field, offset, n):
    """Add a scalar offset to every element."""
    for i in range(n):
        field[i] = field[i] + offset


@pgc.kernel
def compute_scalar_field(scalar, grid: pgc.template(), n_points):
    """Gyroid: sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)."""
    for i in range(n_points):
        ix = i % grid.nx_p1
        iy = (i // grid.nx_p1) % grid.ny_p1
        iz = i // grid.nxy_p1
        x = grid.x0 + float(ix) * grid.dx
        y = grid.y0 + float(iy) * grid.dy
        z = grid.z0 + float(iz) * grid.dz
        scalar[i] = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)


# --- Cell-based MC (baseline, same as example 26) ---

@pgc.kernel
def mc_classify(scalar, num_tri_out, grid: pgc.template(),
                tables: pgc.template(), n_cells, isovalue):
    """Count triangles per cell."""
    for c in range(n_cells):
        ci = c % grid.nx
        cj = (c // grid.nx) % grid.ny
        ck = c // (grid.nx * grid.ny)
        base = ck * grid.nxy_p1 + cj * grid.nx_p1 + ci
        p0 = base
        p1 = base + 1
        p2 = base + grid.nx_p1 + 1
        p3 = base + grid.nx_p1
        p4 = base + grid.nxy_p1
        p5 = base + grid.nxy_p1 + 1
        p6 = base + grid.nxy_p1 + grid.nx_p1 + 1
        p7 = base + grid.nxy_p1 + grid.nx_p1
        case_idx = 0
        if scalar[p0] > isovalue: case_idx = case_idx + 1
        if scalar[p1] > isovalue: case_idx = case_idx + 2
        if scalar[p2] > isovalue: case_idx = case_idx + 4
        if scalar[p3] > isovalue: case_idx = case_idx + 8
        if scalar[p4] > isovalue: case_idx = case_idx + 16
        if scalar[p5] > isovalue: case_idx = case_idx + 32
        if scalar[p6] > isovalue: case_idx = case_idx + 64
        if scalar[p7] > isovalue: case_idx = case_idx + 128
        num_tri_out[c] = tables.num_tris[case_idx]


@pgc.kernel
def mc_emit(scalar, offsets, grid: pgc.template(),
            tables: pgc.template(), out_x, out_y, out_z,
            n_cells, isovalue):
    """Emit interpolated triangle vertices (per-cell)."""
    for c in range(n_cells):
        ci = c % grid.nx
        cj = (c // grid.nx) % grid.ny
        ck = c // (grid.nx * grid.ny)
        base = ck * grid.nxy_p1 + cj * grid.nx_p1 + ci
        v0 = scalar[base]
        v1 = scalar[base + 1]
        v2 = scalar[base + grid.nx_p1 + 1]
        v3 = scalar[base + grid.nx_p1]
        v4 = scalar[base + grid.nxy_p1]
        v5 = scalar[base + grid.nxy_p1 + 1]
        v6 = scalar[base + grid.nxy_p1 + grid.nx_p1 + 1]
        v7 = scalar[base + grid.nxy_p1 + grid.nx_p1]

        case_idx = 0
        if v0 > isovalue: case_idx = case_idx + 1
        if v1 > isovalue: case_idx = case_idx + 2
        if v2 > isovalue: case_idx = case_idx + 4
        if v3 > isovalue: case_idx = case_idx + 8
        if v4 > isovalue: case_idx = case_idx + 16
        if v5 > isovalue: case_idx = case_idx + 32
        if v6 > isovalue: case_idx = case_idx + 64
        if v7 > isovalue: case_idx = case_idx + 128

        x_lo = grid.x0 + float(ci) * grid.dx
        y_lo = grid.y0 + float(cj) * grid.dy
        z_lo = grid.z0 + float(ck) * grid.dz

        out_idx = offsets[c] * 3
        for t in range(16):
            edge = tables.tri_table[case_idx * 16 + t]
            if edge >= 0:
                ca = tables.edge_corners[edge * 2]
                cb = tables.edge_corners[edge * 2 + 1]
                va = select8(v0, v1, v2, v3, v4, v5, v6, v7, ca)
                vb = select8(v0, v1, v2, v3, v4, v5, v6, v7, cb)
                interp = (isovalue - va) / (vb - va + 1e-10)
                xa = x_lo + tables.corner_x[ca] * grid.dx
                xb = x_lo + tables.corner_x[cb] * grid.dx
                ya = y_lo + tables.corner_y[ca] * grid.dy
                yb = y_lo + tables.corner_y[cb] * grid.dy
                za = z_lo + tables.corner_z[ca] * grid.dz
                zb = z_lo + tables.corner_z[cb] * grid.dz
                out_x[out_idx] = xa + interp * (xb - xa)
                out_y[out_idx] = ya + interp * (yb - ya)
                out_z[out_idx] = za + interp * (zb - za)
                out_idx = out_idx + 1


# --- Flying Edges: row-based processing ---

@pgc.kernel
def fe_count_rows(scalar, row_tri_count, grid: pgc.template(),
                  tables: pgc.template(), isovalue, n_rows):
    """Pass 1: Count triangles per cell-row (sequential x-scan)."""
    for row in range(n_rows):
        cj = row % grid.ny
        ck = row // grid.ny
        base00 = ck * grid.nxy_p1 + cj * grid.nx_p1
        base10 = base00 + grid.nx_p1
        base01 = base00 + grid.nxy_p1
        base11 = base10 + grid.nxy_p1

        count = 0
        # Preload left column
        s0 = scalar[base00]
        s3 = scalar[base10]
        s4 = scalar[base01]
        s7 = scalar[base11]

        for ci in range(grid.nx):
            # Load right column
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

            # Shift right → left
            s0 = s1
            s3 = s2
            s4 = s5
            s7 = s6

        row_tri_count[row] = count


@pgc.kernel
def fe_count_edges_xyz(scalar, row_xc, row_yc, row_zc,
                       grid: pgc.template(), isovalue, n_node_rows):
    """Count x/y/z edge intersections separately per node-row.

    Separate counts allow computing point IDs on the fly during
    triangle emission without a global edge→point-ID buffer.
    """
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


@pgc.kernel
def fe_emit_points_xyz(scalar, row_xo, row_yo, row_zo,
                       pt_x, pt_y, pt_z,
                       grid: pgc.template(), isovalue, n_node_rows):
    """Emit unique interpolated points using separate x/y/z offsets.

    Point IDs are deterministic: all x-points first, then y, then z.
    Within each type, ordered by node-row then by node position.
    """
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
            px = grid.x0 + float(i) * grid.dx
            py = grid.y0 + float(j) * grid.dy
            pz = grid.z0 + float(k) * grid.dz
            if i < grid.nx:
                s_nx = scalar[node_idx + 1]
                a_nx = 0
                if s_nx > isovalue:
                    a_nx = 1
                if above != a_nx:
                    t = (isovalue - s) / (s_nx - s + 1.0e-10)
                    pt_x[xi] = px + t * grid.dx
                    pt_y[xi] = py
                    pt_z[xi] = pz
                    xi = xi + 1
            if j < grid.ny:
                s_ny = scalar[node_idx + grid.nx_p1]
                a_ny = 0
                if s_ny > isovalue:
                    a_ny = 1
                if above != a_ny:
                    t = (isovalue - s) / (s_ny - s + 1.0e-10)
                    pt_x[yi] = px
                    pt_y[yi] = py + t * grid.dy
                    pt_z[yi] = pz
                    yi = yi + 1
            if k < grid.nz:
                s_nz = scalar[node_idx + grid.nxy_p1]
                a_nz = 0
                if s_nz > isovalue:
                    a_nz = 1
                if above != a_nz:
                    t = (isovalue - s) / (s_nz - s + 1.0e-10)
                    pt_x[zi] = px
                    pt_y[zi] = py
                    pt_z[zi] = pz + t * grid.dz
                    zi = zi + 1


@pgc.kernel
def fe_emit_tris(scalar, row_tri_offset,
                 row_xo, row_yo, row_zo,
                 tri_v0, tri_v1, tri_v2,
                 grid: pgc.template(), tables: pgc.template(),
                 isovalue, n_cell_rows):
    """Emit triangle connectivity by computing point IDs on the fly.

    Tracks running x/y/z edge counts for 4 neighbor node-rows using
    the 8 cell-corner scalars (already loaded for MC classification).
    No edge_ids buffer needed.

    Edge→row/type mapping (dj,dk relative to cell's cj,ck):
      Edge 0:  x at ci,   row(0,0)    Edge 1:  y at ci+1, row(0,0)
      Edge 2:  x at ci,   row(+1,0)   Edge 3:  y at ci,   row(0,0)
      Edge 4:  x at ci,   row(0,+1)   Edge 5:  y at ci+1, row(0,+1)
      Edge 6:  x at ci,   row(+1,+1)  Edge 7:  y at ci,   row(0,+1)
      Edge 8:  z at ci,   row(0,0)    Edge 9:  z at ci+1, row(0,0)
      Edge 10: z at ci,   row(+1,0)   Edge 11: z at ci+1, row(+1,0)
    """
    for row in range(n_cell_rows):
        cj = row % grid.ny
        ck = row // grid.ny
        # Node-row indices for 4 neighbor rows
        r00 = ck * grid.ny_p1 + cj
        r10 = r00 + 1
        r01 = r00 + grid.ny_p1
        r11 = r01 + 1
        # Per-type base offsets from prefix sum
        xo00 = row_xo[r00]
        yo00 = row_yo[r00]
        zo00 = row_zo[r00]
        xo10 = row_xo[r10]
        zo10 = row_zo[r10]
        xo01 = row_xo[r01]
        yo01 = row_yo[r01]
        xo11 = row_xo[r11]
        # Running per-type edge counts (edges at nodes 0..ci-1)
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
        # Preload left column
        s0 = scalar[base00]
        s3 = scalar[base10]
        s4 = scalar[base01]
        s7 = scalar[base11]
        for ci in range(grid.nx):
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
            # Row(cj,ck): s0 is node scalar
            ix00 = 0
            iy00 = 0
            iz00 = 0
            if a0 != a1: ix00 = 1
            if a0 != a3: iy00 = 1
            if a0 != a4: iz00 = 1
            # Row(cj+1,ck): s3 is node scalar
            ix10 = 0
            iz10 = 0
            if a3 != a2: ix10 = 1
            if a3 != a7: iz10 = 1
            # Row(cj,ck+1): s4 is node scalar
            ix01 = 0
            iy01 = 0
            if a4 != a5: ix01 = 1
            if a4 != a7: iy01 = 1
            # Row(cj+1,ck+1): s7 is node scalar
            ix11 = 0
            if a7 != a6: ix11 = 1
            # Point IDs for all 12 MC edges
            # "before" uses running count, "after" adds current node's flag
            pid0  = xo00 + rx00               # x at ci,   row00
            pid1  = yo00 + ry00 + iy00        # y at ci+1, row00
            pid2  = xo10 + rx10               # x at ci,   row10
            pid3  = yo00 + ry00               # y at ci,   row00
            pid4  = xo01 + rx01               # x at ci,   row01
            pid5  = yo01 + ry01 + iy01        # y at ci+1, row01
            pid6  = xo11 + rx11               # x at ci,   row11
            pid7  = yo01 + ry01               # y at ci,   row01
            pid8  = zo00 + rz00               # z at ci,   row00
            pid9  = zo00 + rz00 + iz00        # z at ci+1, row00
            pid10 = zo10 + rz10               # z at ci,   row10
            pid11 = zo10 + rz10 + iz10        # z at ci+1, row10
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
            # Shift right → left
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

grid = GridParams(nx, ny, nz, x0, y0, z0, dx, dy, dz)
tables = MCTables()

# Compute scalar field (shared)
scalar = pgc.field(dtype=pgc.f32, shape=(n_points,))
compute_scalar_field(scalar, grid, n_points)

results = {}


# --- True Flying Edges (merged points, no edge_ids buffer) ---
print("Flying Edges (merged points)...")

# Pass 1a: Count x/y/z edge intersections separately per node-row
row_xc = pgc.field(dtype=pgc.i32, shape=(n_node_rows,))
row_yc = pgc.field(dtype=pgc.i32, shape=(n_node_rows,))
row_zc = pgc.field(dtype=pgc.i32, shape=(n_node_rows,))
fe_count_edges_xyz(scalar, row_xc, row_yc, row_zc, grid, isovalue, n_node_rows)

# 3 separate prefix sums → per-type offsets (CPU, row arrays are small)
xc_np = row_xc.to_numpy()
yc_np = row_yc.to_numpy()
zc_np = row_zc.to_numpy()
total_x = int(np.sum(xc_np))
total_y = int(np.sum(yc_np))
total_z = int(np.sum(zc_np))
total_points_fe = total_x + total_y + total_z

xo_np = np.zeros(n_node_rows, dtype=np.int32)
yo_np = np.zeros(n_node_rows, dtype=np.int32)
zo_np = np.zeros(n_node_rows, dtype=np.int32)
xo_np[1:] = np.cumsum(xc_np[:-1])
yo_np[1:] = np.cumsum(yc_np[:-1])
yo_np += total_x  # y-points start after all x-points
zo_np[1:] = np.cumsum(zc_np[:-1])
zo_np += total_x + total_y  # z-points start after x+y

row_xo = pgc.field(dtype=pgc.i32, shape=(n_node_rows,))
row_yo = pgc.field(dtype=pgc.i32, shape=(n_node_rows,))
row_zo = pgc.field(dtype=pgc.i32, shape=(n_node_rows,))
row_xo.from_numpy(xo_np)
row_yo.from_numpy(yo_np)
row_zo.from_numpy(zo_np)

# Pass 1b: Count triangles per cell-row
row_tri_count = pgc.field(dtype=pgc.i32, shape=(n_cell_rows,))
fe_count_rows(scalar, row_tri_count, grid, tables, isovalue, n_cell_rows)

tri_counts_np = row_tri_count.to_numpy()
tri_offsets_np = np.zeros(n_cell_rows, dtype=np.int32)
tri_offsets_np[1:] = np.cumsum(tri_counts_np[:-1])
total_tris_fe = int(np.sum(tri_counts_np))
row_tri_offset = pgc.field(dtype=pgc.i32, shape=(n_cell_rows,))
row_tri_offset.from_numpy(tri_offsets_np)

print(f"  Unique points: {total_points_fe:,} (x:{total_x:,} y:{total_y:,} z:{total_z:,})")
print(f"  Triangles:     {total_tris_fe:,}")
print(f"  Merge ratio:   {total_tris_fe * 3 / total_points_fe:.1f}x fewer points than unmerged")

# Allocate output — NO edge_ids buffer needed!
pt_x = pgc.field(dtype=pgc.f32, shape=(total_points_fe,))
pt_y = pgc.field(dtype=pgc.f32, shape=(total_points_fe,))
pt_z = pgc.field(dtype=pgc.f32, shape=(total_points_fe,))
tri_v0 = pgc.field(dtype=pgc.i32, shape=(total_tris_fe,))
tri_v1 = pgc.field(dtype=pgc.i32, shape=(total_tris_fe,))
tri_v2 = pgc.field(dtype=pgc.i32, shape=(total_tris_fe,))


def _fe_prefix_sums():
    """Run count + prefix sum passes (shared by warmup and timed runs)."""
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


# Warmup
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

fe_px = pt_x.to_numpy()
fe_py = pt_y.to_numpy()
fe_pz = pt_z.to_numpy()
fe_t0 = tri_v0.to_numpy()
fe_t1 = tri_v1.to_numpy()
fe_t2 = tri_v2.to_numpy()


# --- Cell-based MC (baseline, unmerged) ---
print("\nCell-based MC (unmerged baseline)...")

num_tri = pgc.field(dtype=pgc.i32, shape=(n_cells,))
offsets = pgc.field(dtype=pgc.i32, shape=(n_cells,))

mc_classify(scalar, num_tri, grid, tables, n_cells, isovalue)
total_tris_mc = algorithms.exclusive_scan(num_tri, offsets, n_cells)
total_verts_mc = total_tris_mc * 3

print(f"  Triangles: {total_tris_mc:,}")
print(f"  Vertices:  {total_verts_mc:,} (unmerged)")

mc_out_x = pgc.field(dtype=pgc.f32, shape=(total_verts_mc,))
mc_out_y = pgc.field(dtype=pgc.f32, shape=(total_verts_mc,))
mc_out_z = pgc.field(dtype=pgc.f32, shape=(total_verts_mc,))

for _w in range(warmup):
    mc_classify(scalar, num_tri, grid, tables, n_cells, isovalue)
    algorithms.exclusive_scan(num_tri, offsets, n_cells)
    mc_emit(scalar, offsets, grid, tables, mc_out_x, mc_out_y, mc_out_z,
            n_cells, isovalue)

times = []
for _t in range(trials):
    t0 = time.perf_counter()
    mc_classify(scalar, num_tri, grid, tables, n_cells, isovalue)
    algorithms.exclusive_scan(num_tri, offsets, n_cells)
    mc_emit(scalar, offsets, grid, tables, mc_out_x, mc_out_y, mc_out_z,
            n_cells, isovalue)
    t1 = time.perf_counter()
    times.append(t1 - t0)

mc_time = min(times)
print(f"  Best of {trials}: {mc_time:.4f}s")
results["MC"] = mc_time


# --- VTK comparison ---
vtk_total_tris = None
vtk_total_points = None
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
    print("\nVTK not installed — skipping VTK comparison.")
    print("  Install with: uv pip install vtk")


# --- Validation ---
print("\n--- Validation ---")
assert total_tris_fe == total_tris_mc, f"FE/MC tri mismatch: {total_tris_fe} vs {total_tris_mc}"
print(f"  FE == MC triangle count: OK ({total_tris_fe:,})")

if vtk_total_tris is not None:
    diff = abs(total_tris_fe - vtk_total_tris)
    if diff == 0:
        print(f"  FE == VTK triangles: OK")
    elif diff < total_tris_fe * 0.001:
        print(f"  FE ~ VTK triangles: diff={diff} ({diff/total_tris_fe*100:.2f}%, f32 vs f64)")

if vtk_total_points is not None:
    pdiff = abs(total_points_fe - vtk_total_points)
    if pdiff == 0:
        print(f"  FE == VTK points: OK ({total_points_fe:,})")
    else:
        print(f"  FE points: {total_points_fe:,}, VTK points: {vtk_total_points:,} (diff={pdiff})")

# Verify connectivity: all triangle indices in range
assert np.all(fe_t0 >= 0) and np.all(fe_t0 < total_points_fe), "tri_v0 out of range"
assert np.all(fe_t1 >= 0) and np.all(fe_t1 < total_points_fe), "tri_v1 out of range"
assert np.all(fe_t2 >= 0) and np.all(fe_t2 < total_points_fe), "tri_v2 out of range"
print("  Triangle connectivity: OK (all indices in range)")

# Verify points on isosurface
n_sample = min(100, total_points_fe)
sample_vals = (np.sin(fe_px[:n_sample]) * np.cos(fe_py[:n_sample])
             + np.sin(fe_py[:n_sample]) * np.cos(fe_pz[:n_sample])
             + np.sin(fe_pz[:n_sample]) * np.cos(fe_px[:n_sample]))
max_err = np.max(np.abs(sample_vals - isovalue))
print(f"  Max scalar error at points: {max_err:.6f}")
assert max_err < 0.1
print("  Points on isosurface: OK")


# --- Summary ---
print("\n" + "=" * 65)
print(f"  {'Algorithm':<20} {'Time':>8}  {'Points':>12}  {'vs FE':>8}  {'vs VTK':>8}")
print("-" * 65)
vtk_time = results.get("VTK-FE", None)

t = results["FE"]
vs_vtk = f"{vtk_time / t:>7.2f}x" if vtk_time else "  n/a  "
print(f"  {'FE (merged)':<20} {t:>7.4f}s  {total_points_fe:>12,}  {1.0:>7.2f}x  {vs_vtk}")

t = results["MC"]
vs_fe = fe_time / t
vs_vtk = f"{vtk_time / t:>7.2f}x" if vtk_time else "  n/a  "
print(f"  {'MC (unmerged)':<20} {t:>7.4f}s  {total_verts_mc:>12,}  {vs_fe:>7.2f}x  {vs_vtk}")

if vtk_time:
    vs_fe = fe_time / vtk_time
    vp = vtk_total_points if vtk_total_points else 0
    print(f"  {'VTK-FE (TBB)':<20} {vtk_time:>7.4f}s  {vp:>12,}  {vs_fe:>7.2f}x  {1.0:>7.2f}x")

print("=" * 65)
print(f"""
True FlyingEdges with edge ownership (no edge_ids buffer):
  Merged points: {total_points_fe:,} unique vs {total_verts_mc:,} unmerged ({total_verts_mc / total_points_fe:.1f}x reduction)
  Point IDs computed on the fly via running x/y/z edge counts per row
  Node rows: {n_node_rows:,} (edge counting), Cell rows: {n_cell_rows:,} (triangle counting)
""")
