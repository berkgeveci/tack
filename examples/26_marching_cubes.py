"""26 — Marching cubes isosurface extraction.

Two-pass marching cubes on a structured 3D grid:

1. Classify: compute case index per cell, count triangles
2. Prefix sum: compute scatter offsets
3. Emit: interpolate triangle vertices along cube edges

Uses the cell set abstraction (CellSetStructured3D) and multi-return
@pgc.func for get_cell_points() and get_cell_bounds().  The marching
cubes lookup tables are stored as pgc fields (read-only data on GPU).

Scalar field: a gyroid — sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
This produces a beautiful triply-periodic minimal surface at isovalue 0.

Usage:
  uv run python examples/26_marching_cubes.py
  uv run python examples/26_marching_cubes.py --arch metal
  uv run python examples/26_marching_cubes.py --arch metal --size 200
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
# MARCHING CUBES TABLES
# ================================================================
# Standard tables from VTK (vtkMarchingCubesTriangleCases.cxx).
# Edge-to-corner pairs and triangle table (256 cases × 16 entries).

# 12 edges, each connecting two corners (VTK ordering)
EDGE_CORNERS = np.array([
    0, 1,  1, 2,  3, 2,  0, 3,   # edges 0-3  (bottom face)
    4, 5,  5, 6,  7, 6,  4, 7,   # edges 4-7  (top face)
    0, 4,  1, 5,  3, 7,  2, 6,   # edges 8-11 (vertical)
], dtype=np.int32)

# Corner positions in the unit cube (MC standard ordering)
# x: 0=lo, 1=hi for each corner
CORNER_X = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.float32)
CORNER_Y = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32)
CORNER_Z = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

# Triangle table: 256 cases × 16 entries (-1 terminated)
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

# Derive triangle count per case from the table
NUM_TRIS = np.zeros(256, dtype=np.int32)
for _case in range(256):
    _count = 0
    for _t in range(0, 16, 3):
        if TRI_TABLE[_case * 16 + _t] >= 0:
            _count += 1
    NUM_TRIS[_case] = _count


# ================================================================
# GRID + TABLES AS @pgc.data_oriented
# ================================================================

@pgc.data_oriented
class UniformGrid3D:
    """Structured hex grid with cell bounds computation."""

    def __init__(self, nx, ny, nz, x0, y0, z0, dx, dy, dz):
        self.nx = nx
        self.ny = ny
        self.nxy = nx * ny
        self.nx_p1 = nx + 1
        self.nxy_p1 = (nx + 1) * (ny + 1)
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.dx = dx
        self.dy = dy
        self.dz = dz

    @pgc.func
    def get_cell_points(self, cell_id):
        """Return 8 corner point IDs in MC standard ordering."""
        ci = cell_id % self.nx
        cj = (cell_id // self.nx) % self.ny
        ck = cell_id // self.nxy
        base = ck * self.nxy_p1 + cj * self.nx_p1 + ci
        p0 = base
        p1 = base + 1
        p2 = base + self.nx_p1 + 1
        p3 = base + self.nx_p1
        p4 = base + self.nxy_p1
        p5 = base + self.nxy_p1 + 1
        p6 = base + self.nxy_p1 + self.nx_p1 + 1
        p7 = base + self.nxy_p1 + self.nx_p1
        return p0, p1, p2, p3, p4, p5, p6, p7

    @pgc.func
    def get_cell_bounds(self, cell_id):
        """Return (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)."""
        ci = cell_id % self.nx
        cj = (cell_id // self.nx) % self.ny
        ck = cell_id // self.nxy
        x_lo = self.x0 + float(ci) * self.dx
        y_lo = self.y0 + float(cj) * self.dy
        z_lo = self.z0 + float(ck) * self.dz
        return x_lo, x_lo + self.dx, y_lo, y_lo + self.dy, z_lo, z_lo + self.dz


@pgc.data_oriented
class MCTables:
    """Marching cubes lookup tables stored as pgc fields."""

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
# HELPER: select one of 8 scalar values by index
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


# ================================================================
# KERNELS
# ================================================================

@pgc.kernel
def compute_scalar_field(scalar, grid: pgc.template(), n_points):
    """Gyroid: f(x,y,z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)."""
    for i in range(n_points):
        ix = i % grid.nx_p1
        iy = (i // grid.nx_p1) % (grid.ny + 1)
        iz = i // grid.nxy_p1
        x = grid.x0 + float(ix) * grid.dx
        y = grid.y0 + float(iy) * grid.dy
        z = grid.z0 + float(iz) * grid.dz
        scalar[i] = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)


@pgc.kernel
def classify_cells(scalar, num_tri_out, grid: pgc.template(),
                   tables: pgc.template(), n_cells, isovalue):
    """Count triangles per cell based on MC case index."""
    for c in range(n_cells):
        p0, p1, p2, p3, p4, p5, p6, p7 = grid.get_cell_points(c)
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
def emit_triangles(scalar, offsets, grid: pgc.template(),
                   tables: pgc.template(), out_x, out_y, out_z,
                   n_cells, isovalue):
    """Emit interpolated triangle vertices."""
    for c in range(n_cells):
        p0, p1, p2, p3, p4, p5, p6, p7 = grid.get_cell_points(c)

        v0 = scalar[p0]
        v1 = scalar[p1]
        v2 = scalar[p2]
        v3 = scalar[p3]
        v4 = scalar[p4]
        v5 = scalar[p5]
        v6 = scalar[p6]
        v7 = scalar[p7]

        case_idx = 0
        if v0 > isovalue: case_idx = case_idx + 1
        if v1 > isovalue: case_idx = case_idx + 2
        if v2 > isovalue: case_idx = case_idx + 4
        if v3 > isovalue: case_idx = case_idx + 8
        if v4 > isovalue: case_idx = case_idx + 16
        if v5 > isovalue: case_idx = case_idx + 32
        if v6 > isovalue: case_idx = case_idx + 64
        if v7 > isovalue: case_idx = case_idx + 128

        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = grid.get_cell_bounds(c)

        out_idx = offsets[c] * 3
        for t in range(16):
            edge = tables.tri_table[case_idx * 16 + t]
            if edge >= 0:
                ca = tables.edge_corners[edge * 2]
                cb = tables.edge_corners[edge * 2 + 1]

                va = select8(v0, v1, v2, v3, v4, v5, v6, v7, ca)
                vb = select8(v0, v1, v2, v3, v4, v5, v6, v7, cb)
                interp = (isovalue - va) / (vb - va + 1e-10)

                xa = x_lo + tables.corner_x[ca] * (x_hi - x_lo)
                xb = x_lo + tables.corner_x[cb] * (x_hi - x_lo)
                ya = y_lo + tables.corner_y[ca] * (y_hi - y_lo)
                yb = y_lo + tables.corner_y[cb] * (y_hi - y_lo)
                za = z_lo + tables.corner_z[ca] * (z_hi - z_lo)
                zb = z_lo + tables.corner_z[cb] * (z_hi - z_lo)

                out_x[out_idx] = xa + interp * (xb - xa)
                out_y[out_idx] = ya + interp * (yb - ya)
                out_z[out_idx] = za + interp * (zb - za)
                out_idx = out_idx + 1


# ================================================================
# RUN
# ================================================================

N = _args.size
nx, ny, nz = N, N, N
n_points = (nx + 1) * (ny + 1) * (nz + 1)
n_cells = nx * ny * nz
warmup = _args.warmup
trials = _args.trials
isovalue = 0.0

# Domain: [-pi, pi]^3 for the gyroid
x0, y0, z0 = -np.pi, -np.pi, -np.pi
dx = 2.0 * np.pi / nx
dy = 2.0 * np.pi / ny
dz = 2.0 * np.pi / nz

print(f"Grid: {nx}x{ny}x{nz} = {n_cells:,} cells, {n_points:,} points")
print(f"Backend: {_args.arch}")
print(f"Isovalue: {isovalue}")
print()

grid = UniformGrid3D(nx, ny, nz, x0, y0, z0, dx, dy, dz)
tables = MCTables()

# Compute scalar field
scalar = pgc.field(dtype=pgc.f32, shape=(n_points,))
compute_scalar_field(scalar, grid, n_points)

# Classify
num_tri = pgc.field(dtype=pgc.i32, shape=(n_cells,))
classify_cells(scalar, num_tri, grid, tables, n_cells, isovalue)

# Prefix sum → scatter offsets
offsets = pgc.field(dtype=pgc.i32, shape=(n_cells,))
total_tris = algorithms.exclusive_scan(num_tri, offsets, n_cells)
total_verts = total_tris * 3
print(f"Triangles: {total_tris:,}")
print(f"Vertices:  {total_verts:,}")
print()

# Allocate output
out_x = pgc.field(dtype=pgc.f32, shape=(total_verts,))
out_y = pgc.field(dtype=pgc.f32, shape=(total_verts,))
out_z = pgc.field(dtype=pgc.f32, shape=(total_verts,))

# Benchmark: full pipeline (classify + scan + emit)
print("PGC marching cubes...")

for i in range(warmup):
    classify_cells(scalar, num_tri, grid, tables, n_cells, isovalue)
    algorithms.exclusive_scan(num_tri, offsets, n_cells)
    emit_triangles(scalar, offsets, grid, tables, out_x, out_y, out_z, n_cells, isovalue)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    classify_cells(scalar, num_tri, grid, tables, n_cells, isovalue)
    algorithms.exclusive_scan(num_tri, offsets, n_cells)
    emit_triangles(scalar, offsets, grid, tables, out_x, out_y, out_z, n_cells, isovalue)
    t1 = time.perf_counter()
    times.append(t1 - t0)

pgc_time = min(times)
print(f"  Best of {trials}: {pgc_time:.4f}s")

# Read back a few vertices for validation
pgc_vx = out_x.to_numpy()
pgc_vy = out_y.to_numpy()
pgc_vz = out_z.to_numpy()

results = {"PGC": pgc_time}


# --- VTK comparison ---
vtk_total_tris = None
try:
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import vtkContourFilter
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    print("\nVTK vtkContourFilter...")

    img = vtkImageData()
    img.SetDimensions(nx + 1, ny + 1, nz + 1)
    img.SetOrigin(x0, y0, z0)
    img.SetSpacing(dx, dy, dz)

    scalar_np = scalar.to_numpy()
    vtk_arr = numpy_to_vtk(scalar_np, deep=True)
    vtk_arr.SetName("scalar")
    img.GetPointData().SetScalars(vtk_arr)

    # SynchronizedTemplates3D (default, serial)
    cf = vtkContourFilter()
    cf.SetInputData(img)
    cf.SetValue(0, isovalue)

    for i in range(warmup):
        cf.Modified()
        cf.Update()

    times = []
    for i in range(trials):
        cf.Modified()
        t0 = time.perf_counter()
        cf.Update()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    vtk_time = min(times)
    vtk_output = cf.GetOutput()
    vtk_total_tris = vtk_output.GetNumberOfCells()
    vtk_total_verts = vtk_output.GetNumberOfPoints()
    print(f"  SyncTemplates (serial): {vtk_time:.4f}s")
    print(f"  Triangles: {vtk_total_tris:,}, Vertices: {vtk_total_verts:,}")
    results["VTK-ST"] = vtk_time

    # FlyingEdges3D (fast mode, threaded)
    cf2 = vtkContourFilter()
    cf2.SetInputData(img)
    cf2.SetValue(0, isovalue)
    cf2.SetFastMode(True)

    for i in range(warmup):
        cf2.Modified()
        cf2.Update()

    times = []
    for i in range(trials):
        cf2.Modified()
        t0 = time.perf_counter()
        cf2.Update()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    vtk_fe_time = min(times)
    vtk_fe_output = cf2.GetOutput()
    vtk_fe_tris = vtk_fe_output.GetNumberOfCells()
    vtk_fe_verts = vtk_fe_output.GetNumberOfPoints()
    print(f"  FlyingEdges (threaded): {vtk_fe_time:.4f}s")
    print(f"  Triangles: {vtk_fe_tris:,}, Vertices: {vtk_fe_verts:,}")
    results["VTK-FE"] = vtk_fe_time

    del cf, cf2, img, vtk_output, vtk_fe_output

except ImportError:
    print("\nVTK not installed — skipping VTK comparison.")
    print("  Install with: uv pip install vtk")


# --- Validation ---
print("\n--- Validation ---")
print(f"  PGC triangles: {total_tris:,}")
if vtk_total_tris is not None:
    print(f"  VTK triangles (ST): {vtk_total_tris:,}")
    diff = abs(total_tris - vtk_total_tris)
    if diff == 0:
        print("  Triangle count matches: OK")
    elif diff < total_tris * 0.001:
        print(f"  Triangle count close: diff={diff} ({diff/total_tris*100:.2f}%, float32 vs float64 rounding)")

# Sanity: all vertices should be within the domain
assert pgc_vx.min() >= x0 - dx and pgc_vx.max() <= x0 + (nx + 1) * dx
assert pgc_vy.min() >= y0 - dy and pgc_vy.max() <= y0 + (ny + 1) * dy
assert pgc_vz.min() >= z0 - dz and pgc_vz.max() <= z0 + (nz + 1) * dz
print("  Vertex bounds: OK")

# Check a few vertices are on the isosurface
sample_pts = np.column_stack([pgc_vx[:100], pgc_vy[:100], pgc_vz[:100]])
sample_vals = (np.sin(sample_pts[:, 0]) * np.cos(sample_pts[:, 1])
             + np.sin(sample_pts[:, 1]) * np.cos(sample_pts[:, 2])
             + np.sin(sample_pts[:, 2]) * np.cos(sample_pts[:, 0]))
max_err = np.max(np.abs(sample_vals - isovalue))
print(f"  Max scalar error at vertices: {max_err:.6f}")
assert max_err < 0.1, f"Vertices not on isosurface: max error {max_err}"
print("  Vertices on isosurface: OK")


# --- Summary ---
print("\n" + "=" * 50)
print(f"  {'Type':<12} {'Time':>8}  {'Speedup':>8}")
print("-" * 50)
baseline = results.get("VTK-FE", results.get("VTK-ST", pgc_time))
row_names = ["PGC"]
if "VTK-ST" in results:
    row_names.append("VTK-ST")
if "VTK-FE" in results:
    row_names.append("VTK-FE")
for name in row_names:
    t = results[name]
    speedup = baseline / t
    print(f"  {name:<12} {t:>7.4f}s  {speedup:>7.2f}x")
print("=" * 50)
