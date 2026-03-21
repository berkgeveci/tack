"""31 -- Multi-block Flying Edges: isosurface extraction across AMR patches.

Runs the Flying Edges algorithm independently on each block (FAB) of a
multi-block dataset, producing unified output arrays for points and
connectivity.  Uses pgc.algorithms.flying_edges module.

Usage:
  uv run python examples/31_multiblock_flying_edges.py
  uv run python examples/31_multiblock_flying_edges.py --arch metal
  uv run python examples/31_multiblock_flying_edges.py --nblocks 100
"""

import time
import numpy as np
import pgc
from pgc.algorithms.flying_edges import flying_edges_multiblock, UniformGrid

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--nblocks', type=int, default=8,
                     help='Number of blocks (default 8)')
_parser.add_argument('--block_size', type=int, default=32,
                     help='Cells per dimension per block (default 32)')
_args = _parser.parse_args()
_arch = getattr(pgc, _args.arch)
pgc.init(arch=_arch)


# ================================================================
# SYNTHETIC MULTI-BLOCK DATA
# ================================================================

@pgc.kernel
def compute_gyroid(scalar, grid: pgc.template(), n_points):
    """Gyroid: sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)."""
    for i in range(n_points):
        ix = i % grid.nx_p1
        iy = (i // grid.nx_p1) % grid.ny_p1
        iz = i // grid.nxy_p1
        x = grid.get_x(ix, iy, iz)
        y = grid.get_y(ix, iy, iz)
        z = grid.get_z(ix, iy, iz)
        scalar[i] = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)


def make_blocks(nblocks, block_size):
    """Create a grid of blocks tiling [-pi, pi]^3."""
    nb = nblocks
    nbx = max(1, round(nb ** (1/3)))
    nby = max(1, round((nb / nbx) ** 0.5))
    nbz = max(1, nb // (nbx * nby))
    if nbx * nby * nbz < nb:
        nbz = (nb + nbx * nby - 1) // (nbx * nby)

    domain_lo = -np.pi
    domain_hi = np.pi
    domain_size = domain_hi - domain_lo
    dx = domain_size / (nbx * block_size)
    dy = domain_size / (nby * block_size)
    dz = domain_size / (nbz * block_size)

    blocks = []
    for bk in range(nbz):
        for bj in range(nby):
            for bi in range(nbx):
                if len(blocks) >= nb:
                    break
                nx = ny = nz = block_size
                x0 = domain_lo + bi * nx * dx
                y0 = domain_lo + bj * ny * dy
                z0 = domain_lo + bk * nz * dz
                grid = UniformGrid(nx, ny, nz, x0, y0, z0, dx, dy, dz)
                n_points = (nx + 1) * (ny + 1) * (nz + 1)
                scalar = pgc.field(dtype=pgc.f32, shape=(n_points,))
                compute_gyroid(scalar, grid, n_points)
                blocks.append({'scalar': scalar, 'grid': grid})

    return blocks, (nbx, nby, nbz), (dx, dy, dz)


# ================================================================
# RUN
# ================================================================

nblocks = _args.nblocks
block_size = _args.block_size
isovalue = 0.0

print(f"Multi-block Flying Edges")
print(f"  Backend:    {_args.arch}")
print(f"  Blocks:     {nblocks} x {block_size}^3 cells each")
print(f"  Isovalue:   {isovalue}")
print()

# Create synthetic blocks
t0 = time.perf_counter()
blocks, (nbx, nby, nbz), (dx, dy, dz) = make_blocks(nblocks, block_size)
t_setup = time.perf_counter() - t0
print(f"  Block layout: {nbx} x {nby} x {nbz} = {len(blocks)} blocks")
print(f"  Spacing:    dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
print(f"  Setup time: {t_setup:.4f}s")
print()

# Warmup
flying_edges_multiblock(blocks, isovalue)

# Timed run
t0 = time.perf_counter()
result = flying_edges_multiblock(blocks, isovalue)
t_fe = time.perf_counter() - t0

if result is None:
    print("  No isosurface produced.")
else:
    print(f"  Points:     {result['total_points']:,}")
    print(f"  Triangles:  {result['total_tris']:,}")
    print(f"  Time:       {t_fe:.4f}s")

    # Verify connectivity
    conn = result['conn']  # shape (m, 3)
    assert np.all(conn >= 0) and np.all(conn < result['total_points'])
    print("  Connectivity: OK (all indices in range)")

    # Verify points on isosurface
    pts = result['points']  # shape (n, 3)
    n_sample = min(100, result['total_points'])
    idxs = np.random.choice(result['total_points'], n_sample, replace=False)
    sx = pts[idxs, 0].astype(np.float64)
    sy = pts[idxs, 1].astype(np.float64)
    sz = pts[idxs, 2].astype(np.float64)
    vals = np.sin(sx)*np.cos(sy) + np.sin(sy)*np.cos(sz) + np.sin(sz)*np.cos(sx)
    max_err = float(np.max(np.abs(vals - isovalue)))
    print(f"  Max scalar error at points: {max_err:.6f}")
    print(f"  Total cells across blocks: {len(blocks) * block_size**3:,}")
