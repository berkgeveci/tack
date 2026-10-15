"""31 -- Multi-block contour: isosurface extraction across AMR patches.

Each block (an AMR patch, a FAB) is a uniform grid of its own, a
tack.dataset dataset; the isosurface is one contour per block. Blocks share
their boundary points with their neighbours, so the pieces meet.

Usage:
  uv run python packages/tack-vis/examples/31_multiblock_contour.py
  uv run python packages/tack-vis/examples/31_multiblock_contour.py --arch metal
  uv run python packages/tack-vis/examples/31_multiblock_contour.py --nblocks 100
"""

import argparse
import time

import numpy as np

import tack
import tack.dataset as td

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--nblocks', type=int, default=8,
                     help='Number of blocks (default 8)')
_parser.add_argument('--block_size', type=int, default=32,
                     help='Cells per dimension per block (default 32)')
_args = _parser.parse_args()
_arch = getattr(tack, _args.arch)
tack.init(arch=_arch)


# ================================================================
# SYNTHETIC MULTI-BLOCK DATA
# ================================================================

@tack.kernel
def compute_gyroid(xyz, scalar):
    """Gyroid: sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)."""
    for p in range(scalar.shape[0]):
        x = xyz.get(p)                      # the block computes point p's position
        scalar[p] = sin(x[0]) * cos(x[1]) + sin(x[1]) * cos(x[2]) + sin(x[2]) * cos(x[0])


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
                n = block_size
                origin = (domain_lo + bi * n * dx, domain_lo + bj * n * dy,
                          domain_lo + bk * n * dz)
                block = td.uniform_grid((n + 1, n + 1, n + 1), origin=origin,
                                        spacing=(dx, dy, dz))
                scalar = tack.empty((block.num_points,))
                compute_gyroid(block.geometry.values, scalar)
                block.fields["gyroid"] = td.Field(td.H1(block), scalar)
                blocks.append(block)

    return blocks, (nbx, nby, nbz), (dx, dy, dz)


# ================================================================
# RUN
# ================================================================

nblocks = _args.nblocks
block_size = _args.block_size
isovalue = 0.0

print("Multi-block contour")
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
[td.contour(block, "gyroid", isovalue) for block in blocks]

# Timed run
t0 = time.perf_counter()
pieces = [td.contour(block, "gyroid", isovalue) for block in blocks]
tack.sync()
t_contour = time.perf_counter() - t0

total_points = sum(piece.num_points for piece in pieces)
total_triangles = sum(piece.num_cells for piece in pieces)
if total_triangles == 0:
    print("  No isosurface produced.")
else:
    print(f"  Points:     {total_points:,}")
    print(f"  Triangles:  {total_triangles:,}")
    print(f"  Time:       {t_contour:.4f}s")

    # Verify connectivity
    for piece in pieces:
        conn = piece.topology.connectivity.numpy()
        assert np.all(conn >= 0) and np.all(conn < piece.num_points)
    print("  Connectivity: OK (all indices in range)")

    # Verify points on isosurface
    pts = np.concatenate([piece.positions() for piece in pieces if piece.num_points])
    n_sample = min(100, len(pts))
    idxs = np.random.choice(len(pts), n_sample, replace=False)
    sx = pts[idxs, 0].astype(np.float64)
    sy = pts[idxs, 1].astype(np.float64)
    sz = pts[idxs, 2].astype(np.float64)
    vals = np.sin(sx)*np.cos(sy) + np.sin(sy)*np.cos(sz) + np.sin(sz)*np.cos(sx)
    max_err = float(np.max(np.abs(vals - isovalue)))
    print(f"  Max scalar error at points: {max_err:.6f}")
    print(f"  Total cells across blocks: {len(blocks) * block_size**3:,}")
