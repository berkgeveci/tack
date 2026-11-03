# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Diffuse heat along a line cut into six blocks, refreshing ghost cells with a Plan
before every step: in one process, or over MPI ranks with --mpi, to the same bits.

    python comm_ghosts.py --check
    mpiexec -n 3 python comm_ghosts.py --mpi --check
"""

import argparse
import hashlib

import numpy as np

import tack
from tack import communication as tc

BLOCKS = 6          # blocks in all, however many ranks share them
CELLS = 40          # cells each block owns
STEPS = 200


# --8<-- [start:kernel]
@tack.kernel
def diffuse(u, out):
    for i in range(1, u.shape[0] - 1):
        out[i] = u[i] + 0.25 * (u[i - 1] - 2.0 * u[i] + u[i + 1])
# --8<-- [end:kernel]


def run(arch="cpu", check=False, mpi=False):
    tack.set_default_device(arch)
    if mpi:
        from mpi4py import MPI      # after the device: a GPU-aware MPI starts on it

        world = tc.World(MPI.COMM_WORLD)
    else:
        world = tc.World()
    if BLOCKS % world.size:
        raise SystemExit(f"run on 1, 2, 3 or 6 ranks, not {world.size}")
    # --8<-- [start:ghosts]
    blocks = tc.Blocks([arch] * (BLOCKS // world.size), world)
    n = CELLS * blocks.count                       # cells in the whole line

    # Block g owns cells g * CELLS to (g + 1) * CELLS - 1. It stores them at
    # 1 to CELLS, with a ghost cell at each end, 0 and CELLS + 1.
    start = np.exp(-((np.arange(n) - n / 3) / 8.0) ** 2)
    u = {g: tack.array(np.pad(start[g * CELLS:(g + 1) * CELLS], 1)) for g in blocks.local}
    v = {g: tack.zeros(CELLS + 2, tack.f64) for g in blocks.local}

    # The plan: each block sends its first owned cell to the block before it
    # and its last to the block after it, which write them into their ghosts.
    sends = {g: {} for g in blocks.local}
    receives = {g: {} for g in blocks.local}
    for g in blocks.local:
        if g > 0:
            sends[g][g - 1], receives[g][g - 1] = [1], [0]
        if g < blocks.count - 1:
            sends[g][g + 1], receives[g][g + 1] = [CELLS], [CELLS + 1]
    ghosts = tc.Plan(blocks, sends, receives)

    for step in range(STEPS):
        ghosts.run(u)                              # refresh every ghost cell
        for g in blocks.local:
            diffuse(u[g], v[g])
        u, v = v, u

    # Every block's owned cells, in order of global id, on every rank.
    owned = blocks.all_gather({g: u[g].numpy()[1:CELLS + 1] for g in blocks.local})
    line = np.concatenate(owned)
    # --8<-- [end:ghosts]
    if check:
        whole, other = tack.array(np.pad(start, 1)), tack.zeros(n + 2, tack.f64)
        for step in range(STEPS):
            diffuse(whole, other)
            whole, other = other, whole
        assert np.array_equal(line, whole.numpy()[1:n + 1])      # bit for bit
    if world.rank == 0:
        digest = hashlib.sha256(line.tobytes()).hexdigest()[:16]
        print(f"{world.size} rank(s), {blocks.count} blocks: max {line.max():.6f}, "
              f"sha256 {digest}")
    return line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mpi", action="store_true", help="run over MPI.COMM_WORLD")
    args = parser.parse_args()
    run(args.arch, args.check, args.mpi)
