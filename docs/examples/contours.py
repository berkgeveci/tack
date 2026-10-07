# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Marching squares using count, scan, and write.

Teaching adaptation of Taichi's marching_squares.py, Copyright (c) The Taichi Authors.
Apache-2.0; see LICENSE-APACHE-2.0.txt.
Source: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/algorithm/marching_squares.py
Uses an analytic circle and paired edge crossings instead of the original noise and table.
"""

import argparse

import numpy as np

import tack
from tack.algorithms import exclusive_scan


@tack.func
def corners(scalar, i, j):
    return [scalar[i, j], scalar[i + 1, j], scalar[i + 1, j + 1], scalar[i, j + 1]]


# --8<-- [start:count]
@tack.kernel
def count_segments(scalar, counts, n):
    for c in range(n * n):
        i, j = c // n, c % n
        s = corners(scalar, i, j)
        crossings = 0
        for edge in range(4):
            if (s[edge] > 0.0) != (s[(edge + 1) % 4] > 0.0):
                crossings += 1
        counts[c] = crossings // 2
# --8<-- [end:count]


# --8<-- [start:write]
@tack.kernel
def write_segments(scalar, offsets, lines, n):
    for c in range(n * n):
        i, j = c // n, c % n
        s = corners(scalar, i, j)
        x = tack.Vector([float(i), float(i + 1), float(i + 1), float(i)])
        y = tack.Vector([float(j), float(j), float(j + 1), float(j + 1)])
        px = tack.local_array(tack.f32, 4)
        py = tack.local_array(tack.f32, 4)
        hits = 0
        for edge in range(4):
            other = (edge + 1) % 4
            if (s[edge] > 0.0) != (s[other] > 0.0):
                t = -s[edge] / (s[other] - s[edge])
                px[hits] = x[edge] + t * (x[other] - x[edge])
                py[hits] = y[edge] + t * (y[other] - y[edge])
                hits += 1
        for k in range(hits // 2):
            slot = offsets[c] + k
            lines[slot, 0] = [px[2 * k] * 2.0 / n - 1.0, py[2 * k] * 2.0 / n - 1.0]
            lines[slot, 1] = [px[2 * k + 1] * 2.0 / n - 1.0, py[2 * k + 1] * 2.0 / n - 1.0]
# --8<-- [end:write]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    n = 24 if check else 48
    axis = np.linspace(-1, 1, n + 1, dtype=np.float32)
    scalar = tack.field_like(axis[:, None] ** 2 + axis[None, :] ** 2 - np.float32(0.7**2))
    # --8<-- [start:allocate]
    counts = tack.field(dtype=tack.i32, shape=(n * n,))
    offsets = tack.field(dtype=tack.i32, shape=(n * n,))
    count_segments(scalar, counts, n)
    total = exclusive_scan(counts, offsets, n * n)
    if total == 0:
        return np.empty((0, 2, 2), dtype=np.float32), n
    lines = tack.Vector.field(2, dtype=tack.f32, shape=(total, 2))
    write_segments(scalar, offsets, lines, n)
    # --8<-- [end:allocate]
    points = lines.to_numpy(vectors=True)
    if check:
        assert points.shape == (total, 2, 2) and total > 0
        assert np.isfinite(points).all()
        # Independent geometry check: linear edge interpolation approximates this circle.
        residual = np.abs(np.sum(points**2, axis=-1) - 0.7**2)
        assert residual.max() < (2.0 / n) ** 2
        assert np.all(np.linalg.norm(points[:, 0] - points[:, 1], axis=1) > 0)
        host_counts = counts.to_numpy()
        host_offsets = offsets.to_numpy()
        np.testing.assert_array_equal(host_offsets, np.cumsum(host_counts) - host_counts)
        assert int(host_counts.sum()) == total
    return points, n


def plot(result, output):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    lines, n = result
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_collection(LineCollection(lines, colors="#2563eb", linewidths=2))
    ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", xlabel="x", ylabel="y")
    grid = np.linspace(-1, 1, n + 1)
    ax.set_xticks(grid[::4], minor=True)
    ax.set_yticks(grid[::4], minor=True)
    ax.grid(which="minor", alpha=0.25)
    ax.set_title(f"{len(lines)} segments from {n} × {n} cells")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", help="Save a figure (requires matplotlib)")
    args = parser.parse_args()
    result = run(args.arch, args.check)
    if args.output:
        plot(result, args.output)
    print("Contours: check passed" if args.check else f"Contours: {len(result[0])} segments")
