# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Two hot spots diffusing on a square plate, with zero-temperature edges.

Adapted from Tack's 17_heat_equation.py; see the tutorial for attribution.
"""

import argparse

import numpy as np

import tack


# --8<-- [start:step]
@tack.kernel
def heat_step(old, new, r, n):
    for i, j in tack.ndrange(n, n):
        if 0 < i < n - 1 and 0 < j < n - 1:
            lap = old[i - 1, j] + old[i + 1, j] + old[i, j - 1] + old[i, j + 1]
            new[i, j] = old[i, j] + r * (lap - 4.0 * old[i, j])
        else:
            new[i, j] = 0.0
# --8<-- [end:step]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    n, steps, r = (32, 40, 0.2) if check else (96, 600, 0.2)
    yy, xx = np.mgrid[:n, :n]
    initial = (np.exp(-((xx - n / 3) ** 2 + (yy - n / 3) ** 2) / (n / 12) ** 2)
               + 0.7 * np.exp(-((xx - 2 * n / 3) ** 2 + (yy - 2 * n / 3) ** 2)
                              / (n / 10) ** 2)).astype(np.float32)
    initial[[0, -1], :] = 0
    initial[:, [0, -1]] = 0
    old = tack.field_like(initial)
    new = tack.field(dtype=tack.f32, shape=(n, n))
    snapshots = [initial.copy()]
    reference = initial.copy()
    # --8<-- [start:loop]
    for step in range(steps):
        heat_step(old, new, r, n)
        old, new = new, old                 # swap Python references, not field contents
        if (step + 1) % (steps // 3) == 0:
            snapshots.append(old.to_numpy().copy())
    # --8<-- [end:loop]
    result = old.to_numpy()
    if check:
        for _ in range(steps):
            next_u = np.zeros_like(reference)
            next_u[1:-1, 1:-1] = reference[1:-1, 1:-1] + np.float32(r) * (
                reference[:-2, 1:-1] + reference[2:, 1:-1]
                + reference[1:-1, :-2] + reference[1:-1, 2:]
                - np.float32(4) * reference[1:-1, 1:-1])
            reference = next_u
        np.testing.assert_allclose(result, reference, atol=2e-6, rtol=2e-5)
        assert np.all(result >= 0) and result.max() <= initial.max()
        assert np.count_nonzero(result[[0, -1], :]) == 0
        assert np.count_nonzero(result[:, [0, -1]]) == 0
    return snapshots


def plot(snapshots, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3 * len(snapshots), 3))
    for ax, data, label in zip(axes, snapshots, ("Initial", "200 steps", "400 steps", "600 steps"),
                               strict=True):
        ax.imshow(data, origin="lower", cmap="inferno", vmin=0, vmax=snapshots[0].max())
        ax.set_title(label)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", help="Save a figure (requires matplotlib; use without --check)")
    args = parser.parse_args()
    snapshots = run(args.arch, args.check)
    if args.output:
        if args.check:
            parser.error("use --output without --check to plot the documented timesteps")
        plot(snapshots, args.output)
    print("Heat: check passed" if args.check else "Heat: simulation complete")
