# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trail-following agents, adapted from Taichi's Physarum example.

Copyright (c) The Taichi Authors. Apache-2.0; see LICENSE-APACHE-2.0.txt.
Source: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/physarum.py
Model and visual inspiration: Sage Jenson, https://sagejenson.com/physarum
"""

import argparse

import numpy as np

import tack
from tack import random


# --8<-- [start:model]
@tack.data_oriented
class Agents:
    sense_angle = 0.6
    sense_distance = 4.0
    turn_angle = 0.3

    def __init__(self, n, size):
        self.n = n
        self.size = size
        self.pos = tack.Vector.field(2, dtype=tack.f32, shape=(n,))
        self.heading = tack.field(dtype=tack.f32, shape=(n,))

    @tack.func
    def sense(self, trail, p, angle):
        q = int(p + self.sense_distance * [cos(angle), sin(angle)]) % self.size
        return trail[q]

    @tack.kernel
    def initialize(self):
        for i in range(self.n):
            state = random.seed(i, 0)
            x, state = random.uniform(state)
            y, state = random.uniform(state)
            angle, state = random.uniform(state)
            self.pos[i] = [x * self.size, y * self.size]
            self.heading[i] = angle * 6.283185307
# --8<-- [end:model]


# --8<-- [start:move]
@tack.kernel
def move(agents: tack.template(), trail, frame):
    for i in range(agents.n):
        p, angle = agents.pos[i], agents.heading[i]
        left = agents.sense(trail, p, angle - agents.sense_angle)
        center = agents.sense(trail, p, angle)
        right = agents.sense(trail, p, angle + agents.sense_angle)
        if left > center and left > right:
            angle -= agents.turn_angle
        elif right > center and right > left:
            angle += agents.turn_angle
        elif center < left and center < right:
            coin, state = random.uniform(random.seed(i, frame + 1))
            angle += agents.turn_angle * (2 * int(coin < 0.5) - 1)
        agents.pos[i] = (p + tack.Vector([cos(angle), sin(angle)])) % agents.size
        agents.heading[i] = angle


@tack.kernel
def deposit(agents: tack.template(), trail):
    for i in range(agents.n):
        cell = int(agents.pos[i]) % agents.size
        tack.atomic_add(trail, cell, 1.0)


@tack.kernel
def diffuse(old, new, size, evaporation):
    for i, j in tack.ndrange(size, size):
        total = 0.0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                total += old[(i + di) % size, (j + dj) % size]
        new[i, j] = total * evaporation / 9.0
# --8<-- [end:move]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    n, size, frames = (128, 24, 8) if check else (4096, 128, 300)
    agents = Agents(n, size)
    agents.initialize()
    old = tack.field(dtype=tack.f32, shape=(size, size))
    new = tack.field(dtype=tack.f32, shape=(size, size))
    evaporation = 0.96
    expected_mass = 0.0
    snapshots = []
    # --8<-- [start:loop]
    for frame in range(frames):
        move(agents, old, frame)
        deposit(agents, old)
        diffuse(old, new, size, evaporation)
        old, new = new, old
        expected_mass = (expected_mass + n) * evaporation
        if (frame + 1) % (frames // 3 if not check else frames) == 0:
            snapshots.append(old.to_numpy())
    # --8<-- [end:loop]
    if check:
        trail = old.to_numpy()
        pos = agents.pos.to_numpy(vectors=True)
        assert np.all(np.isfinite(trail)) and trail.min() >= 0
        assert np.all((pos >= 0) & (pos < size))
        # Every deposit adds one, and periodic averaging preserves mass.
        np.testing.assert_allclose(trail.sum(dtype=np.float64), expected_mass, rtol=3e-6)
    return snapshots


def plot(snapshots, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    scale = max(np.log1p(x).max() for x in snapshots)
    for ax, trail, frame in zip(axes, snapshots, (100, 200, 300), strict=True):
        ax.imshow(np.log1p(trail.T), origin="lower", cmap="magma", vmin=0, vmax=scale)
        ax.set_title(f"Frame {frame}")
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
    images = run(args.arch, args.check)
    if args.output:
        if args.check:
            parser.error("use --output without --check")
        plot(images, args.output)
    print("Physarum: check passed" if args.check else "Physarum: simulation complete")
