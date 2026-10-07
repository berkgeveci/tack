# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-pairs and tiled N-body forces, adapted from NVIDIA Warp's tile example.

Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. Apache-2.0;
see LICENSE-APACHE-2.0.txt.
Source: https://github.com/NVIDIA/warp/blob/main/warp/examples/tile/example_tile_nbody.py
Algorithm: Lars Nyland, Mark Harris, Jan Prins, GPU Gems 3, chapter 31.
"""

import argparse

import numpy as np

import tack
from tack.runtime.dispatch import get_backend


@tack.func
def interaction(p, q):
    r = q - p
    inv = 1.0 / sqrt(r.dot(r) + 0.01)
    return r * inv * inv * inv


# --8<-- [start:plain]
@tack.kernel
def acceleration_plain(pos, acceleration, n):
    for i in range(n):
        a = tack.Vector([0.0, 0.0, 0.0])
        for j in range(n):
            a += interaction(pos[i], pos[j])
        acceleration[i] = a
# --8<-- [end:plain]


# --8<-- [start:tiled]
@tack.kernel
def acceleration_tiled(pos, acceleration, n):
    for i in range(n):
        tx = tack.shared(tack.f32, 256)
        ty = tack.shared(tack.f32, 256)
        tz = tack.shared(tack.f32, 256)
        lane = tack.thread_id()
        a = tack.Vector([0.0, 0.0, 0.0])
        for tile in range(n // 256):
            q = pos[tile * 256 + lane]
            tx[lane], ty[lane], tz[lane] = q.x, q.y, q.z
            tack.barrier()                    # finish filling before reading
            for j in range(256):
                a += interaction(pos[i], [tx[j], ty[j], tz[j]])
            tack.barrier()                    # finish reading before reusing
        acceleration[i] = a
# --8<-- [end:tiled]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    n = 256 if check else 512
    xyz = np.random.default_rng(42).normal(size=(n, 3)).astype(np.float32)
    pos = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    acceleration = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    pos.from_numpy(xyz)
    # --8<-- [start:choose]
    if get_backend().supports_workgroups:
        if n % 256:
            raise ValueError("the tiled example needs a multiple of 256 bodies")
        acceleration_tiled(pos, acceleration, n)
    else:
        acceleration_plain(pos, acceleration, n)
    # --8<-- [end:choose]
    result = acceleration.to_numpy(vectors=True)
    if check:
        r = xyz.astype(np.float64)[None, :, :] - xyz.astype(np.float64)[:, None, :]
        distance_sq = (r * r).sum(axis=-1) + 0.01
        reference = (r / distance_sq[..., None] ** 1.5).sum(axis=1)
        np.testing.assert_allclose(result, reference, atol=8e-4, rtol=3e-5)
        if get_backend().supports_workgroups:
            baseline = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
            acceleration_plain(pos, baseline, n)
            np.testing.assert_allclose(result, baseline.to_numpy(vectors=True), atol=8e-4, rtol=3e-5)
    return xyz, result


def plot(result, output):
    import matplotlib.pyplot as plt

    pos, acceleration = result
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pos[:, 0], pos[:, 1], s=4, color="#2563eb")
    scaled = acceleration / (np.linalg.norm(acceleration, axis=1).max() + 1e-12)
    ax.quiver(pos[::8, 0], pos[::8, 1], scaled[::8, 0], scaled[::8, 1], color="#ea580c")
    ax.set(aspect="equal", xlabel="x", ylabel="y", title="Bodies and gravitational acceleration")
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
    print("N-body: check passed" if args.check else "N-body: force calculation complete")
