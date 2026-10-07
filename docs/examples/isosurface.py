# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Generate a sphere, extract its surface, and render the retained Tack fields.

Teaching adaptation of Tack's examples/32_pathtrace.py.
"""

import argparse

import numpy as np

import tack
from tack.algorithms.flying_edges import UniformGrid, flying_edges
from tack.rendering import Actor, Canvas, PerspectiveCamera, PointLight, Scene, render


# --8<-- [start:scalar]
@tack.kernel
def sphere(scalar, grid: tack.template(), count):
    for idx in range(count):
        i = idx % grid.nx_p1
        j = (idx // grid.nx_p1) % grid.ny_p1
        k = idx // grid.nxy_p1
        x, y, z = grid.get_x(i, j, k), grid.get_y(i, j, k), grid.get_z(i, j, k)
        scalar[idx] = x * x + y * y + z * z - 0.7 * 0.7
# --8<-- [end:scalar]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    n, resolution, samples = (12, 48, 1) if check else (32, 256, 8)
    # --8<-- [start:extract]
    grid = UniformGrid(n, n, n, -1.0, -1.0, -1.0, 2.0 / n, 2.0 / n, 2.0 / n)
    scalar = tack.field(dtype=tack.f32, shape=((n + 1) ** 3,))
    sphere(scalar, grid, scalar.size)
    mesh = flying_edges(scalar, grid, isovalue=0.0)
    if mesh is None:
        raise RuntimeError("the level set did not intersect this grid")
    # --8<-- [end:extract]
    # --8<-- [start:render]
    scene = Scene()
    scene.add(Actor(mesh["points_field"], mesh["conn_field"], color=(0.2, 0.6, 0.9), smooth=True))
    scene.add(PointLight(position=(3, 4, 5), intensity=80.0))
    camera = PerspectiveCamera((2, 1.5, 2.5), (0, 0, 0), width=resolution, height=resolution)
    canvas = Canvas(resolution, resolution)
    render(canvas, scene, camera, samples=samples, max_bounces=1)
    image = canvas.to_numpy()
    # --8<-- [end:render]
    if check:
        points, triangles = mesh["points"], mesh["conn"]
        assert points.shape == (mesh["total_points"], 3)
        assert triangles.shape == (mesh["total_tris"], 3)
        assert triangles.min() >= 0 and triangles.max() < len(points)
        residual = np.abs((points**2).sum(axis=1) - 0.7**2)
        assert residual.max() < (2.0 / n) ** 2
        depth = canvas.depth_to_numpy()
        assert np.count_nonzero(depth >= 0) > 50
        assert np.all(np.isfinite(depth))
        assert image.shape == (resolution, resolution, 4) and image.dtype == np.uint8
    return image


def plot(image, output):
    import matplotlib.pyplot as plt

    plt.imsave(output, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", help="Save the rendered image (requires matplotlib)")
    args = parser.parse_args()
    image = run(args.arch, args.check)
    if args.output:
        plot(image, args.output)
    print("Isosurface: check passed" if args.check else "Isosurface: render complete")
