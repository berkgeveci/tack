# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Generate a sphere, extract its surface, and render the retained Tack fields.

Teaching adaptation of Tack's examples/32_pathtrace.py.
"""

import argparse

import numpy as np

import tack
import tack.data as td
from tack.rendering import Actor, Canvas, PerspectiveCamera, PointLight, Scene, render


# --8<-- [start:scalar]
@tack.kernel
def sphere(xyz, scalar, radius):
    for p in range(scalar.shape[0]):
        x = xyz.get(p)                      # point p's position
        scalar[p] = x.dot(x) - radius * radius
# --8<-- [end:scalar]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    n, resolution, samples = (12, 48, 1) if check else (32, 256, 8)
    # --8<-- [start:extract]
    grid = td.uniform_grid((n + 1, n + 1, n + 1), origin=(-1.0, -1.0, -1.0),
                           spacing=(2.0 / n, 2.0 / n, 2.0 / n))
    scalar = tack.field(tack.f32, shape=(grid.num_points,))
    sphere(grid.geometry.values, scalar, 0.7)
    grid.fields["s"] = td.Field(td.H1(grid), scalar)
    surface = td.contour(grid, "s", 0.0)
    if surface.num_cells == 0:
        raise RuntimeError("the level set did not intersect this grid")
    # --8<-- [end:extract]
    # --8<-- [start:render]
    positions = surface.geometry.values                # a field of 3-vectors
    points = positions.reshape(positions.size)          # the same memory, flat
    triangles = surface.topology.connectivity           # i32 for a mesh this size
    scene = Scene()
    scene.add(Actor(points, triangles, color=(0.2, 0.6, 0.9), smooth=True))
    scene.add(PointLight(position=(3, 4, 5), intensity=80.0))
    camera = PerspectiveCamera((2, 1.5, 2.5), (0, 0, 0), width=resolution, height=resolution)
    canvas = Canvas(resolution, resolution)
    render(canvas, scene, camera, samples=samples, max_bounces=1)
    image = canvas.to_numpy()
    # --8<-- [end:render]
    if check:
        xyz = surface.positions()
        rows = triangles.to_numpy().reshape(-1, 3)
        assert set(surface.topology.types.to_numpy()) == {5}           # triangles
        assert rows.min() >= 0 and rows.max() < len(xyz)
        residual = np.abs((xyz**2).sum(axis=1) - 0.7**2)
        assert residual.max() < (2.0 / n) ** 2
        depth = canvas.depth_to_numpy()
        assert np.count_nonzero(depth >= 0) > 50
        assert np.all(np.isfinite(depth))
        assert image.shape == (resolution, resolution, 4) and image.dtype == np.uint8
    return image


def save_figure(image, output):
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkIOImage import vtkPNGWriter

    # The Tack canvas has top-first rows; VTK image data has bottom-first rows.
    pixels = np.ascontiguousarray(image[::-1]).reshape(-1, 4)
    data = vtkImageData()
    data.SetDimensions(image.shape[1], image.shape[0], 1)
    data.GetPointData().SetScalars(numpy_to_vtk(pixels, deep=True))
    writer = vtkPNGWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(data)
    writer.Write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", help="Save the rendered image (requires VTK)")
    args = parser.parse_args()
    image = run(args.arch, args.check)
    if args.output:
        save_figure(image, args.output)
    print("Isosurface: check passed" if args.check else "Isosurface: render complete")
