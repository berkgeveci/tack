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
@tack.device_class
class Agents:
    sense_angle = 0.6
    sense_distance = 4.0
    turn_angle = 0.3

    def __init__(self, n, size):
        self.n = n
        self.size = size
        self.pos = tack.empty((n,), tack.vec2(tack.f32))
        self.heading = tack.empty((n,))

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
def move(agents, trail, frame):
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
        agents.pos[i] = (p + tack.vec2(cos(angle), sin(angle))) % agents.size
        agents.heading[i] = angle


@tack.kernel
def deposit(agents, trail):
    for i in range(agents.n):
        cell = int(agents.pos[i]) % agents.size
        tack.atomic_add(trail, cell, 1.0)


@tack.kernel
def diffuse(old, new, size, evaporation):
    for i, j in tack.ndindex(size, size):
        total = 0.0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                total += old[(i + di) % size, (j + dj) % size]
        new[i, j] = total * evaporation / 9.0
# --8<-- [end:move]


def run(arch="cpu", check=False):
    tack.set_default_device(arch)
    n, size, frames = (128, 24, 8) if check else (4096, 128, 300)
    agents = Agents(n, size)
    agents.initialize()
    old = tack.zeros((size, size))
    new = tack.zeros((size, size))
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
            snapshots.append(old.numpy())
    # --8<-- [end:loop]
    if check:
        trail = old.numpy()
        pos = agents.pos.numpy()
        assert np.all(np.isfinite(trail)) and trail.min() >= 0
        assert np.all((pos >= 0) & (pos < size))
        # Every deposit adds one, and periodic averaging preserves mass.
        np.testing.assert_allclose(trail.sum(dtype=np.float64), expected_mass, rtol=3e-6)
    return snapshots


def save_figure(snapshots, output):
    import vtkmodules.vtkRenderingFreeType  # register text rendering
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  # register the rendering backend
    from vtkmodules.vtkCommonCore import VTK_RGB, VTK_TEXT_CENTERED, VTK_TEXT_TOP, vtkFloatArray
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkImagingCore import vtkImageMapToColors
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        VTK_NORMALIZED_VIEWPORT,
        vtkColorTransferFunction,
        vtkImageActor,
        vtkRenderer,
        vtkRenderWindow,
        vtkTextActor,
        vtkWindowToImageFilter,
    )
    scale = max(float(np.log1p(x).max()) for x in snapshots)
    # One shared colour range makes changes across panels comparable.
    colour = vtkColorTransferFunction()
    for fraction, rgb in ((0.0, (0.0, 0.0, 0.0)), (0.25, (0.2, 0.04, 0.35)),
                          (0.5, (0.6, 0.1, 0.35)), (0.75, (0.95, 0.4, 0.1)),
                          (1.0, (1.0, 0.95, 0.5))):
        colour.AddRGBPoint(fraction * scale, *rgb)
    window = vtkRenderWindow(off_screen_rendering=True, multi_samples=0,
                             size=(400 * len(snapshots), 440))
    for index, (snapshot, label) in enumerate(zip(snapshots, ("Frame 100", "Frame 200", "Frame 300"), strict=True)):
        data = np.ascontiguousarray(np.log1p(snapshot.T))
        image = vtkImageData(dimensions=(data.shape[1], data.shape[0], 1))
        # Attribute data has no settable scalars property: SetScalars makes them active.
        image.point_data.SetScalars(vtkFloatArray(data.ravel()))
        actor = vtkImageActor(interpolate=False)
        image >> vtkImageMapToColors(lookup_table=colour, output_format=VTK_RGB) >> actor.mapper
        renderer = vtkRenderer(background=(1, 1, 1),
                               viewport=(index / len(snapshots), 0, (index + 1) / len(snapshots), 1))
        renderer.AddActor(actor)
        title = vtkTextActor(input=label, position=(0.5, 0.98))
        title.position_coordinate.coordinate_system = VTK_NORMALIZED_VIEWPORT
        text = title.text_property
        text.justification = VTK_TEXT_CENTERED
        text.vertical_justification = VTK_TEXT_TOP
        text.font_size = 20
        text.color = (0, 0, 0)
        renderer.AddViewProp(title)
        camera = renderer.active_camera
        cx, cy = (data.shape[1] - 1) / 2, (data.shape[0] - 1) / 2
        camera.position = (cx, cy, 1)
        camera.focal_point = (cx, cy, 0)
        camera.parallel_projection = True
        camera.parallel_scale = data.shape[0] * 0.6
        window.AddRenderer(renderer)
    window.Render()
    capture = vtkWindowToImageFilter(input=window, read_front_buffer=False)
    writer = vtkPNGWriter(file_name=str(output))
    capture >> writer
    writer.Write()  # calling the writer, or updating it, writes raw pixels instead of a PNG
    window.Finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", help="Save a figure (requires VTK; use without --check)")
    args = parser.parse_args()
    images = run(args.arch, args.check)
    if args.output:
        if args.check:
            parser.error("use --output without --check")
        save_figure(images, args.output)
    print("Physarum: check passed" if args.check else "Physarum: simulation complete")
