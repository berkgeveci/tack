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
    for i, j in tack.ndindex(n, n):
        if 0 < i < n - 1 and 0 < j < n - 1:
            lap = old[i - 1, j] + old[i + 1, j] + old[i, j - 1] + old[i, j + 1]
            new[i, j] = old[i, j] + r * (lap - 4.0 * old[i, j])
        else:
            new[i, j] = 0.0
# --8<-- [end:step]


def run(arch="cpu", check=False):
    tack.set_default_device(arch)
    n, steps, r = (32, 40, 0.2) if check else (96, 600, 0.2)
    yy, xx = np.mgrid[:n, :n]
    initial = (np.exp(-((xx - n / 3) ** 2 + (yy - n / 3) ** 2) / (n / 12) ** 2)
               + 0.7 * np.exp(-((xx - 2 * n / 3) ** 2 + (yy - 2 * n / 3) ** 2)
                              / (n / 10) ** 2)).astype(np.float32)
    initial[[0, -1], :] = 0
    initial[:, [0, -1]] = 0
    old = tack.array(initial)
    new = tack.empty((n, n))
    snapshots = [initial.copy()]
    reference = initial.copy()
    # --8<-- [start:loop]
    for step in range(steps):
        heat_step(old, new, r, n)
        old, new = new, old                 # swap Python references, not field contents
        if (step + 1) % (steps // 3) == 0:
            snapshots.append(old.numpy().copy())
    # --8<-- [end:loop]
    result = old.numpy()
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
    scale = float(snapshots[0].max())
    # One shared colour range makes changes across panels comparable.
    colour = vtkColorTransferFunction()
    for fraction, rgb in ((0.0, (0.0, 0.0, 0.0)), (0.25, (0.2, 0.04, 0.35)),
                          (0.5, (0.6, 0.1, 0.35)), (0.75, (0.95, 0.4, 0.1)),
                          (1.0, (1.0, 0.95, 0.5))):
        colour.AddRGBPoint(fraction * scale, *rgb)
    window = vtkRenderWindow(off_screen_rendering=True, multi_samples=0,
                             size=(400 * len(snapshots), 440))
    for index, (snapshot, label) in enumerate(zip(snapshots, ("Initial", "200 steps", "400 steps", "600 steps"), strict=True)):
        data = np.ascontiguousarray(snapshot)
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
    snapshots = run(args.arch, args.check)
    if args.output:
        if args.check:
            parser.error("use --output without --check to plot the documented timesteps")
        save_figure(snapshots, args.output)
    print("Heat: check passed" if args.check else "Heat: simulation complete")
