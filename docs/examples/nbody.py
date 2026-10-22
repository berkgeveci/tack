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
        a = tack.vec3(0.0, 0.0, 0.0)
        for j in range(n):
            a += interaction(pos[i], pos[j])
        acceleration[i] = a
# --8<-- [end:plain]


# --8<-- [start:tiled]
TILE = tack.constant(256)


@tack.kernel
def acceleration_tiled(pos, acceleration, n):
    for i in range(n):
        xyz = tack.shared(tack.f32, TILE * 3)  # folded to 768 scalar components
        lane = tack.thread_id()
        a = tack.vec3(0.0, 0.0, 0.0)
        for tile in range(n // TILE):
            q = pos[tile * TILE + lane]
            xyz[3 * lane], xyz[3 * lane + 1], xyz[3 * lane + 2] = q.x, q.y, q.z
            tack.barrier()                    # finish filling before reading
            for j in range(TILE):
                a += interaction(pos[i], [xyz[3 * j], xyz[3 * j + 1], xyz[3 * j + 2]])
            tack.barrier()                    # finish reading before reusing
        acceleration[i] = a
# --8<-- [end:tiled]


def run(arch="cpu", check=False):
    tack.set_default_device(arch)
    n = 256 if check else 512
    xyz = np.random.default_rng(42).normal(size=(n, 3)).astype(np.float32)
    pos = tack.empty((n,), tack.vec3(tack.f32))
    acceleration = tack.empty((n,), tack.vec3(tack.f32))
    tack.copyto(pos, xyz)
    # --8<-- [start:choose]
    if get_backend().supports_workgroups:
        if n % TILE:
            raise ValueError("the tiled example needs a multiple of 256 bodies")
        acceleration_tiled(pos, acceleration, n)
    else:
        acceleration_plain(pos, acceleration, n)
    # --8<-- [end:choose]
    result = acceleration.numpy()
    if check:
        r = xyz.astype(np.float64)[None, :, :] - xyz.astype(np.float64)[:, None, :]
        distance_sq = (r * r).sum(axis=-1) + 0.01
        reference = (r / distance_sq[..., None] ** 1.5).sum(axis=1)
        np.testing.assert_allclose(result, reference, atol=8e-4, rtol=3e-5)
        if get_backend().supports_workgroups:
            baseline = tack.empty((n,), tack.vec3(tack.f32))
            acceleration_plain(pos, baseline, n)
            np.testing.assert_allclose(result, baseline.numpy(), atol=8e-4, rtol=3e-5)
    return xyz, result


def save_figure(result, output):
    import vtkmodules.vtkRenderingFreeType
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.vtkCommonCore import VTK_TEXT_CENTERED, VTK_TEXT_TOP, vtkFloatArray, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCore import VTK_SCALE_BY_VECTOR, VTK_USE_VECTOR, vtkGlyph3D
    from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
    from vtkmodules.vtkFiltersSources import vtkArrowSource
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        VTK_NORMALIZED_VIEWPORT,
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkTextActor,
        vtkWindowToImageFilter,
    )

    pos, acceleration = result
    # Project both positions and forces into the x-y plane.
    xyz = pos.copy()
    xyz[:, 2] = 0
    # vtkFloatArray keeps the float32 positions: vtkPoints(data=numpy) would store doubles.
    bodies = vtkPolyData(points=vtkPoints(data=vtkFloatArray(xyz)))
    mapper = vtkPolyDataMapper()
    bodies >> vtkVertexGlyphFilter() >> mapper
    actor = vtkActor(mapper=mapper)
    actor.property.color = (0.15, 0.39, 0.92)
    actor.property.point_size = 4
    actor.property.render_points_as_spheres = True
    samples = vtkPolyData(points=vtkPoints(data=vtkFloatArray(np.ascontiguousarray(xyz[::8]))))
    scaled = acceleration / (np.linalg.norm(acceleration, axis=1).max() + 1e-12)
    scaled[:, 2] = 0
    # Attribute data has no settable vectors property: SetVectors makes them active.
    samples.point_data.SetVectors(vtkFloatArray(np.ascontiguousarray(scaled[::8])))
    glyphs = vtkGlyph3D(vector_mode=VTK_USE_VECTOR, scale_mode=VTK_SCALE_BY_VECTOR,
                        scale_factor=0.8, orient=True)
    arrow = vtkArrowSource(shaft_radius=0.04, tip_radius=0.15)
    # The glyph source is input port 1, which has no property or >> form.
    glyphs.SetSourceConnection(arrow.output_port)
    arrow_mapper = vtkPolyDataMapper(scalar_visibility=False)
    samples >> glyphs >> arrow_mapper
    arrows = vtkActor(mapper=arrow_mapper)
    arrows.property.color = (0.92, 0.35, 0.05)
    arrows.property.lighting = False
    renderer = vtkRenderer(background=(1, 1, 1))
    renderer.AddActor(actor)
    renderer.AddActor(arrows)
    camera = renderer.active_camera
    camera.position = (0, 0, 1)
    camera.focal_point = (0, 0, 0)
    camera.parallel_projection = True
    camera.parallel_scale = float(np.abs(xyz).max()) * 1.15
    title = vtkTextActor(input="Bodies and gravitational acceleration (x-y)", position=(0.5, 0.98))
    title.position_coordinate.coordinate_system = VTK_NORMALIZED_VIEWPORT
    text = title.text_property
    text.justification = VTK_TEXT_CENTERED
    text.vertical_justification = VTK_TEXT_TOP
    text.font_size = 20
    text.color = (0, 0, 0)
    renderer.AddViewProp(title)
    window = vtkRenderWindow(off_screen_rendering=True, multi_samples=0, size=(640, 640))
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
    parser.add_argument("--output", help="Save a figure (requires VTK)")
    args = parser.parse_args()
    result = run(args.arch, args.check)
    if args.output:
        save_figure(result, args.output)
    print("N-body: check passed" if args.check else "N-body: force calculation complete")
