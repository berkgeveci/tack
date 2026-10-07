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
TILE = tack.constant(256)


@tack.kernel
def acceleration_tiled(pos, acceleration, n):
    for i in range(n):
        xyz = tack.shared(tack.f32, TILE * 3)  # folded to 768 scalar components
        lane = tack.thread_id()
        a = tack.Vector([0.0, 0.0, 0.0])
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
    tack.init(arch=getattr(tack, arch))
    n = 256 if check else 512
    xyz = np.random.default_rng(42).normal(size=(n, 3)).astype(np.float32)
    pos = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    acceleration = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    pos.from_numpy(xyz)
    # --8<-- [start:choose]
    if get_backend().supports_workgroups:
        if n % TILE:
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


def save_figure(result, output):
    import vtkmodules.vtkRenderingFreeType
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkGlyph3D
    from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
    from vtkmodules.vtkFiltersSources import vtkArrowSource
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
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
    points = vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))
    bodies = vtkPolyData()
    bodies.SetPoints(points)
    vertices = vtkVertexGlyphFilter()
    vertices.SetInputData(bodies)
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(vertices.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.15, 0.39, 0.92)
    actor.GetProperty().SetPointSize(4)
    actor.GetProperty().RenderPointsAsSpheresOn()
    samples = vtkPolyData()
    origins = vtkPoints()
    origins.SetData(numpy_to_vtk(np.ascontiguousarray(xyz[::8]), deep=True))
    samples.SetPoints(origins)
    scaled = acceleration / (np.linalg.norm(acceleration, axis=1).max() + 1e-12)
    scaled[:, 2] = 0
    samples.GetPointData().SetVectors(numpy_to_vtk(np.ascontiguousarray(scaled[::8]), deep=True))
    arrow = vtkArrowSource()
    arrow.SetShaftRadius(0.04)
    arrow.SetTipRadius(0.15)
    glyphs = vtkGlyph3D()
    glyphs.SetInputData(samples)
    glyphs.SetSourceConnection(arrow.GetOutputPort())
    glyphs.SetVectorModeToUseVector()
    glyphs.SetScaleModeToScaleByVector()
    glyphs.SetScaleFactor(0.8)
    glyphs.OrientOn()
    arrow_mapper = vtkPolyDataMapper()
    arrow_mapper.SetInputConnection(glyphs.GetOutputPort())
    arrow_mapper.ScalarVisibilityOff()
    arrows = vtkActor()
    arrows.SetMapper(arrow_mapper)
    arrows.GetProperty().SetColor(0.92, 0.35, 0.05)
    arrows.GetProperty().LightingOff()
    renderer = vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor)
    renderer.AddActor(arrows)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 1)
    camera.SetFocalPoint(0, 0, 0)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(float(np.abs(xyz).max()) * 1.15)
    title = vtkTextActor()
    title.SetInput("Bodies and gravitational acceleration (x-y)")
    title.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    title.SetPosition(0.5, 0.98)
    title.GetTextProperty().SetJustificationToCentered()
    title.GetTextProperty().SetVerticalJustificationToTop()
    title.GetTextProperty().SetFontSize(20)
    title.GetTextProperty().SetColor(0, 0, 0)
    renderer.AddViewProp(title)
    window = vtkRenderWindow()
    window.SetOffScreenRendering(True)
    window.SetMultiSamples(0)
    window.SetSize(640, 640)
    window.AddRenderer(renderer)
    window.Render()
    capture = vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.ReadFrontBufferOff()
    writer = vtkPNGWriter()
    writer.SetFileName(str(output))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
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
