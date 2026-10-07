# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A falling fluid block using MLS-MPM, adapted from Taichi's mpm88.py.

Copyright (c) The Taichi Authors; original example by Yuanming Hu.
Apache-2.0; see LICENSE-APACHE-2.0.txt.
Source: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm88.py
"""

import argparse

import numpy as np

import tack

GRID = tack.constant(64)
DX = tack.constant(1.0 / GRID)
DT = tack.constant(2e-4)
VOLUME = tack.constant((DX * 0.5) ** 2)
MASS = tack.constant(VOLUME)  # density is one
STIFFNESS = tack.constant(400.0)


@tack.kernel
def clear_grid(momentum, mass):
    for i, j in tack.ndrange(GRID, GRID):
        momentum[i, j] = 0.0
        mass[i, j] = 0.0


# --8<-- [start:scatter]
@tack.kernel
def particle_to_grid(x, v, C, J, momentum, mass, n):
    for p in range(n):
        xp = x[p] / DX
        base = int(xp - 0.5)
        fx = xp - base
        weights = tack.Matrix([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2,
                               0.5 * (fx - 0.5) ** 2])
        stress = -DT * 4.0 * STIFFNESS * VOLUME * (J[p] - 1.0) / DX**2
        affine = tack.Matrix([[stress, 0.0], [0.0, stress]]) + MASS * C[p]
        for i, j in tack.ndrange(3, 3):
            offset = tack.Vector([i, j])
            dpos = (offset - fx) * DX
            weight = weights[i, 0] * weights[j, 1]
            tack.atomic_add(momentum, base + offset, weight * (MASS * v[p] + affine @ dpos))
            tack.atomic_add(mass, base + offset, weight * MASS)
# --8<-- [end:scatter]


@tack.kernel
def grid_update(momentum, mass):
    for i, j in tack.ndrange(GRID, GRID):
        if mass[i, j] > 0:
            momentum[i, j] /= mass[i, j]  # momentum becomes velocity
            momentum[i, j].y -= DT * 9.8
        if i < 3 and momentum[i, j].x < 0:
            momentum[i, j].x = 0
        if i >= GRID - 3 and momentum[i, j].x > 0:
            momentum[i, j].x = 0
        if j < 3 and momentum[i, j].y < 0:
            momentum[i, j].y = 0
        if j >= GRID - 3 and momentum[i, j].y > 0:
            momentum[i, j].y = 0


# --8<-- [start:gather]
@tack.kernel
def grid_to_particle(x, v, C, J, grid_v, n):
    for p in range(n):
        xp = x[p] / DX
        base = int(xp - 0.5)
        fx = xp - base
        weights = tack.Matrix([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2,
                               0.5 * (fx - 0.5) ** 2])
        new_v = tack.Vector([0.0, 0.0])
        new_C = tack.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i, j in tack.ndrange(3, 3):
            offset = tack.Vector([i, j])
            dpos = (offset - fx) * DX
            weight = weights[i, 0] * weights[j, 1]
            gv = grid_v[base + offset]
            new_v += weight * gv
            new_C += 4.0 * weight * gv.outer_product(dpos) / DX**2
        v[p] = new_v
        x[p] += DT * new_v
        J[p] *= 1.0 + DT * new_C.trace()
        C[p] = new_C
# --8<-- [end:gather]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    n, steps = (128, 10) if check else (2048, 900)
    initial = (np.random.default_rng(0).random((n, 2)) * 0.3 + [0.35, 0.45]).astype(np.float32)
    x = tack.Vector.field(2, dtype=tack.f32, shape=(n,))
    v = tack.Vector.field(2, dtype=tack.f32, shape=(n,))
    C = tack.Matrix.field(2, 2, dtype=tack.f32, shape=(n,))
    J = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(initial)
    J.fill(1.0)
    momentum = tack.Vector.field(2, dtype=tack.f32, shape=(GRID, GRID))
    mass = tack.field(dtype=tack.f32, shape=(GRID, GRID))
    snapshots = [initial.copy()]
    # --8<-- [start:loop]
    for step in range(steps):
        clear_grid(momentum, mass)
        particle_to_grid(x, v, C, J, momentum, mass, n)
        grid_update(momentum, mass)
        grid_to_particle(x, v, C, J, momentum, n)
        if (step + 1) % (steps // 3 if not check else steps) == 0:
            snapshots.append(x.to_numpy(vectors=True))
    # --8<-- [end:loop]
    if check:
        positions = x.to_numpy(vectors=True)
        velocity = v.to_numpy(vectors=True)
        # The weights partition unity, so scatter must conserve particle mass.
        np.testing.assert_allclose(mass.to_numpy().sum(dtype=np.float64), n * float(MASS), rtol=2e-6)
        assert np.all(np.isfinite(positions)) and np.all((positions > 0) & (positions < 1))
        assert np.all(J.to_numpy() > 0)
        # Before contact, a uniform gravitational acceleration moves every particle together.
        np.testing.assert_allclose(velocity[:, 1], -9.8 * float(DT) * steps, atol=2e-5)
        np.testing.assert_allclose(positions[:, 0], initial[:, 0], atol=2e-6)
    return snapshots


def save_figure(snapshots, output):
    import vtkmodules.vtkRenderingFreeType
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
    from vtkmodules.vtkFiltersSources import vtkOutlineSource
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkTextActor,
        vtkWindowToImageFilter,
    )

    window = vtkRenderWindow()
    window.SetOffScreenRendering(True)
    window.SetMultiSamples(0)
    window.SetSize(1600, 440)
    for index, (xy, step) in enumerate(zip(snapshots, (0, 300, 600, 900), strict=True)):
        points = vtkPoints()
        points.SetData(numpy_to_vtk(np.column_stack((xy, np.zeros(len(xy)))), deep=True))
        particles = vtkPolyData()
        particles.SetPoints(points)
        vertices = vtkVertexGlyphFilter()
        vertices.SetInputData(particles)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(vertices.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.15, 0.39, 0.92)
        actor.GetProperty().SetPointSize(2)
        outline = vtkOutlineSource()
        outline.SetBounds(0, 1, 0, 1, 0, 0)
        border_mapper = vtkPolyDataMapper()
        border_mapper.SetInputConnection(outline.GetOutputPort())
        border = vtkActor()
        border.SetMapper(border_mapper)
        border.GetProperty().SetColor(0.2, 0.2, 0.2)
        renderer = vtkRenderer()
        renderer.SetBackground(1, 1, 1)
        renderer.SetViewport(index / 4, 0, (index + 1) / 4, 1)
        renderer.AddActor(actor)
        renderer.AddActor(border)
        title = vtkTextActor()
        title.SetInput(f"Step {step}")
        title.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        title.SetPosition(0.5, 0.98)
        title.GetTextProperty().SetJustificationToCentered()
        title.GetTextProperty().SetVerticalJustificationToTop()
        title.GetTextProperty().SetFontSize(20)
        title.GetTextProperty().SetColor(0, 0, 0)
        renderer.AddViewProp(title)
        camera = renderer.GetActiveCamera()
        camera.SetPosition(0.5, 0.5, 1)
        camera.SetFocalPoint(0.5, 0.5, 0)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(0.6)
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
    parser.add_argument("--output", help="Save a figure (requires VTK; use without --check)")
    args = parser.parse_args()
    snapshots = run(args.arch, args.check)
    if args.output:
        if args.check:
            parser.error("use --output without --check")
        save_figure(snapshots, args.output)
    print("MPM: check passed" if args.check else "MPM: simulation complete")
