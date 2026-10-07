# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Marching squares using count, scan, and write.

Teaching adaptation of Taichi's marching_squares.py, Copyright (c) The Taichi Authors.
Apache-2.0; see LICENSE-APACHE-2.0.txt.
Source: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/algorithm/marching_squares.py
Uses an analytic circle and paired edge crossings instead of the original noise and table.
"""

import argparse

import numpy as np

import tack
from tack.algorithms import exclusive_scan


@tack.func
def corners(scalar, i, j):
    return [scalar[i, j], scalar[i + 1, j], scalar[i + 1, j + 1], scalar[i, j + 1]]


# --8<-- [start:count]
@tack.kernel
def count_segments(scalar, counts, n):
    for c in range(n * n):
        i, j = c // n, c % n
        s = corners(scalar, i, j)
        crossings = 0
        for edge in range(4):
            if (s[edge] > 0.0) != (s[(edge + 1) % 4] > 0.0):
                crossings += 1
        counts[c] = crossings // 2
# --8<-- [end:count]


# --8<-- [start:write]
@tack.kernel
def write_segments(scalar, offsets, lines, n):
    for c in range(n * n):
        i, j = c // n, c % n
        s = corners(scalar, i, j)
        x = tack.Vector([float(i), float(i + 1), float(i + 1), float(i)])
        y = tack.Vector([float(j), float(j), float(j + 1), float(j + 1)])
        px = tack.local_array(tack.f32, 4)
        py = tack.local_array(tack.f32, 4)
        hits = 0
        for edge in range(4):
            other = (edge + 1) % 4
            if (s[edge] > 0.0) != (s[other] > 0.0):
                t = -s[edge] / (s[other] - s[edge])
                px[hits] = x[edge] + t * (x[other] - x[edge])
                py[hits] = y[edge] + t * (y[other] - y[edge])
                hits += 1
        for k in range(hits // 2):
            slot = offsets[c] + k
            lines[slot, 0] = [px[2 * k] * 2.0 / n - 1.0, py[2 * k] * 2.0 / n - 1.0]
            lines[slot, 1] = [px[2 * k + 1] * 2.0 / n - 1.0, py[2 * k + 1] * 2.0 / n - 1.0]
# --8<-- [end:write]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    n = 24 if check else 48
    axis = np.linspace(-1, 1, n + 1, dtype=np.float32)
    scalar = tack.field_like(axis[:, None] ** 2 + axis[None, :] ** 2 - np.float32(0.7**2))
    # --8<-- [start:allocate]
    counts = tack.field(dtype=tack.i32, shape=(n * n,))
    offsets = tack.field(dtype=tack.i32, shape=(n * n,))
    count_segments(scalar, counts, n)
    total = exclusive_scan(counts, offsets, n * n)
    if total == 0:
        return np.empty((0, 2, 2), dtype=np.float32), n
    lines = tack.Vector.field(2, dtype=tack.f32, shape=(total, 2))
    write_segments(scalar, offsets, lines, n)
    # --8<-- [end:allocate]
    points = lines.to_numpy(vectors=True)
    if check:
        assert points.shape == (total, 2, 2) and total > 0
        assert np.isfinite(points).all()
        # Independent geometry check: linear edge interpolation approximates this circle.
        residual = np.abs(np.sum(points**2, axis=-1) - 0.7**2)
        assert residual.max() < (2.0 / n) ** 2
        assert np.all(np.linalg.norm(points[:, 0] - points[:, 1], axis=1) > 0)
        host_counts = counts.to_numpy()
        host_offsets = offsets.to_numpy()
        np.testing.assert_array_equal(host_offsets, np.cumsum(host_counts) - host_counts)
        assert int(host_counts.sum()) == total
    return points, n


def save_figure(result, output):
    import vtkmodules.vtkRenderingFreeType
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor2D
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkTextActor,
        vtkWindowToImageFilter,
    )

    segments, n = result
    xy = segments.reshape(-1, 2)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.column_stack((xy, np.zeros(len(xy)))), deep=True))
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(segments) + 1, dtype=np.int64) * 2,
                                        deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(xy), dtype=np.int64), deep=True))
    lines = vtkPolyData()
    lines.SetPoints(points)
    lines.SetLines(cells)
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(lines)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.15, 0.39, 0.92)
    actor.GetProperty().SetLineWidth(3)
    renderer = vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 1)
    camera.SetFocalPoint(0, 0, 0)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(1.3)
    axes = vtkCubeAxesActor2D()
    axes.SetBounds(-1, 1, -1, 1, 0, 0)
    axes.SetCamera(camera)
    axes.SetXLabel("x")
    axes.SetYLabel("y")
    axes.SetNumberOfLabels(5)
    axes.SetZAxisVisibility(False)
    axes.GetAxisTitleTextProperty().SetColor(0, 0, 0)
    axes.GetAxisLabelTextProperty().SetColor(0, 0, 0)
    axes.GetProperty().SetColor(0.2, 0.2, 0.2)
    renderer.AddViewProp(axes)
    title = vtkTextActor()
    title.SetInput(f"{len(segments)} segments from {n} x {n} cells")
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
    print("Contours: check passed" if args.check else f"Contours: {len(result[0])} segments")
