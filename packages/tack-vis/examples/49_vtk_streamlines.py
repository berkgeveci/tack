"""49 -- Streamlines in a VTK pipeline: Tack traces, VTK makes tubes and renders.

A velocity field on an image is written and read back as a .vti file; Tack's
Streamlines algorithm traces paths from seeds VTK makes (vtkPointSource), and
VTK turns the paths into tubes and renders them over the image's outline. The
arrays cross between VTK and Tack shared, never copied.

    vtkXMLImageDataReader             the velocity field (.vti)
      >> Streamlines(field="velocity", seeds=...)   Tack: the paths, as polylines
      >> vtkGeometryFilter >> vtkTubeFilter          VTK: tubes around them
      >> vtkXMLPolyDataWriter                        the tubes (.vtp)
      (rendered with the image's outline, with --png)

The field turns about the z axis and rises: v = (-y, x, rise). Interpolated
trilinearly it is exact, so each path is a helix, and the check is that every
point of a path stays at its seed's distance from the axis, and rises as far
as the time it took says.

Usage:
  uv run python packages/tack-vis/examples/49_vtk_streamlines.py
  uv run python packages/tack-vis/examples/49_vtk_streamlines.py --png helices.png
"""

import argparse
import pathlib

import numpy as np

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu'],
                     help='the VTK algorithms run on the CPU')
_parser.add_argument('--seeds', type=int, default=40, help='number of seeds (default 40)')
_parser.add_argument('--rise', type=float, default=0.1, help='the field\'s z speed')
_parser.add_argument('--output', default='results/vtk_pipelines',
                     help='directory for the files written')
_parser.add_argument('--png', help='render the tubes to this PNG file')
_args = _parser.parse_args()

import vtkmodules.vtkCommonDataModel  # registers the pythonic constructors
from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkFiltersSources import vtkPointSource
from vtkmodules.vtkIOXML import (
    vtkXMLImageDataReader,
    vtkXMLImageDataWriter,
    vtkXMLPolyDataWriter,
)

import tack
from tack.interop.vtk.algorithms import Streamlines

tack.set_default_device(_args.arch)
output = pathlib.Path(_args.output)
output.mkdir(parents=True, exist_ok=True)


# ================================================================
# THE VELOCITY FIELD, WRITTEN AS A .VTI FILE
# ================================================================

image = vtkImageData(dimensions=(41, 41, 41), origin=(-1.0, -1.0, 0.0),
                     spacing=(0.05, 0.05, 0.05))
z, y, x = np.mgrid[0:41, 0:41, 0:41] * 0.05 + np.array([0.0, -1.0, -1.0]).reshape(3, 1, 1, 1)
velocity = np.stack([-y.ravel(), x.ravel(), np.full(x.size, _args.rise)], axis=1)
array = vtkDoubleArray(np.ascontiguousarray(velocity))
array.SetName("velocity")
image.point_data.AddArray(array)
field_path = output / "helix_velocity.vti"
writer = vtkXMLImageDataWriter(file_name=str(field_path))
writer.SetInputData(image)
writer.Write()
print(f"wrote {field_path}: {image.number_of_points} points")


# ================================================================
# THE PIPELINE
# ================================================================

seed_source = vtkPointSource(center=(0.3, 0.0, 0.1), radius=0.25,
                             number_of_points=_args.seeds)()
seeds = np.asarray(seed_source.points.data, np.float64)
seeds[:, 2] = 0.1                                # all start at one height
reader = vtkXMLImageDataReader(file_name=str(field_path))
streamlines = Streamlines(field="velocity", seeds=seeds, step_size=0.02, steps=300)
tubes = vtkTubeFilter(radius=0.006, number_of_sides=10)
paths = (reader >> streamlines)()
if streamlines.error is not None:
    raise streamlines.error
tube_surface = (reader >> streamlines >> vtkGeometryFilter() >> tubes)()
tubes_path = output / "helix_tubes.vtp"
out = vtkXMLPolyDataWriter(file_name=str(tubes_path))
out.SetInputData(tube_surface)
out.Write()
print(f"{paths.number_of_cells} paths, {paths.number_of_points} points; wrote {tubes_path}: "
      f"{tube_surface.number_of_cells} tube polygons")


# ================================================================
# THE CHECK: HELICES
# ================================================================

points = np.asarray(paths.points.data, np.float64)
time = np.asarray(paths.point_data["time"], np.float64)
offsets = np.asarray(paths.cells.offsets_array)
connectivity = np.asarray(paths.cells.connectivity_array)
worst_radius = worst_rise = 0.0
for k in range(paths.number_of_cells):
    ids = connectivity[offsets[k]:offsets[k + 1]]
    path, t = points[ids], time[ids]
    radius = np.linalg.norm(path[:, :2], axis=1)
    worst_radius = max(worst_radius, float(np.abs(radius - radius[0]).max()))
    worst_rise = max(worst_rise, float(np.abs(path[:, 2] - (path[0, 2] + _args.rise * t)).max()))
print(f"every path a helix: radius within {worst_radius:.2e} of its seed's, "
      f"height within {worst_rise:.2e} of rise x time")
assert paths.number_of_cells == len(seeds)
assert worst_radius < 1e-6 and worst_rise < 1e-9


# ================================================================
# RENDERING
# ================================================================

if _args.png:
    import vtkmodules.vtkInteractionStyle
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkWindowToImageFilter,
    )

    renderer = vtkRenderer(background=(0.12, 0.12, 0.16))
    tube_mapper = vtkPolyDataMapper(scalar_visibility=True)
    tube_mapper.SetInputData(tube_surface)
    tube_mapper.SetScalarModeToUsePointFieldData()   # no pythonic form in 9.7.1
    tube_mapper.SelectColorArray("time")
    tube_mapper.scalar_range = (0.0, float(time.max()))
    renderer.AddActor(vtkActor(mapper=tube_mapper))
    outline = vtkPolyDataMapper()
    outline.SetInputConnection((reader >> vtkOutlineFilter()).GetOutputPort())
    renderer.AddActor(vtkActor(mapper=outline))
    window = vtkRenderWindow(off_screen_rendering=True, size=(800, 600))
    window.AddRenderer(renderer)
    renderer.ResetCamera()
    renderer.active_camera.Elevation(-60)
    window.Render()
    png = vtkPNGWriter(file_name=_args.png)
    png.SetInputConnection(vtkWindowToImageFilter(input=window).GetOutputPort())
    png.Write()
    print(f"rendered {_args.png}")
