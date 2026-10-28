"""47 -- Tack's filters in a VTK pipeline: read, filter, write, render.

VTK's readers and writers do the I/O and VTK renders; Tack's filters sit in the
pipeline between them as VTK algorithms (tack.interop.vtk.algorithms), sharing
memory with VTK both ways, so no array is copied going in or out.

The pipeline:

    vtkXMLUnstructuredGridReader      a tetrahedral mesh of VTK's wavelet
      >> Gradients(field="RTData")     Tack: the gradient on the points
      >> Contour(field="RTData")       Tack: the isosurface, carrying the gradient
      >> vtkXMLUnstructuredGridWriter  the surface, as a .vtu file
      (rendered, coloured by the gradient's magnitude, with --png)

The mesh is written first (the wavelet tetrahedralized by Tack), so the example
reads a real file. The isosurface is checked against VTK's own contour of the
same file: the same number of triangles and the same area.

Usage:
  uv run python packages/tack-vis/examples/47_vtk_pipeline.py
  uv run python packages/tack-vis/examples/47_vtk_pipeline.py --png surface.png
"""

import argparse
import pathlib

import numpy as np

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu'],
                     help='the VTK algorithms run on the CPU')
_parser.add_argument('--extent', type=int, default=20,
                     help='wavelet extent: (2*extent+1)^3 points (default 20)')
_parser.add_argument('--value', type=float, default=160.0, help='the isovalue')
_parser.add_argument('--output', default='results/vtk_pipelines',
                     help='directory for the files written')
_parser.add_argument('--png', help='render the isosurface to this PNG file')
_args = _parser.parse_args()

import vtkmodules.vtkCommonDataModel  # registers the pythonic constructors
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource
from vtkmodules.vtkIOXML import (
    vtkXMLUnstructuredGridReader,
    vtkXMLUnstructuredGridWriter,
)

import tack
from tack.interop.vtk.algorithms import Contour, Gradients, Tetrahedralize

tack.set_default_device(_args.arch)
output = pathlib.Path(_args.output)
output.mkdir(parents=True, exist_ok=True)


def write(data, path):
    """Write ``data``: a writer called on its input writes it (and returns ``None``
    whether or not it could, so check the file)."""
    path.unlink(missing_ok=True)
    vtkXMLUnstructuredGridWriter(file_name=str(path))(data)
    if not path.exists():
        raise RuntimeError(f"could not write {path}")


# ================================================================
# THE INPUT FILE: VTK'S WAVELET, TETRAHEDRALIZED BY TACK
# ================================================================

e = _args.extent
mesh = (vtkRTAnalyticSource(whole_extent=(-e, e, -e, e, -e, e)) >> Tetrahedralize())()
mesh_path = output / "wavelet_tets.vtu"
write(mesh, mesh_path)
print(f"wrote {mesh_path}: {mesh.number_of_cells} tetrahedra, {mesh.number_of_points} points")


# ================================================================
# THE PIPELINE
# ================================================================

reader = vtkXMLUnstructuredGridReader(file_name=str(mesh_path))
gradients = Gradients(field="RTData", at="points", name="gradient")
contour = Contour(field="RTData", value=_args.value)
surface = (reader >> gradients >> contour)()
for algorithm in (gradients, contour):
    if algorithm.error is not None:
        raise algorithm.error
surface_path = output / "isosurface.vtu"
write(surface, surface_path)
print(f"wrote {surface_path}: {surface.number_of_cells} triangles, "
      f"arrays {sorted(surface.point_data.keys())}")


# ================================================================
# THE SAME ISOSURFACE BY VTK
# ================================================================

theirs = vtkContourFilter(compute_scalars=False)
theirs.SetValue(0, _args.value)
theirs.SetInputArrayToProcess(0, 0, 0, 0, "RTData")
reference = (reader >> theirs)()


def area(data):
    return float(vtkCellSizeFilter()(data).cell_data["Area"].sum())


ours_area, their_area = area(surface), area(reference)
print(f"isosurface at {_args.value:g}: Tack {surface.number_of_cells} triangles, area "
      f"{ours_area:.6g}; VTK {reference.number_of_cells}, area {their_area:.6g}")
assert surface.number_of_cells == reference.number_of_cells
assert abs(ours_area - their_area) <= 1e-5 * their_area
assert "gradient" in surface.point_data.keys()


# ================================================================
# RENDERING
# ================================================================

if _args.png:
    import vtkmodules.vtkInteractionStyle
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.vtkCommonCore import vtkScalarsToColors
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        VTK_GET_ARRAY_BY_NAME,
        VTK_SCALAR_MODE_USE_POINT_FIELD_DATA,
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkWindowToImageFilter,
    )

    magnitude = np.linalg.norm(surface.point_data["gradient"], axis=1)
    mapper = vtkPolyDataMapper(scalar_visibility=True,
                               scalar_mode=VTK_SCALAR_MODE_USE_POINT_FIELD_DATA,
                               array_access_mode=VTK_GET_ARRAY_BY_NAME, array_name="gradient",
                               scalar_range=(0.0, float(magnitude.max())))
    mapper.lookup_table.vector_mode = vtkScalarsToColors.MAGNITUDE
    surface >> vtkGeometryFilter() >> mapper
    renderer = vtkRenderer(background=(0.12, 0.12, 0.16))
    renderer.AddActor(vtkActor(mapper=mapper))
    window = vtkRenderWindow(off_screen_rendering=True, size=(800, 600))
    window.AddRenderer(renderer)
    renderer.ResetCamera()
    renderer.active_camera.Azimuth(30)
    renderer.active_camera.Elevation(20)
    window.Render()
    # A PNG writer called on its input writes raw pixels (VTK 9.7.1): connect it
    # and Write().
    png = vtkPNGWriter(file_name=_args.png)
    vtkWindowToImageFilter(input=window) >> png
    png.Write()
    print(f"rendered {_args.png}")
