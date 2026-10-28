"""48 -- A dataset in pieces through a VTK pipeline: ghost cells, contour, merge.

A vtkPartitionedDataSet goes through Tack's VTK algorithms whole: filters that
work piece by piece run on each, and those whose answer needs every piece
(GhostCells, GenerateGlobalIds, MergePieces) take them together. Every array is
shared with VTK, never copied, on the way in and out.

The pipeline:

    vtkXMLPartitionedDataSetReader    the pieces of a tetrahedral mesh (.vtpd)
      >> GhostCells()                 Tack: a layer of each neighbour's cells
      >> Contour(field="RTData")      Tack: each piece's isosurface, its ghost
                                      triangles marked as the neighbour's
      >> MergePieces()                Tack: one surface, shared points once
      >> vtkXMLUnstructuredGridWriter

The pieces are written first: VTK's wavelet cut into slabs (vtkExtractVOI),
tetrahedralized by Tack, so they meet face to face. The merged surface is
checked against the isosurface of the whole mesh, contoured in one piece.

Usage:
  uv run python packages/tack-vis/examples/48_vtk_pieces.py
  uv run python packages/tack-vis/examples/48_vtk_pieces.py --pieces 6 --png pieces.png
"""

import argparse
import itertools
import pathlib

import numpy as np

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu'],
                     help='the VTK algorithms run on the CPU')
_parser.add_argument('--extent', type=int, default=16,
                     help='wavelet extent: (2*extent+1)^3 points (default 16)')
_parser.add_argument('--pieces', type=int, default=4, help='slabs along x (default 4)')
_parser.add_argument('--value', type=float, default=160.0, help='the isovalue')
_parser.add_argument('--output', default='results/vtk_pipelines',
                     help='directory for the files written')
_parser.add_argument('--png', help='render the pieces\' isosurfaces to this PNG file')
_args = _parser.parse_args()

import vtkmodules.vtkCommonDataModel  # registers the pythonic constructors
from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet
from vtkmodules.vtkImagingCore import vtkExtractVOI, vtkRTAnalyticSource
from vtkmodules.vtkIOParallelXML import vtkXMLPartitionedDataSetWriter
from vtkmodules.vtkIOXML import (
    vtkXMLPartitionedDataSetReader,
    vtkXMLUnstructuredGridWriter,
)

import tack
from tack.interop.vtk.algorithms import (
    Contour,
    GhostCells,
    MergePieces,
    Tetrahedralize,
)

tack.set_default_device(_args.arch)
output = pathlib.Path(_args.output)
output.mkdir(parents=True, exist_ok=True)


def checked(algorithm, input):
    """``algorithm``'s output for ``input``, raising what it raised."""
    out = algorithm(input)
    if algorithm.error is not None:
        raise algorithm.error
    return out


def write(writer, data, path):
    """``data`` written by ``writer`` (a writer called on its input returns ``None``
    whether or not it wrote, so check the file)."""
    path.unlink(missing_ok=True)
    writer(data)
    if not path.exists():
        raise RuntimeError(f"could not write {path}")


# ================================================================
# THE PIECES: SLABS OF THE WAVELET, TETRAHEDRALIZED, WRITTEN
# ================================================================

e = _args.extent
whole = vtkRTAnalyticSource(whole_extent=(-e, e, -e, e, -e, e))()
cuts = np.linspace(-e, e, _args.pieces + 1).round().astype(int)
slabs = vtkPartitionedDataSet()
for lo, hi in itertools.pairwise(cuts):
    # Neighbours share their plane of points, as a grid's pieces do.
    slabs.append(vtkExtractVOI(voi=(int(lo), int(hi), -e, e, -e, e))(whole))
pieces = checked(Tetrahedralize(), slabs)
pieces_path = output / "wavelet_pieces.vtpd"
write(vtkXMLPartitionedDataSetWriter(file_name=str(pieces_path)), pieces, pieces_path)
counts = [piece.number_of_cells for piece in pieces]
print(f"wrote {pieces_path}: {len(counts)} pieces of {counts} tetrahedra")


# ================================================================
# THE PIPELINE
# ================================================================

reader = vtkXMLPartitionedDataSetReader(file_name=str(pieces_path))
ghosts, contour = GhostCells(), Contour(field="RTData", value=_args.value)
# No tolerance: pieces sharing an edge interpolate its crossing from the same end
# (the first by position), so the copies are the same bits and merge exactly.
merge = MergePieces()
surfaces = (reader >> ghosts >> contour)()
for algorithm in (ghosts, contour):
    if algorithm.error is not None:
        raise algorithm.error
merged = checked(merge, surfaces)
for i, piece in enumerate(surfaces):
    ghost_triangles = int((piece.cell_data["vtkGhostType"] != 0).sum())
    print(f"  piece {i}: {piece.number_of_cells} triangles, {ghost_triangles} of them its "
          f"neighbours' (ghosts)")
surface_path = output / "pieces_isosurface.vtu"
write(vtkXMLUnstructuredGridWriter(file_name=str(surface_path)), merged, surface_path)
print(f"wrote {surface_path}: {merged.number_of_cells} triangles, {merged.number_of_points} "
      f"points")


# ================================================================
# THE WHOLE MESH'S ISOSURFACE, IN ONE PIECE
# ================================================================

one = checked(Contour(field="RTData", value=_args.value), checked(Tetrahedralize(), whole))
print(f"the whole mesh's isosurface: {one.number_of_cells} triangles, {one.number_of_points} "
      f"points")
assert merged.number_of_cells == one.number_of_cells
assert merged.number_of_points == one.number_of_points


# ================================================================
# RENDERING: EACH PIECE'S OWN TRIANGLES, A COLOUR EACH
# ================================================================

if _args.png:
    import vtkmodules.vtkInteractionStyle
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.vtkFiltersCore import vtkThreshold
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkWindowToImageFilter,
    )

    colours = [(0.89, 0.42, 0.33), (0.36, 0.62, 0.84), (0.55, 0.78, 0.45), (0.95, 0.77, 0.36),
               (0.70, 0.52, 0.82), (0.40, 0.80, 0.78)]
    renderer = vtkRenderer(background=(0.12, 0.12, 0.16))
    for i, piece in enumerate(surfaces):
        # Each piece's own triangles: ghost marks 0, in the cell data.
        owned = vtkThreshold(lower_threshold=0, upper_threshold=0)
        owned.SetInputArrayToProcess(0, 0, 0, 1, "vtkGhostType")
        mapper = vtkPolyDataMapper(scalar_visibility=False)
        piece >> owned >> vtkGeometryFilter() >> mapper
        actor = vtkActor(mapper=mapper)
        actor.property.color = colours[i % len(colours)]
        renderer.AddActor(actor)
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
