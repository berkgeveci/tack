# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""50 -- An Exodus file through Tack's filters: a collection in, a collection out.

VTK's IOSS reader reads disk_out_ref.ex2, a simulation of flow over a spinning
disk (VTK's test data; ParaView's tutorials use it). It gives a
vtkPartitionedDataSetCollection: the element block, and the node and side sets,
which it leaves unread and empty. Tack's algorithms take the collection as it is
and give one of the same shape, each entry named as the file names it:

    vtkIOSSReader                              disk_out_ref.ex2, a collection
      >> Contour(field="Temp", value=400)      Tack: the 400 K isosurface
    vtkIOSSReader                              (seeds: a vtkLineSource on port 1)
      >> Streamlines(field="V") >> Tubes()     Tack: paths through the flow, as tubes
    vtkIOSSReader >> ExtractSurface()          Tack: the disk and its housing
      (rendered together, with --png)

The checks: the isosurface is VTK's (vtkContourFilter of the same block: the same
triangles and area); a path starts at each seed inside the mesh, as VTK's probe
finds them, and stays in the housing; the collections keep their entries.

The file is fetched once by its content hash (see _vtk_data.py), or read from
TACK_VTK_DATA, a VTK build's ExternalData directory.

Usage:
  uv run python packages/tack-vis/examples/50_vtk_exodus.py
  uv run python packages/tack-vis/examples/50_vtk_exodus.py --arch cuda --png disk.png
"""

import argparse

import numpy as np

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'cuda', 'hip', 'level_zero'],
                     help='the device Tack\'s filters run on (the file is in f64)')
_parser.add_argument('--seeds', type=int, default=40, help='seeds on the line (default 40)')
_parser.add_argument('--png', help='render the paths, isosurface and disk to this PNG file')
_args = _parser.parse_args()

import _vtk_data
import vtkmodules.vtkCommonDataModel as dm  # registers the pythonic constructors
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkProbeFilter
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter
from vtkmodules.vtkIOIOSS import vtkIOSSReader

import tack
from tack.interop.vtk.algorithms import Contour, ExtractSurface, Streamlines, Tubes

tack.set_default_device(_args.arch)
device = _args.arch


def names(collection):
    return [collection.GetMetaData(i).Get(dm.vtkCompositeDataSet.NAME())
            for i in range(collection.GetNumberOfPartitionedDataSets())]


def partitions(collection, i):
    entry = collection.GetPartitionedDataSet(i)
    return [entry.GetPartition(j) for j in range(entry.GetNumberOfPartitions())]


def area(surface):
    sizes = vtkCellSizeFilter(compute_area=True, compute_length=False, compute_volume=False,
                              compute_vertex_count=False)(surface)
    return float(np.asarray(sizes.cell_data["Area"]).sum())


# ================================================================
# THE FILE
# ================================================================

reader = vtkIOSSReader(file_name=_vtk_data.path("disk_out_ref.ex2"))
collection = reader()
block = partitions(collection, 0)[0]
print(f"read disk_out_ref.ex2: {collection.GetNumberOfPartitionedDataSets()} entries "
      f"{names(collection)}; {names(collection)[0]} has {block.number_of_cells} hexahedra")


# ================================================================
# THE ISOSURFACE, AGAINST VTK'S
# ================================================================

contour = Contour(field="Temp", value=400.0, device=device)
isosurface = (reader >> contour)()
if contour.error is not None:
    raise contour.error
assert isinstance(isosurface, dm.vtkPartitionedDataSetCollection)
assert names(isosurface) == names(collection)
ours = partitions(isosurface, 0)[0]

theirs = vtkContourFilter(compute_scalars=False)
theirs.SetValue(0, 400.0)
theirs.SetInputArrayToProcess(0, 0, 0, 0, "Temp")
reference = theirs(block)
print(f"isosurface at 400 K: {ours.number_of_cells} triangles, area {area(ours):.4f} "
      f"(VTK's: {reference.number_of_cells}, {area(reference):.4f})")
assert ours.number_of_cells == reference.number_of_cells
assert abs(area(ours) - area(reference)) < 1e-6 * area(reference)


# ================================================================
# STREAMLINES FROM A LINE OF SEEDS, AS TUBES
# ================================================================

line = vtkLineSource(point1=(0.0, -5.0, -9.5), point2=(0.0, 5.0, 9.5),
                     resolution=_args.seeds - 1)
streamlines = Streamlines(field="V", step_size=0.02, steps=4000, step_unit="length",
                          device=device)
streamlines.SetSourceConnection(line.GetOutputPort())
tubes = Tubes(radius=0.06, sides=10, device=device)
tube_surface = (reader >> streamlines >> tubes)()
if streamlines.error is not None:
    raise streamlines.error
paths = partitions(streamlines.GetOutputDataObject(0), 0)[0]

# The seeds inside the mesh, as VTK's probe finds them.
probe = vtkProbeFilter(source_data=block)
valid = probe(line()).point_data["vtkValidPointMask"]   # a vtkCharArray: bytes to NumPy
inside = int(np.asarray(valid).view(np.int8).sum())
points = np.asarray(paths.points.data)
radius = np.linalg.norm(points[:, :2], axis=1)
bounds = block.GetBounds()
lengths = [0.0] * paths.number_of_cells
offsets = np.asarray(paths.lines.offsets_array)
connectivity = np.asarray(paths.lines.connectivity_array)
for k in range(paths.number_of_cells):
    path = points[connectivity[offsets[k]:offsets[k + 1]]]
    lengths[k] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
# A path is one polyline, or more where a particle was pushed out at a wall and
# came back in: its stretches, numbered by "segment", all with its "seed".
seeds = np.asarray(paths.cell_data["seed"])
print(f"{len(set(seeds))} paths ({paths.number_of_cells} stretches) from {inside} seeds "
      f"inside the mesh (of {_args.seeds}), {paths.number_of_points} points, the longest "
      f"stretch {max(lengths):.1f} long; {partitions(tube_surface, 0)[0].number_of_cells} "
      "tube polygons")
assert len(set(seeds)) == inside
# A path ends a step past the wall it leaves through, at most.
step = streamlines.step_size
assert radius.max() <= max(bounds[1], bounds[3]) + step
assert bounds[4] - step <= points[:, 2].min() and points[:, 2].max() <= bounds[5] + step

surface = ExtractSurface(device=device)
housing = (reader >> surface)()
print(f"the disk and housing: {partitions(housing, 0)[0].number_of_cells} quadrilaterals")


# ================================================================
# RENDERING
# ================================================================

if _args.png:
    import _scene

    time = np.asarray(paths.point_data["time"])
    way = _scene.render(_args.png, [
        _scene.Layer(tube_surface, array="time", value_range=(0.0, float(time.max()))),
        _scene.Layer(isosurface, color=(0.95, 0.55, 0.25), opacity=0.6),
        _scene.Layer(housing, color=(0.8, 0.8, 0.85), opacity=0.15)],
        size=(800, 900), elevation=-70, zoom=1.3)
    print(f"rendered {_args.png} ({way})")
