# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""53 -- An AMR file through Tack's filters: a hierarchy whole, and box by box.

VTK's Enzo reader reads moving7_0010 (VTK's test data), a cosmological
simulation's hierarchy of eight levels, each box refined by 2: a
vtkOverlappingAMR, whose cells under a finer box VTK blanks. Tack takes it as
one AMR dataset with those cells marked, so every region counts once:

    vtkAMREnzoReader                          the hierarchy
      >> Statistics(fields=["Density"])       Tack: over the cells no finer box covers
      >> Integrate(fields=["Density"])        Tack: the mass, by quadrature
      >> LogValues(field=("Density", "cells")) >> SlicePlane(z = 0.75)
                                              Tack: box by box, a slice per level
      (the slice rendered by log density, each box's outline on it, with --png)

The checks: the statistics count, and the integral sums, the cells VTK's
blanking leaves visible (in NumPy, over the blocks); the slices of the levels
tile the plane once, their areas adding to the domain's cross-section.

The files are fetched once by their content hashes (see _vtk_data.py), or read
from TACK_VTK_DATA, a VTK build's ExternalData directory.

Usage:
  uv run python packages/tack-vis/examples/53_vtk_amr.py
  uv run python packages/tack-vis/examples/53_vtk_amr.py --arch cuda --png enzo.png
"""

import argparse
import os

import numpy as np

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'cuda', 'hip', 'level_zero'],
                     help='the device Tack\'s filters run on')
_parser.add_argument('--z', type=float, default=0.75, help='the slice\'s height (default 0.75)')
_parser.add_argument('--png', help='render the slice to this PNG file')
_args = _parser.parse_args()

import _vtk_data
import vtkmodules.vtkCommonDataModel as dm  # registers the pythonic constructors
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter
from vtkmodules.vtkIOAMR import vtkAMREnzoReader

import tack
from tack.interop.vtk.algorithms import Integrate, LogValues, SlicePlane, Statistics

if _args.arch == "level_zero":
    # A VTK that holds Level Zero memory (its own device arrays, or Viskores on
    # Kokkos' SYCL) takes back only arrays in its own context: make the device
    # in that context where VTK has one, else Tack's own.
    from tack.interop.vtk import init_level_zero
    try:
        device = init_level_zero()
    except RuntimeError:                    # this VTK holds no Level Zero memory
        device = tack.set_default_device("level_zero")
else:
    device = tack.set_default_device(_args.arch)
HIDDEN = 8                                  # vtkDataSetAttributes.HIDDENCELL


def blocks(amr):
    """``(level, index, box)`` for each box of the hierarchy."""
    return [(level, k, amr.GetDataSetAsCartesianGrid(level, k))
            for level in range(amr.GetNumberOfLevels())
            for k in range(amr.GetNumberOfBlocks(level))]


def area(surface):
    if surface is None or surface.number_of_cells == 0:
        return 0.0
    sizes = vtkCellSizeFilter(compute_area=True, compute_length=False, compute_volume=False,
                              compute_vertex_count=False)(surface)
    return float(np.asarray(sizes.cell_data["Area"]).sum())


# ================================================================
# THE HIERARCHY
# ================================================================

reader = vtkAMREnzoReader(
    file_name=os.path.join(_vtk_data.path("AMR/Enzo/DD0010"), "moving7_0010.hierarchy"),
    max_level=10)
reader.UpdateInformation()
for i in range(reader.GetNumberOfCellArrays()):
    reader.SetCellArrayStatus(reader.GetCellArrayName(i), 1)
amr = reader()
levels = amr.GetNumberOfLevels()
print(f"moving7_0010: {levels} levels of {[amr.GetNumberOfBlocks(l) for l in range(levels)]} "
      f"boxes, {sum(b.number_of_cells for _, _, b in blocks(amr))} cells")

# What VTK's blanking leaves visible, in NumPy: the reference for the checks.
visible, mass = 0, 0.0
for _, _, box in blocks(amr):
    density = np.asarray(box.cell_data["Density"], np.float64)
    shown = (np.asarray(box.cell_data["vtkGhostType"]).astype(np.int64) & HIDDEN) == 0
    visible += int(shown.sum())
    mass += float(density[shown].sum()) * float(np.prod(box.GetSpacing()))


# ================================================================
# THE HIERARCHY WHOLE: STATISTICS AND THE MASS
# ================================================================

table = Statistics(fields=["Density"], device=device)(amr)
column = {table.GetColumnName(c): table.GetColumn(c).GetValue(0)
          for c in range(table.GetNumberOfColumns()) if table.GetColumnName(c) != "field"}
print(f"density over {int(column['n'])} cells (VTK shows {visible}): "
      f"from {column['min']:.3g} to {column['max']:.3g}")
assert int(column["n"]) == visible

integrals = Integrate(fields=["Density"], device=device)(amr)
ours = float(np.asarray(integrals.cell_data["Density"])[0])
volume = float(np.asarray(integrals.cell_data["Volume"])[0])
print(f"the mass, the density's integral: {ours:.6g} (the visible cells' sum: {mass:.6g}), "
      f"over a volume of {volume:g}")
assert abs(ours - mass) < 1e-6 * mass and abs(volume - 1.0) < 1e-12


# ================================================================
# BOX BY BOX: A SLICE OF EACH LEVEL
# ================================================================

log = LogValues(field=("Density", "cells"), base=10.0, name="log_density", device=device)
plane = SlicePlane(origin=(0.5, 0.5, _args.z), normal=(0.0, 0.0, 1.0), device=device)
sliced = (reader >> log >> plane)()
for algorithm in (log, plane):
    if algorithm.error is not None:
        raise algorithm.error
assert isinstance(sliced, dm.vtkPartitionedDataSetCollection)
areas = []
for level in range(sliced.GetNumberOfPartitionedDataSets()):
    pieces = sliced.GetPartitionedDataSet(level)
    areas.append(sum(area(pieces.GetPartition(j)) for j in range(pieces.GetNumberOfPartitions())))
print("the slice's area by level: " + ", ".join(
    f"{sliced.GetMetaData(l).Get(dm.vtkCompositeDataSet.NAME())} {a:.3g}"
    for l, a in enumerate(areas)) + f"; in all {sum(areas):.12f}")
assert len(areas) == levels and abs(sum(areas) - 1.0) < 1e-9


# ================================================================
# RENDERING
# ================================================================

if _args.png:
    import _scene
    from vtkmodules.vtkCommonCore import vtkPoints

    values = [np.asarray(p.cell_data["log_density"])
              for level in range(levels)
              for p in [sliced.GetPartitionedDataSet(level).GetPartition(j)
                        for j in range(sliced.GetPartitionedDataSet(level).GetNumberOfPartitions())]
              if p is not None and p.number_of_cells]
    low, high = np.percentile(np.concatenate(values), [1, 99.9])

    # Each box the plane cuts, outlined a little above it.
    outlines = dm.vtkPolyData()
    corners, lines = [], dm.vtkCellArray()
    for _, _, box in blocks(amr):
        x0, x1, y0, y1, z0, z1 = box.GetBounds()
        if not z0 <= _args.z <= z1:
            continue
        start = len(corners)
        corners += [(x0, y0, _args.z + 1e-4), (x1, y0, _args.z + 1e-4),
                    (x1, y1, _args.z + 1e-4), (x0, y1, _args.z + 1e-4)]
        lines.InsertNextCell(5, [start, start + 1, start + 2, start + 3, start])
    points = vtkPoints()
    for corner in corners:
        points.InsertNextPoint(corner)
    outlines.points = points
    outlines.lines = lines

    # Framed on the level-1 boxes the plane cuts, where the refinement is.
    fine = [box.GetBounds() for level, _, box in blocks(amr)
            if level == 1 and box.GetBounds()[4] <= _args.z <= box.GetBounds()[5]]
    focus = (min(b[0] for b in fine), max(b[1] for b in fine),
             min(b[2] for b in fine), max(b[3] for b in fine), _args.z, _args.z)
    way = _scene.render(_args.png, [
        _scene.Layer(sliced, array="log_density", cells=True,
                     value_range=(float(low), float(high))),
        _scene.Layer(outlines, color=(1.0, 1.0, 1.0))],
        size=(800, 800), focus=focus)
    print(f"rendered {_args.png} ({way})")
