# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""51 -- An OpenFOAM case through Tack's filters: a multiblock in, a multiblock out.

VTK's OpenFOAM reader reads squareBend (VTK's test data), compressible flow
through a square duct with a bend, 112,000 hexahedra. With its patches read
too it gives a vtkMultiBlockDataSet: "internalMesh", the cells, and "boundary",
a multiblock of the inlet, outlet and wall patches as polygons. Tack's
algorithms run on each block and give a multiblock of the same shape:

    vtkOpenFOAMReader                          squareBend, the last step
      >> Gradients(field=("U", "points"))      Tack: the velocity's gradient
      >> FlowQuantities(gradient="gradient")   Tack: vorticity, the Q-criterion
      >> SlicePlane(normal=(0, 0, 1))          Tack: the duct's middle plane
    the inlet and outlet patches >> Integrate  Tack: integrals over each
      (the slice rendered by the Q-criterion, with --png)

The checks: the gradient is VTK's (vtkGradientFilter of the cells' block); the
slice of the cells is VTK's (vtkCutter: the same area); the patches' integrals
are vtkIntegrateAttributes'; the trees keep their blocks and names.

The case is fetched once by its content hashes (see _vtk_data.py), or read from
TACK_VTK_DATA, a VTK build's ExternalData directory.

Usage:
  uv run python packages/tack-vis/examples/51_vtk_openfoam.py
  uv run python packages/tack-vis/examples/51_vtk_openfoam.py --arch cuda --png bend.png
"""

import argparse

import numpy as np

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'cuda', 'hip', 'level_zero'],
                     help='the device Tack\'s filters run on')
_parser.add_argument('--png', help='render the slice and the walls to this PNG file')
_args = _parser.parse_args()

import _vtk_data
import vtkmodules.vtkCommonDataModel as dm  # registers the pythonic constructors
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.vtkFiltersGeneral import vtkGradientFilter
from vtkmodules.vtkFiltersParallel import vtkIntegrateAttributes
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter
from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader

import tack
from tack.interop.vtk.algorithms import FlowQuantities, Gradients, Integrate, SlicePlane

tack.set_default_device(_args.arch)
device = _args.arch


def tree(multiblock, indent="  "):
    """The block names of a multiblock, nested."""
    lines = []
    for i in range(multiblock.GetNumberOfBlocks()):
        block = multiblock.GetBlock(i)
        name = multiblock.GetMetaData(i).Get(dm.vtkCompositeDataSet.NAME())
        if isinstance(block, dm.vtkMultiBlockDataSet):
            lines.append(f"{indent}{name}")
            lines.extend(tree(block, indent + "  "))
        else:
            lines.append(f"{indent}{name}: {block.GetClassName()}, "
                         f"{block.number_of_cells} cells")
    return lines


def area(surface):
    sizes = vtkCellSizeFilter(compute_area=True, compute_length=False, compute_volume=False,
                              compute_vertex_count=False)(surface)
    return float(np.asarray(sizes.cell_data["Area"]).sum())


# ================================================================
# THE CASE, WITH ITS PATCHES
# ================================================================

reader = vtkOpenFOAMReader(file_name=_vtk_data.path("OpenFOAM/squareBend/squareBend.foam"))
reader.UpdateInformation()
for i in range(reader.GetNumberOfPatchArrays()):
    reader.SetPatchArrayStatus(reader.GetPatchArrayName(i), 1)
steps = reader.GetOutputInformation(0).Get(vtkStreamingDemandDrivenPipeline.TIME_STEPS())
reader.UpdateTimeStep(steps[-1])
case = reader.GetOutputDataObject(0)
cells = case.GetBlock(0)
print(f"read squareBend at t = {steps[-1]:g}:")
print("\n".join(tree(case)))


# ================================================================
# GRADIENT, VORTICITY AND A SLICE, ON EVERY BLOCK
# ================================================================

gradients = Gradients(field=("U", "points"), at="points", name="gradient", device=device)
flow = FlowQuantities(gradient="gradient", device=device)
middle = SlicePlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), device=device)
with_flow = (reader >> gradients >> flow)()
sliced = (reader >> gradients >> flow >> middle)()
for algorithm in (gradients, flow, middle):
    if algorithm.error is not None:
        raise algorithm.error
print("the slice:")
print("\n".join(tree(sliced)))
assert [line.split(":")[0] for line in tree(sliced)] == [line.split(":")[0]
                                                         for line in tree(case)]

# The gradient at the points, against VTK's.
vtk_gradient = vtkGradientFilter(result_array_name="gradient")
vtk_gradient.SetInputArrayToProcess(0, 0, 0, dm.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U")
theirs = np.asarray(vtk_gradient(cells).point_data["gradient"])
ours = np.asarray(with_flow.GetBlock(0).point_data["gradient"])
worst = float((np.linalg.norm(ours - theirs, axis=1)
               / np.maximum(np.linalg.norm(theirs, axis=1), 1e-30)).max())
print(f"the velocity gradient at {len(ours)} points: within {worst:.1e} of VTK's")
assert worst < 1e-5

# The slice of the cells, against VTK's.
cutter = vtkCutter(cut_function=dm.vtkPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)))
ours, theirs = area(sliced.GetBlock(0)), area(cutter(cells))
print(f"the middle plane: area {ours:.6f} (VTK's {theirs:.6f})")
assert abs(ours - theirs) < 1e-5 * theirs


# ================================================================
# INTEGRALS OVER THE INLET AND THE OUTLET
# ================================================================

boundary = case.GetBlock(1)
for i in range(boundary.GetNumberOfBlocks()):
    name = boundary.GetMetaData(i).Get(dm.vtkCompositeDataSet.NAME())
    if name not in ("inlet", "outlet"):
        continue
    patch = boundary.GetBlock(i)
    # OpenFOAM's reader gives each field on the points and the cells: the points'.
    ours = Integrate(fields=[(f, "points") for f in ("p", "T", "rho")], device=device)(patch)
    theirs = vtkIntegrateAttributes()(patch)
    measured = float(np.asarray(ours.cell_data["Area"])[0])
    print(f"{name}: area {measured:.4f}, " + ", ".join(
        f"mean {f} {float(np.asarray(ours.point_data[f])[0]) / measured:.4g}"
        for f in ("p", "T", "rho")))
    for f in ("p", "T", "rho"):
        a = float(np.asarray(ours.point_data[f])[0])
        b = float(np.asarray(theirs.point_data[f])[0])
        assert abs(a - b) <= 1e-5 * abs(b), (name, f, a, b)


# ================================================================
# RENDERING
# ================================================================

if _args.png:
    import _scene

    plane = sliced.GetBlock(0)
    q = np.asarray(plane.point_data["q_criterion"])
    way = _scene.render(_args.png, [
        _scene.Layer(plane, array="q_criterion",
                     value_range=tuple(float(v) for v in np.percentile(q, [2, 98]))),
        _scene.Layer(case.GetBlock(1), color=(0.8, 0.8, 0.85), opacity=0.2)],  # the patches
        size=(1000, 700), zoom=1.4)
    print(f"rendered {_args.png} ({way})")
