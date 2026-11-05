# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""52 -- Pathlines through a time series in a VTK pipeline, driving it and in situ.

VTK's OpenFOAM reader reads the lid-driven cavity (VTK's test data): a square
cavity whose top moves at 1 m/s, six steps from t = 0 to 2.5 s. Tack's
Pathlines algorithm follows particles through the steps in either of its modes:

- **Driving**, it asks the reader for each step itself, one per pass, and gives
  the paths once, after the last step (time_range chooses the steps).
- **In situ**, as a simulation would run it, it is handed each step in turn and
  gives the paths so far at every one.

    vtkOpenFOAMReader                                   the cavity's six steps
      >> Pathlines(field=("U", "points"), seeds=...)    Tack: paths from t = 1 s

The flow is steady from t = 1 s, so a path from then on is a streamline, and the
check is that Tack's pathlines end where its streamlines (Streamlines, at the
last step) are 1.5 s along; and that the in situ paths are the driving ones,
bit for bit. With --png the streamlines are drawn as tubes over the speed on the
cavity's middle plane, and the pathlines' points at the steps on them.

The case is fetched once by its content hashes (see _vtk_data.py), or read from
TACK_VTK_DATA, a VTK build's ExternalData directory.

Usage:
  uv run python packages/tack-vis/examples/52_vtk_pathlines.py
  uv run python packages/tack-vis/examples/52_vtk_pathlines.py --arch cuda --png cavity.png
"""

import argparse

import numpy as np

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'cuda', 'hip', 'level_zero'],
                     help='the device Tack\'s filters run on')
_parser.add_argument('--step', type=float, default=0.005,
                     help='the longest integration step, in seconds (default 0.005)')
_parser.add_argument('--png', help='render the paths over the speed to this PNG file')
_args = _parser.parse_args()

import _vtk_data
import vtkmodules.vtkCommonDataModel as dm  # registers the pythonic constructors
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline
from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader

import tack
from tack.interop.vtk.algorithms import Magnitude, Pathlines, SlicePlane, Streamlines, Tubes

tack.set_default_device(_args.arch)
device = _args.arch
VELOCITY = ("U", "points")      # the reader gives U on the points and on the cells


def ends(lines):
    """``{seed: (last point, its time)}`` of a polydata of paths with ``seed`` on
    their points or cells."""
    points = np.asarray(lines.points.data)
    offsets = np.asarray(lines.lines.offsets_array)
    connectivity = np.asarray(lines.lines.connectivity_array)
    time = np.asarray(lines.point_data["time"])
    if "seed" in lines.point_data.keys():
        seeds = np.asarray(lines.point_data["seed"])[connectivity[offsets[:-1]]]
    else:
        seeds = np.asarray(lines.cell_data["seed"])
    out = {}
    for k, seed in enumerate(seeds):
        last = connectivity[offsets[k + 1] - 1]
        if int(seed) not in out or time[last] > out[int(seed)][1]:
            out[int(seed)] = (points[last], float(time[last]))
    return out


# ================================================================
# THE CASE
# ================================================================

reader = vtkOpenFOAMReader(file_name=_vtk_data.path("OpenFOAM/cavity/cavity.foam"))
reader.UpdateInformation()
steps = list(reader.GetOutputInformation(0).Get(
    vtkStreamingDemandDrivenPipeline.TIME_STEPS()))
print(f"the cavity: steps at {steps} s")
seeds = np.array([[x, y, 0.005] for x in (0.02, 0.04, 0.06, 0.08)
                  for y in (0.03, 0.05, 0.07)])
width = 0.1


# ================================================================
# DRIVING: THE FILTER TAKES THE STEPS ITSELF
# ================================================================

pathlines = Pathlines(field=VELOCITY, seeds=seeds, step_size=_args.step,
                      time_range=(1.0, steps[-1]), device=device)
driven = (reader >> pathlines)()
if pathlines.error is not None:
    raise pathlines.error
assert isinstance(driven, dm.vtkMultiBlockDataSet)       # the reader's tree, kept
paths = driven.GetBlock(0)
assert not driven.GetInformation().Has(dm.vtkDataObject.DATA_TIME_STEP())  # time is in the data
taken = sorted(set(np.asarray(paths.point_data["time"]).tolist()))
print(f"driving: {paths.number_of_cells} paths through the steps at {taken} s, "
      f"{paths.number_of_points} points")
assert paths.number_of_cells == len(seeds) and taken == [t for t in steps if t >= 1.0]


# ================================================================
# THE CHECK: IN STEADY FLOW, PATHLINES ARE STREAMLINES
# ================================================================

reader.UpdateTimeStep(steps[-1])
streamlines = Streamlines(field=VELOCITY, seeds=seeds, step_size=_args.step,
                          steps=round((steps[-1] - 1.0) / _args.step),
                          step_unit="time", device=device)
traced = (reader >> streamlines)()
if streamlines.error is not None:
    raise streamlines.error
paths_end, lines_end = ends(paths), ends(traced.GetBlock(0))
worst = max(float(np.linalg.norm(paths_end[s][0] - lines_end[s][0])) for s in paths_end)
print(f"each pathline ends within {worst:.1e} m of its streamline 1.5 s along "
      f"(the cavity is {width} m wide)")
# Times are in the field's f32, summed over the steps.
assert all(abs(lines_end[s][1] - (steps[-1] - 1.0)) < 1e-5 for s in lines_end)
assert worst < 1e-4 * width


# ================================================================
# IN SITU: EACH STEP HANDED TO THE FILTER
# ================================================================

live = Pathlines(field=VELOCITY, seeds=seeds, step_size=_args.step, mode="in situ",
                 device=device)
for t in steps:
    if t < 1.0:
        continue
    reader.UpdateTimeStep(t)
    live.SetInputDataObject(0, reader.GetOutputDataObject(0))
    live.Update()
    so_far = live.GetOutputDataObject(0)
    stamp = so_far.GetInformation().Get(dm.vtkDataObject.DATA_TIME_STEP())
    print(f"in situ at t = {stamp:g} s: {so_far.GetBlock(0).number_of_points} points so far")
same = np.array_equal(np.asarray(so_far.GetBlock(0).points.data), np.asarray(paths.points.data))
print(f"in situ, after the last step: the driving paths {'bit for bit' if same else 'NOT'}")
assert same


# ================================================================
# RENDERING
# ================================================================

if _args.png:
    import _scene

    speed = Magnitude(field=VELOCITY, name="speed", device=device)
    plane = SlicePlane(origin=(0.0, 0.0, 0.005), normal=(0.0, 0.0, 1.0), device=device)
    middle = (reader >> speed >> plane)().GetBlock(0)
    tubes = Tubes(radius=0.0006, sides=10, device=device)(traced).GetBlock(0)
    way = _scene.render(_args.png, [
        _scene.Layer(middle, array="speed", value_range=(0.0, 0.4)),
        _scene.Layer(tubes, color=(0.95, 0.95, 0.95)),
        _scene.Layer(paths, color=(0.95, 0.3, 0.2), point_size=9)],   # the steps' points
        size=(800, 800))
    print(f"rendered {_args.png} ({way})")
