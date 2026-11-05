# VTK's Python tests on Tack

VTK's own Python tests, run with Tack's VTK algorithms (`tack.interop.vtk.algorithms`)
in place of VTK's filters: a robustness test of Tack's filters on the data and
pipelines VTK's tests build. It is not a check that Tack replaces VTK's filters one
for one -- Tack keeps its own array names, cell orders and conventions.

## How it works

`shims.py` puts a stand-in in place of each VTK class it knows (`vtkContourFilter`,
`vtkCutter`, `vtkClipDataSet`, `vtkStreamTracer`, ...; `TRANSLATIONS`), in its
`vtkmodules` module, so a test script runs unchanged. A stand-in keeps a real VTK
filter as its twin: every call the stand-in does not answer (`SetValue`,
`SetCutFunction`, ...) goes to the twin, so the twin holds the test's settings. When
the pipeline runs the stand-in, it reads the twin's settings, runs the Tack wrapper
they translate to, and gives its output (converted to the class VTK's filter gives).
It also runs the twin on the same input and records the two outputs compared:
counts, bounds over the points cells use, total length, area and volume, and each
common array's range, with a verdict (`same`, `same geometry` where only counts
differ, `differ`). A setting with no Tack counterpart raises `Unsupported`, and a
Tack error is caught: the stand-in then gives the twin's output, so the test still
runs, and the record says why.

`run_one.py` runs one test as VTK's test driver does (`rtImageTest`), rendering
offscreen, with VTK's filters (`vtk`) or Tack's (`tack`), its image checked against
VTK's baseline. `drive.py` runs a list of tests both ways in parallel into
`summary.json`; `report.py` summarizes it: each test's outcome, per VTK filter how
Tack's output compared, the errors, the unsupported settings, the outputs that
differ, and the active attributes VTK sets.

## Running it

It needs a VTK source tree (for the tests), VTK's test data and baselines (a VTK
build's `ExternalData`), and a VTK with rendering for Python (a local build on
`PYTHONPATH`, or the wheel):

```sh
uv sync --extra cpu --extra dev --extra interop
grep -rl "vtkContourFilter" ~/VTK/Filters/*/Testing/Python/*.py \
    | sed "s#$HOME/VTK/##" > tests.txt                # the tests to run
uv run python tools/vtk_test_ports/drive.py ~/VTK ~/VTK-build/ExternalData out tests.txt 4
uv run python tools/vtk_test_ports/report.py out/summary.json
```

Each test's output, its image and diff, and the stand-ins' `records.json` are in
`out/<module>_<test>/{vtk,tack}/`.

## What it found (2026-11-05)

Of VTK's 444 Python tests that use a filter Tack has, 239 pass with Tack's filters
in place after the fixes it led to: higher-order field writes (#343), triangulation
of concave polygons (#344), transformed vectors and normals (#345), active
attributes and tube normals (#348), vector components (#349), serendipity cells
with 64-bit data and flat curvilinear extents (#350), hidden points, connectivity
on polydata and empty polygons (#352), polydata outputs (#353). The rest are
expected differences (integrators, triangulations, exact higher order), VTK's names
and cell orders, or settings the stand-ins do not translate.
