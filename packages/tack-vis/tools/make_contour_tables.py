"""Generate tack.data's contour case tables from VTK's own cells.

For every cell shape and every way its points can be above or below the
isovalue, contour one reference cell with VTK: scalars 1 at the points
above, 0 below, isovalue 0.5. Every output point is then the midpoint of
the cell edge it lies on, which names the edge, so each case's triangles
come out in VTK's order and orientation, as edges in VTK's GetEdge
numbering -- the numbering tack.data.shapes uses.

    uv run python packages/tack-vis/tools/make_contour_tables.py \\
        > packages/tack-vis/src/tack/data/_contour_tables.py

A point's bit in the case index is 1 when its value is at least the
isovalue, as in VTK.
"""

import sys

import numpy as np
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkHexahedron,
    vtkMergePoints,
    vtkPyramid,
    vtkTetra,
    vtkVoxel,
    vtkWedge,
)

SHAPES = {"TETRA": vtkTetra, "VOXEL": vtkVoxel, "HEXAHEDRON": vtkHexahedron,
          "WEDGE": vtkWedge, "PYRAMID": vtkPyramid}


def reference_cell(cls):
    cell = cls()
    n = cell.GetNumberOfPoints()
    pcoords = cell.GetParametricCoords()
    for i in range(n):
        cell.GetPointIds().SetId(i, i)
        cell.GetPoints().SetPoint(i, [pcoords[3 * i + k] for k in range(3)])
    return cell


def cases(cls):
    cell = reference_cell(cls)
    n = cell.GetNumberOfPoints()
    points = np.array([cell.GetPoints().GetPoint(i) for i in range(n)])
    edges = [tuple(cell.GetEdge(e).GetPointIds().GetId(k) for k in range(2))
             for e in range(cell.GetNumberOfEdges())]
    midpoints = {tuple(np.round((points[a] + points[b]) / 2, 9)): e
                 for e, (a, b) in enumerate(edges)}
    assert len(midpoints) == len(edges), "edge midpoints must be distinct"
    table = []
    for case in range(2 ** n):
        scalars = vtkDoubleArray()
        for i in range(n):
            scalars.InsertNextValue(float((case >> i) & 1))
        out = vtkPoints()
        locator = vtkMergePoints()
        locator.InitPointInsertion(out, [-1, 2, -1, 2, -1, 2])
        verts, lines, polys = vtkCellArray(), vtkCellArray(), vtkCellArray()
        cell.Contour(0.5, scalars, locator, verts, lines, polys, None, None, None, 0, None)
        triangles = []
        offsets = [polys.GetOffsetsArray().GetValue(i) for i in range(polys.GetNumberOfCells() + 1)]
        connectivity = polys.GetConnectivityArray()
        for t in range(polys.GetNumberOfCells()):
            ids = [connectivity.GetValue(i) for i in range(offsets[t], offsets[t + 1])]
            assert len(ids) == 3, f"case {case}: a polygon of {len(ids)} points"
            triangles.append([midpoints[tuple(np.round(out.GetPoint(i), 9))] for i in ids])
        table.append(triangles)
    return table


def main():
    import vtkmodules
    out = sys.stdout
    out.write('"""Contour case tables, generated from VTK %s by\n'
              'packages/tack-vis/tools/make_contour_tables.py. Do not edit.\n\n'
              'For each shape, CASES is the number of triangles of each case and\n'
              'EDGES the cell edges their points lie on, MAX_TRIANGLES * 3 per\n'
              'case (-1 pads).\n"""\n\nimport tack\n' % vtkmodules.__version__)
    for name, cls in SHAPES.items():
        table = cases(cls)
        most = max(len(t) for t in table)
        counts = [len(t) for t in table]
        edges = []
        for triangles in table:
            flat = [e for tri in triangles for e in tri]
            edges.extend(flat + [-1] * (3 * most - len(flat)))
        out.write(f"\n{name}_MAX_TRIANGLES = {most}\n")
        out.write(f"{name}_CASES = tack.constant({_wrapped(counts)}, tack.i32)\n")
        out.write(f"{name}_EDGES = tack.constant({_wrapped(edges)}, tack.i32)\n")


def _wrapped(values):
    """A tuple literal of ``values``, one line of at most 96 columns per row."""
    import textwrap
    body = textwrap.fill(", ".join(str(v) for v in values), width=92,
                         initial_indent="    ", subsequent_indent="    ")
    return "(\n" + body + ",\n)"


if __name__ == "__main__":
    main()
