"""Viskores' test datasets (``MakeTestDataSet``), transcribed, for tests ported from
Viskores' filter unit tests. The source is
``viskores/cont/testlib/MakeTestDataSet.cxx``.

Each ``make_...`` function gives the data alone, as plain objects: ``Explicit``
(points, VTK cell types, each cell's point ids in Viskores' order, which is
VTK's), ``Uniform`` or ``Rectilinear``, with point and cell fields in Viskores'
order (x fastest for grids). Viskores stores coordinates and fields as Float32;
here they are float64 of the decimals the C++ writes. 2D points get z = 0.
``to_tack`` and ``to_vtk`` build a ``tack.dataset.DataSet`` and a VTK dataset of
one."""

from collections import namedtuple

import numpy as np

import tack
import tack.dataset as td

Explicit = namedtuple("Explicit", "points types cells point_fields cell_fields")
Uniform = namedtuple("Uniform", "dims origin spacing point_fields cell_fields")
Rectilinear = namedtuple("Rectilinear", "x y z point_fields cell_fields")

VERTEX, LINE, TRIANGLE, POLYGON, QUAD, TETRA, HEXAHEDRON, WEDGE, PYRAMID = (
    1, 3, 5, 7, 9, 10, 12, 13, 14)


def _f(values):
    return np.asarray(values, np.float64)


def _vars(point, cell):
    return {"pointvar": _f(point)}, {"cellvar": _f(cell)}


def _explicit(points, cells, point, cell):
    points = _f(points)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    return Explicit(points, [t for t, _ in cells], [list(c) for _, c in cells],
                    *_vars(point, cell))


def make_2d_uniform_0():
    return Uniform((3, 2), (0.0, 0.0), (1.0, 1.0),
                   *_vars([10.1, 20.1, 30.1, 40.1, 50.1, 60.1], [100.1, 200.1]))


def make_2d_uniform_1():
    """5x5.txt."""
    pointvar = [100.0, 78.0, 49.0, 17.0, 1.0, 94.0, 71.0, 47.0, 33.0, 6.0, 52.0, 44.0, 50.0,
                45.0, 48.0, 8.0, 12.0, 46.0, 91.0, 43.0, 0.0, 5.0, 51.0, 76.0, 83.0]
    return Uniform((5, 5), (0.0, 0.0), (1.0, 1.0), *_vars(pointvar, range(16)))


def make_3d_uniform_0():
    pointvar = [10.1, 20.1, 30.1, 40.1, 50.2, 60.2, 70.2, 80.2, 90.3, 100.3, 110.3, 120.3,
                130.4, 140.4, 150.4, 160.4, 170.5, 180.5]
    return Uniform((3, 2, 3), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
                   *_vars(pointvar, [100.1, 100.2, 100.3, 100.4]))


def make_3d_uniform_1():
    """5b 5x5x5: zero on the boundary, a ring of values inside each of z = 1, 2, 3."""
    z0 = [0.0] * 25
    rings = ([99.0, 90.0, 85.0, 95.0, 80.0, 95.0, 85.0, 90.0, 99.0],
             [75.0, 50.0, 65.0, 55.0, 15.0, 45.0, 60.0, 40.0, 70.0],
             [97.0, 87.0, 82.0, 92.0, 77.0, 92.0, 82.0, 87.0, 97.0])
    pointvar = list(z0)
    for ring in rings:
        layer = [0.0] * 5
        for j in range(3):
            layer += [0.0, *ring[3 * j:3 * j + 3], 0.0]
        pointvar += layer + [0.0] * 5
    pointvar += z0
    return Uniform((5, 5, 5), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), *_vars(pointvar, range(64)))


def make_2d_rectilinear_0():
    return Rectilinear([0.0, 1.0, 2.0], [0.0, 1.0], [0.0], *_vars(range(6), range(2)))


def make_3d_rectilinear_0():
    return Rectilinear([0.0, 1.0, 2.0], [0.0, 1.0], [0.0, 1.0, 2.0],
                       *_vars(range(18), range(4)))


def make_2d_explicit_0():
    """Triangles, quads and a hexagon (``POLYGON``): ``to_tack`` makes it a
    ``PolygonalTopology``."""
    points = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2),
              (2, 2), (3, 2), (0, 3), (3, 3), (1, 4), (2, 4)]
    cells = [(TRIANGLE, (0, 1, 5)), (QUAD, (1, 2, 6, 5)), (QUAD, (5, 6, 10, 9)),
             (QUAD, (4, 5, 9, 8)), (TRIANGLE, (2, 3, 7)), (QUAD, (6, 7, 11, 10)),
             (POLYGON, (9, 10, 13, 15, 14, 12))]
    pointvar = [100.0, 78.0, 49.0, 17.0, 94.0, 71.0, 47.0, 33.0, 52.0, 44.0, 50.0, 45.0, 8.0,
                12.0, 46.0, 91.0]
    return _explicit(points, cells, pointvar, range(7))


def make_3d_explicit_0():
    points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 2, 0)]
    cells = [(TRIANGLE, (0, 1, 2)), (QUAD, (2, 1, 3, 4))]
    return _explicit(points, cells, [10.1, 20.1, 30.2, 40.2, 50.3], [100.1, 100.2])


def make_3d_explicit_1():
    """Explicit 0 again (Viskores builds it with ``CellSetExplicit::AddCell``)."""
    return make_3d_explicit_0()


def make_3d_explicit_2():
    """One hexahedron, its points not in VTK's order (Viskores' measure is -1)."""
    points = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 1, 1),
              (0, 1, 1)]
    return _explicit(points, [(HEXAHEDRON, range(8))],
                     [10.1, 20.1, 30.2, 40.2, 50.3, 60.2, 70.2, 80.3], [100.1])


def make_3d_explicit_3():
    """One tetrahedron (Viskores' measure is -1/6)."""
    points = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 1, 0)]
    return _explicit(points, [(TETRA, (0, 1, 2, 3))], [10.1, 10.1, 10.2, 30.2], [100.1])


def make_3d_explicit_4():
    """Two hexahedra side by side."""
    points = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 1, 1),
              (0, 1, 1), (2, 0, 0), (2, 0, 1), (2, 1, 1), (2, 1, 0)]
    cells = [(HEXAHEDRON, (0, 4, 5, 1, 3, 7, 6, 2)), (HEXAHEDRON, (1, 5, 11, 8, 2, 6, 10, 9))]
    pointvar = [10.1, 20.1, 30.2, 40.2, 50.3, 60.2, 70.2, 80.3, 90.0, 10.0, 11.0, 12.0]
    return _explicit(points, cells, pointvar, [100.1, 110.0])


def make_3d_explicit_5():
    """A hexahedron, a pyramid, a tetrahedron and a wedge."""
    points = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 1, 1),
              (0, 1, 1), (2, 0.5, 0.5), (0, 2, 0), (1, 2, 0)]
    cells = [(HEXAHEDRON, (0, 1, 5, 4, 3, 2, 6, 7)), (PYRAMID, (1, 5, 6, 2, 8)),
             (TETRA, (5, 8, 10, 6)), (WEDGE, (4, 7, 9, 5, 6, 10))]
    pointvar = [10.1, 20.1, 30.2, 40.2, 50.3, 60.2, 70.2, 80.3, 90.0, 10.0, 11.0]
    return _explicit(points, cells, pointvar, [100.1, 110.0, 120.2, 130.5])


def make_3d_explicit_6():
    """Lines, vertices, a triangle, a quad and two tetrahedra."""
    points = [(-0.707, -0.354, -0.354), (0.000, -0.854, 0.146), (0.000, -0.146, 0.854),
              (-0.707, 0.354, 0.354), (10.0, 10.0, 10.0), (5.0, 5.0, 5.0), (0.0, 0.0, 2.0),
              (0.0, 0.0, -2.0)]
    cells = [(LINE, (0, 1)), (LINE, (2, 3)), (VERTEX, (4,)), (VERTEX, (5,)),
             (TRIANGLE, (2, 3, 5)), (QUAD, (0, 1, 2, 3)), (TETRA, (0, 2, 3, 6)),
             (TETRA, (3, 2, 0, 7))]
    pointvar = [100.0, 78.0, 49.0, 17.0, 94.0, 71.0, 47.0, 57.0]
    return _explicit(points, cells, pointvar, range(8))


def make_3d_explicit_zoo():
    """Hexahedra, tetrahedra, pyramids and wedges: 30 points, 25 cells."""
    points = [(0.00, 0.00, 0.00), (1.00, 0.00, 0.00), (2.00, 0.00, 0.00), (0.00, 0.00, 1.00),
              (1.00, 0.00, 1.00), (2.00, 0.00, 1.00), (0.00, 1.00, 0.00), (1.00, 1.00, 0.00),
              (2.00, 1.00, 0.00), (0.00, 1.00, 1.00), (1.00, 1.00, 1.00), (2.00, 1.00, 1.00),
              (0.00, 2.00, 0.00), (1.00, 2.00, 0.00), (2.00, 2.00, 0.00), (0.00, 2.00, 1.00),
              (1.00, 2.00, 1.00), (2.00, 2.00, 1.00), (1.00, 3.00, 1.00), (2.75, 0.00, 1.00),
              (3.00, 0.00, 0.75), (3.00, 0.25, 1.00), (3.00, 1.00, 1.00), (3.00, 1.00, 0.00),
              (2.57, 2.00, 1.00), (3.00, 1.75, 1.00), (3.00, 1.75, 0.75), (3.00, 0.00, 0.00),
              (2.57, 0.42, 0.57), (2.59, 1.43, 0.71)]
    cells = [(HEXAHEDRON, (0, 3, 4, 1, 6, 9, 10, 7)), (HEXAHEDRON, (1, 4, 5, 2, 7, 10, 11, 8)),
             (TETRA, (23, 26, 24, 29)), (TETRA, (24, 26, 25, 29)), (TETRA, (8, 17, 11, 29)),
             (TETRA, (17, 24, 25, 29)), (PYRAMID, (24, 17, 8, 23, 29)),
             (PYRAMID, (23, 8, 11, 22, 29)), (PYRAMID, (25, 22, 11, 17, 29)),
             (PYRAMID, (26, 23, 22, 25, 29)), (PYRAMID, (23, 8, 2, 27, 28)),
             (PYRAMID, (22, 11, 8, 23, 28)), (PYRAMID, (11, 5, 2, 8, 28)),
             (PYRAMID, (21, 19, 5, 11, 28)), (TETRA, (11, 22, 21, 28)), (TETRA, (5, 19, 20, 28)),
             (PYRAMID, (23, 27, 20, 21, 28)), (PYRAMID, (20, 27, 2, 5, 28)),
             (TETRA, (19, 21, 20, 28)), (PYRAMID, (7, 6, 12, 13, 16)),
             (PYRAMID, (6, 9, 15, 12, 16)), (PYRAMID, (6, 7, 10, 9, 16)),
             (TETRA, (12, 15, 16, 18)), (WEDGE, (8, 14, 17, 7, 13, 16)),
             (WEDGE, (11, 8, 17, 10, 7, 16))]
    pointvar = [4.0, 5.0, 9.5, 5.5, 6.0, 9.5, 5.0, 5.5, 5.7, 6.5, 6.4, 6.9, 6.6, 6.1, 7.1, 7.2,
                7.3, 7.4, 9.1, 9.2, 9.3, 5.4, 9.5, 9.6, 6.7, 9.8, 6.0, 4.3, 4.9, 4.1]
    return _explicit(points, cells, pointvar, pointvar[:25])


def make_3d_explicit_polygonal():
    """A cube's surface of triangles, quads and quads typed ``POLYGON``: ``to_tack``
    makes it a ``PolygonalTopology``."""
    points = [(-0.707, -0.354, -0.354), (0.000, -0.854, 0.146), (0.000, -0.146, 0.854),
              (-0.707, 0.354, 0.354), (0.000, 0.146, -0.854), (0.000, 0.854, -0.146),
              (0.707, 0.354, 0.354), (0.707, -0.354, -0.354)]
    cells = [(TRIANGLE, (0, 1, 3)), (TRIANGLE, (1, 2, 3)), (QUAD, (4, 5, 6, 7)),
             (TRIANGLE, (0, 4, 1)), (TRIANGLE, (4, 7, 1)), (POLYGON, (3, 2, 6, 5)),
             (QUAD, (0, 3, 5, 4)), (POLYGON, (1, 7, 6, 2))]
    pointvar = [100.0, 78.0, 49.0, 17.0, 94.0, 71.0, 47.0, 33.0]
    return _explicit(points, cells, pointvar, range(8))


def _offsets(d):
    return np.concatenate([[0], np.cumsum([len(c) for c in d.cells])]).astype(np.int64)


def _connectivity(d):
    return np.asarray([p for c in d.cells for p in c], np.int64)


def _values(values, dtype):
    values = np.asarray(values)
    if values.dtype.kind == "f":
        values = values.astype(np.float64 if dtype == tack.f64 else np.float32)
        if values.ndim == 2:
            return tack.array(values, tack.vec(values.shape[1], dtype))
    return tack.array(values)


def to_tack(d, dtype=tack.f64):
    """``d`` as a ``tack.dataset.DataSet`` in ``dtype``. Explicit data with a
    ``POLYGON`` becomes a ``PolygonalTopology`` (as ``vtk_to_dataset`` does: its
    triangles and quads polygons too), refused if it holds other shapes."""
    if isinstance(d, Uniform):
        data = td.uniform_grid(d.dims, origin=d.origin, spacing=d.spacing, dtype=dtype)
    elif isinstance(d, Rectilinear):
        data = td.rectilinear_grid(d.x, d.y, d.z, dtype=dtype)
    elif POLYGON in d.types:
        if not set(d.types) <= {TRIANGLE, POLYGON, QUAD}:
            raise ValueError("polygons mix only with triangles and quads")
        topology = td.PolygonalTopology(_offsets(d), _connectivity(d),
                                        num_points=len(d.points))
        data = td.DataSet(topology, d.points, dtype=dtype)
    else:
        topology = td.UnstructuredTopology(np.asarray(d.types, np.uint8), _offsets(d),
                                           _connectivity(d), num_points=len(d.points))
        data = td.DataSet(topology, d.points, dtype=dtype)
    for name, values in d.point_fields.items():
        data.fields[name] = td.Field(td.H1(data), _values(values, dtype))
    for name, values in d.cell_fields.items():
        data.fields[name] = td.Field(td.Constant(data), _values(values, dtype))
    return data


def to_vtk(d):
    """``d`` as a ``vtkUnstructuredGrid``, ``vtkImageData`` or ``vtkRectilinearGrid``."""
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints, vtkUnsignedCharArray
    from vtkmodules.vtkCommonDataModel import (
        vtkCellArray,
        vtkImageData,
        vtkRectilinearGrid,
        vtkUnstructuredGrid,
    )

    if isinstance(d, Uniform):
        pad = 3 - len(d.dims)
        grid = vtkImageData(dimensions=tuple(d.dims) + (1,) * pad,
                            origin=tuple(d.origin) + (0.0,) * pad,
                            spacing=tuple(d.spacing) + (1.0,) * pad)
    elif isinstance(d, Rectilinear):
        axes = [_f(a) for a in (d.x, d.y, d.z)]
        grid = vtkRectilinearGrid(dimensions=tuple(len(a) for a in axes),
                                  x_coordinates=vtkDoubleArray(axes[0]),
                                  y_coordinates=vtkDoubleArray(axes[1]),
                                  z_coordinates=vtkDoubleArray(axes[2]))
    else:
        grid = vtkUnstructuredGrid()
        grid.points = vtkPoints(data=d.points)
        grid.cells = (vtkUnsignedCharArray(np.asarray(d.types, np.uint8)),
                      vtkCellArray(offsets=_offsets(d), connectivity=_connectivity(d)))
    for name, values in d.point_fields.items():
        grid.point_data[name] = np.ascontiguousarray(values)
    for name, values in d.cell_fields.items():
        grid.cell_data[name] = np.ascontiguousarray(values)
    return grid


ALL = {
    "2d_uniform_0": make_2d_uniform_0,
    "2d_uniform_1": make_2d_uniform_1,
    "3d_uniform_0": make_3d_uniform_0,
    "3d_uniform_1": make_3d_uniform_1,
    "2d_rectilinear_0": make_2d_rectilinear_0,
    "3d_rectilinear_0": make_3d_rectilinear_0,
    "2d_explicit_0": make_2d_explicit_0,
    "3d_explicit_0": make_3d_explicit_0,
    "3d_explicit_1": make_3d_explicit_1,
    "3d_explicit_2": make_3d_explicit_2,
    "3d_explicit_3": make_3d_explicit_3,
    "3d_explicit_4": make_3d_explicit_4,
    "3d_explicit_5": make_3d_explicit_5,
    "3d_explicit_6": make_3d_explicit_6,
    "3d_explicit_zoo": make_3d_explicit_zoo,
    "3d_explicit_polygonal": make_3d_explicit_polygonal,
}
