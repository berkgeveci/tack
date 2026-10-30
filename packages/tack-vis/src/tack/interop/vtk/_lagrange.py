"""VTK's Lagrange cells -- ``VTK_LAGRANGE_CURVE``, ``VTK_LAGRANGE_QUADRILATERAL``,
``VTK_LAGRANGE_HEXAHEDRON``, ``VTK_LAGRANGE_TRIANGLE`` and ``VTK_LAGRANGE_TETRAHEDRON``
-- as Tack's bases on equispaced nodes, both ways. A triangle's or tetrahedron's
``C(p + d, d)`` nodes are VTK's in its recursive order (corners, edges, faces,
interior), matched to Tack's simplex lattice (``bases.simplex_lattice``) by the
parametric coordinates VTK gives each (``GetParametricCoords``), on the same
reference cell.

A Lagrange cell of order ``p`` lists its ``(p + 1)^d`` nodes in VTK's order
(corners, then edge, face and interior nodes; ``vtkLagrangeHexahedron
.PointIndexFromIJK``), on equispaced nodes over ``[0, 1]`` in each axis: Tack's
``"equispaced"`` family on the same reference cell, the corners in the linear
cell's order. Its points are shared with the cells around it, as an ``H1``
space's nodes are, so a grid of such cells is a linear topology of their corners
and fields in ``H1(order=p, basis="equispaced")``.
"""

import numpy as np

import tack

#: VTK type: (Tack's linear shape, dimension).
LAGRANGE = {68: ("Line", 1), 70: ("Quad", 2), 72: ("Hexahedron", 3),
            69: ("Triangle", 2), 71: ("Tetra", 3), 73: ("Wedge", 3)}
_LINEAR = {68: 3, 70: 9, 72: 12, 69: 5, 71: 10, 73: 13}
#: The types whose nodes are matched by VTK's parametric coordinates: the
#: simplices and the wedge.
_SIMPLEX = (69, 71, 73)
#: VTK's Bezier cells: their Lagrange twin, whose points they number alike. Their
#: points and point data are control points and coefficients, Tack's
#: ``basis="bernstein"`` (rational weights are refused).
BEZIER = {75: 68, 77: 70, 79: 72, 76: 69, 78: 71, 80: 73}
#: VTK's classic quadratic cells: the Lagrange twin whose shape and order-2 nodes
#: they share, and Tack's basis for them: the full order-2 space (triangles,
#: tetrahedra) or serendipity (quad8, hex20, wedge15). Matched by VTK's parametric
#: coordinates.
QUADRATIC = {22: (69, None), 24: (71, None), 23: (70, "serendipity"),
             25: (72, "serendipity"), 26: (73, "serendipity")}


def _simplex_vtk_index(vtk_type, order):
    """For each of Tack's simplex nodes (``bases.simplex_lattice``), its place in a VTK
    Lagrange triangle or tetrahedron of ``order``: matched by VTK's own parametric
    coordinates for each of its points."""
    from math import comb

    from vtkmodules.vtkCommonDataModel import (
        vtkLagrangeTetra,
        vtkLagrangeTriangle,
        vtkLagrangeWedge,
    )

    from tack.dataset import bases, shapes

    dimension = LAGRANGE[vtk_type][1]
    count = (comb(order + 2, 2) * (order + 1) if vtk_type == 73
             else comb(order + dimension, dimension))
    cell = {71: vtkLagrangeTetra, 69: vtkLagrangeTriangle, 73: vtkLagrangeWedge}[vtk_type]()
    cell.GetPointIds().SetNumberOfIds(count)
    cell.GetPoints().SetNumberOfPoints(count)
    if vtk_type == 73:
        # A wedge's degrees may differ by direction; VTK wants them set first.
        cell.SetUniformOrderFromNumPoints(count)
    else:
        cell.Initialize()
    pcoords = np.asarray(cell.GetParametricCoords(), np.float64).reshape(-1, 3)[:, :dimension]
    steps = np.rint(pcoords * order).astype(np.int64)
    lattice = bases.lattice_ijk({69: shapes.Triangle, 71: shapes.Tetra,
                                 73: shapes.Wedge}[vtk_type], order)
    place = {tuple(row): k for k, row in enumerate(steps.tolist())}
    out = np.array([place[tuple(row)] for row in lattice.tolist()], np.int64)
    if sorted(out.tolist()) != list(range(count)):
        raise AssertionError(f"VTK's Lagrange nodes of order {order} are not the lattice's")
    return out


def _vtk_index(vtk_type, order):
    """For each lexicographic node ``n`` (first axis fastest), its place in a VTK
    Lagrange cell of ``order``."""
    from vtkmodules.vtkCommonDataModel import vtkLagrangeHexahedron, vtkLagrangeQuadrilateral

    if vtk_type in _SIMPLEX:
        return _simplex_vtk_index(vtk_type, order)
    q = order + 1
    if vtk_type == 72:
        return np.array([vtkLagrangeHexahedron.PointIndexFromIJK(i, j, k, [order] * 3)
                         for k in range(q) for j in range(q) for i in range(q)], np.int64)
    if vtk_type == 70:
        return np.array([vtkLagrangeQuadrilateral.PointIndexFromIJK(i, j, [order] * 2)
                         for j in range(q) for i in range(q)], np.int64)
    # A curve: its two ends, then its interior nodes in order.
    return np.array([0 if i == 0 else 1 if i == order else i + 1 for i in range(q)],
                    np.int64)


def _quadratic_vtk_index(vtk_type, stored):
    """For each of Tack's stored order-2 nodes (``stored``, lexicographic numbers in
    ``bases.lattice_ijk``), its place in a VTK quadratic cell of ``vtk_type``, by the
    parametric coordinates VTK gives its points."""
    from vtkmodules.vtkCommonDataModel import (
        vtkQuadraticHexahedron,
        vtkQuadraticQuad,
        vtkQuadraticTetra,
        vtkQuadraticTriangle,
        vtkQuadraticWedge,
    )

    from tack.dataset import bases, shapes

    lagrange = QUADRATIC[vtk_type][0]
    dimension = LAGRANGE[lagrange][1]
    cell = {22: vtkQuadraticTriangle, 23: vtkQuadraticQuad, 24: vtkQuadraticTetra,
            25: vtkQuadraticHexahedron, 26: vtkQuadraticWedge}[vtk_type]()
    pcoords = np.asarray(cell.GetParametricCoords(), np.float64)
    pcoords = pcoords[:3 * cell.GetNumberOfPoints()].reshape(-1, 3)[:, :dimension]
    place = {tuple(row): k for k, row in enumerate(np.rint(pcoords * 2).astype(np.int64).tolist())}
    lattice = bases.lattice_ijk(getattr(shapes, LAGRANGE[lagrange][0]), 2)
    return np.array([place[tuple(int(v) for v in lattice[n])] for n in stored], np.int64)


def lagrange_order(vtk_type, count):
    """The order of a ``vtk_type`` Lagrange cell of ``count`` nodes."""
    from math import comb

    dimension = LAGRANGE[vtk_type][1]
    if vtk_type in _SIMPLEX:
        size = ((lambda p: comb(p + 2, 2) * (p + 1)) if vtk_type == 73
                else (lambda p: comb(p + dimension, dimension)))
        order = next((p for p in range(64) if size(p) >= count), -1)
        if size(order) != count:
            raise ValueError(f"a Lagrange simplex of {count} nodes has no order in "
                             f"{dimension}D")
        return order
    order = round(count ** (1.0 / dimension)) - 1
    if (order + 1) ** dimension != count:
        raise ValueError(f"a Lagrange cell of {count} nodes has no order in {dimension}D")
    return order


def to_dataset(grid, types, dtype, id_dtype, shift, unit_of, float_fields):
    """A ``vtkUnstructuredGrid`` of one kind of Lagrange or Bezier cell, all of one
    order, as a dataset: linear cells on the corner points, the geometry and point
    data in ``H1(order=p, basis="equispaced")`` (Lagrange) or ``"bernstein"``
    (Bezier), cell data ``Constant``."""
    from tack.dataset import H1, Constant, DataSet, Field, UnstructuredTopology, shapes
    from tack.dataset.topology import uniform_cells
    from tack.interop.mfem import _h1_indices

    (vtk_type,) = types
    basis = "equispaced"
    quadratic = vtk_type if vtk_type in QUADRATIC else None
    if quadratic is not None:
        vtk_type, basis = QUADRATIC[quadratic]
    weights = grid.point_data.GetRationalWeights()
    skipped = {"vtkGhostType", weights.GetName() if weights is not None else None}
    if vtk_type in BEZIER:
        if weights is not None and np.any(np.asarray(weights) != 1.0):
            raise NotImplementedError("rational Bezier cells (weights other than 1) are not "
                                      "supported")
        vtk_type, basis = BEZIER[vtk_type], "bernstein"
    cells = grid.cells
    offsets = np.asarray(cells.offsets_array, np.int64)
    connectivity = np.asarray(cells.connectivity_array, np.int64)
    counts = np.diff(offsets)
    n = len(counts)
    if n == 0 or len(set(counts.tolist())) != 1:
        raise NotImplementedError("Lagrange cells of several orders in one grid are not "
                                  "supported")
    order = 2 if quadratic is not None else lagrange_order(vtk_type, int(counts[0]))
    per_cell = int(counts[0])
    rows = connectivity.reshape(n, per_cell)
    corners_of = (6 if vtk_type == 73 else LAGRANGE[vtk_type][1] + 1 if vtk_type in _SIMPLEX
                  else 2 ** LAGRANGE[vtk_type][1])
    corner_ids, compact = np.unique(rows[:, :corners_of], return_inverse=True)
    points = np.asarray(grid.points.data, np.float64)
    types_, cell_offsets = uniform_cells(_LINEAR[vtk_type], n, tack.i64)
    topology = UnstructuredTopology(types_, cell_offsets,
                                    compact.reshape(-1).astype(np.int64),
                                    num_points=len(corner_ids), id_dtype=id_dtype)
    data = DataSet(topology, points[corner_ids] - shift, dtype=dtype)
    if order == 1:
        space = H1(data)
        vtk_of_dof = corner_ids
    else:
        space = H1(data, order=order, **({"basis": basis} if basis else {}))
        indices = _h1_indices(data, space, per_cell)
        if quadratic is not None:
            shape = getattr(shapes, LAGRANGE[vtk_type][0])
            stored = space.stored_nodes(shape)
            if len(stored) != per_cell:
                raise ValueError(f"a VTK quadratic cell of type {quadratic} has "
                                 f"{len(stored)} points, not {per_cell}")
            lex_to_vtk = _quadratic_vtk_index(quadratic, stored)
        else:
            lex_to_vtk = _vtk_index(vtk_type, order)
        vtk_points = rows[:, lex_to_vtk]
        vtk_of_dof = np.full(space.size, -1, np.int64)
        vtk_of_dof[indices.reshape(-1)] = vtk_points.reshape(-1)
        if (vtk_of_dof < 0).any() or not np.array_equal(vtk_of_dof[indices.reshape(-1)],
                                                        vtk_points.reshape(-1)):
            raise NotImplementedError("the Lagrange cells do not share their points as an "
                                      "H1 space shares its nodes (each cell's own copies)")
        data.geometry = Field(space, _vectors(points[vtk_of_dof] - shift, dtype))
    for name in grid.point_data.keys():
        array = grid.point_data[name]
        values = np.asarray(array)
        if name in skipped:
            continue
        values = values.reshape(values.shape[0], -1)[vtk_of_dof]
        data.fields[name] = Field(space, float_fields(values), unit_of(array))
    for name in grid.cell_data.keys():
        if name in ("vtkGhostType", "HigherOrderDegrees"):
            continue
        array = grid.cell_data[name]
        values = np.asarray(array)
        values = values.reshape(values.shape[0], -1)
        data.fields[name if name not in data.fields else f"{name} (cells)"] = Field(
            Constant(data), float_fields(values), unit_of(array))
    return data


def _vectors(values, dtype):
    out = tack.empty((values.shape[0],), tack.vec(3, dtype))
    if values.shape[0]:
        tack.copyto(out, values.astype(dtype.numpy_dtype))
    return out


def tensor_order(data):
    """The order a Lagrange export of ``data`` takes: the largest of its geometry and
    ``H1`` fields, on segments, quads and hexahedra alone; 1 when there is nothing
    above order 1 or other cells."""
    from tack.dataset import H1, shapes

    tensor = {shapes.Line, shapes.Quad, shapes.Hexahedron, shapes.Pixel, shapes.Voxel}
    simplex = {shapes.Triangle, shapes.Tetra}
    groups = [g for g in data.topology.groups() if g.count]
    kinds = {g.shape for g in groups}
    if not groups or not (kinds <= tensor or kinds <= simplex or kinds == {shapes.Wedge}) or \
            len({g.shape.DIMENSION for g in groups}) != 1:
        return 1
    orders = [f.space.order for f in [data.geometry, *data.fields.values()]
              if isinstance(f.space, H1)]
    return max(orders, default=1)


def bernstein(data):
    """Whether ``data``'s geometry or an ``H1`` field is in a Bernstein basis."""
    from tack.dataset import H1, bases

    return any(isinstance(f.space, H1) and f.space.basis == bases.BERNSTEIN
               for f in [data.geometry, *data.fields.values()])


def serendipity(data):
    """Whether ``data``'s geometry or an ``H1`` field is serendipity."""
    from tack.dataset import H1, bases

    return any(isinstance(f.space, H1) and f.space.basis == bases.SERENDIPITY
               for f in [data.geometry, *data.fields.values()])


def to_vtk(data, order, bezier=False, quadratic=False):
    """``data``'s cells as VTK Lagrange cells of ``order``: points at the equispaced
    nodes of ``H1(order)`` (each shared node once), the geometry and ``H1``
    fields evaluated there (exact for a field of that order or below), cell
    fields as cell data. With ``bezier``, VTK Bezier cells: the points and point
    data the coefficients of ``H1(order, basis="bernstein")`` (the same
    polynomials, the same numbering). With ``quadratic`` (order 2), VTK's classic
    quadratic cells: quad8, hex20 and wedge15 the coefficients of serendipity, and
    quadratic triangles and tetrahedra the full order-2 space's."""
    from math import comb

    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid

    from tack.dataset import H1, Constant, Values, bases, project
    from tack.interop.mfem import _h1_indices

    shape = next(g.shape for g in data.topology.groups() if g.count)
    simplex = bases.simplex_shape(shape)
    wedge = bases.wedge_shape(shape)
    vtk_type = 73 if wedge else ({2: 69, 3: 71} if simplex else
                                 {1: 68, 2: 70, 3: 72})[shape.DIMENSION]
    if quadratic:
        return _to_quadratic_vtk(data, shape, vtk_type)
    space = H1(data, order=order, basis="bernstein" if bezier else "equispaced")
    per_cell = (comb(order + 2, 2) * (order + 1) if wedge else
                comb(order + shape.DIMENSION, shape.DIMENSION) if simplex
                else (order + 1) ** shape.DIMENSION)
    indices = _h1_indices(data, space, per_cell)
    vtk_to_lex = np.argsort(_vtk_index(vtk_type, order))
    connectivity = indices[:, vtk_to_lex].reshape(-1)
    positions = project(data, data.geometry, space).values.numpy().astype(np.float64)
    positions = positions + np.asarray(data.origin)
    grid = vtkUnstructuredGrid()
    grid.points = vtkPoints(data=positions)
    offsets = np.arange(0, (data.num_cells + 1) * per_cell, per_cell, dtype=np.int64)
    written = next((b for b, lagrange in BEZIER.items() if lagrange == vtk_type),
                   None) if bezier else vtk_type
    grid.cells = (written, vtkCellArray(offsets=offsets,
                                        connectivity=connectivity.astype(np.int64)))
    for name, field in data.fields.items():
        if isinstance(field.space, H1) or (isinstance(field.space, Values)
                                           and field.space.on == "points"):
            grid.point_data[name] = project(data, field, space).values.numpy()
        elif isinstance(field.space, Constant) or (isinstance(field.space, Values)
                                                   and field.space.on == "cells"):
            from tack import arrays

            grid.cell_data[name] = arrays.materialize(field.values).numpy()
    return grid


def _to_quadratic_vtk(data, shape, lagrange):
    """``to_vtk``'s VTK quadratic cells: each cell's stored order-2 nodes (serendipity's
    corners and edges on quads, hexahedra and wedges), each shared node one point."""
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid

    from tack import arrays
    from tack.dataset import H1, Constant, Values, project
    from tack.interop.mfem import _h1_indices

    vtk_type = next((q for q, (twin, _) in QUADRATIC.items() if twin == lagrange), None)
    if vtk_type is None:
        raise NotImplementedError(f"VTK has no classic quadratic {shape.__name__} cell")
    basis = QUADRATIC[vtk_type][1]
    space = H1(data, order=2, **({"basis": basis} if basis else {}))
    stored = space.stored_nodes(shape)
    per_cell = len(stored)
    indices = _h1_indices(data, space, per_cell)
    vtk_to_tack = np.argsort(_quadratic_vtk_index(vtk_type, stored))
    connectivity = indices[:, vtk_to_tack].reshape(-1)
    positions = project(data, data.geometry, space).values.numpy().astype(np.float64)
    grid = vtkUnstructuredGrid()
    grid.points = vtkPoints(data=positions + np.asarray(data.origin))
    offsets = np.arange(0, (data.num_cells + 1) * per_cell, per_cell, dtype=np.int64)
    grid.cells = (vtk_type, vtkCellArray(offsets=offsets,
                                         connectivity=connectivity.astype(np.int64)))
    for name, field in data.fields.items():
        if isinstance(field.space, H1) or (isinstance(field.space, Values)
                                           and field.space.on == "points"):
            grid.point_data[name] = project(data, field, space).values.numpy()
        elif isinstance(field.space, Constant) or (isinstance(field.space, Values)
                                                   and field.space.on == "cells"):
            grid.cell_data[name] = arrays.materialize(field.values).numpy()
    return grid
