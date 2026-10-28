"""VTK's Lagrange cells -- ``VTK_LAGRANGE_CURVE``, ``VTK_LAGRANGE_QUADRILATERAL`` and
``VTK_LAGRANGE_HEXAHEDRON`` -- as Tack's tensor bases on equispaced nodes, both
ways.

A Lagrange cell of order ``p`` lists its ``(p + 1)^d`` nodes in VTK's order
(corners, then edge, face and interior nodes; ``vtkLagrangeHexahedron
.PointIndexFromIJK``), on equispaced nodes over ``[0, 1]`` in each axis: Tack's
``"equispaced"`` family on the same reference cell, the corners in the linear
cell's order. Its points are shared with the cells around it, as an ``H1``
space's nodes are, so a grid of such cells is a linear topology of their corners
and fields in ``H1(order=p, nodes="equispaced")``.
"""

import numpy as np

import tack

#: VTK type: (Tack's linear shape, dimension).
LAGRANGE = {68: ("Line", 1), 70: ("Quad", 2), 72: ("Hexahedron", 3)}
_LINEAR = {68: 3, 70: 9, 72: 12}


def _vtk_index(vtk_type, order):
    """For each lexicographic node ``n`` (first axis fastest), its place in a VTK
    Lagrange cell of ``order``."""
    from vtkmodules.vtkCommonDataModel import vtkLagrangeHexahedron, vtkLagrangeQuadrilateral

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


def lagrange_order(vtk_type, count):
    """The order of a ``vtk_type`` Lagrange cell of ``count`` nodes."""
    dimension = LAGRANGE[vtk_type][1]
    order = round(count ** (1.0 / dimension)) - 1
    if (order + 1) ** dimension != count:
        raise ValueError(f"a Lagrange cell of {count} nodes has no order in {dimension}D")
    return order


def to_dataset(grid, types, dtype, id_dtype, shift, unit_of, float_fields):
    """A ``vtkUnstructuredGrid`` of one kind of Lagrange cell, all of one order, as a
    dataset: linear cells on the corner points, the geometry and point data in
    ``H1(order=p, nodes="equispaced")``, cell data ``Constant``."""
    from tack.dataset import H1, Constant, DataSet, Field, UnstructuredTopology
    from tack.dataset.topology import uniform_cells
    from tack.interop.mfem import _h1_indices

    (vtk_type,) = types
    cells = grid.cells
    offsets = np.asarray(cells.offsets_array, np.int64)
    connectivity = np.asarray(cells.connectivity_array, np.int64)
    counts = np.diff(offsets)
    n = len(counts)
    if n == 0 or len(set(counts.tolist())) != 1:
        raise NotImplementedError("Lagrange cells of several orders in one grid are not "
                                  "supported")
    order = lagrange_order(vtk_type, int(counts[0]))
    per_cell = int(counts[0])
    rows = connectivity.reshape(n, per_cell)
    corners_of = 2 ** LAGRANGE[vtk_type][1]
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
        space = H1(data, order=order, nodes="equispaced")
        indices = _h1_indices(data, space, per_cell)
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
        if name == "vtkGhostType":
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

    allowed = {shapes.Line, shapes.Quad, shapes.Hexahedron, shapes.Pixel, shapes.Voxel}
    groups = [g for g in data.topology.groups() if g.count]
    if not groups or any(g.shape not in allowed for g in groups) or \
            len({g.shape.DIMENSION for g in groups}) != 1:
        return 1
    orders = [f.space.order for f in [data.geometry, *data.fields.values()]
              if isinstance(f.space, H1)]
    return max(orders, default=1)


def to_vtk(data, order):
    """``data``'s cells as VTK Lagrange cells of ``order``: points at the equispaced
    nodes of ``H1(order)`` (each shared node once), the geometry and ``H1``
    fields evaluated there (exact for a field of that order or below), cell
    fields as cell data."""
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid

    from tack.dataset import H1, Constant, Values, project
    from tack.interop.mfem import _h1_indices

    shape = next(g.shape for g in data.topology.groups() if g.count)
    vtk_type = {1: 68, 2: 70, 3: 72}[shape.DIMENSION]
    space = H1(data, order=order, nodes="equispaced")
    per_cell = (order + 1) ** shape.DIMENSION
    indices = _h1_indices(data, space, per_cell)
    vtk_to_lex = np.argsort(_vtk_index(vtk_type, order))
    connectivity = indices[:, vtk_to_lex].reshape(-1)
    positions = project(data, data.geometry, space).values.numpy().astype(np.float64)
    positions = positions + np.asarray(data.origin)
    grid = vtkUnstructuredGrid()
    grid.points = vtkPoints(data=positions)
    offsets = np.arange(0, (data.num_cells + 1) * per_cell, per_cell, dtype=np.int64)
    grid.cells = (vtk_type, vtkCellArray(offsets=offsets,
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
