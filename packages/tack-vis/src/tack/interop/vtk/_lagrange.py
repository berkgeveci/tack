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


#: VTK's other higher-order cells: (Tack's shape, the order of Tack's space that holds
#: them). Their nodes are found by VTK's parametric coordinates; where Tack's cell
#: has nodes they lack, VTK's own interpolation functions give them (exactly: each
#: space lies in Tack's of that order).
_CLASSIC = {21: ("Line", 2), 22: ("Triangle", 2), 23: ("Quad", 2), 24: ("Tetra", 2),
            25: ("Hexahedron", 2), 26: ("Wedge", 2), 28: ("Quad", 2),
            29: ("Hexahedron", 2), 30: ("Quad", 2), 31: ("Wedge", 2), 32: ("Wedge", 2),
            33: ("Hexahedron", 2), 34: ("Triangle", 3), 35: ("Line", 3)}
#: VTK's linear cells beside higher-order ones: order 1 in the grid's space.
_LINEAR_SHAPES = {3: "Line", 5: "Triangle", 9: "Quad", 10: "Tetra", 12: "Hexahedron",
                  13: "Wedge", 14: "Pyramid"}
#: The types whose space is Tack's order-2 serendipity space (or, on simplices, the
#: full one): a grid of these alone reads into it.
_SERENDIPITY = {21, 22, 23, 24, 25, 26}
#: Higher-order types Tack has no cell for.
REFUSED = {27: "quadratic pyramids", 36: "quadratic polygons", 37: "triquadratic pyramids",
           74: "Lagrange pyramids", 81: "Bezier pyramids"}
#: Every higher-order type the importer takes.
HIGHER_ORDER = set(LAGRANGE) | set(BEZIER) | set(_CLASSIC)


def _shape_of(vtk_type):
    """``(Tack shape, order or None)``: the shape a VTK higher-order type is, and the
    order of Tack's space for it (``None``: a Lagrange or Bezier cell's own, from its
    node count)."""
    from tack.dataset import shapes

    if vtk_type in _LINEAR_SHAPES:
        return getattr(shapes, _LINEAR_SHAPES[vtk_type]), 1
    if vtk_type in _CLASSIC:
        name, order = _CLASSIC[vtk_type]
        return getattr(shapes, name), order
    twin = BEZIER.get(vtk_type, vtk_type)
    return getattr(shapes, LAGRANGE[twin][0]), None


def _parametric_nodes(shape, order, stored=None):
    """Tack's nodes of ``shape`` at ``order`` (``stored`` lexicographic ones, or all),
    as parametric points ``(n, 3)``."""
    from tack.dataset import bases

    lattice = np.asarray(bases.lattice_ijk(shape, order), np.float64) / order
    lattice = lattice if stored is None else lattice[stored]
    out = np.zeros((len(lattice), 3))
    out[:, :lattice.shape[1]] = lattice
    return out


def _node_matrix(cell, vtk_type, nodes):
    """How a VTK cell's node values give Tack's ``nodes`` (parametric points): an index
    per node where it is one of the cell's own nodes (``-1`` where not), and the
    interpolation weights ``(len(nodes), cell points)`` from VTK's functions."""
    n = cell.GetNumberOfPoints()
    own = np.asarray(cell.GetParametricCoords(), np.float64)[:3 * n].reshape(n, 3)
    if vtk_type == 35:                                   # vtkCubicLine's r is on [-1, 1]
        own = own.copy()
        own[:, 0] = (own[:, 0] + 1.0) / 2.0
    index = np.full(len(nodes), -1, np.int64)
    for k, node in enumerate(nodes):
        hit = np.flatnonzero(np.abs(own - node).max(axis=1) < 1e-9)
        if len(hit):
            index[k] = hit[0]
    weights = np.zeros((len(nodes), n))
    row = [0.0] * n
    for k, node in enumerate(nodes):
        at = list(node)
        if vtk_type == 35:
            at[0] = 2.0 * at[0] - 1.0
        cell.InterpolateFunctions(at, row)
        weights[k] = row
    return index, weights


@tack.kernel
def _node_places(cells, u, node_of, starts, out):
    # Each cell's node n (lexicographic, node_of its place in the view's order):
    # where the space keeps it.
    for c in cells:
        e = cells.entity_id(c)
        for n in range(node_of.shape[0]):
            out[starts[e] + n] = u.dof_index(c, node_of[n])


def to_dataset(grid, types, dtype, id_dtype, shift, unit_of, float_fields):
    """A ``vtkUnstructuredGrid`` of VTK's higher-order cells -- Lagrange, Bezier, the
    classic quadratic, biquadratic and triquadratic ones, cubic lines -- of any
    shapes, dimensions and orders together, as a dataset: linear cells on their
    corners, the geometry and point data in ``H1`` of the grid's largest order
    (equispaced; ``"bernstein"`` for Bezier cells; serendipity for a grid of quad8,
    hex20 and wedge15 with the quadratic simplices), cell data ``Constant``.

    Each type's nodes are matched to Tack's by VTK's parametric coordinates and
    copied; where Tack's cell has nodes a VTK cell lacks (a hex20 in a grid of
    order 2, a quadratic cell beside cubic ones, a quadratic-linear quad), VTK's
    interpolation functions give them, exactly. A node shared by cells takes its
    first cell's value. Pyramids above order 1, quadratic polygons and rational
    Bezier cells are refused, and Bezier cells beside others."""
    from tack.arrays import ConstantArray
    from tack.dataset import H1, Constant, DataSet, Field, UnstructuredTopology, bases, ids

    refused = sorted(REFUSED[t] for t in types if t in REFUSED)
    if refused:
        raise NotImplementedError(f"{', '.join(refused)}: Tack has no such higher-order cell")
    bezier = {t for t in types if t in BEZIER}
    if bezier and bezier != set(types):
        raise NotImplementedError("Bezier cells beside other higher-order cells")
    weights_array = grid.point_data.GetRationalWeights()
    skipped = {"vtkGhostType", weights_array.GetName() if weights_array is not None else None}
    if bezier and weights_array is not None and np.any(np.asarray(weights_array) != 1.0):
        raise NotImplementedError("rational Bezier cells (weights other than 1) are not "
                                  "supported")
    cell_types = np.asarray(grid.cell_types, np.int64).reshape(-1)
    offsets = np.asarray(grid.cells.offsets_array, np.int64)
    connectivity = np.asarray(grid.cells.connectivity_array, np.int64)
    counts = np.diff(offsets)
    n = len(counts)
    # Each cell's (VTK type, node count): its kind, and its Tack shape and order.
    kinds = {}
    for vtk_type, count in set(zip(cell_types.tolist(), counts.tolist(), strict=True)):
        shape, order = _shape_of(vtk_type)
        if order is None:
            order = lagrange_order(BEZIER.get(vtk_type, vtk_type), count)
        kinds[(vtk_type, count)] = (shape, order)
    order = max(o for _, o in kinds.values())
    if bezier and len({o for _, o in kinds.values()}) > 1:
        raise NotImplementedError("Bezier cells of several orders (degree elevation is not "
                                  "supported)")
    basis = ("bernstein" if bezier else "serendipity"
             if order == 2 and set(types) <= _SERENDIPITY | set(_LINEAR_SHAPES)
             and set(types) & {23, 25, 26}
             else "equispaced")

    # The topology: each cell's corners (VTK lists them first), renumbered.
    shape_of_cell = np.empty(n, object)
    corner_counts = np.empty(n, np.int64)
    for (vtk_type, count), (shape, _) in kinds.items():
        mask = (cell_types == vtk_type) & (counts == count)
        shape_of_cell[mask] = shape
        corner_counts[mask] = shape.NUM_POINTS
    corner_rows = [connectivity[offsets[c]:offsets[c] + corner_counts[c]] for c in range(n)]
    corners = np.concatenate(corner_rows) if n else np.zeros(0, np.int64)
    corner_ids, compact = np.unique(corners, return_inverse=True)
    linear_types = np.array([s.ID for s in shape_of_cell], np.uint8)
    cell_offsets = np.concatenate([[0], np.cumsum(corner_counts)])
    idt = ids.at_least(id_dtype or tack.i32, max(len(corner_ids), len(compact)))
    topology = UnstructuredTopology(
        tack.array(linear_types), tack.array(cell_offsets.astype(idt.numpy_dtype)),
        tack.array(compact.astype(idt.numpy_dtype)), num_points=len(corner_ids),
        id_dtype=idt)
    points = np.asarray(grid.points.data, np.float64)
    data = DataSet(topology, points[corner_ids] - shift, dtype=dtype)

    point_arrays = [name for name in grid.point_data.keys() if name not in skipped]
    sources = [points] + [np.asarray(grid.point_data[name]).reshape(len(points), -1)
                          for name in point_arrays]
    if order == 1:
        space = H1(data)
        node_values = [source[corner_ids] for source in sources]
    else:
        space = H1(data, order=order, basis=basis)
        # Where each cell's nodes live in the space, cell by cell.
        stored = {s: space.stored_nodes(s) for s in set(shape_of_cell.tolist())}
        sizes = np.array([len(stored[s]) for s in shape_of_cell], np.int64)
        starts = np.concatenate([[0], np.cumsum(sizes)])
        places = tack.empty((max(int(starts[-1]), 1),), ids.at_least(idt, space.size))
        field = Field(space, ConstantArray(0.0, space.size))
        starts_array = tack.array(starts[:-1].astype(np.int64))
        for group in data.launch_groups("cells", [field]):
            if group.count:
                node_of = bases.view_order(group.shape, order, space._any_order(group.shape))
                node_of = np.asarray(node_of)[stored[group.shape]]
                _node_places(data.domain_view("cells", group), field.view(group),
                             tack.array(node_of.astype(np.int32)), starts_array, places)
        places = places.numpy()[:int(starts[-1])]
        node_values = [np.zeros((space.size, source.shape[1])) for source in sources]
        written = np.zeros(space.size, bool)
        shared_bad = False
        for (vtk_type, count), (shape, _) in kinds.items():
            cells_of = np.flatnonzero((cell_types == vtk_type) & (counts == count))
            nodes = _parametric_nodes(shape, order, stored[shape])
            index, weights = _node_matrix(grid.GetCell(int(cells_of[0])), vtk_type, nodes)
            rows = np.stack([connectivity[offsets[c]:offsets[c] + count] for c in cells_of])
            at = np.stack([places[starts[c]:starts[c + 1]] for c in cells_of])
            if (index >= 0).all():
                # Tack's nodes are the cell's own: copied, to the bit.
                vtk_points = rows[:, index]
                for out, source in zip(node_values, sources, strict=True):
                    out[at.reshape(-1)] = source[vtk_points.reshape(-1)]
                seen = np.full(space.size, -1, np.int64)
                seen[at.reshape(-1)] = vtk_points.reshape(-1)
                shared_bad |= bool((seen[at.reshape(-1)] != vtk_points.reshape(-1)).any())
            else:
                if bezier:
                    raise NotImplementedError("Bezier cells whose nodes are not Tack's")
                for out, source in zip(node_values, sources, strict=True):
                    values = np.einsum("nj,cjk->cnk", weights, source[rows])
                    # The cell's own nodes copied, as above; the others interpolated.
                    own = index >= 0
                    values[:, own] = source[rows[:, index[own]]]
                    fresh = ~written[at]
                    out[at[fresh]] = values[fresh]
            written[at.reshape(-1)] = True
        if shared_bad:
            raise NotImplementedError("the higher-order cells do not share their points as an "
                                      "H1 space shares its nodes (each cell's own copies)")
        data.geometry = Field(space, _vectors(node_values[0] - shift, dtype))
    for name, values in zip(point_arrays, node_values[1:], strict=True):
        array = grid.point_data[name]
        if not np.issubdtype(np.asarray(array).dtype, np.floating) and order > 1:
            values = values.astype(np.asarray(array).dtype)
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
