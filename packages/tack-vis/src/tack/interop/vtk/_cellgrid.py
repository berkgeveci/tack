"""VTK cell grids (``vtkCellGrid``, the discontinuous-Galerkin cells of VTK's
Filters/CellGrid) as Tack datasets.

A cell grid keeps, per cell type (``vtkDGHex``, ...), a corner connectivity and,
per attribute and cell type, its function space, basis, order and DOF sharing
(``vtkCellAttribute``'s cell-type information, readable from Python since VTK's
``GetCellTypes``/``GetDOFSharing``/... accessors). Tack reads it so:

- Cells: each type's cells, types in VTK's cell order (by their cell specs'
  offsets), on the cell spec's corner ids (VTK's DG corner orders are its classic
  cells', which are Tack's).
- The shape attribute (geometry) must be ``HGRAD`` and shared. Its order-1
  complete basis (``C1``) is the points themselves, copied; a higher order is
  resampled onto ``H1(order=p, basis="equispaced")``.
- ``constant`` attributes are ``Constant`` fields.
- ``HGRAD`` ``C1``: shared on the shape's points, an ``H1`` field of the group's
  values; per cell, an ``L2`` field of its corner values, copied.
- Every other ``HGRAD`` basis (``C2``, serendipity ``I2``, the Gauss-point ``G``
  bases, arbitrary-order ``A``) and ``Bezier``: VTK evaluates the attribute at
  the nodes of Tack's equispaced space of that order (``vtkCellGridEvaluator``'s
  interpolate-only mode), per cell into ``L2``, shared into ``H1``. Where the two
  spaces are the same polynomials (every one but the enriched ``F`` bases and
  pyramids above order 1, which are left out) this is exact, to rounding.
- ``HCURL`` and ``HDIV`` have no Tack space: left out, with a warning.

Reference coordinates: VTK's tensor axes (an edge's, a quad's, a hexahedron's,
a wedge's ``t``) are ``[-1, 1]``, Tack's ``[0, 1]``; simplices share the unit
simplex; Tack's pyramid is the unit cube with its top collapsed, VTK's a base on
``[-1, 1]^2`` under an apex at ``(0, 0, 1)``.
"""

import warnings

import numpy as np

import tack
from tack.dataset import shapes

#: VTK's DG cell types and Tack's shapes.
SHAPES = {"vtkDGVert": shapes.Vertex, "vtkDGEdge": shapes.Line, "vtkDGTri": shapes.Triangle,
          "vtkDGQuad": shapes.Quad, "vtkDGTet": shapes.Tetra, "vtkDGHex": shapes.Hexahedron,
          "vtkDGWdg": shapes.Wedge, "vtkDGPyr": shapes.Pyramid}

_registered = False


def _register():
    """VTK's DG cells and their responders, once (queries return nothing without)."""
    global _registered
    if not _registered:
        from vtkmodules.vtkFiltersCellGrid import vtkFiltersCellGrid

        vtkFiltersCellGrid.RegisterCellsAndResponders()
        _registered = True


def _text(token):
    """A ``vtkStringToken``'s string ("" for the empty token)."""
    return token if isinstance(token, str) else (token.Data() if token.HasData() else "")


class _Info:
    """An attribute's cell-type information for one cell type, with its arrays."""

    def __init__(self, attribute, cell_type):
        self.sharing = _text(attribute.GetDOFSharing(cell_type))
        self.space = _text(attribute.GetFunctionSpace(cell_type))
        self.basis = _text(attribute.GetBasis(cell_type))
        self.order = int(attribute.GetOrder(cell_type))
        values = attribute.GetArrayForCellTypeAndRole(cell_type, "values")
        connectivity = attribute.GetArrayForCellTypeAndRole(cell_type, "connectivity")
        self.values = None if values is None else np.asarray(values)
        self.values_name = None if values is None else values.GetName()
        self.connectivity = None if connectivity is None else \
            np.asarray(connectivity).reshape(connectivity.GetNumberOfTuples(), -1)
        self.components = attribute.GetNumberOfComponents()

    def key(self):
        return self.sharing != "", self.space, self.basis, self.order


def _to_vtk_parameters(shape, points):
    """Tack's parametric points of ``shape`` (``(n, d)``) as VTK's reference
    coordinates (``(n, 3)``): tensor axes from ``[0, 1]`` to ``[-1, 1]``."""
    points = np.asarray(points, np.float64)
    out = np.zeros((len(points), 3))
    out[:, :points.shape[1]] = points
    if shape in (shapes.Line, shapes.Quad, shapes.Hexahedron):
        out[:, :shape.DIMENSION] = 2.0 * out[:, :shape.DIMENSION] - 1.0
    elif shape is shapes.Wedge:
        out[:, 2] = 2.0 * out[:, 2] - 1.0
    elif shape is shapes.Pyramid:
        # Tack's pyramid is the unit cube with its top collapsed; VTK's has its base
        # on [-1, 1]^2 at t = 0 and its apex at (0, 0, 1).
        z = out[:, 2].copy()
        out[:, 0] = (2.0 * out[:, 0] - 1.0) * (1.0 - z)
        out[:, 1] = (2.0 * out[:, 1] - 1.0) * (1.0 - z)
    return out


def _lattice(shape, order):
    """Tack's equispaced nodes of ``shape`` at ``order``, in node order, as its
    parametric points; at order 1 its corners, in corner order (the order of an
    order-1 field's values)."""
    from tack.dataset import bases

    lattice = np.asarray(bases.lattice_ijk(shape, order), np.float64) / order
    if order == 1:
        kind, entity, _, _, _ = bases.dof_tables(shape, 1)
        corner_node = np.empty(shape.NUM_POINTS, np.int64)
        corner_node[entity[kind == bases.CORNER]] = np.flatnonzero(kind == bases.CORNER)
        return lattice[corner_node]
    return lattice


def evaluate(grid, attribute, cell_type, cells, parameters):
    """``attribute`` evaluated by VTK in cells ``cells`` (indices among
    ``cell_type``'s) at VTK reference coordinates ``parameters`` (``(n, 3)``):
    ``(n, components)`` doubles."""
    from vtkmodules.vtkCommonCore import (
        vtkDoubleArray,
        vtkStringToken,
        vtkTypeUInt32Array,
        vtkTypeUInt64Array,
    )
    from vtkmodules.vtkCommonDataModel import vtkCellGridEvaluator

    _register()
    types = vtkTypeUInt32Array()
    types.InsertNextValue(vtkStringToken(cell_type).GetHash())
    offsets = vtkTypeUInt64Array()
    offsets.InsertNextValue(0)
    indices = vtkTypeUInt64Array()
    indices.SetNumberOfTuples(len(cells))
    np.asarray(indices)[:] = np.asarray(cells, np.uint64)
    points = vtkDoubleArray()
    points.SetNumberOfComponents(3)
    points.SetNumberOfTuples(len(cells))
    np.asarray(points)[:] = parameters
    query = vtkCellGridEvaluator()
    query.SetCellAttribute(attribute)
    query.InterpolateCellParameters(types, offsets, indices, points)
    grid.Query(query)
    values = np.asarray(query.GetInterpolatedValues())
    if len(values) != len(cells):
        raise RuntimeError("VTK's cell grid evaluator interpolated no values: interpolating "
                           "given cells and parameters needs a VTK with its fix "
                           "(vtkCellGridEvaluator's interpolate-only mode)")
    return values.reshape(len(cells), -1)


def _resampled(grid, attribute, cell_types, counts, order):
    """``attribute`` at Tack's equispaced nodes of ``order`` in every cell, cell by
    cell in ``cell_types`` order: ``(cells' nodes, components)``, each cell's in its
    node order, and the nodes a cell of each type has."""
    parts, per_cell = [], []
    for cell_type, count in zip(cell_types, counts, strict=True):
        shape = SHAPES[cell_type]
        lattice = _lattice(shape, order)
        per_cell.append(len(lattice))
        if not count:
            continue
        cells = np.repeat(np.arange(count), len(lattice))
        parameters = np.tile(_to_vtk_parameters(shape, lattice), (count, 1))
        parts.append(evaluate(grid, attribute, cell_type, cells, parameters))
    width = attribute.GetNumberOfComponents()
    return (np.concatenate(parts) if parts else np.zeros((0, width))), per_cell


def _corners(grid, cell_type, shape_info):
    """``cell_type``'s cells' corner ids: its cell spec's first corners, or (without
    one) those of a corner-first shape basis's connectivity."""
    spec = grid.GetCellType(cell_type).GetCellSourceConnectivity(-1)
    if spec is not None:
        # Corner-first (VTK's sides read a cell spec's first corners): a spec may
        # share a quadratic shape's connectivity.
        rows = np.asarray(spec).reshape(spec.GetNumberOfTuples(), -1)
        return rows[:, :SHAPES[cell_type].NUM_POINTS]
    if shape_info.basis not in ("C", "I"):
        raise NotImplementedError(f"{cell_type} cells without a cell spec, their shape in basis "
                                  f"{shape_info.basis}: no corners to read")
    return shape_info.connectivity[:, :SHAPES[cell_type].NUM_POINTS]


def to_dataset(grid, dtype=tack.f32, id_dtype=None, shift=(0.0, 0.0, 0.0)):
    """A ``vtkCellGrid`` as a ``DataSet`` (see the module's docstring)."""
    from tack.dataset import H1, L2, Constant, DataSet, Field, UnstructuredTopology, ids
    from tack.interop.mfem import _h1_indices

    _register()
    # VTK numbers cells type by type from each cell spec's offset (its first id):
    # Tack's cells follow that order.
    cell_types = sorted((t for t in map(_text, grid.GetCellTypes()) if t in SHAPES),
                        key=lambda t: grid.GetCellType(t).GetCellSourceOffset(-1))
    unknown = [t for t in map(_text, grid.GetCellTypes()) if t not in SHAPES]
    if unknown:
        raise NotImplementedError(f"cell grids of {', '.join(unknown)} cells")
    counts = [grid.GetCellType(t).GetNumberOfCells() for t in cell_types]
    shape_attribute = grid.GetShapeAttribute()
    if shape_attribute is None:
        raise ValueError("a cell grid without a shape attribute has no geometry")
    infos = {t: _Info(shape_attribute, t) for t in cell_types}
    for t, info in infos.items():
        if info.space != "HGRAD" or not info.sharing:
            raise NotImplementedError(
                f"a cell grid's shape is an HGRAD attribute with shared values, not "
                f"{info.space or 'none'} {'shared' if info.sharing else 'per cell'} on {t}")
        if info.order > 1 and SHAPES[t] is shapes.Pyramid:
            raise NotImplementedError(
                f"a curved pyramid ({info.basis}{info.order}): Tack's pyramids are of order 1")
        if info.connectivity is None:
            raise NotImplementedError(f"a cell grid's shape on {t} without a connectivity")
    if len({infos[t].values_name for t in cell_types}) > 1:
        raise NotImplementedError("a cell grid whose cell types' shapes keep their points "
                                  "in different arrays")
    points = infos[cell_types[0]].values.reshape(-1, 3) if cell_types else np.zeros((0, 3))
    corners = [_corners(grid, t, infos[t]) for t in cell_types]
    sizes = [SHAPES[t].NUM_POINTS for t in cell_types]
    n = sum(counts)
    connectivity = np.concatenate([c.reshape(-1) for c in corners]) if corners else \
        np.zeros(0, np.int64)
    order = max(info.order for info in infos.values()) if infos else 1
    used = None
    if order > 1:
        # A curved shape's other nodes are not points: the topology keeps the corners
        # (renumbered), and the geometry is resampled at Tack's nodes below.
        used, connectivity = np.unique(connectivity, return_inverse=True)
        points = points[used]
    idt = ids.at_least(id_dtype or tack.i32, max(len(points), len(connectivity)))
    offsets = np.concatenate([[0], np.cumsum(np.repeat(sizes, counts))]) if n else \
        np.zeros(1, np.int64)
    types = np.repeat([int(SHAPES[t].ID) for t in cell_types], counts).astype(np.uint8)
    topology = UnstructuredTopology(tack.array(types), tack.array(offsets.astype(
        idt.numpy_dtype)), tack.array(connectivity.astype(idt.numpy_dtype)),
        num_points=len(points), id_dtype=idt)
    vec = tack.vec(3, dtype)
    positions = tack.empty((len(points),), vec)
    tack.copyto(positions, (points - np.asarray(shift)).astype(dtype.numpy_dtype))
    data = DataSet(topology, positions, origin=shift)
    if order > 1:
        space = H1(data, order=order, basis="equispaced")
        nodes, per_cell = _resampled(grid, shape_attribute, cell_types, counts, order)
        if len(set(per_cell)) > 1:
            raise NotImplementedError("a curved cell grid of several cell types")
        values = np.zeros((space.size, 3))
        values[_h1_indices(data, space, per_cell[0]).reshape(-1)] = nodes - np.asarray(shift)
        geometry = tack.empty((space.size,), vec)
        tack.copyto(geometry, values.astype(dtype.numpy_dtype))
        data.geometry = Field(space, geometry)

    left_out = []
    for attribute in grid.GetCellAttributeList():
        if attribute is shape_attribute or _text(attribute.GetName()) == \
                _text(shape_attribute.GetName()):
            continue
        name = _text(attribute.GetName())
        present = [t for t in cell_types if t in map(_text, attribute.GetCellTypes())]
        if present != cell_types:
            left_out.append(f"{name} (not on every cell type)")
            continue
        info = {t: _Info(attribute, t) for t in cell_types}
        keys = {i.key() for i in info.values()}
        if len(keys) > 1:
            left_out.append(f"{name} (its cell types in different spaces)")
            continue
        shared, space_name, basis, field_order = keys.pop()
        width = attribute.GetNumberOfComponents()
        element = dtype if width == 1 else tack.vec(width, dtype)
        if space_name == "constant":
            values = np.concatenate([i.values.reshape(-1, width) for i in info.values()])
            data.fields[name] = Field(Constant(data), _array(values, element, dtype))
        elif space_name == "HGRAD" and basis == "C" and field_order == 1:
            if shared:
                same = all(np.array_equal(info[t].connectivity[:, :SHAPES[t].NUM_POINTS],
                                          corners[k]) for k, t in enumerate(cell_types))
                values = info[cell_types[0]].values.reshape(-1, width)
                values = values if used is None or len(values) <= used.max() else values[used]
                if not same or len(values) != len(points):
                    left_out.append(f"{name} (shared on other points than the shape's)")
                    continue
                data.fields[name] = Field(H1(data), _array(values, element, dtype))
            else:
                values = np.concatenate([i.values.reshape(-1, width) for i in info.values()])
                data.fields[name] = Field(L2(data), _array(values, element, dtype))
        elif space_name in ("HGRAD", "Bezier") and basis != "F" and \
                shapes.Pyramid not in [SHAPES[t] for t in cell_types]:
            # Pyramids' other bases are rational (and Tack's order 1 is not): left out.
            target = max(field_order, 1)
            nodes, per_cell = _resampled(grid, attribute, cell_types, counts, target)
            if shared and target == 1:
                space = H1(data)
                values = np.zeros((len(points), width))
                values[connectivity] = nodes
            elif shared:
                space = H1(data, order=target, basis="equispaced")
                if len(set(per_cell)) > 1:
                    left_out.append(f"{name} (shared on several cell types)")
                    continue
                values = np.zeros((space.size, width))
                values[_h1_indices(data, space, per_cell[0]).reshape(-1)] = nodes
            else:
                space = L2(data, order=target, basis="equispaced", layout="blocks") \
                    if target > 1 else L2(data)
                values = nodes
            data.fields[name] = Field(space, _array(values, element, dtype))
        else:
            left_out.append(f"{name} ({space_name} {basis}{field_order})")
    if left_out:
        warnings.warn("vtk_to_dataset left out cell grid attributes Tack has no space for: "
                      + ", ".join(left_out), stacklevel=3)
    return data


def _array(values, element, dtype):
    """Host ``values`` (``(n, width)``) as a Tack array of ``element``."""
    values = np.asarray(values, np.float64)
    out = tack.empty((len(values),), element)
    tack.copyto(out, (values.reshape(-1) if values.shape[1] == 1 else values)
                .astype(dtype.numpy_dtype))
    return out


# ── Writing ─────────────────────────────────────────────────────────

#: Tack's shapes and VTK's DG cell types.
CELL_TYPES = {shape: name for name, shape in SHAPES.items()}

_SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _space_name(width):
    """VTK's name of a ``width``-component attribute's space (``"ℝ³"``)."""
    return "ℝ" + str(width).translate(_SUPERSCRIPTS)


def _vtk_array(values, name, ids=False):
    """Host ``values`` (``(n, components)``) as a VTK array named ``name``."""
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkIdTypeArray

    values = np.ascontiguousarray(values, np.int64 if ids else np.float64)
    out = (vtkIdTypeArray if ids else vtkDoubleArray)()
    out.SetName(name)
    out.SetNumberOfComponents(values.shape[1] if values.ndim > 1 else 1)
    out.SetNumberOfTuples(values.shape[0])
    np.asarray(out).reshape(values.shape)[...] = values
    return out


def _host(values, width):
    """A field's values on the host, ``(n, width)`` doubles."""
    from tack import arrays

    return np.asarray(arrays.to_host(arrays.materialize(values)), np.float64).reshape(-1, width)


def to_cell_grid(data):
    """``data`` as a ``vtkCellGrid``: each shape's cells a DG cell type (in Tack's
    cell order within it), the geometry its shape attribute and each field an
    attribute -- order 1 in VTK's complete basis ``C`` (shared through the points
    for ``H1``, per cell for ``L2``), higher orders in its arbitrary-order basis
    ``A`` (lexicographic, the first axis fastest, equispaced, as Tack's
    equispaced nodes), ``Constant`` fields ``constant``. Fields of other spaces
    are left out with a warning."""
    from vtkmodules.vtkCommonDataModel import vtkCellAttribute, vtkCellGrid
    from vtkmodules.vtkFiltersCellGrid import (
        vtkDGEdge,
        vtkDGHex,
        vtkDGPyr,
        vtkDGQuad,
        vtkDGTet,
        vtkDGTri,
        vtkDGVert,
        vtkDGWdg,
    )

    from tack import arrays
    from tack.dataset import H1, L2, Constant, UnstructuredTopology, Values
    from tack.dataset.algorithms.extract_cells import extract_cells
    from tack.dataset.algorithms.project import project
    from tack.interop.mfem import _h1_indices

    _register()
    classes = {"vtkDGVert": vtkDGVert, "vtkDGEdge": vtkDGEdge, "vtkDGTri": vtkDGTri,
               "vtkDGQuad": vtkDGQuad, "vtkDGTet": vtkDGTet, "vtkDGHex": vtkDGHex,
               "vtkDGWdg": vtkDGWdg, "vtkDGPyr": vtkDGPyr}
    if not isinstance(data.topology, UnstructuredTopology):
        if not getattr(data.topology, "reference_cells", True):
            raise NotImplementedError("a cell grid of polygons, polyhedra or polylines: VTK's "
                                      "DG cells have reference elements")
        data = extract_cells(data, np.arange(data.num_cells))
    if not isinstance(data.geometry.space, H1):
        raise NotImplementedError("a cell grid's shape is shared (an H1 geometry)")
    topology = data.topology
    types, offsets, connectivity = (
        np.asarray(arrays.to_host(arrays.materialize(a))).reshape(-1)
        for a in (topology.types, topology.offsets, topology.connectivity))
    by_id = {int(shape.ID): shape for shape in SHAPES.values()}
    seen = list(dict.fromkeys(types.tolist()))           # types by first appearance
    present = [by_id[t] for t in seen if t in by_id]
    if set(types.tolist()) - set(by_id):
        raise NotImplementedError("a cell grid of shapes VTK's DG cells do not have "
                                  "(pixels and voxels are taken as quads and hexahedra)")
    cells = {shape: np.flatnonzero(types == shape.ID) for shape in present}
    grid = vtkCellGrid()
    corners = {}
    first = 0
    for shape in present:
        name = CELL_TYPES[shape]
        rows = np.stack([connectivity[offsets[c]:offsets[c] + shape.NUM_POINTS]
                         for c in cells[shape]])
        conn = _vtk_array(rows, "conn", ids=True)
        grid.GetAttributes(name).AddArray(conn)
        metadata = classes[name]()
        grid.AddCellMetadata(metadata)
        metadata.SetCellSpec(conn, first)          # its cells' ids from first
        corners[shape] = conn
        first += len(cells[shape])

    left_out = []

    def table(space, shape):
        """Each of ``shape``'s cells' nodes' places in ``space`` (lexicographic)."""
        per_cell = len(_lattice(shape, space.order))
        if len(present) > 1:
            raise NotImplementedError("a curved or high-order cell grid of several shapes")
        return _h1_indices(data, space, per_cell)[cells[shape]]

    def attribute(name, width, shape_attribute=False):
        out = vtkCellAttribute()
        out.Initialize(name, _space_name(width), width)
        if shape_attribute:
            grid.SetShapeAttribute(out)
        else:
            grid.AddCellAttribute(out)
        return out

    def shared(out, field, group, width, offset=None):
        space = field.space
        if space.order == 1:
            values = _host(field.values, width)[:data.num_points]
            if offset is not None:
                values = values + offset
            array = _vtk_array(values, out.GetName().Data(), ids=False)
            grid.GetAttributes(group).AddArray(array)
            for shape in present:
                out.SetCellTypeInfo(CELL_TYPES[shape], group, "HGRAD", "C", 1)
                out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "values", array)
                out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "connectivity", corners[shape])
            return
        equispaced = H1(data, order=space.order, basis="equispaced")
        if space is not equispaced:
            field = project(data, field, equispaced)
        values = _host(field.values, width)
        if offset is not None:
            values = values + offset
        group = f"{group} {space.order}"
        array = _vtk_array(values, out.GetName().Data())
        grid.GetAttributes(group).AddArray(array)
        for shape in present:
            ids = _vtk_array(table(equispaced, shape), f"{out.GetName().Data()} conn", ids=True)
            grid.GetAttributes(CELL_TYPES[shape]).AddArray(ids)
            out.SetCellTypeInfo(CELL_TYPES[shape], group, "HGRAD", "A", space.order)
            out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "values", array)
            out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "connectivity", ids)

    def per_cell(out, field, width):
        space = field.space
        target = space.order
        if target > 1 and ((space.basis or "gauss_lobatto") != "equispaced"
                           or space.layout != "blocks"):
            space = L2(data, order=target, basis="equispaced", layout="blocks")
            field = project(data, field, space)
        values = _host(field.values, width)
        def count(shape):
            return shape.NUM_POINTS if target == 1 else len(_lattice(shape, target))

        starts = np.concatenate([[0], np.cumsum([count(by_id[int(t)]) for t in types])])
        for shape in present:
            n = count(shape)
            rows = np.stack([values[starts[c]:starts[c] + n].reshape(-1) for c in cells[shape]])
            array = _vtk_array(rows, out.GetName().Data())
            grid.GetAttributes(CELL_TYPES[shape]).AddArray(array)
            out.SetCellTypeInfo(CELL_TYPES[shape], "", "HGRAD", "C" if target == 1 else "A",
                                target)
            out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "values", array)

    shape = attribute("shape", 3, shape_attribute=True)
    shared(shape, data.geometry, "coordinates", 3, offset=np.asarray(data.origin))
    for name, field in data.fields.items():
        space = field.space
        width = arrays.width_of(field.values) or 1
        if isinstance(space, Values) and space.on == "points":
            space, field = H1(data), field.__class__(H1(data), field.values)
        if (isinstance(space, Values) and space.on == "cells") or isinstance(space, Constant) \
                or (isinstance(space, L2) and not space.varies and space.order == 0):
            out = attribute(name, width)
            values = _host(field.values, width)
            for s in present:
                array = _vtk_array(values[cells[s]], name)
                grid.GetAttributes(CELL_TYPES[s]).AddArray(array)
                out.SetCellTypeInfo(CELL_TYPES[s], "", "constant", "C", 0)
                out.SetArrayForCellTypeAndRole(CELL_TYPES[s], "values", array)
        elif isinstance(space, H1) and space.dofs is None and space.basis in (
                None, "equispaced", "gauss_lobatto") and (space.order == 1 or len(present) == 1):
            shared(attribute(name, width), field, "point-data", width)
        elif isinstance(space, L2) and not space.varies and space.basis in (
                None, "equispaced", "gauss_lobatto"):
            per_cell(attribute(name, width), field, width)
        else:
            left_out.append(f"{name} ({space!r})")
    if left_out:
        warnings.warn("dataset_to_vtk left out fields a cell grid has no basis for here: "
                      + ", ".join(left_out), stacklevel=3)
    return grid
