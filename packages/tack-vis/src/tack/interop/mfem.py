"""MFEM meshes and grid functions as ``tack.dataset`` datasets, through PyMFEM.

``mfem_to_dataset(mesh, fields)`` copies an ``mfem.ser.Mesh`` and named
``GridFunction``s on it. The mesh's elements are the topology, and its
element attributes an i32 ``Constant`` field, ``"attribute"``.

MFEM lists every element's corners in VTK's order, the prism included:
both put the triangle (0, 1, 2) with its normal toward (3, 4, 5), and a
positively oriented MFEM prism has a positive Jacobian as a VTK wedge, as
VTK's own wedges do. (MFEM's VTK writer swaps a prism's corners 1 and 2,
and 4 and 5 -- ``PrismMap`` in mesh/vtk.cpp -- which turns such a prism
inside out against VTK's documented and implemented convention.)

Each grid function becomes a field in the space its collection matches:

- ``H1`` order 1 (``H1_*_P1``): an ``H1`` field, one value per vertex, which
  is how MFEM numbers an order-1 H1 space's DOFs;
- ``L2`` order 0: a ``Constant`` field;
- ``L2`` order 1: an ``L2`` field of each element's values at its corners,
  where MFEM evaluates its own basis (its DOFs sit at interior Gauss points,
  not corners). On tetrahedra, hexahedra and prisms the two spaces are the
  same, so this is exact; on pyramids MFEM's order-1 L2 space is not the
  collapsed trilinear one, and the corners interpolate it;
- ``H1`` order 2: an ``H1`` order-2 field, MFEM's function evaluated at each
  element's quadratic nodes (corners, edge middles, quad-face middles, a
  hexahedron's center). MFEM's order-2 nodes are these same points, so it
  is exact; pyramids, whose MFEM basis is another, are refused.

- ``H1`` from order 3 and ``L2`` from order 2, on quads and hexahedra: Tack's
  space of that order (``tack.dataset.bases``). On nodes Tack has -- MFEM's
  default Gauss-Lobatto for ``H1`` and Gauss-Legendre for ``L2``, Gauss-Lobatto
  and closed uniform for either -- the values are copied, permuted into Tack's
  numbering: both number a cell's nodes lexicographically on the same
  reference cell, and a shared node is every element's, which is checked. On
  any other basis (Bernstein, open uniform) the function is evaluated at Tack's
  Gauss-Lobatto nodes, the same polynomials, so exactly.
- The same on triangles and tetrahedra, whose Tack nodes are equispaced: MFEM's
  closed uniform values are copied (matched by reference coordinates), and any
  other basis -- MFEM's default Gauss-Lobatto-based simplex nodes for ``H1``,
  Gauss-Legendre for ``L2`` -- is evaluated at Tack's equispaced lattice.

A curved mesh gets a geometry of its nodes' order the same way. Vector grid
functions (``vdim > 1``) become fields of vectors, in either ordering.
``mfem_grid_function`` gives a field back as an MFEM ``GridFunction``. PyMFEM
(``pip install mfem``) is imported on first use.
"""

import numpy as np

import tack

__all__ = ["mfem_field", "mfem_grid_function", "mfem_to_dataset"]

#: VTK type of each MFEM geometry (fem/geom.hpp): segment, triangle,
#: square, tetrahedron, cube, prism, pyramid.
_VTK_TYPE = {1: 3, 2: 5, 3: 9, 4: 10, 5: 12, 6: 13, 7: 14}

#: VTK corner j is MFEM vertex _CORNERS[geometry][j]; every shape's are the
#: same, but kept as a table for a shape whose are not.
_CORNERS = {}


def _mfem():
    import mfem.ser

    return mfem.ser


def _corners(geometry, n):
    return _CORNERS.get(geometry, tuple(range(n)))


def _topology(mesh, id_dtype=None):
    from tack.dataset import UnstructuredTopology

    types, rows = [], []
    for e in range(mesh.GetNE()):
        geometry = mesh.GetElementBaseGeometry(e)
        if geometry not in _VTK_TYPE:
            raise NotImplementedError(f"MFEM geometry {geometry} has no linear VTK cell")
        vertices = mesh.GetElementVertices(e)
        types.append(_VTK_TYPE[geometry])
        rows.append([vertices[k] for k in _corners(geometry, len(vertices))])
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
    connectivity = (np.concatenate(rows) if rows else np.zeros(0)).astype(np.int64)
    return UnstructuredTopology(np.array(types, np.uint8), offsets, connectivity,
                                num_points=mesh.GetNV(), id_dtype=id_dtype)


def _collection(fes):
    """``(family, order)`` of a finite element space: ``"H1"`` or ``"L2"``."""
    # PyMFEM returns the collection as its base class, so go by its name:
    # "H1_3D_P1", "L2_3D_P0", "L2_T1_3D_P1" (an L2 basis type), "H1@U_3D_P3"
    # (an H1 basis type), "H1Pos_3D_P3" (Bernstein), ...
    name = fes.FEColl().Name()
    family = name.split("_", 1)[0].split("@", 1)[0]
    family = "H1" if family == "H1Pos" else family
    if family not in ("H1", "L2"):
        raise NotImplementedError(f"MFEM collection {name} has no space here yet")
    return family, fes.GetMaxElementOrder() if hasattr(fes, "GetMaxElementOrder") \
        else fes.GetOrder(0)


#: An MFEM collection's basis type, by its name, as Tack's nodes: H1's default is
#: Gauss-Lobatto ("H1_"), "@U" closed uniform; L2's default Gauss-Legendre
#: ("L2_"), "_T1" Gauss-Lobatto, "_T4" closed uniform. Anything else (Bernstein,
#: open uniform, ...) has no Tack nodes: its function is evaluated at Tack's.
def _nodes_of(fes):
    name = fes.FEColl().Name()
    if name.startswith("H1_"):
        return "gauss_lobatto"
    if name.startswith("H1@U_"):
        return "equispaced"
    if name.startswith("L2_") and not name.startswith("L2_T"):
        return "gauss_legendre"
    if name.startswith("L2_T1_"):
        return "gauss_lobatto"
    if name.startswith("L2_T4_"):
        return "equispaced"
    return None


def _by_dof(fes, values):
    """A grid function's values as ``(ndofs, vdim)``, whatever its ordering."""
    ndofs, vdim = fes.GetNDofs(), fes.GetVDim()
    values = np.asarray(values, dtype=float)
    if fes.GetOrdering() == 0:                    # byNODES: all of component 0 first
        return values.reshape(vdim, ndofs).T
    return values.reshape(ndofs, vdim)


def _field_values(array, dtype):
    """A host array of ``(n,)`` or ``(n, width)`` as a device array."""
    array = np.ascontiguousarray(array)
    if array.ndim == 1 or array.shape[1] == 1:
        array = array.reshape(-1)
        field = tack.empty((array.shape[0],), dtype)
    else:
        field = tack.empty((array.shape[0],), tack.vec(array.shape[1], dtype))
    if array.size:
        tack.copyto(field, array.astype(dtype.numpy_dtype))
    return field


def _corner_values(mesh, gf):
    """Each element's values at its corners, VTK's order, element by element."""
    m = _mfem()
    vdim = gf.FESpace().GetVDim()
    out = []
    vector = m.Vector(vdim)
    for e in range(mesh.GetNE()):
        geometry = mesh.GetElementBaseGeometry(e)
        reference = m.Geometries.GetVertices(geometry)
        for k in _corners(geometry, reference.GetNPoints()):
            ip = reference.IntPoint(k)
            if vdim == 1:
                out.append([gf.GetValue(e, ip)])
            else:
                gf.GetVectorValue(e, ip, vector)
                out.append(np.array(vector.GetDataArray()))
    return np.array(out, dtype=float).reshape(-1, vdim)


@tack.kernel
def _quadratic_layout(cells, u, indices, nodes, width):
    for c in cells:
        e = cells.entity_id(c)
        for k in range(cells.NUM_QUADRATIC):
            indices[e * width + k] = u.dof_index(c, k)
            nodes[e * width + k] = cells.quadratic_node(k)


def _quadratic_values(data, mesh, gf):
    """``gf`` at every value of ``data``'s H1 order-2 space: each element evaluates it
    at its quadratic nodes, whose parametric coordinates are MFEM's reference
    coordinates too (the corners agree, and so the references)."""
    from tack.arrays import ConstantArray
    from tack.dataset import H1, Field, for_each

    m = _mfem()
    space = H1(data, order=2)
    width = 27
    n = data.num_cells
    from tack.dataset.ids import at_least

    index = at_least(data.id_dtype, space.size)
    indices = tack.empty((n * width,), index)
    indices.fill(-1)
    nodes = tack.empty((n * width,), tack.vec(3, tack.f64 if tack.f64 in _supported() else tack.f32))
    for_each(_quadratic_layout, data, "cells", Field(space, ConstantArray(0.0, space.size)),
             indices, nodes, width)
    indices = indices.numpy().reshape(n, width)
    nodes = nodes.numpy().reshape(n, width, 3)
    vdim = gf.FESpace().GetVDim()
    out = np.zeros((space.size, vdim))
    vector = m.Vector(vdim)
    ip = m.IntegrationPoint()
    for e in range(n):
        for k in np.flatnonzero(indices[e] >= 0):
            ip.Set3(*(float(x) for x in nodes[e, k]))
            if vdim == 1:
                out[indices[e, k], 0] = gf.GetValue(e, ip)
            else:
                gf.GetVectorValue(e, ip, vector)
                out[indices[e, k]] = vector.GetDataArray()
    return space, out


@tack.kernel
def _tensor_layout(cells, u, node_of, indices, per_cell):
    # Each cell's node n (lexicographic) where the space keeps it; node_of[n] is its
    # number in the view's cell order (bases.view_order).
    for c in cells:
        e = cells.entity_id(c)
        for n in range(per_cell):
            indices[e * per_cell + n] = u.dof_index(c, node_of[n])


def _h1_indices(data, space, per_cell):
    """``(cells, (p + 1)^d)``: where ``space`` keeps each cell's lexicographic node."""
    from tack.arrays import ConstantArray
    from tack.dataset import Field, bases
    from tack.dataset.ids import at_least

    indices = tack.empty((data.num_cells * per_cell,), at_least(data.id_dtype, space.size))
    field = Field(space, ConstantArray(0.0, space.size))
    for group in data.launch_groups("cells", [field]):
        if group.count:
            node_of = tack.array(bases.view_order(group.shape, space.order))
            _tensor_layout(data.domain_view("cells", group), field.view(group), node_of,
                           indices, per_cell)
    return indices.numpy().reshape(data.num_cells, per_cell)


def _lexicographic(fes, element, nodes, order):
    """MFEM's DOF ``k`` of ``element`` is node ``lex[k]`` in Tack's order (lexicographic,
    the first axis fastest): matched by its reference coordinates, which MFEM's
    and Tack's reference cells share, against the 1D ``nodes``."""
    fe = fes.GetFE(element)
    rule = fe.GetNodes()
    dim = fe.GetDim()
    simplex = _simplex_lattice_of(fe, order)
    if simplex is not None:
        # A triangle's or tetrahedron's nodes: equispaced, matched to Tack's lattice.
        number = {tuple(row): n for n, row in enumerate(simplex.tolist())}
        lex = np.empty(rule.GetNPoints(), np.int64)
        for k in range(rule.GetNPoints()):
            point = rule.IntPoint(k)
            at = np.array([point.x, point.y, point.z][:dim]) * order
            steps = tuple(int(v) for v in np.rint(at))
            if np.abs(at - steps).max() > 1e-9 or steps not in number:
                raise AssertionError("MFEM's simplex nodes are not equispaced")
            lex[k] = number[steps]
        return lex
    q = order + 1
    lex = np.empty(rule.GetNPoints(), np.int64)
    for k in range(rule.GetNPoints()):
        point = rule.IntPoint(k)
        at = [point.x, point.y, point.z][:dim]
        index = [int(np.argmin(np.abs(nodes - x))) for x in at]
        lex[k] = sum(i * q ** axis for axis, i in enumerate(index))
    if sorted(lex) != list(range(q ** dim)):
        raise AssertionError("MFEM's nodes are not the tensor product of Tack's")
    return lex


def _simplex_lattice_of(fe, order):
    """Tack's lattice of ``order`` for an MFEM triangle or tetrahedron element, else
    ``None``: MFEM's reference simplices are Tack's (corners at the origin and the
    unit points)."""
    from tack.dataset import bases, shapes

    shape = {2: shapes.Triangle, 4: shapes.Tetra}.get(fe.GetGeomType())
    return bases.simplex_lattice(shape, order) if shape is not None else None


def _element_dofs(fes, nodes, order):
    """``(ne, nodes a cell)``: each element's MFEM DOF at each of Tack's nodes."""
    m = _mfem()
    ne = fes.GetNE()
    q = fes.GetFE(0).GetDof() if ne else 0
    out = np.empty((ne, q), np.int64)
    lex = {}
    dofs = m.intArray()
    for e in range(ne):
        geometry = fes.GetMesh().GetElementBaseGeometry(e)
        if geometry not in lex:
            lex[geometry] = _lexicographic(fes, e, nodes, order)
        fes.GetElementDofs(e, dofs)
        out[e, lex[geometry]] = dofs.ToList()
    return out


def _check_tensor(data, what):
    """Cells of one family Tack has bases of any order on: tensor shapes, or
    triangles and tetrahedra. Returns whether they are simplices."""
    from tack.dataset import bases

    groups = [g for g in data.topology.groups() if g.count]
    for group in groups:
        if not bases.has_bases(group.shape):
            raise NotImplementedError(f"{what} is on segments, quads, hexahedra, triangles "
                                      f"and tetrahedra here, not {group.shape.__name__} cells")
    families = {bases.simplex_shape(g.shape) for g in groups}
    if len(families) > 1:
        raise NotImplementedError(f"{what} takes simplices or tensor shapes, not both")
    return families.pop() if families else False


def _tensor_values(data, mesh, gf, family, order):
    """``gf``, an H1 or L2 grid function of order ``order`` on quads or hexahedra, as
    ``(space, values)`` in Tack's space of that order. On nodes Tack has
    (``_nodes_of``) its values are copied, permuted into Tack's numbering; on
    any other basis it is evaluated at Tack's Gauss-Lobatto nodes (the same
    polynomials, so exactly)."""
    from tack.dataset import H1, L2, bases

    simplex = _check_tensor(data, f"MFEM {family} order {order}")
    fes = gf.FESpace()
    nodes = _nodes_of(fes)
    if simplex and nodes != "equispaced":
        nodes = None                  # Tack's simplices are equispaced: evaluate there
    space = (H1 if family == "H1" else L2)(
        data, order=order, nodes=nodes or ("equispaced" if simplex else "gauss_lobatto"))
    n = data.num_cells
    dim = data.topology.dimension
    from math import comb

    per_cell = comb(order + dim, dim) if simplex else (order + 1) ** dim
    if family == "H1":
        indices = _h1_indices(data, space, per_cell)
    else:
        indices = space.indices()                        # either layout
    vdim = fes.GetVDim()
    if nodes is not None:
        values = _by_dof(fes, gf.GetDataArray())
        mfem_dofs = _element_dofs(fes, bases.nodes(nodes, order), order)
        out = np.zeros((space.size, vdim))
        out[indices.reshape(-1)] = values[mfem_dofs.reshape(-1)]
        # A shared node is every element's: one value, or the numberings disagree.
        if not np.array_equal(out[indices.reshape(-1)], values[mfem_dofs.reshape(-1)]):
            raise AssertionError("MFEM's and Tack's numberings of a shared node disagree")
        return space, out
    m = _mfem()
    if simplex:
        lattice = bases.lattice_ijk(next(g.shape for g in data.topology.groups() if g.count),
                                    order) / order
        points = np.c_[lattice, np.zeros((len(lattice), 3 - dim))]
    else:
        x = bases.nodes("gauss_lobatto", order)
        q = order + 1
        points = [[x[k % q], x[(k // q) % q] if dim >= 2 else 0.0,
                   x[k // (q * q)] if dim == 3 else 0.0] for k in range(per_cell)]
    out = np.zeros((space.size, vdim))
    vector = m.Vector(vdim)
    ip = m.IntegrationPoint()
    for e in range(n):
        for k in range(per_cell):
            at = [float(v) for v in points[k]]
            ip.Set3(*at)
            if vdim == 1:
                out[indices[e, k], 0] = gf.GetValue(e, ip)
            else:
                gf.GetVectorValue(e, ip, vector)
                out[indices[e, k]] = vector.GetDataArray()
    return space, out


def mfem_grid_function(data, field, mesh, fes=None):
    """``field``, an ``H1`` or ``L2`` field of ``data`` on quads or hexahedra (the dataset
    ``mfem_to_dataset`` made from ``mesh``), as an MFEM ``GridFunction``: on
    ``fes`` if given (its collection's nodes must be the field's), else on a new
    space of the field's family, order and nodes (``H1_FECollection`` or
    ``L2_FECollection`` with the matching basis type), ``byVDIM`` for vectors.
    The values are copied, permuted into MFEM's numbering. Keep the returned
    function's ``fes`` and ``fec`` attributes alive as long as it is used."""
    from tack import arrays
    from tack.dataset import H1, L2, bases

    m = _mfem()
    space = field.space
    if not isinstance(space, (H1, L2)) or space.varies:
        raise NotImplementedError(f"an MFEM grid function of {space!r} is not supported")
    simplex = _check_tensor(data, "mfem_grid_function")
    order = space.order
    family = space.nodes or ("equispaced" if simplex else "gauss_lobatto")
    if isinstance(space, L2) and order <= 1 and not space.tensor:
        raise NotImplementedError("L2 of corner values (order 0 and 1) has no MFEM basis "
                                  "of its own: project it into L2 on Gauss-Legendre nodes")
    dim = data.topology.dimension
    values = arrays.materialize(field.values).numpy().reshape(space.size, -1)
    vdim = values.shape[1]
    fec = None
    if fes is None:
        kind = {"gauss_lobatto": m.BasisType.GaussLobatto,
                "equispaced": m.BasisType.ClosedUniform,
                "gauss_legendre": m.BasisType.GaussLegendre}[family]
        fec = (m.H1_FECollection if isinstance(space, H1) else m.L2_FECollection)(
            order, dim, kind)
        fes = m.FiniteElementSpace(mesh, fec, vdim, m.Ordering.byVDIM)
    if _nodes_of(fes) != family:
        raise ValueError(f"the MFEM space's nodes are not the field's ({family})")
    n = data.num_cells
    from math import comb

    per_cell = comb(order + dim, dim) if simplex else (order + 1) ** dim
    if isinstance(space, H1):
        indices = _h1_indices(data, space, per_cell).reshape(-1)
    else:
        indices = space.indices().reshape(-1)            # either layout
    mfem_dofs = _element_dofs(fes, bases.nodes(family, order), order).reshape(-1)
    by_dof = np.zeros((fes.GetNDofs(), vdim))
    by_dof[mfem_dofs] = values[indices]
    gf = m.GridFunction(fes)
    flat = by_dof.T.reshape(-1) if fes.GetOrdering() == 0 else by_dof.reshape(-1)
    gf.Assign(m.Vector(flat.astype(float)))
    gf.fes, gf.fec = fes, fec
    return gf


def _supported():
    from tack.runtime.dispatch import get_backend

    return get_backend().supported_dtypes


def mfem_field(data, mesh, gf, dtype=tack.f32):
    """``gf``, an MFEM grid function on ``mesh``, as a field of ``data`` (the dataset
    ``mfem_to_dataset`` made from ``mesh``)."""
    from tack.dataset import H1, L2, Constant, Field

    fes = gf.FESpace()
    family, order = _collection(fes)
    if family == "H1" and order == 1:
        if fes.GetNDofs() != mesh.GetNV():
            raise ValueError("an order-1 H1 space has a DOF per vertex")
        return Field(H1(data), _field_values(_by_dof(fes, gf.GetDataArray()), dtype))
    if family == "L2" and order == 0:
        return Field(Constant(data), _field_values(_by_dof(fes, gf.GetDataArray()), dtype))
    if family == "L2" and order == 1:
        return Field(L2(data), _field_values(_corner_values(mesh, gf), dtype))
    if family == "H1" and order == 2 and _nodes_of(fes) is not None:
        space, values = _quadratic_values(data, mesh, gf)
        return Field(space, _field_values(values, dtype))
    if order >= 2:
        space, values = _tensor_values(data, mesh, gf, family, order)
        return Field(space, _field_values(values, dtype))
    raise NotImplementedError(f"MFEM {family} order {order} has no space here yet")


def mfem_to_dataset(mesh, fields=None, dtype=tack.f32, id_dtype=None):
    """A ``tack.dataset.DataSet`` of an MFEM mesh and named grid functions on it.

    The geometry is the vertex positions (z = 0 for a 2D mesh), or for a curved
    mesh an ``H1`` field of positions of its nodes' order (above order 2, on
    quads and hexahedra). Element attributes become ``fields["attribute"]``.
    ``id_dtype`` is the topology's (``tack.dataset.ids``).
    """
    from tack.dataset import Constant, DataSet, Field

    nodes = mesh.GetNodes()
    curved = nodes is not None and _collection(nodes.FESpace())[1] > 1
    order = _collection(nodes.FESpace())[1] if curved else 1
    topology = _topology(mesh, id_dtype)
    if curved:
        positions = np.asarray(mesh.GetVertexArray(), dtype=float).reshape(
            mesh.GetNV(), mesh.SpaceDimension())
    elif nodes is not None:
        # Order-1 nodes are the vertices' positions, and may have moved since
        # the vertex array was set (Mesh.Transform moves the nodes).
        positions = _by_dof(nodes.FESpace(), nodes.GetDataArray())
    else:
        positions = np.asarray(mesh.GetVertexArray(), dtype=float).reshape(
            mesh.GetNV(), mesh.SpaceDimension())
    if positions.shape[1] < 3:
        positions = np.hstack([positions, np.zeros((positions.shape[0],
                                                    3 - positions.shape[1]))])
    data = DataSet(topology, positions, dtype=dtype)
    if curved:
        # The corners first, to have a dataset to number the nodes on; then the
        # curved geometry, MFEM's nodes at those nodes.
        if order == 2 and _nodes_of(nodes.FESpace()) is not None:
            space, values = _quadratic_values(data, mesh, nodes)
        else:
            space, values = _tensor_values(data, mesh, nodes, "H1", order)
        data.geometry = Field(space, _field_values(values, dtype))
    attributes = np.array([mesh.GetAttribute(e) for e in range(mesh.GetNE())], np.int32)
    data.fields["attribute"] = Field(Constant(data), _field_values(attributes, tack.i32))
    for name, gf in (fields or {}).items():
        data.fields[name] = mfem_field(data, mesh, gf, dtype=dtype)
    return data
