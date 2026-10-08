"""MFEM meshes and grid functions as ``tack.data`` datasets, through PyMFEM.

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
  collapsed trilinear one, and the corners interpolate it.

Vector grid functions (``vdim > 1``) become fields of vectors, in either
ordering. Higher orders and curved meshes are refused until their bases
exist. PyMFEM (``pip install mfem``) is imported on first use.
"""

import numpy as np

import tack

__all__ = ["mfem_field", "mfem_to_dataset"]

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


def _topology(mesh):
    from tack.data import UnstructuredTopology

    types, rows = [], []
    for e in range(mesh.GetNE()):
        geometry = mesh.GetElementBaseGeometry(e)
        if geometry not in _VTK_TYPE:
            raise NotImplementedError(f"MFEM geometry {geometry} has no linear VTK cell")
        vertices = mesh.GetElementVertices(e)
        types.append(_VTK_TYPE[geometry])
        rows.append([vertices[k] for k in _corners(geometry, len(vertices))])
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int32)
    connectivity = (np.concatenate(rows) if rows else np.zeros(0)).astype(np.int32)
    return UnstructuredTopology(np.array(types, np.uint8), offsets, connectivity,
                                num_points=mesh.GetNV())


def _collection(fes):
    """``(family, order)`` of a finite element space: ``"H1"`` or ``"L2"``."""
    # PyMFEM returns the collection as its base class, so go by its name:
    # "H1_3D_P1", "L2_3D_P0", "L2_T1_3D_P1" (an L2 basis type), ...
    name = fes.FEColl().Name()
    family = name.split("_", 1)[0]
    if family not in ("H1", "L2"):
        raise NotImplementedError(f"MFEM collection {name} has no space here yet")
    return family, fes.GetMaxElementOrder() if hasattr(fes, "GetMaxElementOrder") \
        else fes.GetOrder(0)


def _by_dof(fes, values):
    """A grid function's values as ``(ndofs, vdim)``, whatever its ordering."""
    ndofs, vdim = fes.GetNDofs(), fes.GetVDim()
    values = np.asarray(values, dtype=float)
    if fes.GetOrdering() == 0:                    # byNODES: all of component 0 first
        return values.reshape(vdim, ndofs).T
    return values.reshape(ndofs, vdim)


def _field_values(array, dtype):
    """A host array of ``(n,)`` or ``(n, width)`` as a device field."""
    array = np.ascontiguousarray(array)
    if array.ndim == 1 or array.shape[1] == 1:
        array = array.reshape(-1)
        field = tack.field(dtype, shape=(array.shape[0],))
    else:
        field = tack.Vector.field(array.shape[1], dtype, shape=(array.shape[0],))
    if array.size:
        field.from_numpy(array.astype(dtype.numpy_dtype))
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


def mfem_field(data, mesh, gf, dtype=tack.f32):
    """``gf``, an MFEM grid function on ``mesh``, as a field of ``data`` (the dataset
    ``mfem_to_dataset`` made from ``mesh``)."""
    from tack.data import H1, L2, Constant, Field

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
    raise NotImplementedError(f"MFEM {family} order {order} has no space here yet")


def mfem_to_dataset(mesh, fields=None, dtype=tack.f32):
    """A ``tack.data.DataSet`` of an MFEM mesh and named grid functions on it.

    The geometry is the vertex positions (z = 0 for a 2D mesh); a mesh with
    curved nodes of order above one is refused. Element attributes become
    ``fields["attribute"]``.
    """
    from tack.data import Constant, DataSet, Field

    nodes = mesh.GetNodes()
    if nodes is not None and _collection(nodes.FESpace())[1] > 1:
        raise NotImplementedError("curved MFEM meshes (nodes of order above 1) are not "
                                  "supported yet")
    topology = _topology(mesh)
    if nodes is not None:
        # Order-1 nodes are the vertices' positions, and may have moved since
        # the vertex array was set (Mesh.Transform moves the nodes).
        positions = _by_dof(nodes.FESpace(), nodes.GetDataArray())
    else:
        positions = np.asarray(mesh.GetVertexArray(), dtype=float).reshape(mesh.GetNV(), -1)
    if positions.shape[1] < 3:
        positions = np.hstack([positions, np.zeros((positions.shape[0],
                                                    3 - positions.shape[1]))])
    data = DataSet(topology, positions, dtype=dtype)
    attributes = np.array([mesh.GetAttribute(e) for e in range(mesh.GetNE())], np.int32)
    data.fields["attribute"] = Field(Constant(data), _field_values(attributes, tack.i32))
    for name, gf in (fields or {}).items():
        data.fields[name] = mfem_field(data, mesh, gf, dtype=dtype)
    return data
