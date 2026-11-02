# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Experimental, serial CPU import of Basix P elements into Tack.

Directly compatible bases borrow coefficients through permutations. The explicit
``conversion='lazy'`` option also borrows coefficients, evaluating a change of
basis on every read. It is an experiment, not a public adapter or fast path.
"""

from functools import lru_cache
from math import comb

import basix
import numpy as np

import tack
import tack.dataset as td
from tack.arrays import ConstantArray, Permutation, _Implicit, _metadata
from tack.dataset import bases, shapes
from tack.dataset.topology import uniform_cells

_CELLS = {
    'interval': (shapes.Line, [0, 1]),
    'triangle': (shapes.Triangle, [0, 1, 2]),
    'quadrilateral': (shapes.Quad, [0, 1, 3, 2]),
    'tetrahedron': (shapes.Tetra, [0, 1, 2, 3]),
    'hexahedron': (shapes.Hexahedron, [0, 1, 3, 2, 4, 5, 7, 6]),
    'prism': (shapes.Wedge, [0, 1, 2, 3, 4, 5]),
    'pyramid': (shapes.Pyramid, [0, 1, 3, 2, 4]),
}


def _vertices(cell):
    out = basix.geometry(getattr(basix.CellType, cell))[_CELLS[cell][1]].copy()
    if cell == 'pyramid':
        out[-1] = [0.5, 0.5, 1.0]  # Tack's collapsed coordinates
    return out


def _basix_points(cell, points):
    out = np.asarray(points).copy()
    if cell == 'pyramid':
        out[:, :2] *= 1.0 - out[:, 2:3]
    return out


def random_points(cell, count, seed=12):
    dim = _CELLS[cell][0].DIMENSION
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.08, 0.92, (count, dim))
    if cell in ('triangle', 'tetrahedron', 'prism'):
        d = 2 if cell == 'prism' else dim
        weights = rng.exponential(size=(count, d + 1))
        points[:, :d] = weights[:, 1:] / weights.sum(axis=1, keepdims=True)
    return points


def target_points(cell, p, family):
    shape = _CELLS[cell][0]
    if p == 0:
        if cell in ('triangle', 'prism'):
            return np.array([[1/3, 1/3] + ([0.5] if cell == 'prism' else [])])
        if cell == 'tetrahedron':
            return np.array([[0.25, 0.25, 0.25]])
        if cell == 'pyramid':
            return np.array([[0.4, 0.4, 0.2]])  # Tack's current parametric center
        return np.full((1, shape.DIMENSION), 0.5)
    if p == 1 and family != 'gauss_legendre':
        return _vertices(cell)
    if not bases.has_bases(shape):
        raise NotImplementedError(f'Tack has no order-{p} basis for {cell}')
    lattice = bases.lattice_ijk(shape, p)
    if bases.simplex_shape(shape) or cell == 'prism':
        return lattice / p
    return bases.nodes(family, p)[lattice]


def target_basis(cell, p, family, points):
    shape = _CELLS[cell][0]
    if p == 0:
        return np.ones((len(points), 1))
    if cell == 'pyramid':
        r, s, t = points.T
        return np.array([(1-r)*(1-s)*(1-t), r*(1-s)*(1-t),
                         r*s*(1-t), (1-r)*s*(1-t), t]).T
    if bases.simplex_shape(shape):
        out = bases.simplex_basis(shape, p, points)
        if family == 'bernstein':
            out = np.linalg.solve(bases.simplex_bernstein(shape, p).T, out.T).T
    elif cell == 'prism':
        out = bases.wedge_basis(p, points)
        if family == 'bernstein':
            out = np.linalg.solve(bases.wedge_bernstein(p).T, out.T).T
    else:
        lattice = bases.lattice_ijk(shape, p)
        nodes = bases.nodes(family, p)
        out = np.ones((len(points), len(lattice)))
        for axis in range(shape.DIMENSION):
            one = np.ones((len(points), p + 1))
            for j in range(p + 1):
                if family == 'bernstein':
                    one[:, j] = comb(p, j) * points[:, axis]**j * (1-points[:, axis])**(p-j)
                else:
                    for k in range(p + 1):
                        if k != j:
                            one[:, j] *= (points[:, axis] - nodes[k]) / (nodes[j] - nodes[k])
            out *= one[:, lattice[:, axis]]
    if p == 1 and family != 'gauss_legendre':
        out = out[:, bases.dof_tables(shape, p)[4]]
    return out


@lru_cache(maxsize=None)
def _basis_plan(cell, p, variant, discontinuous):
    e = basix.create_element(basix.ElementFamily.P, getattr(basix.CellType, cell), p,
                             getattr(basix.LagrangeVariant, variant),
                             discontinuous=discontinuous)
    shape = _CELLS[cell][0]
    if cell == 'pyramid' and p > 1:
        raise NotImplementedError('Tack currently supports pyramid geometry/fields at order 1')
    candidates = ['bernstein'] if variant == 'bernstein' and p > 1 else []
    candidates += (['equispaced'] if bases.simplex_shape(shape) or cell == 'prism'
                   else ['gauss_lobatto', 'equispaced'])
    if discontinuous and bases.tensor_shape(shape):
        candidates.append('gauss_legendre')
    for family in dict.fromkeys(candidates):
        nodes = target_points(cell, p, family)
        samples = np.r_[nodes, random_points(cell, 19)]
        target = target_basis(cell, p, family, samples)
        source = e.tabulate(0, _basix_points(cell, samples))[0, :, :, 0]
        if target.shape[1] != e.dim:
            continue
        distances = np.max(np.abs(target[:, :, None] - source[:, None, :]), axis=0)
        permutation = distances.argmin(axis=1)
        if (len(np.unique(permutation)) == len(permutation)
                and np.max(distances[np.arange(len(permutation)), permutation]) < 2e-11):
            return family, permutation, None
    # Exact re-expression in a supported nodal space, checked away from its nodes.
    family = 'equispaced'
    nodes = target_points(cell, p, family)
    if len(nodes) != e.dim:
        raise NotImplementedError('Source and target polynomial spaces have different dimensions')
    matrix = e.tabulate(0, _basix_points(cell, nodes))[0, :, :, 0]
    samples = random_points(cell, 31, seed=91)
    reconstructed = target_basis(cell, p, family, samples) @ matrix
    expected = e.tabulate(0, _basix_points(cell, samples))[0, :, :, 0]
    if np.max(np.abs(reconstructed - expected)) > 2e-10:
        raise NotImplementedError('Source polynomial space is not represented exactly')
    return family, None, matrix


def import_mesh(m):
    if m.comm.size != 1:
        raise NotImplementedError('Prototype is serial')
    cell = m.topology.cell_type.name
    if cell not in _CELLS or len(m.geometry.dofmaps) != 1:
        raise NotImplementedError('Prototype needs one supported cell type')
    shape, corners = _CELLS[cell]
    rows = m.geometry.dofmaps[0]
    coordinate = m.geometry.cmaps[0]
    if coordinate.degree != 1 or rows.shape[1] != shape.NUM_POINTS:
        raise NotImplementedError('Prototype takes order-1 geometry')
    try:
        element = basix.create_element(basix.ElementFamily.P, getattr(basix.CellType, cell),
                                       1, basix.LagrangeVariant(coordinate.variant))
    except RuntimeError as exc:
        raise NotImplementedError('Coordinate basis must interpolate the vertices') from exc
    at_vertices = element.tabulate(0, basix.geometry(element.cell_type))[0, :, :, 0]
    if not np.allclose(at_vertices, np.eye(shape.NUM_POINTS), rtol=0, atol=1e-12):
        raise NotImplementedError('Coordinate basis must interpolate the vertices')
    connectivity = np.ascontiguousarray(rows[:, corners], dtype=np.int32)
    types, offsets = uniform_cells(shape.ID, len(rows), tack.i32)
    topology = td.UnstructuredTopology(types, offsets, tack.array(connectivity),
                                       num_points=len(m.geometry.x))
    if m.geometry.x.dtype != np.float64:
        raise NotImplementedError('Prototype takes float64 geometry')
    xyz = tack.array_from_ptr(m.geometry.x, (len(m.geometry.x),),
                              tack.vec(3, tack.f64), device='cpu')
    data = td.DataSet(topology, xyz, dtype=tack.f64)
    data._dolfinx_owner = m
    data._dolfinx_cell = cell
    return data


@tack.kernel
def _dof_indices(cells, field, count, out, points):
    for c in cells:
        for j in range(count):
            out[cells.entity_id(c) * count + j] = field.dof_index(c, j)
            if cells.entity_id(c) == 0:
                points[j] = field.node_parametric(j)


@tack.device_class
class _BasisMappedValues(_Implicit):
    """Read-only live change of basis. Uses Tack's internal implicit-array hook."""

    def __init__(self, source, dofs, matrix, owners, local_nodes, width):
        self.source = source
        self.indices = tack.array(np.ascontiguousarray(dofs.reshape(-1)))
        self.weights = tack.array(np.ascontiguousarray(matrix.reshape(-1)))
        self.owners = tack.array(owners)
        self.local_nodes = tack.array(local_nodes)
        self.SOURCE_NODES = dofs.shape[1]
        _metadata(self, len(owners), width if width > 1 else None, tack.f64)

    @tack.func
    def get(self, k):
        start = self.owners[k] * self.SOURCE_NODES
        row = self.local_nodes[k] * self.SOURCE_NODES
        value = self.source[self.indices[start]] * self.weights[row]
        for j in range(1, self.SOURCE_NODES):
            value += self.source[self.indices[start + j]] * self.weights[row + j]
        return value


def _lazy_values(data, space, source, source_table, matrix, block_size, nodes):
    placeholder = td.Field(space, ConstantArray(0.0, space.size, dtype=tack.f64))
    nlocal = matrix.shape[0]
    table = tack.empty((data.num_cells * nlocal,), tack.i32)
    native_points = tack.empty((nlocal,), tack.vec(3, tack.f64))
    td.for_each(_dof_indices, data, 'cells', placeholder, nlocal, table, native_points)
    # Entity-numbered quadratic H1 uses corner/edge order, while its table
    # representation uses a lattice. Match the actual view's order explicitly.
    points = native_points.numpy()[:, :nodes.shape[1]]
    distances = np.linalg.norm(nodes[:, None, :] - points[None, :, :], axis=2)
    permutation = distances.argmin(axis=1)
    if (len(np.unique(permutation)) != nlocal
            or np.max(distances[np.arange(nlocal), permutation]) > 1e-12):
        raise NotImplementedError('Target view node ordering does not match the conversion')
    indices = table.numpy().reshape(-1, nlocal)[:, permutation].reshape(-1)
    owners = np.full(space.size, -1, np.int32)
    local_nodes = np.full(space.size, -1, np.int32)
    for i, dof in enumerate(indices):
        if owners[dof] < 0:
            owners[dof], local_nodes[dof] = divmod(i, nlocal)
    if np.any(owners < 0):
        raise NotImplementedError('Unreferenced target DOFs need an ownership rule')
    return _BasisMappedValues(source, source_table, matrix, owners, local_nodes, block_size)


def import_function(data, function, *, conversion='error'):
    if conversion not in ('error', 'lazy'):
        raise ValueError("conversion must be 'error' or 'lazy'")
    V = function.function_space
    e = V.element.basix_element
    if V.mesh is not data._dolfinx_owner:
        raise ValueError('Function and dataset must use the same source mesh')
    if e.family != basix.ElementFamily.P or e.value_shape:
        raise NotImplementedError('Prototype takes scalar or blocked Basix P elements')
    if V.element.needs_dof_transformations:
        raise NotImplementedError('Non-permutation DOF orientation transformations need handling')
    p, block_size = e.degree, V.dofmap.bs
    if len(function.x.array) % block_size or function.x.array.dtype != np.float64:
        raise NotImplementedError('Prototype takes complete, real float64 coefficient blocks')
    if block_size == 1:
        raw = tack.from_dlpack(function.x.array, device='cpu')
    else:
        raw = tack.array_from_ptr(function.x.array, (len(function.x.array) // block_size,),
                                   tack.vec(block_size, tack.f64), device='cpu')
    assert np.shares_memory(np.from_dlpack(raw), function.x.array)
    family, permutation, matrix = _basis_plan(data._dolfinx_cell, p,
                                               e.lagrange_variant.name, e.discontinuous)
    kwargs = {'order': p, 'basis': family}
    if e.discontinuous:
        kwargs['layout'] = 'blocks'
    constructor = td.L2 if e.discontinuous else td.H1
    source_table = V.dofmap.list
    if matrix is not None:
        if conversion == 'error':
            raise NotImplementedError('Basis differs; opt in with conversion="lazy"')
        space = constructor(data, **kwargs)
        nodes = target_points(data._dolfinx_cell, p, family)
        values = _lazy_values(data, space, raw, source_table, matrix, block_size, nodes)
    elif e.discontinuous:
        indices = np.ascontiguousarray(source_table[:, permutation].reshape(-1))
        space = constructor(data, **kwargs)
        values = Permutation(tack.array(indices), raw)
    else:
        grid = V.mesh.geometry.dofmaps[0]
        n = len(function.x.array) // block_size
        foreign_to_tack = np.full(n, -1, dtype=np.int32)
        for j in range(len(e.entity_dofs[0])):
            local = e.entity_dofs[0][j][0]
            for source, point in zip(source_table[:, local], grid[:, j]):
                if foreign_to_tack[source] not in (-1, point):
                    raise NotImplementedError('Periodic or constrained vertex numbering')
                foreign_to_tack[source] = point
        if len(np.unique(foreign_to_tack[foreign_to_tack >= 0])) != data.num_points:
            raise NotImplementedError('Unused geometry points or incomplete vertex addressing')
        remaining = np.flatnonzero(foreign_to_tack < 0)
        foreign_to_tack[remaining] = np.arange(data.num_points, n, dtype=np.int32)
        inverse = np.argsort(foreign_to_tack).astype(np.int32)
        if p > 1:
            kwargs['dofs'] = np.ascontiguousarray(foreign_to_tack[source_table[:, permutation]])
        space = constructor(data, **kwargs)
        values = Permutation(tack.array(inverse), raw)
    field = td.Field(space, values)
    field._dolfinx_owner = function
    field._dolfinx_import = {'mode': 'lazy' if matrix is not None else 'direct',
                             'basis': family, 'source_variant': e.lagrange_variant.name}
    return field


if __name__ == '__main__':
    from verify import main

    main()
