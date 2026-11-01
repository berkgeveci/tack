# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Exploratory H1/L2 import on serial, straight DOLFINx quadrilateral meshes.

Coefficient and geometry storage are borrowed; integer addressing arrays are
allocated. This is a proof of concept, not a public adapter or general FE import.
"""

import basix
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

import tack
import tack.dataset as td
from tack.arrays import Permutation
from tack.dataset import bases, shapes
from tack.dataset.topology import uniform_cells


def import_mesh(m):
    if m.comm.size != 1:
        raise NotImplementedError('Prototype is serial')
    rows = m.geometry.dofmaps[0]
    if m.topology.cell_type != mesh.CellType.quadrilateral or rows.shape[1] != 4:
        raise NotImplementedError('Prototype takes straight quadrilateral geometry')
    connectivity = np.ascontiguousarray(rows[:, [0, 1, 3, 2]])
    types, offsets = uniform_cells(shapes.QUAD, len(rows), tack.i32)
    topology = td.UnstructuredTopology(types, offsets, tack.array(connectivity),
                                       num_points=len(m.geometry.x))
    xyz = tack.array_from_ptr(m.geometry.x, (len(m.geometry.x),),
                              tack.vec(3, tack.f64), device='cpu')
    data = td.DataSet(topology, xyz, dtype=tack.f64)
    data._dolfinx_owner = m
    return data


def import_function(data, function):
    V = function.function_space
    e = V.element.basix_element
    if V.mesh is not data._dolfinx_owner:
        raise ValueError('Function and dataset must use the same source mesh')
    if e.family != basix.ElementFamily.P:
        raise NotImplementedError('Prototype takes Lagrange elements')
    if e.value_shape:
        raise NotImplementedError('Prototype takes scalar or blocked Lagrange elements')
    p = e.degree
    block_size = V.dofmap.bs
    if len(function.x.array) % block_size:
        raise ValueError('Incomplete coefficient block')
    if function.x.array.dtype != np.float64:
        raise NotImplementedError('Prototype takes real float64 coefficients')
    if block_size == 1:
        raw_values = tack.from_dlpack(function.x.array, device='cpu')
    else:
        raw_values = tack.array_from_ptr(function.x.array,
                                         (len(function.x.array) // block_size,),
                                         tack.vec(block_size, tack.f64), device='cpu')
    assert np.shares_memory(np.from_dlpack(raw_values), function.x.array)
    family = 'gauss_lobatto'
    nodes = bases.nodes(family, p) if p else np.array([0.5])
    target_points = np.array([(r, s) for s in nodes for r in nodes])
    if e.discontinuous and p == 1:
        target_points = target_points[[0, 1, 3, 2]]
    if len(e.points) != len(target_points):
        raise NotImplementedError('Element has no matching nodal basis')
    distances = np.linalg.norm(target_points[:, None, :] - e.points[None, :, :], axis=2)
    local_permutation = distances.argmin(axis=1)
    if np.max(distances[np.arange(len(target_points)), local_permutation]) >= 1e-12:
        raise NotImplementedError('Source nodes differ from Tack Gauss-Lobatto nodes')
    assert len(np.unique(local_permutation)) == len(target_points)
    grid = V.mesh.geometry.dofmaps[0]
    source_table = V.dofmap.list
    if e.discontinuous:
        indices = np.ascontiguousarray(source_table[:, local_permutation].reshape(-1))
        space = td.L2(data, order=p, basis=family, layout='blocks')
        field = td.Field(space, Permutation(tack.array(indices), raw_values))
        field._dolfinx_owner = function
        return field
    n = len(function.x.array) // block_size
    foreign_to_tack = np.full(n, -1, dtype=np.int32)
    for j in range(4):
        local = e.entity_dofs[0][j][0]
        for source, point in zip(source_table[:, local], grid[:, j]):
            assert foreign_to_tack[source] in (-1, point)
            foreign_to_tack[source] = point
    assert len(np.unique(foreign_to_tack[foreign_to_tack >= 0])) == data.num_points
    remaining = np.flatnonzero(foreign_to_tack < 0)
    foreign_to_tack[remaining] = np.arange(data.num_points, n, dtype=np.int32)
    tack_to_foreign = np.argsort(foreign_to_tack).astype(np.int32)

    table = np.ascontiguousarray(foreign_to_tack[source_table[:, local_permutation]])
    space = td.H1(data, order=p, basis=family,
                   **({'dofs': table} if p > 1 else {}))
    values = Permutation(tack.array(tack_to_foreign), raw_values)
    field = td.Field(space, values)
    field._dolfinx_owner = function
    return field


@tack.kernel
def evaluate(cells, field, pcs, per_cell, positions, values):
    for c in cells:
        e = cells.entity_id(c)
        for j in range(per_cell):
            k = e * per_cell + j
            positions[k] = cells.position(c, pcs[k])
            values[k] = field.value(c, pcs[k])


def check(data, function, field):
    count = 7
    pc = np.zeros((data.num_cells * count, 3))
    pc[:, :2] = np.random.default_rng(12).uniform(0.05, 0.95, (len(pc), 2))
    pcs = tack.empty((len(pc),), tack.vec(3, tack.f64))
    tack.copyto(pcs, pc)
    xyz = tack.empty_like(pcs)
    block_size = function.function_space.dofmap.bs
    value_type = tack.f64 if block_size == 1 else tack.vec(block_size, tack.f64)
    values = tack.empty((len(pc),), value_type)
    td.for_each(evaluate, data, 'cells', field, pcs, count, xyz, values)
    cell_ids = np.repeat(np.arange(data.num_cells, dtype=np.int32), count)
    reference = function.eval(xyz.numpy(), cell_ids)
    if function.function_space.dofmap.bs == 1:
        reference = reference.reshape(-1)
    error = np.max(np.abs(values.numpy() - reference))
    np.testing.assert_allclose(values.numpy(), reference, rtol=2e-12, atol=2e-12)
    return error


if __name__ == '__main__':
    tack.set_default_device('cpu')
    m = mesh.create_unit_square(MPI.COMM_SELF, 2, 2, cell_type=mesh.CellType.quadrilateral)
    data = import_mesh(m)
    assert np.shares_memory(np.from_dlpack(data.geometry.values), m.geometry.x)
    rng = np.random.default_rng(23)
    for family, orders in (('Lagrange', (1, 2, 3, 4)), ('DG', (0, 1, 2, 3))):
        for p in orders:
            for block_size in (1, 2):
                element = (family, p) if block_size == 1 else (family, p, (block_size,))
                V = fem.functionspace(m, element)
                u = fem.Function(V)
                # Random coefficients exercise every basis mode, not just quadratics.
                u.x.array[:] = rng.normal(size=len(u.x.array))
                field = import_function(data, u)
                error = check(data, u, field)
                # After completed reads, mutate the source vector to verify live aliasing.
                u.x.array[:] *= 2
                function_error = check(data, u, field)
                print(f'{family} p={p}, components={block_size}: '
                      f'{len(u.x.array)} shared coefficients; '
                      f'max error {max(error, function_error):.3g}; '
                      'live source updates: PASS', flush=True)
    print('Geometry and coefficient arrays shared; integer maps allocated.', flush=True)
