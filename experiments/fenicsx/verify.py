# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Live DOLFINx comparisons for the experimental importer; no VTK required."""

import argparse
import json
from collections import Counter
from pathlib import Path

import basix
import basix.ufl
import numpy as np
from dolfinx import fem, mesh
from interop import _CELLS, _basix_points, import_function, import_mesh, random_points
from mpi4py import MPI

import tack
import tack.dataset as td
from tack.arrays import ExtractComponent


def make_mesh(cell):
    if cell == 'interval':
        return mesh.create_unit_interval(MPI.COMM_SELF, 3)
    if cell in ('triangle', 'quadrilateral'):
        return mesh.create_unit_square(MPI.COMM_SELF, 2, 2, cell_type=getattr(mesh.CellType, cell))
    if cell in ('tetrahedron', 'hexahedron', 'prism'):
        return mesh.create_box(MPI.COMM_SELF, [[0, 0, 0], [1, 1, 1]], [1, 1, 1],
                               cell_type=getattr(mesh.CellType, cell))
    # Two pyramids share their base. The second reverses the reference axes.
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                  [0.5, 0.5, 1], [0.5, 0.5, -1]], float)
    coordinate = basix.ufl.element('P', 'pyramid', 1, shape=(3,))
    cells = np.array([[0, 1, 2, 3, 4], [0, 2, 1, 3, 5]], np.int64)
    return mesh.create_mesh(MPI.COMM_SELF, cells, coordinate, x)


@tack.kernel
def evaluate(cells, field, pcs, per_cell, positions, values):
    for c in cells:
        for j in range(per_cell):
            k = cells.entity_id(c) * per_cell + j
            positions[k] = cells.position(c, pcs[k])
            values[k] = field.value(c, pcs[k])


@tack.kernel
def gradients(cells, field, pcs, per_cell, out):
    for c in cells:
        for j in range(per_cell):
            k = cells.entity_id(c) * per_cell + j
            pc = pcs[k]
            jac = cells.geometry_jacobian(c, pc)
            d = field.parametric_gradient(c, pc)
            g = tack.vec3(0.0, 0.0, 0.0)
            if cells.DIMENSION == 3:
                g = jac.inverse().transpose() @ d
            else:
                a = tack.vec3(jac[0, 0], jac[1, 0], jac[2, 0])
                if cells.DIMENSION == 1:
                    g = a * (d[0] / a.dot(a))
                else:
                    b = tack.vec3(jac[0, 1], jac[1, 1], jac[2, 1])
                    aa = a.dot(a)
                    ab = a.dot(b)
                    bb = b.dot(b)
                    g = ((bb*d[0] - ab*d[1])*a + (aa*d[1] - ab*d[0])*b) / (aa*bb - ab*ab)
            out[k] = g


def check(data, function, field, *, derivatives=True):
    per_cell = 7
    cell = data._dolfinx_cell
    dim = _CELLS[cell][0].DIMENSION
    reference_points = random_points(cell, per_cell)
    pc = np.zeros((data.num_cells * per_cell, 3))
    pc[:, :dim] = np.tile(reference_points, (data.num_cells, 1))
    pcs = tack.empty((len(pc),), tack.vec(3, tack.f64))
    tack.copyto(pcs, pc)
    xyz = tack.empty_like(pcs)
    block_size = function.function_space.dofmap.bs
    value_type = tack.f64 if block_size == 1 else tack.vec(block_size, tack.f64)
    values = tack.empty((len(pc),), value_type)
    td.for_each(evaluate, data, 'cells', field, pcs, per_cell, xyz, values)
    cell_ids = np.repeat(np.arange(data.num_cells, dtype=np.int32), per_cell)
    expected = function.eval(xyz.numpy(), cell_ids)
    if block_size == 1:
        expected = expected.reshape(-1)
    actual = values.numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-10)
    value_error = float(np.max(np.abs(actual - expected)))
    gradient_error = 0.0
    if derivatives:
        V = function.function_space
        element = V.element.basix_element
        X = _basix_points(cell, reference_points)
        tab = element.tabulate(1, X)
        geometry = basix.create_element(basix.ElementFamily.P, getattr(basix.CellType, cell),
                                        1, basix.LagrangeVariant.equispaced)
        geom_tab = geometry.tabulate(1, X)
        derivative_ids = [basix.index(*(int(i == axis) for i in range(dim))) for axis in range(dim)]
        coeffs = function.x.array.reshape(-1, block_size)[V.dofmap.list]
        coord = V.mesh.geometry.x[V.mesh.geometry.dofmaps[0]]
        expected_gradients = []
        for c in range(data.num_cells):
            jac = np.stack([geom_tab[i, :, :, 0] @ coord[c] for i in derivative_ids], axis=-1)
            d = np.stack([tab[i, :, :, 0] @ coeffs[c] for i in derivative_ids], axis=-1)
            expected_gradients.append(np.einsum('qdi,qbd->qbi', np.linalg.pinv(jac), d))
        expected_gradients = np.concatenate(expected_gradients)
        for component in range(block_size):
            scalar = field if block_size == 1 else td.Field(field.space,
                                                             ExtractComponent(field.values, component))
            result = tack.empty_like(pcs)
            td.for_each(gradients, data, 'cells', scalar, pcs, per_cell, result)
            actual_gradient = result.numpy()
            reference_gradient = expected_gradients[:, component]
            np.testing.assert_allclose(actual_gradient, reference_gradient, rtol=2e-9, atol=2e-9)
            gradient_error = max(gradient_error, float(np.max(np.abs(actual_gradient-reference_gradient))))
    return value_error, gradient_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shapes', default=','.join(_CELLS))
    parser.add_argument('--variants', default='equispaced,gll_warped,gll_isaac,gll_centroid,'
                        'chebyshev_warped,gl_warped,legendre,bernstein')
    parser.add_argument('--orders', default='0,1,2,3,4')
    parser.add_argument('--components', default='1,2')
    parser.add_argument('--no-derivatives', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    tack.set_default_device('cpu')
    rng = np.random.default_rng(23)
    report = {'passed': [], 'source_unsupported': [], 'target_unsupported': [], 'failed': []}
    for cell in args.shapes.split(','):
        m = make_mesh(cell)
        data = import_mesh(m)
        assert np.shares_memory(np.from_dlpack(data.geometry.values), m.geometry.x)
        for variant in args.variants.split(','):
            for discontinuous in (False, True):
                for p in map(int, args.orders.split(',')):
                    if p == 0 and not discontinuous:
                        continue
                    for block_size in map(int, args.components.split(',')):
                        case = {'shape': cell, 'variant': variant, 'order': p,
                                'space': 'DG' if discontinuous else 'H1', 'components': block_size}
                        label = ' '.join(str(v) for v in case.values())
                        stage = 'basix'
                        try:
                            element = basix.ufl.element('P', cell, p,
                                lagrange_variant=getattr(basix.LagrangeVariant, variant),
                                discontinuous=discontinuous,
                                **({'shape': (block_size,)} if block_size > 1 else {}))
                            stage = 'dolfinx'
                            V = fem.functionspace(m, element)
                        except (RuntimeError, ValueError, NotImplementedError) as exc:
                            report['source_unsupported'].append(dict(case, stage=stage, reason=str(exc)))
                            continue
                        u = fem.Function(V)
                        u.x.array[:] = rng.normal(size=len(u.x.array))
                        try:
                            if cell == 'pyramid' and (p > 1 or (p == 1 and V.element.basix_element.dim != 5)):
                                try:
                                    import_function(data, u, conversion='lazy')
                                except NotImplementedError as exc:
                                    report['target_unsupported'].append(dict(case, reason=str(exc)))
                                    continue
                                raise AssertionError('Higher-order pyramid was not refused')
                            field = import_function(data, u, conversion='lazy')
                            if field._dolfinx_import['mode'] == 'lazy':
                                try:
                                    import_function(data, u)
                                except NotImplementedError:
                                    pass
                                else:
                                    raise AssertionError('Basis conversion was not explicit')
                            error, gradient_error = check(data, u, field, derivatives=not args.no_derivatives)
                            u.x.array[:] *= 2
                            update_error, update_gradient = check(data, u, field,
                                                                   derivatives=not args.no_derivatives)
                            row = dict(case, **field._dolfinx_import,
                                       value_error=max(error, update_error),
                                       gradient_error=max(gradient_error, update_gradient))
                            report['passed'].append(row)
                            print('PASS', label, row['mode'],
                                  f'value={row["value_error"]:.2g} grad={row["gradient_error"]:.2g}', flush=True)
                        except Exception as exc:
                            report['failed'].append(dict(case, reason=f'{type(exc).__name__}: {exc}'))
                            print('FAIL', label, type(exc).__name__, str(exc).splitlines()[0], flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('RESULT', {key: len(value) for key, value in report.items()}, flush=True)
    print('MODES', dict(Counter(row['mode'] for row in report['passed'])), flush=True)
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
