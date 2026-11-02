# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Solved PDEs followed by Tack's public analysis algorithms."""

import argparse
import itertools
import json
from pathlib import Path

import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import LinearProblem
from interop import _CELLS, _basix_points, import_function, import_mesh, target_points
from mpi4py import MPI

import tack
import tack.dataset as td

_PREFIX = itertools.count()
_JIT = {'cache_dir': '/private/tmp/fenicsx-tack-analysis-jit'}


def domain_for(cell, n):
    if cell in ('triangle', 'quadrilateral'):
        return mesh.create_unit_square(MPI.COMM_SELF, n, n, cell_type=getattr(mesh.CellType, cell))
    return mesh.create_box(MPI.COMM_SELF, [[0, 0, 0], [1, 1, 1]], [n, n, n],
                           cell_type=getattr(mesh.CellType, cell))


def solve(cell, n, p, variant, kind):
    domain = domain_for(cell, n)
    dim = domain.geometry.dim
    vector = kind in ('rotation', 'swirl')
    source_variant = 'equispaced' if variant == 'legendre' else variant
    element = basix.ufl.element('P', cell, p,
        lagrange_variant=getattr(basix.LagrangeVariant, source_variant),
        **({'shape': (3,)} if vector else {}))
    V = fem.functionspace(domain, element)
    x = ufl.SpatialCoordinate(domain)
    if kind == 'quadratic':
        exact = sum((x[i] - 0.5)**2 for i in range(dim))
        boundary = lambda x: np.sum((x[:dim] - 0.5)**2, axis=0)
    elif kind == 'sine':
        exact = ufl.sin(np.pi * x[0])
        for i in range(1, dim):
            exact *= ufl.sin(np.pi * x[i])
        boundary = lambda x: np.prod(np.sin(np.pi * x[:dim]), axis=0)
    else:
        vz = 0.1 if dim == 3 else 0.0
        speed = 1 + (x[0]-0.5)**2 + (x[1]-0.5)**2 if kind == 'swirl' else 1
        exact = ufl.as_vector((-(x[1] - 0.5)*speed, (x[0] - 0.5)*speed, vz))

        def boundary(x):
            speed = 1 + (x[0]-0.5)**2 + (x[1]-0.5)**2 if kind == 'swirl' else 1
            return np.vstack((-(x[1] - 0.5)*speed, (x[0] - 0.5)*speed,
                               np.full(x.shape[1], vz)))
    g = fem.Function(V)
    g.interpolate(boundary)
    if cell == 'prism':
        # DOLFINx's convenience boundary-facet locator rejects mixed facet types.
        # This test uses nodal Lagrange DOFs, so its public geometric locator applies.
        dofs = fem.locate_dofs_geometrical(V, lambda x: np.any(
            np.isclose(x[:dim], 0) | np.isclose(x[:dim], 1), axis=0))
    else:
        facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1,
                                               lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, facets)
    bc = fem.dirichletbc(g, dofs)
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 8})
    a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
    rhs = -ufl.div(ufl.grad(exact))
    L = ufl.inner(rhs, test) * dx
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f'tack_analysis_{next(_PREFIX)}_',
                             petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                             jit_options=_JIT)
    solution = problem.solve()
    solution.x.scatter_forward()
    assert problem.solver.getConvergedReason() > 0
    residual = problem.b.duplicate()
    problem.A.mult(solution.x.petsc_vec, residual)
    residual.axpy(-1, problem.b)
    relative_residual = residual.norm() / max(problem.b.norm(), 1.0)
    residual.destroy()
    assert relative_residual < 1e-10, relative_residual
    projection_metrics = {'linear_systems': 1}
    if variant == 'legendre':
        W = fem.functionspace(domain, basix.ufl.element('P', cell, p,
            lagrange_variant=basix.LagrangeVariant.legendre, discontinuous=True))
        w, z = ufl.TrialFunction(W), ufl.TestFunction(W)
        projection = LinearProblem(ufl.inner(w, z)*dx, ufl.inner(solution, z)*dx,
            petsc_options_prefix=f'tack_projection_{next(_PREFIX)}_',
            petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'}, jit_options=_JIT)
        projected = projection.solve()
        projected.x.scatter_forward()
        assert projection.solver.getConvergedReason() > 0
        residual = projection.b.duplicate()
        projection.A.mult(projected.x.petsc_vec, residual)
        residual.axpy(-1, projection.b)
        projection_residual = residual.norm()/max(projection.b.norm(), 1.0)
        residual.destroy()
        assert projection_residual < 1e-10
        projection_error = np.sqrt(max(float(fem.assemble_scalar(fem.form(
            (projected-solution)**2*dx, jit_options=_JIT))), 0))
        assert projection_error < 2e-10
        projection_metrics = {'linear_systems': 2, 'projection_residual': projection_residual,
                               'projection_l2_error': float(projection_error)}
        solution = projected
    error2 = fem.assemble_scalar(fem.form(ufl.inner(solution-exact, solution-exact) * dx,
                                         jit_options=_JIT))
    error = float(np.sqrt(max(float(error2), 0.0)))
    if kind != 'sine':
        assert error < 2e-10, error
    data = import_mesh(domain)
    field = import_function(data, solution, conversion='lazy')
    data.fields['u'] = field
    return solution, data, {'cells': data.num_cells, 'dofs': len(solution.x.array),
                            'linear_residual': relative_residual, 'pde_l2_error': error,
                            'import_mode': field._dolfinx_import['mode'],
                            'reflected_cells': field._dolfinx_import['reflected_cells'],
                            **projection_metrics}


def source_values(function, points):
    domain = function.function_space.mesh
    points = np.ascontiguousarray(points, dtype=np.float64)
    tree = geometry.bb_tree(domain, domain.topology.dim, padding=1e-12)
    candidates = geometry.compute_collisions_points(tree, points)
    collisions = geometry.compute_colliding_cells(domain, candidates, points)
    cells = np.array([collisions.links(i)[0] if len(collisions.links(i)) else -1
                      for i in range(len(points))], np.int32)
    valid = cells >= 0
    width = function.function_space.dofmap.bs
    values = np.zeros((len(points), width))
    if np.any(valid):
        values[valid] = function.eval(points[valid], cells[valid]).reshape(-1, width)
    return values[:, 0] if width == 1 else values, valid


def check_basic(solution, data, kind):
    domain = solution.function_space.mesh
    dim = domain.geometry.dim
    vector = solution.function_space.dofmap.bs > 1
    # Independent quadrature over the solved FE function.
    dx = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 10})
    source_integral = ([fem.assemble_scalar(fem.form(solution[j] * dx, jit_options=_JIT))
                        for j in range(3)] if vector else
                       fem.assemble_scalar(fem.form(solution * dx, jit_options=_JIT)))
    integrated = td.integrate(data, ['u'])
    np.testing.assert_allclose(integrated.measure, 1.0, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(integrated.values['u'], source_integral, rtol=3e-10, atol=3e-10)
    if kind == 'quadratic':
        np.testing.assert_allclose(integrated.values['u'], dim/12, atol=3e-10)
    if vector:
        np.testing.assert_allclose(integrated.values['u'], [0, 0, 0.1 if dim == 3 else 0], atol=3e-10)

    # Actual public probe, including independent source-side cell location.
    points = np.zeros((97, 3))
    points[:96, :dim] = np.random.default_rng(67).uniform(0.03, 0.97, (96, dim))
    points[-1, :dim] = 0.5
    points[-1, 0] = 1.2
    expected, valid = source_values(solution, points)
    probed = td.probe(data, points, fields=['u'])
    np.testing.assert_array_equal(probed.fields['valid'].values.numpy().astype(bool), valid)
    probe_values = probed.fields['u'].values.numpy()
    np.testing.assert_allclose(probe_values, expected, rtol=2e-9, atol=2e-9)

    # Public gradient filter, compared to a compiled DOLFINx UFL expression.
    cell = data._dolfinx_cell
    pc = _basix_points(cell, target_points(cell, 0, 'equispaced'))
    expression = fem.Expression(ufl.grad(solution), pc, jit_options=_JIT)
    cells = np.arange(data.num_cells, dtype=np.int32)
    source_gradient = expression.eval(domain, cells)
    if vector:
        reference = np.zeros((data.num_cells, 3, 3))
        reference[:, :, :dim] = source_gradient.reshape(data.num_cells, 3, dim)
        reference = reference.reshape(-1, 9)
    else:
        reference = np.zeros((data.num_cells, 3))
        reference[:, :dim] = source_gradient.reshape(-1, dim)
    gradient = td.gradients(data, 'u')
    actual_gradient = gradient.values.numpy()
    np.testing.assert_allclose(actual_gradient, reference, rtol=3e-9, atol=3e-9)
    if kind == 'quadratic':
        expected_points = 2 * (data.positions() - 0.5)
        expected_points[:, dim:] = 0
        point_gradient = td.gradients(data, 'u', at='points').values.numpy()
        np.testing.assert_allclose(point_gradient, expected_points, rtol=3e-9, atol=3e-9)
    return {
        'integral': np.asarray(integrated.values['u']).tolist(),
        'integral_source_error': float(np.max(np.abs(np.asarray(integrated.values['u'])-source_integral))),
        'probe_source_error': float(np.max(np.abs(probe_values-expected))),
        'gradient_source_error': float(np.max(np.abs(actual_gradient-reference))),
    }, gradient


def check_cuts(solution, data):
    dim = solution.function_space.mesh.geometry.dim
    radius = 0.3
    expected_measure = 2*np.pi*radius if dim == 2 else 4*np.pi*radius**2
    contour_errors, slice_errors = [], []
    max_level_error, max_slice_field_error = 0.0, 0.0
    for subdivision in ((4, 8, 16) if dim == 3 else (4, 8)):
        if subdivision < 16:
            surface = td.contour(data, 'u', radius**2, fields=['u'], subdivision=subdivision)
            assert surface.num_cells > 0
            points = surface.positions()
            source, valid = source_values(solution, points)
            assert valid.all()
            level_error = float(np.max(np.abs(source-radius**2)))
            assert level_error < 2e-8, level_error
            max_level_error = max(max_level_error, level_error)
            measure = td.integrate(surface, []).measure
            contour_errors.append(float(abs(measure-expected_measure)))
        if dim == 3:
            cut = td.slice_plane(data, [0.43, 0, 0], [1, 0, 0], fields=['u'],
                                 subdivision=subdivision)
            assert cut.num_cells > 0
            points = cut.positions()
            np.testing.assert_allclose(points[:, 0], 0.43, atol=2e-10)
            expected, valid = source_values(solution, points)
            assert valid.all()
            actual = cut.fields['u'].values.numpy()
            np.testing.assert_allclose(actual, expected, rtol=3e-9, atol=3e-9)
            max_slice_field_error = max(max_slice_field_error, float(np.max(np.abs(actual-expected))))
            integral = td.integrate(cut, ['u'])
            np.testing.assert_allclose(integral.measure, 1.0, atol=2e-10)
            slice_errors.append(abs(integral.values['u']-((0.43-0.5)**2 + 2/12)))
    assert contour_errors[1] < 0.6*contour_errors[0] + 2e-10, contour_errors
    assert contour_errors[1] / expected_measure < 0.025, contour_errors
    if slice_errors:
        assert all(b < a + 2e-10 for a, b in itertools.pairwise(slice_errors)), slice_errors
        assert slice_errors[-1] < 0.2*slice_errors[0] + 2e-10, slice_errors
        assert slice_errors[-1] < 0.004, slice_errors
    return {'contour_subdivisions': [4, 8], 'contour_measure_errors': contour_errors,
            'contour_level_source_error': max_level_error,
            'slice_subdivisions': [4, 8, 16] if dim == 3 else [],
            'slice_integral_errors': slice_errors, 'slice_field_source_error': max_slice_field_error}


def check_flow(data, gradient, kind):
    dim = _CELLS[data._dolfinx_cell][0].DIMENSION
    quantities = td.flow_quantities(data, gradient)
    centers = td.cell_centers(data).values.numpy()
    r2 = np.sum((centers[:, :2]-0.5)**2, axis=1)
    q = (1+r2)*(1+3*r2) if kind == 'swirl' else np.ones(len(centers))
    curl = 2+4*r2 if kind == 'swirl' else np.full(len(centers), 2.0)
    np.testing.assert_allclose(quantities.divergence.values.numpy(), 0.0, atol=3e-9)
    np.testing.assert_allclose(quantities.q_criterion.values.numpy(), q, atol=3e-9)
    vort = quantities.vorticity.values.numpy()
    np.testing.assert_allclose(vort, np.c_[0*curl, 0*curl, curl], atol=3e-9)
    seeds = np.array([[0.65, 0.5, 0.25 if dim == 3 else 0],
                      [0.5, 0.7, 0.4 if dim == 3 else 0]], dtype=float)
    max_error = 0.0
    for neighbors in (False, True):
        lines = td.streamlines(data, 'u', seeds, 0.01, 100, neighbors=neighbors)
        assert lines.num_cells == len(seeds)
        xyz = lines.positions()
        times = lines.fields['time'].values.numpy()
        offsets, indices = (a.numpy() for a in lines.topology.lines())
        seed_ids = lines.fields['seed'].values.numpy()
        for i, seed_id in enumerate(seed_ids):
            ids = indices[offsets[i]:offsets[i+1]]
            t = times[ids]
            np.testing.assert_allclose(t[-1], 1.0, atol=2e-10)
            initial = seeds[seed_id]
            a, b = initial[:2] - 0.5
            angle = t*(1+a*a+b*b) if kind == 'swirl' else t
            expected = np.c_[0.5 + a*np.cos(angle) - b*np.sin(angle),
                              0.5 + a*np.sin(angle) + b*np.cos(angle),
                              initial[2] + (0.1*t if dim == 3 else 0*t)]
            error = float(np.max(np.abs(xyz[ids]-expected)))
            assert error < 2e-8, error
            max_error = max(max_error, error)
    return {'streamline_analytic_error': max_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='all')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    tack.set_default_device('cpu')
    cases = [
        ('quadratic-triangle', 'triangle', 4, 2, 'equispaced', 'quadratic'),
        ('quadratic-quad', 'quadrilateral', 4, 2, 'equispaced', 'quadratic'),
        ('quadratic-tetrahedron', 'tetrahedron', 2, 2, 'equispaced', 'quadratic'),
        ('quadratic-hex', 'hexahedron', 2, 2, 'gll_warped', 'quadratic'),
        ('quadratic-prism', 'prism', 2, 2, 'equispaced', 'quadratic'),
        ('quadratic-bernstein', 'triangle', 4, 2, 'bernstein', 'quadratic'),
        ('quadratic-lazy', 'triangle', 3, 3, 'gll_warped', 'quadratic'),
        *[(f'sine-{n}', 'triangle', n, 2, 'equispaced', 'sine') for n in (4, 8, 16)],
        ('rotation-hex', 'hexahedron', 2, 2, 'equispaced', 'rotation'),
        ('swirl-lazy', 'triangle', 4, 3, 'gll_warped', 'swirl'),
        ('dg-legendre-triangle', 'triangle', 4, 2, 'legendre', 'quadratic'),
        ('dg-legendre-prism', 'prism', 2, 2, 'legendre', 'quadratic'),
    ]
    selected = cases if args.case == 'all' else [case for case in cases if case[0] in args.case.split(',')]
    if not selected:
        parser.error('No matching case')
    report = {'passed': [], 'failed': []}
    for name, cell, n, p, variant, kind in selected:
        print('SOLVE', name, flush=True)
        row, stage = {}, 'solve'
        try:
            solution, data, row = solve(cell, n, p, variant, kind)
            print('ANALYZE', name, row['import_mode'], flush=True)
            stage = 'integrate/probe/gradients'
            basic, gradient = check_basic(solution, data, kind)
            row.update(basic)
            if kind == 'quadratic':
                stage = 'contour/slice'
                row.update(check_cuts(solution, data))
            if kind in ('rotation', 'swirl'):
                stage = 'flow/streamlines'
                row.update(check_flow(data, gradient, kind))
            row.update(case=name, shape=cell, order=p, variant=variant, problem=kind)
            report['passed'].append(row)
            print('PASS', name, json.dumps(row), flush=True)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            report['failed'].append({'case': name, 'stage': stage, 'partial_metrics': row,
                                      'reason': f'{type(exc).__name__}: {exc}'})
            print('FAIL', name, str(exc).splitlines()[0], flush=True)
    sine = [row['pde_l2_error'] for row in report['passed'] if row['problem'] == 'sine']
    if len(sine) == 3:
        assert sine[1] < sine[0]/4 and sine[2] < sine[1]/4, sine
        report['convergence'] = {'poisson_l2_errors': sine,
                                 'observed_orders': np.log2(np.array(sine[:-1])/sine[1:]).tolist()}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print('RESULT',len(report['passed']),'passed,',len(report['failed']),'failed',flush=True)
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
