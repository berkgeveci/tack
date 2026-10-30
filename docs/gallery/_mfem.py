# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The problems of the MFEM gallery, solved with PyMFEM (Tack's ``interop`` extra):
MFEM's examples (https://mfem.org, ``examples/exN.cpp``; PyMFEM's ``exN.py``),
each one's steps written out on MFEM's meshes as PyMFEM ships them, refined to a
few thousand elements. tack-examples' ``tack_examples/mfem/`` ports run the same
problems and hold Tack to MFEM's evaluation of them.

Each solve returns the ``GridFunction``; the objects it points at are kept on it
(``keep``), since PyMFEM does not.
"""

import math
from pathlib import Path

import numpy as np


def mesh(name, cells=None):
    """MFEM's mesh ``name`` (``"star.mesh"``) as PyMFEM ships it, refined uniformly
    while it has fewer than ``cells`` / 2^dim elements."""
    import mfem
    import mfem.ser as mfem_ser

    path = Path(mfem.__file__).parent / "external" / "ser" / "data" / name
    out = mfem_ser.Mesh(str(path), 1, 1)
    while cells is not None and out.GetNE() * 2 ** out.Dimension() <= cells:
        out.UniformRefinement()
    return out


def _solved(mfem, a, b, x, ess_tdof_list, copy_interior=0):
    """``a x = b`` with ``x``'s essential values, by PCG with a Gauss-Seidel smoother."""
    A, B, X = mfem.OperatorPtr(), mfem.Vector(), mfem.Vector()
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior)
    M = mfem.GSSmoother(A.AsSparseMatrix())
    mfem.PCG(A, M, B, X, 0, 2000, 1e-12, 0.0)
    a.RecoverFEMSolution(X, b, x)


def poisson(mesh, order):
    """ex1: -Delta u = 1, u = 0 on the boundary, in H1 of ``order``."""
    import mfem.ser as mfem

    fec = mfem.H1_FECollection(order, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec)
    ess_tdof_list = mfem.intArray()
    ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
    ess_bdr.Assign(1)
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)
    one = mfem.ConstantCoefficient(1.0)
    b = mfem.LinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    a.Assemble()
    _solved(mfem, a, b, x, ess_tdof_list)
    x.keep = (fec, fespace)
    return x


def elasticity(mesh, order):
    """ex2: the two-material beam, fixed at one end and pulled down at the other; the
    displacement in H1 of ``order``, dim components."""
    import mfem.ser as mfem

    dim = mesh.Dimension()
    fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.FiniteElementSpace(mesh, fec, dim)
    ess_tdof_list = mfem.intArray()
    ess_bdr = mfem.intArray([1] + [0] * (mesh.bdr_attributes.Max() - 1))
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)
    f = mfem.VectorArrayCoefficient(dim)
    for i in range(dim - 1):
        f.Set(i, mfem.ConstantCoefficient(0.0))
    pull_force = mfem.Vector([0] * mesh.bdr_attributes.Max())
    pull_force[1] = -1.0e-2
    f.Set(dim - 1, mfem.PWConstCoefficient(pull_force))
    b = mfem.LinearForm(fespace)
    b.AddBoundaryIntegrator(mfem.VectorBoundaryLFIntegrator(f))
    b.Assemble()
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)
    lamb, mu = mfem.Vector(mesh.attributes.Max()), mfem.Vector(mesh.attributes.Max())
    lamb.Assign(1.0)
    mu.Assign(1.0)
    lamb[0], mu[0] = 50.0, 50.0
    coefficients = (mfem.PWConstCoefficient(lamb), mfem.PWConstCoefficient(mu))
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.ElasticityIntegrator(*coefficients))
    a.Assemble()
    _solved(mfem, a, b, x, ess_tdof_list)
    x.keep = (fec, fespace, f, pull_force, coefficients)
    return x


def adaptive_poisson(mesh, order, max_dofs):
    """ex6: ex1's problem, solved, estimated (Zienkiewicz-Zhu) and refined where the
    error is largest until ``max_dofs``: a nonconforming mesh."""
    import mfem.ser as mfem

    dim, sdim = mesh.Dimension(), mesh.SpaceDimension()
    mesh.EnsureNCMesh()
    fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.FiniteElementSpace(mesh, fec)
    a, b = mfem.BilinearForm(fespace), mfem.LinearForm(fespace)
    one, zero = mfem.ConstantCoefficient(1.0), mfem.ConstantCoefficient(0.0)
    integ = mfem.DiffusionIntegrator(one)
    a.AddDomainIntegrator(integ)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)
    ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
    ess_bdr.Assign(1)
    flux_fespace = mfem.FiniteElementSpace(mesh, fec, sdim)
    estimator = mfem.ZienkiewiczZhuEstimator(integ, x, flux_fespace, own_flux_fes=False)
    estimator.SetAnisotropic()
    refiner = mfem.ThresholdRefiner(estimator)
    refiner.SetTotalErrorFraction(0.7)
    while True:
        a.Assemble()
        b.Assemble()
        ess_tdof_list = mfem.intArray()
        x.ProjectBdrCoefficient(zero, ess_bdr)
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)
        _solved(mfem, a, b, x, ess_tdof_list, 1)
        if fespace.GetTrueVSize() > max_dofs:
            break
        refiner.Apply(mesh)
        if refiner.Stop():
            break
        fespace.Update()
        x.Update()
        a.Update()
        b.Update()
    x.keep = (fec, fespace, flux_fespace, estimator, refiner, a, b, integ, one, zero)
    return x


def sphere(order, refinements=2):
    """ex7's mesh: the cube's quads, refined, their nodes in H1 of ``order`` snapped
    onto the unit sphere."""
    import mfem.ser as mfem

    vertices = [[-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
                [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]]
    elements = [[3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
                [4, 5, 6, 7]]
    out = mfem.Mesh(2, len(vertices), len(elements), 0, 3)
    for v in vertices:
        out.AddVertex(v)
    for k, e in enumerate(elements):
        out.AddQuad(e, k + 1)
    out.FinalizeQuadMesh(1, 1, True)
    fec = mfem.H1_FECollection(order, out.Dimension())
    nodal = mfem.FiniteElementSpace(out, fec, out.SpaceDimension())
    out.SetNodalFESpace(nodal)
    for _ in range(refinements):
        out.UniformRefinement()
    nodes = out.GetNodes()
    fes = nodes.FESpace()
    values = nodes.GetDataArray()
    for i in range(fes.GetNDofs()):
        at = [fes.DofToVDof(i, d) for d in range(3)]
        values[at] /= np.linalg.norm(values[at])
    out.keep = (fec, nodal)
    return out


def laplace_beltrami(mesh, order):
    """ex7: -Delta u + u = 7 xy / |x|^2 on the sphere's surface, in H1 of ``order``."""
    import mfem.ser as mfem

    class Rhs(mfem.PyCoefficient):
        def EvalValue(self, x):
            return 7 * x[0] * x[1] / np.sum(np.asarray(x) ** 2)

    fec = mfem.H1_FECollection(order, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec)
    rhs, one = Rhs(), mfem.ConstantCoefficient(1.0)
    b = mfem.LinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(rhs))
    b.Assemble()
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    a.AddDomainIntegrator(mfem.MassIntegrator(one))
    a.Assemble()
    _solved(mfem, a, b, x, mfem.intArray())
    x.keep = (fec, fespace, rhs, one)
    return x


def advection(mesh, t_final, order=3, dt=0.01):
    """ex9, problem 0: a smooth bump carried by a constant velocity across a periodic
    mesh, in DG of ``order`` (upwind fluxes), stepped by RK4 to ``t_final``."""
    import mfem.ser as mfem

    dim = mesh.Dimension()
    bb_min, bb_max = (np.asarray(v) for v in mesh.GetBoundingBox(order))
    velocity_of = {2: [math.sqrt(2 / 3), math.sqrt(1 / 3)],
                   3: [math.sqrt(3 / 6), math.sqrt(2 / 6), math.sqrt(1 / 6)]}[dim]

    class Velocity(mfem.VectorPyCoefficient):
        def EvalValue(self, x):
            return velocity_of

    class U0(mfem.PyCoefficient):
        def EvalValue(self, x):
            X = 2 * (np.asarray(x) - (bb_min + bb_max) / 2) / (bb_max - bb_min)
            rx, ry, cx, cy, w = 0.45, 0.25, 0.0, -0.2, 10.0
            return (math.erfc(w * (X[0] - cx - rx)) * math.erfc(-w * (X[0] - cx + rx))
                    * math.erfc(w * (X[1] - cy - ry)) * math.erfc(-w * (X[1] - cy + ry))) / 16

    class Inflow(mfem.PyCoefficient):
        def EvalValue(self, x):
            return 0.0

    fec = mfem.DG_FECollection(order, dim, mfem.BasisType.GaussLobatto)
    fes = mfem.FiniteElementSpace(mesh, fec)
    velocity, u0, inflow = Velocity(dim), U0(), Inflow()
    m = mfem.BilinearForm(fes)
    m.AddDomainIntegrator(mfem.MassIntegrator())
    k = mfem.BilinearForm(fes)
    k.AddDomainIntegrator(mfem.ConvectionIntegrator(velocity, -1.0))
    k.AddInteriorFaceIntegrator(mfem.TransposeIntegrator(
        mfem.DGTraceIntegrator(velocity, 1.0, -0.5)))
    k.AddBdrFaceIntegrator(mfem.TransposeIntegrator(
        mfem.DGTraceIntegrator(velocity, 1.0, -0.5)))
    b = mfem.LinearForm(fes)
    b.AddBdrFaceIntegrator(mfem.BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5))
    m.Assemble()
    m.Finalize()
    k.Assemble(0)
    k.Finalize(0)
    b.Assemble()
    M, K = m.SpMat(), k.SpMat()

    class Evolution(mfem.PyTimeDependentOperator):
        # du/dt = M^-1 (K u + b)
        def __init__(self):
            mfem.PyTimeDependentOperator.__init__(self, M.Size())
            self.z = mfem.Vector(M.Size())
            self.prec = mfem.DSmoother()
            self.solver = mfem.CGSolver()
            self.solver.SetPreconditioner(self.prec)
            self.solver.SetOperator(M)
            self.solver.iterative_mode = False
            self.solver.SetRelTol(1e-9)
            self.solver.SetAbsTol(0.0)
            self.solver.SetMaxIter(100)
            self.solver.SetPrintLevel(0)

        def Mult(self, x, y):
            K.Mult(x, self.z)
            self.z += b
            self.solver.Mult(self.z, y)

    u = mfem.GridFunction(fes)
    u.ProjectCoefficient(u0)
    evolution = Evolution()
    ode = mfem.RK4Solver()
    ode.Init(evolution)
    t = 0.0
    while t < t_final - dt / 2:
        t, dt = ode.Step(u, t, dt)
    u.keep = (fec, fes, velocity, u0, inflow, m, k, b, evolution, ode)
    return u


def dg_elasticity(mesh, order):
    """ex17: ex2's beam displaced by -0.2 x along its last axis on two sides, in DG of
    ``order`` with dim components, by the symmetric interior penalty method."""
    import mfem.ser as mfem

    dim = mesh.Dimension()
    alpha, kappa = -1.0, float((order + 1) ** 2)

    class Initial(mfem.VectorPyCoefficient):
        def __init__(self):
            mfem.VectorPyCoefficient.__init__(self, dim)

        def EvalValue(self, x):
            u = [0.0] * dim
            u[-1] = -0.2 * x[0]
            return tuple(u)

    fec = mfem.DG_FECollection(order, dim, mfem.BasisType.GaussLobatto)
    fespace = mfem.FiniteElementSpace(mesh, fec, dim)
    dir_bdr = mfem.intArray(mesh.bdr_attributes.Max())
    dir_bdr.Assign(0)
    dir_bdr[0] = dir_bdr[1] = 1
    x = mfem.GridFunction(fespace)
    initial = Initial()
    x.ProjectCoefficient(initial)
    lamb, mu = mfem.Vector(mesh.attributes.Max()), mfem.Vector(mesh.attributes.Max())
    lamb.Assign(1.0)
    mu.Assign(1.0)
    lamb[0], mu[0] = 50.0, 50.0
    lambda_c, mu_c = mfem.PWConstCoefficient(lamb), mfem.PWConstCoefficient(mu)
    b = mfem.LinearForm(fespace)
    b.AddBdrFaceIntegrator(mfem.DGElasticityDirichletLFIntegrator(
        initial, lambda_c, mu_c, alpha, kappa), dir_bdr)
    b.Assemble()
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.ElasticityIntegrator(lambda_c, mu_c))
    a.AddInteriorFaceIntegrator(mfem.DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa))
    a.AddBdrFaceIntegrator(mfem.DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa), dir_bdr)
    a.Assemble()
    A, B, X = mfem.SparseMatrix(), mfem.Vector(), mfem.Vector()
    a.FormLinearSystem(mfem.intArray(), x, b, A, X, B)
    M = mfem.GSSmoother(A)
    mfem.PCG(A, M, B, X, 0, 5000, 1e-12, 0.0)
    a.RecoverFEMSolution(X, b, x)
    x.keep = (fec, fespace, initial, lambda_c, mu_c, lamb, mu, a, b, dir_bdr)
    return x
