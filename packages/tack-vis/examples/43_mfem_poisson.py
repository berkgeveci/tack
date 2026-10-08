"""43 -- A Poisson solution from MFEM, curved and quadratic, through tack.data.

Solves -laplace(u) = 1 with u = 0 on the boundary in PyMFEM (MFEM's first
example), in H1 of order 2 on a curved mesh, brings the mesh and solution
into a tack.data dataset -- the geometry an order-2 H1 field, the solution
another -- and checks them against MFEM before running filters on them.

The mesh is --mesh, any mesh MFEM reads (MFEM's data folder has curved ones:
fichera-q2.mesh, escher-p2.mesh, ...); without it, a Cartesian mesh of
hexahedra bent by a smooth map. Pyramids have no order-2 space here.

Usage:
  pip install mfem "llvmlite==0.46.0"     # numba's newest wants a newer llvmlite
  python examples/43_mfem_poisson.py [--arch cpu|metal|cuda|hip|level_zero]
                                     [--mesh FILE] [--refine N] [--output DIR]

With --output DIR it writes the solution's contour, a slice and the boundary
as VTK files.
"""

import argparse
import os

import numpy as np

import tack

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--arch", default="cpu",
                    choices=["cpu", "metal", "cuda", "hip", "level_zero"])
parser.add_argument("--mesh", help="a mesh file MFEM reads")
parser.add_argument("--refine", type=int, default=1, help="uniform refinements")
parser.add_argument("--output", help="directory to write VTK files into")
args = parser.parse_args()
if args.arch == "level_zero":
    from tack.interop.vtk import init_level_zero
    init_level_zero()
else:
    tack.init(arch=getattr(tack, args.arch))

import mfem.ser as mfem

import tack.data as td
from tack.data import algorithms as alg
from tack.interop.mfem import mfem_to_dataset


class Bend(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        return [x[0] + 0.15 * np.sin(1.3 * x[1]), x[1] + 0.1 * x[0] * x[2],
                x[2] + 0.12 * np.cos(x[0])]


if args.mesh:
    mesh = mfem.Mesh(args.mesh, 1, 1)
else:
    mesh = mfem.Mesh.MakeCartesian3D(4, 3, 3, mfem.Element.HEXAHEDRON, 3.0, 1.5, 2.0)
    mesh.SetCurvature(2)
    mesh.Transform(Bend(3))
for _ in range(args.refine):
    mesh.UniformRefinement()
if mesh.GetNodes() is None:
    mesh.SetCurvature(2)

# MFEM's example 1: -laplace(u) = 1, u = 0 on the boundary, in H1 of order 2.
fec = mfem.H1_FECollection(2, mesh.Dimension())
fes = mfem.FiniteElementSpace(mesh, fec)
boundary = mfem.intArray([1] * mesh.bdr_attributes.Max())
fixed = mfem.intArray()
fes.GetEssentialTrueDofs(boundary, fixed)
one = mfem.ConstantCoefficient(1.0)
b = mfem.LinearForm(fes)
b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
b.Assemble()
u = mfem.GridFunction(fes)
u.Assign(0.0)
a = mfem.BilinearForm(fes)
a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
a.Assemble()
A, B, X = mfem.OperatorPtr(), mfem.Vector(), mfem.Vector()
a.FormLinearSystem(fixed, u, b, A, X, B)
matrix = mfem.OperatorHandle2SparseMatrix(A)
mfem.PCG(matrix, mfem.GSSmoother(matrix), B, X, 0, 1000, 1e-12, 0.0)
a.RecoverFEMSolution(X, b, u)
peak = float(np.max(u.GetDataArray()))
print(f"MFEM: {mesh.GetNE()} elements, {fes.GetNDofs()} order-2 DOFs, u up to {peak:.4f}")

data = mfem_to_dataset(mesh, {"u": u})
field = data.fields["u"]
print(f"tack: {data.num_cells} cells, {data.topology.faces().num_faces} faces, "
      f"{data.topology.edges().num_edges} edges; geometry {data.geometry.space!r}, "
      f"u {field.space!r} with {field.space.size} values")

# The same function: values and gradients at element centers, against MFEM.
values = alg.values_at_centers(data, field).values.to_numpy()
gradients = alg.gradients(data, field).values.to_numpy(vectors=True)
value_error = gradient_error = 0.0
g = mfem.Vector(3)
for e in range(mesh.GetNE()):
    center = mfem.Geometries.GetCenter(mesh.GetElementBaseGeometry(e))
    T = mesh.GetElementTransformation(e)
    T.SetIntPoint(center)
    u.GetGradient(T, g)
    # In double: a float32 less a Python float would round MFEM's value first.
    value_error = max(value_error, abs(float(values[e]) - u.GetValue(e, center)))
    gradient_error = max(gradient_error, np.abs(gradients[e].astype(np.float64)
                                                - g.GetDataArray()).max())
print(f"against MFEM at element centers: values within {value_error:.1e}, "
      f"gradients within {gradient_error:.1e}")

# Filters on a curved, quadratic dataset. They read each cell's corners: the
# contour and slice are of the field and geometry at the corners, linearized.
iso = td.contour(data, "u", 0.5 * peak)
cut = td.slice_plane(data, data.positions().mean(axis=0), (1.0, 0.3, 0.2))
outside = td.external_faces(data)
flux = alg.divergence(data, alg.upwind_flux(data, field, (1.0, 0.0, 0.0)))
print(f"contour of u at half its peak: {iso.num_cells} triangles; slice: {cut.num_cells}; "
      f"boundary: {outside.num_cells} faces; net upwind outflow summed over cells "
      f"{flux.values.to_numpy().sum():.2e}")

if args.output:
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

    from tack.interop.vtk import dataset_to_vtk

    os.makedirs(args.output, exist_ok=True)
    for name, surface in (("contour", iso), ("slice", cut), ("boundary", outside)):
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(os.path.join(args.output, f"poisson_{name}.vtu"))
        writer.SetInputData(dataset_to_vtk(surface))
        writer.Write()
        print(f"wrote {writer.GetFileName()}")
