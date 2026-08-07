"""39 -- VTK interop: hand data between Tack and VTK without copying.

Tack computes, VTK filters, Tack computes again -- and the data never
moves. Both speak DLPack, so an exchange is one call in each direction and
the memory is shared, not copied.

    field = vtk_to_field(vtk_array)   # VTK  -> Tack
    array = field_to_vtk(field)       # Tack -> VTK

A vtkDataArray is tuples x components and a Tack field is n-dimensional,
so the shapes line up on their own: a (n, 3) field is n tuples of 3.
Nothing has to be declared.

The pipeline below is the one that matters in practice -- a Tack kernel
generating a field, a VTK filter consuming it, and a Tack kernel measuring
the result. On CUDA or HIP the same code keeps everything on the device:
DLPack carries the device pointer, and neither side copies to the host.
That path additionally needs VTK built with the Viskores accelerators
(-DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmCore=YES), which is what wraps
device memory as a vtkmDataArray.

Usage:
  python examples/39_vtk_interop.py [--arch cpu|metal|cuda|hip]

Needs VTK with vtkmodules.util.dlpack_support.
"""

import argparse

import numpy as np

import tack
from tack.interop.vtk import field_to_vtk, vtk_to_field
from tack.runtime.dispatch import get_backend

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="cpu",
                    choices=["cpu", "metal", "cuda", "hip", "level_zero"])
args = parser.parse_args()
tack.init(arch=getattr(tack, args.arch))

NX = NY = NZ = 64
N_POINTS = NX * NY * NZ


# ── Step 1: generate the grid with a Tack kernel ─────────────────────
#
# Points as (n, 3) and scalars as (n,) -- the shapes VTK wants, so no
# reshaping happens anywhere below.

@tack.kernel
def gyroid(points, scalars, nx, ny, nz):
    two_pi = 6.283185307179586
    for idx in range(nx * ny * nz):
        i = idx % nx
        j = (idx // nx) % ny
        k = idx // (nx * ny)

        x = float(i) / float(nx - 1)
        y = float(j) / float(ny - 1)
        z = float(k) / float(nz - 1)

        points[idx, 0] = x
        points[idx, 1] = y
        points[idx, 2] = z

        a = x * two_pi
        b = y * two_pi
        c = z * two_pi
        scalars[idx] = (tack.sin(a) * tack.cos(b) +
                        tack.sin(b) * tack.cos(c) +
                        tack.sin(c) * tack.cos(a))


points = tack.field(dtype=tack.f32, shape=(N_POINTS, 3))
scalars = tack.field(dtype=tack.f32, shape=(N_POINTS,))
gyroid(points, scalars, NX, NY, NZ)

print(f"Tack ({get_backend().label}): {NX}x{NY}x{NZ} = {N_POINTS:,} points")
print(f"  points  {points.shape}  -> {points.shape[1]} components")
print(f"  scalars {scalars.shape}")


# ── Step 2: hand them to VTK ─────────────────────────────────────────
#
# No component count to pass: points.shape already says 3.

coord_array = field_to_vtk(points, name="Points")
scalar_array = field_to_vtk(scalars, name="gyroid")

print("\nVTK arrays (shared memory, not copied):")
print(f"  {coord_array.GetClassName()}: "
      f"{coord_array.GetNumberOfTuples():,} x {coord_array.GetNumberOfComponents()}")
print(f"  {scalar_array.GetClassName()}: "
      f"{scalar_array.GetNumberOfTuples():,} x {scalar_array.GetNumberOfComponents()}")

import vtk

vtk_points = vtk.vtkPoints()
vtk_points.SetData(coord_array)

grid = vtk.vtkStructuredGrid()
grid.SetDimensions(NX, NY, NZ)
grid.SetPoints(vtk_points)
grid.GetPointData().SetScalars(scalar_array)


# ── Step 3: run a VTK filter over Tack's memory ──────────────────────

from vtkmodules.vtkFiltersCore import vtkContourFilter

contour = vtkContourFilter()
contour.SetInputData(grid)
contour.SetInputArrayToProcess(
    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "gyroid")
contour.SetValue(0, 0.0)
contour.Update()

output = contour.GetOutput()
print(f"\nContour: {output.GetNumberOfPoints():,} points, "
      f"{output.GetNumberOfCells():,} triangles")

if output.GetNumberOfPoints() == 0:
    raise SystemExit("contour produced nothing")


# ── Step 4: bring the result back into Tack ──────────────────────────

result = vtk_to_field(output.GetPoints().GetData())
print(f"\nBack in Tack: {result.shape} -- {result.shape[1]} components, "
      f"still VTK's memory")


# ── Step 5: compute on it, to show it is really there ────────────────

@tack.kernel
def bounds(pts, lo, hi, n):
    for i in range(n):
        for c in range(3):
            tack.atomic_min(lo, c, pts[i, c])
            tack.atomic_max(hi, c, pts[i, c])


n_out = result.shape[0]
lo = tack.field(dtype=tack.f32, shape=(3,))
hi = tack.field(dtype=tack.f32, shape=(3,))
lo.from_numpy(np.full(3, 1e30, dtype=np.float32))
hi.from_numpy(np.full(3, -1e30, dtype=np.float32))

bounds(result, lo, hi, n_out)

lo_v, hi_v = lo.to_numpy(), hi.to_numpy()
print(f"  bounds x [{lo_v[0]:.3f}, {hi_v[0]:.3f}]  "
      f"y [{lo_v[1]:.3f}, {hi_v[1]:.3f}]  "
      f"z [{lo_v[2]:.3f}, {hi_v[2]:.3f}]")

print("\nTack kernel -> VTK filter -> Tack kernel, no copies")
