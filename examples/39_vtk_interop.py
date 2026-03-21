"""39 -- VTK interop: vtkmDataArray → vtkStructuredGrid → viskores filter.

Demonstrates zero-copy pipeline:
  1. Create raw arrays (simulating GPU memory)
  2. Wrap in vtkmDataArray via vtkMemoryDescriptor + vtkmDataArrayFactory
  3. Build vtkStructuredGrid from vtkmDataArrays
  4. Run viskores-accelerated contour filter
  5. Extract output memory descriptors

On CUDA/HIP, replace "host" with "cuda"/"hip" and pass device pointers.
The same pipeline keeps data on GPU through the entire filter execution.

Usage:
  python examples/39_vtk_interop.py
"""

import numpy as np
import vtk
from vtkmodules.vtkCommonCore import vtkMemoryDescriptor
from vtkmodules.vtkAcceleratorsVTKmCore import vtkmDataArrayFactory


# ================================================================
# Step 1: Create raw data arrays (simulating external memory)
# ================================================================

nx, ny, nz = 64, 64, 64
n_points = nx * ny * nz

# Coordinates (interleaved xyz, AoS layout)
coords = np.zeros(n_points * 3, dtype=np.float32)
idx = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            coords[idx * 3]     = float(i) / (nx - 1)
            coords[idx * 3 + 1] = float(j) / (ny - 1)
            coords[idx * 3 + 2] = float(k) / (nz - 1)
            idx += 1

# Scalar field: gyroid
scalars = np.zeros(n_points, dtype=np.float32)
idx = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            x = coords[idx * 3] * 2 * np.pi
            y = coords[idx * 3 + 1] * 2 * np.pi
            z = coords[idx * 3 + 2] * 2 * np.pi
            scalars[idx] = (np.sin(x) * np.cos(y) +
                            np.sin(y) * np.cos(z) +
                            np.sin(z) * np.cos(x))
            idx += 1

print(f"Grid: {nx}x{ny}x{nz} = {n_points:,} points")


# ================================================================
# Step 2: Wrap in vtkmDataArray via factory (zero-copy)
# ================================================================

def make_vtkm_array(data, n_tuples, n_components, name):
    """Wrap a numpy array as a vtkmDataArray via the factory."""
    desc = vtkMemoryDescriptor()
    desc.Set(data.ctypes.data, data.nbytes, "host")

    factory = vtkmDataArrayFactory()
    factory.SetNumberOfTuples(n_tuples)
    factory.SetNumberOfComponents(n_components)
    factory.SetDataType(10)  # VTK_FLOAT
    factory.AddBuffer(desc)

    arr = factory.CreateArray()
    arr.SetName(name)
    return arr


coord_array = make_vtkm_array(coords, n_points, 3, "Points")
scalar_array = make_vtkm_array(scalars, n_points, 1, "gyroid")

print(f"Coord array:  {coord_array.GetClassName()}")
print(f"Scalar array: {scalar_array.GetClassName()}")


# ================================================================
# Step 3: Build vtkStructuredGrid
# ================================================================

points = vtk.vtkPoints()
points.SetData(coord_array)

grid = vtk.vtkStructuredGrid()
grid.SetDimensions(nx, ny, nz)
grid.SetPoints(points)
grid.GetPointData().SetScalars(scalar_array)

print(f"\nStructured grid: {grid.GetNumberOfPoints():,} points, "
      f"{grid.GetNumberOfCells():,} cells")


# ================================================================
# Step 4: Run viskores contour filter
# ================================================================

contour = vtk.vtkContourFilter()
contour.SetInputData(grid)
contour.SetValue(0, 0.0)
contour.Update()

output = contour.GetOutput()
print(f"\nContour result:")
print(f"  Points:    {output.GetNumberOfPoints():,}")
print(f"  Triangles: {output.GetNumberOfCells():,}")

if output.GetNumberOfPoints() > 0:
    out_pts = output.GetPoints().GetData()
    print(f"  Output points type: {out_pts.GetClassName()}")


# ================================================================
# Step 5: Extract output memory descriptors
# ================================================================

    descs = out_pts.GetMemoryDescriptors()
    print(f"\n  Output memory descriptors: {descs.GetNumberOfItems()}")
    for i in range(descs.GetNumberOfItems()):
        d = descs.GetItemAsObject(i)
        print(f"    [{i}] role={d.GetRole()}, space={d.GetMemorySpace()}, "
              f"size={d.GetSizeInBytes():,} bytes, ptr=0x{d.GetPointer():x}")

    print(f"\nFull pipeline: raw memory → vtkmDataArray → "
          f"vtkStructuredGrid → viskores contour → output ✓")
