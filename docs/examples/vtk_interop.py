# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Share a VTK image's scalars with Tack on CPU, update them, and export to VTK.

Requires a VTK build with vtkmodules.util.dlpack_support. Rendering is not needed.
"""

import argparse

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData

import tack
from tack.interop.vtk import field_to_vtk, vtk_to_field


# --8<-- [start:kernel]
@tack.kernel
def radius_squared(values, nx, ny):
    for i in range(values.shape[0]):
        x = float(i % nx) - float(nx - 1) * 0.5
        y = float((i // nx) % ny) - float(ny - 1) * 0.5
        z = float(i // (nx * ny)) - 2.0
        values[i] = x * x + y * y + z * z
# --8<-- [end:kernel]


def run():
    tack.init(arch=tack.cpu)
    # --8<-- [start:share]
    image = vtkImageData()
    image.SetDimensions(9, 7, 5)
    initial = numpy_to_vtk(np.zeros(image.GetNumberOfPoints(), dtype=np.float32), deep=True)
    image.GetPointData().SetScalars(initial)
    values = vtk_to_field(initial)  # shares VTK's allocation; retains its owner
    radius_squared(values, 9, 7)
    result = field_to_vtk(values, name="radius_squared")  # shares the Tack field
    image.GetPointData().SetScalars(result)
    # --8<-- [end:share]
    # A VTK structured image's x index varies fastest.
    z, y, x = np.mgrid[:5, :7, :9]
    expected = ((x - 4) ** 2 + (y - 3) ** 2 + (z - 2) ** 2).ravel().astype(np.float32)
    np.testing.assert_array_equal(vtk_to_numpy(initial), expected)
    np.testing.assert_array_equal(vtk_to_numpy(result), expected)
    # The exported array still aliases the original: an edit crosses both views.
    values.fill(123.0)
    assert vtk_to_numpy(initial)[0] == 123 and vtk_to_numpy(result)[0] == 123
    radius_squared(values, 9, 7)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Write the updated VTK image as a .vti file")
    args = parser.parse_args()
    image = run()
    if args.output:
        from vtkmodules.vtkIOXML import vtkXMLImageDataWriter

        writer = vtkXMLImageDataWriter()
        writer.SetFileName(args.output)
        writer.SetInputData(image)
        if writer.Write() != 1:
            raise RuntimeError("VTK could not write the image")
    print("VTK interop: check passed")
