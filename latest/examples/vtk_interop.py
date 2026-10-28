# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Share a VTK image's scalars with Tack on CPU, update them, and export to VTK.

Requires a VTK build with vtkmodules.util.dlpack_support. Rendering is not needed.
"""

import argparse

import numpy as np
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkCommonDataModel import vtkImageData

import tack
from tack.interop.vtk import array_to_vtk, vtk_to_array


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
    tack.set_default_device("cpu")
    # --8<-- [start:share]
    image = vtkImageData(dimensions=(9, 7, 5))
    initial = vtkFloatArray(number_of_tuples=image.number_of_points)  # VTK allocates it
    initial.Fill(0.0)
    image.point_data.SetScalars(initial)  # attribute data has no settable scalars property
    values = vtk_to_array(initial)  # shares VTK's allocation; retains its owner
    radius_squared(values, 9, 7)
    result = array_to_vtk(values, name="radius_squared")  # shares the Tack field
    image.point_data.SetScalars(result)
    # --8<-- [end:share]
    # A VTK structured image's x index varies fastest.
    z, y, x = np.mgrid[:5, :7, :9]
    expected = ((x - 4) ** 2 + (y - 3) ** 2 + (z - 2) ** 2).ravel().astype(np.float32)
    np.testing.assert_array_equal(np.asarray(initial), expected)
    np.testing.assert_array_equal(np.asarray(result), expected)
    # The exported array still aliases the original: an edit crosses both views.
    values.fill(123.0)
    assert np.asarray(initial)[0] == 123 and np.asarray(result)[0] == 123
    radius_squared(values, 9, 7)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Write the updated VTK image as a .vti file")
    args = parser.parse_args()
    image = run()
    if args.output:
        from vtkmodules.vtkIOXML import vtkXMLImageDataWriter

        writer = vtkXMLImageDataWriter(file_name=args.output)
        image >> writer
        # Write(), not writer(image): the call returns None and only logs a failure.
        if writer.Write() != 1:
            raise RuntimeError("VTK could not write the image")
    print("VTK interop: check passed")
