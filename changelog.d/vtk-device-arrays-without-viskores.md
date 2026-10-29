### Changed

- The VTK interop shares GPU memory with any VTK that holds device arrays, not
  only one built with Viskores: VTK's `vtkDeviceDataArray` (CUDA, needing only
  the driver) serves as well. Nothing in the interop changed; its device tests
  now run wherever such a VTK is found.
