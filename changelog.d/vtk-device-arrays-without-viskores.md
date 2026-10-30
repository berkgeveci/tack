### Changed

- The VTK interop shares GPU memory with any VTK that holds device arrays, not
  only one built with Viskores: VTK's `vtkDeviceDataArray` (CUDA and Level Zero,
  needing only the driver) serves as well. Nothing in the interop changed; its
  device tests now run wherever such a VTK is found.
- `tack.interop.vtk.init_level_zero()` works with a VTK that has its own Level
  Zero context, without Viskores; its error message and docs no longer say
  Viskores on Kokkos' SYCL backend is required.
