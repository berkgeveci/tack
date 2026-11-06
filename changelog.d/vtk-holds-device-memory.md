### Fixed

- A VTK with DLPack but without the device's memory (the nightly wheels, whose
  Viskores runs on the CPU alone yet lists CUDA among
  `dlpack_support.supported_device_types()`) no longer fails every VTK
  algorithm on a GPU as it hands its output back: the interop tries a
  four-value array once per type of device (`_vtk_holds`) and copies to the
  host where VTK cannot hold the memory.
