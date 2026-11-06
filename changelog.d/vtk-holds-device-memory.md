### Fixed

- A VTK with DLPack but without the device's memory (VTK master's nightly
  wheels, whose Viskores runs on the CPU alone yet lists CUDA and ROCm among
  `dlpack_support.supported_device_types()`) no longer fails every VTK
  algorithm on a GPU as it hands its output back: the interop shares one value
  once per device (`_vtk_holds`, by memory domain, VTK's error for a refusal not
  shown), and copies to the host where VTK cannot hold the memory.
